import torch
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """single epoch step for training the model

    sets in training mode, performs forward pass, performs optimizer step,
    calculates loss and classification metrics

    Args:
    model: PyTorch model we want to train.
    dataloader: Pytorch dataloader that contains the data we want our model to train on.
    loss_fn: the Pytorch function chosen to calculate loss.
    optimizer: the PyTorch optimizer chosen to minimize loss.
    device: our target device

    Returns:
    A tuple of training loss, training accuracy, precision and recall
    """
    # setup: train mode and initialize loss and metrics
    model.train()

    train_loss = 0

    accuracy = MulticlassAccuracy(num_classes=39)
    precision = MulticlassPrecision(num_classes=39)
    recall = MulticlassRecall(num_classes=39)

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # ensuring device is set correctly
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # extract predicted class
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        
        # Convert one-hot encoded labels back to class indices
        if y.ndim == 2:
            _, y = y.max(dim=1)

        # calculate the classification metrics
        accuracy.update(y_pred_class, y)
        precision.update(y_pred_class, y)
        recall.update(y_pred_class, y)

    # average loss
    train_loss = train_loss / len(dataloader)

    # compute metrics for all batches
    train_acc = accuracy.compute()
    train_prec = precision.compute()
    train_rec = recall.compute()

    # reset metrics
    accuracy.reset()
    precision.reset()
    recall.reset()

    return train_loss, train_acc, train_prec, train_rec


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """single epoch step for testing the model

    sets in evaluation mode, performs forward pass, calculates loss and classification metrics

    Args:
    model: PyTorch model we want to test.
    dataloader: Pytorch dataloader that contains the data we want our model to tested on.
    loss_fn: the Pytorch function chosen to calculate loss.
    device: our target device

    Returns:
    A tuple of testing loss, testing accuracy, precision and recall
    """
    # setup: eval mode and initialize loss and metrics
    model.eval()

    test_loss = 0
    
    accuracy = MulticlassAccuracy(num_classes=39)
    precision = MulticlassPrecision(num_classes=39)
    recall = MulticlassRecall(num_classes=39)

    # turn on inference context manager
    with torch.inference_mode():
        # loop through data loader data batches
        for batch, (X, y) in enumerate(dataloader):
            # send data to target device
            X, y = X.to(device), y.to(device)

            # forward pass
            test_pred_logits = model(X)

            # calculate accumulated loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # extract predictions
            y_pred_class = test_pred_logits.argmax(dim=1)

            # Convert one-hot encoded labels back to class indices
            if y.ndim == 2:
                _, y = y.max(dim=1)

            # calculate the classification metrics
            accuracy.update(y_pred_class, y)
            precision.update(y_pred_class, y)
            recall.update(y_pred_class, y)
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    
    # compute metrics for all batches
    test_acc = accuracy.compute()
    test_prec = precision.compute()
    test_rec = recall.compute()

    return test_loss, test_acc, test_prec, test_rec



# Add writer parameter to train()
def train(model: torch.nn.Module, 
    train_dataloader: torch.utils.data.DataLoader, 
    test_dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    device: torch.device, 
    writer_local: torch.utils.tensorboard.writer.SummaryWriter, 
    writer_drive: torch.utils.tensorboard.writer.SummaryWriter,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
    early_stopping: EarlyStopping = None
) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_accuracy": [],
               "train_precision": [],
               "train_recall": [],
               "test_loss": [],
               "test_accuracy": [],
               "test_precision": [],
               "test_recall": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_prec, train_rec = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device)
        test_loss, test_acc, test_prec, test_rec = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)

        # Print out what's happening
        print(
            f"epoch: {epoch + 1} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_acc:.4f} | "
            f"test loss: {test_loss:.4f} | "
            f"test accuracy: {test_acc:.4f}"
        )

        # add epoch results to our dictionary
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_acc)
        results["train_precision"].append(train_prec)
        results["train_recall"].append(train_rec)

        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_acc)
        results["test_precision"].append(test_prec)
        results["test_recall"].append(test_rec)

        # log results to SummaryWriter
        if writer_local:
            writer_local.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer_local.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_accuracy": train_acc,
                                                "test_accuracy": test_acc}, 
                               global_step=epoch)
            writer_local.add_scalars(main_tag="Precision", 
                               tag_scalar_dict={"train_precision": train_prec,
                                                "test_precision": test_prec},
                               global_step=epoch)
            writer_local.add_scalars(main_tag="Recall", 
                               tag_scalar_dict={"train_recall": train_rec,
                                                "test_recall": test_rec},
                               global_step=epoch)
            writer_local.close()
        
        if writer_drive:
            writer_drive.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer_drive.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_accuracy": train_acc,
                                                "test_accuracy": test_acc}, 
                               global_step=epoch)
            writer_drive.add_scalars(main_tag="Precision", 
                               tag_scalar_dict={"train_precision": train_prec,
                                                "test_precision": test_prec},
                               global_step=epoch)
            writer_drive.add_scalars(main_tag="Recall", 
                               tag_scalar_dict={"train_recall": train_rec,
                                                "test_recall": test_rec},
                               global_step=epoch)
            writer_drive.close()
        
        # Step the scheduler
        if scheduler:
            scheduler.step()

        # Check early stopping
        if early_stopping:
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("[INFO] Early stopping")
                break

    return results