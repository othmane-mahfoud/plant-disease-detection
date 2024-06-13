import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR
from helpers.modeling import create_base_model, create_hybrid
from helpers.utils import save_model
from helpers.setup import set_device
from torcheval.metrics import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

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
    trial,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
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

        # Step the scheduler
        if scheduler:
            scheduler.step()

        # For pruning (stops trial early if not promising)
        trial.report(results["test_accuracy"][-1].item(), epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return results



def objective(trial, train_dataloader, test_dataloader):

    device = set_device()

    # Hyperparameters to optimize
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    dropout_proba = trial.suggest_float("dropout_proba", 0.2, 0.5)

    # Create model
    model = create_hybrid(device=device, out_features=39, dropout_proba=dropout_proba)
    model.to(device)

    # Select optimizer
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001) # I added weight decay here to prevent overfitting
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # learning rate scheduler

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Train the model
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=10,
        device=device,
        trial=trial,
        scheduler=scheduler,
    )

    # Save the model
    save_model(model, "models", f"HybridModel_trial_{trial.number}.pth")

    # Return validation accuracy for optimization
    return results["test_accuracy"][-1].item()
