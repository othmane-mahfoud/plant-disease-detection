import os
import random
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

NUM_WORKERS = 1

# Define albumentations transforms including CutMix
def get_transforms():
    return A.Compose([
        A.Resize(264, 264),  # Resize to 264x264
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        ToTensorV2(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Custom CutMix function
def cutmix(images, targets, alpha=1.0):
    indices = torch.randperm(images.size(0))
    shuffled_images = images[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = shuffled_images[:, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
    
    targets_a, targets_b = targets, shuffled_targets
    return images, targets_a, targets_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Custom dataset class to apply CutMix
class CutMixDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, cutmix_alpha=1.0):
        super().__init__(root, transform=None)
        self.cutmix_alpha = cutmix_alpha
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=image)
            image = augmented['image']
        return image, target

    def collate_fn(self, batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        targets = torch.tensor(targets)
        images, targets_a, targets_b, lam = cutmix(images, targets, self.cutmix_alpha)
        return images, targets_a, targets_b, lam
    

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    test_transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS,
    with_augmentation: bool=False
):
    """Creates training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    test_transform: torchvision transforms to perform on the testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader).
    """

    if with_augmentation:
        train_data = CutMixDataset(root=train_dir, transform=get_transforms())
        train_dataloader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_data.collate_fn,
        )
    else:
        train_data = datasets.ImageFolder(train_dir, transform=test_transform)
        train_dataloader = DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True,
        )

    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False, # don't need to shuffle test data
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader
