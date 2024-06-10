import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader, default_collate

def get_base_transforms():
    return v2.Compose([
        v2.PILToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_advanced_transforms():
    return v2.Compose([
        v2.PILToTensor(),
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomRotation(90),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def collate_fn(batch):
    cutmix = v2.CutMix(num_classes=39)
    return cutmix(*default_collate(batch))

def create_dataloaders(train_dir, test_dir, batch_size, num_workers, augmentation):
    base_transform = get_base_transforms()
    advanced_transform = get_advanced_transforms()
    
    if augmentation == "advanced":
        train_dataset = datasets.ImageFolder(root=train_dir, transform=advanced_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=base_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif augmentation == "cutmix":
        train_dataset = datasets.ImageFolder(root=train_dir, transform=base_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=base_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        train_dataset = datasets.ImageFolder(root=train_dir, transform=base_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=base_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader

def imshow(img, title, is_grid):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)  # Clip values to be in the range [0, 1]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    if is_grid == False:
        plt.axis('off') 
    plt.show()


def display_img_grid(dataloader, title, is_cutmix=False):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images), title=title, is_grid=True)

    if is_cutmix:
        labels = labels.argmax(dim=1).numpy()
        print('\n'.join(f'{dataloader.dataset.classes[int(label)]}' for label in labels))
    else:
        print('\n'.join('%5s' % dataloader.dataset.classes[labels[j]] for j in range(len(labels))))


def display_img_sample(dataloader, title, is_cutmix=False, sample_size=3):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    if is_cutmix:
        labels = labels.argmax(dim=1).cpu().numpy()
        for i in range(sample_size):
            imshow(images[i], dataloader.dataset.classes[int(labels[i])], is_grid=False)
    else:
        for i in range(sample_size):
            imshow(images[i], dataloader.dataset.classes[int(labels[i])],  is_grid=False)