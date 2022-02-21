import os
import torch
import torchvision
from data import DataLoader

def get_dataloader(dataset, path, batchsize=512, train_transforms=[], val_transforms=[], mean=0, std=1, num_workers=4):
    normalization = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]
    
    composed_train_transforms = torchvision.transforms.Compose(train_transforms+normalization)
    composed_val_transforms = torchvision.transforms.Compose(val_transforms+normalization)
    
    if dataset == 'imagenette':
        train_path = os.path.join(path, 'train')
        val_path = os.path.join(path, 'val')
        train = torchvision.datasets.ImageFolder(
            root=train_path,
            transform=composed_train_transforms,
        )
        val = torchvision.datasets.ImageFolder(
            root=val_path,
            transform=composed_val_transforms,
        )

    if dataset == 'MNIST':
        train = torchvision.datasets.MNIST(path, train=True, transform=composed_train_transforms, download=True)
        val = torchvision.datasets.MNIST(path, train=False, transform=composed_val_transforms, download=True)
    if dataset == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(path, train=True, transform=composed_train_transforms, download=True)
        val = torchvision.datasets.CIFAR10(path, train=False, transform=composed_val_transforms, download=True)
    if dataset == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(path, train=True, transform=composed_train_transforms, download=True)
        val = torchvision.datasets.CIFAR100(path, train=False, transform=composed_val_transforms, download=True)
    
    dataloader_kwargs = dict(batch_size=batchsize, 
                             pin_memory=True,
                             shuffle=True,
                             num_workers=num_workers,
    )
    
    dataloaders = {
        'train': DataLoader(train, **dataloader_kwargs),
        'validation': DataLoader(val, **dataloader_kwargs),
    }
    return dataloaders

def load_imagenette(path, batchsize=32, train_transforms=[], val_transforms=[], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), num_workers=4):
    normalization = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]
    
    composed_val_transforms= torchvision.transforms.Compose(val_transforms+normalization)
    composed_train_transforms = torchvision.transforms.Compose(train_transforms+normalization)

    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    
    train = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=composed_train_transforms,
    )
    val = torchvision.datasets.ImageFolder(
        root=val_path,
        transform=composed_val_transforms,
    )
    
    dataloader_kwargs = dict(batch_size=batchsize, 
                             pin_memory=True,
                             shuffle=True,
                             num_workers=num_workers,
    )
    
    dataloaders = {
        'train': DataLoader(train, **dataloader_kwargs),
        'validation': DataLoader(val, **dataloader_kwargs),
    }
    return dataloaders

def load_torchvision_dataset(dataset, path, batchsize=512, train_transforms=[], val_transforms=[], mean=0, std=1, num_workers=4):
    normalization = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std)
    ]
    composed_train_transforms = torchvision.transforms.Compose(train_transforms+normalization)
    composed_val_transforms = torchvision.transforms.Compose(val_transforms+normalization)
    
    if dataset == 'MNIST':
        train = torchvision.datasets.MNIST(path, train=True, transform=composed_train_transforms, download=True)
        test = torchvision.datasets.MNIST(path, train=False, transform=composed_val_transforms, download=True)
    if dataset == 'CIFAR10':
        train = torchvision.datasets.CIFAR10(path, train=True, transform=composed_train_transforms, download=True)
        test = torchvision.datasets.CIFAR10(path, train=False, transform=composed_val_transforms, download=True)
    if dataset == 'CIFAR100':
        train = torchvision.datasets.CIFAR100(path, train=True, transform=composed_train_transforms, download=True)
        test = torchvision.datasets.CIFAR100(path, train=False, transform=composed_val_transforms, download=True)
    
    dataloader_kwargs = dict(batch_size=batchsize, 
                             pin_memory=True,
                             shuffle=True,
                             num_workers=num_workers,
    )
    
    dataloaders = {
        'train': DataLoader(train, **dataloader_kwargs),
        'validation': DataLoader(test, **dataloader_kwargs),
    }
    return dataloaders