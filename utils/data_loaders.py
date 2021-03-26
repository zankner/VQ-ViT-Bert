import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_val_loaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(args.data_dir,
                                     train=True,
                                     transform=train_transform,
                                     download=True)
    val_dataset = datasets.CIFAR10(args.data_dir,
                                   train=True,
                                   transform=val_transform,
                                   download=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.val_size * num_train))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=args.num_workers,
    )

    return train_loader, val_loader


def get_test_loader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.CIFAR10(args.data_dir,
                                    train=False,
                                    transform=test_transform,
                                    download=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return test_loader