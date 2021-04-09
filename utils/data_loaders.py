import os
import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

CIFAR_TRANSFORM = {"size": 32}
IMAGENET_TRANSFORM = {"size": 256}


def get_train_val_loaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset == "image-net" or args.dataset == "image-net-tar":
        size = IMAGENET_TRANSFORM["size"]
    elif args.dataset == "cifar":
        size = CIFAR_TRANSFORM["size"]

    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])

    if args.dataset == "image-net":
        train_dataset = datasets.ImageFolder(os.path.join(
            args.data_dir, "train"),
                                             transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"),
                                           transform=val_transform)
    elif args.dataset == "image-net-tar":
        train_dataset = datasets.ImageNet(args.data_dir,
                                          split="train",
                                          transform=train_transform)
        val_dataset = datasets.ImageNet(args.data_dir,
                                        split="val",
                                        transform=val_transform)
    elif args.dataset == "cifar":
        train_dataset = datasets.CIFAR10(args.data_dir,
                                         train=True,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(args.data_dir,
                                       train=True,
                                       transform=val_transform,
                                       download=True)

    if args.val_size:
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
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
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