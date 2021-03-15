import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms, datasets
import datetime
from vae import VQVae
from vae.model_utils import train_step, validate_step


def train(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(32),
         transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(args.data_dir,
                                     train=True,
                                     transform=train_transforms,
                                     download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size)

    test_dataset = datasets.CIFAR10(args.data_dir,
                                    train=False,
                                    transform=test_transforms,
                                    download=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size)

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model = VQVae(args.vocab_size, args.num_embeddings, args.num_blocks,
                  args.feature_dim, args.channels, args.commitment_cost)
    model.eval()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.step_size)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_step(train_loader, model, criterion, optimizer, epoch, device,
                   args)
        scheduler.step()

        validate_step(test_loader, model, criterion, device, args)
