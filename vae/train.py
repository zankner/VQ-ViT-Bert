import os
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
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

    dataset = datasets.CIFAR10(args.data_dir,
                               train=True,
                               transform=train_transforms,
                               download=True)
    train_len = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_len, len(dataset) - train_len])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size)

    val_dataset = datasets.CIFAR10(args.data_dir,
                                   train=False,
                                   transform=test_transforms,
                                   download=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size)

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join(args.summary_dir, log_time)
    os.mkdir(summary_path)
    writer = SummaryWriter(summary_path)

    checkpoint_dir = os.path.join(args.checkpoint_dir, log_time)
    os.mkdir(checkpoint_dir)

    model = VQVae(args.vocab_size, args.num_embeddings, args.num_blocks,
                  args.feature_dim, args.channels, args.commitment_cost)
    model.eval()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.step_size)

    criterion = nn.MSELoss()
    criterion.to(device)

    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_step(train_loader, model, criterion, optimizer, epoch, device,
                   writer, args)
        scheduler.step()

        val_loss = validate_step(val_loader, model, criterion, device, args)

        # Checkpoint model
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
