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
from utils import get_train_val_loaders, get_test_loader


def train(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    train_loader, val_loader = get_train_val_loaders(args)

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join(args.summary_dir, log_time)
    os.mkdir(summary_path)
    writer = SummaryWriter(summary_path)

    checkpoint_dir = os.path.join(args.checkpoint_dir, log_time)
    os.mkdir(checkpoint_dir)

    model = VQVae(args.num_codebook_indeces, args.embedding_dim,
                  args.num_blocks, args.feature_dim, args.channels,
                  args.commitment_cost)
    model.eval()
    model.to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

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
