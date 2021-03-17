import os
import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from vit import ViT
from vit import TokensDataset, MPP
from vae.model_utils import train_step, validate_step


def train(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    train_dataset = TokensDataset(args.data_dir, args.extension)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size)

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join("vit_runs", log_time)
    os.mkdir(summary_path)
    writer = SummaryWriter(summary_path)

    transformer = ViT(args.dim, args.depth, args.heads, args.mlp_dim,
                      args.vocab_size, args.embedding_dim, args.dim_head,
                      args.dropout, args.emb_dropout)
    mpp = MPP(transformer, args.mask_prob, args.replace_prob, args.num_tokens,
              args.random_token_prob, args.mask_token_id, args.pad_token_id,
              args.mask_ignore_token_ids)
    mpp.eval()
    mpp.to(device)

    optimizer = optim.Adam(mpp.parameters(), weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.step_size)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_step(train_loader, mpp, optimizer, epoch, device, writer, args)
        scheduler.step()
