import os
import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from vit import ViT
from vit import MPP
from vit.model_utils import train_step, validate_step
from vae import VQVae, OpenAIDiscreteVAE
from utils import get_train_val_loaders


def pretrain(args):
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

    if args.architecture == "dall-e":
        vae = OpenAIDiscreteVAE()
    else:
        assert args.vae_ckpt != None, "Vae checkpoint must be specified when loading a custom trained VAE"

        vae = VQVae(args.num_codebook_indeces, args.embedding_dim,
                    args.num_blocks, args.feature_dim, args.channels)
        ckpt_dir = os.path.join(args.vae_ckpt, "checkpoint.pt")
        vae_ckpt = torch.load(ckpt_dir)['model_state_dict']
        vae.load_state_dict(vae_ckpt)

    transformer = ViT(vae, args.dim, args.depth, args.heads, args.mlp_dim,
                      args.vocab_size, args.num_codebook_indeces,
                      args.dim_head, args.dropout, args.emb_dropout)
    mpp = MPP(transformer, args.vocab_size, args.dim, args.mask_prob,
              args.replace_prob, args.random_token_prob, args.mask_token_id,
              args.pad_token_id, args.cls_token_id, args.mask_ignore_token_ids)
    mpp.eval()
    mpp.to(device)

    optimizer = optim.AdamW(mpp.parameters(),
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.step_size)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        train_step(train_loader, mpp, optimizer, epoch, device, writer, args)
        scheduler.step()

        val_loss = validate_step(val_loader, mpp, device, epoch, writer, args)

        # Checkpoint model
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': mpp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
