import os
import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from vit import ViT, LinearClassifier, MPP
from vit.model_utils import fine_tune_train_step, fine_tune_validate_step
from vae import VQVae, OpenAIDiscreteVAE
from utils import get_train_val_loaders


def fine_tune(args):
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
        vae = VQVae(args.num_codebook_indeces, args.embedding_dim,
                    args.num_blocks, args.feature_dim, args.channels)

    transformer = ViT(vae, args.dim, args.depth, args.heads, args.mlp_dim,
                      args.vocab_size, args.num_codebook_indeces,
                      args.dim_head, args.dropout, args.emb_dropout)
    mpp = MPP(transformer, args.vocab_size, args.dim, args.mask_prob,
              args.replace_prob, args.random_token_prob, args.mask_token_id,
              args.pad_token_id, args.cls_token_id, args.mask_ignore_token_ids)

    ckpt_dir = os.path.join(args.mpp_ckpt, "checkpoint.pt")
    mpp_ckpt = torch.load(ckpt_dir)['model_state_dict']
    mpp.load_state_dict(mpp_ckpt)

    classifier = LinearClassifier(mpp.transformer, args.dim, args.out_dim)
    classifier.eval()
    classifier.to(device)

    if args.freeze_transformer:
        params = classifier.classification_head.parameters()
    else:
        params = classifier.parameters()
    optimizer = optim.Adam(params,
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.step_size)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        fine_tune_train_step(train_loader, classifier, criterion, optimizer,
                             epoch, device, writer, args)
        scheduler.step()

        val_loss = fine_tune_validate_step(val_loader, classifier, criterion,
                                           device, epoch, writer, args)

        # Checkpoint model
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
