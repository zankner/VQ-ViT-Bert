import os
import datetime
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from vit import ViT, LinearClassifier, MPP, TokensDataset
from vit.model_utils import fine_tune_train_step, fine_tune_validate_step


def fine_tune(args):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    train_dataset = TokensDataset(args.data_dir, args.extension)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size)

    test_dataset = TokensDataset(args.data_dir, args.extension)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size)

    log_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = os.path.join("vit_runs", log_time)
    os.mkdir(summary_path)
    writer = SummaryWriter(summary_path)

    checkpoint_dir = os.path.join(args.checkpoint_dir, log_time)
    os.mkdir(checkpoint_dir)

    transformer_temp = ViT(args.dim, args.depth, args.heads, args.mlp_dim,
                           args.vocab_size, args.embedding_dim, args.dim_head,
                           args.dropout, args.emb_dropout)
    mpp = MPP(transformer_temp, args.mask_prob, args.replace_prob,
              args.num_tokens, args.random_token_prob, args.mask_token_id,
              args.pad_token_id, args.mask_ignore_token_ids)
    saved_path = os.path.join(args.transformer_ckpt, args.transformer_dir,
                              "checkpoint.pt")
    mpp_saved = torch.load(saved_path)['model_state_dict']
    mpp.load_state_dict(mpp_saved)

    classifier = LinearClassifier(mpp.transformer, args.dim, args.out_dim)
    classifier.eval()
    classifier.to(device)

    optimizer = optim.Adam(classifier.parameters(),
                           weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=args.step_size)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    start_epoch = 1
    for epoch in range(start_epoch, args.epochs + 1):
        fine_tune_train_step(train_loader, classifier, criterion, optimizer,
                             epoch, device, writer, args)
        scheduler.step()

        val_loss = fine_tune_validate_step(test_loader, classifier, criterion,
                                           device, epoch, writer, args)

        # Checkpoint model
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
