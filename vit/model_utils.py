import torch
import time
from utils import AverageMeter, ProgressMeter


def train_step(train_loader, model, optimizer, epoch, device, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (tokens, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        tokens = tokens.to(device)

        # compute output
        loss = model(tokens)

        # measure perplexity and record loss
        losses.update(loss.item(), tokens.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        # do SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            writer.add_scalar('training_loss', losses.avg,
                              epoch * len(train_loader) + i)


def validate_step(val_loader, model, device, epoch, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(val_loader), [batch_time, losses],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (tokens, _) in enumerate(val_loader):
            tokens = tokens.to(device)

            # compute output
            loss = model(tokens)

            # measure perplexity and record loss
            losses.update(loss.item(), tokens.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                writer.add_scalar('validation_loss', losses.avg,
                                  epoch * len(val_loader) + i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Loss {loss.avg:.3f}'.format(loss=loss))

        return loss.avg