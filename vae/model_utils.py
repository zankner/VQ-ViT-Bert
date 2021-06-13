import torch
import time
from utils import AverageMeter, ProgressMeter


def train_step(train_loader, model, criterion, optimizer, epoch, device,
               writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    perplexities = AverageMeter('Perplexity', ':.4e')

    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, perplexities],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)

        # compute output
        output, vq_loss, perplexity = model(images)
        loss = criterion(output, images) + vq_loss

        # measure perplexity and record loss
        losses.update(loss.item(), images.size(0))
        perplexities.update(perplexity.item(), images.size(0))

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
    writer.add_scalar('loss/train', losses.avg, epoch)
    writer.add_scalar('perlexity/train', perplexities.avg, epoch)


def validate_step(val_loader, model, criterion, device, epoch, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    perplexities = AverageMeter('Perplexity', ':.4e')
    progress = ProgressMeter(len(val_loader),
                             [batch_time, losses, perplexities],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, vq_loss, perplexity = model(images)
            loss = criterion(output, images) + vq_loss

            # measure perplexity and record loss
            losses.update(loss.item(), images.size(0))
            perplexities.update(perplexity.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Loss {losses.avg:.3f}'.format(losses=losses))

    writer.add_scalar('loss/val', losses.avg, epoch)
    writer.add_scalar('perlexity/val', perplexities.avg, epoch)
    return losses.avg