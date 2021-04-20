import torch
import time
from utils import AverageMeter, ProgressMeter
from utils import accuracy


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
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)

        # compute output
        loss = model(images)

        # measure perplexity and record loss
        losses.update(loss.mean().item(), images.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.mean().backward()

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
        for i, (images, _) in enumerate(val_loader):
            images = images.to(device)

            # compute output
            loss = model(images)

            # measure perplexity and record loss
            losses.update(loss.mean().item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                writer.add_scalar('validation_loss', losses.avg,
                                  epoch * len(val_loader) + i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Loss {losses.avg:.3f}'.format(losses=losses))

        return losses.avg


def fine_tune_train_step(train_loader, model, criterion, optimizer, epoch,
                         device, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.mean().item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.mean().backward()

        # do SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            writer.add_scalar('train_acc_1', top1.avg,
                              epoch * len(train_loader) + i)
            writer.add_scalar('train_acc_5', top5.avg,
                              epoch * len(train_loader) + i)


def fine_tune_validate_step(val_loader, model, criterion, device, epoch,
                            writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.mean().item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                writer.add_scalar('val_acc_1', top1.avg,
                                  epoch * len(val_loader) + i)
                writer.add_scalar('val_acc_5', top5.avg,
                                  epoch * len(val_loader) + i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))

    return top1.avg
