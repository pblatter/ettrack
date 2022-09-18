import os
import time
import torch
import torchvision

from collections import OrderedDict
from supernet_backbone.lib_back.utils.helpers import AverageMeter, accuracy, reduce_tensor

def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', use_amp=False, model_ema=None, logger=None, writer=None):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    prec1_m = AverageMeter()
    prec5_m = AverageMeter()

    model.train()

    maxup_cnt = 0

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    optimizer.zero_grad()
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()

        output = model(input)

        if args.loss_scale:
            loss = loss_fn(output, target) / args.accumulation_steps
        else:
            loss = loss_fn(output, target)

        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        if not args.distributed:
            # losses_m.update(loss.item(), input.size(0))
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            prec1 = reduce_tensor(prec1, args.world_size)
            prec5 = reduce_tensor(prec5, args.world_size)
        else:
            reduced_loss = loss.data

        loss.backward()

        if ((batch_idx + 1) % args.accumulation_steps == 0) or last_batch:
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        losses_m.update(reduced_loss.item(), input.size(0))
        prec1_m.update(prec1.item(), output.size(0))
        prec5_m.update(prec5.item(), output.size(0))

        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{}] '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f}) '
                    'Prec@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                    'Prec@5: {top5.val:>7.4f} ({top5.avg:>7.4f}) '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                    'LR: {lr:.3e}'
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        loss=losses_m,
                        top1=prec1_m,
                        top5=prec5_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

                if args.maxup:
                    logger.info("Maxup Count: {}/{}".format(maxup_cnt, batch_idx+1))

                writer.add_scalar('Loss/train', prec1_m.avg, epoch * len(loader) + batch_idx)
                writer.add_scalar('Accuracy/train', prec1_m.avg, epoch * len(loader) + batch_idx)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch * len(loader) + batch_idx)

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(
                model, optimizer, args, epoch, model_ema=model_ema, use_amp=use_amp, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)


        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])