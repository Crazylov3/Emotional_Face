import sys

import torch
from timm.utils import AverageMeter, accuracy
import math


def train_one_epoch(model, pdar, optimizer, criterion, device, max_norm=0):
    model.train()
    criterion.train()
    loss_metric = AverageMeter()
    acc1_metric = AverageMeter()
    acc2_metric = AverageMeter()
    for img, target in pdar:
        img = img.to(device)
        target = target.to(device)
        oup = model(img)
        loss = criterion(oup, target)
        acc1, acc2 = accuracy(oup, target, topk=(1, 2))

        if not math.isfinite(loss):
            print("Loss is {}, stopping training".format(loss))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        loss_metric.update(loss.item())
        acc1_metric.update(acc1.item())
        acc2_metric.update(acc2.item())

        pdar.set_postfix({
            "train loss": loss_metric.avg,
            "top_1 accuracy": acc1_metric.avg,
            "top_2 accuracy": acc2_metric.avg
        })

    return loss_metric.avg, acc1_metric.avg, acc2_metric.avg


@torch.no_grad()
def evaluate(model, pdar, criterion, device):
    model.eval()
    criterion.eval()
    loss_metric = AverageMeter()
    acc1_metric = AverageMeter()
    acc2_metric = AverageMeter()
    for img, target in pdar:
        img = img.to(device)
        target = target.to(device)
        oup = model(img)
        loss = criterion(oup, target)
        acc1, acc2 = accuracy(oup, target, topk=(1, 2))

        loss_metric.update(loss.item())
        acc1_metric.update(acc1.item())
        acc2_metric.update(acc2.item())

        pdar.set_postfix({
            "test loss": loss_metric.avg,
            "top_1 accuracy": acc1_metric.avg,
            "top_2 accuracy": acc2_metric.avg
        })

    return loss_metric.avg, acc1_metric.avg, acc2_metric.avg
