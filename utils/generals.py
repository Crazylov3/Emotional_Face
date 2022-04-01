import torch
from torchvision.transforms import transforms as tf
import math
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
import glob
import re


def save_strategy(acc1, acc2, best_acc1, best_acc2):
    return (acc1 * 0.9 + acc2 * 0.1) > (best_acc1 * 0.9 + best_acc2 * 0.1)


def save_checkpoint(epoch, optimizer, lr_scheduler, loss, model, acc1_2, path):
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": loss,
        "lr_scheduler": lr_scheduler.state_dict(),
        "acc": acc1_2
    }, path)
    print("---> Saved checkpoint!")


def load_checkpoint(path):
    ckpt = torch.load(path)
    state_dict = ckpt["model"]
    epoch = ckpt["epoch"]
    optim = ckpt["optimizer"]
    loss = ckpt["loss"]
    lr_scheduler = ckpt["loss"]
    acc = ckpt["acc"]
    return state_dict, epoch, optim, lr_scheduler, loss, acc


def get_transform(w, h):
    transform = tf.Compose([
        tf.RandomVerticalFlip(),
        tf.RandomRotation(0.4),
        tf.Grayscale(),
        tf.Resize((h, w)),
        tf.ToTensor(),
        tf.Normalize((0.5,), (0.5,))

    ])
    return transform


def get_lr_scheduler(optimizer, epochs, lrf=0.05):
    return LambdaLR(optimizer, lr_lambda=lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{sep}{n}"
