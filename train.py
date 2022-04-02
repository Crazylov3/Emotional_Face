import os.path
import wandb
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader
from models.transformer import SwinTransformer
from utils.dataset import EmotionalDataSet
from utils.generals import load_checkpoint, save_checkpoint, get_transform, get_lr_scheduler, save_strategy, \
    increment_path, intersect_dicts
from trainer.engine import train_one_epoch, evaluate
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    class_mapping = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "sad": 5,
        "surprise": 6
    }
    imgsz = w = h = 224
    in_channel = 1
    num_classes = len(class_mapping.keys())
    embed_dim = 96
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    learning_rate = 1e-3
    decay = 5e-3
    epochs = 200
    save_root = increment_path("/content/drive/MyDrive/Emotional/runs/exp", exist_ok=False)
    save_ckpt_path = save_root + "/" + "checkpoint.pth"
    pretrained_weights = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_epoch = 10
    use_tensorboard = True
    use_wandb = True
    cache_train = True

    df_test = pd.read_csv("data/test.csv")
    df_train = pd.read_csv("data/train.csv")

    trainset = EmotionalDataSet(
        df_train,
        root="data/train",
        cache_train=cache_train,
        cache_path="data/train.cache",
        transformer=get_transform(w, h)
    )

    testset = EmotionalDataSet(
        df_test,
        root="data/test",
        cache_train=cache_train,
        cache_path="data/test.cache",
        transformer=get_transform(w, h)
    )

    trainloader = DataLoader(trainset, shuffle=True, batch_size=32, num_workers=2)
    testloader = DataLoader(testset, shuffle=True, batch_size=32, num_workers=2)

    model = SwinTransformer(img_size=imgsz,
                            in_chans=in_channel,
                            num_classes=num_classes,
                            embed_dim=embed_dim,
                            depths=depths,
                            num_heads=num_heads,
                            drop_path_rate=0.2
                            )

    criterion = LabelSmoothingCrossEntropy()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=decay, betas=(0.937, 0.999))
    lr_scheduler = get_lr_scheduler(optimizer, epochs=epochs)

    if pretrained_weights.endswith(".pth") and os.path.isfile(pretrained_weights):
        try:
            state_dict, start_epoch, optim, lr_, loss, acc = load_checkpoint(save_ckpt_path)
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(optim)
            lr_scheduler.load_state_dict(lr_)
            best_acc1, best_acc2 = acc
        except:
            ckpt = torch.load(pretrained_weights)["model"]
            state_dict = intersect_dicts(ckpt, model.state_dict())
            model.load_state_dict(state_dict)
            start_epoch = 1
            best_acc1 = best_acc2 = 0
    else:
        start_epoch = 1
        best_acc1 = best_acc2 = 0

    model = nn.DataParallel(model).to(device)
    criterion.to(device)

    writer = SummaryWriter(save_root) if use_tensorboard else None
    wandb_run = wandb.init(
        project="Emotional Face",
        name=save_root.split("/")[-1],
        resume="allow",
        config={
            "img size": imgsz,
            "lr": learning_rate,
            "epochs": epochs,
            "embed_dim": embed_dim,
            "depths": depths,
            "num_heads": num_heads,
            "decay": decay
        }) if use_wandb else None

    for epoch in range(start_epoch, epochs + 1):
        train_pdar = tqdm(trainloader, desc=f"Training Epoch {epoch}/{epochs}")
        train_loss, train_acc1, train_acc2 = train_one_epoch(model, train_pdar, optimizer, criterion, device)

        test_pdar = tqdm(testloader, desc=f"Evaluating")
        test_loss, test_acc1, test_acc2 = evaluate(model, test_pdar, criterion, device)

        if use_tensorboard:
            writer.add_scalars(
                "EmotionalExp",
                {
                    'train/loss': train_loss,
                    'train/acc1': train_acc1,
                    'train/acc2': train_acc2,
                    'test/loss': test_loss,
                    'test/acc1': test_acc1,
                    'test/acc2': test_acc1
                },
                epoch
            )
        if use_wandb:
            wandb_run.log({
                'train/loss': train_loss,
                'train/acc1': train_acc1,
                'train/acc2': train_acc2,
                'test/loss': test_loss,
                'test/acc1': test_acc1,
                'test/acc2': test_acc1})

        if (epoch > save_epoch) and save_strategy(test_acc1, train_acc2, best_acc1, best_acc2):
            save_checkpoint(epoch, optimizer, lr_scheduler, test_loss, model, [test_acc1, test_acc2], save_ckpt_path)

    if use_wandb:
        wandb.finish()
