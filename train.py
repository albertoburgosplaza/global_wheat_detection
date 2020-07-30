import os
import pandas as pd


from torch.utils.data import DataLoader
from utils import collate_fn
import ast


import config
from dataset import *
from model import *
from engine import *


import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transforms():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def run_training(fold:int):

    df = pd.read_csv(os.path.join(config.DATA_PATH, "train_folds.csv"))

    train_dataset = WheatDataset(
        df,
        f"{config.DATA_PATH}/train",
        df[df["kfold"] != fold].image_id.values,
        get_train_transforms()
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    valid_dataset = WheatDataset(
        df,
        f"{config.DATA_PATH}/train",
        df[df["kfold"] == fold].image_id.values,
        get_valid_transforms()
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )


    model = WheatModel(config.NUM_CLASSES)
    model.to(config.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    lr = config.LR
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000000, gamma=0.1)

    epochs = config.EPOCHS
    epoch = 1
    for e in range(0,epochs):
        model.train()
        print(f'Starting epoch {epoch}')
        loss = train_one_epoch(model, train_data_loader, optimizer, print_freq=50) / len(train_data_loader)
        print(f'Epoch {epoch}, loss: {loss}')
        epoch = epoch + 1


if __name__ == "__main__":
    fold = 0
    run_training(fold)