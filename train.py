import os
import pandas as pd
import torch


import config
from dataset import get_train_data_loader, get_valid_data_loader
from model import WheatModel
from engine import train_one_epoch, evaluate
from callbacks import EarlyStopping


def run_training(fold:int):

    df = pd.read_csv(os.path.join(config.DATA_PATH, "train_folds.csv"))

    train_data_loader = get_train_data_loader(df, fold)
    valid_data_loader = get_valid_data_loader(df, fold)

    model = WheatModel(config.NUM_CLASSES)
    model.to(config.DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=config.LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000000, gamma=0.1)
    es = EarlyStopping(patience=5, mode="max", verbose=True)

    print_freq = round(len(train_data_loader)/4)
    for epoch in range(0, config.EPOCHS):
        loss = train_one_epoch(model, train_data_loader, optimizer, epoch, print_freq)
        m_ap = evaluate(model, valid_data_loader, epoch)

        es(m_ap, model)
        if es.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    run_training(fold=0)