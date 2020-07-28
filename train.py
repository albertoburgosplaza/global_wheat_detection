import os
import pandas as pd
from dataset import WheatDataset
from torch.utils.data import DataLoader
from utils import collate_fn
import ast


DATA_PATH = "C:/Users/alberto/Documents/datasets/global-wheat-detection"
fold = 0


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_PATH, "train_folds.csv"))

    train_dataset = WheatDataset(
        df,
        f"{DATA_PATH}/train",
        df[df["kfold"] != fold].image_id.values
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_dataset = WheatDataset(
        df,
        f"{DATA_PATH}/train",
        df[df["kfold"] == fold].image_id.values
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    image, target = next(iter(train_data_loader))
    print(target)