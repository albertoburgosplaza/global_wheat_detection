import os
import ast
import pandas as pd
from sklearn import model_selection

import config


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(config.DATA_PATH, "train.csv"))
    df.bbox = df.bbox.apply(ast.literal_eval)
    df = (
        df.groupby(["image_id", "source"])["bbox"].apply(list).reset_index(name="boxes")
    )
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    source = df.source.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for fold, (_, val_) in enumerate(kf.split(X=df, y=source)):
        df.loc[val_, "kfold"] = fold
    df.to_csv(os.path.join(config.DATA_PATH, "train_folds.csv"), index=False)
