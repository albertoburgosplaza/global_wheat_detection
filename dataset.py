import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import ast
import torch


from torch.utils.data import DataLoader
from utils import collate_fn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


import config
from utils import x1y1wh_to_x1y1x2y2


class WheatDataset(Dataset):

    def __init__(self, image_dir, image_ids, dataframe=None, transforms=None, test=False):
        super().__init__()

        self.df = dataframe
        self.image_dir = image_dir
        self.image_ids = image_ids
        if config.DEBUG:
            self.image_ids = image_ids[:round(len(image_ids)*0.05)]
        self.transforms = transforms
        self.test = test

    def __getitem__(self, idx:int):
        image_id = self.image_ids[idx]


        image = cv2.imread(f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0


        if self.test:
            if self.transforms:
                sample = self.transforms(**{
                    'image': image
                })
                image = sample['image']

            return image, image_id

        else:
            boxes = self.df.loc[self.df["image_id"] == image_id, "boxes"].apply(ast.literal_eval)

            if len(boxes) > 0:
                boxes = torch.FloatTensor(list(boxes))[0]
                boxes = x1y1wh_to_x1y1x2y2(boxes)
                labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
            else:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0), dtype=torch.int64)
            
            target = {}
            target['boxes'] = boxes
            target['labels'] = labels

            if self.transforms:
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': boxes,
                    'labels': labels
                })
                image = sample['image']
                if len(sample['bboxes']) > 0:
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).float()

            return image, target

    def __len__(self) -> int:
        return len(self.image_ids)


def get_train_data_loader(df, fold):
    train_dataset = WheatDataset(
        f"{config.DATA_PATH}/train",
        df[df["kfold"] != fold].image_id.values,
        df,
        get_train_transforms()
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    return train_data_loader


def get_valid_data_loader(df, fold):
    valid_dataset = WheatDataset(
        f"{config.DATA_PATH}/train",
        df[df["kfold"] == fold].image_id.values,
        df,
        get_valid_transforms()
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    return valid_data_loader


def get_test_data_loader():
    test_dataset = WheatDataset(
        f"{config.DATA_PATH}/test",
        [f.split(".")[:-1][0] for f in os.listdir(f"{config.DATA_PATH}/test")],
        transforms=get_test_transforms(),
        test=True
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn
    )

    return test_data_loader


def get_train_transforms():
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                 val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                       contrast_limit=0.2, p=0.9),
            ], p=0.9),
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=512, width=512, p=1.0),
        A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        ToTensorV2(p=1.0),
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transforms():
    return A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_test_transforms():
    return A.Compose([
        ToTensorV2(p=1.0)
    ])