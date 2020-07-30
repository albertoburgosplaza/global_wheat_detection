import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import ast
import torch


import config
from utils import x1y1wh_to_x1y1x2y2


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, image_ids, transforms=None, test=False):
        super().__init__()

        self.df = dataframe
        self.image_dir = image_dir
        self.image_ids = image_ids
        if config.DEBUG:
            self.image_ids = image_ids[:round(len(image_ids)*.1)]
        self.transforms = transforms
        self.test = test

    def __getitem__(self, idx:int):
        image_id = self.image_ids[idx]


        image = cv2.imread(f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0


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
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0).float()

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)