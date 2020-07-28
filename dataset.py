import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import ast
import torch

class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, image_ids, transforms=None, test=False):
        super().__init__()

        self.df = dataframe
        self.image_dir = image_dir
        self.image_ids = image_ids
        self.transforms = transforms
        self.test = test

    def __getitem__(self, idx:int):
        image_id = self.image_ids[idx]
        boxes = self.df.loc[self.df["image_id"] == image_id, "bboxes"].apply(ast.literal_eval)

        if len(boxes) > 0:
            boxes = torch.FloatTensor(list(boxes))
            # x1y1wh_to_x1y1x2y2
            x1 = boxes[:,0]
            y1 = boxes[:,1]
            w = boxes[:,2]
            h = boxes[:,3]
            x2 = x1 + w
            y2 = y1 + h
            boxes = torch.stack([x1,y1,x2,y2], dim=1)
            labels = torch.zeros([boxes.shape[0]], dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0), dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        image = cv2.imread(f"{self.image_dir}/{image_id}.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)