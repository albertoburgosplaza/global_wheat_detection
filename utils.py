import torch


def collate_fn(batch):
    return tuple(zip(*batch))


def x1y1x2y2_to_x1y1wh(boxes):
    """
    Converts boxes from x1y1x2y2 (upper left, bottom right) format 
    to x1y1wh (upper left, width and height) format
    """
    boxes[:,2] = boxes[:,2] - boxes[:,0]
    boxes[:,3] = boxes[:,3] - boxes[:,1]
    return boxes

def x1y1wh_to_x1y1x2y2(boxes):
    """
    Converts boxes from x1y1wh (upper left, width and height) format 
    to x1y1x2y2 (upper left, bottom right) format
    """
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    w = boxes[:,2]
    h = boxes[:,3]
    x2 = x1 + w
    y2 = y1 + h
    boxes = torch.stack([x1,y1,x2,y2], dim=1)
    return boxes