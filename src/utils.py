import torch
import config
import numpy as np
import cv2
from ensemble_boxes import weighted_boxes_fusion


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


def format_prediction_string(boxes, scores):
    pred_strings = []
    for s, b in zip(scores, boxes.astype(int)):
        pred_strings.append(f'{s:.4f} {b[0]} {b[1]} {b[2] - b[0]} {b[3] - b[1]}')

    return " ".join(pred_strings)


def make_ensemble_predictions(images, models):
    images = list(image.to(config.DEVICE) for image in images)    
    result = []
    for net in models:
        with torch.no_grad():
            outputs = net(images)
        result.append(outputs)

    return result


def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.55, skip_box_thr=0.7, weights=None):
    boxes = [prediction[image_index]['boxes'].data.cpu().numpy()/(image_size-1) for prediction in predictions]
    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels


def draw_bounding_boxes(image, boxes, labels=None, color=(0,255,0)):
    for b in boxes:
        x = b[0]
        y = b[1]
        w = b[2]
        h = b[3]
        cv2.rectangle(image, (x,y), (x+w,y+h), color, 2)

    return image


def save_labeled_image(image, boxes, image_id):
    boxes = x1y1x2y2_to_x1y1wh(boxes)
    image = np.ascontiguousarray(image.permute(1, 2, 0).cpu().numpy())
    image = draw_bounding_boxes(image, boxes)
    cv2.imwrite(f"{image_id}.png", cv2.cvtColor(image*255.0, cv2.COLOR_BGR2RGB))