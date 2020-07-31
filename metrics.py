import numpy as np
import config


def intersection_over_union(x1_p, y1_p, x2_p, y2_p, x1_g, y1_g, x2_g, y2_g):
    # Determine the (x, y)-coordinates of the intersection rectangle
    x1_i = max(x1_p, x1_g)
    y1_i = max(y1_p, y1_g)
    x2_i = min(x2_p, x2_g)
    y2_i = min(y2_p, y2_g)

    # Compute the area of intersection rectangle
    i_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    
    # Compute the area of both the prediction and ground-truth rectangles
    p_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    g_area = (x2_g - x1_g + 1) * (y2_g - y1_g + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = i_area / float(p_area + g_area - i_area)

    return iou


def average_precision(boxes, boxes_gt):
    iou = []

    for p in range(0, len(boxes)):
        x1_p, y1_p, x2_p, y2_p = boxes[p]
        for g in range(0, len(boxes_gt)):
            x1_g, y1_g, x2_g, y2_g = boxes_gt[g]
            iou.append(intersection_over_union(x1_p, y1_p, x2_p, y2_p, x1_g, y1_g, x2_g, y2_g))

    iou = np.array(iou)
    iou = iou.reshape(len(boxes), len(boxes_gt))
    ap = 0
    
    for t in np.linspace(.5, .75, 6):
        tp = 0
        fp = 0
        fn = 0

        for p in range(0, len(boxes)):
            hits = (iou > t)[p,:].sum()
            if hits:
                tp = tp + 1
                fp = fp + hits - 1
            else:
                fp = fp + 1

        for g in range(0, len(boxes_gt)):
            hits = (iou > t)[:,g].sum()
            if not(hits):
                fn = fn + 1
                
        ap = ap + (tp / (tp + fp + fn))
        
    ap = ap / 6

    return ap


def batch_average_precision(outputs, targets):
    m_ap_batch = 0

    for i in range(0, len(outputs)):
        scores = outputs[i]['scores'].cpu().detach().numpy()
        boxes = outputs[i]['boxes'].cpu().detach().numpy()
        boxes = boxes[scores > config.SCORE_THRESHOLD]
        boxes_gt = targets[i]['boxes'].cpu().detach().numpy()
        
        m_ap_batch += average_precision(boxes, boxes_gt)
            
    return m_ap_batch / len(outputs)