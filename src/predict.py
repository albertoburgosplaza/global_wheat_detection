import torch
import pandas as pd
from ensemble_boxes import *
import numpy as np
import gc


import config
from dataset import get_test_data_loader
from model import load_trained_model
from utils import format_prediction_string, make_ensemble_predictions, run_wbf, save_labeled_image


models = []

def run_predict():
    test_data_loader = get_test_data_loader()

    for fold in range(0, config.NUM_FOLDS):
        models.append(load_trained_model(f"checkpoint-f{fold}.pt"))

    results = []

    for images, image_ids in test_data_loader:
        predictions = make_ensemble_predictions(images, models)
        for i, image in enumerate(images):
            boxes, scores, labels = run_wbf(predictions, image_index=i)
            boxes = boxes.astype(np.int32)
            image_id = image_ids[i]

            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }
            results.append(result)

            save_labeled_image(image, boxes, image_id)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv('submission.csv', index=False)


def run_predict_one_fold(fold:int):
    test_data_loader = get_test_data_loader()

    model = load_trained_model(f"checkpoint-f{fold}.pt")

    results = []

    for images, image_ids in test_data_loader:
        images = list(image.to(config.DEVICE) for image in images)
        outputs = model(images)
        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].cpu().detach().numpy()
            scores = outputs[i]['scores'].cpu().detach().numpy()
            boxes = boxes[scores > config.SCORE_THRESHOLD]
            scores = scores[scores > config.SCORE_THRESHOLD]
            image_id = image_ids[i]

            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }

            results.append(result)

            save_labeled_image(image, boxes, image_id)
    
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    run_predict()
    # run_predict_one_fold(fold=1)