import torch
import pandas as pd


import config
from dataset import get_test_data_loader
from model import WheatModel
from utils import format_prediction_string


def run_predict(fold:int):
    test_data_loader = get_test_data_loader()

    model = WheatModel(config.NUM_CLASSES, pretrained=False)
    model.to(config.DEVICE)
    model.load_state_dict(torch.load(f"checkpoint-f{fold}.pt"))

    print(f"Starting prediction")
    model.eval()

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
    
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    run_predict(fold=0)