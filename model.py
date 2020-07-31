import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import config


class WheatModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(WheatModel, self).__init__()

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained_backbone=pretrained,
            pretrained=pretrained,
        )

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        loss_dict = self.model(images, targets)
        return loss_dict


def load_trained_model(checkpoint_path):
    model = WheatModel(config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    model.to(config.DEVICE)

    return model