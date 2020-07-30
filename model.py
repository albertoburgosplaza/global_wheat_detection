from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class WheatModel(nn.Module):
    def __init__(self, num_classes):
        super(WheatModel, self).__init__()

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        loss_dict = self.model(images, targets)
        return loss_dict