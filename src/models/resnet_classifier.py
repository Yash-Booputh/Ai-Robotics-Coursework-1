"""
ResNet Classifier Module

Provides ResNet34 and ResNet50 models with transfer learning
for office items classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Literal


class ResNetClassifier(nn.Module):

    def __init__(
            self,
            num_classes: int,
            model_name: Literal['resnet34', 'resnet50'] = 'resnet34',
            pretrained: bool = True
    ):
        super(ResNetClassifier, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        if model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self._freeze_base_layers()

    def _freeze_base_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def create_resnet_model(num_classes: int, model_name: str = 'resnet34', pretrained: bool = True):
    return ResNetClassifier(num_classes=num_classes, model_name=model_name, pretrained=pretrained)