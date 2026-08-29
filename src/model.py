"""Model definitions for the image classifier."""

import torch.nn as nn
from torchvision import models


class SmallCNN(nn.Module):
    """A compact CNN for 32x32 CIFAR-10 images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Return a model by name.

    Supported: "resnet18" (torchvision, adapted for CIFAR-10) and "smallcnn".
    """
    name = architecture.lower()
    if name == "smallcnn":
        return SmallCNN(num_classes=num_classes)
    if name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)
        # Adapt the stem for small 32x32 inputs.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"unknown architecture: {architecture}")
