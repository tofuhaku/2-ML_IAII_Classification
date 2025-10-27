import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import config

# Define CNN Model with Pre-trained weights (ResNet50)
class SimpsonsClassifier(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(SimpsonsClassifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False

        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Alternative EfficientNet model
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(EfficientNetClassifier, self).__init__()
        try:
            self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        except:
            # Fallback to ResNet if EfficientNet not available
            # self.backbone = models.resnet34(pretrained=True)
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)