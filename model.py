# model.py

import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES

def get_simpsons_cnn():
    """
    Load ResNet-50 pre-trained model and modify the final fully connected layer to adapt to 50 classes.
    """
    # Load ResNet-50 pre-trained model on ImageNet
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # --- Transfer Learning: Adjust Output Layer ---
    num_ftrs = model.fc.in_features
    # Set new fully connected layer
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    return model