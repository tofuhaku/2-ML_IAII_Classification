"""
CNN Model architecture for The Simpsons Character Recognition
Uses ResNet as backbone with pre-training and fine-tuning capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config

class SimpsonsClassifier(nn.Module):
    """
    CNN Classifier for The Simpsons characters using pre-trained ResNet
    """

    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True, freeze_backbone=True):
        """
        Args:
            num_classes (int): Number of character classes
            pretrained (bool): Whether to use pre-trained weights
            freeze_backbone (bool): Whether to freeze backbone layers initially
        """
        super(SimpsonsClassifier, self).__init__()

        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)

        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        self.backbone.fc = nn.Identity()  # Remove original FC layer

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(256, num_classes)
        )

        # Initialize classifier weights
        self._initialize_classifier_weights()

        print(f"Initialized SimpsonsClassifier with {num_classes} classes")
        print(f"Backbone frozen: {freeze_backbone}")
        print(f"Using pre-trained weights: {pretrained}")

    def _initialize_classifier_weights(self):
        """Initialize classifier weights using Xavier initialization"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        # Extract features using backbone
        features = self.backbone(x)

        # Apply classifier
        output = self.classifier(features)

        return output

    def unfreeze_backbone(self):
        """Unfreeze backbone layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone layers unfrozen for fine-tuning")

    def freeze_backbone(self):
        """Freeze backbone layers"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone layers frozen")

    def get_feature_extractor(self):
        """Get feature extractor (backbone without classifier)"""
        return self.backbone

    def get_classifier(self):
        """Get classifier head"""
        return self.classifier

class CustomCNN(nn.Module):
    """
    Custom CNN architecture for comparison with transfer learning approach
    """

    def __init__(self, num_classes=config.NUM_CLASSES):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),

            # Fifth block
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        # Calculate the size of flattened features
        # For input size 224x224, after 5 maxpool operations (each /2): 224/32 = 7
        self.feature_size = 512 * 7 * 7

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(4096, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

        print(f"Initialized CustomCNN with {num_classes} classes")

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """

    def __init__(self, alpha=1, gamma=2, num_classes=config.NUM_CLASSES):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions from model (before softmax) [N, C]
            targets: ground truth labels [N]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for better generalization
    """

    def __init__(self, num_classes=config.NUM_CLASSES, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions from model [N, C]
            targets: ground truth labels [N]
        """
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)

        # Apply label smoothing
        targets_smooth = (1 - self.smoothing) * targets_one_hot + self.smoothing / self.num_classes

        loss = -torch.sum(targets_smooth * log_probs, dim=1).mean()
        return loss

def create_model(model_type='resnet', pretrained=True, freeze_backbone=True):
    """
    Create and return the specified model

    Args:
        model_type (str): Type of model ('resnet' or 'custom')
        pretrained (bool): Whether to use pre-trained weights
        freeze_backbone (bool): Whether to freeze backbone initially

    Returns:
        nn.Module: The created model
    """

    if model_type.lower() == 'resnet':
        model = SimpsonsClassifier(
            num_classes=config.NUM_CLASSES,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
    elif model_type.lower() == 'custom':
        model = CustomCNN(num_classes=config.NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Move model to device
    model = model.to(config.DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

def get_loss_function(loss_type='cross_entropy', class_weights=None):
    """
    Get the specified loss function

    Args:
        loss_type (str): Type of loss function
        class_weights (torch.Tensor): Class weights for handling imbalance

    Returns:
        nn.Module: The loss function
    """

    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def count_parameters(model):
    """Count the number of parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")

    # Test ResNet-based model
    print("\n" + "="*50)
    print("ResNet-based Model:")
    resnet_model = create_model('resnet', pretrained=True, freeze_backbone=True)

    # Test custom CNN
    print("\n" + "="*50)
    print("Custom CNN Model:")
    custom_model = create_model('custom')

    # Test forward pass
    print("\n" + "="*50)
    print("Testing forward pass...")
    dummy_input = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)

    with torch.no_grad():
        resnet_output = resnet_model(dummy_input)
        custom_output = custom_model(dummy_input)

    print(f"ResNet output shape: {resnet_output.shape}")
    print(f"Custom CNN output shape: {custom_output.shape}")

    # Test loss functions
    print("\n" + "="*50)
    print("Testing loss functions...")
    dummy_targets = torch.randint(0, config.NUM_CLASSES, (2,)).to(config.DEVICE)

    ce_loss = get_loss_function('cross_entropy')
    focal_loss = get_loss_function('focal')
    ls_loss = get_loss_function('label_smoothing')

    print(f"Cross Entropy Loss: {ce_loss(resnet_output, dummy_targets).item():.4f}")
    print(f"Focal Loss: {focal_loss(resnet_output, dummy_targets).item():.4f}")
    print(f"Label Smoothing Loss: {ls_loss(resnet_output, dummy_targets).item():.4f}")

    print("\nModel testing completed!")