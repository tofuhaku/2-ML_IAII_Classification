"""
Data preprocessing module for The Simpsons Character Recognition
Handles image loading, augmentation, and dataset creation
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import random
from sklearn.utils.class_weight import compute_class_weight
import config

class SimpsonsDataset(Dataset):
    """Custom Dataset class for The Simpsons characters"""

    def __init__(self, data_dir, transform=None, is_training=True):
        """
        Args:
            data_dir (str): Directory with character folders
            transform (callable, optional): Optional transform to be applied on a sample
            is_training (bool): Whether this is training data
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_training = is_training
        self.images = []
        self.labels = []

        if is_training:
            self._load_training_data()
        else:
            self._load_test_data()

    def _load_training_data(self):
        """Load training data from character folders"""
        for char_name in config.CHARACTERS:
            char_dir = os.path.join(self.data_dir, char_name)
            if os.path.exists(char_dir):
                for img_file in os.listdir(char_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(char_dir, img_file)
                        self.images.append(img_path)
                        self.labels.append(config.CHAR_TO_IDX[char_name])

        print(f"Loaded {len(self.images)} training images across {len(set(self.labels))} classes")

    def _load_test_data(self):
        """Load test data from test folder"""
        for img_file in os.listdir(self.data_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(self.data_dir, img_file)
                self.images.append(img_path)
                # For test data, we don't have labels, so we'll use -1
                self.labels.append(-1)

        print(f"Loaded {len(self.images)} test images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get a single item from the dataset"""
        img_path = self.images[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.is_training:
            label = self.labels[idx]
            return image, label
        else:
            # For test data, return image and filename
            filename = os.path.basename(img_path)
            return image, filename

class AdvancedAugmentation:
    """Advanced data augmentation for robust training"""

    def __init__(self):
        pass

    def add_noise(self, image):
        """Add Gaussian noise to image"""
        noise = torch.randn_like(image) * config.NOISE_STD
        return torch.clamp(image + noise, 0, 1)

    def __call__(self, image):
        """Apply random advanced augmentation"""
        # Add noise with 30% probability
        if random.random() < 0.3:
            image = self.add_noise(image)

        return image

def get_transforms(is_training=True):
    """Get image transforms for training or validation/test"""

    if is_training:
        # Training transforms with heavy augmentation
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomRotation(config.ROTATION_DEGREES),
            transforms.ColorJitter(
                brightness=config.COLOR_JITTER_BRIGHTNESS,
                contrast=config.COLOR_JITTER_CONTRAST,
                saturation=config.COLOR_JITTER_SATURATION,
                hue=config.COLOR_JITTER_HUE
            ),
            transforms.RandomHorizontalFlip(p=config.RANDOM_HORIZONTAL_FLIP_PROB),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            AdvancedAugmentation()
        ])
    else:
        # Validation/Test transforms - minimal preprocessing
        transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return transform

def create_data_loaders():
    """Create train and validation data loaders"""

    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)

    # Create dataset with training transforms
    train_transform = get_transforms(is_training=True)
    full_dataset = SimpsonsDataset(
        data_dir=config.TRAIN_DATA_PATH,
        transform=train_transform,
        is_training=True
    )

    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(config.VALIDATION_SPLIT * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    # Create validation dataset with different transforms
    val_transform = get_transforms(is_training=False)
    val_dataset_transformed = SimpsonsDataset(
        data_dir=config.TRAIN_DATA_PATH,
        transform=val_transform,
        is_training=True
    )

    # Apply validation indices to the transformed dataset
    val_dataset.dataset = val_dataset_transformed

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader

def create_test_loader():
    """Create test data loader"""

    test_transform = get_transforms(is_training=False)
    test_dataset = SimpsonsDataset(
        data_dir=config.TEST_DATA_PATH,
        transform=test_transform,
        is_training=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )

    print(f"Test samples: {len(test_dataset)}")

    return test_loader

def compute_class_weights(train_loader):
    """Compute class weights for handling class imbalance"""

    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())

    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )

    class_weights_tensor = torch.FloatTensor(class_weights).to(config.DEVICE)

    print(f"Computed class weights for {len(class_weights)} classes")

    return class_weights_tensor

def analyze_dataset(data_loader):
    """Analyze dataset statistics"""

    print("Dataset Analysis:")
    print("="*50)

    # Count samples per class
    class_counts = {}
    total_samples = 0

    for _, labels in data_loader:
        for label in labels:
            if label.item() != -1:  # Skip test data
                char_name = config.IDX_TO_CHAR[label.item()]
                class_counts[char_name] = class_counts.get(char_name, 0) + 1
                total_samples += 1

    print(f"Total samples: {total_samples}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"Average samples per class: {total_samples / len(class_counts):.1f}")

    # Show top and bottom classes by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 5 classes by sample count:")
    for char, count in sorted_classes[:5]:
        print(f"  {char}: {count}")

    print("\nBottom 5 classes by sample count:")
    for char, count in sorted_classes[-5:]:
        print(f"  {char}: {count}")

    return class_counts

if __name__ == "__main__":
    # Test the data loading functionality
    print("Testing data preprocessing...")

    train_loader, val_loader = create_data_loaders()
    test_loader = create_test_loader()

    # Analyze dataset
    analyze_dataset(train_loader)

    # Test loading a batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break

    print("Data preprocessing test completed!")