# data_loader.py

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import os
from config import IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT, CLASS_TO_IDX

# --- Data Transforms ---
def get_train_transforms():
    """ Training data augmentations and normalization """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        # ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_test_transforms():
    """ Validation and test data normalization (no random augmentations) """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- Custom test dataset ---
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get all image file names and sort them to ensure id correspondence
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # return image, img_name # Return image tensor and file name
        return image, idx # Return image tensor and file index

# --- DataLoader creation function ---
def get_data_loaders(train_dir, test_dir):
    """Load train/val/test datasets and create DataLoaders"""

    # 1. Train/validation set processing
    full_train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms()
    )
    full_train_dataset.class_to_idx = CLASS_TO_IDX

    # Calculate split sizes
    train_size = int((1 - VAL_SPLIT) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    # Random split
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Set validation set transforms (must reassign non-random transforms)
    val_dataset.dataset.transform = get_val_test_transforms()

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. Test set processing
    test_dataset = TestDataset(root_dir=test_dir, transform=get_val_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader