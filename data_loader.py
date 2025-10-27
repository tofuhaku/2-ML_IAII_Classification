# data_loader.py
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import config

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class AddSpeckleNoise(object):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_level
        noisy_tensor = tensor * (1 + noise)
        return torch.clamp(noisy_tensor, 0, 1)

class AddPoissonNoise(object):
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, tensor):
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))
        noisy_tensor = tensor + noise / 255.0
        return torch.clamp(noisy_tensor, 0, 1)

class AddSaltPepperNoise(object):
    def __init__(self, salt_prob=0.05, pepper_prob=0.05):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob

    def __call__(self, tensor):
        noise = torch.rand(tensor.size())
        tensor = tensor.clone()
        tensor[noise < self.salt_prob] = 1
        tensor[noise > 1 - self.pepper_prob] = 0
        return tensor

class AddRandomBackground(object):
    def __init__(self, backgrounds_dir):
        self.backgrounds_dir = backgrounds_dir
        if not os.path.isdir(backgrounds_dir):
            raise FileNotFoundError(f"Backgrounds directory not found: {backgrounds_dir}")
        self.background_files = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)]
        if not self.background_files:
            raise ValueError(f"No background images found in {backgrounds_dir}.")

    def __call__(self, img):
        img_rgba = img.convert("RGBA")
        mask = Image.new("L", img_rgba.size, 0)
        for x in range(img_rgba.width):
            for y in range(img_rgba.height):
                r, g, b, a = img_rgba.getpixel((x, y))
                if (r < 20 and g < 20 and b < 20) or (r > 235 and g > 235 and b > 235):
                    mask.putpixel((x, y), 0)
                else:
                    mask.putpixel((x, y), 255)

        bg_path = random.choice(self.background_files)
        background = Image.open(bg_path).convert("RGBA")
        
        padding_x = random.randint(10, 30)
        padding_y = random.randint(10, 30)
        new_size = (img_rgba.width + 2 * padding_x, img_rgba.height + 2 * padding_y)
        
        background = background.resize(new_size, Image.LANCZOS)
        paste_position = (padding_x, padding_y)
        background.paste(img_rgba, paste_position, mask)
        return background.convert("RGB")

class SimpsonsDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []

        if is_train:
            for char_name in config.CHARACTER_NAMES:
                char_dir = os.path.join(data_dir, char_name)
                if os.path.exists(char_dir):
                    for img_file in os.listdir(char_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append(os.path.join(char_dir, img_file))
                            self.labels.append(config.CLASS_TO_IDX[char_name])
        else: # Test set
            for img_file in sorted(os.listdir(data_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(data_dir, img_file))
                    self.labels.append(os.path.splitext(img_file)[0]) # Image ID

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# --- Data Transformation and Loader Functions ---
def get_transforms(mean, std):
    """Returns train and validation/test transforms."""
    train_transform = transforms.Compose([
        transforms.RandomApply([AddRandomBackground(config.BACKGROUND_DIR)], p=0.7),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply([transforms.RandomRotation(20)], p=0.2),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.RandomApply([AddGaussianNoise(0., 0.07)], p=0.2),
        AddGaussianNoise(0., 0.001),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(mean, std):
    """Creates and returns training and validation dataloaders."""
    train_transform, val_transform = get_transforms(mean, std)

    train_full_aug = SimpsonsDataset(config.TRAIN_PATH, transform=train_transform, is_train=True)
    val_full_clean = SimpsonsDataset(config.TRAIN_PATH, transform=val_transform, is_train=True)

    indices = list(range(len(train_full_aug)))
    train_idx, val_idx = train_test_split(indices, test_size=config.VAL_SPLIT, random_state=42)
    
    train_subset = Subset(train_full_aug, train_idx)
    val_subset = Subset(val_full_clean, val_idx)

    print(f"Total training samples: {len(train_full_aug)}")
    print(f"Training subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")

    train_loader = DataLoader(train_subset, config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_subset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader

def get_test_loader(mean, std):
    """Creates and returns the test dataloader."""
    _, test_transform = get_transforms(mean, std)
    
    test_dataset = SimpsonsDataset(config.TEST_PATH, transform=test_transform, is_train=False)
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return test_loader