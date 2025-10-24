import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')

TRAIN_PATH = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\train'
TEST_PATH = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\test-renamed_images'
BACKGROUND_DIR = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\background'

# Define character names
CHARACTER_NAMES = [
    'abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble',
    'bart_simpson', 'brandine_spuckler', 'carl_carlson', 'charles_montgomery_burns',
    'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'disco_stu',
    'dolph_starbeam', 'duff_man', 'edna_krabappel', 'fat_tony',
    'gary_chalmers', 'gil', 'groundskeeper_willie', 'homer_simpson',
    'jimbo_jones', 'kearney_zzyzwicz', 'kent_brockman', 'krusty_the_clown',
    'lenny_leonard', 'lionel_hutz', 'lisa_simpson', 'lunchlady_doris',
    'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby',
    'milhouse_van_houten', 'miss_hoover', 'moe_szyslak', 'ned_flanders',
    'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner',
    'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier',
    'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'timothy_lovejoy',
    'troy_mcclure', 'waylon_smithers'
]
BATCH_SIZE = 128

# Create character to index mapping
char_to_idx = {char: idx for idx, char in enumerate(CHARACTER_NAMES)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

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
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

class AddPoissonNoise(object):
    def __init__(self, lam=1.0):
        self.lam = lam

    def __call__(self, tensor):
        noise = torch.poisson(self.lam * torch.ones(tensor.shape))
        noisy_tensor = tensor + noise / 255.0
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

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
            raise FileNotFoundError(f"Backgrounds directory not found at {backgrounds_dir}. Please check the BACKGROUNDS_DIR variable.")
        self.background_files = [os.path.join(backgrounds_dir, f) for f in os.listdir(backgrounds_dir)]
        if not self.background_files:
            raise ValueError(f"No images found in {backgrounds_dir}.")

    def __call__(self, img):
        # Convert character image to RGBA
        img_rgba = img.convert("RGBA")

        # Create a mask from the black and white background by iterating pixels
        mask = Image.new("L", img_rgba.size, 0)
        for x in range(img_rgba.width):
            for y in range(img_rgba.height):
                r, g, b, a = img_rgba.getpixel((x, y))
                # Threshold for near-black or near-white
                if (r < 20 and g < 20 and b < 20) or (r > 235 and g > 235 and b > 235):
                    mask.putpixel((x, y), 0)  # Transparent
                else:
                    mask.putpixel((x, y), 255) # Opaque

        # Choose a random background
        bg_path = random.choice(self.background_files)
        background = Image.open(bg_path).convert("RGBA")

        # Add random padding
        padding_x = random.randint(10, 30)
        padding_y = random.randint(10, 30)
        new_size = (img_rgba.width + 2 * padding_x, img_rgba.height + 2 * padding_y)

        # Resize background to the new size (character + padding)
        background = background.resize(new_size, Image.LANCZOS)

        # Define paste position (top-left corner)
        paste_position = (padding_x, padding_y)
        
        # Paste the character image onto the background using the generated mask
        background.paste(img_rgba, paste_position, mask)

        return background.convert("RGB")

# Data transforms
train_transform = transforms.Compose([
    AddRandomBackground(BACKGROUND_DIR),
    transforms.Resize((224, 224)),
    
    # Geometric
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.6, p=1.0)], p=0.3),
    transforms.RandomApply([transforms.RandomAffine(degrees=(-70, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))], p=0.3),
    transforms.RandomApply([transforms.ElasticTransform(alpha=100.0)], p=0.2),

    # Color and Brightness
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.5),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomInvert(p=0.1),
    transforms.RandomApply([transforms.RandomPosterize(bits=4)], p=0.2),
    transforms.RandomApply([transforms.RandomSolarize(threshold=0.5)], p=0.1),
    transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.3),

    # Filtering
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),

    # Convert to Tensor
    transforms.ToTensor(),

    # Noise (Applied on Tensor)
    transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.2),
    transforms.RandomApply([AddPoissonNoise(lam=0.1)], p=0.2),
    transforms.RandomApply([AddSpeckleNoise(noise_level=0.1)], p=0.2),
    transforms.RandomApply([AddSaltPepperNoise(salt_prob=0.03, pepper_prob=0.03)], p=0.2),
    AddGaussianNoise(0., 0.001), 

    # Normalize (Must be last)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class SimpsonsDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []

        if is_train:
            # Load training data
            for char_name in CHARACTER_NAMES:
                char_dir = os.path.join(data_dir, char_name)
                if os.path.exists(char_dir):
                    for img_file in os.listdir(char_dir):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.images.append(os.path.join(char_dir, img_file))
                            self.labels.append(char_to_idx[char_name])
        else:
            # Load test data
            for img_file in sorted(os.listdir(data_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(data_dir, img_file))
                    # Extract ID from filename
                    img_id = os.path.splitext(img_file)[0]
                    self.labels.append(img_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            return image, self.labels[idx]
        else:
            return image, self.labels[idx]  # Return image ID for test set

# Define CNN Model with Pre-trained weights
class SimpsonsClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpsonsClassifier, self).__init__()
        # Use ResNet50 as backbone with pretrained weights
        # self.backbone = models.resnet50(pretrained=True)
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
    def __init__(self, num_classes=50):
        super(EfficientNetClassifier, self).__init__()
        try:
            # from torchvision.models import efficientnet_b0
            # self.backbone = efficientnet_b0(pretrained=True)
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

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_train / total_train:.2f}%'
            })

        train_acc = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct_val / total_val:.2f}%'
                })

        val_acc = 100 * correct_val / total_val
        val_loss = val_loss / len(val_loader)

        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step()

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_simpsons_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

    return train_losses, val_losses, train_accs, val_accs

def predict_test_set(model, test_loader):
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Predicting test set')
        for images, img_ids in test_bar:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(predicted)):
                predictions.append(idx_to_char[predicted[i].item()])
                image_ids.append(img_ids[i])

    return image_ids, predictions

def main():
    print("Loading datasets...")

    # Load training dataset
    train_dataset = SimpsonsDataset(TRAIN_PATH, transform=train_transform, is_train=True)
    print(f"Total training samples: {len(train_dataset)}")

    # Split training data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Enhanced splitting with different transforms
    train_full_aug = SimpsonsDataset(TRAIN_PATH, transform=train_transform, is_train=True)
    val_full_clean = SimpsonsDataset(TRAIN_PATH, transform=val_transform, is_train=True)
    train_indices = list(range(len(train_full_aug)))
    train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=42)
    train_subset = torch.utils.data.Subset(train_full_aug, train_idx)
    val_subset = torch.utils.data.Subset(val_full_clean, val_idx)

    # Create validation dataset with different transforms
    # val_dataset = SimpsonsDataset('dataset/train', transform=val_transform, is_train=True)
    # val_subset.dataset = val_dataset

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")

    # Create data loaders
    train_loader = DataLoader(train_subset, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model
    print("Initializing model...")
    model = SimpsonsClassifier(num_classes=50).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("Starting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, num_epochs=50, lr=0.001
    )

    # Load best model for testing
    print("Loading best model for testing...")
    model.load_state_dict(torch.load('best_simpsons_model.pth'))

    # Load test dataset
    test_dataset = SimpsonsDataset(TEST_PATH, transform=test_transform, is_train=False)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Test samples: {len(test_dataset)}")

    # Make predictions
    print("Making predictions on test set...")
    image_ids, predictions = predict_test_set(model, test_loader)

    # Create submission CSV
    submission_df = pd.DataFrame({
        'id': image_ids,
        'character': predictions
    })

    # Sort by ID (convert to int for proper sorting)
    submission_df['id_int'] = submission_df['id'].astype(int)
    submission_df = submission_df.sort_values('id_int').drop('id_int', axis=1)

    submission_df.to_csv('submission.csv', index=False)
    print("Predictions saved to 'submission.csv'")

    # Print some sample predictions
    print("\nSample predictions:")
    # print(submission_df.head(10))

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    # plt.show()

    print(f"Best validation accuracy: {max(val_accs):.2f}%")
    print("Training completed!")

if __name__ == "__main__":
    main()