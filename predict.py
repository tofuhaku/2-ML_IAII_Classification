import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm import tqdm
import sys

# --- CONFIGURATION ---
# ‼️ Please ensure these paths are correct
TEST_PATH = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\test-renamed_images'
MODEL_WEIGHTS_PATH = 'best_simpsons_model.pth'
OUTPUT_CSV_PATH = 'submission.csv'
    
BATCH_SIZE = 128
NUM_WORKERS = 8

# --- CHARACTER MAPPING ---
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
idx_to_char = {idx: char for idx, char in enumerate(CHARACTER_NAMES)}


# ✅ SOLUTION: Move class definitions to the top level (outside of any function)

class SimpsonsDataset(Dataset):
    """Dataset for loading Simpsons test images."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.image_ids = []
        
        # Sort files to ensure consistent order
        for img_file in sorted(os.listdir(data_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(data_dir, img_file))
                img_id = os.path.splitext(img_file)[0]
                self.image_ids.append(img_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_ids[idx]

class SimpsonsClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(SimpsonsClassifier, self).__init__()
        # self.backbone = models.resnet50(pretrained=True)
        self.backbone = models.resnet50(weights=None)

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

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super(EfficientNetClassifier, self).__init__()
        # Load architecture without pre-trained weights, as we will load our own file
        self.backbone = efficientnet_b0(weights=None) 
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)


def main():
    """
    Main function to run the prediction process.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Check for required files and directories
    if not os.path.exists(TEST_PATH):
        print(f"Error: Test directory not found at '{TEST_PATH}'")
        sys.exit(1)
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights file not found at '{MODEL_WEIGHTS_PATH}'")
        sys.exit(1)
        
    # Define the same transformations as used for validation/testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.321, 0.2833, 0.2236], std=[0.3274, 0.2947, 0.2796])
    ])

    
    # Load data
    print("Loading test dataset...")
    test_dataset = SimpsonsDataset(TEST_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Found {len(test_dataset)} images in the test set.")

    # Initialize model and load weights
    print("Initializing model...")
    model = EfficientNetClassifier(num_classes=len(CHARACTER_NAMES)).to(device)
    # model = SimpsonsClassifier(num_classes=len(CHARACTER_NAMES)).to(device)
    
    print(f"Loading model weights from '{MODEL_WEIGHTS_PATH}'...")
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    
    # Run prediction
    model.eval()
    predictions = []
    image_ids = []

    print("Starting prediction...")
    with torch.no_grad():
        for images, ids_batch in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)
            
            predicted_chars = [idx_to_char[idx.item()] for idx in predicted_indices]
            
            predictions.extend(predicted_chars)
            image_ids.extend(ids_batch)
            
    # Generate and save submission file
    print("Generating submission file...")
    submission_df = pd.DataFrame({'id': image_ids, 'character': predictions})
    submission_df['id_int'] = submission_df['id'].astype(int)
    submission_df = submission_df.sort_values('id_int').drop('id_int', axis=1)

    submission_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nPrediction complete! Results saved to '{OUTPUT_CSV_PATH}'")


if __name__ == "__main__":
    main()