# predict.py
import torch
import pandas as pd
from tqdm import tqdm
import json
import os
import config
from model import EfficientNetClassifier  # or SimpsonsClassifier
from data_loader import get_test_loader

def generate_predictions(model, test_loader):
    """Generates predictions for the test set."""
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc='Predicting test set')
        for images, img_ids in test_bar:
            images = images.to(config.DEVICE)
            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)

            for i in range(len(predicted_indices)):
                predictions.append(config.IDX_TO_CLASS[predicted_indices[i].item()])
                image_ids.append(img_ids[i])
                
    return image_ids, predictions

def main():
    # 1. Check if model weights and stats exist
    if not os.path.exists(config.MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at '{config.MODEL_WEIGHTS_PATH}'. Please train the model first.")
        return
    if not os.path.exists(config.STATS_PATH):
        print(f"Error: Dataset stats not found at '{config.STATS_PATH}'. Please run calculate_stats.py first.")
        return

    # 2. Load dataset stats
    with open(config.STATS_PATH, 'r') as f:
        stats = json.load(f)
    mean, std = stats['mean'], stats['std']

    # 3. Initialize model and load weights
    model = EfficientNetClassifier().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_WEIGHTS_PATH, map_location=config.DEVICE))
    print("Model weights loaded successfully.")

    # 4. Get test dataloader
    test_loader = get_test_loader(mean, std)

    # 5. Make predictions
    print("Making predictions...")
    image_ids, predictions = generate_predictions(model, test_loader)
    
    # 6. Save predictions to CSV
    submission_df = pd.DataFrame({'id': image_ids, 'character': predictions})
    submission_df['id_int'] = submission_df['id'].astype(int)
    submission_df = submission_df.sort_values('id_int').drop('id_int', axis=1)
    
    submission_df.to_csv(config.OUTPUT_CSV_PATH, index=False)
    print(f"Predictions saved to '{config.OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()