# main.py

import torch
import torch.optim as optim
import torch.nn as nn
from config import DEVICE, TRAIN_DIR, TEST_DIR, LEARNING_RATE, NUM_EPOCHS, MODEL_PATH
from data_loader import get_data_loaders
from model import get_simpsons_cnn
from train import train_model, predict_and_save_csv
import os
import sys

def main():
    print(f"--- The Simpsons Character Recognizer Starting (Using {DEVICE}) ---")

    # 1. Load Data
    try:
        train_loader, val_loader, test_loader = get_data_loaders(TRAIN_DIR, TEST_DIR)
        print(f"Dataset loaded successfully: Train ({len(train_loader.dataset)}), Val ({len(val_loader.dataset)}), Test ({len(test_loader.dataset)})")
    except Exception as e:
        print(f"Fatal error: Data loading failed. Please check TRAIN_DIR ({TRAIN_DIR}) and TEST_DIR ({TEST_DIR}) for correctness.")
        print(f"Error details: {e}")
        sys.exit(1)

    # 2. Prepare model, loss function, and optimizer
    model = get_simpsons_cnn().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Train model
    # Train only if model weights do not exist
    if not os.path.exists(MODEL_PATH):
        print("\n--- Start training ---")
        train_model(model, criterion, optimizer, train_loader, val_loader, NUM_EPOCHS)
    else:
        print(f"\n--- Model weights {MODEL_PATH} already exist, skipping training phase ---")

    # 4. Predict on test set and output CSV
    print("\n--- Start predicting on test set ---")
    predict_and_save_csv(model, test_loader)

    print("--- Training and prediction process completed ---")

if __name__ == '__main__':
    # Set PyTorch start method to avoid potential issues with multi-processing data loading
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()