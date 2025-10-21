# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import os
from config import DEVICE, MODEL_PATH, OUTPUT_CSV, IDX_TO_CLASS

# --- Test and Validation data ---
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    """ Model training and validation loop """
    best_acc = 0.0

    print(f"--- Start training on {DEVICE} ---")

    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        train_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = train_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # Save best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model: Validation Accuracy {best_acc:.4f}")

# --- Prediction and CSV output function ---
def predict_and_save_csv(model, test_loader):
    """Use the trained model to make predictions on the test set and save as CSV"""

    # Check if model weights exist and load
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model weights {MODEL_PATH} not found, will use current model for predictions (results may be inaccurate).")
    else:
        print(f"Loading best model weights: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()
    results = []

    with torch.no_grad():
        for inputs, indices in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Translate predictions to character names
            predictions = [IDX_TO_CLASS[p.item()] for p in preds.cpu()]

            # Process indices as IDs (indices start from 0, IDs start from 1)
            image_ids = [(idx.item() + 1) for idx in indices.cpu()]

            for image_id, character in zip(image_ids, predictions):
                results.append({'id': image_id, 'character': character})

    # Create DataFrame and save as CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPrediction saved to {OUTPUT_CSV}")
    # print("--- Prediction Results Preview ---")
    # print(df.head())