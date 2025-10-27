# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import config
from model import EfficientNetClassifier # or SimpsonsClassifier
from data_loader import get_dataloaders
from calculate_stats import calculate_and_save_stats

def train_model(model, train_loader, val_loader):
    """Main training loop."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
        
        for images, labels in train_bar:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100 * correct_train / total_train:.2f}%'})

        # Validation phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]')
        
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100 * correct_val / total_val:.2f}%'})

        # Record metrics for the epoch
        train_acc = 100 * correct_train / total_train
        train_loss = running_loss / len(train_loader)
        val_acc = 100 * correct_val / total_val
        val_loss = val_loss / len(val_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step()
        print(f'\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_WEIGHTS_PATH)
            print(f'New best model saved with Val Acc: {val_acc:.2f}%')
        print('-' * 70)
            
    return history

def plot_curves(history):
    """Plots training and validation curves."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved to 'training_curves.png'")

def main():
    # 1. Load or calculate dataset stats
    if os.path.exists(config.STATS_PATH):
        with open(config.STATS_PATH, 'r') as f:
            stats = json.load(f)
        mean, std = stats['mean'], stats['std']
        print(f"Loaded stats from {config.STATS_PATH}: Mean={mean}, Std={std}")
    else:
        mean, std = calculate_and_save_stats()

    # 2. Get Dataloaders
    train_loader, val_loader = get_dataloaders(mean, std)
    
    # 3. Initialize Model
    model = EfficientNetClassifier().to(config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # 4. Train model
    print("Starting training...")
    history = train_model(model, train_loader, val_loader)
    
    # 5. Plot and save results
    plot_curves(history)
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print("Training completed!")

if __name__ == "__main__":
    main()