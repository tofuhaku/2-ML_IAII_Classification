"""
Training script for The Simpsons Character Recognition
Handles model training with pre-training, fine-tuning, and early stopping
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import config
from model import create_model, get_loss_function
from data_preprocess import create_data_loaders, compute_class_weights

class EarlyStopping:
    """Early stopping to avoid overfitting"""

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class Trainer:
    """Main trainer class for The Simpsons classifier"""

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = config.DEVICE

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # Mixed precision training for GPU acceleration
        self.use_mixed_precision = self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            print("✅ Mixed precision training enabled")
        else:
            self.scaler = None
            print("Mixed precision disabled (CPU mode)")

        # Create directories
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)

        print(f"Trainer initialized on device: {self.device}")

        # GPU memory optimization
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def train_epoch(self):
        """Train for one epoch with mixed precision support"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc='Training')

        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            if self.use_mixed_precision:
                # Mixed precision forward pass
                with autocast():
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)

                # Mixed precision backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping with scaler
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Update weights with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular precision training
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            current_acc = 100.0 * correct / total
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self):
        """Validate for one epoch with mixed precision support"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='Validation')

            for data, targets in progress_bar:
                data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                if self.use_mixed_precision:
                    # Mixed precision validation
                    with autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                else:
                    # Regular precision validation
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)

                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Store predictions for analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                # Update progress bar
                current_acc = 100.0 * correct / total
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        epoch_loss = total_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc, all_predictions, all_targets

    def train(self, num_epochs):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}")
        print("="*60)

        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.PATIENCE,
            min_delta=0.001,
            restore_best_weights=True
        )

        best_val_acc = 0.0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)

            # Training phase
            train_loss, train_acc = self.train_epoch()

            # Validation phase
            val_loss, val_acc, val_predictions, val_targets = self.validate_epoch()

            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Print epoch results
            epoch_time = time.time() - epoch_start_time
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"Time: {epoch_time:.2f}s")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f"New best validation accuracy: {best_val_acc:.2f}%")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss)

            # Unfreeze backbone for fine-tuning
            if (epoch + 1) == config.UNFREEZE_EPOCH and hasattr(self.model, 'unfreeze_backbone'):
                self.model.unfreeze_backbone()
                # Reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
                print("Backbone unfrozen for fine-tuning!")

            # Early stopping check
            if early_stopping(val_loss, self.model):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")

        # Save final model
        self.save_model('final_model.pth')

        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(config.MODEL_SAVE_PATH, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': {
                'num_classes': config.NUM_CLASSES,
                'img_size': config.IMG_SIZE,
                'model_type': 'resnet'
            }
        }, filepath)
        print(f"Model saved to {filepath}")

    def save_checkpoint(self, epoch, val_loss):
        """Save training checkpoint"""
        filepath = os.path.join(config.CHECKPOINT_PATH, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(config.MODEL_SAVE_PATH, 'training_history.png'), dpi=300)
        plt.show()

def create_optimizer(model, optimizer_type='adam'):
    """Create optimizer"""
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer

def create_scheduler(optimizer, scheduler_type='reduce_on_plateau'):
    """Create learning rate scheduler"""
    if scheduler_type.lower() == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=True
        )
    elif scheduler_type.lower() == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        scheduler = None

    return scheduler

def main():
    """Main training function"""
    print("Initializing training...")

    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders()

    # Compute class weights for handling imbalance
    print("Computing class weights...")
    class_weights = compute_class_weights(train_loader)

    # Create model
    print("Creating model...")
    model = create_model(
        model_type='resnet',
        pretrained=True,
        freeze_backbone=config.FREEZE_PRETRAINED_LAYERS
    )

    # Create loss function
    print("Creating loss function...")
    criterion = get_loss_function('cross_entropy', class_weights)

    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(model, 'adamw')

    # Create scheduler
    print("Creating scheduler...")
    scheduler = create_scheduler(optimizer, 'reduce_on_plateau')

    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler)

    # Start training
    print("Starting training...")
    train_losses, val_losses, train_accuracies, val_accuracies = trainer.train(config.EPOCHS)

    # Plot training history
    print("Plotting training history...")
    trainer.plot_training_history()

    print("Training completed successfully!")

if __name__ == "__main__":
    main()