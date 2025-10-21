"""
Inference script for The Simpsons Character Recognition
Handles model inference and CSV output generation
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from torchvision import transforms
import config
from model import create_model
from data_preprocess import create_test_loader, get_transforms

class SimpsonsInference:
    """Inference class for The Simpsons character classification"""

    def __init__(self, model_path):
        """
        Initialize inference with trained model

        Args:
            model_path (str): Path to the trained model
        """
        self.device = config.DEVICE
        self.model_path = model_path
        self.model = None
        self.transform = get_transforms(is_training=False)

        self._load_model()

        print(f"Inference initialized on device: {self.device}")

    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}")

        # Create model architecture
        self.model = create_model(
            model_type='resnet',
            pretrained=False,  # We'll load our trained weights
            freeze_backbone=False
        )

        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print("Model loaded successfully!")

    def predict_single_image(self, image_path):
        """
        Predict a single image

        Args:
            image_path (str): Path to the image

        Returns:
            tuple: (predicted_class_name, confidence_score, all_probabilities)
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_class_idx = predicted.item()
            confidence_score = confidence.item()

            predicted_class_name = config.IDX_TO_CHAR[predicted_class_idx]
            all_probs = probabilities.cpu().numpy()[0]

        return predicted_class_name, confidence_score, all_probs

    def predict_batch(self, test_loader):
        """
        Predict a batch of images using data loader

        Args:
            test_loader: PyTorch DataLoader for test data

        Returns:
            tuple: (predictions, confidences, filenames)
        """
        self.model.eval()
        predictions = []
        confidences = []
        filenames = []

        with torch.no_grad():
            for images, batch_filenames in tqdm(test_loader, desc="Predicting"):
                images = images.to(self.device)

                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                batch_confidences, batch_predictions = torch.max(probabilities, 1)

                # Convert to character names
                for i in range(len(batch_predictions)):
                    pred_idx = batch_predictions[i].item()
                    pred_char = config.IDX_TO_CHAR[pred_idx]
                    conf = batch_confidences[i].item()

                    predictions.append(pred_char)
                    confidences.append(conf)
                    filenames.append(batch_filenames[i])

        return predictions, confidences, filenames

    def predict_test_set(self):
        """
        Predict entire test set and return results

        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        print("Creating test data loader...")
        test_loader = create_test_loader()

        print("Running inference on test set...")
        predictions, confidences, filenames = self.predict_batch(test_loader)

        # Create DataFrame
        results_df = pd.DataFrame({
            'filename': filenames,
            'character': predictions,
            'confidence': confidences
        })

        # Extract image ID from filename (remove .jpg extension)
        results_df['id'] = results_df['filename'].str.replace('.jpg', '')

        # Convert id to integer for proper sorting
        results_df['id'] = results_df['id'].astype(int)

        # Sort by id
        results_df = results_df.sort_values('id').reset_index(drop=True)

        print(f"Predictions completed for {len(results_df)} images")

        return results_df

    def generate_submission_csv(self, output_path=None):
        """
        Generate submission CSV file

        Args:
            output_path (str): Path for output CSV file

        Returns:
            str: Path to the generated CSV file
        """
        if output_path is None:
            output_path = config.OUTPUT_CSV

        # Get predictions
        results_df = self.predict_test_set()

        # Create submission format (id, character)
        submission_df = results_df[['id', 'character']].copy()

        # Save to CSV
        submission_df.to_csv(output_path, index=False)

        print(f"Submission CSV saved to: {output_path}")
        print(f"Submission shape: {submission_df.shape}")

        # Show sample predictions
        print("\nSample predictions:")
        print(submission_df.head(10))

        # Show prediction distribution
        print("\nPrediction distribution:")
        char_counts = submission_df['character'].value_counts()
        print(char_counts.head(10))

        return output_path

    def analyze_predictions(self, results_df):
        """
        Analyze prediction results

        Args:
            results_df (pd.DataFrame): DataFrame with predictions
        """
        print("Prediction Analysis:")
        print("="*50)

        # Basic statistics
        print(f"Total predictions: {len(results_df)}")
        print(f"Unique characters predicted: {results_df['character'].nunique()}")
        print(f"Average confidence: {results_df['confidence'].mean():.3f}")
        print(f"Min confidence: {results_df['confidence'].min():.3f}")
        print(f"Max confidence: {results_df['confidence'].max():.3f}")

        # Confidence distribution
        print("\nConfidence distribution:")
        print(f"  High confidence (>0.9): {(results_df['confidence'] > 0.9).sum()}")
        print(f"  Medium confidence (0.5-0.9): {((results_df['confidence'] > 0.5) & (results_df['confidence'] <= 0.9)).sum()}")
        print(f"  Low confidence (<0.5): {(results_df['confidence'] <= 0.5).sum()}")

        # Character prediction distribution
        print("\nCharacter prediction distribution:")
        char_counts = results_df['character'].value_counts()
        print("Top 10 predicted characters:")
        for char, count in char_counts.head(10).items():
            percentage = (count / len(results_df)) * 100
            print(f"  {char}: {count} ({percentage:.1f}%)")

        # Low confidence predictions
        low_conf = results_df[results_df['confidence'] < 0.5]
        if len(low_conf) > 0:
            print(f"\nLow confidence predictions ({len(low_conf)} images):")
            print(low_conf[['filename', 'character', 'confidence']].head())

    def predict_single_file(self, image_path, show_top_k=5):
        """
        Predict a single image file and show detailed results

        Args:
            image_path (str): Path to the image file
            show_top_k (int): Number of top predictions to show
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found!")
            return

        print(f"Predicting image: {image_path}")

        predicted_class, confidence, all_probs = self.predict_single_image(image_path)

        print(f"\nTop prediction: {predicted_class} (confidence: {confidence:.3f})")

        # Show top-k predictions
        top_k_indices = np.argsort(all_probs)[::-1][:show_top_k]

        print(f"\nTop {show_top_k} predictions:")
        for i, idx in enumerate(top_k_indices):
            char_name = config.IDX_TO_CHAR[idx]
            prob = all_probs[idx]
            print(f"  {i+1}. {char_name}: {prob:.3f}")

        return predicted_class, confidence, all_probs

def load_best_model():
    """Load the best trained model"""
    model_path = os.path.join(config.MODEL_SAVE_PATH, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"Best model not found at {model_path}")
        print("Looking for alternative models...")

        # Try final model
        final_model_path = os.path.join(config.MODEL_SAVE_PATH, 'final_model.pth')
        if os.path.exists(final_model_path):
            model_path = final_model_path
            print(f"Using final model: {model_path}")
        else:
            raise FileNotFoundError("No trained model found! Please train the model first.")

    return model_path

def main():
    """Main inference function"""
    print("Starting inference...")

    # Load the best model
    model_path = load_best_model()

    # Create inference object
    inference = SimpsonsInference(model_path)

    # Generate submission CSV
    print("Generating submission CSV...")
    csv_path = inference.generate_submission_csv()

    # Analyze results
    print("Analyzing predictions...")
    results_df = inference.predict_test_set()
    inference.analyze_predictions(results_df)

    print(f"\nInference completed! Submission saved to: {csv_path}")

if __name__ == "__main__":
    main()