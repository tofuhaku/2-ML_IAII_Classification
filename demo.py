"""
Demo script for The Simpsons Character Recognition System
Demonstrates basic usage of the CNN classifier
"""

import os
import sys

def print_demo_header():
    """Print demo header"""
    print("="*70)
    print("    The Simpsons Character Recognition - Demo Script")
    print("="*70)

def demo_system_info():
    """Demo: Show system information"""
    print("\n1. System Information and Supported Characters")
    print("-" * 50)
    os.system("python main.py --mode info")

def demo_data_analysis():
    """Demo: Analyze dataset"""
    print("\n2. Dataset Analysis")
    print("-" * 50)
    os.system("python main.py --mode analyze")

def demo_training():
    """Demo: Train the model"""
    print("\n3. Model Training")
    print("-" * 50)
    print("Starting training process...")
    print("Note: This will take several hours depending on your hardware")

    choice = input("Do you want to proceed with training? (y/n): ")
    if choice.lower() == 'y':
        os.system("python main.py --mode train")
    else:
        print("Skipping training...")

def demo_inference():
    """Demo: Run inference"""
    print("\n4. Model Inference")
    print("-" * 50)

    # Check if model exists
    model_path = "models/best_model.pth"
    if os.path.exists(model_path):
        print("Running inference on test set...")
        os.system("python main.py --mode test")
    else:
        print("No trained model found. Please train the model first.")

def demo_single_prediction():
    """Demo: Predict single image"""
    print("\n5. Single Image Prediction")
    print("-" * 50)

    # Check if model exists
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print("No trained model found. Please train the model first.")
        return

    # Look for sample images in test directory
    test_dir = "dataset/test-renamed_images"
    if os.path.exists(test_dir):
        sample_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        if sample_images:
            sample_image = os.path.join(test_dir, sample_images[0])
            print(f"Predicting sample image: {sample_image}")
            os.system(f"python main.py --mode predict --image {sample_image}")
        else:
            print("No test images found.")
    else:
        print("Test directory not found.")

def main():
    """Main demo function"""
    print_demo_header()

    print("This demo will showcase the capabilities of The Simpsons Character Recognition System.")
    print("Please ensure you have:")
    print("1. Installed all required packages (pip install -r requirements.txt)")
    print("2. Set up the dataset in the correct directory structure")
    print("\nPress Enter to continue...")
    input()

    try:
        # Demo 1: System info
        demo_system_info()

        # Demo 2: Data analysis
        if os.path.exists("dataset/train/train"):
            demo_data_analysis()
        else:
            print("\nSkipping data analysis - training data not found")

        # Demo 3: Training (optional)
        if os.path.exists("dataset/train/train"):
            demo_training()
        else:
            print("\nSkipping training - training data not found")

        # Demo 4: Inference
        if os.path.exists("dataset/test-renamed_images"):
            demo_inference()
        else:
            print("\nSkipping inference - test data not found")

        # Demo 5: Single prediction
        demo_single_prediction()

        print("\n" + "="*70)
        print("Demo completed! Check the generated files:")
        print("- models/best_model.pth (trained model)")
        print("- submission.csv (predictions)")
        print("- models/training_history.png (training plots)")
        print("="*70)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Please check your environment setup.")

if __name__ == "__main__":
    main()