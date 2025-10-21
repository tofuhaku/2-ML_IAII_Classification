"""
Main execution script for The Simpsons Character Recognition
Complete CNN-based image classifier for 50 characters using PyTorch

Usage:
    python main.py --mode train    # Train the model
    python main.py --mode test     # Run inference and generate CSV
    python main.py --mode both     # Train and test
    python main.py --mode predict --image path/to/image.jpg  # Predict single image
"""

import os
import argparse
import sys
import torch
import config
from train import main as train_main
from inference import main as inference_main, SimpsonsInference
from data_preprocess import analyze_dataset, create_data_loaders

def print_header():
    """Print application header"""
    print("="*70)
    print("    The Simpsons Character Recognition System")
    print("    50-Class CNN Classifier using PyTorch")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print("="*70)

def setup_directories():
    """Create necessary directories"""
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)
    print(f"Created directories: {config.MODEL_SAVE_PATH}, {config.CHECKPOINT_PATH}")

def check_data_availability():
    """Check if training and test data are available"""
    train_available = os.path.exists(config.TRAIN_DATA_PATH)
    test_available = os.path.exists(config.TEST_DATA_PATH)

    print(f"Training data available: {train_available} ({config.TRAIN_DATA_PATH})")
    print(f"Test data available: {test_available} ({config.TEST_DATA_PATH})")

    if not train_available and not test_available:
        print("ERROR: No data found! Please check the dataset paths in config.py")
        sys.exit(1)

    return train_available, test_available

def analyze_data():
    """Analyze the dataset"""
    print("\nAnalyzing dataset...")
    try:
        train_loader, val_loader = create_data_loaders()
        analyze_dataset(train_loader)
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

def train_model():
    """Train the model"""
    print("\nStarting training...")
    try:
        train_main()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)

def test_model():
    """Run inference and generate CSV"""
    print("\nStarting inference...")
    try:
        inference_main()
        print("Inference completed successfully!")
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)

def predict_single_image(image_path):
    """Predict a single image"""
    print(f"\nPredicting single image: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return

    try:
        # Load the best model
        from inference import load_best_model
        model_path = load_best_model()

        # Create inference object
        inference = SimpsonsInference(model_path)

        # Predict
        inference.predict_single_file(image_path, show_top_k=5)

    except Exception as e:
        print(f"Error during prediction: {e}")

def print_system_info():
    """Print system information"""
    print("\nSystem Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")

        # Show current GPU usage
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(current_device) / 1e9
            cached = torch.cuda.memory_reserved(current_device) / 1e9
            print(f"Current GPU: {current_device}")
            print(f"Memory allocated: {allocated:.2f} GB")
            print(f"Memory cached: {cached:.2f} GB")
    else:
        print("No CUDA devices found. Running on CPU.")
        print("To use GPU, ensure:")
        print("  1. NVIDIA GPU is installed")
        print("  2. CUDA drivers are installed")
        print("  3. PyTorch with CUDA support is installed")

def print_character_list():
    """Print all 50 characters"""
    print("\nSupported Characters (50 total):")
    print("-" * 50)
    for i, char in enumerate(config.CHARACTERS, 1):
        print(f"{i:2d}. {char}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='The Simpsons Character Recognition')
    parser.add_argument('--mode', choices=['train', 'test', 'both', 'predict', 'analyze', 'info'],
                       default='both', help='Mode to run')
    parser.add_argument('--image', type=str, help='Path to image for prediction (use with --mode predict)')

    args = parser.parse_args()

    # Print header
    print_header()

    # Setup directories
    setup_directories()

    # Check data availability
    train_available, test_available = check_data_availability()

    # Execute based on mode
    if args.mode == 'info':
        print_system_info()
        print_character_list()

    elif args.mode == 'analyze':
        if train_available:
            analyze_data()
        else:
            print("Training data not available for analysis!")

    elif args.mode == 'train':
        if train_available:
            train_model()
        else:
            print("Training data not available!")

    elif args.mode == 'test':
        if test_available:
            test_model()
        else:
            print("Test data not available!")

    elif args.mode == 'both':
        if train_available:
            train_model()
        else:
            print("Skipping training - no training data available")

        if test_available:
            test_model()
        else:
            print("Skipping testing - no test data available")

    elif args.mode == 'predict':
        if args.image:
            predict_single_image(args.image)
        else:
            print("Please provide --image path for prediction mode")

    print("\nProgram completed!")

if __name__ == "__main__":
    main()