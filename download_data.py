#!/usr/bin/env python3
"""
The Simpsons Character Recognition - Data Downloader
Download and organize The Simpsons character dataset
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil

def create_directory_structure():
    """Create necessary directory structure"""
    print("📁 Creating directory structure...")

    # Create main data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # Create subdirectories
    (data_dir / 'train').mkdir(exist_ok=True)
    (data_dir / 'val').mkdir(exist_ok=True)
    (data_dir / 'test').mkdir(exist_ok=True)

    print("✅ Directory structure created")
    return data_dir

def download_sample_dataset():
    """Download a sample dataset or create instructions for manual download"""
    print("📥 Setting up The Simpsons dataset...")

    data_dir = Path('data')

    # Create instruction file for manual dataset setup
    instructions = """
    📋 The Simpsons Character Recognition Dataset Setup Instructions
    ================================================================

    Since The Simpsons dataset requires manual download, please follow these steps:

    1. 📥 Download Dataset Options:
       Option A - Kaggle Dataset:
       - Visit: https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset
       - Download: simpsons_dataset.zip

       Option B - Alternative Dataset:
       - Search for "The Simpsons Character Recognition Dataset" on Kaggle
       - Look for datasets with 50+ character classes

    2. 📂 Extract and Organize:
       - Extract the downloaded zip file
       - Organize images in the following structure:

       data/
       ├── train/
       │   ├── character1/
       │   │   ├── image1.jpg
       │   │   ├── image2.jpg
       │   │   └── ...
       │   ├── character2/
       │   └── ...
       ├── val/
       │   ├── character1/
       │   └── ...
       └── test/
           ├── character1/
           └── ...

    3. 🎯 Character Classes (50 main characters):
       - Abraham Simpson, Apu Nahasapeemapetilon, Bart Simpson
       - Chief Wiggum, Comic Book Guy, Edna Krabappel
       - Fat Tony, Gil, Homer Simpson, Krusty the Clown
       - Lisa Simpson, Marge Simpson, Milhouse Van Houten
       - Moe Szyslak, Ned Flanders, Nelson Muntz
       - Principal Skinner, Professor Frink, Ralph Wiggum
       - Selma Bouvier, Sideshow Bob, Snake Jailbird
       - And 27 more characters...

    4. 🔧 Automated Organization (Optional):
       After downloading, run:
       python organize_dataset.py

    5. ✅ Verify Setup:
       Run: python main.py --mode train
    """

    with open('DATA_SETUP_INSTRUCTIONS.txt', 'w', encoding='utf-8') as f:
        f.write(instructions)

    print("📋 Instructions saved to DATA_SETUP_INSTRUCTIONS.txt")

    # Create sample directory structure with dummy folders
    sample_characters = [
        'homer_simpson', 'marge_simpson', 'bart_simpson', 'lisa_simpson',
        'maggie_simpson', 'ned_flanders', 'moe_szyslak', 'apu_nahasapeemapetilon',
        'chief_wiggum', 'milhouse_van_houten', 'nelson_muntz', 'ralph_wiggum',
        'martin_prince', 'krusty_the_clown', 'sideshow_bob', 'comic_book_guy',
        'professor_frink', 'principal_skinner', 'edna_krabappel', 'snake_jailbird'
    ]

    for split in ['train', 'val', 'test']:
        for char in sample_characters[:10]:  # Create 10 sample character folders
            char_dir = data_dir / split / char
            char_dir.mkdir(parents=True, exist_ok=True)

            # Create a placeholder file
            placeholder = char_dir / 'README.txt'
            with open(placeholder, 'w') as f:
                f.write(f"Place {char.replace('_', ' ').title()} images here")

    print("📁 Sample directory structure created with 10 character classes")
    return data_dir

def create_dataset_organizer():
    """Create a script to organize downloaded dataset"""
    organizer_script = '''#!/usr/bin/env python3
"""
Dataset Organizer for The Simpsons Character Recognition
Automatically organize downloaded dataset into train/val/test splits
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def organize_dataset(source_dir, target_dir='data'):
    """
    Organize dataset into train/val/test splits

    Args:
        source_dir: Path to downloaded dataset
        target_dir: Target directory for organized dataset
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directories
    for split in ['train', 'val', 'test']:
        (target_path / split).mkdir(parents=True, exist_ok=True)

    # Find all character directories
    character_dirs = [d for d in source_path.iterdir() if d.is_dir()]

    print(f"Found {len(character_dirs)} character directories")

    for char_dir in character_dirs:
        char_name = char_dir.name.lower().replace(' ', '_')
        print(f"Processing {char_name}...")

        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(char_dir.glob(ext))

        if len(image_files) < 10:
            print(f"  Warning: Only {len(image_files)} images found for {char_name}")
            continue

        # Shuffle images
        random.shuffle(image_files)

        # Split into train/val/test (70/15/15)
        n_total = len(image_files)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)

        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]

        # Create character directories
        for split in ['train', 'val', 'test']:
            (target_path / split / char_name).mkdir(parents=True, exist_ok=True)

        # Copy files
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for i, img_file in enumerate(files):
                target_file = target_path / split / char_name / f"{char_name}_{i:04d}{img_file.suffix}"
                shutil.copy2(img_file, target_file)

        print(f"  ✅ {char_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python organize_dataset.py <source_directory>")
        print("Example: python organize_dataset.py extracted_simpsons_dataset")
        sys.exit(1)

    source_directory = sys.argv[1]
    if not os.path.exists(source_directory):
        print(f"Error: Directory {source_directory} does not exist")
        sys.exit(1)

    organize_dataset(source_directory)
    print("\\n✅ Dataset organization completed!")
    print("Run 'python main.py --mode train' to start training")
'''

    with open('organize_dataset.py', 'w', encoding='utf-8') as f:
        f.write(organizer_script)

    print("🔧 Created organize_dataset.py script")

def main():
    """Main function to set up dataset"""
    print("🎬 The Simpsons Character Recognition - Dataset Setup")
    print("=" * 60)

    # Create directory structure
    data_dir = create_directory_structure()

    # Setup sample dataset structure
    download_sample_dataset()

    # Create dataset organizer
    create_dataset_organizer()

    print("\n" + "=" * 60)
    print("✅ Dataset setup completed!")
    print("\n📋 Next Steps:")
    print("1. Read DATA_SETUP_INSTRUCTIONS.txt for dataset download guide")
    print("2. Download The Simpsons dataset from Kaggle")
    print("3. Run: python organize_dataset.py <downloaded_folder>")
    print("4. Start training: python main.py --mode train")
    print("\n🚀 Your GPU is ready for training!")

if __name__ == "__main__":
    main()