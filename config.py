"""
Configuration file for The Simpsons Character Recognition
Contains all hyperparameters, character mappings, and system settings
"""

import torch

# Device configuration - Enhanced GPU setup
def setup_device():
    """Setup device with enhanced GPU configuration"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU Available: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Enable GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return device
    else:
        print("CUDA not available, using CPU")
        return torch.device('cpu')

DEVICE = setup_device()

# All 50 characters as specified
CHARACTERS = [
    'abraham_grampa_simpson',
    'agnes_skinner',
    'apu_nahasapeemapetilon',
    'barney_gumble',
    'bart_simpson',
    'brandine_spuckler',
    'carl_carlson',
    'charles_montgomery_burns',
    'chief_wiggum',
    'cletus_spuckler',
    'comic_book_guy',
    'disco_stu',
    'dolph_starbeam',
    'duff_man',
    'edna_krabappel',
    'fat_tony',
    'gary_chalmers',
    'gil',
    'groundskeeper_willie',
    'homer_simpson',
    'jimbo_jones',
    'kearney_zzyzwicz',
    'kent_brockman',
    'krusty_the_clown',
    'lenny_leonard',
    'lionel_hutz',
    'lisa_simpson',
    'lunchlady_doris',
    'maggie_simpson',
    'marge_simpson',
    'martin_prince',
    'mayor_quimby',
    'milhouse_van_houten',
    'miss_hoover',
    'moe_szyslak',
    'ned_flanders',
    'nelson_muntz',
    'otto_mann',
    'patty_bouvier',
    'principal_skinner',
    'professor_john_frink',
    'rainier_wolfcastle',
    'ralph_wiggum',
    'selma_bouvier',
    'sideshow_bob',
    'sideshow_mel',
    'snake_jailbird',
    'timothy_lovejoy',
    'troy_mcclure',
    'waylon_smithers'
]

# Create character to index mapping
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARACTERS)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}
NUM_CLASSES = len(CHARACTERS)

# Data paths
TRAIN_DATA_PATH = 'dataset/train'
VAL_DATA_PATH = 'dataset/val'  # Will create validation split from training data
TEST_DATA_PATH = 'dataset/test-renamed_images'

# Image preprocessing parameters
IMG_SIZE = 224  # ResNet standard input size
BATCH_SIZE = 32 if DEVICE.type == 'cpu' else 256  # Larger batch size for GPU
NUM_WORKERS = 0   # Set to 0 for Windows to avoid multiprocessing issues

# Training parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15  # Early stopping patience

# Data augmentation parameters
ROTATION_DEGREES = 15
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2
COLOR_JITTER_SATURATION = 0.2
COLOR_JITTER_HUE = 0.1
RANDOM_HORIZONTAL_FLIP_PROB = 0.5
NOISE_STD = 0.1

# Model parameters
PRETRAINED_MODEL = 'resnet50'  # Using ResNet50 as base model for pre-training
DROPOUT_RATE = 0.5
FREEZE_PRETRAINED_LAYERS = True  # Whether to freeze pre-trained layers initially

# Training schedule
UNFREEZE_EPOCH = 20  # Epoch to start fine-tuning pre-trained layers
REDUCE_LR_PATIENCE = 8  # Patience for learning rate reduction
REDUCE_LR_FACTOR = 0.5  # Factor to reduce learning rate

# Model saving
MODEL_SAVE_PATH = 'models'
CHECKPOINT_PATH = 'checkpoints'
BEST_MODEL_NAME = 'best_simpsons_classifier.pth'

# Output
OUTPUT_CSV = 'submission.csv'

# Validation split
VALIDATION_SPLIT = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42