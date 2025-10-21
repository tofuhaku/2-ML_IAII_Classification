import torch
import os

# --- Parameter ---
NUM_CLASSES = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
IMAGE_SIZE = 224 # Pre-train model (ResNet) default 224x224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths (please modify according to your actual paths)
TRAIN_DIR = 'D:/Programs/ML/ML_IAII/2-ML_IAII_Classification/dataset/train'
TEST_DIR = 'D:/Programs/ML/ML_IAII/2-ML_IAII_Classification/dataset/test-renamed_images'

# File settings
MODEL_PATH = 'best_simpsons_cnn.pth'
OUTPUT_CSV = 'submission.csv'
VAL_SPLIT = 0.15 # 15% for validation

# Character names list
CHARACTER_NAMES = [
    'abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble', 'bart_simpson',
    'brandine_spuckler', 'carl_carlson', 'charles_montgomery_burns', 'chief_wiggum', 'cletus_spuckler',
    'comic_book_guy', 'disco_stu', 'dolph_starbeam', 'duff_man', 'edna_krabappel', 'fat_tony',
    'gary_chalmers', 'gil', 'groundskeeper_willie', 'homer_simpson', 'jimbo_jones', 'kearney_zzyzwicz',
    'kent_brockman', 'krusty_the_clown', 'lenny_leonard', 'lionel_hutz', 'lisa_simpson', 'lunchlady_doris',
    'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby', 'milhouse_van_houten', 'miss_hoover',
    'moe_szyslak', 'ned_flanders', 'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner',
    'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier', 'sideshow_bob',
    'sideshow_mel', 'snake_jailbird', 'timothy_lovejoy', 'troy_mcclure', 'waylon_smithers'
]

# Mapping from index to character name and vice versa
IDX_TO_CLASS = {i: name for i, name in enumerate(CHARACTER_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CHARACTER_NAMES)}

# Catalogue class count check
if len(IDX_TO_CLASS) != NUM_CLASSES:
    raise ValueError(f"Character ({len(IDX_TO_CLASS)}) and NUM_CLASSES ({NUM_CLASSES}) do not match.")