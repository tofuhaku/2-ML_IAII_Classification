import torch
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\train'
TEST_PATH = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\test-renamed_images'
BACKGROUND_DIR = r'D:\Programs\ML\ML_IAII\2-ML_IAII_Classification\dataset\background'
MODEL_WEIGHTS_PATH = 'best_model.pth'
OUTPUT_CSV_PATH = 'submission.csv'
STATS_PATH = 'stats.csv'

BATCH_SIZE = 128
NUM_EPOCHS = 50
NUM_WORKERS = 8
NUM_CLASSES = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
VAL_SPLIT = 0.2

# Define character names
CHARACTER_NAMES = [
    'abraham_grampa_simpson', 'agnes_skinner', 'apu_nahasapeemapetilon', 'barney_gumble',
    'bart_simpson', 'brandine_spuckler', 'carl_carlson', 'charles_montgomery_burns',
    'chief_wiggum', 'cletus_spuckler', 'comic_book_guy', 'disco_stu',
    'dolph_starbeam', 'duff_man', 'edna_krabappel', 'fat_tony',
    'gary_chalmers', 'gil', 'groundskeeper_willie', 'homer_simpson',
    'jimbo_jones', 'kearney_zzyzwicz', 'kent_brockman', 'krusty_the_clown',
    'lenny_leonard', 'lionel_hutz', 'lisa_simpson', 'lunchlady_doris',
    'maggie_simpson', 'marge_simpson', 'martin_prince', 'mayor_quimby',
    'milhouse_van_houten', 'miss_hoover', 'moe_szyslak', 'ned_flanders',
    'nelson_muntz', 'otto_mann', 'patty_bouvier', 'principal_skinner',
    'professor_john_frink', 'rainier_wolfcastle', 'ralph_wiggum', 'selma_bouvier',
    'sideshow_bob', 'sideshow_mel', 'snake_jailbird', 'timothy_lovejoy',
    'troy_mcclure', 'waylon_smithers'
]

# Mapping from index to character name and vice versa
IDX_TO_CLASS = {i: name for i, name in enumerate(CHARACTER_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CHARACTER_NAMES)}
