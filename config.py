# Shared settings for HAM10000 dataset preparation.

# Edit values here (ratios, paths, class labels) and rerun
# `python data_preparation.py` to regenerate the output CSV files.

import os

# --- Paths ---
# BASE_DIR is the folder that contains this config.py file.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Folder with raw HAM10000 images and metadata file.
DATA_DIR = os.path.join(BASE_DIR, "HAM10000")
# HAM10000 metadata CSV (no .csv extension in the original dataset).
METADATA_FILE = os.path.join(DATA_DIR, "HAM10000_metadata")
# Folder where prepared CSV outputs are written.
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

# Make sure the output folder exists before we write to it.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Train / Validation / Test split ratios (must add up to 1.0) ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Semi-supervised learning setup ---
# Fraction of training data that keeps its labels (rest is "unlabeled").
LABELED_DATA_PERCENTAGE = 0.10
# Minimum labeled samples per class, so rare classes are still represented.
MIN_SAMPLES_PER_CLASS = 5

# --- Image preprocessing settings (used by utils.preprocess_image) ---
IMAGE_SIZE = (224, 224)  # Standard CNN input size (ResNet, VGG, etc.)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet channel means
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet channel standard deviations

# --- Class labels ---
# Text label (from HAM10000 "dx" column) -> integer label used by models.
CLASS_MAPPING = {
    "nv": 0,     # Melanocytic nevi
    "mel": 1,    # Melanoma
    "bkl": 2,    # Benign keratosis-like lesions
    "bcc": 3,    # Basal cell carcinoma
    "akiec": 4,  # Actinic keratoses
    "vasc": 5,   # Vascular lesions
    "df": 6,     # Dermatofibroma
}

CLASS_NAMES = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
NUM_CLASSES = len(CLASS_NAMES)

# --- Reproducibility ---
# Fixed seed so every run produces the same splits.
RANDOM_SEED = 42

# --- Preparation options ---
# If True, keep only one image per lesion_id (avoids data leakage).
REMOVE_DUPLICATES = True
# If True, splits preserve the class distribution of the full dataset.
STRATIFY_SPLIT = True

# --- Output CSV file paths ---
TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VAL_CSV = os.path.join(OUTPUT_DIR, "val.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")
TRAIN_LABELED_CSV = os.path.join(OUTPUT_DIR, "train_labeled.csv")
TRAIN_UNLABELED_CSV = os.path.join(OUTPUT_DIR, "train_unlabeled.csv")
STATS_FILE = os.path.join(OUTPUT_DIR, "dataset_statistics.txt")
