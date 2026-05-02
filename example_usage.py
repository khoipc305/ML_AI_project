"""Minimal example showing how to use the prepared dataset.

Run this AFTER `python data_preparation.py` so the CSV files exist:

    python example_usage.py

The model training team can copy this pattern to build their data loader.
"""

import pandas as pd

from config import TEST_CSV, TRAIN_CSV, TRAIN_LABELED_CSV, TRAIN_UNLABELED_CSV, VAL_CSV
from utils import normalize_image, preprocess_image


def main():
    # Load every prepared CSV split into a pandas DataFrame.
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    train_labeled_df = pd.read_csv(TRAIN_LABELED_CSV)
    train_unlabeled_df = pd.read_csv(TRAIN_UNLABELED_CSV)

    # Quick sanity check on how many samples ended up in each split.
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    print(f"Labeled train: {len(train_labeled_df)}")
    print(f"Unlabeled train: {len(train_unlabeled_df)}")

    # Load + normalize the first training image as a demo.
    sample = train_df.iloc[0]
    image = preprocess_image(sample["image_path"])  # resized, scaled to [0, 1]
    image = normalize_image(image)                  # ImageNet mean/std normalization

    print("\nExample image loaded successfully")
    print(f"Image ID: {sample['image_id']}")
    print(f"Class: {sample['dx']} / label {sample['label']}")
    print(f"Image shape: {image.shape}")


if __name__ == "__main__":
    main()
