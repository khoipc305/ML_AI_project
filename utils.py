# Helper functions for HAM10000 dataset preparation and image loading.
# These functions are used by `data_preparation.py` and can also be imported
# directly by the model training code (e.g. to load and normalize images).


import os

import cv2
import numpy as np
import pandas as pd

from config import BASE_DIR, CLASS_MAPPING, DATA_DIR, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD


def load_metadata(metadata_path):
    # Read the HAM10000 metadata CSV into a DataFrame.
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} metadata rows")
    return df


def resolve_image_path(image_path):
    # Turn a stored image_path into an absolute path on this machine.
    # Stored paths are relative (e.g. "HAM10000/ISIC_0024306.jpg") so the
    # project works for anyone who clones the repo, regardless of where
    # they put it.
    if os.path.isabs(image_path):
        return image_path
    return os.path.join(BASE_DIR, image_path)


def verify_dataset(df, data_dir=DATA_DIR):
    # Attach each row's image path and drop rows whose image file is missing.
    df = df.copy()
    # Absolute path used only to check that the file exists on disk.
    abs_paths = df["image_id"].apply(lambda image_id: os.path.join(data_dir, f"{image_id}.jpg"))
    df = df[abs_paths.apply(os.path.exists)].copy()
    # Store a path relative to BASE_DIR so the CSV is portable across machines.
    # Forward slashes work on Windows, macOS, and Linux.
    df["image_path"] = df["image_id"].apply(lambda image_id: f"HAM10000/{image_id}.jpg")
    print(f"Verified {len(df)} images")
    return df


def remove_duplicate_lesions(df):
    # Keep only one image per lesion_id to avoid train/test leakage.
    before = len(df)
    df = df.drop_duplicates(subset="lesion_id", keep="first").copy()
    print(f"Removed {before - len(df)} duplicate lesion images")
    return df


def preprocess_image(image_path, target_size=IMAGE_SIZE):
    # Load an image, convert to RGB, resize, and scale pixels to [0, 1].
    # Accepts both absolute paths and paths stored in our CSV files
    # (which are relative to the project folder).
    image_path = resolve_image_path(image_path)
    image = cv2.imread(image_path)                    # BGR uint8
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    # Convert to RGB
    image = cv2.resize(image, target_size)            # Resize to IMAGE_SIZE
    return image.astype(np.float32) / 255.0           # Scale to [0, 1] floats


def normalize_image(image, mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    # Apply per-channel mean/std normalization (ImageNet stats by default).
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return (image - mean) / std


def encode_labels(df):
    # Add an integer `label` column based on the text `dx` column.
    df = df.copy()
    df["label"] = df["dx"].map(CLASS_MAPPING)
    # Drop any row whose diagnosis is not in CLASS_MAPPING.
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df
