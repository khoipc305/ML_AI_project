#HAM10000 dataset preparation pipeline.

# Run this file to produce the CSV splits used by the model training team:
# python data_preparation.py

# Outputs are written to `processed_data/` (see paths in config.py):
#   - train.csv / val.csv / test.csv           -> stratified split
#   - train_labeled.csv / train_unlabeled.csv  -> semi-supervised split
#   - dataset_statistics.txt                   -> summary counts


import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CLASS_NAMES,
    LABELED_DATA_PERCENTAGE,
    METADATA_FILE,
    MIN_SAMPLES_PER_CLASS,
    RANDOM_SEED,
    REMOVE_DUPLICATES,
    STATS_FILE,
    STRATIFY_SPLIT,
    TEST_CSV,
    TEST_RATIO,
    TRAIN_CSV,
    TRAIN_LABELED_CSV,
    TRAIN_RATIO,
    TRAIN_UNLABELED_CSV,
    VAL_CSV,
    VAL_RATIO,
)
from utils import encode_labels, load_metadata, remove_duplicate_lesions, verify_dataset


def prepare_dataset():
    # Run the full dataset preparation pipeline end-to-end.
    # 1. Load metadata and keep only rows whose image file exists.
    df = load_metadata(METADATA_FILE)
    df = verify_dataset(df)

    # 2. Optionally drop duplicate images of the same lesion.
    if REMOVE_DUPLICATES:
        df = remove_duplicate_lesions(df)

    # 3. Add a numeric `label` column based on `dx`.
    df = encode_labels(df)
    print_class_counts("Full dataset", df)

    # 4. Split into train/val/test, then split train into labeled/unlabeled.
    train_df, val_df, test_df = split_train_val_test(df)
    train_labeled_df, train_unlabeled_df = split_labeled_unlabeled(train_df)

    # 5. Save CSV outputs and summary statistics.
    save_outputs(train_df, val_df, test_df, train_labeled_df, train_unlabeled_df)
    save_statistics(df, train_df, val_df, test_df, train_labeled_df, train_unlabeled_df)

    print("\nDataset preparation complete.")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Labeled train: {len(train_labeled_df)} | Unlabeled train: {len(train_unlabeled_df)}")

    return df, train_df, val_df, test_df, train_labeled_df, train_unlabeled_df


def split_train_val_test(df):
    # Return (train_df, val_df, test_df) using the ratios from config.
    # First split: hold out the test set from everything else.
    stratify = df["dx"] if STRATIFY_SPLIT else None
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=stratify,
        random_state=RANDOM_SEED,
    )

    # Second split: separate validation from training within the remainder.
    # VAL_RATIO is a fraction of the full dataset, so we rescale it to the
    # train+val subset to keep the final ratios correct.
    val_ratio_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    stratify = train_val_df["dx"] if STRATIFY_SPLIT else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=stratify,
        random_state=RANDOM_SEED,
    )

    return train_df.copy(), val_df.copy(), test_df.copy()


def split_labeled_unlabeled(train_df):
    # Split `train_df` into a small labeled set and a larger unlabeled set.
    # Sampling is done per class so every class has at least
    # MIN_SAMPLES_PER_CLASS labeled examples (useful for semi-supervised
    # learning where most training data is treated as unlabeled).
    
    labeled_parts = []

    for class_name in CLASS_NAMES:
        class_df = train_df[train_df["dx"] == class_name]
        # Pick enough labeled samples, but never fewer than the minimum.
        n_labeled = max(
            MIN_SAMPLES_PER_CLASS,
            int(len(class_df) * LABELED_DATA_PERCENTAGE),
        )
        # Never request more samples than the class actually has.
        n_labeled = min(n_labeled, len(class_df))
        labeled_parts.append(class_df.sample(n=n_labeled, random_state=RANDOM_SEED))

    # Combine labeled rows from every class; unlabeled = the remaining train rows.
    train_labeled_df = pd.concat(labeled_parts).copy()
    train_unlabeled_df = train_df.drop(index=train_labeled_df.index).copy()

    return train_labeled_df, train_unlabeled_df


def save_outputs(train_df, val_df, test_df, train_labeled_df, train_unlabeled_df):
    # Write every split as a CSV file to the paths defined in config.py.
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    train_labeled_df.to_csv(TRAIN_LABELED_CSV, index=False)
    train_unlabeled_df.to_csv(TRAIN_UNLABELED_CSV, index=False)


def save_statistics(df, train_df, val_df, test_df, train_labeled_df, train_unlabeled_df):
    # Write a human-readable summary of split sizes and class counts.
    with open(STATS_FILE, "w") as file:
        file.write("HAM10000 Dataset Summary\n")
        file.write("========================\n\n")
        file.write(f"Full dataset: {len(df)}\n")
        file.write(f"Train: {len(train_df)}\n")
        file.write(f"Validation: {len(val_df)}\n")
        file.write(f"Test: {len(test_df)}\n")
        file.write(f"Labeled train: {len(train_labeled_df)}\n")
        file.write(f"Unlabeled train: {len(train_unlabeled_df)}\n\n")

        for name, split_df in {
            "Full": df,
            "Train": train_df,
            "Validation": val_df,
            "Test": test_df,
            "Labeled train": train_labeled_df,
        }.items():
            file.write(f"{name} class distribution:\n")
            file.write(split_df["dx"].value_counts().to_string())
            file.write("\n\n")


def print_class_counts(title, df):
    # Print how many samples fall into each diagnosis class.
    print(f"\n{title} class distribution:")
    for class_name, count in df["dx"].value_counts().items():
        print(f"{class_name}: {count}")


if __name__ == "__main__":
    # Entry point used when running `python data_preparation.py`.
    prepare_dataset()
