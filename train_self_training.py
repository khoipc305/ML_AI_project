#-------------------------------------------------------------------------
# FILENAME: train_self_training.py
# SPECIFICATION: Iterative self-training on a ResNet-18 CNN warm-started
# from the supervised baseline checkpoint. Generates pseudo-labels on the
# unlabeled training pool, retains samples whose softmax confidence exceeds
# threshold tau, retrains on labeled + confident pseudo-labels, and decays
# tau by 0.05 across three total iterations (tau = 0.90, 0.85, 0.80) as
# specified in Section IV-C.3 of the report.
# FOR: CS 4210 - Course Project
# Sources:
# 1. https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# 2. https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
# 3. https://docs.pytorch.org/tutorials/beginner/basics/intro.html
# 4. D.-H. Lee, "Pseudo-Label: The Simple and Efficient Semi-Supervised
#    Learning Method for Deep Neural Networks," ICML Workshop, 2013.
#-----------------------------------------------------------*/

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

from utils import preprocess_image, normalize_image
from config import DATA_DIR, RANDOM_SEED

# ---- Seed Everything for Reproducibility ----
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---- Self-Training Hyperparameters (per report Section IV-C.3) ----
NUM_ITERATIONS   = 3        # Total self-training iterations
INITIAL_TAU      = 0.90     # Starting confidence threshold
TAU_DECAY        = 0.05     # Threshold reduction per iteration
EPOCHS_PER_ROUND = 20       # Matches the supervised baseline schedule
BATCH_SIZE       = 32
LEARNING_RATE    = 1e-4
NUM_CLASSES      = 7

# Use GPU if available since self-training is significantly heavier than the
# single-pass supervised baseline (3 rounds x 20 epochs over a growing pool).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_image_tensors(csv_path):
    """Load every image referenced in `csv_path` into a single stacked
    Tensor (N, 3, 224, 224), along with a Tensor of labels and the
    original DataFrame so callers can still inspect image_id / dx."""
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in df.iterrows():
        image_path = os.path.join(DATA_DIR, row['image_id'] + '.jpg')   # Source: [3]
        img = preprocess_image(image_path)
        img = normalize_image(img)
        img = img.transpose((2, 0, 1))                                  # HWC -> CHW
        images.append(torch.tensor(img, dtype=torch.float32))           # Source: [3]
        labels.append(row['label'])

    X = torch.stack(images)                                             # Source: [2]
    y = torch.tensor(labels, dtype=torch.long)
    return X, y, df


def generate_pseudo_labels(model, X_unlabeled, threshold):
    """Run the model in eval mode over the unlabeled pool and return the
    indices that pass the confidence threshold along with their predicted
    labels. Uses softmax over logits (Source: [1])."""
    model.eval()
    softmax_layer = nn.Softmax(dim=1)

    # Stream through the unlabeled pool in batches to keep memory bounded
    # even though all tensors are already in RAM.
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_unlabeled),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_max_probs = []
    all_preds = []

    with torch.no_grad():                                               # Source: [1]
        for (images,) in loader:
            images = images.to(device)
            logits = model(images)                                      # Source: [1]
            probs = softmax_layer(logits)                               # Source: [1]
            max_probs, preds = probs.max(dim=1)                         # Source: [1]
            all_max_probs.append(max_probs.cpu())
            all_preds.append(preds.cpu())

    max_probs = torch.cat(all_max_probs)
    preds = torch.cat(all_preds)

    # Boolean mask of samples we are confident enough about to keep.
    confident_mask = max_probs >= threshold
    confident_indices = confident_mask.nonzero(as_tuple=True)[0]
    confident_preds = preds[confident_indices]

    return confident_indices, confident_preds, max_probs


def train_one_round(model, X_combined, y_combined, augmentation, criterion, optimizer):
    """Train the model for EPOCHS_PER_ROUND epochs on the combined
    (labeled + confident pseudo-labeled) dataset. Mirrors the loop in
    train_supervised_baseline.py so the two scripts stay comparable."""

    dataset = torch.utils.data.TensorDataset(X_combined, y_combined)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Cosine annealing matched to the round length (Source: [1])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PER_ROUND)

    for epoch in range(EPOCHS_PER_ROUND):
        model.train()                                                   # Source: [1]
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Augmentation pipeline matches the supervised baseline so the
            # only deliberate change between conditions is the addition of
            # pseudo-labeled data.
            images = augmentation(images)

            optimizer.zero_grad()                                       # Source: [1]
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)
        print(f"  Epoch {epoch+1:2d}/{EPOCHS_PER_ROUND}  Loss: {average_loss:.4f}")
        scheduler.step()


def main():
    # ---- 1. Load labeled and unlabeled splits ----
    print("Loading labeled training set...")
    X_labeled, y_labeled, _ = load_image_tensors('processed_data/train_labeled.csv')
    print(f"  Labeled samples: {len(X_labeled)}")

    print("Loading unlabeled training set...")
    X_unlabeled, _, df_unlabeled = load_image_tensors('processed_data/train_unlabeled.csv')
    print(f"  Unlabeled samples: {len(X_unlabeled)}")

    # ---- 2. Build the model and warm-start from the supervised baseline ----
    # ResNet-18 with 7-class head (Source: [2])
    model = models.resnet18(weights=None)
    number_of_features = model.fc.in_features
    model.fc = nn.Linear(number_of_features, NUM_CLASSES)

    # Load the checkpoint produced by train_supervised_baseline.py (Source: [1])
    print("Warm-starting from resnet18_supervised_baseline.pth...")
    state_dict = torch.load('resnet18_supervised_baseline.pth', map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # Same class weights as the supervised baseline so the only change between
    # conditions is the addition of pseudo-labeled data.
    class_weights = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Augmentation pipeline matches the supervised baseline (Source: [3])
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

    # ---- 3. Iterative self-training loop ----
    tau = INITIAL_TAU

    for iteration in range(1, NUM_ITERATIONS + 1):
        print("\n" + "=" * 60)
        print(f"Self-Training Iteration {iteration}/{NUM_ITERATIONS}  (tau = {tau:.2f})")
        print("=" * 60)

        # 3a. Generate pseudo-labels for every unlabeled image and keep only
        # the ones whose top softmax probability >= tau.
        print("Generating pseudo-labels on unlabeled pool...")
        confident_indices, confident_preds, max_probs = generate_pseudo_labels(
            model, X_unlabeled, tau
        )
        n_kept = len(confident_indices)
        n_total = len(X_unlabeled)
        keep_pct = 100.0 * n_kept / n_total if n_total else 0.0
        print(f"  Retained {n_kept}/{n_total} unlabeled samples ({keep_pct:.1f}%)")

        if n_kept > 0:
            # Per-class breakdown helps us see whether self-training is just
            # reinforcing the majority class (a common confirmation-bias
            # failure mode discussed in the report).
            unique, counts = torch.unique(confident_preds, return_counts=True)
            print("  Pseudo-label class distribution:")
            for cls, cnt in zip(unique.tolist(), counts.tolist()):
                print(f"    class {cls}: {cnt}")

        # 3b. Build the combined training set (labeled + confident pseudo-labeled).
        if n_kept > 0:
            X_pseudo = X_unlabeled[confident_indices]
            y_pseudo = confident_preds
            X_combined = torch.cat([X_labeled, X_pseudo], dim=0)
            y_combined = torch.cat([y_labeled, y_pseudo], dim=0)
        else:
            # Fall back to just the labeled set if no sample passed tau.
            X_combined = X_labeled
            y_combined = y_labeled

        print(f"  Combined training set size: {len(X_combined)}")

        # 3c. Retrain on the combined set.
        # Re-initialize the optimizer each iteration so the cosine schedule
        # restarts cleanly for the new (larger) training pool.
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        print("Retraining on combined set...")
        train_one_round(model, X_combined, y_combined, augmentation, criterion, optimizer)

        # 3d. Decay the threshold for the next iteration.
        tau = max(0.0, tau - TAU_DECAY)

    # ---- 4. Save the final self-training checkpoint ----
    output_path = 'resnet18_self_training.pth'
    torch.save(model.state_dict(), output_path)
    print(f"\nSaved self-training model to {output_path}")


if __name__ == "__main__":
    main()