#-------------------------------------------------------------------------
# FILENAME: evaluate_self_training.py
# SPECIFICATION: Evaluates the self-trained ResNet-18 model on the held-out
# test set using the shared evaluate() utility from eval_script.py, so its
# numbers are directly comparable to the other three conditions in the
# report (KNN, Decision Tree, ResNet-18 supervised baseline).
# FOR: CS 4210 - Course Project
# Sources:
# 1. https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# 2. https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
# 3. https://docs.pytorch.org/tutorials/beginner/basics/intro.html
#-----------------------------------------------------------*/

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

from config import DATA_DIR, RANDOM_SEED
from utils import preprocess_image, normalize_image
from eval_script import evaluate

# ---- Reproducibility ----
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load and preprocess test images ----
    df_test = pd.read_csv('processed_data/test.csv')

    X_test = []
    y_true = []

    for _, row in df_test.iterrows():
        image_path = os.path.join(DATA_DIR, row['image_id'] + '.jpg')
        img = preprocess_image(image_path)
        img = normalize_image(img)
        img = img.transpose((2, 0, 1))                                  # HWC -> CHW for PyTorch
        X_test.append(torch.tensor(img, dtype=torch.float32))           # Source: [3]
        y_true.append(row['label'])

    y_true_numpy = np.array(y_true)
    X_tensor = torch.stack(X_test)
    Y_tensor = torch.tensor(y_true, dtype=torch.long)

    test_dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ---- Build ResNet-18 with the 7-class head and load the checkpoint ----
    model = models.resnet18(weights=None)                               # Source: [2]
    number_of_features = model.fc.in_features
    model.fc = nn.Linear(number_of_features, 7)

    # Load the self-training checkpoint produced by train_self_training.py (Source: [1])
    model.load_state_dict(torch.load('resnet18_self_training.pth', map_location=device))
    model = model.to(device)
    model.eval()                                                        # Source: [1]

    # Softmax to convert logits into class probabilities for ROC-AUC (Source: [1])
    softmax_layer = nn.Softmax(dim=1)

    all_probs = []
    all_preds = []

    with torch.no_grad():                                               # Source: [1]
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = softmax_layer(logits)
            preds = probs.argmax(1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    # Shared evaluation utility writes the figure and prints the metrics.
    evaluate("ResNet-18 Self-Training", y_true_numpy, y_pred, y_prob)


if __name__ == "__main__":
    main()