#-------------------------------------------------------------------------
# FILENAME: train_decision_tree.py
# SPECIFICATION: Trains and tunes a Decision Tree baseline using PCA-reduced images.
#-------------------------------------------------------------------------

import os
import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from config import DATA_DIR, RANDOM_SEED
from utils import preprocess_image, normalize_image
from eval_script import evaluate
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_and_flatten_data(csv_file):
    """Loads images from a CSV, preprocesses, normalizes, and flattens them to 1D."""
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    X = []
    y = []

    for index, row in df.iterrows():
        image_path = os.path.join(DATA_DIR, row['image_id'] + '.jpg')
        
        # Load and preprocess
        img = preprocess_image(image_path)
        img = normalize_image(img)
        
        # Flatten the 3D image (224x224x3) into a 1D array for scikit-learn
        flattened_img = img.flatten()
        
        X.append(flattened_img)
        y.append(row['label'])

    return np.array(X), np.array(y)

def main():
    # 1. Load the labeled training, validation, and test sets
    X_train, y_train = load_and_flatten_data('processed_data/train_labeled.csv')
    X_val, y_val = load_and_flatten_data('processed_data/val.csv')
    X_test, y_test = load_and_flatten_data('processed_data/test.csv')

    # 2. Apply Principal Component Analysis (PCA)
    print("Applying PCA to reduce features to 100 components...")
    pca = PCA(n_components=100, random_state=RANDOM_SEED)
    
    # Fit PCA ONLY on the training data to prevent data leakage, then transform all sets
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    # 3. Hyperparameter Tuning on Validation Set
    print("Tuning Decision Tree maximum depth...")
    depths = [5, 10, 20, None]
    best_depth = None
    best_val_f1 = -1
    best_model = None

    for depth in depths:
        # Initialize and train the tree
        clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=RANDOM_SEED)
        clf.fit(X_train_pca, y_train)
        
        # Predict on validation set to find the best depth
        val_preds = clf.predict(X_val_pca)
        val_f1 = f1_score(y_val, val_preds, average='macro')
        
        print(f" - Max Depth: {depth} | Validation Macro-F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_depth = depth
            best_model = clf

    print(f"\nSelected Best Max Depth: {best_depth}")

    # 4. Final Evaluation on Test Set
    print("Evaluating best model on Test Set...")
    y_pred = best_model.predict(X_test_pca)
    y_prob = best_model.predict_proba(X_test_pca) # Required for ROC-AUC curves

    # 5. Generate metrics and save plots using the shared script
    evaluate("Decision_Tree_Baseline", y_test, y_pred, y_prob)

if __name__ == "__main__":
    main()