#-------------------------------------------------------------------------
# AUTHOR: Andrew Mazmanian
# FILENAME: evaluate_supervised_baseline.py
# SPECIFICATION: Evaluates the trained ResNet-18 model on the test dataset.
# Sources:
# 1. https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# 2. https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
# 3. https://docs.pytorch.org/tutorials/beginner/basics/intro.html
#-----------------------------------------------------------*/

import os
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
from config import DATA_DIR, RANDOM_SEED
from utils import preprocess_image, normalize_image
from eval_script import evaluate
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    
def main():
    df_test = pd.read_csv('processed_data/test.csv')
    
    X_test = []
    y_true = []

    # Load and preprocess test images
    for index, row in df_test.iterrows():
        image_path = os.path.join(DATA_DIR, row['image_id'] + '.jpg')
        img = preprocess_image(image_path)
        img = normalize_image(img)
        img = img.transpose((2, 0, 1)) # Format for PyTorch
        
        # Convert to Tensor (Source: [3])
        X_test.append(torch.tensor(img, dtype=torch.float32)) 
        y_true.append(row['label'])

    y_true_numpy = np.array(y_true) # Saved for later
    X_tensor = torch.stack(X_test)
    Y_tensor = torch.tensor(y_true, dtype=torch.long)
    
    test_dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Source: [2]
    model = models.resnet18(weights=None) 
    number_of_features = model.fc.in_features
    model.fc = nn.Linear(number_of_features, 7)
    
    # Loading the saved state dictionary (Source: [1])
    model.load_state_dict(torch.load('resnet18_supervised_baseline.pth'))
    
    # Put model in evaluation mode to disable dropout/batchnorm updates (Source: [1])
    model.eval()
    
    # We need Softmax to scale raw logits into percentages for the ROC-AUC curves
    # (Source: [1])
    softmax_layer = nn.Softmax(dim=1)
    
    all_probs = []
    all_preds = []
        
    # Disable gradient tracking to save memory during testing (Source: [1])
    with torch.no_grad():
        for images, labels in test_loader:            
            # Get raw numbers from the model (Source: [1])
            logits = model(images)
            
            # Convert raw numbers to probability percentages (Source: [1])
            probs = softmax_layer(logits)
            
            # Get the highest probability class prediction (Source: [1])
            preds = probs.argmax(1)
            
            # Convert PyTorch Tensors back to standard Python/NumPy arrays 
            # so scikit-learn can use them in eval_script.py (Source: [1])
            all_probs.append(probs.numpy())
            all_preds.append(preds.numpy())
            
    # Concatenate all batches into single arrays
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    
    evaluate("ResNet-18 Supervised Baseline", y_true_numpy, y_pred, y_prob)

if __name__ == "__main__":
    main()