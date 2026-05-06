#-------------------------------------------------------------------------
# AUTHOR: Andrew Mazmanian
# FILENAME: train_supervised_baseline.py
# SPECIFICATION: Training a ResNet-18 CNN on 10% labeled data using preprocessed CSVs.# FOR: CS 4210- Course Project
# Sources:
# 1. https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# 2. https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
# 3. https://docs.pytorch.org/tutorials/beginner/basics/intro.html
#-----------------------------------------------------------*/

import random
import os
# NEED TO INSTALL BY RUNNING: pip install torch torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

# ----Loading Real Data from CSV Files----
import pandas as pd
from utils import preprocess_image, normalize_image
from config import DATA_DIR, RANDOM_SEED

import numpy as np

# ---- Seed Everything for Reproducibility ----
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 1. Reads CSV file
df = pd.read_csv('processed_data/train_labeled.csv')

X = [] # Holds images
Y = [] # Holds labels

# 2. Loops through every row in CSV file
for index, row in df.iterrows():
    
    image_path = os.path.join(DATA_DIR, row['image_id'] + '.jpg') # Constructs path to image file (Source: [3])
    # Load's picture from folder
    img = preprocess_image(image_path)
    img = normalize_image(img)
    img = img.transpose((2, 0, 1)) # Format for PyTorch
    image_tensor = torch.tensor(img, dtype=torch.float32) # Converts to Tensor for PyTorch (Source: [3])
    
    X.append(image_tensor)
    Y.append(row['label'])

# PyTorch requires stacking lists into single large Tensor
X_tensor = torch.stack(X) #Implementation Source: [2]
Y_tensor = torch.tensor(Y, dtype=torch.long)

# 4. To loop through X & Y (Source: [3])
real_dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
train_loader = torch.utils.data.DataLoader(real_dataset, batch_size=32, shuffle=True)

print(f"Successfully loaded {len(df)} training images!") #Makes sure we loaded the data correctly

# ----Data Augmentation Pipeline----
# Random flips, Rotations, and Color jitter as specified in our proposal. Implementation Source: [3]
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

# ----Setting up ResNet-18 Model----
# 1. Loading pre-trained model. Source: [2]
model = models.resnet18(weights='DEFAULT')

# 2. Replaces the final layer to output 7 classes as ResNet-18 originally predicts 1000 things...
# ...and we only have 7 skin cancer classes. Source: [3]
number_of_features = model.fc.in_features
model.fc = nn.Linear(number_of_features, 7)

# ----Setting up Training Rules (Source: [1])----
# Dummy weights to handle class imbalance (since Melanocytic nevi is >60% of data)
class_weights = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Sees how wrong the model is
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Define the number of epochs first
num_epochs = 20

# Algorithm updates the model to make it better
optimizer = optim.Adam(model.parameters(), lr=1e-4) #

# Now num_epochs is defined, so this line will work
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ----Manual Training Loop----
num_epochs = 20
for epoch in range(num_epochs):
    
    model.train() # Puts into training mode (Souce: [1])
    total_loss = 0.0

    # Loops through data in batches
    for images, labels in train_loader:
        
        # Applying rotations and flips to the images
        images = train_transforms(images)
        
        # Resets the optimizer, makes a prediction, and calculates the error (loss), ...
        # ... updating model weights based on the error. (Source: [1])
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Printing average error rate for this epoch
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Error Loss: {average_loss:.4f}")
    
    # Step the learning rate down
    scheduler.step()

# ----Saving the Model for Semi-Superivsed Looping (Sofia Truong)----
torch.save(model.state_dict(), 'resnet18_supervised_baseline.pth')