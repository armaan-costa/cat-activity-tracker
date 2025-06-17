import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim

# ==== CONFIG ============
DATA_DIR = "data"
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_EPOCHS = 10
LR = 0.001
MODEL_PATH = "models/cat_activity_model.pth"
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loads dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Train
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# Prints all class labels
print(f"Classes: {full_dataset.classes}")
print(f"Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}")



