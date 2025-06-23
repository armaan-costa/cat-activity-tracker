import os
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim

# ==== CONFIG ============
DATA_DIR = "data"
BATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_EPOCHS = 15
LR = 0.001
MODEL_PATH = "models/cat_activity_model.pth"
# ========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == 'cuda':
    print(f"CUDA GPU found and being used: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA GPU not found... \nCPU is being used.")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # Using means and standard deviations recommended for ResNet18 Image classification
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Loads dataset
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Train
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=True)

# Prints all labels
print(f"Classes: {full_dataset.classes}")
print(f"Training size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

# Taking ResNet18 and fine-tuning it for the data collected.
num_classes = len(full_dataset.classes)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

if __name__ == "__main__":
    
    best_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels, = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * (correct / total)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} || Train Loss: {train_loss:.3f} || Train Accuracy: {train_acc:.2f}%")

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # Choose best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print("The new best model best model has been saved.")
