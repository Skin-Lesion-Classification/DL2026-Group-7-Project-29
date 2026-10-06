"""
train.py - Training script skeleton for HAM10000.
"""
import torch
import torch.nn as nn
from pathlib import Path

from data import get_dataloaders
from models.resnet import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(epochs=15, lr=1e-4):
    print(f"Training on device: {DEVICE}")
    train_loader, val_loader, _ = get_dataloaders()
    model = get_model(num_classes=7, pretrained=True).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    train_model()
