"""
train.py - Training script for ResNet-50 on HAM10000.
"""
import torch
import torch.nn as nn
from pathlib import Path

from data import get_dataloaders, set_seed
from models.resnet import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train_model(epochs=15, lr=1e-4):
    set_seed(42)
    print(f"Training on device: {DEVICE}")
    train_loader, val_loader, _ = get_dataloaders()
    model = get_model(num_classes=7, pretrained=True).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Epoch {epoch} | Train Loss: {tr_loss:.4f}")

if __name__ == "__main__":
    train_model()
