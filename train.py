"""
train.py - Training script skeleton for HAM10000.
"""
import torch
import torch.nn as nn
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print(f"Training on device: {DEVICE}")
    # TODO: Load dataset and model

if __name__ == "__main__":
    train_model()
