"""
train.py - Training Pipeline for ResNet-50 on HAM10000.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score

from data import get_dataloaders, set_seed
from models.resnet import get_model
from losses import get_loss_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_grad_scaler():
    """Support backward compatibility for GradScaler across PyTorch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    return torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for images, targets in loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    return total_loss / len(loader), acc, macro_f1

def train_model(
    epochs=15,
    lr=1e-4,
    loss_type="ce",
    use_alpha=False,
    use_weighted_sampler=False,
    use_aug=False,
    num_workers=2,
    save_dir="results/exp"
):
    if use_weighted_sampler and loss_type == "weighted_ce":
        raise ValueError("Cannot combine 'use_weighted_sampler=True' with 'loss_type=weighted_ce'.")
    if use_weighted_sampler and loss_type == "focal" and use_alpha:
        raise ValueError("Cannot combine 'use_weighted_sampler=True' with 'use_alpha=True'.")

    set_seed(42)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        use_weighted_sampler=use_weighted_sampler,
        use_aug=use_aug,
        num_workers=num_workers
    )
    model = get_model(num_classes=7, pretrained=True).to(DEVICE)

    counts = [train_loader.dataset.labels.count(i) for i in range(7)]
    criterion = get_loss_fn(loss_type, class_counts=counts, use_alpha=use_alpha, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = get_grad_scaler()

    best_macro_f1 = 0.0
    history = []
    print(f"[*] Starting training on device: {DEVICE}")

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | Val Macro F1: {val_f1:6.4f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save(
                {'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_macro_f1': val_f1},
                save_path / "best_model.pth"
            )

    pd.DataFrame(history).to_csv(save_path / "training_history.csv", index=False)
