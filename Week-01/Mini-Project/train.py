"""Train the digit-recognizer CNN on MNIST.

Run:  python train.py
Creates: digit_cnn.pth (best weights), history.json, images/training_curves.png
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files, no pop-up window
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import DigitCNN
from preprocess import MEAN, STD

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
IMAGES_DIR = BASE / "images"
WEIGHTS = BASE / "digit_cnn.pth"
HISTORY = BASE / "history.json"

EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42


def get_loaders():
    """54,000 training / 6,000 validation images (the 10,000 test images are kept for later)."""
    # Training images get small random rotations, shifts and zooms (augmentation)
    # so the model copes with messy hand-drawn digits.
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])
    plain_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,)),
    ])

    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_tf)
    val_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=plain_tf)

    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(train_full), generator=gen).tolist()
    train_ds = Subset(train_full, perm[:54000])
    val_ds = Subset(val_full, perm[54000:])  # same images, but without augmentation

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise just evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)                      # forward pass
            loss = loss_fn(out, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()                  # backpropagation
                optimizer.step()                 # update weights
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(dim=1) == yb).sum().item()
            count += xb.size(0)

    return total_loss / count, correct / count


def plot_curves(history, path):
    """Save loss and accuracy curves (train vs validation) as one image."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    IMAGES_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading MNIST (the first run downloads it, about 11 MB)...")
    train_loader, val_loader = get_loaders()

    model = DigitCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, loss_fn, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        saved = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), WEIGHTS)
            saved = "  <- best so far, saved"

        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | "
              f"{time.time() - start:.0f}s{saved}")

    HISTORY.write_text(json.dumps(history, indent=2))
    plot_curves(history, IMAGES_DIR / "training_curves.png")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Weights saved to:  {WEIGHTS.name}")
    print("Curves saved to:   images/training_curves.png")
    print("Next step: python evaluate.py")


if __name__ == "__main__":
    main()
