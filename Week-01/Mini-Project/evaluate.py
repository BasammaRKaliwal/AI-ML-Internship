"""Evaluate the trained model on the 10,000 MNIST test images.

Run:  python evaluate.py
Creates: images/confusion_matrix.png, images/misclassified.png, results.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import DigitCNN
from preprocess import MEAN, STD

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
IMAGES_DIR = BASE / "images"
WEIGHTS = BASE / "digit_cnn.pth"
RESULTS = BASE / "results.json"


def predict_all(model, loader, device):
    """Return (true labels, predicted labels) for every image in the loader."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            y_pred.append(out.argmax(dim=1).cpu().numpy())
            y_true.append(yb.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def confusion_matrix(y_true, y_pred, n_classes=10):
    """Rows = true digit, columns = predicted digit."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_report(cm):
    """Precision and recall for every digit."""
    diag = np.diag(cm).astype(float)
    col_sums = cm.sum(axis=0).astype(float)   # how often each digit was predicted
    row_sums = cm.sum(axis=1).astype(float)   # how often each digit really appears
    precision = np.divide(diag, col_sums, out=np.zeros_like(diag), where=col_sums > 0)
    recall = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums > 0)
    return precision, recall


def plot_confusion(cm, accuracy, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix (Test accuracy {accuracy * 100:.2f}%)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_misclassified(images, true, pred, path, max_items=12):
    """Show up to 12 test images the model got wrong."""
    rows, cols = 2, 6
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.4))
    fig.suptitle("Misclassified test images")
    for k, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if k < min(max_items, len(images)):
            ax.imshow(images[k], cmap="gray")
            ax.set_title(f"True {true[k]} / Pred {pred[k]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    if not WEIGHTS.exists():
        raise SystemExit("digit_cnn.pth not found. Run  python train.py  first.")

    IMAGES_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))

    y_true, y_pred = predict_all(model, loader, device)
    accuracy = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)
    precision, recall = per_class_report(cm)

    print(f"Test accuracy: {accuracy * 100:.2f}%  ({int((y_true == y_pred).sum())} of {len(y_true)} correct)\n")
    print("Digit | Precision | Recall")
    print("------+-----------+-------")
    for d in range(10):
        print(f"  {d}   |   {precision[d]:.3f}   | {recall[d]:.3f}")

    plot_confusion(cm, accuracy, IMAGES_DIR / "confusion_matrix.png")

    wrong = np.where(y_true != y_pred)[0]
    raw_images = test_ds.data[wrong].numpy()   # original 28x28 pixels, not normalised
    plot_misclassified(raw_images, y_true[wrong], y_pred[wrong], IMAGES_DIR / "misclassified.png")

    RESULTS.write_text(json.dumps({
        "test_accuracy": accuracy,
        "wrong_predictions": int(len(wrong)),
        "precision": [float(p) for p in precision],
        "recall": [float(r) for r in recall],
    }, indent=2))

    print("\nSaved: images/confusion_matrix.png, images/misclassified.png, results.json")
    print("Next step: python app.py")


if __name__ == "__main__":
    main()
