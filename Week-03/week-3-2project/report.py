"""Loss curve and the samples file (NumPy and matplotlib only)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import HISTORY_FILE, IMAGES_DIR, SAMPLES_FILE
from utils import repeat_rate


def plot_history(history):
    IMAGES_DIR.mkdir(exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Training loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss (lower is better)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, np.exp(history["val_loss"]), marker="o", color="tab:green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation perplexity")
    axes[1].set_title("Perplexity: how many characters the model is 'choosing between'")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "loss_curve.png", dpi=150)
    plt.close(fig)


def write_samples(samples, seed, history, config):
    """samples: list of (temperature, text)."""
    best = int(np.argmin(history["val_loss"]))
    lines = ["# Generated samples", "",
             f"- Seed text: `{seed.strip() or '(none)'}`",
             f"- Best validation loss: {history['val_loss'][best]:.3f} (epoch {best + 1}), "
             f"perplexity {np.exp(history['val_loss'][best]):.2f}",
             f"- Model: {config['num_layers']} LSTM layers x {config['hidden_size']} units, "
             f"{config['vocab_size']} characters in the vocabulary", ""]
    for temperature, text in samples:
        lines += [f"## Temperature {temperature}", "",
                  f"Repeated 4-letter chunks: {repeat_rate(text) * 100:.0f}%", "", "```", text.rstrip(), "```", ""]
    SAMPLES_FILE.write_text("\n".join(lines), encoding="utf-8")


def save_history(history):
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
