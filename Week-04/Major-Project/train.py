"""Train the baseline, the improved CNN and a pretrained MobileNetV2, then keep the best as final_model.pt.

Run:   python train.py                       (all three)
       python train.py --configs baseline improved
       python train.py --quick               (2 epochs, a few batches: only checks that everything works)

Creates: models/<name>.pt, final_model.pt, history.json, images/training_curves.png
"""
import argparse
import copy
import json
import shutil
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from common import CONFIGS, build_model, count_parameters, eval_transform, get_device, train_transform
from paths import DATA_DIR, FINAL_MODEL, HISTORY_FILE, MODELS_DIR, SPLIT_INFO
from report import plot_curves


def run_epoch(model, loader, loss_fn, device, optimizer=None, max_batches=0):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader):
            if max_batches and step >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
    return total_loss / count, correct / count


def make_optimizer(model, lr, weight_decay):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_config(key, cfg, classes, device, batch, quick):
    size = cfg["size"]
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform(size, cfg["augment"]))
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform(size))
    if train_ds.classes != classes or val_ds.classes != classes:
        sys.exit("The class folders in data/ do not match split_info.json. Run prepare_data.py again.")
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    torch.manual_seed(42)
    model = build_model(cfg["arch"], len(classes), pretrained=True).to(device)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 2 if quick else cfg["epochs"]
    max_batches = 8 if quick else 0

    freeze_until = epochs // 2 if cfg["arch"] == "transfer" else 0   # phase 1: only the new classifier
    if freeze_until:
        for p in model.features.parameters():
            p.requires_grad = False
    optimizer = make_optimizer(model, cfg["lr"], cfg["weight_decay"])

    print(f"\n=== {cfg['title']} | {count_parameters(model):,} parameters | {size}x{size} px | "
          f"augmentation: {'yes' if cfg['augment'] else 'no'} | {epochs} epochs ===")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_epoch, best_state = -1.0, 0, None
    started = time.time()
    for epoch in range(1, epochs + 1):
        if freeze_until and epoch == freeze_until + 1:               # phase 2: fine-tune the last blocks
            for block in model.features[-4:]:
                for p in block.parameters():
                    p.requires_grad = True
            optimizer = make_optimizer(model, cfg["lr"] / 10, cfg["weight_decay"])
            print("   (fine-tuning the last 4 blocks from now on)")
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer, max_batches)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device, None, max_batches)
        for name, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[name].append(round(value, 4))
        marker = ""
        if val_acc > best_acc:
            best_acc, best_epoch, best_state = val_acc, epoch, copy.deepcopy(model.state_dict())
            marker = "  <- best so far"
        print(f"Epoch {epoch:2d}/{epochs} | train loss {train_loss:.3f} acc {train_acc * 100:5.1f}% | "
              f"val loss {val_loss:.3f} acc {val_acc * 100:5.1f}%{marker}")

    seconds = time.time() - started
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "arch": cfg["arch"], "size": size, "key": key,
                "title": cfg["title"], "val_acc": best_acc, "epoch": best_epoch}, MODELS_DIR / f"{key}.pt")
    history.update(params=count_parameters(model), seconds=round(seconds, 1), best_val_acc=round(best_acc, 4),
                   best_epoch=best_epoch, epochs=epochs)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if not SPLIT_INFO.exists() or not (DATA_DIR / "train").is_dir():
        sys.exit("No prepared data found. Run prepare_data.py first.")
    classes = json.loads(SPLIT_INFO.read_text(encoding="utf-8"))["classes"]
    device = get_device()
    print(f"Device: {device} | categories: {', '.join(classes)}")

    # Training all three starts fresh; training only some keeps the results of the others
    keep_old = HISTORY_FILE.exists() and not args.quick and set(args.configs) != set(CONFIGS)
    history = json.loads(HISTORY_FILE.read_text()) if keep_old else {}
    for key in args.configs:
        history[key] = train_config(key, CONFIGS[key], classes, device, args.batch, args.quick)
        HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")

    plot_curves({k: v for k, v in history.items() if k in CONFIGS}, {k: CONFIGS[k]["title"] for k in CONFIGS})
    best = max((k for k in history if k in CONFIGS), key=lambda k: history[k]["best_val_acc"])
    shutil.copy(MODELS_DIR / f"{best}.pt", FINAL_MODEL)

    print("\nSummary (validation accuracy):")
    for k in CONFIGS:
        if k in history:
            h = history[k]
            print(f"  {CONFIGS[k]['title']:26s} {h['best_val_acc'] * 100:5.1f}%  ({h['seconds']:.0f} s, {h['params']:,} parameters)")
    print(f"\nFinal model: {CONFIGS[best]['title']} -> final_model.pt")
    print("Next step:  python evaluate.py")


if __name__ == "__main__":
    main()
