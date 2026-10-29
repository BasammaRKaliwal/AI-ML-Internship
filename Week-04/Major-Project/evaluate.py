"""Measure every trained model on the test images (which were never used for training or choosing),
then write the confusion matrix, error analysis and the full evaluation report.

Run:   python evaluate.py

Creates: evaluation_report.md, results.json, comparison_table.md,
         images/confusion_matrix.png, images/failure_examples.png, images/confidence_histogram.png
"""
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from common import CONFIGS, eval_transform, get_device, load_checkpoint
from paths import (COMPARISON_FILE, DATA_DIR, FINAL_MODEL, HISTORY_FILE, MODELS_DIR, REPORT_FILE, RESULTS_FILE,
                   SPLIT_INFO)
from report import (build_evaluation_report, comparison_markdown, plot_confidence, plot_confusion, plot_curves,
                    plot_failures)
from utils import confidence_table, confusion_matrix, per_class_report, top_confusions


@torch.no_grad()
def predict_all(model, size, device):
    dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform(size))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        probs.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels), [p for p, _ in dataset.samples], dataset.classes


def main():
    if not FINAL_MODEL.exists() or not HISTORY_FILE.exists():
        sys.exit("No trained models found. Run train.py first.")
    info = json.loads(SPLIT_INFO.read_text(encoding="utf-8"))
    history = json.loads(HISTORY_FILE.read_text())
    classes = info["classes"]
    device = get_device()
    final_key = load_checkpoint(FINAL_MODEL, device)[1]["key"]

    rows, final = [], None
    for key, cfg in CONFIGS.items():
        path = MODELS_DIR / f"{key}.pt"
        if key not in history or not path.exists():
            continue
        model, ckpt = load_checkpoint(path, device)
        probs, y_true, paths, test_classes = predict_all(model, ckpt["size"], device)
        if test_classes != classes:
            sys.exit("The classes in data/test do not match split_info.json. Run prepare_data.py again.")
        y_pred = probs.argmax(1)
        accuracy = float((y_pred == y_true).mean())
        h = history[key]
        rows.append({"key": key, "title": cfg["title"], "description": cfg["description"], "size": ckpt["size"],
                     "augment": cfg["augment"], "epochs": h["epochs"], "params": h["params"], "seconds": h["seconds"],
                     "val_acc": h["best_val_acc"], "test_acc": accuracy, "best_epoch": h["best_epoch"]})
        print(f"{cfg['title']:26s} test accuracy {accuracy * 100:5.1f}%")
        if key == final_key:
            final = (probs, y_true, paths, h)

    probs, y_true, paths, h = final
    y_pred = probs.argmax(1)
    confidence = probs.max(1)
    correct = y_pred == y_true
    cm = confusion_matrix(y_true, y_pred, len(classes))
    precision, recall, f1, support = per_class_report(cm)
    confusions = top_confusions(cm, 5)
    wrong = np.where(~correct)[0]
    stats = {"n_test": int(len(y_true)), "n_wrong": int(len(wrong)),
             "avg_conf_correct": float(confidence[correct].mean()) if correct.any() else 0.0,
             "avg_conf_wrong": float(confidence[wrong].mean()) if len(wrong) else 0.0,
             "wrong_high_conf": int((confidence[wrong] >= 0.8).sum())}
    gap = h["train_acc"][h["best_epoch"] - 1] - h["val_acc"][h["best_epoch"] - 1]

    plot_curves({k: history[k] for k in CONFIGS if k in history}, {k: CONFIGS[k]["title"] for k in CONFIGS})
    plot_confusion(cm, classes, float(correct.mean()))
    plot_confidence(confidence, correct)
    worst = wrong[np.argsort(-confidence[wrong])][:12]
    plot_failures([(paths[i], int(y_true[i]), int(y_pred[i]), float(confidence[i])) for i in worst], classes)

    ctx = {"info": info, "classes": classes, "rows": rows, "final_key": final_key, "cm": cm,
           "precision": precision, "recall": recall, "f1": f1, "support": support, "confusions": confusions,
           "conf_rows": confidence_table(confidence, correct), "stats": stats, "gap": gap}
    REPORT_FILE.write_text(build_evaluation_report(ctx), encoding="utf-8")
    COMPARISON_FILE.write_text(comparison_markdown(rows, final_key), encoding="utf-8")
    RESULTS_FILE.write_text(json.dumps({
        "final_model": final_key, "test_accuracy": round(float(correct.mean()), 4), "stats": stats,
        "configurations": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "description"} for r in rows],
        "per_class": {c: {"precision": round(float(precision[i]), 4), "recall": round(float(recall[i]), 4),
                          "f1": round(float(f1[i]), 4), "images": int(support[i])} for i, c in enumerate(classes)},
    }, indent=2), encoding="utf-8")

    print(f"\nFinal model ({final_key}): test accuracy {correct.mean() * 100:.1f}% ({len(wrong)} wrong of {len(y_true)})")
    print("\nPer category:")
    for i, c in enumerate(classes):
        print(f"  {c:10s} precision {precision[i] * 100:5.1f}%  recall {recall[i] * 100:5.1f}%  ({int(support[i])} images)")
    if stats["n_wrong"]:
        print(f"\n{stats['wrong_high_conf']} of {stats['n_wrong']} wrong predictions had a score of 80% or more.")
    print("\nSaved: evaluation_report.md, comparison_table.md, results.json and the pictures in images/")
    print("Next step:  python app.py")


if __name__ == "__main__":
    main()
