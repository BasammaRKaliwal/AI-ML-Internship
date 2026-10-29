"""Pictures and the written evaluation report (NumPy, PIL and matplotlib only)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from paths import IMAGES_DIR


def pretty(name):
    return name.replace("_", " ").capitalize()


def _save(fig, filename):
    IMAGES_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / filename, dpi=150)
    plt.close(fig)


def plot_class_distribution(info):
    classes = info["classes"]
    splits = ["train", "val", "test"]
    colours = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(classes))
    for s in splits:
        counts = np.array([info["splits"][s].get(c, 0) for c in classes])
        ax.bar([pretty(c) for c in classes], counts, bottom=bottom, label=s, color=colours[s])
        bottom += counts
    for i, total in enumerate(bottom):
        ax.text(i, total + 5, int(total), ha="center", fontsize=9)
    ax.set_ylabel("Images")
    ax.set_title("Class distribution and split sizes")
    ax.legend()
    _save(fig, "class_distribution.png")


def plot_curves(history, titles=None):
    """history: {config: {train_loss, val_loss, train_acc, val_acc, ...}} -> loss on top, accuracy below."""
    names = list(history)
    fig, axes = plt.subplots(2, len(names), figsize=(4.5 * len(names), 6.5), squeeze=False)
    for col, name in enumerate(names):
        h = history[name]
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        title = (titles or {}).get(name, name)
        axes[0, col].plot(epochs, h["train_loss"], label="Training")
        axes[0, col].plot(epochs, h["val_loss"], label="Validation")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Loss")
        axes[1, col].plot(epochs, np.array(h["train_acc"]) * 100, label="Training")
        axes[1, col].plot(epochs, np.array(h["val_acc"]) * 100, label="Validation")
        axes[1, col].set_ylabel("Accuracy (%)")
        axes[1, col].set_xlabel("Epoch")
        for row in (0, 1):
            axes[row, col].grid(alpha=0.3)
            axes[row, col].legend()
    _save(fig, "training_curves.png")


def plot_confusion(cm, classes, accuracy):
    names = [pretty(c) for c in classes]
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, fmt in ((axes[0], cm, "Counts", "{:d}"), (axes[1], normalised * 100, "Row percentages", "{:.0f}")):
        ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix: {title} (test accuracy {accuracy * 100:.1f}%)", fontsize=10)
        for i in range(len(names)):
            for j in range(len(names)):
                value = data[i, j]
                ax.text(j, i, fmt.format(int(value) if title == "Counts" else value), ha="center", va="center",
                        color="white" if normalised[i, j] > 0.5 else "black", fontsize=9)
    _save(fig, "confusion_matrix.png")


def plot_failures(items, classes, filename="failure_examples.png", ncols=4):
    """items: list of (image_path, true_idx, predicted_idx, score)."""
    if not items:
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (path, true_i, pred_i, score) in zip(axes, items):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(f"True: {pretty(classes[true_i])}\nPredicted: {pretty(classes[pred_i])} ({score * 100:.0f}%)",
                     fontsize=9, color="red")
    fig.suptitle("Most confident mistakes on the test set")
    _save(fig, filename)


def plot_confidence(confidence, correct):
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=bool)
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([confidence[correct], confidence[~correct]], bins=bins, stacked=True,
            label=["Correct predictions", "Wrong predictions"], color=["tab:green", "tab:red"])
    ax.set_xlabel("Model score for its chosen category")
    ax.set_ylabel("Test images")
    ax.set_title("A high score does not guarantee a correct answer")
    ax.legend()
    _save(fig, "confidence_histogram.png")


# ---------- the written report ----------

def comparison_markdown(rows, final_key):
    lines = ["| Configuration | Input size | Augmentation | Epochs | Parameters | Training time | Best val. accuracy | Test accuracy |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = " (final model)" if r["key"] == final_key else ""
        lines.append(f"| {r['title']}{mark} | {r['size']} px | {'Yes' if r['augment'] else 'No'} | {r['epochs']} | "
                     f"{r['params']:,} | {r['seconds']:.0f} s | {r['val_acc'] * 100:.1f}% | {r['test_acc'] * 100:.1f}% |")
    return "\n".join(lines) + "\n"


def suggest_improvements(ctx):
    """Data-driven suggestions the student can reword."""
    classes, recall = ctx["classes"], ctx["recall"]
    order = list(np.argsort(recall))
    weak = [pretty(classes[i]) for i in order[:2]]
    tips = []
    if ctx["confusions"]:
        count, i, j = ctx["confusions"][0]
        tips.append(f"**Fix the biggest confusion first.** {pretty(classes[i])} was predicted as {pretty(classes[j])} "
                    f"{count} times. Collect more and more varied photos of both and check that the labels are right.")
    tips.append(f"**Add data for the weakest categories** ({', '.join(weak)}). They have the lowest recall.")
    if ctx["gap"] is not None and ctx["gap"] > 0.05:
        tips.append(f"**Reduce overfitting.** The final model is {ctx['gap'] * 100:.1f} points better on training than "
                    "validation data. Stronger augmentation, more dropout or more data would help.")
    tips += [
        "**Use photos from real situations.** Most public waste datasets show one object on a plain background. "
        "Add pictures with cluttered backgrounds, bad lighting and partly hidden objects.",
        "**Add an 'unknown / not waste' option.** A score threshold already marks unsure predictions, but a model "
        "trained with extra 'other' images would be more honest about things it has never seen.",
        "**Fine-tune more layers or use a larger input size** if the computer allows it.",
        "**Calibrate the scores** (for example with temperature scaling) so that a score of 80% really means "
        "about 80% of such predictions are correct.",
    ]
    return tips


def build_evaluation_report(ctx):
    info, classes = ctx["info"], ctx["classes"]
    n_test = ctx["stats"]["n_test"]
    lines = ["# Evaluation Report: Waste Image Classification", ""]

    # 1 dataset
    lines += ["## 1. Dataset", "",
              f"- **Source:** {info.get('dataset_name', 'not given')}"
              + (f" ({info['dataset_url']})" if info.get("dataset_url") else ""),
              f"- **Categories used ({len(classes)}):** " + ", ".join(pretty(c) for c in classes),
              f"- **Images found:** {sum(info['found'].values())} in the selected categories",
              f"- **Exact duplicate files removed:** {info['exact_duplicates_removed']}",
              f"- **Near-duplicate groups:** {info['near_duplicate_groups']} groups containing "
              f"{info['images_in_near_dup_groups']} images. Each group was kept together in a single split.",
              f"- **Leakage check after splitting** (identical or near-identical images in different splits): "
              f"{info['leakage_check']['exact']} identical, {info['leakage_check']['near']} near-identical",
              "", "| Category | Train | Validation | Test | Total |", "|---|---|---|---|---|"]
    totals = {"train": 0, "val": 0, "test": 0}
    for c in classes:
        row = [info["splits"][s].get(c, 0) for s in ("train", "val", "test")]
        for s, v in zip(("train", "val", "test"), row):
            totals[s] += v
        lines.append(f"| {pretty(c)} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
    lines += [f"| **Total** | **{totals['train']}** | **{totals['val']}** | **{totals['test']}** | **{sum(totals.values())}** |",
              "", "![Class distribution](images/class_distribution.png)", ""]

    # 2 configurations
    lines += ["## 2. Model configurations", "", comparison_markdown(ctx["rows"], ctx["final_key"])]
    for r in ctx["rows"]:
        lines.append(f"- **{r['title']}:** {r['description']}")
    lines += ["", "The best model was chosen by **validation** accuracy. The test set was used only once for the final numbers.", ""]

    # 3 graphs
    lines += ["## 3. Training and validation graphs", "", "![Training curves](images/training_curves.png)", "",
              "When the training line keeps improving but the validation line stops, the model is memorising the "
              "training images (overfitting).", ""]

    # 4 test results
    final = next(r for r in ctx["rows"] if r["key"] == ctx["final_key"])
    lines += ["## 4. Final test results", "",
              f"**Final model: {final['title']}. Test accuracy {final['test_acc'] * 100:.1f}%** "
              f"({ctx['stats']['n_wrong']} wrong out of {n_test} test images).", "",
              "| Category | Precision | Recall | F1 | Test images |", "|---|---|---|---|---|"]
    for i, c in enumerate(classes):
        lines.append(f"| {pretty(c)} | {ctx['precision'][i] * 100:.0f}% | {ctx['recall'][i] * 100:.0f}% | "
                     f"{ctx['f1'][i] * 100:.0f}% | {int(ctx['support'][i])} |")
    lines += ["", "Precision: of the images predicted as this category, how many were right. "
              "Recall: of the real images of this category, how many were found.", "",
              "![Confusion matrix](images/confusion_matrix.png)", ""]
    if ctx["confusions"]:
        lines += ["Most frequent mistakes:", "", "| True category | Predicted as | Times |", "|---|---|---|"]
        for count, i, j in ctx["confusions"]:
            lines.append(f"| {pretty(classes[i])} | {pretty(classes[j])} | {count} |")
        lines.append("")

    # 5 failures
    lines += ["## 5. Failure examples and error analysis", "", "![Failure examples](images/failure_examples.png)", ""]
    st = ctx["stats"]
    if st["n_wrong"]:
        lines += [f"- Average score on correct predictions: {st['avg_conf_correct'] * 100:.0f}%. "
                  f"On wrong predictions: {st['avg_conf_wrong'] * 100:.0f}%.",
                  f"- **{st['wrong_high_conf']} of the {st['n_wrong']} wrong predictions had a score of 80% or more.** "
                  "The model can be confidently wrong.", ""]
    lines += ["| Score range | Test images | Accuracy in that range |", "|---|---|---|"]
    for low, high, n, acc in ctx["conf_rows"]:
        lines.append(f"| {low * 100:.0f}% to {high * 100:.0f}% | {n} | {'-' if acc is None else f'{acc * 100:.0f}%'} |")
    lines += ["", "![Score histogram](images/confidence_histogram.png)", "",
              "### Suggested improvements (reword in your own words)", ""]
    lines += [f"{k}. {tip}" for k, tip in enumerate(suggest_improvements(ctx), 1)]

    lines += ["", "## Limitations", "",
              "- **A model score is not a guarantee.** It only says how strongly the model prefers one category "
              "over the others. It is not the chance of being right, and wrong answers can have high scores.",
              f"- The model knows only {len(classes)} categories and will always pick one of them, even for a banana peel or a battery.",
              "- It was trained on a small public dataset with mostly plain backgrounds, so real photos taken in "
              "other lighting, angles or with several objects may be classified worse.",
              "- Photos are resized to a square, which squeezes objects with a different shape.",
              "- Whether something can be recycled depends on local rules and on how clean it is. "
              "This tool only looks at a picture and must not be used as the only basis for sorting decisions.", ""]
    return "\n".join(lines)
