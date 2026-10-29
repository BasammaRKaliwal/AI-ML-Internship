"""Gradio demo: upload a photo of waste and see the predicted category and the model score.

Run:   python app.py        then open the address it prints (usually http://127.0.0.1:7860)
"""
import sys

import gradio as gr

from paths import FINAL_MODEL
from predictor import UNSURE_BELOW, Predictor

if not FINAL_MODEL.exists():
    sys.exit("No final_model.pt found. Run train.py first (or see README.md for the download instructions).")

predictor = Predictor()

DISCLAIMER = (
    "**A model score is not a guarantee.** It only shows how strongly the model prefers one category over the others. "
    "It is not the probability of being right, and the model can be confidently wrong, especially for objects "
    "or photos unlike its training data. The model only knows these categories: "
    + ", ".join(c.capitalize() for c in predictor.classes)
    + ". It will choose one of them even for something else. Do not rely on it alone to sort waste."
)


def classify(image):
    if image is None:
        return None, "Please upload an image first."
    result = predictor.predict(image)                      # same preprocessing as in training
    label, score = result["label"].capitalize(), result["score"] * 100
    lines = [f"### Prediction: **{label}**", f"Model score: **{score:.1f}%**"]
    if result["unsure"]:
        lines.append(f"\n⚠️ The score is below {UNSURE_BELOW * 100:.0f}%, so the model is **not sure**. "
                     "Treat this answer with extra care.")
    else:
        lines.append("\nA high score still does not guarantee a correct answer.")
    rows = predictor.compare_models(image)
    if rows:
        lines += ["", "**Comparison of the trained models on this image**", "",
                  "| Model | Prediction | Score |", "|---|---|---|"]
        lines += [f"| {t} | {l.capitalize()} | {s * 100:.1f}% |" for t, l, s in rows]
    return {c.capitalize(): s for c, s in result["scores"].items()}, "\n".join(lines)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Upload a waste photo"),
    outputs=[gr.Label(num_top_classes=len(predictor.classes), label="Model scores"), gr.Markdown()],
    title="Waste Image Classification",
    description="Upload a picture of one waste item. The model predicts its category and shows its score.",
    article=DISCLAIMER,
)

if __name__ == "__main__":
    demo.launch()
