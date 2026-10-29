"""Load the trained models and classify one picture, using exactly the preprocessing used in training."""
import numpy as np
import torch
from PIL import Image

from common import CONFIGS, eval_transform, get_device, load_checkpoint
from paths import FINAL_MODEL, MODELS_DIR

UNSURE_BELOW = 0.60       # scores under this are flagged as "not sure"


class Predictor:
    def __init__(self):
        self.device = get_device()
        self.model, ckpt = load_checkpoint(FINAL_MODEL, self.device)
        self.classes = ckpt["classes"]
        self.final_title = ckpt.get("title", "final model")
        self.transform = eval_transform(ckpt["size"])
        self.others = {}                                           # the other trained models, for comparison
        for key, cfg in CONFIGS.items():
            path = MODELS_DIR / f"{key}.pt"
            if path.exists():
                model, c = load_checkpoint(path, self.device)
                self.others[key] = (model, eval_transform(c["size"]), c.get("title", cfg["title"]))

    @torch.no_grad()
    def _scores(self, model, transform, image):
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    def predict(self, image):
        """Returns a dict: label, score, scores (all categories), unsure (True if the score is low)."""
        probs = self._scores(self.model, self.transform, image)
        best = int(np.argmax(probs))
        return {"label": self.classes[best], "score": float(probs[best]),
                "scores": {c: float(p) for c, p in zip(self.classes, probs)},
                "unsure": bool(probs[best] < UNSURE_BELOW)}

    def compare_models(self, image):
        """Prediction of every trained model: list of (title, label, score)."""
        rows = []
        for key, (model, transform, title) in self.others.items():
            probs = self._scores(model, transform, image)
            best = int(np.argmax(probs))
            rows.append((title, self.classes[best], float(probs[best])))
        return rows
