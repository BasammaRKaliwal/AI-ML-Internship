"""Web app: draw a digit in the browser and the model predicts it live.

Run:  python app.py      then open  http://127.0.0.1:5000
"""
import base64
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_file
from PIL import Image

from model import DigitCNN
from preprocess import preprocess_canvas

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "digit_cnn.pth"

if not WEIGHTS.exists():
    raise SystemExit("digit_cnn.pth not found. Run  python train.py  first.")

app = Flask(__name__)

device = torch.device("cpu")
model = DigitCNN().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()


@app.route("/")
def index():
    return send_file(BASE / "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")
    if "," not in image_data:
        return jsonify(error="No image received"), 400

    try:
        raw = base64.b64decode(image_data.split(",", 1)[1])
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return jsonify(error="Could not read the image"), 400

    pixels = preprocess_canvas(img)
    if pixels is None:
        return jsonify(error="Nothing drawn yet")

    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 28, 28)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]

    digit = int(probs.argmax())
    return jsonify(
        digit=digit,
        confidence=float(probs[digit]),
        probabilities=[float(p) for p in probs],
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
