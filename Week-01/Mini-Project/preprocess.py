"""Turns a drawing from the web canvas into a 28x28 image the model understands.

MNIST digits are white on black, scaled to fit a 20x20 box, and centred by their
centre of mass inside a 28x28 frame. Doing exactly the same to the drawing is
what makes the live predictions accurate.
"""
import numpy as np
from PIL import Image

# Normalisation values for MNIST (shared by training, evaluation and the app)
MEAN = 0.1307
STD = 0.3081


def _shift(arr, dy, dx):
    """Shift a 28x28 array by (dy, dx) pixels, filling the gap with zeros."""
    out = np.zeros_like(arr)
    out[max(dy, 0):28 + min(dy, 0), max(dx, 0):28 + min(dx, 0)] = \
        arr[max(-dy, 0):28 + min(-dy, 0), max(-dx, 0):28 + min(-dx, 0)]
    return out


def preprocess_canvas(img):
    """Return a normalised float32 array of shape (28, 28), or None if blank.

    `img` is a PIL image with a black background and a white drawing.
    """
    arr = np.array(img.convert("L"), dtype=np.uint8)

    # 1. Crop to the bounding box of the drawing
    ys, xs = np.where(arr > 20)
    if len(ys) == 0:
        return None
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 2. Scale so the longest side is 20 pixels (keep the aspect ratio)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 3. Paste into the middle of a 28x28 black frame
    canvas = np.zeros((28, 28), dtype=np.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = np.array(small, dtype=np.float32)

    # 4. Move the centre of mass to the middle of the frame
    total = canvas.sum()
    if total > 0:
        idx = np.arange(28)
        cy = (canvas.sum(axis=1) * idx).sum() / total
        cx = (canvas.sum(axis=0) * idx).sum() / total
        canvas = _shift(canvas, int(round(13.5 - cy)), int(round(13.5 - cx)))

    # 5. Scale to 0-1 and normalise like the training data
    canvas = canvas / 255.0
    return ((canvas - MEAN) / STD).astype(np.float32)
