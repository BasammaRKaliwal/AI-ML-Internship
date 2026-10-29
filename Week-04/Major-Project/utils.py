"""NumPy/PIL helpers: duplicate detection, leakage-free splitting, and evaluation numbers.

Nothing here needs PyTorch, so it is easy to read and to test.
"""
import hashlib
import random

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------- duplicate detection ----------

def file_md5(path):
    """Fingerprint of the exact file bytes (identical files give identical values)."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(image, size=8):
    """64-bit 'difference hash': a tiny fingerprint of what the picture looks like.

    Resized or re-saved copies of the same photo get (almost) the same hash.
    """
    small = image.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hash_bytes(hashes):
    """List of 64-bit ints -> array (n, 8) of bytes, so differences can be counted with NumPy."""
    return np.array([list(int(h).to_bytes(8, "big")) for h in hashes], dtype=np.uint8)


def drop_exact_duplicates(md5s):
    """Indices to keep (first copy of every identical file) and how many were dropped."""
    seen, keep = set(), []
    for i, value in enumerate(md5s):
        if value not in seen:
            seen.add(value)
            keep.append(i)
    return keep, len(md5s) - len(keep)


def group_near_duplicates(hashes, max_distance=4):
    """Give every image a group id. Images whose hashes differ in at most `max_distance` of 64 bits
    end up in the same group (directly or through a chain of similar images)."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    codes = hash_bytes(hashes) if n else np.zeros((0, 8), dtype=np.uint8)
    for i in range(n - 1):
        distances = _POPCOUNT[np.bitwise_xor(codes[i], codes[i + 1:])].sum(axis=1)
        for j in np.where(distances <= max_distance)[0]:
            a, b = find(i), find(i + 1 + int(j))
            if a != b:
                parent[b] = a
    roots = [find(i) for i in range(n)]
    relabel = {root: k for k, root in enumerate(dict.fromkeys(roots))}
    return [relabel[r] for r in roots]


# ---------- splitting ----------

def split_by_group(group_ids, labels, fractions=(0.70, 0.15, 0.15), seed=42):
    """Assign every image to split 0 (train), 1 (val) or 2 (test).

    Whole groups of near-identical images always go to the SAME split, so a copy of a training
    image can never show up in the validation or test set. Class proportions are kept roughly equal.
    """
    rng = random.Random(seed)
    n = len(group_ids)
    split_of_group = {}
    result = [-1] * n
    for label in sorted(set(labels)):
        members = {}
        for i in range(n):
            if labels[i] == label:
                members.setdefault(group_ids[i], []).append(i)
        total = sum(len(v) for v in members.values())
        target = [f * total for f in fractions]
        filled = [0, 0, 0]
        groups = list(members)
        rng.shuffle(groups)
        groups.sort(key=lambda g: -len(members[g]))           # place big groups first
        for g in groups:
            if g in split_of_group:                            # group already placed through another class
                s = split_of_group[g]
            else:
                deficits = [(target[k] - filled[k]) / max(target[k], 1e-9) for k in range(3)]
                s = int(np.argmax(deficits))
                split_of_group[g] = s
            filled[s] += len(members[g])
            for i in members[g]:
                result[i] = s
    return result


def count_leakage(md5s, hashes, splits, max_distance=4):
    """How many image pairs in DIFFERENT splits are identical or near-identical (should be 0)."""
    exact, near = 0, 0
    codes = hash_bytes(hashes) if len(hashes) else np.zeros((0, 8), dtype=np.uint8)
    splits = np.asarray(splits)
    for i in range(len(md5s) - 1):
        other = np.arange(i + 1, len(md5s))
        other = other[splits[other] != splits[i]]
        if len(other) == 0:
            continue
        distances = _POPCOUNT[np.bitwise_xor(codes[i], codes[other])].sum(axis=1)
        near += int((distances <= max_distance).sum())
        exact += sum(1 for j in other if md5s[j] == md5s[i])
    return exact, near


# ---------- evaluation numbers ----------

def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_report(cm):
    """precision, recall, f1, support for every class."""
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def top_confusions(cm, k=5):
    """The k most frequent mistakes as (count, true_class, predicted_class)."""
    pairs = [(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    return sorted(pairs, reverse=True)[:k]


def confidence_table(confidence, correct, edges=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)):
    """Accuracy inside each score range: (low, high, number_of_images, accuracy or None)."""
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=bool)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high)
        rows.append((low, min(high, 1.0), int(mask.sum()), float(correct[mask].mean()) if mask.any() else None))
    return rows
