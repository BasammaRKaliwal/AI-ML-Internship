"""Plain-Python/NumPy helpers (no PyTorch), so they are easy to read and to test.

Text -> numbers, random training windows, and the sampling rule that
temperature and top-k control.
"""
import numpy as np


def clean_text(text):
    """Make line endings uniform and drop characters that are hard to model."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("﻿", "")
    return text.strip() + "\n"


def build_vocab(text, min_count=1):
    """Sorted list of the characters in `text` (rare ones can be dropped with min_count)."""
    counts = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    return sorted(ch for ch, c in counts.items() if c >= min_count)


def encode(text, vocab):
    """Text -> array of integer ids. Characters not in the vocabulary are skipped."""
    index = {ch: i for i, ch in enumerate(vocab)}
    return np.array([index[ch] for ch in text if ch in index], dtype=np.int64)


def decode(ids, vocab):
    return "".join(vocab[int(i)] for i in ids)


def split_data(ids, val_fraction=0.1):
    """Train on the first part of the text, validate on the last part (never shuffled)."""
    cut = int(len(ids) * (1 - val_fraction))
    return ids[:cut], ids[cut:]


def random_batch(ids, batch_size, seq_len, rng):
    """`batch_size` random windows. y is x shifted one character to the left (next-character targets)."""
    starts = rng.integers(0, len(ids) - seq_len - 1, size=batch_size)
    x = np.stack([ids[s:s + seq_len] for s in starts])
    y = np.stack([ids[s + 1:s + seq_len + 1] for s in starts])
    return x, y


def sequential_batches(ids, batch_size, seq_len, max_batches=None):
    """Non-overlapping windows in order, for a steady validation score."""
    n_windows = (len(ids) - 1) // seq_len
    starts = np.arange(n_windows) * seq_len
    batches = []
    for i in range(0, n_windows, batch_size):
        chunk = starts[i:i + batch_size]
        x = np.stack([ids[s:s + seq_len] for s in chunk])
        y = np.stack([ids[s + 1:s + seq_len + 1] for s in chunk])
        batches.append((x, y))
        if max_batches and len(batches) >= max_batches:
            break
    return batches


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def sample_from_logits(logits, temperature, rng, top_k=0):
    """Pick the next character id.

    temperature < 1  -> sharper, safer, more repetitive
    temperature > 1  -> flatter, more surprising, more mistakes
    top_k > 0        -> only the k most likely characters can be chosen
    """
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 1e-6:                      # temperature 0 means "always the most likely"
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k and 0 < top_k < len(logits):
        cutoff = np.sort(logits)[-top_k]
        logits = np.where(logits < cutoff, -np.inf, logits)
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def keep_known_characters(text, vocab):
    """Remove characters the model has never seen. Returns (kept_text, removed_characters)."""
    known = set(vocab)
    kept = "".join(ch for ch in text if ch in known)
    removed = sorted({ch for ch in text if ch not in known})
    return kept, removed


def repeat_rate(text, n=4):
    """Share of repeated n-grams in a text (a rough 'is it stuck in a loop?' score, 0 to 1)."""
    if len(text) <= n:
        return 0.0
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return 1 - len(set(grams)) / len(grams)
