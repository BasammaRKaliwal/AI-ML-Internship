"""Load a trained model and generate new text from it."""
import numpy as np
import torch

from common import CharLSTM, MODEL_FILE, get_device
from utils import keep_known_characters, sample_from_logits


def load_checkpoint(path=MODEL_FILE, device=None):
    device = device or get_device()
    # weights_only=False because the file also holds plain Python data (vocabulary, settings)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                              # very old PyTorch has no weights_only argument
        ckpt = torch.load(path, map_location="cpu")
    model = CharLSTM(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["vocab"], ckpt


@torch.no_grad()
def generate(model, vocab, seed="", length=500, temperature=0.8, top_k=0, random_seed=None, device=None):
    """Return (seed_used + newly generated text, list of characters removed from the seed).

    The model first reads the seed (to build up its memory), then picks one character at a time
    and feeds each pick back in as the next input.
    """
    device = device or next(model.parameters()).device
    rng = np.random.default_rng(random_seed)
    index = {ch: i for i, ch in enumerate(vocab)}

    seed, removed = keep_known_characters(seed, vocab)
    if not seed:                                   # nothing usable: start from a newline (or any character)
        seed = "\n" if "\n" in index else vocab[0]

    state = None
    ids = torch.tensor([[index[ch] for ch in seed]], dtype=torch.long, device=device)
    logits, state = model(ids, state)              # read the whole seed in one go
    out = []
    for _ in range(int(length)):
        next_id = sample_from_logits(logits[0, -1].cpu().numpy(), temperature, rng, top_k)
        out.append(vocab[next_id])
        logits, state = model(torch.tensor([[next_id]], device=device), state)
    return seed + "".join(out), removed
