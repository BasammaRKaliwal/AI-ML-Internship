"""Train a character-level LSTM on a text file.

Run:   python train.py
Other: python train.py --epochs 15 --hidden 256
       python train.py --corpus my_lyrics.txt
       python train.py --quick          (about a minute: checks that everything works)

Creates: model.pt, history.json, samples.md, images/loss_curve.png
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from common import CORPUS_FILE, MODEL_FILE, CharLSTM, count_parameters, get_device
from generator import generate, load_checkpoint
from report import plot_history, save_history, write_samples
from utils import (build_vocab, clean_text, encode, random_batch, sequential_batches, split_data)


@torch.no_grad()
def evaluate(model, batches, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    for x, y in batches:
        logits, _ = model(torch.from_numpy(x).to(device))
        total += loss_fn(logits.reshape(-1, logits.size(-1)), torch.from_numpy(y).to(device).reshape(-1)).item()
        count += 1
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default=str(CORPUS_FILE))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=0, help="training steps per epoch (0 = about one pass over the text)")
    parser.add_argument("--seq-len", type=int, default=100, help="how many characters the model reads at once")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--embed", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--max-chars", type=int, default=0, help="use only the first N characters (faster)")
    parser.add_argument("--seed-text", default="ROMEO:\n", help="text used to start the sample shown after each epoch")
    parser.add_argument("--quick", action="store_true", help="tiny run just to check that everything works")
    args = parser.parse_args()

    corpus_path = args.corpus
    try:
        text = clean_text(open(corpus_path, encoding="utf-8", errors="ignore").read())
    except FileNotFoundError:
        sys.exit(f"Could not find {corpus_path}.\nRun  python download_corpus.py  or put your own text file there.")
    if args.max_chars:
        text = text[:args.max_chars]
    if args.quick:
        args.epochs, args.steps, args.hidden, args.layers = 2, 30, 128, 1
        text = text[:200_000]

    vocab = build_vocab(text)
    ids = encode(text, vocab)
    train_ids, val_ids = split_data(ids, 0.1)
    if len(val_ids) < args.seq_len * 2 + 2 or len(train_ids) < args.seq_len * 2 + 2:
        sys.exit(f"The text is too short ({len(text):,} characters). Use a longer file or a smaller --seq-len.")

    device = get_device()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = CharLSTM(len(vocab), args.embed, args.hidden, args.layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss()
    steps = args.steps or max(20, len(train_ids) // (args.seq_len * args.batch))
    val_batches = sequential_batches(val_ids, args.batch, args.seq_len, max_batches=40)

    print(f"Text: {len(text):,} characters | vocabulary: {len(vocab)} characters | device: {device}")
    print(f"Model: {args.layers} LSTM layer(s) x {args.hidden} units | {count_parameters(model):,} parameters")
    print(f"Training: {args.epochs} epochs x {steps} steps (batches of {args.batch} x {args.seq_len} characters)")
    print(f"A model that guesses at random has loss {np.log(len(vocab)):.2f}; lower is better.\n")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for _ in range(steps):
            x, y = random_batch(train_ids, args.batch, args.seq_len, rng)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # stops exploding gradients
            optimizer.step()
            running += loss.item()
        train_loss = running / steps
        val_loss = evaluate(model, val_batches, loss_fn, device)
        scheduler.step(val_loss)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))

        marker = ""
        if val_loss < best_val:                          # keep the best model, not just the last one
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "config": model.config, "vocab": vocab,
                        "corpus": str(corpus_path), "epoch": epoch, "val_loss": val_loss}, MODEL_FILE)
            marker = "  <- saved"
        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {train_loss:.3f} | val loss {val_loss:.3f} "
              f"| perplexity {np.exp(val_loss):5.2f} | {time.time() - started:4.0f}s{marker}")
        preview, _ = generate(model, vocab, args.seed_text, 150, 0.8, random_seed=epoch, device=device)
        print("   sample: " + preview.replace("\n", "\n           ") + "\n")

    # Final report uses the best saved model
    best_model, _, ckpt = load_checkpoint(MODEL_FILE, device)
    samples = [(t, generate(best_model, vocab, args.seed_text, 400, t, random_seed=7, device=device)[0])
               for t in (0.3, 0.8, 1.2)]
    plot_history(history)
    save_history(history)
    write_samples(samples, args.seed_text, history, best_model.config)
    print(f"Done. Best validation loss {best_val:.3f} (epoch {ckpt['epoch']}).")
    print("Saved: model.pt, history.json, samples.md, images/loss_curve.png")
    print("Next step:  streamlit run app.py")


if __name__ == "__main__":
    main()
