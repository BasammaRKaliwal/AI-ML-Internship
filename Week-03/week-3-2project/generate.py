"""Generate text from the command line (no web page needed).

Run:   python generate.py --seed "ROMEO:" --length 400 --temperature 0.8
"""
import argparse
import sys

from common import MODEL_FILE
from generator import generate, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="")
    parser.add_argument("--length", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=None)
    args = parser.parse_args()

    if not MODEL_FILE.exists():
        sys.exit("No model.pt found. Run train.py first.")
    model, vocab, _ = load_checkpoint()
    text, removed = generate(model, vocab, args.seed, args.length, args.temperature, args.top_k, args.random_seed)
    if removed:
        print(f"(Ignored characters the model has never seen: {removed})\n")
    print(text)


if __name__ == "__main__":
    main()
