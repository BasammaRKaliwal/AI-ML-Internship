"""OPTIONAL: try to download TrashNet from Hugging Face into raw_dataset/<category>/.

Run:   pip install datasets
       python download_dataset.py

If this fails (a different dataset layout, no internet), download a waste dataset by hand instead:
any dataset with one folder per category works (see README.md), then use
    python prepare_data.py --source <that folder>
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="garythung/trashnet", help="Hugging Face dataset name")
    parser.add_argument("--out", default=str(BASE / "raw_dataset"))
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Run   pip install datasets   first.")

    print(f"Downloading {args.repo} ...")
    try:
        data = load_dataset(args.repo, split="train")
    except Exception as error:
        sys.exit(f"Download failed: {error}\nUse the manual download instead (see README.md).")
    if "image" not in data.features or "label" not in data.features:
        sys.exit(f"Unexpected columns: {list(data.features)}. Use the manual download instead (see README.md).")

    names = data.features["label"].names
    out = Path(args.out)
    for i, row in enumerate(data):
        folder = out / names[row["label"]]
        folder.mkdir(parents=True, exist_ok=True)
        row["image"].convert("RGB").save(folder / f"{i:05d}.jpg", quality=92)
    print(f"Saved {len(data)} images in {out}")
    print("Categories: " + ", ".join(names))
    print("Next step:  python prepare_data.py --source raw_dataset")


if __name__ == "__main__":
    main()
