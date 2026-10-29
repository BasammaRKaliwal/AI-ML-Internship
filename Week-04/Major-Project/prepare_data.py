"""Turn a folder-per-class image dataset into train / val / test folders WITHOUT duplicate leakage.

Run:
    python prepare_data.py --source raw_dataset
    python prepare_data.py --source C:\\path\\to\\dataset --classes cardboard glass metal paper plastic

--source must contain one sub-folder per category (cardboard, glass, ...).

What it does
  1. Finds the images of the chosen categories.
  2. Removes exact duplicate files.
  3. Finds near-duplicates (resized / re-saved copies of the same photo) with an image fingerprint
     and keeps every such group together in ONE split.
  4. Splits 70% train / 15% validation / 15% test, keeping the class proportions.
  5. Checks that no identical or near-identical image ended up in two different splits.

Creates: data/train, data/val, data/test, split_info.json, images/class_distribution.png
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

from paths import DATA_DIR, IMAGES_DIR, SPLIT_INFO
from report import plot_class_distribution
from utils import (IMAGE_SUFFIXES, count_leakage, dhash, drop_exact_duplicates, file_md5,
                   group_near_duplicates, split_by_group)

DEFAULT_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]
SPLIT_NAMES = ["train", "val", "test"]


def find_class_folders(source, wanted):
    """Match the wanted names to the real folder names, ignoring upper/lower case."""
    folders = {p.name.lower(): p for p in source.iterdir() if p.is_dir()}
    missing = [c for c in wanted if c.lower() not in folders]
    if missing:
        sys.exit(f"These categories were not found in {source}: {', '.join(missing)}\n"
                 f"Folders found there: {', '.join(sorted(folders)) or '(none)'}\n"
                 "Check --source, or choose the categories with --classes.")
    return {c: folders[c.lower()] for c in wanted}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--out", default=str(DATA_DIR))
    parser.add_argument("--hash-distance", type=int, default=4,
                        help="images whose fingerprints differ in at most this many of 64 bits count as near-duplicates")
    parser.add_argument("--max-per-class", type=int, default=0, help="use at most this many images per class (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="TrashNet (Thung and Yang, Stanford, 2016)")
    parser.add_argument("--dataset-url", default="https://github.com/garythung/trashnet")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 3 <= len(args.classes) <= 5:
        print(f"Note: the brief asks for 3 to 5 categories and you chose {len(args.classes)}.")
    source, out = Path(args.source), Path(args.out)
    if not source.is_dir():
        sys.exit(f"Source folder not found: {source}")
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            sys.exit(f"{out} already exists. Add --overwrite to replace it.")
        shutil.rmtree(out)

    folders = find_class_folders(source, args.classes)
    paths, labels, found = [], [], {}
    for index, name in enumerate(args.classes):
        files = sorted(p for p in folders[name].rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if args.max_per_class:
            files = files[:args.max_per_class]
        found[name] = len(files)
        paths += files
        labels += [index] * len(files)
        if len(files) < 20:
            sys.exit(f"Only {len(files)} images found for '{name}'. At least 20 are needed.")
    print(f"Found {len(paths)} images: " + ", ".join(f"{k} {v}" for k, v in found.items()))

    print("Fingerprinting images (takes a moment) ...")
    md5s, hashes, good = [], [], []
    unreadable = 0
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as image:
                hashes.append(dhash(image))
            md5s.append(file_md5(path))
            good.append(i)
        except Exception:
            unreadable += 1
    paths = [paths[i] for i in good]
    labels = [labels[i] for i in good]

    keep, n_exact = drop_exact_duplicates(md5s)
    paths = [paths[i] for i in keep]
    labels = [labels[i] for i in keep]
    md5s = [md5s[i] for i in keep]
    hashes = [hashes[i] for i in keep]

    groups = group_near_duplicates(hashes, args.hash_distance)
    sizes = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1
    multi = [g for g, n in sizes.items() if n > 1]
    in_groups = sum(sizes[g] for g in multi)

    splits = split_by_group(groups, labels, seed=args.seed)
    leak_exact, leak_near = count_leakage(md5s, hashes, splits, args.hash_distance)

    counts = {s: {} for s in SPLIT_NAMES}
    for path, label, split, in zip(paths, labels, splits):
        name, split_name = args.classes[label], SPLIT_NAMES[split]
        target = out / split_name / name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / f"{len(list(target.iterdir())):05d}_{path.name}")
        counts[split_name][name] = counts[split_name].get(name, 0) + 1

    info = {"dataset_name": args.dataset_name, "dataset_url": args.dataset_url, "source": str(source),
            "classes": list(args.classes), "seed": args.seed, "hash_distance": args.hash_distance,
            "found": found, "unreadable_files": unreadable, "exact_duplicates_removed": n_exact,
            "near_duplicate_groups": len(multi), "images_in_near_dup_groups": in_groups,
            "leakage_check": {"exact": leak_exact, "near": leak_near}, "splits": counts}
    SPLIT_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    IMAGES_DIR.mkdir(exist_ok=True)
    plot_class_distribution(info)

    print(f"\nExact duplicate files removed : {n_exact}")
    print(f"Near-duplicate groups kept together: {len(multi)} (covering {in_groups} images)")
    print(f"Unreadable files skipped      : {unreadable}")
    print(f"Leakage check across splits   : {leak_exact} identical, {leak_near} near-identical (both should be 0)")
    print("\nSplit sizes:")
    for s in SPLIT_NAMES:
        print(f"  {s:5s} {sum(counts[s].values()):5d}   " + ", ".join(f"{k} {v}" for k, v in counts[s].items()))
    print(f"\nDone. Data is in {out}\nNext step:  python train.py")
    if leak_exact or leak_near:
        print("WARNING: some images still appear in more than one split. Try a smaller --hash-distance or report this.")


if __name__ == "__main__":
    main()
