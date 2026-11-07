"""
src/data/dataset_prepare.py

Phase 1 ingestion tool. Takes a folder of downloaded images and adds the valid,
unique ones to dataset/raw/<label>/ in a uniform format, logging each image
in dataset/raw/manifest.csv.

Example:
    python src/data/dataset_prepare.py --src D:/downloads/ffhq --label real \
        --source ffhq --limit 2500 --require-face
"""
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 200_000_000  # guard against decompression bombs

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "dataset" / "raw"
MANIFEST = RAW_DIR / "manifest.csv"
FIELDS = ["filename", "label", "source", "original_name", "width", "height", "phash"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Lookup table for fast Hamming distance between 64-bit hashes
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_to_all(hashes: np.ndarray, h: int) -> np.ndarray:
    """Hamming distance between hash h and every hash in `hashes` (uint64 array)."""
    xor = np.bitwise_xor(hashes, np.uint64(h))
    return _POP[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


_cascade = None


def has_face(img: Image.Image) -> bool:
    """Cheap Haar-cascade check. Can miss unusual poses, so use it as a coarse filter."""
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    gray = cv2.cvtColor(np.array(img.resize((256, 256))), cv2.COLOR_RGB2GRAY)
    return len(_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))) > 0


def square_resize(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != size:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def load_manifest() -> list[dict]:
    if not MANIFEST.exists():
        return []
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="folder of downloaded images (searched recursively)")
    ap.add_argument("--label", required=True, choices=["real", "ai_generated"])
    ap.add_argument("--source", required=True, help="short source/generator name, e.g. ffhq, stylegan2")
    ap.add_argument("--limit", type=int, default=5000, help="max images to add from this run")
    ap.add_argument("--size", type=int, default=512, help="output square size")
    ap.add_argument("--min-side", type=int, default=256, help="reject images smaller than this")
    ap.add_argument("--dup-threshold", type=int, default=4, help="max pHash Hamming distance to count as duplicate")
    ap.add_argument("--require-face", action="store_true", help="reject images with no detected face")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.src.is_dir():
        sys.exit(f"Source folder not found: {args.src}")

    out_dir = RAW_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in args.src.rglob("*") if p.suffix.lower() in IMG_EXT]
    random.Random(args.seed).shuffle(files)  # random sample if source > limit
    print(f"Found {len(files)} candidate images in {args.src}")

    # Existing hashes (so repeated runs deduplicate against everything already added)
    existing = load_manifest()
    cap = len(existing) + len(files) + 1
    hashes = np.zeros(cap, dtype=np.uint64)
    n_hashes = 0
    for row in existing:
        hashes[n_hashes] = np.uint64(int(row["phash"], 16))
        n_hashes += 1
    index = sum(1 for r in existing if r["label"] == args.label and r["source"] == args.source)

    new_file = not MANIFEST.exists()
    stats = Counter()
    with open(MANIFEST, "a", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for p in tqdm(files, desc="Processing"):
            if stats["added"] >= args.limit:
                break
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.load()
            except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
                stats["invalid"] += 1
                continue

            if min(im.size) < args.min_side:
                stats["too_small"] += 1
                continue
            if args.require_face and not has_face(im):
                stats["no_face"] += 1
                continue

            orig_w, orig_h = im.size
            im = square_resize(im, args.size)
            h = int(str(imagehash.phash(im)), 16)

            if n_hashes and (hamming_to_all(hashes[:n_hashes], h) <= args.dup_threshold).any():
                stats["duplicate"] += 1
                continue

            index += 1
            name = f"{args.label}_{args.source}_{index:05d}.png"
            im.save(out_dir / name, format="PNG")
            hashes[n_hashes] = np.uint64(h)
            n_hashes += 1
            writer.writerow({
                "filename": name, "label": args.label, "source": args.source,
                "original_name": p.name, "width": orig_w, "height": orig_h,
                "phash": f"{h:016x}",
            })
            stats["added"] += 1

    print("\nSummary:")
    for k in ("added", "invalid", "too_small", "no_face", "duplicate"):
        print(f"  {k:10s}: {stats[k]}")
    if stats["added"] < args.limit:
        print(f"  Note: only {stats['added']} of the requested {args.limit} were added.")


if __name__ == "__main__":
    main()