"""
src/data/dataset_split.py

Phase 2: stratified train/validation/test split of dataset/raw into dataset/processed,
followed by class-balance and leakage verification.

Usage (from the project root):
    python src/data/dataset_split.py                 # create the split (70/15/15)
    python src/data/dataset_split.py --hardlink      # same, but saves disk space
    python src/data/dataset_split.py --force         # rebuild an existing split
    python src/data/dataset_split.py --check-only    # re-verify an existing split
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "dataset" / "raw"
PROCESSED_DIR = ROOT / "dataset" / "processed"
RAW_MANIFEST = RAW_DIR / "manifest.csv"
SPLIT_MANIFEST = PROCESSED_DIR / "split_manifest.csv"

CLASSES = ["real", "ai_generated"]
SPLITS = ["train", "validation", "test"]
COLUMNS = ["filename", "label", "source", "phash"]

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_to_all(hashes: np.ndarray, h: int) -> np.ndarray:
    """Hamming distance between hash h and every hash in `hashes` (uint64 array)."""
    xor = np.bitwise_xor(hashes, np.uint64(h))
    return _POP[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def load_raw_manifest() -> pd.DataFrame:
    if not RAW_MANIFEST.exists():
        sys.exit(f"Manifest not found: {RAW_MANIFEST}. Run dataset_prepare.py first.")
    # dtype=str keeps hex hashes intact (pandas would otherwise parse digit-only hashes as numbers)
    df = pd.read_csv(RAW_MANIFEST, dtype=str)[COLUMNS]
    missing = [r.filename for r in df.itertuples() if not (RAW_DIR / r.label / r.filename).exists()]
    if missing:
        sys.exit(f"{len(missing)} manifest files are missing from dataset/raw (e.g. {missing[:3]}).")
    if df["filename"].duplicated().any():
        sys.exit("Duplicate filenames in the manifest.")
    on_disk = sum(1 for c in CLASSES for _ in (RAW_DIR / c).glob("*.png"))
    if on_disk != len(df):
        print(f"WARNING: {on_disk} PNGs on disk but {len(df)} rows in the manifest. "
              f"Only manifest rows are used.")
    return df


def make_splits(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> dict:
    n = len(df)
    n_val, n_test = round(n * val_frac), round(n * test_frac)
    strata = df["label"] + "|" + df["source"]
    train_df, rest_df = train_test_split(
        df, test_size=n_val + n_test, stratify=strata, random_state=seed, shuffle=True)
    rest_strata = rest_df["label"] + "|" + rest_df["source"]
    val_df, test_df = train_test_split(
        rest_df, test_size=n_test, stratify=rest_strata, random_state=seed, shuffle=True)
    return {
        "train": train_df.assign(split="train"),
        "validation": val_df.assign(split="validation"),
        "test": test_df.assign(split="test"),
    }


def existing_processed_files() -> list:
    return [p for s in SPLITS for c in CLASSES for p in (PROCESSED_DIR / s / c).glob("*.png")]


def materialize(splits: dict, hardlink: bool) -> None:
    for split, sdf in splits.items():
        for label in CLASSES:
            (PROCESSED_DIR / split / label).mkdir(parents=True, exist_ok=True)
        for row in tqdm(sdf.itertuples(), total=len(sdf), desc=f"{split:10s}"):
            src = RAW_DIR / row.label / row.filename
            dst = PROCESSED_DIR / split / row.label / row.filename
            if hardlink:
                try:
                    os.link(src, dst)
                    continue
                except OSError:
                    pass  # fall back to a normal copy
            shutil.copy2(src, dst)


def verify_on_disk(splits: dict) -> bool:
    ok = True
    print("\nOn-disk file check:")
    for split, sdf in splits.items():
        for label in CLASSES:
            expected = set(sdf.loc[sdf["label"] == label, "filename"])
            actual = {p.name for p in (PROCESSED_DIR / split / label).glob("*.png")}
            status = "OK" if expected == actual else "MISMATCH"
            if expected != actual:
                ok = False
            print(f"  {split:10s} {label:13s} manifest={len(expected):5d} disk={len(actual):5d}  {status}")
    return ok


def run_checks(splits: dict, threshold: int) -> bool:
    ok = True
    all_df = pd.concat(splits.values())
    total = len(all_df)

    print("\nSplit sizes and class balance:")
    table = pd.crosstab(all_df["split"], all_df["label"]).reindex(index=SPLITS, columns=CLASSES)
    table["total"] = table.sum(axis=1)
    table["share"] = (table["total"] / total * 100).round(1).astype(str) + "%"
    print(table.to_string())

    print(f"\nLeakage check (near-duplicate = perceptual-hash distance <= {threshold} bits):")
    hashes = {n: np.array([int(h, 16) for h in d["phash"]], dtype=np.uint64) for n, d in splits.items()}
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = set(splits[a]["filename"]) & set(splits[b]["filename"])
            mins = np.array([hamming_to_all(hashes[a], int(h)).min() for h in hashes[b]])
            n_close = int((mins <= threshold).sum())
            print(f"  {a:10s} vs {b:10s}: shared files={len(shared)}, "
                  f"closest pair={int(mins.min())} bits, near-duplicates={n_close}")
            if shared or n_close:
                ok = False
    print("\nLeakage check:", "PASSED" if ok else "FAILED")
    return ok


def load_split_manifest() -> dict:
    if not SPLIT_MANIFEST.exists():
        sys.exit(f"{SPLIT_MANIFEST} not found. Create a split first.")
    df = pd.read_csv(SPLIT_MANIFEST, dtype=str)
    return {s: df[df["split"] == s].copy() for s in SPLITS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leak-threshold", type=int, default=4,
                    help="pHash distance (bits) at or below which two images count as near-duplicates")
    ap.add_argument("--hardlink", action="store_true", help="hard-link instead of copying (saves disk)")
    ap.add_argument("--force", action="store_true", help="delete an existing split and rebuild it")
    ap.add_argument("--check-only", action="store_true", help="only verify an existing split")
    args = ap.parse_args()

    if args.check_only:
        splits = load_split_manifest()
        ok_disk = verify_on_disk(splits)
        ok_leak = run_checks(splits, args.leak_threshold)
        sys.exit(0 if (ok_disk and ok_leak) else 1)

    if args.val_frac + args.test_frac >= 1:
        sys.exit("val-frac + test-frac must be below 1.")

    df = load_raw_manifest()
    existing = existing_processed_files()
    if existing and not args.force:
        sys.exit(f"dataset/processed already contains {len(existing)} images. "
                 f"Re-run with --force to delete them and rebuild the split.")
    for p in existing:
        p.unlink()

    splits = make_splits(df, args.val_frac, args.test_frac, args.seed)
    materialize(splits, args.hardlink)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(splits.values())[COLUMNS + ["split"]].to_csv(SPLIT_MANIFEST, index=False)
    print(f"\nWrote {SPLIT_MANIFEST}")

    ok_disk = verify_on_disk(splits)
    ok_leak = run_checks(splits, args.leak_threshold)
    sys.exit(0 if (ok_disk and ok_leak) else 1)


if __name__ == "__main__":
    main()