"""
src/evaluation/robustness_test.py

Phase 4: how do predictions change when test images are degraded?
Perturbations are applied to each PIL image before the normal evaluation preprocessing.

Run from the project root:
    python -m src.evaluation.robustness_test
"""
import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from src.data.dataset_loader import FaceDataset
from src.data.preprocessing import get_eval_transform, load_rgb_image
from src.evaluation.metrics import (REPORT_DIR, collect_predictions, compute_metrics, fmt,
                                    make_loader)
from src.models.model_loader import load_checkpoint
from src.training.config import GRAPHS_DIR, PROCESSED_DIR, SAVED_MODELS_DIR


# ----------------------------------------------------------------------------- perturbations
class Identity:
    def __call__(self, img, idx):
        return img


class JpegCompression:
    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, img, idx):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class DownUp:
    """Downscale by `scale`, then upscale back to the original size."""

    def __init__(self, scale: float):
        self.scale = scale

    def __call__(self, img, idx):
        w, h = img.size
        small = img.resize((max(1, round(w * self.scale)), max(1, round(h * self.scale))),
                           Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class GaussianBlur:
    def __init__(self, radius: float):
        self.radius = radius

    def __call__(self, img, idx):
        return img.filter(ImageFilter.GaussianBlur(self.radius))


class GaussianNoise:
    """Additive Gaussian noise (sigma in 0-255 pixel units), deterministic per image index."""

    def __init__(self, sigma: float, seed: int = 0):
        self.sigma = sigma
        self.seed = seed

    def __call__(self, img, idx):
        arr = np.asarray(img, dtype=np.float32)
        rng = np.random.default_rng(self.seed + idx)
        noisy = arr + rng.normal(0.0, self.sigma, arr.shape)
        return Image.fromarray(np.clip(noisy, 0, 255).round().astype(np.uint8))


def build_perturbations() -> list:
    return [
        ("JPEG compression", [(f"q={q}", JpegCompression(q)) for q in (95, 75, 50, 30)]),
        ("Resize (down then up)", [(f"{int(s * 100)}%", DownUp(s)) for s in (0.75, 0.5, 0.25)]),
        ("Gaussian blur", [(f"r={r}", GaussianBlur(r)) for r in (0.5, 1.0, 2.0)]),
        ("Gaussian noise", [(f"sigma={s}", GaussianNoise(s)) for s in (2, 5, 10)]),
    ]


class PerturbedDataset(Dataset):
    def __init__(self, samples, perturbation, transform):
        self.samples = list(samples)
        self.perturbation = perturbation
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.perturbation(load_rgb_image(path), idx)
        return self.transform(img), label


# ----------------------------------------------------------------------------- plotting
def plot_robustness(df: pd.DataFrame, clean_acc: float, path) -> None:
    families = [f for f in df["family"].unique() if f != "Clean"]
    fig, axes = plt.subplots(1, len(families), figsize=(4.2 * len(families), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families):
        sub = df[df["family"] == fam]
        bars = ax.bar(sub["level"], sub["accuracy"], color="#4C72B0")
        ax.axhline(clean_acc, color="green", linestyle="--", label=f"clean ({clean_acc:.3f})")
        ax.axhline(0.5, color="gray", linestyle=":", label="chance (0.5)")
        ax.set_title(fam)
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, sub["accuracy"]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=SAVED_MODELS_DIR / "efficientnet_best.pth")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2, help="use 0 if the DataLoader hangs on Windows")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Device: {device} | checkpoint: {args.checkpoint.name} (epoch {ckpt['epoch']})")

    samples = FaceDataset(PROCESSED_DIR / "test", get_eval_transform()).samples
    transform = get_eval_transform()
    settings = [("Clean", "-", Identity())]
    for family, levels in build_perturbations():
        settings += [(family, level, p) for level, p in levels]

    rows = []
    for i, (family, level, perturbation) in enumerate(settings, 1):
        ds = PerturbedDataset(samples, perturbation, transform)
        probs, labels = collect_predictions(model, make_loader(ds, args.batch_size, args.workers), device)
        m = compute_metrics(labels, probs, args.threshold)
        rows.append({
            "family": family, "level": level,
            "accuracy": m["accuracy"], "auc": m["auc"],
            "recall": m["recall"], "specificity": m["specificity"],
            "predicted_ai_rate": float((probs >= args.threshold).mean()),
        })
        print(f"[{i:2d}/{len(settings)}] {family:22s} {level:10s} "
              f"acc {fmt(m['accuracy'])}  auc {fmt(m['auc'])}  "
              f"recall {fmt(m['recall'])}  specificity {fmt(m['specificity'])}")

    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "robustness_results.csv"
    df.to_csv(csv_path, index=False)
    clean_acc = float(df.loc[df["family"] == "Clean", "accuracy"].iloc[0])
    plot_path = GRAPHS_DIR / "robustness_accuracy.png"
    plot_robustness(df, clean_acc, plot_path)

    print("\nSummary (recall = AI images caught, specificity = real images called real,")
    print("predicted_ai_rate = share of all images predicted AI-generated):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nThe Clean row should match test_metrics.json from metrics.py.")
    print(f"Saved: {csv_path}\n       {plot_path}")


if __name__ == "__main__":
    main()