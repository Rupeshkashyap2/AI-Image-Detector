"""
src/explainability/overlay.py

Overlay of the Grad-CAM heatmap on the original image, plus the end-to-end
explain_image() used by the backend, plus a CLI that produces sample sheets and
a faithfulness check.

Run from the project root:
    python -m src.explainability.overlay
"""
import argparse
import base64
import io
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.dataset_loader import CLASS_NAMES, FaceDataset
from src.data.preprocessing import get_eval_transform, load_rgb_image, preprocess_image
from src.explainability.gradcam import GradCAM, TARGET_LAYER_NAME, deletion_test
from src.explainability.heatmap import DEFAULT_COLORMAP, cam_to_heatmap
from src.models.model_loader import load_checkpoint
from src.training.config import PROCESSED_DIR, ROOT, SAVED_MODELS_DIR

GRADCAM_DIR = ROOT / "results" / "gradcam"
REPORT_DIR = ROOT / "results" / "evaluation_report"
PREDICTIONS_CSV = ROOT / "results" / "predictions" / "test_predictions.csv"


# ----------------------------------------------------------------------------- overlay and encoding
def overlay_heatmap(image_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Alpha-blend a colour heatmap onto an RGB uint8 image of the same size."""
    if image_rgb.shape != heatmap_rgb.shape:
        raise ValueError(f"Shape mismatch: image {image_rgb.shape} vs heatmap {heatmap_rgb.shape}")
    blend = (1.0 - alpha) * image_rgb.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    return np.clip(np.rint(blend), 0, 255).astype(np.uint8)


def encode_png_base64(arr: np.ndarray) -> str:
    """RGB uint8 array -> base64 string of a PNG (no 'data:' prefix; the frontend adds it)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------- end to end
@dataclass
class Explanation:
    prediction: str            # class name of the predicted class
    confidence: float          # probability of the predicted class
    probabilities: dict        # class name -> probability
    target_class: str          # class the heatmap explains (the prediction by default)
    cam: np.ndarray            # raw small map in [0, 1], e.g. (7, 7)
    heatmap: np.ndarray        # RGB uint8, same size as the input image
    overlay: np.ndarray        # RGB uint8, same size as the input image


def explain_image(model, img: Image.Image, alpha: float = 0.4, target_class=None,
                  colormap: int = DEFAULT_COLORMAP) -> Explanation:
    """PIL image -> prediction, probabilities, heatmap and overlay, all from one model."""
    img = img.convert("RGB")
    device = next(model.parameters()).device
    res = GradCAM(model)(preprocess_image(img).to(device), target_class)
    heatmap = cam_to_heatmap(res.cam, img.size, colormap)
    overlay = overlay_heatmap(np.asarray(img), heatmap, alpha)
    return Explanation(
        prediction=CLASS_NAMES[res.pred_class],
        confidence=float(res.probs[res.pred_class]),
        probabilities={name: float(res.probs[i]) for i, name in enumerate(CLASS_NAMES)},
        target_class=CLASS_NAMES[res.target_class],
        cam=res.cam, heatmap=heatmap, overlay=overlay,
    )


# ----------------------------------------------------------------------------- sheets (CLI)
def save_sheet(items, path: Path, title: str, model, alpha: float) -> None:
    """items: list of (image path, true label index). One row per image: original, heatmap, overlay."""
    n = len(items)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3.2 * n), squeeze=False)
    print(f"\n{title}")
    for r, (p, true_idx) in enumerate(items):
        p = Path(p)
        img = load_rgb_image(p)
        ex = explain_image(model, img, alpha=alpha)
        true_name = CLASS_NAMES[true_idx]
        p_ai = ex.probabilities["ai_generated"]
        for c, arr in enumerate((np.asarray(img), ex.heatmap, ex.overlay)):
            axes[r, c].imshow(arr)
            axes[r, c].axis("off")
        axes[r, 0].set_title(f"{p.name}\ntrue: {true_name} | pred: {ex.prediction} | p(AI)={p_ai:.3f}",
                             fontsize=8, loc="left")
        if r == 0:
            axes[r, 1].set_title("Grad-CAM heatmap", fontsize=9)
            axes[r, 2].set_title("Overlay", fontsize=9)
        print(f"  {p.name:34s} true={true_name:13s} pred={ex.prediction:13s} p(AI)={p_ai:.3f}")
    fig.suptitle(f"{title}\nRed = regions that raised the score of the predicted class. "
                 f"Explains the model; does not prove an image is AI-generated.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved {path}")


def pick_samples(ds: FaceDataset, per_class: int, seed: int) -> dict:
    rng = random.Random(seed)
    out = {}
    for idx in range(len(CLASS_NAMES)):
        pool = [s for s in ds.samples if s[1] == idx]
        out[idx] = rng.sample(pool, min(per_class, len(pool)))
    return out


def load_top_errors(pred_csv: Path, k: int) -> list:
    """Most confident mistakes from the Phase 4 predictions file."""
    if not pred_csv.exists() or k <= 0:
        return []
    df = pd.read_csv(pred_csv)
    errs = df[df["correct"].astype(str) == "False"].copy()
    errs["confidence"] = np.maximum(errs["prob_ai_generated"], 1 - errs["prob_ai_generated"])
    errs = errs.sort_values("confidence", ascending=False).head(k)
    return [(Path(r.path), CLASS_NAMES.index(r.true_label)) for r in errs.itertuples()
            if Path(r.path).exists()]


def print_deletion_summary(res: dict) -> None:
    print(f"\n=== Faithfulness (deletion) check, mask fraction {res['fraction']:.0%} ===")
    print("Drop = fall in the probability of the originally predicted class after masking.")

    def line(name, s):
        if s is None:
            return
        print(f"  {name:13s} n={s['n']:4d}  drop(top mask)={s['mean_drop_top']:+.4f}  "
              f"drop(bottom mask)={s['mean_drop_bottom']:+.4f}  "
              f"flips top/bottom={s['flip_rate_top']:.3f}/{s['flip_rate_bottom']:.3f}  "
              f"top drop larger in {s['share_top_drop_larger']:.1%} of images")

    line("overall", res["overall"])
    for name, s in res["by_true_class"].items():
        line(name, s)
    print("If the heatmaps are informative, masking the top-ranked regions should hurt more than "
          "masking the bottom-ranked ones.\nThis is a sanity check, not proof of what the model "
          "'really' uses.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=SAVED_MODELS_DIR / "efficientnet_best.pth")
    ap.add_argument("--n", type=int, default=4, help="random test images per class for the sample sheets")
    ap.add_argument("--n-errors", type=int, default=6, help="most confident mistakes to include")
    ap.add_argument("--deletion-n", type=int, default=200, help="images for the deletion check (0 = skip)")
    ap.add_argument("--fraction", type=float, default=0.2, help="pixel fraction masked in the deletion check")
    ap.add_argument("--alpha", type=float, default=0.4, help="heatmap opacity in the overlay")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Device: {device} | checkpoint: {args.checkpoint.name} (epoch {ckpt['epoch']}) "
          f"| target layer: {TARGET_LAYER_NAME}")

    ds = FaceDataset(PROCESSED_DIR / "test", get_eval_transform())
    chosen = pick_samples(ds, args.n, args.seed)
    save_sheet(chosen[0], GRADCAM_DIR / "gradcam_real_samples.png",
               "Grad-CAM: real test images", model, args.alpha)
    save_sheet(chosen[1], GRADCAM_DIR / "gradcam_ai_samples.png",
               "Grad-CAM: AI-generated test images", model, args.alpha)

    errors = load_top_errors(PREDICTIONS_CSV, args.n_errors)
    if errors:
        save_sheet(errors, GRADCAM_DIR / "gradcam_misclassified.png",
                   "Grad-CAM: most confident mistakes on the test set", model, args.alpha)
    else:
        print(f"\nNo predictions file at {PREDICTIONS_CSV}; run src.evaluation.metrics first "
              f"to include misclassified images.")

    if args.deletion_n > 0:
        pool = pick_samples(ds, args.deletion_n // 2, args.seed + 1)
        samples = pool[0] + pool[1]
        res = deletion_test(model, samples, CLASS_NAMES, args.fraction)
        print_deletion_summary(res)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORT_DIR / "gradcam_deletion_test.json", "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved {REPORT_DIR / 'gradcam_deletion_test.json'}")


if __name__ == "__main__":
    main()