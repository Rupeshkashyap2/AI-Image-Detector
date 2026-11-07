"""
src/evaluation/metrics.py

Phase 4: metric computation and the main evaluation run.
  - Clean evaluation on the held-out test split (dataset/processed/test)
  - Optional external-generator evaluation (dataset/external_test)
Positive class = ai_generated (label 1). Default decision threshold = 0.5.

Run from the project root:
    python -m src.evaluation.metrics
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.data.dataset_loader import CLASS_NAMES, FaceDataset
from src.data.preprocessing import get_eval_transform, load_rgb_image
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.roc_curve import plot_roc_curves
from src.models.model_loader import load_checkpoint
from src.training.config import GRAPHS_DIR, PROCESSED_DIR, ROOT, SAVED_MODELS_DIR

RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CM_DIR = RESULTS_DIR / "confusion_matrix"
REPORT_DIR = RESULTS_DIR / "evaluation_report"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PREP_SIZE = 256  # size used when the training data was prepared


# ----------------------------------------------------------------------------- helpers
def _safe_div(a, b) -> float:
    return a / b if b else float("nan")


def fmt(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return None if math.isnan(v) else v
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    return o


def center_square_resize(img: Image.Image, size: int) -> Image.Image:
    """Same geometry as dataset_prepare.py: center-crop to a square, resize to size x size."""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != size:
        img = img.resize((size, size), Image.LANCZOS)
    return img


# ----------------------------------------------------------------------------- metrics
def compute_metrics(labels, probs, threshold: float = 0.5) -> dict:
    """labels: 0 = real, 1 = ai_generated. probs: probability of ai_generated.
    Undefined quantities (zero denominator, single-class AUC) are NaN."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    n = int(len(labels))
    both = len(np.unique(labels)) == 2
    return {
        "n": n,
        "threshold": float(threshold),
        "accuracy": (tp + tn) / n,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "auc": float(roc_auc_score(labels, probs)) if both else float("nan"),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def confidence_table(labels, probs, threshold: float = 0.5) -> list:
    """Accuracy within bins of model confidence (confidence = max class probability)."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    conf = np.maximum(probs, 1.0 - probs)
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0000001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        rows.append({
            "confidence_range": f"{lo:.2f}-{min(hi, 1.0):.2f}",
            "n": n,
            "accuracy": float((preds[m] == labels[m]).mean()) if n else None,
        })
    return rows


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Return (probability of ai_generated, true label) as numpy arrays, in loader order."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


# ----------------------------------------------------------------------------- data
class ImageListDataset(Dataset):
    """(path, label) samples; optionally center-crops and resizes like the training data."""

    def __init__(self, samples, transform, square_size=None):
        self.samples = list(samples)
        self.transform = transform
        self.square_size = square_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_rgb_image(path)
        if self.square_size:
            img = center_square_resize(img, self.square_size)
        return self.transform(img), label


def make_loader(dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


def _is_valid_image(path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def gather_external_samples(external_dir) -> list:
    samples, skipped = [], 0
    for sub, label in (("unseen_real", 0), ("unseen_ai", 1)):
        d = Path(external_dir) / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                if _is_valid_image(p):
                    samples.append((p, label))
                else:
                    skipped += 1
    if skipped:
        print(f"Skipped {skipped} unreadable external images.")
    return samples


# ----------------------------------------------------------------------------- reporting
def print_report(title: str, m: dict) -> None:
    print(f"\n=== {title} (n={m['n']}) ===")
    print(f"  accuracy    {fmt(m['accuracy'])}")
    print(f"  precision   {fmt(m['precision'])}   (images flagged AI that really were AI)")
    print(f"  recall      {fmt(m['recall'])}   (AI images that were caught)")
    print(f"  specificity {fmt(m['specificity'])}   (real images correctly called real)")
    print(f"  f1-score    {fmt(m['f1'])}")
    print(f"  AUC         {fmt(m['auc'])}")
    print(f"  threshold   {m['threshold']}")
    print("  confusion matrix (rows = true, columns = predicted):")
    print("                  pred real    pred ai")
    print(f"    true real    {m['tn']:9d}  {m['fp']:9d}")
    print(f"    true ai      {m['fn']:9d}  {m['tp']:9d}")


def save_predictions(samples, labels, probs, threshold, path) -> pd.DataFrame:
    assert len(samples) == len(labels) == len(probs), "prediction/sample count mismatch"
    labels = np.asarray(labels).astype(int)
    preds = (np.asarray(probs) >= threshold).astype(int)
    df = pd.DataFrame({
        "path": [str(p) for p, _ in samples],
        "true_label": [CLASS_NAMES[v] for v in labels],
        "pred_label": [CLASS_NAMES[v] for v in preds],
        "prob_ai_generated": np.round(np.asarray(probs, dtype=float), 6),
        "correct": labels == preds,
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def print_top_errors(df: pd.DataFrame, k: int = 10) -> None:
    errs = df[~df["correct"]].copy()
    if errs.empty:
        print("\nNo misclassified images.")
        return
    errs["confidence"] = np.maximum(errs["prob_ai_generated"], 1 - errs["prob_ai_generated"])
    errs = errs.sort_values("confidence", ascending=False).head(k)
    print(f"\nMost confident mistakes (top {len(errs)}):")
    for r in errs.itertuples():
        print(f"  {Path(r.path).name:34s} true={r.true_label:13s} "
              f"pred={r.pred_label:13s} p(ai)={r.prob_ai_generated:.3f}")


def evaluate_dataset(tag, title, model, dataset, device, threshold, batch_size, workers):
    loader = make_loader(dataset, batch_size, workers)
    probs, labels = collect_predictions(model, loader, device)
    metrics = compute_metrics(labels, probs, threshold)
    df = save_predictions(dataset.samples, labels, probs, threshold,
                          PREDICTIONS_DIR / f"{tag}_predictions.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], CLASS_NAMES,
                          CM_DIR / f"{tag}_confusion_matrix.png", f"Confusion matrix - {title}")
    print_report(title, metrics)
    return metrics, labels, probs, df


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=SAVED_MODELS_DIR / "efficientnet_best.pth")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2, help="use 0 if the DataLoader hangs on Windows")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--external-dir", type=Path, default=ROOT / "dataset" / "external_test")
    args = ap.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"Device: {device} | checkpoint: {args.checkpoint.name} (saved at epoch {ckpt['epoch']})")
    for d in (PREDICTIONS_DIR, CM_DIR, REPORT_DIR, GRAPHS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- clean test split
    test_ds = FaceDataset(PROCESSED_DIR / "test", get_eval_transform())
    metrics, labels, probs, df = evaluate_dataset(
        "test", "Test set", model, test_ds, device, args.threshold, args.batch_size, args.workers)
    plot_roc_curves([("Test set", labels, probs)], GRAPHS_DIR / "test_roc_curve.png",
                    "ROC curve - test set")

    conf_rows = confidence_table(labels, probs, args.threshold)
    print("\nAccuracy by model confidence:")
    for r in conf_rows:
        acc = "n/a" if r["accuracy"] is None else f"{r['accuracy']:.4f}"
        print(f"  confidence {r['confidence_range']}: n={r['n']:5d}  accuracy={acc}")
    print_top_errors(df)

    val_m = ckpt.get("val_metrics", {})
    print(f"\nFor comparison, validation metrics at the saved checkpoint: "
          f"accuracy {fmt(val_m.get('accuracy'))}, AUC {fmt(val_m.get('auc'))}")

    report = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_epoch": ckpt["epoch"],
        "validation_metrics_at_checkpoint": val_m,
        "test_metrics": metrics,
        "accuracy_by_confidence": conf_rows,
    }
    with open(REPORT_DIR / "test_metrics.json", "w") as f:
        json.dump(to_jsonable(report), f, indent=2)

    # ---- external generator test (optional)
    ext_samples = gather_external_samples(args.external_dir)
    if not ext_samples:
        print(f"\nNo external-test images found under {args.external_dir} "
              f"(unseen_real / unseen_ai). External test skipped.")
    else:
        n_real = sum(1 for _, l in ext_samples if l == 0)
        n_ai = len(ext_samples) - n_real
        print(f"\nExternal set: {n_real} real, {n_ai} AI-generated images.")
        ext_ds = ImageListDataset(ext_samples, get_eval_transform(), square_size=PREP_SIZE)
        ext_m, ext_labels, ext_probs, _ = evaluate_dataset(
            "external", "External test set", model, ext_ds, device,
            args.threshold, args.batch_size, args.workers)
        drawn = plot_roc_curves(
            [("Test set", labels, probs), ("External set", ext_labels, ext_probs)],
            GRAPHS_DIR / "roc_test_vs_external.png", "ROC curve - test vs external")
        if drawn < 2:
            print("  (External ROC/AUC not available: the external set has only one class.)")
        with open(REPORT_DIR / "external_metrics.json", "w") as f:
            json.dump(to_jsonable({"external_metrics": ext_m, "n_real": n_real, "n_ai": n_ai}), f, indent=2)

    print(f"\nSaved reports to {REPORT_DIR}, plots to {CM_DIR} and {GRAPHS_DIR}, "
          f"predictions to {PREDICTIONS_DIR}")


if __name__ == "__main__":
    main()