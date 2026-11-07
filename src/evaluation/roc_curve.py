"""
src/evaluation/roc_curve.py

ROC curve plot for one or more result sets.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve as sk_roc_curve


def plot_roc_curves(curves, path, title="ROC curve") -> int:
    """curves: list of (name, labels, probs). Sets with only one class are skipped.
    Returns the number of curves drawn."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    drawn = 0
    for name, labels, probs in curves:
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = sk_roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")
        drawn += 1
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False positive rate (real images flagged as AI)")
    ax.set_ylabel("True positive rate (AI images caught)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return drawn