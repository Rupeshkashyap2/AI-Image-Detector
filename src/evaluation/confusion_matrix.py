"""
src/evaluation/confusion_matrix.py

Confusion-matrix plot (counts plus row percentages).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm, class_names, path, title="Confusion matrix") -> None:
    """cm: 2x2 array [[tn, fp], [fn, tp]] (rows = true class, columns = predicted)."""
    cm = np.asarray(cm, dtype=int)
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros(cm.shape, dtype=float), where=row_sums > 0) * 100
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.1f}%)" for j in range(cm.shape[1])]
                      for i in range(cm.shape[0])])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"fontsize": 12})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)