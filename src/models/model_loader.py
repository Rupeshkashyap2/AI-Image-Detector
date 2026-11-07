"""
src/models/model_loader.py

Single place to build a model and to save/load checkpoints.
Used by training (Phase 3), evaluation (Phase 4), Grad-CAM (Phase 5) and the backend (Phase 6).
"""
from pathlib import Path

import torch
import torch.nn as nn

from src.data.dataset_loader import CLASS_NAMES
from src.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from src.models.efficientnet import build_efficientnet_b0


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True,
                dropout: float = 0.2, num_classes: int = 2) -> nn.Module:
    if arch == "efficientnet_b0":
        return build_efficientnet_b0(num_classes, pretrained, dropout)
    raise ValueError(f"Unknown architecture: {arch}")


def save_checkpoint(path, model: nn.Module, epoch: int, metrics: dict,
                    arch: str = "efficientnet_b0", dropout: float = 0.2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": arch,
        "num_classes": len(CLASS_NAMES),
        "dropout": float(dropout),
        "class_names": list(CLASS_NAMES),
        "img_size": IMG_SIZE,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "epoch": int(epoch),
        "val_metrics": {k: float(v) for k, v in metrics.items()},
        "model_state_dict": model.state_dict(),
    }, path)


def load_checkpoint(path, device="cpu"):
    """Return (model in eval mode on `device`, checkpoint dict)."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    model = build_model(ckpt["arch"], pretrained=False,
                        dropout=ckpt["dropout"], num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, ckpt