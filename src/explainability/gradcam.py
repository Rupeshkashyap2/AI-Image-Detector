"""
src/explainability/gradcam.py

Phase 5: Grad-CAM for the trained EfficientNetB0 (the same model object used for classification).

Target layer: model.features[-1], the last convolutional block (1280 x 7 x 7 for a 224 x 224 input).
The explained score is the pre-softmax logit of the predicted class (or of a chosen target class).
Grad-CAM shows which spatial regions increased that score. It does not prove an image is
AI-generated; it only reflects evidence the model learned.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocessing import IMG_SIZE, get_eval_transform, load_rgb_image
from src.explainability.heatmap import resize_cam

TARGET_LAYER_NAME = "features[-1]"


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. A constant map (including all zeros) becomes all zeros."""
    cam = np.asarray(cam, dtype=np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-12:
        return np.zeros_like(cam)
    return (cam - lo) / (hi - lo)


@dataclass
class GradCAMResult:
    cam: np.ndarray        # normalized to [0, 1], shape (h, w), e.g. (7, 7)
    raw_cam: np.ndarray    # after ReLU, before normalization
    probs: np.ndarray      # softmax probabilities, shape (num_classes,)
    pred_class: int
    target_class: int


class GradCAM:
    """Grad-CAM for torchvision EfficientNet models (needs .features, .avgpool, .classifier)."""

    def __init__(self, model: nn.Module):
        for attr in ("features", "avgpool", "classifier"):
            if not hasattr(model, attr):
                raise ValueError(f"Model has no '{attr}'; GradCAM expects a torchvision EfficientNet.")
        self.model = model
        self.device = next(model.parameters()).device

    def _forward_with_features(self, x: torch.Tensor):
        feats = self.model.features(x)
        logits = self.model.classifier(torch.flatten(self.model.avgpool(feats), 1))
        return feats, logits

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> GradCAMResult:
        """x: normalized tensor of shape (1, 3, H, W). target_class defaults to the prediction."""
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("x must have shape (1, 3, H, W).")
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                x = x.detach().to(self.device).requires_grad_(True)
                feats, logits = self._forward_with_features(x)
                probs = torch.softmax(logits.detach().float(), dim=1)[0].cpu().numpy()
                pred = int(probs.argmax())
                target = pred if target_class is None else int(target_class)
                if not 0 <= target < logits.shape[1]:
                    raise ValueError(f"target_class must be in [0, {logits.shape[1] - 1}], got {target}.")
                grads = torch.autograd.grad(logits[0, target], feats)[0]
            weights = grads.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
            raw = torch.relu((weights * feats.detach()).sum(dim=1))[0].cpu().numpy()
        finally:
            if was_training:
                self.model.train()
        return GradCAMResult(cam=normalize_cam(raw), raw_cam=raw, probs=probs,
                             pred_class=pred, target_class=target)


# ----------------------------------------------------------------------------- faithfulness check
def _summarize(rows: list):
    if not rows:
        return None
    d_top = np.array([r["p0"] - r["p_top"] for r in rows])
    d_bot = np.array([r["p0"] - r["p_bottom"] for r in rows])
    return {
        "n": len(rows),
        "mean_drop_top": float(d_top.mean()),
        "mean_drop_bottom": float(d_bot.mean()),
        "flip_rate_top": float(np.mean([r["flip_top"] for r in rows])),
        "flip_rate_bottom": float(np.mean([r["flip_bottom"] for r in rows])),
        "share_top_drop_larger": float((d_top > d_bot).mean()),
    }


def deletion_test(model: nn.Module, samples, class_names, fraction: float = 0.2) -> dict:
    """Does masking the highest-ranked Grad-CAM pixels hurt the prediction more than masking
    the lowest-ranked ones? For each image, `fraction` of the 224x224 pixels is set to the
    ImageNet mean colour (0 after normalization), once for the top-ranked and once for the
    bottom-ranked pixels. We record how the probability of the originally predicted class changes.

    Caveat: grey patches are out-of-distribution, so some drop is expected for any mask.
    The comparison between top and bottom masks, not the absolute drop, is the informative part.
    samples: list of (path, true_label_index).
    """
    transform = get_eval_transform()
    model.eval()
    gradcam = GradCAM(model)
    device = gradcam.device
    n_pix = IMG_SIZE * IMG_SIZE
    k = int(round(fraction * n_pix))
    rows = []
    for path, label in samples:
        x = transform(load_rgb_image(path)).unsqueeze(0).to(device)
        res = gradcam(x)
        cam224 = np.clip(resize_cam(res.cam, (IMG_SIZE, IMG_SIZE)), 0.0, 1.0)
        order = np.argsort(cam224.ravel(), kind="stable")
        row = {"label": int(label), "p0": float(res.probs[res.pred_class])}
        for name, idx in (("top", order[-k:]), ("bottom", order[:k])):
            m = np.zeros(n_pix, dtype=bool)
            m[idx] = True
            mask = torch.from_numpy(m.reshape(IMG_SIZE, IMG_SIZE)).to(device)
            xm = x.clone()
            xm[0][:, mask] = 0.0
            with torch.no_grad():
                p = torch.softmax(model(xm).float(), dim=1)[0].cpu().numpy()
            row[f"p_{name}"] = float(p[res.pred_class])
            row[f"flip_{name}"] = int(p.argmax()) != res.pred_class
        rows.append(row)
    return {
        "fraction": fraction,
        "overall": _summarize(rows),
        "by_true_class": {name: _summarize([r for r in rows if r["label"] == i])
                          for i, name in enumerate(class_names)},
    }