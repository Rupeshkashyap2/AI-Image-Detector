"""
backend/app/services/gradcam_service.py

Thin wrapper around the Phase 5 explain_image(): same model object, results as base64 PNGs.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from src.explainability.overlay import encode_png_base64, explain_image


@dataclass
class ExplanationImages:
    heatmap: str
    overlay: str
    analyzed_image: str


def generate_explanation(model, image: Image.Image, alpha: float,
                         target_class: Optional[int]) -> ExplanationImages:
    ex = explain_image(model, image, alpha=alpha, target_class=target_class)
    return ExplanationImages(
        heatmap=encode_png_base64(ex.heatmap),
        overlay=encode_png_base64(ex.overlay),
        analyzed_image=encode_png_base64(np.asarray(image)),
    )