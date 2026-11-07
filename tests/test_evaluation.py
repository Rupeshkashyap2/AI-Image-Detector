"""tests/test_evaluation.py"""
import math

import numpy as np
import pytest
from PIL import Image

from src.evaluation.metrics import compute_metrics
from src.evaluation.robustness_test import DownUp, GaussianBlur, GaussianNoise, JpegCompression


def _img(seed=0, size=64):
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_compute_metrics_known_values():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    probs = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]  # predictions: 0 0 0 1 0 1 1 1
    m = compute_metrics(labels, probs)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 3)
    for key in ("accuracy", "precision", "recall", "specificity", "f1"):
        assert m[key] == pytest.approx(0.75)
    assert m["auc"] == pytest.approx(15 / 16)  # 15 of 16 positive/negative pairs ranked correctly


def test_single_class_metrics_are_nan_not_zero():
    m = compute_metrics([1, 1, 1], [0.9, 0.8, 0.2])  # only AI images
    assert math.isnan(m["auc"])
    assert math.isnan(m["specificity"])
    assert m["recall"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("p", [JpegCompression(50), DownUp(0.5), GaussianBlur(1.0), GaussianNoise(5)])
def test_perturbations_keep_size_and_mode(p):
    out = p(_img(), 0)
    assert out.size == (64, 64) and out.mode == "RGB"


def test_noise_is_deterministic_per_index():
    a = np.array(GaussianNoise(5)(_img(), 3))
    b = np.array(GaussianNoise(5)(_img(), 3))
    c = np.array(GaussianNoise(5)(_img(), 4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_jpeg_changes_pixels():
    assert not np.array_equal(np.array(_img()), np.array(JpegCompression(30)(_img(), 0)))