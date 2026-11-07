"""tests/test_gradcam.py"""
import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset_loader import CLASS_NAMES
from src.explainability.gradcam import GradCAM, normalize_cam
from src.explainability.heatmap import colorize_cam, resize_cam
from src.explainability.overlay import encode_png_base64, explain_image, overlay_heatmap
from src.models.efficientnet import build_efficientnet_b0


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_efficientnet_b0(pretrained=False).eval()


def test_manual_forward_matches_model(model):
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feats, logits = GradCAM(model)._forward_with_features(x)
        assert torch.allclose(logits, model(x), atol=1e-6)
    assert feats.shape == (1, 1280, 7, 7)


def test_cam_shape_and_range(model):
    res = GradCAM(model)(torch.randn(1, 3, 224, 224))
    assert res.cam.shape == (7, 7)
    assert res.cam.dtype == np.float32
    assert 0.0 <= res.cam.min() and res.cam.max() <= 1.0
    assert res.probs.shape == (2,) and res.probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_matches_hook_based_reference(model):
    x = torch.randn(1, 3, 224, 224)
    res = GradCAM(model)(x, target_class=1)
    store = {}

    def fwd_hook(module, inputs, output):
        store["a"] = output
        output.register_hook(lambda g: store.__setitem__("g", g))

    handle = model.features[-1].register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x.clone().requires_grad_(True))
            logits[0, 1].backward()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    weights = store["g"].mean(dim=(2, 3), keepdim=True)
    reference = torch.relu((weights * store["a"].detach()).sum(dim=1))[0].numpy()
    np.testing.assert_allclose(res.raw_cam, reference, rtol=1e-4, atol=1e-6)


def test_target_class_handling(model):
    x = torch.randn(1, 3, 224, 224)
    gc = GradCAM(model)
    res = gc(x)
    assert res.target_class == res.pred_class
    assert gc(x, target_class=0).target_class == 0
    with pytest.raises(ValueError):
        gc(x, target_class=5)
    with pytest.raises(ValueError):
        gc(torch.randn(2, 3, 224, 224))


def test_weights_untouched_and_mode_restored():
    m = build_efficientnet_b0(pretrained=False)
    m.train()
    GradCAM(m)(torch.randn(1, 3, 224, 224))
    assert m.training
    assert all(p.grad is None for p in m.parameters())


def test_works_inside_no_grad(model):
    with torch.no_grad():
        res = GradCAM(model)(torch.randn(1, 3, 224, 224))
    assert res.cam.shape == (7, 7)


def test_normalize_cam_edge_cases():
    assert not np.isnan(normalize_cam(np.zeros((7, 7)))).any()
    assert np.array_equal(normalize_cam(np.zeros((7, 7))), np.zeros((7, 7), dtype=np.float32))
    assert np.array_equal(normalize_cam(np.full((7, 7), 3.0)), np.zeros((7, 7), dtype=np.float32))
    out = normalize_cam(np.array([[1.0, 2.0], [3.0, 5.0]]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_resize_and_colorize_shapes():
    cam = np.random.default_rng(0).random((7, 7)).astype(np.float32)
    big = resize_cam(cam, (300, 200))  # (width, height)
    assert big.shape == (200, 300)
    heat = colorize_cam(big)
    assert heat.shape == (200, 300, 3) and heat.dtype == np.uint8


def test_overlay_alpha_extremes_and_mismatch():
    img = np.random.default_rng(1).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    heat = np.random.default_rng(2).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    assert np.array_equal(overlay_heatmap(img, heat, alpha=0.0), img)
    assert np.array_equal(overlay_heatmap(img, heat, alpha=1.0), heat)
    with pytest.raises(ValueError):
        overlay_heatmap(img, heat[:10])


def test_explain_image_end_to_end(model):
    img = Image.fromarray(np.random.default_rng(3).integers(0, 255, (200, 300, 3), dtype=np.uint8))
    ex = explain_image(model, img)  # non-square: 300 wide, 200 high
    assert ex.heatmap.shape == (200, 300, 3)
    assert ex.overlay.shape == (200, 300, 3)
    assert ex.prediction in CLASS_NAMES and ex.target_class == ex.prediction
    assert sum(ex.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert ex.confidence == pytest.approx(ex.probabilities[ex.prediction])
    decoded = Image.open(io.BytesIO(base64.b64decode(encode_png_base64(ex.overlay))))
    assert decoded.format == "PNG" and decoded.size == (300, 200)