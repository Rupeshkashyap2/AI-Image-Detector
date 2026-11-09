"""
tests/test_prediction_api.py

Backend tests with an UNTRAINED EfficientNetB0: predictions are meaningless here; the tests
check the API contract, validation, error handling, CORS and preprocessing consistency.
"""
import base64
import io
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.utils.preprocessing import PREP_SIZE, center_square_resize
from src.models.efficientnet import build_efficientnet_b0

DEFAULTS = dict(
    model_path=Path("does_not_exist.pth"), device="cpu", max_upload_mb=1.0, min_side=64,
    max_pixels=4_000_000, uncertain_threshold=0.9999, downscale_warn_factor=2.0,
    heatmap_alpha=0.4, allowed_origins=("http://localhost:5173",),
)


def make_settings(**overrides) -> Settings:
    return Settings(**{**DEFAULTS, **overrides})


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_efficientnet_b0(pretrained=False).eval()


@pytest.fixture(scope="module")
def client(model):
    with TestClient(create_app(make_settings(), model=model)) as c:
        yield c


def noise_png(size=(300, 200), seed=0) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def solid_bytes(size, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buf, format=fmt)
    return buf.getvalue()


def post_image(client, data, name="photo.png", mime="image/png", **params):
    return client.post("/predict", files={"file": (name, data, mime)}, params=params)


def decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


# ----------------------------------------------------------------------------- happy path
def test_health_with_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "device": "cpu"}


def test_predict_response_contract(client):
    r = post_image(client, noise_png((300, 200)))  # non-square on purpose
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {"REAL", "AI-GENERATED"}
    assert set(body["class_probabilities"]) == {"REAL", "AI-GENERATED"}
    assert sum(body["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["confidence"] == pytest.approx(body["class_probabilities"][body["prediction"]], abs=1e-4)
    assert body["heatmap_target"] == body["prediction"]
    for key in ("heatmap", "overlay", "analyzed_image"):
        img = decode(body[key])
        assert img.format == "PNG" and img.size == (PREP_SIZE, PREP_SIZE)


def test_explain_ai_targets_ai_class(client):
    r = post_image(client, noise_png(), explain="ai")
    assert r.status_code == 200
    assert r.json()["heatmap_target"] == "AI-GENERATED"


def test_invalid_explain_value_is_rejected(client):
    assert post_image(client, noise_png(), explain="banana").status_code == 422


# ----------------------------------------------------------------------------- validation
def test_rejects_non_image(client):
    r = post_image(client, b"this is not an image", "fake.png")
    assert r.status_code == 400
    assert "not a valid image" in r.json()["detail"]


def test_rejects_empty_file(client):
    assert post_image(client, b"").status_code == 400


def test_rejects_unsupported_format(client):
    assert post_image(client, solid_bytes((100, 100), "GIF"), "a.gif", "image/gif").status_code == 415


def test_rejects_too_small_image(client):
    r = post_image(client, solid_bytes((32, 32)))
    assert r.status_code == 400
    assert "too small" in r.json()["detail"]


def test_rejects_too_many_pixels(client):
    assert post_image(client, solid_bytes((2500, 2000))).status_code == 413  # 5 MP > 4 MP limit


def test_rejects_oversized_upload_early(client):
    # ~3 MB is rejected by the size middleware before the body is parsed
    assert post_image(client, b"\x00" * (3 * 1024 * 1024)).status_code == 413


def test_rejects_oversized_upload_while_reading(client):
    # 1.5 MB passes the middleware but exceeds the 1 MB limit while reading
    r = post_image(client, b"\x00" * int(1.5 * 1024 * 1024))
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]


# ----------------------------------------------------------------------------- warnings
def test_uncertain_flag_and_warning(client):
    body = post_image(client, noise_png()).json()  # untrained model: confidence far below 0.9999
    assert body["uncertain"] is True
    assert any("not confident" in w for w in body["warnings"])


def test_confident_when_threshold_is_minimum(model):
    with TestClient(create_app(make_settings(uncertain_threshold=0.5), model=model)) as c:
        body = post_image(c, noise_png()).json()
    assert body["uncertain"] is False
    assert not any("not confident" in w for w in body["warnings"])


def test_low_resolution_warning(client):
    body = post_image(client, solid_bytes((100, 100))).json()
    assert any("Low-resolution" in w for w in body["warnings"])


def test_downscale_warning(client):
    body = post_image(client, solid_bytes((1024, 1024))).json()
    assert any("reduced" in w for w in body["warnings"])


# ----------------------------------------------------------------------------- degraded mode and CORS
def test_service_without_model_is_degraded():
    with TestClient(create_app(make_settings())) as c:  # checkpoint missing, no model injected
        h = c.get("/health")
        assert h.status_code == 200
        assert h.json()["status"] == "degraded" and h.json()["model_loaded"] is False
        r = post_image(c, noise_png())
        assert r.status_code == 503
        assert "does_not_exist" not in r.text  # no server paths in error messages


def test_cors_allows_only_configured_origin(client):
    headers = {"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"}
    allowed = client.options("/predict", headers=headers)
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    denied = client.options("/predict", headers={**headers, "Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in denied.headers


# ----------------------------------------------------------------------------- preprocessing consistency
def test_prep_geometry_matches_training_and_evaluation():
    from src.data.dataset_prepare import square_resize
    from src.evaluation.metrics import center_square_resize as eval_version

    arr = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    ours = np.asarray(center_square_resize(img))
    assert ours.shape == (PREP_SIZE, PREP_SIZE, 3)
    assert np.array_equal(ours, np.asarray(square_resize(img, PREP_SIZE)))
    assert np.array_equal(ours, np.asarray(eval_version(img, PREP_SIZE)))