"""tests/test_preprocessing.py"""
import io

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.augmentation import get_train_transform
from src.data.dataset_loader import CLASS_NAMES, CLASS_TO_IDX, FaceDataset
from src.data.preprocessing import (
    IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, denormalize,
    get_eval_transform, load_rgb_image, preprocess_image,
)


def _solid(color=(120, 60, 200), size=(256, 256)):
    return Image.new("RGB", size, color)


def _noise(seed=0, size=256):
    arr = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_eval_transform_shape_and_dtype():
    x = get_eval_transform()(_solid())
    assert x.shape == (3, IMG_SIZE, IMG_SIZE)
    assert x.dtype == torch.float32


def test_train_transform_shape():
    x = get_train_transform()(_solid())
    assert x.shape == (3, IMG_SIZE, IMG_SIZE)


def test_eval_transform_is_deterministic():
    img = _noise()
    t = get_eval_transform()
    assert torch.equal(t(img), t(img))


def test_normalization_matches_imagenet():
    x = get_eval_transform()(_solid((255, 255, 255)))
    expected = (1.0 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    assert torch.allclose(x[:, 0, 0], expected, atol=1e-5)


def test_denormalize_roundtrip():
    color = (120, 60, 200)
    x = get_eval_transform()(_solid(color))
    back = denormalize(x)[:, 0, 0]
    assert torch.allclose(back, torch.tensor(color) / 255.0, atol=1e-2)


def test_preprocess_image_adds_batch_dim():
    assert preprocess_image(_solid()).shape == (1, 3, IMG_SIZE, IMG_SIZE)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_load_rgb_image_converts_modes(mode):
    buf = io.BytesIO()
    Image.new(mode, (64, 64)).save(buf, format="PNG")
    buf.seek(0)
    assert load_rgb_image(buf).mode == "RGB"


def test_class_mapping_is_fixed():
    assert CLASS_NAMES == ["real", "ai_generated"]
    assert CLASS_TO_IDX == {"real": 0, "ai_generated": 1}


def test_dataset_reads_labels(tmp_path):
    for name, count in (("real", 2), ("ai_generated", 1)):
        d = tmp_path / name
        d.mkdir()
        for i in range(count):
            _solid().save(d / f"{name}_{i}.png")
    ds = FaceDataset(tmp_path, get_eval_transform())
    assert len(ds) == 3
    assert ds.class_counts() == {"real": 2, "ai_generated": 1}
    x, y = ds[2]
    assert y == CLASS_TO_IDX["ai_generated"]
    assert x.shape == (3, IMG_SIZE, IMG_SIZE)


def test_missing_class_folder_raises(tmp_path):
    (tmp_path / "real").mkdir()
    with pytest.raises(FileNotFoundError):
        FaceDataset(tmp_path, get_eval_transform())