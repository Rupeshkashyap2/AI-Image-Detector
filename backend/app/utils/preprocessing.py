"""
backend/app/utils/preprocessing.py

Geometry applied to every upload before the model's own 224x224 transform.
It is identical to how the training data was prepared (dataset_prepare.py)
and how the external test set is prepared (evaluation/metrics.py):
center-crop to a square, then resize to 256 with LANCZOS.
"""
from PIL import Image

PREP_SIZE = 256


def center_square_resize(img: Image.Image, size: int = PREP_SIZE) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != size:
        img = img.resize((size, size), Image.LANCZOS)
    return img