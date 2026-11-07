
from pathlib import Path
from typing import BinaryIO, Union

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb_image(source: Union[str, Path, BinaryIO]) -> Image.Image:
    """Open an image (path or file-like object), fix EXIF rotation, convert to RGB.

    Raises PIL.UnidentifiedImageError / OSError if the file is not a valid image.
    """
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.load()
    return im


def get_eval_transform() -> transforms.Compose:
    """Resize to 224x224 -> tensor in [0,1] -> ImageNet normalization. No randomness."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


_EVAL_TRANSFORM = get_eval_transform()


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """PIL RGB image -> normalized tensor of shape (1, 3, 224, 224) for inference."""
    return _EVAL_TRANSFORM(img).unsqueeze(0)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization (for visualization). Accepts (3,H,W) or (N,3,H,W)."""
    shape = (1, 3, 1, 1) if t.ndim == 4 else (3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(shape)
    std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(shape)
    return (t * std + mean).clamp(0.0, 1.0)