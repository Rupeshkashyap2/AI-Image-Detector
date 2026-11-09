"""
backend/app/services/image_service.py

Reading an upload (with a hard size cap) and turning it into the model-ready square image.
Nothing is written to disk.
"""
from dataclasses import dataclass

from fastapi import UploadFile
from PIL import Image

from backend.app.config import Settings
from backend.app.utils.image_validation import ImageValidationError, validate_and_load
from backend.app.utils.preprocessing import PREP_SIZE, center_square_resize

CHUNK_SIZE = 1024 * 1024


@dataclass
class PreparedImage:
    image: Image.Image            # PREP_SIZE x PREP_SIZE, RGB
    original_size: tuple          # (width, height) after EXIF rotation
    downscale_factor: float       # shortest original side / PREP_SIZE


async def read_upload_limited(file: UploadFile, max_bytes: int) -> bytes:
    chunks, total = [], 0
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ImageValidationError(f"The file is too large (limit {max_bytes / 1024 / 1024:.0f} MB).", 413)
        chunks.append(chunk)
    return b"".join(chunks)


def prepare_upload(data: bytes, settings: Settings) -> PreparedImage:
    img = validate_and_load(data, settings.max_upload_bytes, settings.min_side, settings.max_pixels)
    original_size = img.size
    return PreparedImage(
        image=center_square_resize(img),
        original_size=original_size,
        downscale_factor=min(original_size) / PREP_SIZE,
    )