"""
backend/app/utils/image_validation.py

Validate raw upload bytes and decode them to an RGB PIL image.
The format is detected from the file content; the client's file name and
content type are ignored. Errors carry an HTTP status and a safe message.
"""
import io

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def validate_and_load(data: bytes, max_bytes: int, min_side: int, max_pixels: int) -> Image.Image:
    if not data:
        raise ImageValidationError("The uploaded file is empty.", 400)
    if len(data) > max_bytes:
        raise ImageValidationError(f"The file is too large (limit {max_bytes / 1024 / 1024:.0f} MB).", 413)

    # Cheap header check first: format and dimensions, without decoding the pixels.
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
    except Image.DecompressionBombError:
        raise ImageValidationError("The image dimensions are too large.", 413) from None
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise ImageValidationError("The file is not a valid image.", 400) from None

    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError("Unsupported image format. Please upload a JPEG, PNG or WebP image.", 415)
    if width * height > max_pixels:
        raise ImageValidationError(f"The image dimensions are too large ({width}x{height}).", 413)
    if min(width, height) < min_side:
        raise ImageValidationError(
            f"The image is too small ({width}x{height}). The shortest side must be at least {min_side} px.", 400)

    # Full decode (fails on truncated or corrupted files), fix EXIF rotation, force RGB.
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            rgb = ImageOps.exif_transpose(im).convert("RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise ImageValidationError("The image file is corrupted or incomplete.", 400) from None
    return rgb