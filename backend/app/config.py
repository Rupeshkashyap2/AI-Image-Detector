"""
backend/app/config.py

Settings read from environment variables (optionally from backend/.env).
Every value has a safe default, so the API runs without any configuration.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / "backend" / ".env")  # silently does nothing if the file is missing

DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"  # Vite dev server


@dataclass(frozen=True)
class Settings:
    model_path: Path
    device: str                    # "auto", "cpu" or "cuda"
    max_upload_mb: float
    min_side: int                  # shortest image side accepted, in pixels
    max_pixels: int                # largest image accepted (width x height)
    uncertain_threshold: float     # confidence below this is flagged as uncertain
    downscale_warn_factor: float   # warn when an image is reduced by more than this (0 = never)
    heatmap_alpha: float
    allowed_origins: tuple

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


def _read(name, default, cast=float, minimum=None, maximum=None):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be a valid {cast.__name__}, got {raw!r}.") from None
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name}={value} is outside the allowed range [{minimum}, {maximum}].")
    return value


def load_settings() -> Settings:
    model_path = Path(os.environ.get("MODEL_PATH", "").strip() or "saved_models/efficientnet_best.pth")
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    device = (os.environ.get("DEVICE", "").strip() or "auto").lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"DEVICE must be auto, cpu or cuda, got {device!r}.")

    origins_raw = os.environ.get("ALLOWED_ORIGINS", "").strip() or DEFAULT_ORIGINS
    origins = tuple(o.strip() for o in origins_raw.split(",") if o.strip())

    return Settings(
        model_path=model_path,
        device=device,
        max_upload_mb=_read("MAX_UPLOAD_MB", 10.0, float, 0.1, 100.0),
        min_side=_read("MIN_IMAGE_SIDE", 128, int, 16, 4096),
        max_pixels=_read("MAX_IMAGE_PIXELS", 50_000_000, int, 10_000, 500_000_000),
        uncertain_threshold=_read("UNCERTAIN_THRESHOLD", 0.90, float, 0.5, 1.0),
        downscale_warn_factor=_read("DOWNSCALE_WARN_FACTOR", 2.0, float, 0.0, 100.0),
        heatmap_alpha=_read("HEATMAP_ALPHA", 0.4, float, 0.0, 1.0),
        allowed_origins=origins,
    )