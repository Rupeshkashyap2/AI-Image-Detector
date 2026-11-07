"""
src/explainability/heatmap.py

Turn a small Grad-CAM map (e.g. 7x7, values in [0, 1]) into a colour heatmap
of the original image size.
"""
import cv2
import numpy as np

DEFAULT_COLORMAP = cv2.COLORMAP_JET  # red = strongest influence on the explained class


def resize_cam(cam: np.ndarray, size) -> np.ndarray:
    """Bilinear resize. size = (width, height), e.g. PIL's image.size."""
    out = cv2.resize(np.asarray(cam, dtype=np.float32), (int(size[0]), int(size[1])),
                     interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0.0, 1.0)


def colorize_cam(cam: np.ndarray, colormap: int = DEFAULT_COLORMAP) -> np.ndarray:
    """cam in [0, 1], shape (H, W) -> RGB uint8 image of shape (H, W, 3)."""
    u8 = np.rint(np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, colormap), cv2.COLOR_BGR2RGB)


def cam_to_heatmap(cam: np.ndarray, size, colormap: int = DEFAULT_COLORMAP) -> np.ndarray:
    """Small CAM -> RGB heatmap resized to size = (width, height)."""
    return colorize_cam(resize_cam(cam, size), colormap)