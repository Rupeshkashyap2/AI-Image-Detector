"""
backend/app/services/prediction_service.py

Holds the trained model (loaded once at startup) and turns a prepared image into a response.
Ordinary prediction runs under torch.no_grad(); gradients are enabled only inside Grad-CAM.
"""
import logging
import threading

import torch

from backend.app.config import Settings
from backend.app.schemas.response import PredictResponse
from backend.app.services.gradcam_service import generate_explanation
from backend.app.services.image_service import PreparedImage
from backend.app.utils.preprocessing import PREP_SIZE
from src.data.dataset_loader import CLASS_NAMES
from src.data.preprocessing import preprocess_image
from src.models.model_loader import load_checkpoint

logger = logging.getLogger("detector.prediction")

API_LABELS = {"real": "REAL", "ai_generated": "AI-GENERATED"}
AI_INDEX = CLASS_NAMES.index("ai_generated")


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.device = None
        self._lock = threading.Lock()  # one inference at a time

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def _resolve_device(self) -> torch.device:
        pref = self.settings.device
        if pref == "cuda" and not torch.cuda.is_available():
            logger.warning("DEVICE=cuda requested but CUDA is unavailable; using CPU.")
            return torch.device("cpu")
        if pref == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(pref)

    def set_model(self, model) -> None:
        model.eval()
        self.model = model
        self.device = next(model.parameters()).device

    def load(self) -> bool:
        """Load the checkpoint. Returns False (and logs why) instead of raising."""
        path = self.settings.model_path
        if not path.exists():
            logger.error("Model checkpoint not found: %s", path)  # server log only, never sent to clients
            return False
        try:
            model, ckpt = load_checkpoint(path, self._resolve_device())
        except Exception:
            logger.exception("Failed to load the model checkpoint.")
            return False
        self.set_model(model)
        logger.info("Model loaded on %s (checkpoint epoch %s).", self.device, ckpt.get("epoch"))
        return True

    def predict(self, prepared: PreparedImage, explain: str = "predicted") -> PredictResponse:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded.")

        with self._lock:
            x = preprocess_image(prepared.image).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            target_idx = AI_INDEX if explain == "ai" else pred_idx
            images = generate_explanation(self.model, prepared.image, self.settings.heatmap_alpha, target_idx)

        confidence = float(probs[pred_idx])
        uncertain = confidence < self.settings.uncertain_threshold

        warnings = []
        if uncertain:
            warnings.append(f"The model is not confident about this image ({confidence:.0%}). "
                            f"Treat this result as inconclusive.")
        width, height = prepared.original_size
        if min(width, height) < PREP_SIZE:
            warnings.append(f"Low-resolution image ({width}x{height}). "
                            f"Results are less reliable below {PREP_SIZE} px.")
        factor_limit = self.settings.downscale_warn_factor
        if factor_limit > 0 and prepared.downscale_factor > factor_limit:
            warnings.append(f"This image was reduced about {prepared.downscale_factor:.1f}x to fit the model's "
                            f"input size. The model has not been validated on heavily downscaled images, "
                            f"so treat the result with caution.")

        return PredictResponse(
            prediction=API_LABELS[CLASS_NAMES[pred_idx]],
            confidence=round(confidence, 4),
            class_probabilities={API_LABELS[name]: round(float(probs[i]), 4)
                                 for i, name in enumerate(CLASS_NAMES)},
            uncertain=uncertain,
            warnings=warnings,
            heatmap_target=API_LABELS[CLASS_NAMES[target_idx]],
            heatmap=images.heatmap,
            overlay=images.overlay,
            analyzed_image=images.analyzed_image,
        )