"""
backend/app/routes/prediction.py

GET /health and POST /predict.
"""
import logging
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app.schemas.response import HealthResponse, PredictResponse
from backend.app.services.image_service import prepare_upload, read_upload_limited
from backend.app.services.prediction_service import PredictionService
from backend.app.utils.image_validation import ImageValidationError

logger = logging.getLogger("detector.api")
router = APIRouter()


def get_service(request: Request) -> PredictionService:
    return request.app.state.service


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: PredictionService = Depends(get_service)) -> HealthResponse:
    """Is the API up, and did the model load?"""
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        device=str(service.device) if service.is_ready else None,
    )


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="A JPEG, PNG or WebP image."),
    explain: Literal["predicted", "ai"] = Query(
        "predicted", description="'predicted' explains the predicted class; 'ai' explains the AI-GENERATED class."),
    service: PredictionService = Depends(get_service),
) -> PredictResponse:
    """Classify one face image as REAL or AI-GENERATED and return a Grad-CAM explanation."""
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="The model is not available right now. Please try again later.")
    settings = request.app.state.settings
    try:
        data = await read_upload_limited(file, settings.max_upload_bytes)
        prepared = await run_in_threadpool(prepare_upload, data, settings)
        return await run_in_threadpool(service.predict, prepared, explain)
    except ImageValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Something went wrong while analysing the image.")