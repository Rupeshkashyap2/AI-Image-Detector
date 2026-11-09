"""
backend/app/main.py

FastAPI application. Run from the project root:
    python -m uvicorn backend.app.main:app
Interactive docs (Swagger UI): http://127.0.0.1:8000/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import Settings, load_settings
from backend.app.routes.prediction import router
from backend.app.services.prediction_service import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("detector.main")


def create_app(settings: Settings | None = None, model=None) -> FastAPI:
    """`model` lets tests inject a model instead of loading a checkpoint."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = PredictionService(settings)
        if model is not None:
            service.set_model(model)
        elif not service.load():
            logger.error("Starting WITHOUT a model: /predict will return 503 until it is fixed.")
        app.state.service = service
        app.state.settings = settings
        yield

    app = FastAPI(
        title="AI Image Detector API",
        version="1.0.0",
        description=("Estimates whether a face image is REAL or AI-GENERATED and explains the "
                     "prediction with Grad-CAM. The result is a probability, not proof."),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def reject_oversized_requests(request: Request, call_next):
        # Cheap early rejection using the declared size. Requests without a Content-Length
        # header are still limited while reading the upload.
        if request.method == "POST":
            declared = request.headers.get("content-length", "")
            if declared.isdigit() and int(declared) > settings.max_upload_bytes + 1024 * 1024:
                return JSONResponse(status_code=413, content={
                    "detail": f"The file is too large (limit {settings.max_upload_bytes / 1024 / 1024:.0f} MB)."})
        return await call_next(request)

    # Added last so it is the outermost layer and adds CORS headers to every response.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": "AI Image Detector API", "docs": "/docs", "health": "/health"}

    return app


app = create_app()