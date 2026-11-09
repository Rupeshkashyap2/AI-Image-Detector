"""
backend/app/schemas/response.py

Response models. Images are base64-encoded PNGs without a "data:" prefix;
the frontend adds "data:image/png;base64,".
"""
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["REAL", "AI-GENERATED"]


class PredictResponse(BaseModel):
    prediction: Label
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the predicted class.")
    class_probabilities: Dict[str, float]
    uncertain: bool = Field(description="True when confidence is below the configured threshold.")
    warnings: List[str] = Field(default_factory=list)
    heatmap_target: Label = Field(description="The class the heatmap explains.")
    heatmap: str
    overlay: str
    analyzed_image: str = Field(description="The square image the model actually saw (heatmap is aligned to it).")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "degraded"]
    model_loaded: bool
    device: Optional[str] = None