"""Classification response schemas."""

from pydantic import BaseModel


class PredictionDetail(BaseModel):
    """Individual prediction with label and confidence."""

    label: str
    confidence: float


class ClassificationResponse(BaseModel):
    """Response schema for classification endpoints."""

    prediction: str
    confidence: float
    all_predictions: list[PredictionDetail]

