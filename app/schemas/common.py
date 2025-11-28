"""Common response schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    models_loaded: bool


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    detail: str | None = None

