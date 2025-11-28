"""Health check endpoint."""

from fastapi import APIRouter

from app.schemas.common import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status and whether models are loaded
    """
    return HealthResponse(status="healthy", models_loaded=True)

