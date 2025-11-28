"""Dependency injection for API routes"""

from fastapi import Request

from app.ml.classifier import Classifier
from app.ml.clip_encoder import CLIPEncoder
from app.ml.recommender import Recommender
from app.services.image_service import ImageService
from app.services.metadata_service import MetadataService


def get_clip_encoder(request: Request) -> CLIPEncoder:
    """Get CLIP encoder from app state."""
    return request.app.state.clip_encoder


def get_category_classifier(request: Request) -> Classifier:
    """Get category classifier from app state."""
    return request.app.state.category_classifier


def get_style_classifier(request: Request) -> Classifier:
    """Get style classifier from app state."""
    return request.app.state.style_classifier


def get_recommender(request: Request) -> Recommender:
    """Get recommender from app state."""
    return request.app.state.recommender


def get_metadata_service(request: Request) -> MetadataService:
    """Get metadata service from app state."""
    return request.app.state.metadata_service


def get_image_service() -> ImageService:
    """Get image service instance."""
    return ImageService()

