"""Application lifespan management for model loading."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.core.config import settings
from app.ml.classifier import Classifier
from app.ml.clip_encoder import CLIPEncoder
from app.ml.recommender import Recommender
from app.services.metadata_service import MetadataService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application lifespan: load models on startup, cleanup on shutdown.

    Args:
        app: FastAPI application instance
    """
    # Startup: Load models
    logger.info("Starting up: Loading ML models...")

    try:
        # Determine device
        device = settings.DEVICE
        if device == "cuda" and not __import__("torch").cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            device = "cpu"

        # Load CLIP encoder
        logger.info("Loading CLIP encoder...")
        clip_encoder = CLIPEncoder(
            model_path=settings.clip_model_path,
            processor_path=settings.clip_processor_path,
            device=device,
        )
        app.state.clip_encoder = clip_encoder
        logger.info("CLIP encoder loaded")

        # Load category classifier
        logger.info("Loading category classifier...")
        category_classifier = Classifier(
            checkpoint_path=settings.category_classifier_path,
            device=device,
        )
        app.state.category_classifier = category_classifier
        logger.info("Category classifier loaded")

        # Load style classifier
        logger.info("Loading style classifier...")
        style_classifier = Classifier(
            checkpoint_path=settings.style_classifier_path,
            device=device,
        )
        app.state.style_classifier = style_classifier
        logger.info("Style classifier loaded")

        # Load FAISS recommender
        logger.info("Loading FAISS recommender...")
        recommender = Recommender(
            index_path=settings.faiss_index_path,
            mapping_path=settings.faiss_mapping_path,
        )
        app.state.recommender = recommender
        logger.info("FAISS recommender loaded")

        # Load metadata service
        logger.info("Loading metadata service...")
        metadata_service = MetadataService(data_dir=settings.data_dir_path)
        app.state.metadata_service = metadata_service
        logger.info("Metadata service loaded")

        logger.info("All models loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load models: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down: Cleaning up...")

