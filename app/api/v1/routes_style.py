"""Style classification endpoint."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_image_service, get_style_classifier
from app.ml.classifier import Classifier
from app.schemas.classify import ClassificationResponse, PredictionDetail
from app.services.image_service import ImageService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/classify/style", response_model=ClassificationResponse)
async def classify_style(
    file: UploadFile = File(...),
    classifier: Classifier = Depends(get_style_classifier),
    image_service: ImageService = Depends(get_image_service),
):
    """
    Classify the style of a fashion item from an image.

    Args:
        file: Uploaded image file
        classifier: Style classifier (injected)
        image_service: Image service (injected)

    Returns:
        Classification results with prediction and confidence scores
    """
    try:
        # Validate and read image
        image = await image_service.read_image(file)

        # Predict
        predicted_class, confidence, all_predictions = classifier.predict(image)

        # Format response
        response = ClassificationResponse(
            prediction=predicted_class,
            confidence=confidence,
            all_predictions=[
                PredictionDetail(label=label, confidence=conf)
                for label, conf in all_predictions
            ],
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Classification failed: {str(e)}"
        )

