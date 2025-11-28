"""Similar items recommendation endpoint."""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.deps import (
    get_clip_encoder,
    get_image_service,
    get_metadata_service,
    get_recommender,
)
from app.core.config import settings
from app.ml.clip_encoder import CLIPEncoder
from app.ml.recommender import Recommender
from app.schemas.recommend import RecommendationResponse, SimilarItem
from app.services.image_service import ImageService
from app.services.metadata_service import MetadataService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/recommend/similar", response_model=RecommendationResponse)
async def recommend_similar(
    file: UploadFile = File(...),
    top_k: int = Query(
        default=10, ge=1, le=50, description="Number of recommendations"
    ),
    clip_encoder: CLIPEncoder = Depends(get_clip_encoder),
    recommender: Recommender = Depends(get_recommender),
    metadata_service: MetadataService = Depends(get_metadata_service),
    image_service: ImageService = Depends(get_image_service),
):
    """
    Find similar fashion items to the uploaded image.

    Args:
        file: Uploaded image file
        top_k: Number of recommendations to return (1-50)
        clip_encoder: CLIP encoder (injected)
        recommender: FAISS recommender (injected)
        metadata_service: Metadata service (injected)
        image_service: Image service (injected)

    Returns:
        List of similar items with metadata
    """
    try:
        # Validate and read image
        image = await image_service.read_image(file)

        # Generate CLIP embedding
        embedding = clip_encoder.encode(image)

        # Search in FAISS
        results = recommender.search(embedding, k=top_k)

        # Enrich with metadata
        similar_items = []
        for result in results:
            img_id = result["img_id"]
            score = result["score"]

            # Get image URL
            image_url = metadata_service.get_image_url(img_id)

            # Get product info
            product_info = metadata_service.get_product_info(img_id)
            product_name = (
                product_info.get("productDisplayName") if product_info else None
            )
            category = (
                product_info.get("subCategory") if product_info else None
            )
            color = product_info.get("baseColour") if product_info else None

            similar_items.append(
                SimilarItem(
                    img_id=img_id,
                    score=score,
                    image_url=image_url,
                    product_name=product_name,
                    category=category,
                    color=color,
                )
            )

        return RecommendationResponse(
            query_processed=True,
            similar_items=similar_items,
            total_results=len(similar_items),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Recommendation failed: {str(e)}"
        )

