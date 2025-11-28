"""Recommendation response schemas."""

from pydantic import BaseModel


class SimilarItem(BaseModel):
    """Single similar item in recommendation results."""

    img_id: str
    score: float
    image_url: str | None = None
    product_name: str | None = None
    category: str | None = None
    color: str | None = None


class RecommendationResponse(BaseModel):
    """Response schema for recommendation endpoint."""

    query_processed: bool
    similar_items: list[SimilarItem]
    total_results: int

