"""Main FastAPI application."""
from dotenv import load_dotenv
load_dotenv()

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import (
    routes_category,
    routes_health,
    routes_recommend,
    routes_style,
)
from app.core.config import settings
from app.core.lifespan import lifespan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Create FastAPI app with lifespan
app = FastAPI(
    title="SimilarPic API",
    description="Fashion Image Classification and Recommendation API",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes_health.router, prefix="/api/v1", tags=["health"])
app.include_router(
    routes_category.router, prefix="/api/v1", tags=["classification"]
)
app.include_router(routes_style.router, prefix="/api/v1", tags=["classification"])
app.include_router(
    routes_recommend.router, prefix="/api/v1", tags=["recommendation"]
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SimilarPic API",
        "version": "1.0.0",
        "docs": "/docs",
    }

