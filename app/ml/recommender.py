"""FAISS-based recommender for similar image search."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class Recommender:
    """
    FAISS-based recommender for finding similar images.

    Uses L2 distance in normalized embedding space.
    """

    def __init__(
        self,
        index_path: Path,
        mapping_path: Path,
    ):
        """
        Initialize recommender with FAISS index and image ID mapping.

        Args:
            index_path: Path to FAISS index file (.index)
            mapping_path: Path to JSON file with img_ids mapping
        """
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.index: Optional[faiss.Index] = None
        self.img_ids: List[str] = []

        self._load_index()

    def _load_index(self):
        """Load FAISS index and image ID mapping."""
        logger.info(f"Loading FAISS index from {self.index_path}")

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.mapping_path.exists():
            raise FileNotFoundError(
                f"Image ID mapping not found: {self.mapping_path}"
            )

        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))

        # Load image ID mapping
        with open(self.mapping_path, "r") as f:
            mapping_data = json.load(f)
            self.img_ids = mapping_data.get("img_ids", [])

        if len(self.img_ids) != self.index.ntotal:
            logger.warning(
                f"Mismatch: index has {self.index.ntotal} vectors, "
                f"but mapping has {len(self.img_ids)} IDs"
            )

        logger.info(
            f"Loaded FAISS index with {self.index.ntotal} vectors and "
            f"{len(self.img_ids)} image IDs"
        )

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> List[Dict[str, float]]:
        """
        Search for similar images.

        Args:
            query_embedding: Normalized embedding array of shape (1, 512)
            k: Number of results to return

        Returns:
            List of dicts with 'img_id' and 'score' (similarity 0-1)
        """
        if self.index is None:
            raise RuntimeError("Index not loaded")

        # Ensure float32 and correct shape
        query_embedding = query_embedding.astype("float32")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Convert distances to similarity scores (1 - distance for L2)
        # Note: For normalized vectors, distance range is [0, 2]
        # Similarity = 1 - (distance / 2) or just 1 - distance for normalized
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.img_ids):
                img_id = self.img_ids[idx]
                # Convert distance to similarity (closer to 0 = more similar)
                # For normalized L2, max distance is 2, so similarity = 1 - (dist/2)
                similarity = max(0.0, 1.0 - (float(dist) / 2.0))
                results.append({"img_id": img_id, "score": similarity})
            else:
                logger.warning(f"Index {idx} out of range for img_ids")

        return results

