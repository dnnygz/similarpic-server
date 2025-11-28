"""Metadata service for looking up product information from CSVs."""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MetadataService:
    """
    Service for looking up product metadata from CSV files.

    Loads images.csv and styles.csv once at initialization for fast lookups.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize metadata service.

        Args:
            data_dir: Path to directory containing images.csv and styles.csv
        """
        self.data_dir = data_dir
        self.images_csv_path = data_dir / "images.csv"
        self.styles_csv_path = data_dir / "styles.csv"

        # Lookup dictionaries
        self.image_urls: Dict[str, str] = {}
        self.styles: Dict[str, Dict] = {}

        self._load_data()

    def _load_data(self):
        """Load CSV files and build lookup dictionaries."""
        logger.info(f"Loading metadata from {self.data_dir}")

        # Load images.csv: filename,link
        if not self.images_csv_path.exists():
            logger.warning(f"images.csv not found at {self.images_csv_path}")
        else:
            # Use on_bad_lines='skip' to handle malformed lines gracefully
            try:
                images_df = pd.read_csv(
                    self.images_csv_path,
                    on_bad_lines="skip",
                    engine="python",  # Python engine is more forgiving
                )
            except TypeError:
                # Fallback for older pandas versions
                images_df = pd.read_csv(
                    self.images_csv_path,
                    error_bad_lines=False,
                    warn_bad_lines=True,
                    engine="python",
                )
            # Extract ID from filename (remove .jpg extension)
            images_df["id"] = images_df["filename"].str.replace(".jpg", "")
            # Build lookup dict: {id: url}
            self.image_urls = dict(
                zip(images_df["id"], images_df["link"])
            )
            logger.info(f"Loaded {len(self.image_urls)} image URLs")

        # Load styles.csv
        if not self.styles_csv_path.exists():
            logger.warning(f"styles.csv not found at {self.styles_csv_path}")
        else:
            # Use on_bad_lines='skip' to handle malformed lines gracefully
            try:
                styles_df = pd.read_csv(
                    self.styles_csv_path,
                    on_bad_lines="skip",
                    engine="python",  # Python engine is more forgiving
                )
            except TypeError:
                # Fallback for older pandas versions
                styles_df = pd.read_csv(
                    self.styles_csv_path,
                    error_bad_lines=False,
                    warn_bad_lines=True,
                    engine="python",
                )
            # Convert id to string for consistent lookup
            styles_df["id"] = styles_df["id"].astype(str)
            # Build lookup dict: {id: {all columns as dict}}
            self.styles = styles_df.set_index("id").to_dict("index")
            logger.info(f"Loaded {len(self.styles)} product styles")

    def get_image_url(self, img_id: str) -> Optional[str]:
        """
        Get image URL for a given image ID.

        Args:
            img_id: Image ID as string

        Returns:
            Image URL or None if not found
        """
        return self.image_urls.get(str(img_id))

    def get_product_info(self, img_id: str) -> Optional[Dict]:
        """
        Get product information for a given image ID.

        Args:
            img_id: Image ID as string

        Returns:
            Dictionary with product info (gender, masterCategory, etc.) or None
        """
        return self.styles.get(str(img_id))

