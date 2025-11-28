#!/usr/bin/env python3
"""Script to download CLIP processor from Hugging Face and save it locally"""

import logging
from pathlib import Path

from transformers import CLIPProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_clip_processor():
    """Download CLIP processor and save to local directory."""
    # Path where processor should be saved
    processor_path = Path("models/clip/clip_processor")
    
    processor_path.mkdir(parents=True, exist_ok=True)
    
    # Check if processor already exists
    if processor_path.exists() and any(processor_path.iterdir()):
        logger.info(f"CLIP processor already exists at {processor_path}")
        logger.info("Skipping download. Delete the directory if you want to re-download.")
        return
    
    logger.info("Downloading CLIP processor from Hugging Face...")
    logger.info("Model: openai/clip-vit-base-patch32")
    
    try:
        # Download processor from Hugging Face
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        logger.info(f"Saving processor to {processor_path}...")
        processor.save_pretrained(str(processor_path))
        
        logger.info("CLIP processor downloaded and saved successfully!")
        logger.info(f"  Location: {processor_path.absolute()}")
        
    except Exception as e:
        logger.error(f"Failed to download CLIP processor: {e}")
        raise


if __name__ == "__main__":
    download_clip_processor()

