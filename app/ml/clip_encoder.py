"""CLIP encoder wrapper for image embeddings."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    Wrapper for CLIP model to generate image embeddings.

    Uses openai/clip-vit-base-patch32 architecture.
    """

    def __init__(
        self,
        model_path: Path,
        processor_path: Path,
        device: str = "cpu",
    ):
        """
        Initialize CLIP encoder.

        Args:
            model_path: Path to CLIP model checkpoint (.pth)
            processor_path: Path to CLIP processor directory
            device: Device to run inference on (cpu or cuda)
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.processor_path = processor_path
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None

        self._load_model()

    def _load_model(self):
        """Load CLIP model and processor."""
        logger.info(f"Loading CLIP model from {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"CLIP model not found: {self.model_path}")

        # Load processor - use Hugging Face if local path doesn't exist or is empty
        processor_source = "openai/clip-vit-base-patch32"
        if self.processor_path.exists():
            # Check if directory has any files (not just empty directory)
            try:
                if any(self.processor_path.iterdir()):
                    processor_source = str(self.processor_path)
                    logger.info(f"Loading CLIP processor from local path: {processor_source}")
                else:
                    logger.warning(
                        f"CLIP processor directory exists but is empty. "
                        f"Loading from Hugging Face: {processor_source}"
                    )
            except (OSError, PermissionError):
                logger.warning(
                    f"Could not access CLIP processor directory. "
                    f"Loading from Hugging Face: {processor_source}"
                )
        else:
            logger.info(
                f"CLIP processor path not found. Loading from Hugging Face: {processor_source}"
            )

        # Load processor
        self.processor = CLIPProcessor.from_pretrained(processor_source)

        # Load base model
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # Load fine-tuned weights
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # If checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        logger.info("CLIP model loaded successfully")

    def encode(self, image: Image.Image) -> np.ndarray:
        """
        Generate normalized L2 embedding for an image.

        Args:
            image: PIL Image in RGB format

        Returns:
            numpy array of shape (1, 512) with L2-normalized embedding
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor not loaded")

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            # L2 normalization
            features = features / features.norm(dim=-1, keepdim=True)

        # Convert to numpy
        return features.cpu().numpy()  # Shape: (1, 512)

