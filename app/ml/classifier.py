"""EfficientNet classifier wrapper for category and style classification."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from app.core.config import settings
from app.ml.transforms import get_inference_transforms

logger = logging.getLogger(__name__)


class FashionClassifier(nn.Module):
    """
    EfficientNet-B3 classifier with custom head.

    Architecture must match exactly with training code.
    Supports two architectures:
    - With BatchNorm: Dropout -> Linear -> ReLU -> BatchNorm -> Dropout -> Linear
    - Without BatchNorm: Dropout -> Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(self, num_classes: int, use_batch_norm: bool = True):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features  # 1536
        
        if use_batch_norm:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(p=0.2),
                nn.Linear(512, num_classes),
            )
        else:
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(512, num_classes),
            )

    def forward(self, x):
        return self.backbone(x)


class Classifier:
    """
    Wrapper for EfficientNet classifier with checkpoint loading and inference.

    Handles model loading, preprocessing, and prediction.
    """

    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        """
        Initialize classifier from checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to run inference on (cpu or cuda)
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self.classes: List[str] = []
        self.transforms = get_inference_transforms()

        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load model checkpoint and metadata."""
        logger.info(f"Loading classifier from {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        if "class_to_idx" in checkpoint:
            # Style classifier structure
            self.class_to_idx = checkpoint["class_to_idx"]
            self.idx_to_class = checkpoint.get("idx_to_class", {})
            self.classes = checkpoint.get("classes", [])
        elif "label_to_idx" in checkpoint:
            # Category classifier structure
            self.class_to_idx = checkpoint["label_to_idx"]
            self.idx_to_class = {}  # Will be built below
            self.classes = []  # Will be built below
        else:
            logger.warning("No class mapping found in checkpoint")
            self.class_to_idx = {}
            self.idx_to_class = {}
            self.classes = []

        num_classes = checkpoint.get("num_classes", len(self.class_to_idx))

        if not self.idx_to_class and self.class_to_idx:
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if not self.classes and self.idx_to_class:
            self.classes = [self.idx_to_class[i] for i in range(len(self.idx_to_class))]

        # detect architecture
        state_dict = checkpoint["model_state_dict"]
        # Check if BatchNorm exists (look for running_mean in classifier.3)
        has_batch_norm = any(
            "backbone.classifier.3.running_mean" in key
            for key in state_dict.keys()
        )
        
        logger.info(
            f"Detected architecture: {'with BatchNorm' if has_batch_norm else 'without BatchNorm'}"
        )

        # Initialize model with correct architecture
        self.model = FashionClassifier(
            num_classes=num_classes, use_batch_norm=has_batch_norm
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(
        self, image: Image.Image, top_k: int = None
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predict class for an image.

        Args:
            image: PIL Image in RGB format
            top_k: Number of top predictions to return (None = all)

        Returns:
            Tuple of:
                - predicted_class: str
                - confidence: float
                - all_predictions: List of (class, confidence) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Preprocess
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transforms(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs = probs.cpu().numpy()[0]

        # Map to class names
        predictions = []
        for idx, prob in enumerate(probs):
            class_name = self.idx_to_class.get(idx, f"Class_{idx}")
            predictions.append((class_name, float(prob)))

        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold
        filtered = [
            (cls, conf)
            for cls, conf in predictions
            if conf >= settings.CONFIDENCE_THRESHOLD
        ]

        if not filtered:
            # If nothing passes threshold, return top prediction anyway
            filtered = [predictions[0]] if predictions else []

        # Get top prediction
        predicted_class, confidence = filtered[0]

        # Limit to top_k if specified
        if top_k is not None:
            filtered = filtered[:top_k]

        return predicted_class, confidence, filtered

