"""Image validation and preprocessing service."""

import logging
from io import BytesIO
from typing import Tuple

from fastapi import UploadFile
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


class ImageService:
    """Service for validating and processing uploaded images."""

    @staticmethod
    def validate_file(file: UploadFile) -> Tuple[bool, str]:
        """
        Validate uploaded file.

        Args:
            file: FastAPI UploadFile

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check extension
        if not file.filename:
            return False, "No filename provided"

        extension = file.filename.split(".")[-1].lower()
        if extension not in settings.ALLOWED_EXTENSIONS:
            return (
                False,
                f"Invalid file extension. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}",
            )

        return True, ""

    @staticmethod
    async def read_image(file: UploadFile) -> Image.Image:
        """
        Read and validate image from uploaded file.

        Args:
            file: FastAPI UploadFile

        Returns:
            PIL.Image in RGB format

        Raises:
            ValueError: If file is invalid or cannot be read
        """
        # Validate extension
        is_valid, error_msg = ImageService.validate_file(file)
        if not is_valid:
            raise ValueError(error_msg)

        # Read file content
        content = await file.read()

        # Check file size
        if len(content) > settings.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large. Maximum size: {settings.MAX_FILE_SIZE / 1_048_576:.1f}MB"
            )

        # Try to open as image
        try:
            image = Image.open(BytesIO(content))
            # Verify it's a valid image by attempting to load it
            image.verify()
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")

        # Reopen after verify (verify closes the image)
        image = Image.open(BytesIO(content))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

