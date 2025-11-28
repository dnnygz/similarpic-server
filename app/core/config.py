"""Application configuration using pydantic-settings."""

from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")

    # Paths
    MODEL_DIR: str = Field(default="models", description="Path to models directory")
    DATA_DIR: str = Field(default="data", description="Path to data directory")

    # ML Configuration
    DEVICE: str = Field(default="cpu", description="Device for inference (cpu or cuda)")
    TOP_K_RECOMMENDATIONS: int = Field(
        default=10, ge=1, le=50, description="Default number of recommendations"
    )
    CONFIDENCE_THRESHOLD: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum confidence for predictions"
    )

    # File upload limits
    MAX_FILE_SIZE: int = Field(
        default=10_485_760, description="Maximum file size in bytes (10MB)"
    )
    ALLOWED_EXTENSIONS: List[str] = Field(
        default_factory=lambda: ["jpg", "jpeg", "png", "webp"],
        description="Allowed image file extensions",
    )

    # CORS
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8000",
        ],
        description="Allowed CORS origins",
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins string into list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def model_dir_path(self) -> Path:
        """Get Path object for model directory."""
        return Path(self.MODEL_DIR)

    @property
    def data_dir_path(self) -> Path:
        """Get Path object for data directory."""
        return Path(self.DATA_DIR)

    @property
    def clip_model_path(self) -> Path:
        """Get path to CLIP model checkpoint."""
        return self.model_dir_path / "clip" / "clip_model.pth"

    @property
    def clip_processor_path(self) -> Path:
        """Get path to CLIP processor directory."""
        return self.model_dir_path / "clip" / "clip_processor"

    @property
    def category_classifier_path(self) -> Path:
        """Get path to category classifier checkpoint."""
        return (
            self.model_dir_path / "classifiers" / "classifier_category_best.pth"
        )

    @property
    def style_classifier_path(self) -> Path:
        """Get path to style classifier checkpoint."""
        return self.model_dir_path / "classifiers" / "classifier_style_best.pth"

    @property
    def faiss_index_path(self) -> Path:
        """Get path to FAISS index file."""
        return self.model_dir_path / "faiss" / "faiss_index_flat.index"

    @property
    def faiss_mapping_path(self) -> Path:
        """Get path to FAISS image ID mapping."""
        return self.model_dir_path / "faiss" / "img_id_mapping.json"

    @property
    def images_csv_path(self) -> Path:
        """Get path to images CSV file."""
        return self.data_dir_path / "images.csv"

    @property
    def styles_csv_path(self) -> Path:
        """Get path to styles CSV file."""
        return self.data_dir_path / "styles.csv"


# Global settings instance
settings = Settings()

