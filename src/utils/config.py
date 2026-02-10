"""
Configuration Management
"""
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import List, Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Accident FIR Automation"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="ALLOWED_ORIGINS"
    )
    
    # Model paths
    MODELS_DIR: Path = Field(default=Path("models"), env="MODELS_DIR")
    YOLO_MODEL_PATH: Path = Field(
        default=Path("models/yolov8_accident.pt"),
        env="YOLO_MODEL_PATH"
    )
    DAMAGE_MODEL_PATH: Path = Field(
        default=Path("models/damage_classifier.h5"),
        env="DAMAGE_MODEL_PATH"
    )
    
    # Model parameters
    YOLO_CONFIDENCE: float = Field(default=0.5, env="YOLO_CONFIDENCE")
    YOLO_IOU: float = Field(default=0.45, env="YOLO_IOU")
    OCR_BACKEND: str = Field(default="paddle", env="OCR_BACKEND")
    USE_GPU: bool = Field(default=False, env="USE_GPU")
    
    # Data directories
    DATA_DIR: Path = Field(default=Path("data"), env="DATA_DIR")
    UPLOAD_DIR: Path = Field(default=Path("data/uploads"), env="UPLOAD_DIR")
    OUTPUT_DIR: Path = Field(default=Path("data/outputs"), env="OUTPUT_DIR")
    
    # File upload settings
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:password@localhost:5432/accident_fir",
        env="DATABASE_URL"
    )
    
    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_DIR: Path = Field(default=Path("logs"), env="LOG_DIR")
    
    # API Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # seconds
    
    # Authentication
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # AWS S3 (optional)
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    AWS_S3_BUCKET: Optional[str] = Field(default=None, env="AWS_S3_BUCKET")
    AWS_REGION: str = Field(default="us-east-1", env="AWS_REGION")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    
    @validator("MODELS_DIR", "YOLO_MODEL_PATH", "DAMAGE_MODEL_PATH", pre=True)
    def validate_paths(cls, v):
        """Ensure paths are Path objects"""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @validator("UPLOAD_DIR", "OUTPUT_DIR", "LOG_DIR")
    def create_directories(cls, v):
        """Create directories if they don't exist"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    WORKERS: int = 8


class TestSettings(Settings):
    """Test environment settings"""
    DEBUG: bool = True
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/accident_fir_test"
    REDIS_URL: str = "redis://localhost:6379/1"


def get_settings() -> Settings:
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()
