from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import os


# Suppress transformers warnings about invalid generation flags
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload")
    
    # Model settings
    model_name: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="HuggingFace model name"
    )
    cache_dir: str = Field(default="./model_cache", description="Model cache directory")
    max_length: int = Field(default=512, ge=1, le=2048, description="Maximum sequence length")
    
    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format")
    
    # Security settings
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    allowed_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = "NLLB_"


# Singleton instance
settings = Settings()
