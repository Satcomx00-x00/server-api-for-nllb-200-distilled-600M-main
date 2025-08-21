from typing import List, Optional
from pydantic import BaseModel, Field, validator


class TranslationRequest(BaseModel):
    """Schema for translation requests."""
    
    text: str = Field(..., description="Text to translate", min_length=1)
    source_language: str = Field(..., description="Source language code (e.g., 'eng_Latn')")
    target_language: str = Field(..., description="Target language code (e.g., 'fra_Latn')")
    
    @validator('text')
    def validate_text(cls, v: str) -> str:
        """Validate text is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or just whitespace")
        return v.strip()
    
    @validator('source_language', 'target_language')
    def validate_language_code(cls, v: str) -> str:
        """Validate language code format."""
        if not v or len(v) < 3:
            raise ValueError("Language code must be at least 3 characters")
        return v


class TranslationResponse(BaseModel):
    """Schema for translation responses."""
    
    translated_text: str = Field(..., description="Translated text")
    source_language: str = Field(..., description="Detected source language")
    target_language: str = Field(..., description="Target language")
    confidence: Optional[float] = Field(None, description="Translation confidence score")


class HealthResponse(BaseModel):
    """Schema for health check responses."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    supported_languages: List[str] = Field(..., description="List of supported language codes")


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
