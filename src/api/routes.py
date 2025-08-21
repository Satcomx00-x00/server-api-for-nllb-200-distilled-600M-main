from fastapi import APIRouter, HTTPException, Depends
from typing import Annotated

from src.models.schemas import (
    TranslationRequest,
    TranslationResponse,
    HealthResponse,
)
from src.services.translation_service import NLLBTranslationService

router = APIRouter(prefix="/api/v1")


# Clean dependency injection
def get_translation_service() -> NLLBTranslationService:
    """Get the translation service instance."""
    from src.services.translation_service import translation_service

    return translation_service


TranslationServiceDep = Annotated[
    NLLBTranslationService, Depends(get_translation_service)
]


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    request: TranslationRequest,
    service: TranslationServiceDep,
) -> TranslationResponse:
    """Translate text from source to target language."""
    try:
        translated_text, confidence = await service.translate(
            text=request.text,
            source_lang=request.source_language,
            target_lang=request.target_language,
        )

        return TranslationResponse(
            translated_text=translated_text,
            source_language=request.source_language,
            target_language=request.target_language,
            confidence=confidence,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check(service: TranslationServiceDep) -> HealthResponse:
    """Server health check endpoint."""
    return HealthResponse(
        status="healthy" if service.is_loaded else "starting",
        model_loaded=service.is_loaded,
        supported_languages=service.get_supported_languages(),
    )


@router.get("/languages")
async def get_supported_languages(service: TranslationServiceDep) -> dict:
    """Get supported language codes."""
    return {"supported_languages": service.get_supported_languages()}
