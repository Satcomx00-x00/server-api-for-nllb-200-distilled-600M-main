from typing import Annotated
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.config.settings import settings
from src.services.translation_service import NLLBTranslationService


# Global service instance
translation_service = NLLBTranslationService()


async def get_translation_service() -> NLLBTranslationService:
    """Dependency to get translation service."""
    return translation_service


# Security setup
security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> None:
    """Verify API key if configured."""
    if settings.api_key:
        if not credentials or credentials.credentials != settings.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )


# Type aliases for cleaner dependencies
TranslationServiceDep = Annotated[NLLBTranslationService, Depends(get_translation_service)]
SecurityDep = Annotated[None, Depends(verify_api_key)]
