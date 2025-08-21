import time
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config.settings import settings


def setup_middleware(app) -> None:
    """Configure middleware for the FastAPI app."""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> Response:
        """Log all requests and responses."""
        start_time = time.time()

        # More concise request logging
        logger.info(f"{request.method} {request.url.path} - Started")

        try:
            # Process request
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log completion with timing and status
            logger.info(
                f"{request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)"
            )

            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"{request.method} {request.url.path} - Error: {str(e)} ({process_time:.3f}s)"
            )
            raise
