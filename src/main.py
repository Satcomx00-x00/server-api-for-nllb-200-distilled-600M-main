import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger

from src.config.settings import settings
from src.api.routes import router
from src.api.middleware import setup_middleware
from src.services.translation_service import translation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting NLLB-200 API server...")
    await translation_service.initialize()
    logger.info("Server started successfully")
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    yield
    
    # Shutdown
    logger.info("Shutting down server...")
    # Cleanup if needed


# Create FastAPI app
app = FastAPI(
    title="NLLB-200 Translation API",
    description="FastAPI server for NLLB-200 multilingual translation",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Include routes
app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    )
