import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from loguru import logger

from src.api.middleware import setup_middleware
from src.api.routes import router
from src.config.settings import settings
from src.services.translation_service import translation_service


def _ensure_cuda():
    """
    If nvidia-smi is missing or fails, install the NVIDIA driver + CUDA toolkit
    via the official Ubuntu 22.04 packages.  Idempotent: does nothing if CUDA
    already works.
    """
    try:
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL)
        return  # CUDA already functional
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    print("CUDA not detected – installing NVIDIA driver + CUDA toolkit …")
    cmds = [
        ["apt", "update"],
        ["apt", "install", "-y", "ubuntu-drivers-common"],
        ["ubuntu-drivers", "autoinstall"],
        ["apt", "install", "-y", "cuda-toolkit"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True)
    # Ensure the new libraries are on the dynamic linker path
    cuda_lib = "/usr/local/cuda/lib64"
    if cuda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print("CUDA installation complete – please restart the container if nvidia-smi still fails.")


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
    lifespan=lifespan,
)

# Setup middleware
setup_middleware(app)

# Include routes
app.include_router(router, prefix="/api/v1")

# Run once at import time
_ensure_cuda()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
