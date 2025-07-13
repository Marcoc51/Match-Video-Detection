"""
Main FastAPI application for Match Video Detection.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from .middleware import setup_cors
from .routes import health, predict, download, models, monitoring
from .config import get_config, validate_configuration
from .job_manager import start_job_manager, stop_job_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    config = get_config()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Match Video Detection API...")
    
    # Validate configuration
    if not validate_configuration():
        logger.error("Configuration validation failed!")
        raise RuntimeError("Invalid configuration")
    
    # Start job manager
    start_job_manager()
    logger.info("Job manager started")
    
    logger.info(f"API started successfully on {config.host}:{config.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Match Video Detection API...")
    stop_job_manager()
    logger.info("API shutdown complete")


# Create FastAPI app with lifespan
config = get_config()
app = FastAPI(
    title=config.title,
    description=config.description,
    version=config.version,
    lifespan=lifespan
)

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_credentials,
    allow_methods=config.cors_methods,
    allow_headers=config.cors_headers,
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(predict.router, tags=["prediction"])
app.include_router(download.router, tags=["download"])
app.include_router(models.router, tags=["models"])
app.include_router(monitoring.router, tags=["monitoring"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger = logging.getLogger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    logger = logging.getLogger(__name__)
    logger.info(f"{request.method} {request.url}")
    
    response = await call_next(request)
    
    logger.info(f"{request.method} {request.url} - {response.status_code}")
    return response


# For local development: run with `python -m src.api.main`
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=config.reload
    ) 