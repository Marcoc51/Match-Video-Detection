"""
Health check routes for the API.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import psutil
import os
from pathlib import Path

from ..models import HealthResponse, APIInfo, SystemStatus
from ..config import get_config
from ..job_manager import get_job_manager

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse: Current health status of the API
    """
    config = get_config()
    job_manager = get_job_manager()
    
    # Calculate uptime (simplified - in production you'd track start time)
    uptime = 0  # TODO: Implement proper uptime tracking
    
    return HealthResponse(
        status="ok",
        message="Match Video Detection API is running",
        timestamp=datetime.now(),
        version=config.version,
        uptime=uptime
    )

@router.get("/", response_model=APIInfo)
def root():
    """
    Root endpoint with API information.
    
    Returns:
        APIInfo: API information and available endpoints
    """
    config = get_config()
    
    endpoints = {
        "health": "/health",
        "system_status": "/system/status",
        "predict": "/predict",
        "jobs": "/jobs",
        "job_status": "/jobs/{job_id}",
        "cancel_job": "/jobs/{job_id}/cancel",
        "download_video": "/download/{job_id}/video",
        "download_stats": "/download/{job_id}/stats",
        "models": "/models",
        "docs": "/docs",
        "redoc": "/redoc"
    }
    
    features = [
        "Player and ball detection",
        "Pass detection and visualization",
        "Possession tracking",
        "Cross detection",
        "Team assignment",
        "Speed and distance estimation",
        "Video processing and analysis"
    ]
    
    return APIInfo(
        name=config.title,
        version=config.version,
        description=config.description,
        endpoints=endpoints,
        features=features,
        documentation="/docs"
    )

@router.get("/system/status", response_model=SystemStatus)
def system_status():
    """
    Get system status information.
    
    Returns:
        SystemStatus: Current system resource usage and job statistics
    """
    job_manager = get_job_manager()
    
    # Get system metrics
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    memory_usage = memory.percent
    disk = psutil.disk_usage('/')
    disk_usage = disk.percent
    
    # Get GPU usage if available (simplified)
    gpu_usage = None
    try:
        # This would require additional libraries like nvidia-ml-py
        # For now, we'll leave it as None
        pass
    except:
        pass
    
    # Get job statistics
    job_stats = job_manager.get_system_status()
    
    # Calculate uptime (simplified)
    uptime = 0  # TODO: Implement proper uptime tracking
    
    return SystemStatus(
        cpu_usage=cpu_usage,
        memory_usage=memory_usage,
        gpu_usage=gpu_usage,
        disk_usage=disk_usage,
        active_jobs=job_stats["active_jobs"],
        queue_size=job_stats["pending_jobs"],
        uptime=uptime
    )

@router.get("/health/detailed")
def detailed_health_check():
    """
    Detailed health check with system information.
    
    Returns:
        dict: Detailed health information including system resources
    """
    config = get_config()
    job_manager = get_job_manager()
    
    # Validate configuration
    config_issues = config.validate_config()
    
    # Get system information
    system_info = {
        "cpu_count": psutil.cpu_count(),
        "memory_total": psutil.virtual_memory().total,
        "memory_available": psutil.virtual_memory().available,
        "disk_total": psutil.disk_usage('/').total,
        "disk_free": psutil.disk_usage('/').free,
    }
    
    # Get job statistics
    job_stats = job_manager.get_system_status()
    
    health_status = {
        "status": "ok" if not config_issues else "warning",
        "timestamp": datetime.now().isoformat(),
        "version": config.version,
        "configuration": {
            "valid": len(config_issues) == 0,
            "issues": config_issues
        },
        "system": system_info,
        "jobs": job_stats,
        "services": {
            "job_manager": "running",
            "model_loaded": Path(config.model_path_absolute).exists(),
            "output_directories": {
                "output_dir": Path(config.output_dir).exists(),
                "video_output_dir": Path(config.video_output_dir).exists(),
                "results_output_dir": Path(config.results_output_dir).exists(),
            }
        }
    }
    
    return health_status 