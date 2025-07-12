"""
Models routes for API.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from typing import List, Dict

from ..config import get_config

router = APIRouter()

@router.get("/models")
def list_models():
    """
    List available models and their status.
    
    Returns:
        dict: Information about available models
    """
    config = get_config()
    models_dir = config.project_root / "models"
    
    models_info = {
        "available_models": [],
        "default_model": config.model_path,
        "models_directory": str(models_dir)
    }
    
    # Check if models directory exists
    if not models_dir.exists():
        models_info["error"] = "Models directory not found"
        return models_info
    
    # Look for model files
    model_extensions = [".pt", ".pth", ".onnx", ".pb"]
    
    for model_file in models_dir.rglob("*"):
        if model_file.is_file() and model_file.suffix.lower() in model_extensions:
            model_info = {
                "name": model_file.name,
                "path": str(model_file.relative_to(config.project_root)),
                "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                "type": model_file.suffix.lower(),
                "status": "available" if model_file.exists() else "missing"
            }
            models_info["available_models"].append(model_info)
    
    # Check default model status
    default_model_path = config.model_path_absolute
    models_info["default_model_status"] = {
        "exists": default_model_path.exists(),
        "path": str(default_model_path),
        "size_mb": round(default_model_path.stat().st_size / (1024 * 1024), 2) if default_model_path.exists() else None
    }
    
    return models_info

@router.get("/models/{model_name}")
def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model file
    
    Returns:
        dict: Detailed model information
    
    Raises:
        HTTPException: If model is not found
    """
    config = get_config()
    models_dir = config.project_root / "models"
    
    # Find the model file
    model_path = None
    for model_file in models_dir.rglob(model_name):
        if model_file.is_file():
            model_path = model_file
            break
    
    if not model_path:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    # Get model information
    stat = model_path.stat()
    
    model_info = {
        "name": model_path.name,
        "path": str(model_path.relative_to(config.project_root)),
        "absolute_path": str(model_path),
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "size_gb": round(stat.st_size / (1024 * 1024 * 1024), 3),
        "type": model_path.suffix.lower(),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "is_default": str(model_path) == str(config.model_path_absolute),
        "status": "available"
    }
    
    return model_info

@router.get("/models/status")
def get_models_status():
    """
    Get overall status of models.
    
    Returns:
        dict: Models status information
    """
    config = get_config()
    models_dir = config.project_root / "models"
    
    status = {
        "models_directory_exists": models_dir.exists(),
        "default_model_exists": config.model_path_absolute.exists(),
        "total_models": 0,
        "total_size_mb": 0,
        "model_types": {}
    }
    
    if models_dir.exists():
        model_extensions = [".pt", ".pth", ".onnx", ".pb"]
        
        for model_file in models_dir.rglob("*"):
            if model_file.is_file() and model_file.suffix.lower() in model_extensions:
                status["total_models"] += 1
                size_mb = model_file.stat().st_size / (1024 * 1024)
                status["total_size_mb"] += size_mb
                
                ext = model_file.suffix.lower()
                if ext not in status["model_types"]:
                    status["model_types"][ext] = 0
                status["model_types"][ext] += 1
        
        status["total_size_mb"] = round(status["total_size_mb"], 2)
    
    return status 