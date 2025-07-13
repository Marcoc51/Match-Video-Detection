"""
Configuration management for the API.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class APIConfig(BaseSettings):
    """API configuration settings."""
    
    # API Settings
    title: str = Field(default="Match Video Detection API", description="API title")
    version: str = Field(default="1.0.0", description="API version")
    description: str = Field(default="API for detecting events in football match videos", description="API description")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=True, description="Enable auto-reload for development")
    
    # CORS Settings
    cors_origins: list = Field(default=["*"], description="Allowed CORS origins")
    cors_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_methods: list = Field(default=["*"], description="Allowed CORS methods")
    cors_headers: list = Field(default=["*"], description="Allowed CORS headers")
    
    # File Upload Settings
    max_file_size: int = Field(default=500 * 1024 * 1024, description="Maximum file size in bytes (500MB)")
    allowed_video_formats: list = Field(
        default=[".mp4", ".avi", ".mov", ".mkv", ".webm"], 
        description="Allowed video file formats"
    )
    temp_dir: str = Field(default="/tmp", description="Temporary directory for file processing")
    
    # Processing Settings
    max_concurrent_jobs: int = Field(default=3, description="Maximum concurrent processing jobs")
    job_timeout: int = Field(default=3600, description="Job timeout in seconds (1 hour)")
    cleanup_interval: int = Field(default=300, description="Cleanup interval in seconds (5 minutes)")
    
    # Model Settings
    model_path: str = Field(default="models/yolo/best.pt", description="Path to YOLO model")
    confidence_threshold: float = Field(default=0.5, description="Detection confidence threshold")
    iou_threshold: float = Field(default=0.45, description="Detection IoU threshold")
    
    # Output Settings
    output_dir: str = Field(default="outputs", description="Output directory")
    video_output_dir: str = Field(default="outputs/videos", description="Video output directory")
    results_output_dir: str = Field(default="outputs/results", description="Results output directory")
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="outputs/logs/api.log", description="Log file path")
    
    # MLflow Settings
    mlflow_tracking_uri: str = Field(default="sqlite:///mlruns.db", description="MLflow tracking URI")
    mlflow_experiment_name: str = Field(default="football-analysis", description="MLflow experiment name")
    
    # Security Settings
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    require_api_key: bool = Field(default=False, description="Require API key for requests")
    
    class Config:
        env_file = ".env"
        env_prefix = "API_"
        protected_namespaces = ('settings_',)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            self.temp_dir,
            self.output_dir,
            self.video_output_dir,
            self.results_output_dir,
            Path(self.log_file).parent
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        # Try to find project root by looking for key files
        current = Path(__file__).parent  # src/api
        current = current.parent  # src
        current = current.parent  # project root
        if (current / "main.py").exists() or (current / "requirements.txt").exists():
            return current
        return Path.cwd()
    
    @property
    def model_path_absolute(self) -> Path:
        """Get absolute path to the model."""
        if Path(self.model_path).is_absolute():
            return Path(self.model_path)
        return self.project_root / self.model_path
    
    def validate_config(self) -> list:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check if model exists
        if not self.model_path_absolute.exists():
            issues.append(f"Model not found at: {self.model_path_absolute}")
        
        # Check if output directories are writable
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create output directory: {e}")
        
        # Check if temp directory is writable
        try:
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create temp directory: {e}")
        
        return issues


# Global configuration instance
config = APIConfig()


def get_config() -> APIConfig:
    """Get the global configuration instance."""
    return config


def validate_configuration() -> bool:
    """Validate the configuration and return True if valid."""
    issues = config.validate_config()
    if issues:
        print("Configuration validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    return True 