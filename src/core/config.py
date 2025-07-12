"""
Configuration settings for the Match Video Detection application.
"""

from pathlib import Path
from typing import Dict, Any
import os

from src.utils.colors import *

class Config:
    """Configuration class for the application."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    # Model paths
    YOLO_MODEL_PATH = MODELS_DIR / "yolo" / "best.pt"
    
    # Data paths
    RAW_VIDEOS_DIR = DATA_DIR / "raw"
    PROCESSED_VIDEOS_DIR = DATA_DIR / "processed"
    FRAMES_DIR = DATA_DIR / "frames"
    LABELED_DIR = DATA_DIR / "labeled"
    PREDICTIONS_DIR = DATA_DIR / "predictions"
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
    
    # Video processing settings
    DEFAULT_FPS = 30
    FRAME_SKIP = 1  # Process every nth frame
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # Tracking settings
    TRACKING_DISTANCE_THRESHOLD = 250
    TRACKING_INITIALIZATION_DELAY = 3
    TRACKING_HIT_COUNTER_MAX = 90
    
    # Ball tracking settings
    BALL_TRACKING_DISTANCE_THRESHOLD = 150
    BALL_TRACKING_INITIALIZATION_DELAY = 20
    BALL_TRACKING_HIT_COUNTER_MAX = 2000
    
    # Team assignment settings
    TEAM_COLORS = {
        1: RED,    # Red for Home team
        2: BLUE    # Blue for Away team
    }
    
    # Goalkeeper IDs (known goalkeeper mappings)
    GOALKEEPER_TEAM_MAP = {
        285: 2,  # Away team goalkeeper
        155: 1   # Home team goalkeeper
    }
    
    # Pass detection settings
    PASS_MIN_DISTANCE = 50  # Minimum distance for a pass
    PASS_MAX_TIME = 3.0     # Maximum time for a pass (seconds)
    
    # Cross detection settings
    CROSS_MIN_DISTANCE = 100  # Minimum distance for a cross
    CROSS_ANGLE_THRESHOLD = 30  # Minimum angle for a cross (degrees)
    
    # Possession tracking settings
    POSSESSION_FPS = 30
    POSSESSION_MIN_TIME = 0.5  # Minimum time for possession (seconds)
    
    # Visualization settings
    DRAW_PLAYER_BOXES = True
    DRAW_BALL_TRAJECTORY = True
    DRAW_PASS_LINES = True
    DRAW_POSSESSION_OVERLAY = True
    DRAW_SPEED_OVERLAY = True
    DRAW_DISTANCE_OVERLAY = True
    
    # Colors for visualization
    COLORS = {
        "player_home": RED,      # Red
        "player_away": BLUE,      # Blue
        "ball": GREEN,             # Green
        "referee": YELLOW,        # Yellow
        "pass_line": WHITE,    # White
        "cross_line": PINK,     # Magenta
        "possession_home": LIGHT_RED,  # Light red
        "possession_away": LIGHT_BLUE,  # Light blue
        "speed_text": WHITE,   # White
        "distance_text": WHITE # White
    }
    
    # MLflow settings
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = "football-analysis"
    MLFLOW_MODEL_REGISTRY_NAME = "football-detection-model"
    
    # Training settings
    TRAINING_BATCH_SIZE = 16
    TRAINING_EPOCHS = 100
    TRAINING_LEARNING_RATE = 0.01
    TRAINING_IMG_SIZE = 640
    
    @classmethod
    def get_model_path(cls, model_name: str = "best.pt") -> Path:
        """Get the path to a specific model."""
        return cls.MODELS_DIR / "yolo" / model_name
    
    @classmethod
    def get_output_path(cls, video_name: str) -> Path:
        """Get the output path for a processed video."""
        return cls.OUTPUTS_DIR / f"analyzed_{video_name}"
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary directories."""
        directories = [
            cls.MODELS_DIR,
            cls.DATA_DIR,
            cls.OUTPUTS_DIR,
            cls.RAW_VIDEOS_DIR,
            cls.PROCESSED_VIDEOS_DIR,
            cls.FRAMES_DIR,
            cls.LABELED_DIR,
            cls.PREDICTIONS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_paths(cls) -> Dict[str, bool]:
        """Validate that all required paths exist."""
        validation_results = {
            "project_root": cls.PROJECT_ROOT.exists(),
            "models_dir": cls.MODELS_DIR.exists(),
            "yolo_model": cls.YOLO_MODEL_PATH.exists(),
            "data_dir": cls.DATA_DIR.exists(),
            "outputs_dir": cls.OUTPUTS_DIR.exists()
        }
        return validation_results
    
    @classmethod
    def get_settings_dict(cls) -> Dict[str, Any]:
        """Get all settings as a dictionary."""
        return {
            "project_root": str(cls.PROJECT_ROOT),
            "models_dir": str(cls.MODELS_DIR),
            "data_dir": str(cls.DATA_DIR),
            "outputs_dir": str(cls.OUTPUTS_DIR),
            "api_host": cls.API_HOST,
            "api_port": cls.API_PORT,
            "default_fps": cls.DEFAULT_FPS,
            "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
            "tracking_distance_threshold": cls.TRACKING_DISTANCE_THRESHOLD,
            "team_colors": cls.TEAM_COLORS,
            "goalkeeper_team_map": cls.GOALKEEPER_TEAM_MAP
        }


def get_config() -> Config:
    """Get the application configuration instance.
    
    Returns:
        Config instance with all application settings
    """
    return Config() 