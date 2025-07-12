"""
Data Loader Block for Football Video Analysis Pipeline
Handles video input from various sources (API, file system, cloud storage)
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager, JobState

logger = logging.getLogger(__name__)


@data_loader
def load_video_data(*args, **kwargs) -> Dict[str, Any]:
    """
    Load video data from various sources.
    
    This block handles:
    - API-triggered video processing
    - File system video loading
    - Cloud storage integration
    - Video metadata extraction
    
    Returns:
        Dict containing video data and metadata
    """
    try:
        # Get configuration
        config = get_config()
        
        # Get job information from upstream trigger
        job_id = kwargs.get('job_id')
        if job_id:
            return load_from_api_job(job_id, config)
        else:
            return load_from_file_system(kwargs, config)
            
    except Exception as e:
        logger.error(f"Error in data loader: {e}")
        raise


def load_from_api_job(job_id: int, config) -> Dict[str, Any]:
    """Load video data from API job."""
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    
    if not job:
        raise ValueError(f"Job {job_id} not found")
    
    # Update job status
    job_manager.update_job(job_id, progress=15, message="Loading video data...")
    
    # Get video file path
    video_path = get_video_path_from_job(job, config)
    
    # Extract video metadata
    metadata = extract_video_metadata(video_path)
    
    # Prepare data for pipeline
    data = {
        'job_id': job_id,
        'video_path': str(video_path),
        'filename': job.filename,
        'features': job.features.dict() if job.features else {},
        'metadata': metadata,
        'source': 'api',
        'timestamp': job.created_at.isoformat()
    }
    
    logger.info(f"Loaded video data for job {job_id}: {job.filename}")
    return data


def load_from_file_system(kwargs: Dict, config) -> Dict[str, Any]:
    """Load video data from file system."""
    video_path = kwargs.get('video_path')
    if not video_path:
        raise ValueError("video_path parameter is required")
    
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Extract video metadata
    metadata = extract_video_metadata(video_path)
    
    # Prepare data for pipeline
    data = {
        'job_id': None,
        'video_path': str(video_path),
        'filename': video_path.name,
        'features': kwargs.get('features', {}),
        'metadata': metadata,
        'source': 'file_system',
        'timestamp': kwargs.get('timestamp')
    }
    
    logger.info(f"Loaded video data from file system: {video_path}")
    return data


def get_video_path_from_job(job, config) -> Path:
    """Get video file path from job information."""
    # For API jobs, the video should be in the temp directory
    temp_dir = Path(config.temp_dir)
    
    # Look for the video file in temp directory
    video_path = temp_dir / job.filename
    if video_path.exists():
        return video_path
    
    # If not found, check if it was moved to a processing directory
    processing_dir = temp_dir / "processing"
    if processing_dir.exists():
        for video_file in processing_dir.glob(f"*{Path(job.filename).suffix}"):
            if job.filename in video_file.name:
                return video_file
    
    raise FileNotFoundError(f"Video file not found for job {job.job_id}")


def extract_video_metadata(video_path: Path) -> Dict[str, Any]:
    """Extract metadata from video file."""
    try:
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        
        metadata = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            'file_size': video_path.stat().st_size,
            'format': video_path.suffix.lower()
        }
        
        cap.release()
        
        return metadata
        
    except Exception as e:
        logger.warning(f"Could not extract video metadata: {e}")
        return {
            'fps': 30.0,
            'frame_count': 0,
            'width': 1920,
            'height': 1080,
            'duration': 0,
            'file_size': video_path.stat().st_size,
            'format': video_path.suffix.lower()
        }


@test
def test_load_video_data():
    """Test the data loader with sample data."""
    # Test with file system source
    test_data = load_from_file_system({
        'video_path': 'test_video.mp4',
        'features': {'passes': True, 'possession': True}
    }, get_config())
    
    assert 'video_path' in test_data
    assert 'metadata' in test_data
    assert test_data['source'] == 'file_system'
    
    print("✅ Data loader test passed") 