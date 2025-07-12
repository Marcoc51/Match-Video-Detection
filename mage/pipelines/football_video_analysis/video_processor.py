"""
Video Processor Block for Football Video Analysis Pipeline
Handles video preprocessing, frame extraction, and quality optimization
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager

logger = logging.getLogger(__name__)


@transformer
def process_video(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Process video for analysis.
    
    This block handles:
    - Video quality optimization
    - Frame rate standardization
    - Resolution adjustment
    - Frame extraction for analysis
    - Video format conversion
    
    Args:
        data: Input data containing video information
        
    Returns:
        Dict containing processed video data and frames
    """
    try:
        video_path = data['video_path']
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=25, message="Processing video...")
        
        logger.info(f"Processing video: {video_path}")
        
        # Process the video
        processed_data = preprocess_video(video_path, data['metadata'])
        
        # Extract frames for analysis
        frames_data = extract_frames(video_path, processed_data)
        
        # Combine results
        result = {
            **data,
            'processed_video_path': processed_data['processed_path'],
            'frames': frames_data['frames'],
            'frame_indices': frames_data['indices'],
            'processing_info': processed_data['info']
        }
        
        logger.info(f"Video processing completed: {len(frames_data['frames'])} frames extracted")
        return result
        
    except Exception as e:
        logger.error(f"Error in video processor: {e}")
        raise


def preprocess_video(video_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Preprocess video for optimal analysis."""
    video_path = Path(video_path)
    
    # Create temporary directory for processed video
    temp_dir = Path(tempfile.mkdtemp())
    processed_path = temp_dir / f"processed_{video_path.name}"
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine target parameters
    target_fps = 30.0  # Standardize to 30 FPS
    target_width = 1920
    target_height = 1080
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(processed_path), fourcc, target_fps, (target_width, target_height))
    
    frame_skip = max(1, int(fps / target_fps))
    processed_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to match target FPS
            if processed_frames % frame_skip == 0:
                # Resize frame
                frame_resized = cv2.resize(frame, (target_width, target_height))
                
                # Apply basic enhancement
                frame_enhanced = enhance_frame(frame_resized)
                
                # Write frame
                out.write(frame_enhanced)
            
            processed_frames += 1
            
            # Progress logging
            if processed_frames % 100 == 0:
                logger.info(f"Processed {processed_frames}/{frame_count} frames")
    
    finally:
        cap.release()
        out.release()
    
    # Get processed video info
    processed_info = {
        'original_fps': fps,
        'target_fps': target_fps,
        'original_resolution': f"{width}x{height}",
        'target_resolution': f"{target_width}x{target_height}",
        'frame_skip': frame_skip,
        'processed_frames': processed_frames,
        'processing_applied': ['resize', 'fps_standardization', 'enhancement']
    }
    
    return {
        'processed_path': str(processed_path),
        'info': processed_info
    }


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Apply basic frame enhancement."""
    # Convert to LAB color space for better enhancement
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply slight sharpening
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    return enhanced


def extract_frames(video_path: str, processed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract frames for analysis."""
    video_path = Path(processed_data['processed_path'])
    
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    indices = []
    
    # Extract every 10th frame for analysis (3 FPS equivalent)
    frame_interval = 10
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
                indices.append(frame_count)
            
            frame_count += 1
            
            # Limit to first 1000 frames for memory management
            if len(frames) >= 1000:
                break
    
    finally:
        cap.release()
    
    logger.info(f"Extracted {len(frames)} frames for analysis")
    
    return {
        'frames': frames,
        'indices': indices,
        'frame_interval': frame_interval
    }


@test
def test_video_processor():
    """Test the video processor with sample data."""
    # Create sample data
    sample_data = {
        'video_path': 'test_video.mp4',
        'metadata': {
            'fps': 30.0,
            'frame_count': 100,
            'width': 1920,
            'height': 1080
        }
    }
    
    # Test frame enhancement
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    enhanced_frame = enhance_frame(test_frame)
    
    assert enhanced_frame.shape == test_frame.shape
    assert enhanced_frame.dtype == test_frame.dtype
    
    print("✅ Video processor test passed") 