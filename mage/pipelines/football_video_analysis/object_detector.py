"""
Object Detector Block for Football Video Analysis Pipeline
Handles player and ball detection using YOLO models
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager
from src.detection.yolo_detector import YOLODetector

logger = logging.getLogger(__name__)


@transformer
def detect_objects(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Detect players and ball in video frames.
    
    This block handles:
    - Player detection using YOLO
    - Ball detection using YOLO
    - Confidence filtering
    - Bounding box extraction
    - Detection quality assessment
    
    Args:
        data: Input data containing video frames
        
    Returns:
        Dict containing detection results
    """
    try:
        frames = data['frames']
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=35, message="Detecting objects...")
        
        logger.info(f"Starting object detection on {len(frames)} frames")
        
        # Initialize detector
        detector = initialize_detector()
        
        # Detect objects in frames
        detections = detect_in_frames(frames, detector, data.get('features', {}))
        
        # Quality check
        quality_metrics = assess_detection_quality(detections)
        
        # Combine results
        result = {
            **data,
            'detections': detections,
            'quality_metrics': quality_metrics,
            'detection_config': {
                'model_path': detector.model_path,
                'confidence_threshold': detector.confidence_threshold,
                'iou_threshold': detector.iou_threshold
            }
        }
        
        logger.info(f"Object detection completed: {len(detections)} frame detections")
        return result
        
    except Exception as e:
        logger.error(f"Error in object detector: {e}")
        raise


def initialize_detector() -> YOLODetector:
    """Initialize the YOLO detector."""
    config = get_config()
    
    # Initialize detector with configuration
    detector = YOLODetector(
        model_path=config.model_path_absolute,
        confidence_threshold=config.confidence_threshold,
        iou_threshold=config.iou_threshold
    )
    
    logger.info(f"Initialized detector with model: {config.model_path_absolute}")
    return detector


def detect_in_frames(frames: List[np.ndarray], detector: YOLODetector, features: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect objects in video frames."""
    detections = []
    
    for i, frame in enumerate(frames):
        try:
            # Detect objects in frame
            frame_detections = detector.detect(frame)
            
            # Filter detections based on features
            filtered_detections = filter_detections(frame_detections, features)
            
            # Add frame information
            frame_result = {
                'frame_index': i,
                'detections': filtered_detections,
                'timestamp': i / 30.0,  # Assuming 30 FPS
                'detection_count': len(filtered_detections)
            }
            
            detections.append(frame_result)
            
            # Progress logging
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(frames)} frames")
                
        except Exception as e:
            logger.warning(f"Error detecting in frame {i}: {e}")
            # Add empty detection for failed frame
            detections.append({
                'frame_index': i,
                'detections': [],
                'timestamp': i / 30.0,
                'detection_count': 0,
                'error': str(e)
            })
    
    return detections


def filter_detections(frame_detections: List[Dict], features: Dict[str, Any]) -> List[Dict]:
    """Filter detections based on enabled features."""
    if not frame_detections:
        return []
    
    filtered = []
    
    for detection in frame_detections:
        class_name = detection.get('class_name', '').lower()
        
        # Always include players and ball
        if 'player' in class_name or 'ball' in class_name:
            filtered.append(detection)
        
        # Include referee if available
        elif 'referee' in class_name:
            filtered.append(detection)
    
    return filtered


def assess_detection_quality(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess the quality of detections."""
    if not detections:
        return {
            'total_frames': 0,
            'frames_with_detections': 0,
            'average_detections_per_frame': 0,
            'detection_consistency': 0,
            'quality_score': 0
        }
    
    total_frames = len(detections)
    frames_with_detections = sum(1 for d in detections if d['detection_count'] > 0)
    total_detections = sum(d['detection_count'] for d in detections)
    average_detections = total_detections / total_frames if total_frames > 0 else 0
    
    # Calculate detection consistency
    detection_counts = [d['detection_count'] for d in detections]
    consistency = 1 - (np.std(detection_counts) / (np.mean(detection_counts) + 1e-6))
    consistency = max(0, min(1, consistency))  # Clamp between 0 and 1
    
    # Calculate overall quality score
    coverage_ratio = frames_with_detections / total_frames
    quality_score = (coverage_ratio * 0.4 + consistency * 0.3 + min(average_detections / 10, 1) * 0.3)
    
    return {
        'total_frames': total_frames,
        'frames_with_detections': frames_with_detections,
        'total_detections': total_detections,
        'average_detections_per_frame': average_detections,
        'detection_consistency': consistency,
        'coverage_ratio': coverage_ratio,
        'quality_score': quality_score,
        'quality_level': get_quality_level(quality_score)
    }


def get_quality_level(quality_score: float) -> str:
    """Get quality level based on score."""
    if quality_score >= 0.8:
        return 'excellent'
    elif quality_score >= 0.6:
        return 'good'
    elif quality_score >= 0.4:
        return 'fair'
    else:
        return 'poor'


@test
def test_object_detector():
    """Test the object detector with sample data."""
    # Create sample detection data
    sample_detections = [
        {
            'frame_index': 0,
            'detections': [
                {'class_name': 'player', 'confidence': 0.9, 'bbox': [100, 100, 200, 300]},
                {'class_name': 'ball', 'confidence': 0.8, 'bbox': [150, 150, 170, 170]}
            ],
            'detection_count': 2
        },
        {
            'frame_index': 1,
            'detections': [
                {'class_name': 'player', 'confidence': 0.85, 'bbox': [110, 110, 210, 310]}
            ],
            'detection_count': 1
        }
    ]
    
    # Test quality assessment
    quality = assess_detection_quality(sample_detections)
    
    assert quality['total_frames'] == 2
    assert quality['frames_with_detections'] == 2
    assert quality['total_detections'] == 3
    assert 'quality_score' in quality
    assert 'quality_level' in quality
    
    print("✅ Object detector test passed") 