"""
Detection module for Match Video Detection.

This module contains all object detection functionality including
YOLO models, detection results processing, and detection utilities.
"""

from .yolo_detector import YOLODetector
from .detection_result import DetectionResult
from .detection_utils import DetectionUtils

__all__ = [
    'YOLODetector',
    'DetectionResult', 
    'DetectionUtils'
]
