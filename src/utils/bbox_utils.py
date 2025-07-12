"""Bounding box utility functions for video analysis."""

import numpy as np
from typing import List, Tuple, Union


def get_center_of_bbox(bbox: List[float]) -> Tuple[float, float]:
    """Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Center point as (x, y)
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must have 4 coordinates [x1, y1, x2, y2]")
    
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def get_bbox_width(bbox: List[float]) -> float:
    """Calculate the width of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Width of the bounding box
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must have 4 coordinates [x1, y1, x2, y2]")
    
    return bbox[2] - bbox[0]


def get_bbox_height(bbox: List[float]) -> float:
    """Calculate the height of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Height of the bounding box
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must have 4 coordinates [x1, y1, x2, y2]")
    
    return bbox[3] - bbox[1]


def get_bbox_area(bbox: List[float]) -> float:
    """Calculate the area of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Area of the bounding box
    """
    return get_bbox_width(bbox) * get_bbox_height(bbox)


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box as [x1, y1, x2, y2]
        bbox2: Second bounding box as [x1, y1, x2, y2]
        
    Returns:
        IoU value between 0 and 1
    """
    if len(bbox1) != 4 or len(bbox2) != 4:
        raise ValueError("Bounding boxes must have 4 coordinates [x1, y1, x2, y2]")
    
    # Calculate intersection
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    # Check if there is intersection
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union
    area1 = get_bbox_area(bbox1)
    area2 = get_bbox_area(bbox2)
    union_area = area1 + area2 - intersection_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        point1: First point as (x, y)
        point2: Second point as (x, y)
        
    Returns:
        Euclidean distance between the points
    """
    if len(point1) != 2 or len(point2) != 2:
        raise ValueError("Points must have 2 coordinates (x, y)")
    
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def measure_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Alias for calculate_distance for backward compatibility."""
    return calculate_distance(p1, p2)


def measure_xy_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate x and y distance components between two points.
    
    Args:
        p1: First point as (x, y)
        p2: Second point as (x, y)
        
    Returns:
        Tuple of (x_distance, y_distance)
    """
    if len(p1) != 2 or len(p2) != 2:
        raise ValueError("Points must have 2 coordinates (x, y)")
    
    return p1[0] - p2[0], p1[1] - p2[1]


def get_foot_position(bbox: List[float]) -> Tuple[float, float]:
    """Calculate the foot position (bottom center) of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Foot position as (x, y)
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must have 4 coordinates [x1, y1, x2, y2]")
    
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, y2
