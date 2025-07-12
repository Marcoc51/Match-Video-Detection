"""
Utility functions for detection processing and analysis.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .detection_result import DetectionResult, Detection
from ..core.config import Config


class DetectionUtils:
    """
    Utility class for detection processing and analysis.
    
    This class provides static methods for common detection tasks
    such as filtering, analysis, and visualization.
    """
    
    @staticmethod
    def calculate_iou(bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    @staticmethod
    def filter_overlapping_detections(
        detections: List[Detection],
        iou_threshold: float = 0.5,
        keep_highest_confidence: bool = True
    ) -> List[Detection]:
        """
        Filter out overlapping detections based on IoU threshold.
        
        Args:
            detections: List of Detection objects
            iou_threshold: IoU threshold for considering detections as overlapping
            keep_highest_confidence: Whether to keep the detection with highest confidence
            
        Returns:
            Filtered list of Detection objects
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        filtered_detections = []
        
        for detection in sorted_detections:
            is_overlapping = False
            
            for filtered_detection in filtered_detections:
                iou = DetectionUtils.calculate_iou(detection.bbox, filtered_detection.bbox)
                if iou > iou_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    @staticmethod
    def find_closest_detection(
        target_point: Tuple[float, float],
        detections: List[Detection],
        max_distance: Optional[float] = None
    ) -> Optional[Detection]:
        """
        Find the detection closest to a target point.
        
        Args:
            target_point: Target point (x, y)
            detections: List of Detection objects
            max_distance: Maximum allowed distance (optional)
            
        Returns:
            Closest Detection object, or None if no detection within max_distance
        """
        if not detections:
            return None
        
        closest_detection = None
        min_distance = float('inf')
        
        for detection in detections:
            distance = DetectionUtils.calculate_distance(target_point, detection.center)
            
            if distance < min_distance:
                if max_distance is None or distance <= max_distance:
                    min_distance = distance
                    closest_detection = detection
        
        return closest_detection
    
    @staticmethod
    def analyze_detection_trends(
        detection_results: List[DetectionResult],
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze detection trends across multiple frames.
        
        Args:
            detection_results: List of DetectionResult objects
            class_name: Specific class to analyze (optional)
            
        Returns:
            Dictionary containing trend analysis
        """
        if not detection_results:
            return {
                "total_frames": 0,
                "detection_counts": [],
                "average_detections_per_frame": 0,
                "detection_rate": 0.0
            }
        
        detection_counts = []
        frames_with_detections = 0
        
        for result in detection_results:
            if class_name:
                count = len(result.get_detections_by_class(class_name))
            else:
                count = len(result.detections)
            
            detection_counts.append(count)
            if count > 0:
                frames_with_detections += 1
        
        total_frames = len(detection_results)
        
        return {
            "total_frames": total_frames,
            "detection_counts": detection_counts,
            "average_detections_per_frame": np.mean(detection_counts),
            "detection_rate": frames_with_detections / total_frames if total_frames > 0 else 0.0,
            "max_detections": max(detection_counts),
            "min_detections": min(detection_counts),
            "std_detections": np.std(detection_counts)
        }
    
    @staticmethod
    def create_detection_heatmap(
        detection_results: List[DetectionResult],
        frame_shape: Tuple[int, int, int],
        class_name: Optional[str] = None,
        blur_radius: int = 15
    ) -> np.ndarray:
        """
        Create a heatmap showing where detections occur most frequently.
        
        Args:
            detection_results: List of DetectionResult objects
            frame_shape: Shape of the frames (height, width, channels)
            class_name: Specific class to include in heatmap (optional)
            blur_radius: Radius for Gaussian blur
            
        Returns:
            Heatmap as numpy array
        """
        height, width, _ = frame_shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for result in detection_results:
            detections = result.detections
            if class_name:
                detections = result.get_detections_by_class(class_name)
            
            for detection in detections:
                x1, y1, x2, y2 = map(int, detection.bbox)
                # Add detection area to heatmap
                heatmap[y1:y2, x1:x2] += 1
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Apply Gaussian blur for smooth visualization
        heatmap = cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)
        
        return heatmap
    
    @staticmethod
    def save_detection_results(
        detection_results: List[DetectionResult],
        output_path: Path,
        format: str = "json"
    ) -> None:
        """
        Save detection results to file.
        
        Args:
            detection_results: List of DetectionResult objects
            output_path: Path to save the results
            format: Output format ("json" or "csv")
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            import json
            data = [result.to_dict() for result in detection_results]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'
                ])
                
                for frame_idx, result in enumerate(detection_results):
                    for detection in result.detections:
                        x1, y1, x2, y2 = detection.bbox
                        writer.writerow([
                            frame_idx,
                            detection.class_name,
                            detection.confidence,
                            x1, y1, x2, y2
                        ])
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def load_detection_results(
        file_path: Path,
        format: str = "json"
    ) -> List[DetectionResult]:
        """
        Load detection results from file.
        
        Args:
            file_path: Path to the results file
            format: Input format ("json" or "csv")
            
        Returns:
            List of DetectionResult objects
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")
        
        if format.lower() == "json":
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            results = []
            for frame_data in data:
                detections = []
                for det_data in frame_data['detections']:
                    detection = Detection(
                        bbox=tuple(det_data['bbox']),
                        confidence=det_data['confidence'],
                        class_id=det_data['class_id'],
                        class_name=det_data['class_name'],
                        center=tuple(det_data['center']) if det_data['center'] else None
                    )
                    detections.append(detection)
                
                result = DetectionResult(detections, tuple(frame_data['frame_shape']))
                results.append(result)
            
            return results
        
        elif format.lower() == "csv":
            import csv
            results = []
            current_frame = 0
            current_detections = []
            frame_shape = None
            
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_idx = int(row['frame'])
                    
                    # If we've moved to a new frame, save the previous one
                    if frame_idx != current_frame and current_detections:
                        if frame_shape is None:
                            # Estimate frame shape (you might want to store this in the CSV)
                            frame_shape = (1080, 1920, 3)
                        result = DetectionResult(current_detections, frame_shape)
                        results.append(result)
                        current_detections = []
                        current_frame = frame_idx
                    
                    # Create detection object
                    detection = Detection(
                        bbox=(float(row['x1']), float(row['y1']), 
                              float(row['x2']), float(row['y2'])),
                        confidence=float(row['confidence']),
                        class_id=0,  # Not stored in CSV
                        class_name=row['class_name']
                    )
                    current_detections.append(detection)
            
            # Add the last frame
            if current_detections:
                if frame_shape is None:
                    frame_shape = (1080, 1920, 3)
                result = DetectionResult(current_detections, frame_shape)
                results.append(result)
            
            return results
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def validate_detection_result(result: DetectionResult) -> bool:
        """
        Validate a detection result for consistency.
        
        Args:
            result: DetectionResult object to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check frame shape
            if len(result.frame_shape) != 3:
                return False
            
            # Check detections
            for detection in result.detections:
                # Check bbox format
                if len(detection.bbox) != 4:
                    return False
                
                x1, y1, x2, y2 = detection.bbox
                if x1 >= x2 or y1 >= y2:
                    return False
                
                # Check confidence
                if not (0 <= detection.confidence <= 1):
                    return False
                
                # Check class_id
                if detection.class_id < 0:
                    return False
            
            return True
        
        except Exception:
            return False 