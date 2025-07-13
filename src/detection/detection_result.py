"""
Detection result classes and utilities for football video analysis.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.utils.colors import RED, BLUE, GREEN, YELLOW, WHITE


class Detection:
    """Represents a single detection result."""
    
    def __init__(
        self,
        bbox: List[float],
        confidence: float,
        class_id: int,
        class_name: str
    ):
        """Initialize a detection.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence score
            class_id: Class ID
            class_name: Class name
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the detection."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get the area of the detection."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def __str__(self) -> str:
        """String representation of the detection."""
        return f"{self.class_name} (conf: {self.confidence:.2f}) at {self.bbox}"


class DetectionResult:
    """Container for detection results from a single frame."""
    
    def __init__(self, detections: List[Detection], frame: Optional[np.ndarray] = None):
        """Initialize detection result.
        
        Args:
            detections: List of detections
            frame: Original frame (optional)
        """
        self.detections = detections
        self.frame = frame
    
    def get_detections_by_class(self, class_name: str) -> List[Detection]:
        """Get all detections of a specific class.
        
        Args:
            class_name: Name of the class to filter by
            
        Returns:
            List of detections of the specified class
        """
        return [d for d in self.detections if d.class_name == class_name]
    
    def get_detections_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Get all detections above a confidence threshold.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detections above the threshold
        """
        return [d for d in self.detections if d.confidence >= min_confidence]
    
    def get_highest_confidence_detection(self, class_name: str) -> Optional[Detection]:
        """Get the detection with highest confidence for a specific class.
        
        Args:
            class_name: Name of the class
            
        Returns:
            Detection with highest confidence or None if not found
        """
        class_detections = self.get_detections_by_class(class_name)
        if not class_detections:
            return None
        return max(class_detections, key=lambda d: d.confidence)
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """Draw detections on a frame.
        
        Args:
            frame: Frame to draw on
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Frame with detections drawn
        """
        result_frame = frame.copy()
        
        # Color mapping for different classes
        colors = {
            "player": RED,      # Red
            "ball": GREEN,        # Green
            "referee": BLUE,     # Blue
            "goalkeeper": YELLOW, # Yellow
            "default": WHITE   # White
        }
        
        for detection in self.detections:
            # Get color for this class
            color = colors.get(detection.class_name, colors.get("default", WHITE))
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            label_parts = []
            if show_labels:
                label_parts.append(detection.class_name)
            if show_confidence:
                label_parts.append(f"{detection.confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # Calculate text position
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 > 20 else y1 + text_size[1] + 10
                
                # Draw text background
                cv2.rectangle(
                    result_frame,
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0], text_y + 5),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result_frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    WHITE,
                    2
                )
        
        return result_frame
    
    def save_annotations(self, output_path: Path) -> None:
        """Save detection annotations to a file.
        
        Args:
            output_path: Path to save annotations
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            for detection in self.detections:
                x1, y1, x2, y2 = detection.bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # YOLO format: class_id center_x center_y width height
                f.write(f"{detection.class_id} {center_x} {center_y} {width} {height}\n")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection result to dictionary.
        
        Returns:
            Dictionary representation of the detection result
        """
        return {
            "detections": [
                {
                    "bbox": d.bbox,
                    "confidence": d.confidence,
                    "class_id": d.class_id,
                    "class_name": d.class_name
                }
                for d in self.detections
            ],
            "frame_shape": self.frame.shape if self.frame is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectionResult':
        """Create detection result from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            DetectionResult instance
        """
        detections = [
            Detection(
                bbox=d["bbox"],
                confidence=d["confidence"],
                class_id=d["class_id"],
                class_name=d["class_name"]
            )
            for d in data["detections"]
        ]
        return cls(detections)
    
    @classmethod
    def from_yolo_result(cls, yolo_result, frame_shape: Tuple[int, int, int]) -> 'DetectionResult':
        """Create detection result from YOLO result.
        
        Args:
            yolo_result: YOLO detection result
            frame_shape: Shape of the original frame (height, width, channels)
            
        Returns:
            DetectionResult instance
        """
        detections = []
        
        if hasattr(yolo_result, 'boxes') and yolo_result.boxes is not None:
            boxes = yolo_result.boxes
            
            # Get coordinates, confidence, and class information
            if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                bboxes = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else np.ones(len(bboxes))
                class_ids = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else np.zeros(len(bboxes))
                
                # Class name mapping (adjust based on your model's classes)
                class_names = {
                    0: "player",
                    1: "ball", 
                    2: "referee",
                    3: "goalkeeper"
                }
                
                for i, (bbox, conf, class_id) in enumerate(zip(bboxes, confidences, class_ids)):
                    class_name = class_names.get(int(class_id), f"class_{int(class_id)}")
                    
                    detection = Detection(
                        bbox=bbox.tolist(),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=class_name
                    )
                    detections.append(detection)
        
        return cls(detections)
    
    def __len__(self) -> int:
        """Get number of detections."""
        return len(self.detections)
    
    def __getitem__(self, index: int) -> Detection:
        """Get detection by index."""
        return self.detections[index]
    
    def __iter__(self):
        """Iterate over detections."""
        return iter(self.detections) 