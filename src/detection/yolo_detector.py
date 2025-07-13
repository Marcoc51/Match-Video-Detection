"""
YOLO-based object detector for football match analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO

from ..core.config import Config
from .detection_result import DetectionResult


class YOLODetector:
    """
    YOLO-based object detector for football match analysis.
    
    This class provides a high-level interface for object detection
    using YOLO models, with support for batch processing, confidence
    filtering, and result formatting.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        device: str = "auto"
    ):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to the YOLO model file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ("cpu", "cuda", or "auto")
        """
        # Use config defaults if not provided
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
        self.iou_threshold = iou_threshold or Config.IOU_THRESHOLD
        self.device = device
        
        # Load model
        if model_path is None:
            model_path = Config.YOLO_MODEL_PATH
        
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        
        self.model_path = model_path
        self.model = YOLO(str(model_path))
        
        # Set model parameters
        self.model.conf = self.confidence_threshold
        self.model.iou = self.iou_threshold
        # Device is set automatically in newer ultralytics versions
        # self.model.device = self.device  # This line is removed for compatibility
        
        print(f"YOLO detector initialized with model: {model_path}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"Device: {self.device}")
    
    def detect_frame(
        self, 
        frame: np.ndarray,
        save_results: bool = False,
        verbose: bool = False
    ) -> DetectionResult:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame as numpy array
            save_results: Whether to save detection results
            verbose: Whether to print verbose output
            
        Returns:
            DetectionResult object containing detections
        """
        try:
            # Run inference
            results = self.model.predict(
                frame, 
                save=save_results,
                verbose=verbose,
                conf=self.confidence_threshold,
                iou=self.iou_threshold
            )
            
            # Process results
            detection_result = DetectionResult.from_yolo_result(results[0], frame.shape)
            
            return detection_result
            
        except Exception as e:
            print(f"Error during frame detection: {e}")
            # Return empty detection result
            return DetectionResult([], frame.shape)
    
    def detect_frames(
        self, 
        frames: List[np.ndarray],
        batch_size: int = 1,
        save_results: bool = False,
        verbose: bool = False
    ) -> List[DetectionResult]:
        """
        Detect objects in multiple frames.
        
        Args:
            frames: List of input frames
            batch_size: Number of frames to process at once
            save_results: Whether to save detection results
            verbose: Whether to print verbose output
            
        Returns:
            List of DetectionResult objects
        """
        detection_results = []
        
        # Process frames in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            try:
                # Run batch inference
                results = self.model.predict(
                    batch_frames,
                    save=save_results,
                    verbose=verbose,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold
                )
                
                # Process each result
                for j, result in enumerate(results):
                    frame_idx = i + j
                    if frame_idx < len(frames):
                        detection_result = DetectionResult.from_yolo_result(
                            result, frames[frame_idx].shape
                        )
                        detection_results.append(detection_result)
                
            except Exception as e:
                print(f"Error during batch detection (frames {i}-{i+batch_size-1}): {e}")
                # Add empty results for failed batch
                for j in range(len(batch_frames)):
                    frame_idx = i + j
                    if frame_idx < len(frames):
                        detection_results.append(
                            DetectionResult([], frames[frame_idx].shape)
                        )
        
        return detection_results
    
    def detect_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        save_results: bool = True,
        verbose: bool = False
    ) -> List[DetectionResult]:
        """
        Detect objects in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            save_results: Whether to save detection results
            verbose: Whether to print verbose output
            
        Returns:
            List of DetectionResult objects for each frame
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        detection_results = []
        
        try:
            # Read all frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            # Detect objects in all frames
            detection_results = self.detect_frames(
                frames, 
                batch_size=1,  # Process one frame at a time for videos
                save_results=save_results,
                verbose=verbose
            )
            
            # Save annotated video if requested
            if output_path and len(detection_results) > 0:
                self._save_annotated_video(
                    frames, detection_results, output_path, video_path
                )
        
        finally:
            cap.release()
        
        return detection_results
    
    def _save_annotated_video(
        self,
        frames: List[np.ndarray],
        detection_results: List[DetectionResult],
        output_path: Path,
        original_video_path: Path
    ) -> None:
        """Save video with detection annotations."""
        if len(frames) == 0 or len(detection_results) == 0:
            print("No frames or detection results to save")
            return
        
        # Get video properties from original video
        cap = cv2.VideoCapture(str(original_video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        try:
            # Write annotated frames
            for i, (frame, detection_result) in enumerate(zip(frames, detection_results)):
                annotated_frame = detection_result.draw_detections(frame)
                out.write(annotated_frame)
                
                if i % 100 == 0:
                    print(f"Processed frame {i}/{len(frames)}")
        
        finally:
            out.release()
        
        print(f"Annotated video saved to: {output_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "model_type": "YOLO",
            "classes": self.model.names if hasattr(self.model, 'names') else {}
        }
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """Update the confidence threshold for detections."""
        self.confidence_threshold = new_threshold
        self.model.conf = new_threshold
        print(f"Confidence threshold updated to: {new_threshold}")
    
    def update_iou_threshold(self, new_threshold: float) -> None:
        """Update the IoU threshold for NMS."""
        self.iou_threshold = new_threshold
        self.model.iou = new_threshold
        print(f"IoU threshold updated to: {new_threshold}")


# Convenience function for backward compatibility
def detect_objects_in_frame(frame: np.ndarray) -> DetectionResult:
    """
    Convenience function for detecting objects in a single frame.
    
    Args:
        frame: Input frame as numpy array
        
    Returns:
        DetectionResult object containing detections
    """
    detector = YOLODetector()
    return detector.detect_frame(frame) 