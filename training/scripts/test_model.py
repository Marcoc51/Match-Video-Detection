#!/usr/bin/env python3
"""
Test script for YOLO model inference and visualization.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import List, Tuple, Optional

from src.utils.colors import GREEN, WHITE


def load_model(model_path: Path) -> torch.nn.Module:
    """Load YOLO model from path.
    
    Args:
        model_path: Path to the YOLO model file
        
    Returns:
        Loaded YOLO model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model using torch hub or local path
    if str(model_path).endswith('.pt'):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    return model


def detect_objects(model: torch.nn.Module, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[dict]:
    """Detect objects in a frame using YOLO model.
    
    Args:
        model: YOLO model
        frame: Input frame
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        List of detected objects with bbox, confidence, and class
    """
    # Run inference
    results = model(frame)
    
    detections = []
    
    # Process results
    if results.pred[0] is not None:
        for *xyxy, conf, cls in results.pred[0]:
            if conf >= confidence_threshold:
                detection = {
                    'bbox': [int(x) for x in xyxy],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': results.names[int(cls)]
                }
                detections.append(detection)
    
    return detections


def draw_detections(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
    """Draw detection bounding boxes on frame.
    
    Args:
        frame: Input frame
        detections: List of detections
        
    Returns:
        Frame with detections drawn
    """
    result_frame = frame.copy()
    
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), GREEN, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Draw label background
        cv2.rectangle(
            result_frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            GREEN,
            -1
        )
        
        # Draw label text
        cv2.putText(
            result_frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            2
        )
    
    return result_frame


def process_video(
    model_path: Path,
    video_path: Path,
    output_path: Path,
    confidence_threshold: float = 0.5,
    max_frames: Optional[int] = None
) -> None:
    """Process video with YOLO model and save results.
    
    Args:
        model_path: Path to YOLO model
        video_path: Path to input video
        output_path: Path to save output video
        confidence_threshold: Minimum confidence threshold
        max_frames: Maximum number of frames to process (None for all)
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    processed_frames = 0
    
    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we've reached max frames
        if max_frames is not None and processed_frames >= max_frames:
            break
        
        # Detect objects
        detections = detect_objects(model, frame, confidence_threshold)
        
        # Draw detections
        result_frame = draw_detections(frame, detections)
        
        # Write frame
        out.write(result_frame)
        
        frame_count += 1
        processed_frames += 1
        
        # Print progress
        if frame_count % 30 == 0:  # Every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Processing complete. Output saved to: {output_path}")
    print(f"Processed {processed_frames} frames")


def process_image(
    model_path: Path,
    image_path: Path,
    output_path: Path,
    confidence_threshold: float = 0.5
) -> None:
    """Process single image with YOLO model.
    
    Args:
        model_path: Path to YOLO model
        image_path: Path to input image
        output_path: Path to save output image
        confidence_threshold: Minimum confidence threshold
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Detect objects
    detections = detect_objects(model, image, confidence_threshold)
    
    # Draw detections
    result_image = draw_detections(image, detections)
    
    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_image)
    
    print(f"Processing complete. Output saved to: {output_path}")
    print(f"Found {len(detections)} objects")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test YOLO model on video or image")
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLO model")
    parser.add_argument("--input", type=Path, required=True, help="Path to input video or image")
    parser.add_argument("--output", type=Path, required=True, help="Path to output file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process (video only)")
    
    args = parser.parse_args()
    
    # Check if input is video or image
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    input_ext = args.input.suffix.lower()
    
    if input_ext in video_extensions:
        print("Processing video...")
        process_video(
            args.model,
            args.input,
            args.output,
            args.confidence,
            args.max_frames
        )
    elif input_ext in image_extensions:
        print("Processing image...")
        process_image(
            args.model,
            args.input,
            args.output,
            args.confidence
        )
    else:
        raise ValueError(f"Unsupported file format: {input_ext}")


if __name__ == "__main__":
    main() 