#!/usr/bin/env python3
"""
Visualization script for box crosses detection results.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from src.utils.colors import RED, GREEN, BLUE, YELLOW, WHITE


def load_detections(detection_file: Path) -> List[dict]:
    """Load detection results from file.
    
    Args:
        detection_file: Path to detection results file
        
    Returns:
        List of detections
    """
    detections = []
    
    if detection_file.suffix == '.txt':
        # YOLO format: class_id center_x center_y width height
        with open(detection_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert to bbox format [x1, y1, x2, y2]
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'confidence': 1.0  # Default confidence
                    })
    
    return detections


def draw_detection_box(frame: np.ndarray, bbox: List[float], color: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    """Draw a detection bounding box on frame.
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        color: BGR color tuple
        thickness: Line thickness
        
    Returns:
        Frame with bounding box drawn
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_ball_position(frame: np.ndarray, ball_bbox: List[float]) -> np.ndarray:
    """Draw ball position indicator.
    
    Args:
        frame: Input frame
        ball_bbox: Ball bounding box [x1, y1, x2, y2]
        
    Returns:
        Frame with ball position drawn
    """
    if ball_bbox:
        ball_center_x = int((ball_bbox[0] + ball_bbox[2]) / 2)
        ball_center_y = int((ball_bbox[1] + ball_bbox[3]) / 2)
        cv2.circle(frame, (ball_center_x, ball_center_y), 10, YELLOW, -1)
    
    return frame


def draw_statistics(frame: np.ndarray, stats: dict) -> np.ndarray:
    """Draw statistics overlay on frame.
    
    Args:
        frame: Input frame
        stats: Dictionary containing statistics
        
    Returns:
        Frame with statistics overlay
    """
    y_offset = 30
    
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(
            frame, 
            text, 
            (10, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            WHITE, 
            2
        )
        y_offset += 30
    
    return frame


def draw_cross_indicator(frame: np.ndarray, is_box_cross: bool, position: Tuple[int, int]) -> np.ndarray:
    """Draw cross indicator.
    
    Args:
        frame: Input frame
        is_box_cross: Whether this is a box cross
        position: Position to draw indicator
        
    Returns:
        Frame with cross indicator
    """
    x, y = position
    color = GREEN if is_box_cross else RED
    cv2.circle(frame, (x, y), 15, color, 2)
    
    # Add text
    text = "CROSS" if is_box_cross else "NO CROSS"
    cv2.putText(
        frame, 
        text, 
        (10, 180), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        WHITE, 
        2
    )
    
    return frame


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Draw color legend.
    
    Args:
        frame: Input frame
        
    Returns:
        Frame with legend
    """
    legend_items = [
        ("Player", RED),
        ("Ball", GREEN),
        ("Referee", BLUE),
        ("Cross", YELLOW)
    ]
    
    y_offset = 220
    
    for item_name, color in legend_items:
        # Draw color box
        cv2.rectangle(frame, (10, y_offset - 15), (30, y_offset + 5), color, -1)
        
        # Draw text
        cv2.putText(
            frame, 
            item_name, 
            (40, y_offset), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            WHITE, 
            2
        )
        y_offset += 30
    
    return frame


def visualize_detections(
    video_path: Path,
    detection_files: List[Path],
    output_path: Path,
    show_ball: bool = True,
    show_stats: bool = True
) -> None:
    """Visualize detection results on video.
    
    Args:
        video_path: Path to input video
        detection_files: List of detection result files
        output_path: Path to save output video
        show_ball: Whether to show ball position
        show_stats: Whether to show statistics
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Load all detection files
    all_detections = {}
    for detection_file in detection_files:
        frame_num = int(detection_file.stem.split('_')[-1])  # Extract frame number
        all_detections[frame_num] = load_detections(detection_file)
    
    # Create video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    total_crosses = 0
    
    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame
        frame_detections = all_detections.get(frame_count, [])
        
        # Draw detections
        for detection in frame_detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            
            # Choose color based on class
            if class_id == 0:  # Player
                color = RED
            elif class_id == 1:  # Ball
                color = GREEN
            elif class_id == 2:  # Referee
                color = BLUE
            else:  # Cross
                color = YELLOW
                total_crosses += 1
            
            # Draw bounding box
            frame = draw_detection_box(frame, bbox, color)
        
        # Show ball position if requested
        if show_ball:
            ball_detections = [d for d in frame_detections if d['class_id'] == 1]
            if ball_detections:
                frame = draw_ball_position(frame, ball_detections[0]['bbox'])
        
        # Show statistics if requested
        if show_stats:
            stats = {
                "Frame": frame_count,
                "Detections": len(frame_detections),
                "Total Crosses": total_crosses
            }
            frame = draw_statistics(frame, stats)
        
        # Draw legend
        frame = draw_legend(frame)
        
        # Write frame
        out.write(frame)
        
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:  # Every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Visualization complete. Output saved to: {output_path}")
    print(f"Total crosses detected: {total_crosses}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visualize box crosses detection results")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    parser.add_argument("--detections", type=Path, nargs='+', required=True, help="Detection result files")
    parser.add_argument("--output", type=Path, required=True, help="Path to save output video")
    parser.add_argument("--no-ball", action="store_true", help="Don't show ball position")
    parser.add_argument("--no-stats", action="store_true", help="Don't show statistics")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    # Check if detection files exist
    for detection_file in args.detections:
        if not detection_file.exists():
            raise FileNotFoundError(f"Detection file not found: {detection_file}")
    
    # Visualize detections
    visualize_detections(
        args.video,
        args.detections,
        args.output,
        show_ball=not args.no_ball,
        show_stats=not args.no_stats
    )


if __name__ == "__main__":
    main() 