"""Video utility functions for processing and manipulation."""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Union


def read_video(video_path: Union[str, Path]) -> List[np.ndarray]:
    """Read video file and return list of frames.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        List of video frames as numpy arrays
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file cannot be opened
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")
    
    return frames


def save_video(
    output_video_frames: List[np.ndarray], 
    output_video_path: Union[str, Path], 
    fps: float = 30.0
) -> None:
    """Save list of frames as video file.
    
    Args:
        output_video_frames: List of frames to save
        output_video_path: Path where to save the video
        fps: Frames per second for the output video
        
    Raises:
        ValueError: If no frames provided or invalid frame dimensions
    """
    if not output_video_frames:
        raise ValueError("No frames provided for video saving")
    
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get frame dimensions from first frame
    height, width = output_video_frames[0].shape[:2]
    
    # Determine codec based on file extension
    if output_video_path.suffix.lower() == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_video_path.suffix.lower() == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(
        str(output_video_path), 
        fourcc, 
        fps, 
        (width, height)
    )
    
    try:
        for frame in output_video_frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("All frames must have the same dimensions")
            out.write(frame)
    finally:
        out.release()


def draw_bezier_pass(
    frame: np.ndarray, 
    start: Tuple[int, int], 
    end: Tuple[int, int], 
    color: Tuple[int, int, int], 
    thickness: int = 2, 
    curve_height: int = 60
) -> np.ndarray:
    """Draw a Bézier curve representing a pass on the frame.
    
    Args:
        frame: Input frame to draw on
        start: Starting point (x, y)
        end: Ending point (x, y)
        color: BGR color tuple
        thickness: Line thickness
        curve_height: Height of the curve
        
    Returns:
        Frame with the Bézier curve drawn on it
    """
    if frame is None:
        raise ValueError("Frame cannot be None")
    
    # Convert points to numpy arrays
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)
    
    # Calculate control point for the curve
    mid = (start + end) / 2
    
    # Perpendicular direction for the curve
    direction = end - start
    perp = np.array([-direction[1], direction[0]])
    norm = np.linalg.norm(perp)
    
    if norm > 0:
        perp = perp / norm
        control = mid + perp * curve_height
    else:
        # If start and end are the same, use a simple curve
        control = mid + np.array([0, curve_height])
    
    # Generate points along the Bézier curve
    points = []
    for t in np.linspace(0, 1, 30):
        point = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end
        points.append(point.astype(int))
    
    points = np.array(points).reshape((-1, 1, 2))
    
    # Draw the curve
    cv2.polylines(frame, [points], False, color, thickness)
    
    # Draw an arrowhead at the end
    if len(points) > 1:
        cv2.arrowedLine(
            frame, 
            tuple(points[-2][0]), 
            tuple(points[-1][0]), 
            color, 
            thickness, 
            tipLength=0.3
        )
    
    return frame