"""
Tracking utility functions for football video analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from src.utils.colors import BLACK, WHITE, RED, GREEN, YELLOW


def draw_tracking_visualization(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    show_ids: bool = True,
    show_trajectories: bool = True,
    trajectory_length: int = 30
) -> np.ndarray:
    """Draw tracking visualization on a frame.
    
    Args:
        frame: Input frame
        tracks: List of track dictionaries
        show_ids: Whether to show track IDs
        show_trajectories: Whether to show track trajectories
        trajectory_length: Length of trajectory to show
        
    Returns:
        Frame with tracking visualization
    """
    result_frame = frame.copy()
    
    for track in tracks:
        bbox = track.get('bbox', [])
        track_id = track.get('track_id', 0)
        class_name = track.get('class_name', 'unknown')
        confidence = track.get('confidence', 0.0)
        trajectory = track.get('trajectory', [])
        
        if not bbox or len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on class
        if class_name == 'player':
            color = RED
        elif class_name == 'ball':
            color = GREEN
        elif class_name == 'referee':
            color = YELLOW
        else:
            color = WHITE
        
        # Draw bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw trajectory
        if show_trajectories and trajectory:
            trajectory_points = trajectory[-trajectory_length:]
            if len(trajectory_points) > 1:
                points = np.array(trajectory_points, dtype=np.int32)
                cv2.polylines(result_frame, [points], False, color, 2)
        
        # Draw track ID
        if show_ids:
            label = f"ID: {track_id}"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            
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


def draw_triangle_marker(
    frame: np.ndarray,
    center: Tuple[int, int],
    size: int = 20,
    color: Tuple[int, int, int] = RED,
    direction: str = "up"
) -> np.ndarray:
    """Draw a triangle marker on the frame.
    
    Args:
        frame: Input frame
        center: Center point of the triangle
        size: Size of the triangle
        color: Color of the triangle
        direction: Direction of the triangle ("up", "down", "left", "right")
        
    Returns:
        Frame with triangle marker
    """
    cx, cy = center
    
    if direction == "up":
        triangle_points = np.array([
            [cx, cy - size],
            [cx - size//2, cy + size//2],
            [cx + size//2, cy + size//2]
        ], np.int32)
    elif direction == "down":
        triangle_points = np.array([
            [cx, cy + size],
            [cx - size//2, cy - size//2],
            [cx + size//2, cy - size//2]
        ], np.int32)
    elif direction == "left":
        triangle_points = np.array([
            [cx - size, cy],
            [cx + size//2, cy - size//2],
            [cx + size//2, cy + size//2]
        ], np.int32)
    elif direction == "right":
        triangle_points = np.array([
            [cx + size, cy],
            [cx - size//2, cy - size//2],
            [cx - size//2, cy + size//2]
        ], np.int32)
    else:
        return frame
    
    # Draw filled triangle
    cv2.fillPoly(frame, [triangle_points], color)
    
    # Draw black border
    cv2.drawContours(frame, [triangle_points], 0, BLACK, 2)
    
    return frame


def draw_overlay_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = WHITE,
    thickness: int = 2,
    background_color: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """Draw text with optional background overlay.
    
    Args:
        frame: Input frame
        text: Text to draw
        position: Position (x, y) for the text
        font_scale: Font scale
        color: Text color
        thickness: Text thickness
        background_color: Background color (optional)
        
    Returns:
        Frame with text overlay
    """
    result_frame = frame.copy()
    
    # Calculate text size
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    x, y = position
    
    # Draw background if specified
    if background_color:
        cv2.rectangle(
            result_frame,
            (x, y - text_size[1] - 5),
            (x + text_size[0], y + 5),
            background_color,
            -1
        )
    
    # Draw text
    cv2.putText(
        result_frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    return result_frame


def create_tracking_summary(
    tracks: List[Dict[str, Any]],
    frame_count: int,
    fps: float = 30.0
) -> Dict[str, Any]:
    """Create a summary of tracking results.
    
    Args:
        tracks: List of track dictionaries
        frame_count: Total number of frames
        fps: Frames per second
        
    Returns:
        Dictionary containing tracking summary
    """
    if not tracks:
        return {
            "total_tracks": 0,
            "total_frames": frame_count,
            "duration_seconds": frame_count / fps,
            "tracks_per_class": {},
            "average_confidence": 0.0
        }
    
    # Count tracks by class
    tracks_per_class = {}
    confidences = []
    
    for track in tracks:
        class_name = track.get('class_name', 'unknown')
        tracks_per_class[class_name] = tracks_per_class.get(class_name, 0) + 1
        
        confidence = track.get('confidence', 0.0)
        if confidence > 0:
            confidences.append(confidence)
    
    return {
        "total_tracks": len(tracks),
        "total_frames": frame_count,
        "duration_seconds": frame_count / fps,
        "tracks_per_class": tracks_per_class,
        "average_confidence": np.mean(confidences) if confidences else 0.0
    }


def save_tracking_results(
    tracks: List[Dict[str, Any]],
    output_path: Path,
    format: str = "json"
) -> None:
    """Save tracking results to file.
    
    Args:
        tracks: List of track dictionaries
        output_path: Path to save results
        format: Output format ("json" or "csv")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        import json
        with open(output_path, 'w') as f:
            json.dump(tracks, f, indent=2)
    elif format.lower() == "csv":
        import pandas as pd
        
        # Flatten track data for CSV
        csv_data = []
        for track in tracks:
            row = {
                'track_id': track.get('track_id', 0),
                'class_name': track.get('class_name', 'unknown'),
                'confidence': track.get('confidence', 0.0),
                'bbox_x1': track.get('bbox', [0, 0, 0, 0])[0],
                'bbox_y1': track.get('bbox', [0, 0, 0, 0])[1],
                'bbox_x2': track.get('bbox', [0, 0, 0, 0])[2],
                'bbox_y2': track.get('bbox', [0, 0, 0, 0])[3],
                'frame': track.get('frame', 0)
            }
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_tracking_results(input_path: Path, format: str = "json") -> List[Dict[str, Any]]:
    """Load tracking results from file.
    
    Args:
        input_path: Path to load results from
        format: Input format ("json" or "csv")
        
    Returns:
        List of track dictionaries
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Tracking results file not found: {input_path}")
    
    if format.lower() == "json":
        import json
        with open(input_path, 'r') as f:
            return json.load(f)
    elif format.lower() == "csv":
        import pandas as pd
        
        df = pd.read_csv(input_path)
        tracks = []
        
        for _, row in df.iterrows():
            track = {
                'track_id': int(row['track_id']),
                'class_name': row['class_name'],
                'confidence': float(row['confidence']),
                'bbox': [float(row['bbox_x1']), float(row['bbox_y1']), 
                        float(row['bbox_x2']), float(row['bbox_y2'])],
                'frame': int(row['frame'])
            }
            tracks.append(track)
        
        return tracks
    else:
        raise ValueError(f"Unsupported format: {format}") 