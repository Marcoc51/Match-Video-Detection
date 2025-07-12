"""Speed and distance estimation for football video analysis."""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.utils.colors import BLACK


class SpeedAndDistanceEstimator:
    """Estimates speed and distance for tracked objects."""
    
    def __init__(self, fps: float = 30.0):
        """Initialize the speed and distance estimator.
        
        Args:
            fps: Frames per second of the video
        """
        self.fps = fps
        self.time_per_frame = 1.0 / fps
    
    def add_speed_and_distance_to_tracks(self, tracks: Dict[str, List[Dict]]) -> None:
        """Add speed and distance calculations to tracks.
        
        Args:
            tracks: Dictionary containing object tracks
        """
        for object_type, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_data in frame_tracks.items():
                    if 'position' in track_data:
                        # Calculate speed if we have previous frame data
                        if frame_num > 0 and track_id in tracks[object_type][frame_num - 1]:
                            prev_track = tracks[object_type][frame_num - 1][track_id]
                            if 'position' in prev_track:
                                prev_pos = prev_track['position']
                                curr_pos = track_data['position']
                                
                                # Calculate distance
                                distance = self._calculate_distance(prev_pos, curr_pos)
                                
                                # Calculate speed (pixels per second)
                                speed = distance / self.time_per_frame
                                
                                # Store results
                                track_data['distance'] = distance
                                track_data['speed'] = speed
                            else:
                                track_data['distance'] = 0.0
                                track_data['speed'] = 0.0
                        else:
                            track_data['distance'] = 0.0
                            track_data['speed'] = 0.0
    
    def _calculate_distance(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate Euclidean distance between two positions.
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Distance between the positions
        """
        return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    
    def get_average_speed(self, tracks: Dict[str, List[Dict]], object_type: str = "players") -> float:
        """Calculate average speed for a specific object type.
        
        Args:
            tracks: Dictionary containing object tracks
            object_type: Type of object to calculate average speed for
            
        Returns:
            Average speed in pixels per second
        """
        if object_type not in tracks:
            return 0.0
        
        speeds = []
        for frame_tracks in tracks[object_type]:
            for track_data in frame_tracks.values():
                if 'speed' in track_data:
                    speeds.append(track_data['speed'])
        
        return np.mean(speeds) if speeds else 0.0
    
    def get_total_distance(self, tracks: Dict[str, List[Dict]], track_id: int, object_type: str = "players") -> float:
        """Calculate total distance traveled by a specific track.
        
        Args:
            tracks: Dictionary containing object tracks
            track_id: ID of the track to calculate distance for
            object_type: Type of object
            
        Returns:
            Total distance traveled
        """
        if object_type not in tracks:
            return 0.0
        
        total_distance = 0.0
        for frame_tracks in tracks[object_type]:
            if track_id in frame_tracks and 'distance' in frame_tracks[track_id]:
                total_distance += frame_tracks[track_id]['distance']
        
        return total_distance
    
    def draw_speed_overlay(
        self, 
        frame: np.ndarray, 
        tracks: Dict[str, List[Dict]], 
        frame_num: int,
        show_speed: bool = True,
        show_distance: bool = True
    ) -> np.ndarray:
        """Draw speed and distance information overlay on frame.
        
        Args:
            frame: Input frame
            tracks: Dictionary containing object tracks
            frame_num: Current frame number
            show_speed: Whether to show speed information
            show_distance: Whether to show distance information
            
        Returns:
            Frame with speed/distance overlay
        """
        result_frame = frame.copy()
        
        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 150), WHITE, -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
        
        y_offset = 30
        
        # Draw speed information
        if show_speed:
            for object_type in ["players", "ball"]:
                if object_type in tracks and frame_num < len(tracks[object_type]):
                    frame_tracks = tracks[object_type][frame_num]
                    speeds = []
                    
                    for track_data in frame_tracks.values():
                        if 'speed' in track_data:
                            speeds.append(track_data['speed'])
                    
                    if speeds:
                        avg_speed = np.mean(speeds)
                        max_speed = np.max(speeds)
                        
                        cv2.putText(
                            result_frame,
                            f"{object_type.capitalize()} Avg Speed: {avg_speed:.1f} px/s",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            BLACK,
                            2
                        )
                        y_offset += 25
                        
                        cv2.putText(
                            result_frame,
                            f"{object_type.capitalize()} Max Speed: {max_speed:.1f} px/s",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            BLACK,
                            2
                        )
                        y_offset += 25
        
        # Draw distance information
        if show_distance:
            for object_type in ["players", "ball"]:
                if object_type in tracks and frame_num < len(tracks[object_type]):
                    frame_tracks = tracks[object_type][frame_num]
                    distances = []
                    
                    for track_data in frame_tracks.values():
                        if 'distance' in track_data:
                            distances.append(track_data['distance'])
                    
                    if distances:
                        total_distance = np.sum(distances)
                        
                        cv2.putText(
                            result_frame,
                            f"{object_type.capitalize()} Total Distance: {total_distance:.1f} px",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            BLACK,
                            2
                        )
                        y_offset += 25
        
        return result_frame
    
    def export_speed_analysis(
        self, 
        tracks: Dict[str, List[Dict]], 
        output_path: Path
    ) -> None:
        """Export speed and distance analysis to CSV file.
        
        Args:
            tracks: Dictionary containing object tracks
            output_path: Path to save the analysis
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("frame,object_type,track_id,speed,distance,total_distance\n")
            
            for object_type in tracks:
                for frame_num, frame_tracks in enumerate(tracks[object_type]):
                    for track_id, track_data in frame_tracks.items():
                        speed = track_data.get('speed', 0.0)
                        distance = track_data.get('distance', 0.0)
                        total_distance = self.get_total_distance(tracks, track_id, object_type)
                        
                        f.write(f"{frame_num},{object_type},{track_id},{speed:.2f},{distance:.2f},{total_distance:.2f}\n")
    
    def get_speed_statistics(self, tracks: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Get comprehensive speed and distance statistics.
        
        Args:
            tracks: Dictionary containing object tracks
            
        Returns:
            Dictionary containing speed and distance statistics
        """
        stats = {}
        
        for object_type in tracks:
            speeds = []
            distances = []
            
            for frame_tracks in tracks[object_type]:
                for track_data in frame_tracks.values():
                    if 'speed' in track_data:
                        speeds.append(track_data['speed'])
                    if 'distance' in track_data:
                        distances.append(track_data['distance'])
            
            if speeds:
                stats[object_type] = {
                    'avg_speed': np.mean(speeds),
                    'max_speed': np.max(speeds),
                    'min_speed': np.min(speeds),
                    'std_speed': np.std(speeds),
                    'total_distance': np.sum(distances),
                    'avg_distance_per_frame': np.mean(distances) if distances else 0.0
                }
        
        return stats