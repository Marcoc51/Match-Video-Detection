"""
Track management and lifecycle handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from .tracking_result import Track, TrackingResult
from ..detection.detection_result import DetectionResult
from ..core.config import Config


class TrackManager:
    """
    Manages track lifecycle and provides track processing utilities.
    
    This class handles track creation, maintenance, filtering, and
    post-processing operations like interpolation and smoothing.
    """
    
    def __init__(self):
        """Initialize track manager."""
        self.tracking_result = TrackingResult()
    
    def create_tracks_from_detections(
        self,
        detection_results: List[DetectionResult],
        track_assignments: Dict[str, Dict[int, int]]  # object_type -> {detection_idx -> track_id}
    ) -> TrackingResult:
        """
        Create tracks from detection results and track assignments.
        
        Args:
            detection_results: List of DetectionResult objects
            track_assignments: Dictionary mapping detections to track IDs
            
        Returns:
            TrackingResult with created tracks
        """
        self.tracking_result = TrackingResult()
        self.tracking_result.frame_count = len(detection_results)
        
        # Process each frame
        for frame_idx, detection_result in enumerate(detection_results):
            # Process players
            player_detections = detection_result.get_player_detections()
            for det_idx, detection in enumerate(player_detections):
                track_id = track_assignments.get("players", {}).get(det_idx)
                if track_id is not None:
                    self._add_detection_to_track(
                        track_id, "players", frame_idx, detection
                    )
            
            # Process ball
            ball_detection = detection_result.get_ball_detection()
            if ball_detection:
                track_id = track_assignments.get("ball", {}).get(0, 1)  # Default ball track ID
                self._add_detection_to_track(
                    track_id, "ball", frame_idx, ball_detection
                )
            
            # Process referees
            referee_detections = detection_result.get_referee_detections()
            for det_idx, detection in enumerate(referee_detections):
                track_id = track_assignments.get("referees", {}).get(det_idx)
                if track_id is not None:
                    self._add_detection_to_track(
                        track_id, "referees", frame_idx, detection
                    )
        
        return self.tracking_result
    
    def _add_detection_to_track(
        self,
        track_id: int,
        object_type: str,
        frame_idx: int,
        detection
    ) -> None:
        """Add a detection to an existing or new track."""
        # Get or create track
        track = self.tracking_result.get_track(track_id, object_type)
        if track is None:
            track = Track(
                track_id=track_id,
                class_name=detection.class_name,
                class_id=detection.class_id
            )
            self.tracking_result.add_track(track, object_type)
        
        # Add detection to track
        position = detection.center
        track.add_detection(
            frame_idx=frame_idx,
            bbox=detection.bbox,
            position=position,
            confidence=detection.confidence
        )
    
    def interpolate_missing_positions(
        self,
        object_type: str,
        max_gap: int = 10
    ) -> None:
        """
        Interpolate missing positions in tracks.
        
        Args:
            object_type: Type of object to interpolate
            max_gap: Maximum gap size to interpolate
        """
        tracks = self.tracking_result.get_tracks_by_type(object_type)
        
        for track in tracks.values():
            if len(track.frame_indices) < 2:
                continue
            
            # Find gaps in frame indices
            frame_indices = sorted(track.frame_indices)
            positions = [track.get_position_at_frame(idx) for idx in frame_indices]
            
            # Create DataFrame for interpolation
            df = pd.DataFrame({
                'frame': frame_indices,
                'x': [pos[0] for pos in positions],
                'y': [pos[1] for pos in positions]
            })
            
            # Find missing frames
            all_frames = range(min(frame_indices), max(frame_indices) + 1)
            missing_frames = [f for f in all_frames if f not in frame_indices]
            
            # Interpolate missing positions
            for missing_frame in missing_frames:
                if missing_frame < min(frame_indices) or missing_frame > max(frame_indices):
                    continue
                
                # Find surrounding frames
                before_frames = [f for f in frame_indices if f < missing_frame]
                after_frames = [f for f in frame_indices if f > missing_frame]
                
                if not before_frames or not after_frames:
                    continue
                
                gap_size = min(missing_frame - max(before_frames), 
                              min(after_frames) - missing_frame)
                
                if gap_size <= max_gap:
                    # Linear interpolation
                    before_frame = max(before_frames)
                    after_frame = min(after_frames)
                    
                    before_pos = track.get_position_at_frame(before_frame)
                    after_pos = track.get_position_at_frame(after_frame)
                    
                    if before_pos and after_pos:
                        # Calculate interpolation weight
                        weight = (missing_frame - before_frame) / (after_frame - before_frame)
                        
                        # Interpolate position
                        interpolated_x = before_pos[0] + weight * (after_pos[0] - before_pos[0])
                        interpolated_y = before_pos[1] + weight * (after_pos[1] - before_pos[1])
                        interpolated_pos = (interpolated_x, interpolated_y)
                        
                        # Add interpolated position to track
                        track.add_detection(
                            frame_idx=missing_frame,
                            bbox=track.get_bbox_at_frame(before_frame) or (0, 0, 0, 0),
                            position=interpolated_pos,
                            confidence=0.5  # Lower confidence for interpolated positions
                        )
    
    def smooth_tracks(
        self,
        object_type: str,
        window_size: int = 5
    ) -> None:
        """
        Apply smoothing to track positions using moving average.
        
        Args:
            object_type: Type of object to smooth
            window_size: Size of smoothing window
        """
        tracks = self.tracking_result.get_tracks_by_type(object_type)
        
        for track in tracks.values():
            if len(track.positions) < window_size:
                continue
            
            # Smooth x and y coordinates separately
            x_coords = [pos[0] for pos in track.positions]
            y_coords = [pos[1] for pos in track.positions]
            
            # Apply moving average
            smoothed_x = self._moving_average(x_coords, window_size)
            smoothed_y = self._moving_average(y_coords, window_size)
            
            # Update positions
            track.positions = list(zip(smoothed_x, smoothed_y))
    
    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Calculate moving average of a list of values."""
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            window = data[start:end]
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def filter_short_tracks(
        self,
        object_type: str,
        min_duration: int = 10
    ) -> None:
        """
        Remove tracks that are too short.
        
        Args:
            object_type: Type of object to filter
            min_duration: Minimum track duration in frames
        """
        tracks = self.tracking_result.get_tracks_by_type(object_type)
        tracks_to_remove = []
        
        for track_id, track in tracks.items():
            if track.get_duration() < min_duration:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del tracks[track_id]
    
    def filter_low_confidence_tracks(
        self,
        object_type: str,
        min_confidence: float = 0.3
    ) -> None:
        """
        Remove tracks with low average confidence.
        
        Args:
            object_type: Type of object to filter
            min_confidence: Minimum average confidence
        """
        tracks = self.tracking_result.get_tracks_by_type(object_type)
        tracks_to_remove = []
        
        for track_id, track in tracks.items():
            if track.get_average_confidence() < min_confidence:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del tracks[track_id]
    
    def merge_close_tracks(
        self,
        object_type: str,
        distance_threshold: float = 50.0,
        time_threshold: int = 5
    ) -> None:
        """
        Merge tracks that are close in space and time.
        
        Args:
            object_type: Type of object to merge
            distance_threshold: Maximum distance for merging
            time_threshold: Maximum time gap for merging
        """
        tracks = self.tracking_result.get_tracks_by_type(object_type)
        tracks_list = list(tracks.items())
        
        for i, (track_id1, track1) in enumerate(tracks_list):
            for j, (track_id2, track2) in enumerate(tracks_list[i+1:], i+1):
                if self._should_merge_tracks(track1, track2, distance_threshold, time_threshold):
                    self._merge_tracks(track1, track2)
                    del tracks[track_id2]
    
    def _should_merge_tracks(
        self,
        track1: Track,
        track2: Track,
        distance_threshold: float,
        time_threshold: int
    ) -> bool:
        """Check if two tracks should be merged."""
        # Check if tracks are close in time
        min_time1, max_time1 = min(track1.frame_indices), max(track1.frame_indices)
        min_time2, max_time2 = min(track2.frame_indices), max(track2.frame_indices)
        
        time_gap = min(abs(max_time1 - min_time2), abs(max_time2 - min_time1))
        if time_gap > time_threshold:
            return False
        
        # Check if tracks are close in space
        pos1 = track1.get_position_at_frame(max_time1)
        pos2 = track2.get_position_at_frame(min_time2)
        
        if pos1 and pos2:
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            return distance <= distance_threshold
        
        return False
    
    def _merge_tracks(self, track1: Track, track2: Track) -> None:
        """Merge track2 into track1."""
        # Add all detections from track2 to track1
        for i, frame_idx in enumerate(track2.frame_indices):
            track1.add_detection(
                frame_idx=frame_idx,
                bbox=track2.bboxes[i],
                position=track2.positions[i],
                confidence=track2.confidences[i],
                has_ball=track2.has_ball[i] if i < len(track2.has_ball) else False
            )
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all tracks."""
        return self.tracking_result.get_track_statistics()
    
    def export_tracks_to_csv(self, output_path: Path) -> None:
        """Export all tracks to CSV format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("track_id,object_type,frame_idx,x,y,confidence,has_ball\n")
            
            for object_type in ["players", "ball", "referees"]:
                tracks = self.tracking_result.get_tracks_by_type(object_type)
                
                for track in tracks.values():
                    for i, frame_idx in enumerate(track.frame_indices):
                        pos = track.positions[i]
                        conf = track.confidences[i]
                        has_ball = track.has_ball[i] if i < len(track.has_ball) else False
                        
                        f.write(f"{track.track_id},{object_type},{frame_idx},"
                               f"{pos[0]:.2f},{pos[1]:.2f},{conf:.3f},{has_ball}\n")
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get a track by ID from any object type."""
        for object_type in ["players", "ball", "referees"]:
            track = self.tracking_result.get_track(track_id, object_type)
            if track:
                return track
        return None 