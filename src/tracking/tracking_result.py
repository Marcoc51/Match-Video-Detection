"""
Tracking result data structures and processing.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pickle


@dataclass
class Track:
    """Represents a single object track across multiple frames."""
    track_id: int
    class_name: str
    class_id: int
    bboxes: List[Tuple[float, float, float, float]] = field(default_factory=list)
    positions: List[Tuple[float, float]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    team: Optional[int] = None
    team_color: Optional[Tuple[int, int, int]] = None
    has_ball: List[bool] = field(default_factory=list)
    
    def add_detection(
        self,
        frame_idx: int,
        bbox: Tuple[float, float, float, float],
        position: Tuple[float, float],
        confidence: float,
        has_ball: bool = False
    ) -> None:
        """Add a detection to this track."""
        self.frame_indices.append(frame_idx)
        self.bboxes.append(bbox)
        self.positions.append(position)
        self.confidences.append(confidence)
        self.has_ball.append(has_ball)
    
    def get_position_at_frame(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        """Get position at a specific frame."""
        try:
            idx = self.frame_indices.index(frame_idx)
            return self.positions[idx]
        except ValueError:
            return None
    
    def get_bbox_at_frame(self, frame_idx: int) -> Optional[Tuple[float, float, float, float]]:
        """Get bounding box at a specific frame."""
        try:
            idx = self.frame_indices.index(frame_idx)
            return self.bboxes[idx]
        except ValueError:
            return None
    
    def get_duration(self) -> int:
        """Get the duration of this track in frames."""
        if not self.frame_indices:
            return 0
        return max(self.frame_indices) - min(self.frame_indices) + 1
    
    def get_average_confidence(self) -> float:
        """Get average confidence across all detections."""
        if not self.confidences:
            return 0.0
        return np.mean(self.confidences)
    
    def is_active_at_frame(self, frame_idx: int) -> bool:
        """Check if track is active at given frame."""
        return frame_idx in self.frame_indices


class TrackingResult:
    """
    Container for tracking results across multiple frames.
    
    This class manages tracks for different object types (players, ball, referees)
    and provides methods for accessing and analyzing tracking data.
    """
    
    def __init__(self):
        """Initialize empty tracking result."""
        self.tracks = {
            "players": {},      # track_id -> Track
            "ball": {},         # track_id -> Track  
            "referees": {}      # track_id -> Track
        }
        self.frame_count = 0
        self.metadata = {}
    
    def add_track(self, track: Track, object_type: str) -> None:
        """
        Add a track to the tracking result.
        
        Args:
            track: Track object to add
            object_type: Type of object ("players", "ball", "referees")
        """
        if object_type not in self.tracks:
            raise ValueError(f"Invalid object type: {object_type}")
        
        self.tracks[object_type][track.track_id] = track
    
    def get_track(self, track_id: int, object_type: str) -> Optional[Track]:
        """Get a specific track by ID and type."""
        return self.tracks[object_type].get(track_id)
    
    def get_tracks_by_type(self, object_type: str) -> Dict[int, Track]:
        """Get all tracks of a specific type."""
        return self.tracks.get(object_type, {})
    
    def get_active_tracks_at_frame(self, frame_idx: int, object_type: str) -> Dict[int, Track]:
        """Get all tracks active at a specific frame."""
        active_tracks = {}
        for track_id, track in self.tracks[object_type].items():
            if track.is_active_at_frame(frame_idx):
                active_tracks[track_id] = track
        return active_tracks
    
    def get_player_tracks(self) -> Dict[int, Track]:
        """Get all player tracks."""
        return self.tracks["players"]
    
    def get_ball_tracks(self) -> Dict[int, Track]:
        """Get all ball tracks."""
        return self.tracks["ball"]
    
    def get_referee_tracks(self) -> Dict[int, Track]:
        """Get all referee tracks."""
        return self.tracks["referees"]
    
    def get_track_count(self, object_type: str) -> int:
        """Get number of tracks for a specific type."""
        return len(self.tracks[object_type])
    
    def get_total_track_count(self) -> int:
        """Get total number of tracks across all types."""
        return sum(len(tracks) for tracks in self.tracks.values())
    
    def filter_tracks_by_duration(self, min_duration: int, object_type: str) -> Dict[int, Track]:
        """Filter tracks by minimum duration."""
        filtered_tracks = {}
        for track_id, track in self.tracks[object_type].items():
            if track.get_duration() >= min_duration:
                filtered_tracks[track_id] = track
        return filtered_tracks
    
    def filter_tracks_by_confidence(self, min_confidence: float, object_type: str) -> Dict[int, Track]:
        """Filter tracks by minimum average confidence."""
        filtered_tracks = {}
        for track_id, track in self.tracks[object_type].items():
            if track.get_average_confidence() >= min_confidence:
                filtered_tracks[track_id] = track
        return filtered_tracks
    
    def get_track_statistics(self) -> Dict[str, Any]:
        """Get statistics about all tracks."""
        stats = {
            "total_tracks": self.get_total_track_count(),
            "frame_count": self.frame_count,
            "by_type": {}
        }
        
        for object_type, tracks in self.tracks.items():
            if tracks:
                durations = [track.get_duration() for track in tracks.values()]
                confidences = [track.get_average_confidence() for track in tracks.values()]
                
                stats["by_type"][object_type] = {
                    "count": len(tracks),
                    "avg_duration": np.mean(durations),
                    "avg_confidence": np.mean(confidences),
                    "min_duration": min(durations),
                    "max_duration": max(durations)
                }
            else:
                stats["by_type"][object_type] = {
                    "count": 0,
                    "avg_duration": 0,
                    "avg_confidence": 0,
                    "min_duration": 0,
                    "max_duration": 0
                }
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracking result to dictionary format."""
        tracks_dict = {}
        for object_type, tracks in self.tracks.items():
            tracks_dict[object_type] = {}
            for track_id, track in tracks.items():
                tracks_dict[object_type][track_id] = {
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "class_id": track.class_id,
                    "bboxes": track.bboxes,
                    "positions": track.positions,
                    "confidences": track.confidences,
                    "frame_indices": track.frame_indices,
                    "team": track.team,
                    "team_color": track.team_color,
                    "has_ball": track.has_ball
                }
        
        return {
            "tracks": tracks_dict,
            "frame_count": self.frame_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackingResult':
        """Create TrackingResult from dictionary."""
        result = cls()
        result.frame_count = data.get("frame_count", 0)
        result.metadata = data.get("metadata", {})
        
        tracks_data = data.get("tracks", {})
        for object_type, tracks in tracks_data.items():
            for track_id, track_data in tracks.items():
                track = Track(
                    track_id=track_data["track_id"],
                    class_name=track_data["class_name"],
                    class_id=track_data["class_id"],
                    bboxes=track_data["bboxes"],
                    positions=track_data["positions"],
                    confidences=track_data["confidences"],
                    frame_indices=track_data["frame_indices"],
                    team=track_data.get("team"),
                    team_color=track_data.get("team_color"),
                    has_ball=track_data.get("has_ball", [])
                )
                result.add_track(track, object_type)
        
        return result
    
    def save(self, file_path: Path) -> None:
        """Save tracking result to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, file_path: Path) -> 'TrackingResult':
        """Load tracking result from file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Tracking result file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return cls.from_dict(data)
    
    def __len__(self) -> int:
        """Return total number of tracks."""
        return self.get_total_track_count()
    
    def __contains__(self, track_id: int) -> bool:
        """Check if track ID exists in any object type."""
        for tracks in self.tracks.values():
            if track_id in tracks:
                return True
        return False 