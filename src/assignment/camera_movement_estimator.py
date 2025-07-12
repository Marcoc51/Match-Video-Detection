"""Camera movement estimation for video analysis."""

import pickle
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import os

from src.utils.bbox_utils import measure_distance, measure_xy_distance
from src.utils.colors import *


class CameraMovementEstimator:
    """Estimates camera movement in video frames."""

    def __init__(self, frame: np.ndarray):
        """Initialize the camera movement estimator.
        
        Args:
            frame: Reference frame for feature detection
        """
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features,
        )

    def add_adjust_positions_to_tracks(
        self, 
        tracks: Dict[str, List[Dict]], 
        camera_movement_per_frame: List[List[float]]
    ) -> None:
        """Add adjusted positions to tracks based on camera movement.
        
        Args:
            tracks: Dictionary containing object tracks
            camera_movement_per_frame: List of camera movements per frame
        """
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    if 'position' in track_info:
                        position = track_info['position']
                        camera_movement = camera_movement_per_frame[frame_num]
                        position_adjusted = (
                            position[0] - camera_movement[0], 
                            position[1] - camera_movement[1]
                        )
                        tracks[object_name][frame_num][track_id]['position_adjusted'] = \
                            position_adjusted

    def get_camera_movement(
        self, 
        frames: List[np.ndarray], 
        read_from_stub: bool = False, 
        stub_path: Optional[Path] = None
    ) -> List[List[float]]:
        """Get camera movement for each frame.
        
        Args:
            frames: List of video frames
            read_from_stub: Whether to read from cached stub file
            stub_path: Path to stub file for caching
            
        Returns:
            List of camera movements [x, y] for each frame
        """
        # Read from stub 
        if read_from_stub and stub_path is not None and stub_path.exists():
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            if new_features is not None and old_features is not None:
                for i, (new, old) in enumerate(zip(new_features, old_features)):
                    new_features_point = new.ravel()
                    old_features_point = old.ravel()

                    distance = measure_distance(new_features_point, old_features_point)
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = measure_xy_distance(
                            old_features_point, new_features_point
                        )

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
            
            old_gray = frame_gray.copy()

        if stub_path is not None:
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(
        self, 
        frames: List[np.ndarray], 
        camera_movement_per_frame: List[List[float]]
    ) -> List[np.ndarray]:
        """Draw camera movement information on frames.
        
        Args:
            frames: List of video frames
            camera_movement_per_frame: List of camera movements per frame
            
        Returns:
            List of frames with camera movement overlay
        """
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), WHITE, -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame, 
                f"Camera Movement X: {x_movement:.2f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                BLACK,
                3
            )
            frame = cv2.putText(
                frame, 
                f"Camera Movement Y: {y_movement:.2f}",
                (10, 75), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                BLACK,
                3
            )

            output_frames.append(frame) 

        return output_frames