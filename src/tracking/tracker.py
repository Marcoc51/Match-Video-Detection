"""
Main tracking module for football match analysis.

This module provides the main Tracker class that integrates detection,
tracking, and visualization for football match videos.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import supervision as sv

from ..detection.yolo_detector import YOLODetector
from .tracking_result import TrackingResult
from .track_manager import TrackManager
from ..utils.bbox_utils import get_bbox_width, get_center_of_bbox, get_foot_position
from ..utils.colors import *


class Tracker:
    """
    Main tracker class for football match analysis.
    
    This class integrates YOLO detection with ByteTrack tracking
    and provides comprehensive tracking functionality for players,
    ball, and referees.
    """
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        confidence_threshold: float = None,
        iou_threshold: float = None,
        device: str = "auto"
    ):
        """
        Initialize the tracker.
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        # Initialize YOLO detector
        self.detector = YOLODetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        
        # Initialize ByteTrack tracker
        self.tracker = sv.ByteTrack()
        
        # Initialize track manager
        self.track_manager = TrackManager()
        
        # Tracking results
        self.tracking_result = TrackingResult()
        
        print("Tracker initialized successfully")

    def get_object_tracks(
        self, 
        frames: List[np.ndarray],
        read_from_stub: bool = False,
        stub_path: Optional[Path] = None
    ) -> Dict[str, List[Dict]]:
        """
        Get object tracks from video frames.
        
        Args:
            frames: List of video frames
            read_from_stub: Whether to read from cached stub file
            stub_path: Path to stub file for caching
            
        Returns:
            Dictionary containing tracks for players, referees, and ball
        """
        # Load from stub if requested and available
        if read_from_stub and stub_path is not None and stub_path.exists():
            import pickle
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            print(f"Loaded tracks from stub: {stub_path}")
            return tracks

        # Detect objects in all frames
        print("Detecting objects in frames...")
        detection_results = self.detector.detect_frames(frames)
        
        # Initialize tracking structure
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        # Process each frame
        for frame_num, detection_result in enumerate(detection_results):
            # Initialize frame dictionaries
            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})
            
            # Convert DetectionResult to supervision format for tracking
            if len(detection_result.detections) > 0:
                # Extract data from DetectionResult
                bboxes = []
                confidences = []
                class_ids = []
                
                for detection in detection_result.detections:
                    bboxes.append(detection.bbox)
                    confidences.append(detection.confidence)
                    class_ids.append(detection.class_id)
                
                # Create supervision Detections object
                detection_supervision = sv.Detections(
                    xyxy=np.array(bboxes),
                    confidence=np.array(confidences),
                    class_id=np.array(class_ids)
                )
                
                # Class name mapping
                class_names = {0: "player", 1: "ball", 2: "referee", 3: "goalkeeper"}
                class_names_inv = {v.lower(): k for k, v in class_names.items()}
                
                # Convert goalkeeper to player for tracking
                for object_idx, class_id in enumerate(detection_supervision.class_id):
                    if class_names.get(class_id) == "goalkeeper":
                        detection_supervision.class_id[object_idx] = class_names_inv['player']
                
                # Track objects using ByteTrack
                detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
                
                # Process tracked objects
                for detection in detections_with_tracks:
                    bbox = detection[0].tolist()
                    class_id = detection[3]
                    track_id = detection[4]
                    
                    if class_id == class_names_inv['player']:
                        tracks['players'][frame_num][track_id] = {"bbox": bbox}
                    elif class_id == class_names_inv['referee']:
                        tracks['referees'][frame_num][track_id] = {"bbox": bbox}
                
                # Process ball detections (not tracked by ByteTrack)
                for detection in detection_supervision:
                    bbox = detection[0].tolist()
                    class_id = detection[3]
                    
                    if class_id == class_names_inv['ball']:
                        tracks['ball'][frame_num][1] = {"bbox": bbox}
        
        # Save to stub if requested
        if stub_path is not None:
            import pickle
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
            print(f"Saved tracks to stub: {stub_path}")
        
        return tracks
    
    def interpolate_ball_positions(self, ball_positions: List[Dict]) -> List[Dict]:
        """
        Interpolate missing ball positions using pandas.
        
        Args:
            ball_positions: List of ball position dictionaries
            
        Returns:
            List of interpolated ball positions
        """
        # Extract bboxes from ball positions
        bboxes = [pos.get(1, {}).get('bbox', []) for pos in ball_positions]
        
        # Create DataFrame for interpolation
        df_ball_positions = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])
        
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        # Convert back to original format
        interpolated_positions = [{1: {"bbox": bbox}} for bbox in df_ball_positions.to_numpy().tolist()]
        
        return interpolated_positions

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _  = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center = (x_center, y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle=235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                cv2.FILLED
            )
            
            x1_text = x1_rect+12
            if track_id > 9 and track_id <= 99:
                x1_text = x1_rect+8
            elif track_id > 99:
                x1_text = x1_rect+4
                
            cv2.putText(
                frame,
                str(track_id),
                (int(x1_text),int(y2_rect-5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                BLACK,
                2
            )

    def draw_triangle(self, frame, bbox, color):
        y2 = int(bbox[3])

        x_center, _  = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        triangle_points = np.array([
            [x_center, y2],
            [x_center - int(width//2), y2 - int(width//2)],
            [x_center + int(width//2), y2 - int(width//2)]
        ], np.int32)

        cv2.fillPoly(frame, [triangle_points], color)

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 100), WHITE, -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw team ball control information
        if frame_num < len(team_ball_control):
            team_id = team_ball_control[frame_num]
            team_name = "Home" if team_id == 1 else "Away" if team_id == 2 else "None"
            team_color = RED if team_id == 1 else BLUE if team_id == 2 else BLACK
            
            cv2.putText(
                frame, 
                f"Ball Control: {team_name}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                team_color,
                3
            )
            
            cv2.putText(
                frame, 
                f"Team ID: {team_id}", 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                team_color,
                3
            )

    def draw_pass_arrow(self, frame, pos1, pos2):
        # Draw a blue arrow between two points
        cv2.arrowedLine(frame, pos1, pos2, BLUE, 3, tipLength=0.3)

    def draw_annotations(self, video_frames, tracks, team_ball_control, passes=None):
        annotated_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy()
            
            # Draw team ball control
            self.draw_team_ball_control(annotated_frame, frame_num, team_ball_control)
            
            # Draw player ellipses
            if frame_num < len(tracks['players']):
                for player_id, player_data in tracks['players'][frame_num].items():
                    bbox = player_data['bbox']
                    team_id = player_data.get('team', 0)
                    
                    # Choose color based on team
                    if team_id == 1:
                        color = RED
                    elif team_id == 2:
                        color = BLUE
                    else:
                        color = GREY
                    
                    self.draw_ellipse(annotated_frame, bbox, color, player_id)
            
            # Draw referee triangles
            if frame_num < len(tracks['referees']):
                for ref_id, ref_data in tracks['referees'][frame_num].items():
                    bbox = ref_data['bbox']
                    self.draw_triangle(annotated_frame, bbox, YELLOW)
            
            # Draw ball
            if frame_num < len(tracks['ball']):
                for ball_id, ball_data in tracks['ball'][frame_num].items():
                    bbox = ball_data['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.circle(annotated_frame, (int((x1+x2)/2), int((y1+y2)/2)), 5, GREEN, -1)
            
            # Draw passes if provided
            if passes:
                for pass_data in passes:
                    if pass_data['frame'] == frame_num:
                        start_pos = pass_data['start_pos']
                        end_pos = pass_data['end_pos']
                        self.draw_pass_arrow(annotated_frame, start_pos, end_pos)
            
            annotated_frames.append(annotated_frame)
        
        return annotated_frames

    def add_position_to_tracks(self,tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, frame_tracks in enumerate(object_tracks):
                for track_id, track_data in frame_tracks.items():
                    bbox = track_data['bbox']
                    center = get_center_of_bbox(bbox)
                    tracks[object_type][frame_num][track_id]['position'] = center