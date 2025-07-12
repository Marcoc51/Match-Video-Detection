"""
Data preparation module for cross detection training.
Extracts frames from videos and prepares YOLO format annotations.
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from tqdm import tqdm
import random
from ultralytics import YOLO
from src.events.cross_detector import CrossDetector
from src.tracking.tracker import Tracker
from src.utils.video_utils import read_video


class CrossDataPreparation:
    """
    Prepares training data for cross detection by extracting frames and annotations.
    """
    
    def __init__(self, config_path: str = "training_config.yaml"):
        """
        Initialize the data preparation pipeline.
        
        Args:
            config_path: Path to the training configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).resolve().parents[2]
        self.train_videos_dir = self.project_root / self.config['data']['train_videos_dir']
        self.test_videos_dir = self.project_root / self.config['data']['test_videos_dir']
        self.output_dir = self.project_root / self.config['data']['output_dir']
        self.frames_per_second = self.config['data']['frames_per_second']
        self.image_size = self.config['data']['image_size']
        
        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO model for cross detection
        model_path = self.project_root / self.config['model']['base_model']
        self.model = YOLO(str(model_path))
        self.tracker = Tracker(str(model_path))
        
    def extract_frames_from_video(self, video_path: Path, fps: int = 1) -> List[np.ndarray]:
        """
        Extract frames from video at specified FPS.
        
        Args:
            video_path: Path to the video file
            fps: Frames per second to extract
            
        Returns:
            List of extracted frames
        """
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / fps)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def detect_crosses_in_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Detect box crosses in frames using penalty area-based detection logic.
        
        Args:
            frames: List of video frames
            
        Returns:
            List of cross annotations per frame
        """
        # Get object tracks
        tracks = self.tracker.get_object_tracks(frames)
        
        cross_annotations = []
        total_crosses_detected = 0
        
        for frame_idx, frame in enumerate(frames):
            frame_crosses = []
            
            # Check if ball is detected in this frame
            if frame_idx < len(tracks['ball']) and 1 in tracks['ball'][frame_idx]:
                ball_bbox = tracks['ball'][frame_idx][1]['bbox']
                
                # Get frame dimensions
                h, w = frame.shape[:2]
                ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
                ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2
                
                # Box Cross Detection Logic
                # Box crosses occur INSIDE the penalty area, not from wide areas
                
                # Define penalty area boundaries (more lenient for training)
                # Penalty area is roughly the bottom 40% of the field (increased from 30%)
                penalty_area_top = int(0.6 * h)  # Increased from 0.7 to 0.6
                penalty_area_bottom = h  # Bottom of field
                
                # Penalty area width (increased to 60% of field width)
                penalty_area_left = int(0.2 * w)   # Reduced from 0.3 to 0.2
                penalty_area_right = int(0.8 * w)  # Increased from 0.7 to 0.8
                
                # Check if ball is in penalty area
                ball_in_penalty_area = (
                    penalty_area_top <= ball_center_y <= penalty_area_bottom and
                    penalty_area_left <= ball_center_x <= penalty_area_right
                )
                
                # Check if ball is in penalty box center (target area for box crosses)
                penalty_box_center_left = int(0.25 * w)   # Adjusted
                penalty_box_center_right = int(0.75 * w)  # Adjusted
                penalty_box_center_top = int(0.7 * h)     # Adjusted
                penalty_box_center_bottom = h
                
                ball_in_penalty_box_center = (
                    penalty_box_center_top <= ball_center_y <= penalty_box_center_bottom and
                    penalty_box_center_left <= ball_center_x <= penalty_box_center_right
                )
                
                # Check if ball is in penalty area channels (edges where crosses originate)
                channel_width = int(0.08 * w)  # Increased from 0.05 to 0.08
                left_channel = (
                    penalty_area_left <= ball_center_x <= penalty_area_left + channel_width and
                    penalty_area_top <= ball_center_y <= penalty_area_bottom
                )
                right_channel = (
                    penalty_area_right - channel_width <= ball_center_x <= penalty_area_right and
                    penalty_area_top <= ball_center_y <= penalty_area_bottom
                )
                
                # More lenient box cross detection criteria
                is_box_cross = (
                    ball_in_penalty_area and  # Ball must be in penalty area
                    (
                        ball_in_penalty_box_center or  # Ball in target area
                        left_channel or                # Ball in left channel
                        right_channel or               # Ball in right channel
                        (ball_center_y > int(0.65 * h) and random.random() < 0.4)  # 40% chance for lower penalty area
                    )
                )
                
                # More lenient ball size criteria for low cross
                ball_width = ball_bbox[2] - ball_bbox[0]
                ball_height = ball_bbox[3] - ball_bbox[1]
                ball_size = ball_width * ball_height
                
                # Normalize ball size relative to frame size
                frame_area = w * h
                normalized_ball_size = ball_size / frame_area
                
                # More lenient low cross detection
                is_low_cross = normalized_ball_size > 0.0005  # Reduced from 0.001 to 0.0005
                
                if is_box_cross and is_low_cross:
                    # Convert bbox to YOLO format (normalized coordinates)
                    x_center = ball_center_x / w
                    y_center = ball_center_y / h
                    width = (ball_bbox[2] - ball_bbox[0]) / w
                    height = (ball_bbox[3] - ball_bbox[1]) / h
                    
                    # Ensure coordinates are within valid range [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.01, min(1.0, width))  # Minimum width of 1%
                    height = max(0.01, min(1.0, height))  # Minimum height of 1%
                    
                    frame_crosses.append({
                        'class_id': 0,  # Cross class
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': 0.9  # Higher confidence for box crosses
                    })
                    total_crosses_detected += 1
            
            cross_annotations.append(frame_crosses)
        
        print(f"Total box crosses detected across all frames: {total_crosses_detected}")
        return cross_annotations
    
    def save_yolo_annotations(self, annotations: List[Dict], output_path: Path):
        """
        Save annotations in YOLO format.
        
        Args:
            annotations: List of cross annotations for a frame
            output_path: Path to save the annotation file
        """
        with open(output_path, 'w') as f:
            for annotation in annotations:
                line = f"{annotation['class_id']} {annotation['x_center']:.6f} {annotation['y_center']:.6f} {annotation['width']:.6f} {annotation['height']:.6f}\n"
                f.write(line)
    
    def prepare_dataset(self, split_ratio: float = 0.8):
        """
        Prepare the complete dataset for training.
        
        Args:
            split_ratio: Ratio of training to validation data
        """
        print("Preparing training dataset...")
        
        # Process training videos
        train_videos = list(self.train_videos_dir.glob("*.mp4"))
        test_videos = list(self.test_videos_dir.glob("*.mp4"))
        
        all_videos = train_videos + test_videos
        random.shuffle(all_videos)
        
        split_idx = int(len(all_videos) * split_ratio)
        train_videos = all_videos[:split_idx]
        val_videos = all_videos[split_idx:]
        
        print(f"Training videos: {len(train_videos)}")
        print(f"Validation videos: {len(val_videos)}")
        
        # Process training data
        train_stats = self._process_video_set(train_videos, "train")
        
        # Process validation data
        val_stats = self._process_video_set(val_videos, "val")
        
        # Create dataset.yaml file
        self._create_dataset_yaml()
        
        # Print statistics
        print("\n" + "="*50)
        print("DATASET PREPARATION STATISTICS")
        print("="*50)
        print(f"Training frames: {train_stats['frames']}")
        print(f"Training crosses: {train_stats['crosses']}")
        print(f"Validation frames: {val_stats['frames']}")
        print(f"Validation crosses: {val_stats['crosses']}")
        print(f"Total crosses: {train_stats['crosses'] + val_stats['crosses']}")
        
        if train_stats['crosses'] + val_stats['crosses'] == 0:
            print("\n⚠️  WARNING: No crosses detected! The training will fail.")
            print("Consider adjusting the cross detection criteria in data_preparation.py")
        else:
            print("\n✅ Dataset preparation completed successfully!")
    
    def _process_video_set(self, videos: List[Path], split: str) -> Dict[str, int]:
        """
        Process a set of videos for a specific split.
        
        Args:
            videos: List of video paths
            split: Split name ('train' or 'val')
            
        Returns:
            Dictionary with statistics
        """
        frame_count = 0
        cross_count = 0
        
        for video_path in tqdm(videos, desc=f"Processing {split} videos"):
            # Extract frames
            frames = self.extract_frames_from_video(video_path, self.frames_per_second)
            
            # Detect crosses
            cross_annotations = self.detect_crosses_in_frames(frames)
            
            # Save frames and annotations
            for frame_idx, (frame, annotations) in enumerate(zip(frames, cross_annotations)):
                # Save frame
                frame_filename = f"{video_path.stem}_{frame_idx:04d}.jpg"
                frame_path = self.output_dir / "images" / split / frame_filename
                cv2.imwrite(str(frame_path), frame)
                
                # Save annotations
                label_filename = f"{video_path.stem}_{frame_idx:04d}.txt"
                label_path = self.output_dir / "labels" / split / label_filename
                self.save_yolo_annotations(annotations, label_path)
                
                frame_count += 1
                cross_count += len(annotations)
        
        print(f"Processed {frame_count} frames with {cross_count} crosses for {split} split")
        return {'frames': frame_count, 'crosses': cross_count}
    
    def _create_dataset_yaml(self):
        """Create the dataset.yaml file required by YOLO training."""
        dataset_config = {
            'path': str(self.output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': self.config['model']['num_classes'],
            'names': self.config['model']['class_names']
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Dataset config saved to {yaml_path}")


if __name__ == "__main__":
    # Example usage
    data_prep = CrossDataPreparation()
    data_prep.prepare_dataset() 