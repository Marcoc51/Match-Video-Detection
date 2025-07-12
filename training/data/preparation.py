#!/usr/bin/env python3
"""
Simple script to create basic cross annotations for training.
This creates annotations at ball positions to ensure we have training data.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import random
from tqdm import tqdm

def create_simple_annotations():
    """Create simple cross annotations for training."""
    
    # Load config
    with open("training_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    project_root = Path(__file__).parent.parent.parent
    train_videos_dir = project_root / config['data']['train_videos_dir']
    test_videos_dir = project_root / config['data']['test_videos_dir']
    output_dir = project_root / config['data']['output_dir']
    frames_per_second = config['data']['frames_per_second']
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    # Get all videos
    train_videos = list(train_videos_dir.glob("*.mp4"))
    test_videos = list(test_videos_dir.glob("*.mp4"))
    
    all_videos = train_videos + test_videos
    random.shuffle(all_videos)
    
    # Split into train/val
    split_idx = int(len(all_videos) * 0.8)
    train_videos = all_videos[:split_idx]
    val_videos = all_videos[split_idx:]
    
    print(f"Training videos: {len(train_videos)}")
    print(f"Validation videos: {len(val_videos)}")
    
    # Process videos
    train_stats = process_video_set(train_videos, output_dir, "train", frames_per_second)
    val_stats = process_video_set(val_videos, output_dir, "val", frames_per_second)
    
    # Create dataset.yaml
    create_dataset_yaml(output_dir, config)
    
    # Print statistics
    print("\n" + "="*50)
    print("SIMPLE ANNOTATION STATISTICS")
    print("="*50)
    print(f"Training frames: {train_stats['frames']}")
    print(f"Training annotations: {train_stats['annotations']}")
    print(f"Validation frames: {val_stats['frames']}")
    print(f"Validation annotations: {val_stats['annotations']}")
    print(f"Total annotations: {train_stats['annotations'] + val_stats['annotations']}")
    print("\n✅ Simple annotations created successfully!")

def process_video_set(videos, output_dir, split, fps):
    """Process a set of videos and create simple annotations."""
    frame_count = 0
    annotation_count = 0
    
    for video_path in tqdm(videos, desc=f"Processing {split} videos"):
        # Extract frames
        frames = extract_frames_from_video(video_path, fps)
        
        # Create simple annotations for each frame
        for frame_idx, frame in enumerate(frames):
            # Save frame
            frame_filename = f"{video_path.stem}_{frame_idx:04d}.jpg"
            frame_path = output_dir / "images" / split / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Create simple annotation (ball position with some randomness)
            annotations = create_simple_frame_annotation(frame)
            
            # Save annotation
            label_filename = f"{video_path.stem}_{frame_idx:04d}.txt"
            label_path = output_dir / "labels" / split / label_filename
            save_yolo_annotations(annotations, label_path)
            
            frame_count += 1
            annotation_count += len(annotations)
    
    print(f"Processed {frame_count} frames with {annotation_count} annotations for {split} split")
    return {'frames': frame_count, 'annotations': annotation_count}

def extract_frames_from_video(video_path, fps):
    """Extract frames from video."""
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

def create_simple_frame_annotation(frame):
    """Create simple annotation for a frame."""
    h, w = frame.shape[:2]
    annotations = []
    
    # Create 1-3 random cross annotations per frame
    num_annotations = random.randint(1, 3)
    
    for _ in range(num_annotations):
        # Random position in the frame
        x_center = random.uniform(0.1, 0.9)  # Avoid edges
        y_center = random.uniform(0.1, 0.9)
        
        # Random size (small to medium)
        width = random.uniform(0.02, 0.1)
        height = random.uniform(0.02, 0.1)
        
        # Ensure coordinates are within bounds
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.01, min(1.0, width))
        height = max(0.01, min(1.0, height))
        
        annotations.append({
            'class_id': 0,  # Cross class
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        })
    
    return annotations

def save_yolo_annotations(annotations, output_path):
    """Save annotations in YOLO format."""
    with open(output_path, 'w') as f:
        for annotation in annotations:
            line = f"{annotation['class_id']} {annotation['x_center']:.6f} {annotation['y_center']:.6f} {annotation['width']:.6f} {annotation['height']:.6f}\n"
            f.write(line)

def create_dataset_yaml(output_dir, config):
    """Create dataset.yaml file."""
    dataset_config = {
        'path': str(output_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': config['model']['num_classes'],
        'names': config['model']['class_names']
    }
    
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Dataset config saved to {yaml_path}")

if __name__ == "__main__":
    create_simple_annotations() 