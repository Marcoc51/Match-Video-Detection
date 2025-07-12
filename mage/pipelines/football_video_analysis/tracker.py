"""
Tracker Block for Football Video Analysis Pipeline
Handles object tracking across video frames
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from collections import defaultdict
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager
from src.tracking.tracker import Tracker
from src.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


@transformer
def track_objects(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Track objects across video frames.
    
    This block handles:
    - Player tracking across frames
    - Ball tracking across frames
    - Track association and management
    - Track quality assessment
    - Trajectory analysis
    
    Args:
        data: Input data containing detections
        
    Returns:
        Dict containing tracking results
    """
    try:
        detections = data['detections']
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=45, message="Tracking objects...")
        
        logger.info(f"Starting object tracking on {len(detections)} frames")
        
        # Initialize trackers
        player_tracker = initialize_tracker('player')
        ball_tracker = initialize_tracker('ball')
        
        # Track objects
        tracking_results = track_across_frames(detections, player_tracker, ball_tracker)
        
        # Analyze trajectories
        trajectory_analysis = analyze_trajectories(tracking_results)
        
        # Quality assessment
        tracking_quality = assess_tracking_quality(tracking_results)
        
        # Combine results
        result = {
            **data,
            'tracking_results': tracking_results,
            'trajectory_analysis': trajectory_analysis,
            'tracking_quality': tracking_quality,
            'tracking_config': {
                'max_disappeared': 30,  # frames
                'min_hits': 3,
                'iou_threshold': 0.3
            }
        }
        
        logger.info(f"Object tracking completed: {len(tracking_results['tracks'])} tracks")
        return result
        
    except Exception as e:
        logger.error(f"Error in tracker: {e}")
        raise


def initialize_tracker(object_type: str) -> Tracker:
    """Initialize tracker for specific object type."""
    config = {
        'max_disappeared': 30,  # frames
        'min_hits': 3,
        'iou_threshold': 0.3
    }
    
    if object_type == 'ball':
        # Ball tracking needs more sensitive parameters
        config.update({
            'max_disappeared': 15,
            'min_hits': 2,
            'iou_threshold': 0.2
        })
    
    tracker = Tracker(**config)
    logger.info(f"Initialized {object_type} tracker")
    return tracker


def track_across_frames(detections: List[Dict[str, Any]], player_tracker: Tracker, ball_tracker: Tracker) -> Dict[str, Any]:
    """Track objects across all frames."""
    tracks = {
        'players': [],
        'ball': [],
        'frame_tracks': []
    }
    
    player_tracks = defaultdict(list)
    ball_tracks = defaultdict(list)
    
    for frame_data in detections:
        frame_index = frame_data['frame_index']
        frame_detections = frame_data['detections']
        
        # Separate players and ball
        players = []
        ball = None
        
        for detection in frame_detections:
            class_name = detection.get('class_name', '').lower()
            if 'player' in class_name:
                players.append(detection)
            elif 'ball' in class_name:
                ball = detection
        
        # Track players
        player_trackers = player_tracker.update(players, frame_index)
        
        # Track ball
        ball_tracker_result = ball_tracker.update([ball] if ball else [], frame_index)
        
        # Store frame results
        frame_tracks = {
            'frame_index': frame_index,
            'timestamp': frame_data['timestamp'],
            'player_tracks': player_trackers,
            'ball_track': ball_tracker_result[0] if ball_tracker_result else None
        }
        
        tracks['frame_tracks'].append(frame_tracks)
        
        # Update individual track histories
        for track in player_trackers:
            player_tracks[track['track_id']].append({
                'frame_index': frame_index,
                'timestamp': frame_data['timestamp'],
                'bbox': track['bbox'],
                'confidence': track['confidence']
            })
        
        if ball_tracker_result:
            ball_tracks[ball_tracker_result[0]['track_id']].append({
                'frame_index': frame_index,
                'timestamp': frame_data['timestamp'],
                'bbox': ball_tracker_result[0]['bbox'],
                'confidence': ball_tracker_result[0]['confidence']
            })
    
    # Convert track histories to final format
    tracks['players'] = [
        {
            'track_id': track_id,
            'type': 'player',
            'trajectory': trajectory,
            'start_frame': trajectory[0]['frame_index'],
            'end_frame': trajectory[-1]['frame_index'],
            'duration': trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
        }
        for track_id, trajectory in player_tracks.items()
    ]
    
    tracks['ball'] = [
        {
            'track_id': track_id,
            'type': 'ball',
            'trajectory': trajectory,
            'start_frame': trajectory[0]['frame_index'],
            'end_frame': trajectory[-1]['frame_index'],
            'duration': trajectory[-1]['timestamp'] - trajectory[0]['timestamp']
        }
        for track_id, trajectory in ball_tracks.items()
    ]
    
    return tracks


def analyze_trajectories(tracking_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze object trajectories."""
    analysis = {
        'player_analysis': [],
        'ball_analysis': [],
        'interaction_analysis': []
    }
    
    # Analyze player trajectories
    for player_track in tracking_results['players']:
        trajectory = player_track['trajectory']
        
        # Calculate movement metrics
        total_distance = calculate_trajectory_distance(trajectory)
        average_speed = total_distance / player_track['duration'] if player_track['duration'] > 0 else 0
        
        # Calculate area coverage
        bboxes = [point['bbox'] for point in trajectory]
        coverage_area = calculate_coverage_area(bboxes)
        
        player_analysis = {
            'track_id': player_track['track_id'],
            'total_distance': total_distance,
            'average_speed': average_speed,
            'coverage_area': coverage_area,
            'movement_pattern': classify_movement_pattern(trajectory),
            'track_quality': assess_track_quality(trajectory)
        }
        
        analysis['player_analysis'].append(player_analysis)
    
    # Analyze ball trajectory
    for ball_track in tracking_results['ball']:
        trajectory = ball_track['trajectory']
        
        # Calculate ball movement
        total_distance = calculate_trajectory_distance(trajectory)
        average_speed = total_distance / ball_track['duration'] if ball_track['duration'] > 0 else 0
        
        # Analyze ball possession
        possession_analysis = analyze_ball_possession(trajectory, tracking_results['frame_tracks'])
        
        ball_analysis = {
            'track_id': ball_track['track_id'],
            'total_distance': total_distance,
            'average_speed': average_speed,
            'possession_analysis': possession_analysis,
            'track_quality': assess_track_quality(trajectory)
        }
        
        analysis['ball_analysis'].append(ball_analysis)
    
    # Analyze player-ball interactions
    interaction_analysis = analyze_player_ball_interactions(
        tracking_results['players'], 
        tracking_results['ball'], 
        tracking_results['frame_tracks']
    )
    analysis['interaction_analysis'] = interaction_analysis
    
    return analysis


def calculate_trajectory_distance(trajectory: List[Dict]) -> float:
    """Calculate total distance traveled in trajectory."""
    if len(trajectory) < 2:
        return 0.0
    
    total_distance = 0.0
    
    for i in range(1, len(trajectory)):
        prev_bbox = trajectory[i-1]['bbox']
        curr_bbox = trajectory[i]['bbox']
        
        # Calculate center points
        prev_center = [(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2]
        curr_center = [(curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2]
        
        # Calculate Euclidean distance
        distance = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
        total_distance += distance
    
    return total_distance


def calculate_coverage_area(bboxes: List[List[float]]) -> float:
    """Calculate area covered by trajectory."""
    if not bboxes:
        return 0.0
    
    # Find bounding box of all positions
    min_x = min(bbox[0] for bbox in bboxes)
    max_x = max(bbox[2] for bbox in bboxes)
    min_y = min(bbox[1] for bbox in bboxes)
    max_y = max(bbox[3] for bbox in bboxes)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return width * height


def classify_movement_pattern(trajectory: List[Dict]) -> str:
    """Classify player movement pattern."""
    if len(trajectory) < 3:
        return 'static'
    
    # Calculate movement direction changes
    direction_changes = 0
    prev_direction = None
    
    for i in range(1, len(trajectory)):
        prev_bbox = trajectory[i-1]['bbox']
        curr_bbox = trajectory[i]['bbox']
        
        prev_center = [(prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2]
        curr_center = [(curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2]
        
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        
        if abs(dx) > 5 or abs(dy) > 5:  # Minimum movement threshold
            current_direction = np.arctan2(dy, dx)
            
            if prev_direction is not None:
                angle_diff = abs(current_direction - prev_direction)
                if angle_diff > np.pi/4:  # 45 degrees
                    direction_changes += 1
            
            prev_direction = current_direction
    
    # Classify based on direction changes
    if direction_changes == 0:
        return 'linear'
    elif direction_changes < 3:
        return 'curved'
    else:
        return 'complex'


def assess_track_quality(trajectory: List[Dict]) -> Dict[str, Any]:
    """Assess the quality of a track."""
    if not trajectory:
        return {'score': 0, 'level': 'poor', 'issues': ['empty_trajectory']}
    
    issues = []
    score = 1.0
    
    # Check for gaps
    frame_indices = [point['frame_index'] for point in trajectory]
    gaps = [frame_indices[i+1] - frame_indices[i] for i in range(len(frame_indices)-1)]
    max_gap = max(gaps) if gaps else 0
    
    if max_gap > 10:
        issues.append('large_gaps')
        score -= 0.2
    
    # Check for confidence consistency
    confidences = [point['confidence'] for point in trajectory]
    avg_confidence = np.mean(confidences)
    
    if avg_confidence < 0.7:
        issues.append('low_confidence')
        score -= 0.3
    
    # Check trajectory length
    if len(trajectory) < 5:
        issues.append('short_trajectory')
        score -= 0.2
    
    # Determine quality level
    if score >= 0.8:
        level = 'excellent'
    elif score >= 0.6:
        level = 'good'
    elif score >= 0.4:
        level = 'fair'
    else:
        level = 'poor'
    
    return {
        'score': max(0, score),
        'level': level,
        'issues': issues,
        'avg_confidence': avg_confidence,
        'max_gap': max_gap
    }


def analyze_ball_possession(trajectory: List[Dict], frame_tracks: List[Dict]) -> Dict[str, Any]:
    """Analyze ball possession patterns."""
    # This is a simplified analysis - in a real implementation,
    # you would use more sophisticated algorithms
    return {
        'possession_changes': 0,  # Placeholder
        'possession_duration': trajectory[-1]['timestamp'] - trajectory[0]['timestamp'],
        'analysis_method': 'simplified'
    }


def analyze_player_ball_interactions(players: List[Dict], ball: List[Dict], frame_tracks: List[Dict]) -> List[Dict]:
    """Analyze interactions between players and ball."""
    # This is a placeholder for interaction analysis
    # In a real implementation, you would analyze proximity and movement patterns
    return []


def assess_tracking_quality(tracking_results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall tracking quality."""
    total_tracks = len(tracking_results['players']) + len(tracking_results['ball'])
    
    if total_tracks == 0:
        return {
            'total_tracks': 0,
            'tracking_success_rate': 0,
            'average_track_quality': 0,
            'quality_level': 'poor'
        }
    
    # Calculate average track quality
    all_tracks = tracking_results['players'] + tracking_results['ball']
    quality_scores = []
    
    for track in all_tracks:
        trajectory = track['trajectory']
        quality = assess_track_quality(trajectory)
        quality_scores.append(quality['score'])
    
    avg_quality = np.mean(quality_scores) if quality_scores else 0
    
    # Determine overall quality level
    if avg_quality >= 0.8:
        quality_level = 'excellent'
    elif avg_quality >= 0.6:
        quality_level = 'good'
    elif avg_quality >= 0.4:
        quality_level = 'fair'
    else:
        quality_level = 'poor'
    
    return {
        'total_tracks': total_tracks,
        'player_tracks': len(tracking_results['players']),
        'ball_tracks': len(tracking_results['ball']),
        'average_track_quality': avg_quality,
        'quality_level': quality_level,
        'tracking_coverage': len([f for f in tracking_results['frame_tracks'] if f['player_tracks'] or f['ball_track']]) / len(tracking_results['frame_tracks'])
    }


@test
def test_tracker():
    """Test the tracker with sample data."""
    # Create sample tracking data
    sample_trajectory = [
        {'frame_index': 0, 'timestamp': 0.0, 'bbox': [100, 100, 120, 140], 'confidence': 0.9},
        {'frame_index': 1, 'timestamp': 0.033, 'bbox': [105, 105, 125, 145], 'confidence': 0.85},
        {'frame_index': 2, 'timestamp': 0.067, 'bbox': [110, 110, 130, 150], 'confidence': 0.9}
    ]
    
    # Test trajectory analysis
    distance = calculate_trajectory_distance(sample_trajectory)
    coverage = calculate_coverage_area([point['bbox'] for point in sample_trajectory])
    quality = assess_track_quality(sample_trajectory)
    
    assert distance > 0
    assert coverage > 0
    assert 'score' in quality
    assert 'level' in quality
    
    print("✅ Tracker test passed") 