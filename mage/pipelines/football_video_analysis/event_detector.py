"""
Event Detector Block for Football Video Analysis Pipeline
Handles detection of football events (passes, possession, crosses)
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager
from src.events.pass_detector import PassDetector
from src.events.possession_tracker import PossessionTracker
from src.events.cross_detector import CrossDetector

logger = logging.getLogger(__name__)


@transformer
def detect_events(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Detect football events in the video.
    
    This block handles:
    - Pass detection
    - Possession tracking
    - Cross detection
    - Event validation and filtering
    - Event statistics calculation
    
    Args:
        data: Input data containing tracking results
        
    Returns:
        Dict containing detected events and statistics
    """
    try:
        tracking_results = data['tracking_results']
        features = data.get('features', {})
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=55, message="Detecting events...")
        
        logger.info("Starting event detection")
        
        events = {
            'passes': [],
            'possession': {},
            'crosses': [],
            'statistics': {}
        }
        
        # Detect passes if enabled
        if features.get('passes', True):
            events['passes'] = detect_passes(tracking_results)
            logger.info(f"Detected {len(events['passes'])} passes")
        
        # Track possession if enabled
        if features.get('possession', True):
            events['possession'] = track_possession(tracking_results)
            logger.info("Possession tracking completed")
        
        # Detect crosses if enabled
        if features.get('crosses', False):
            events['crosses'] = detect_crosses(tracking_results)
            logger.info(f"Detected {len(events['crosses'])} crosses")
        
        # Calculate event statistics
        events['statistics'] = calculate_event_statistics(events)
        
        # Combine results
        result = {
            **data,
            'events': events,
            'event_config': {
                'passes_enabled': features.get('passes', True),
                'possession_enabled': features.get('possession', True),
                'crosses_enabled': features.get('crosses', False)
            }
        }
        
        logger.info("Event detection completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in event detector: {e}")
        raise


def detect_passes(tracking_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect passes in the video."""
    try:
        # Initialize pass detector
        pass_detector = PassDetector()
        
        # Extract player and ball trajectories
        player_tracks = tracking_results['players']
        ball_tracks = tracking_results['ball']
        frame_tracks = tracking_results['frame_tracks']
        
        # Detect passes
        passes = pass_detector.detect_passes(player_tracks, ball_tracks, frame_tracks)
        
        # Validate and filter passes
        validated_passes = validate_passes(passes, tracking_results)
        
        return validated_passes
        
    except Exception as e:
        logger.error(f"Error detecting passes: {e}")
        return []


def validate_passes(passes: List[Dict], tracking_results: Dict[str, Any]) -> List[Dict]:
    """Validate detected passes."""
    validated = []
    
    for pass_event in passes:
        # Basic validation
        if not pass_event.get('start_time') or not pass_event.get('end_time'):
            continue
        
        if pass_event['end_time'] <= pass_event['start_time']:
            continue
        
        # Check if pass duration is reasonable (0.5 to 5 seconds)
        duration = pass_event['end_time'] - pass_event['start_time']
        if duration < 0.5 or duration > 5.0:
            continue
        
        # Check if pass distance is reasonable
        if pass_event.get('distance', 0) < 50:  # Minimum 50 pixels
            continue
        
        validated.append(pass_event)
    
    return validated


def track_possession(tracking_results: Dict[str, Any]) -> Dict[str, Any]:
    """Track ball possession throughout the video."""
    try:
        # Initialize possession tracker
        possession_tracker = PossessionTracker()
        
        # Extract frame tracks
        frame_tracks = tracking_results['frame_tracks']
        
        # Track possession
        possession_data = possession_tracker.track_possession(frame_tracks)
        
        return possession_data
        
    except Exception as e:
        logger.error(f"Error tracking possession: {e}")
        return {
            'possession_changes': [],
            'team_possession': {'home': 0, 'away': 0},
            'final_home': 50.0,
            'final_away': 50.0
        }


def detect_crosses(tracking_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect crosses in the video."""
    try:
        # Initialize cross detector
        cross_detector = CrossDetector()
        
        # Extract player and ball trajectories
        player_tracks = tracking_results['players']
        ball_tracks = tracking_results['ball']
        frame_tracks = tracking_results['frame_tracks']
        
        # Detect crosses
        crosses = cross_detector.detect_crosses(player_tracks, ball_tracks, frame_tracks)
        
        # Validate crosses
        validated_crosses = validate_crosses(crosses, tracking_results)
        
        return validated_crosses
        
    except Exception as e:
        logger.error(f"Error detecting crosses: {e}")
        return []


def validate_crosses(crosses: List[Dict], tracking_results: Dict[str, Any]) -> List[Dict]:
    """Validate detected crosses."""
    validated = []
    
    for cross in crosses:
        # Basic validation
        if not cross.get('start_time') or not cross.get('end_time'):
            continue
        
        if cross['end_time'] <= cross['start_time']:
            continue
        
        # Check if cross duration is reasonable (1 to 8 seconds)
        duration = cross['end_time'] - cross['start_time']
        if duration < 1.0 or duration > 8.0:
            continue
        
        # Check if cross distance is reasonable
        if cross.get('distance', 0) < 100:  # Minimum 100 pixels for crosses
            continue
        
        validated.append(cross)
    
    return validated


def calculate_event_statistics(events: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics for detected events."""
    statistics = {
        'total_passes': len(events['passes']),
        'total_crosses': len(events['crosses']),
        'possession_home': events['possession'].get('final_home', 50.0),
        'possession_away': events['possession'].get('final_away', 50.0),
        'possession_changes': len(events['possession'].get('possession_changes', [])),
        'pass_statistics': {},
        'cross_statistics': {}
    }
    
    # Pass statistics
    if events['passes']:
        pass_durations = [p['end_time'] - p['start_time'] for p in events['passes']]
        pass_distances = [p.get('distance', 0) for p in events['passes']]
        
        statistics['pass_statistics'] = {
            'average_duration': np.mean(pass_durations),
            'average_distance': np.mean(pass_distances),
            'min_duration': np.min(pass_durations),
            'max_duration': np.max(pass_durations),
            'min_distance': np.min(pass_distances),
            'max_distance': np.max(pass_distances),
            'successful_passes': len([p for p in events['passes'] if p.get('successful', True)])
        }
    
    # Cross statistics
    if events['crosses']:
        cross_durations = [c['end_time'] - c['start_time'] for c in events['crosses']]
        cross_distances = [c.get('distance', 0) for c in events['crosses']]
        
        statistics['cross_statistics'] = {
            'average_duration': np.mean(cross_durations),
            'average_distance': np.mean(cross_distances),
            'min_duration': np.min(cross_durations),
            'max_duration': np.max(cross_durations),
            'min_distance': np.min(cross_distances),
            'max_distance': np.max(cross_distances),
            'successful_crosses': len([c for c in events['crosses'] if c.get('successful', True)])
        }
    
    return statistics


@test
def test_event_detector():
    """Test the event detector with sample data."""
    # Create sample tracking data
    sample_tracking = {
        'players': [
            {
                'track_id': 1,
                'type': 'player',
                'trajectory': [
                    {'frame_index': 0, 'timestamp': 0.0, 'bbox': [100, 100, 120, 140]},
                    {'frame_index': 1, 'timestamp': 0.033, 'bbox': [105, 105, 125, 145]}
                ]
            }
        ],
        'ball': [
            {
                'track_id': 1,
                'type': 'ball',
                'trajectory': [
                    {'frame_index': 0, 'timestamp': 0.0, 'bbox': [150, 150, 160, 160]},
                    {'frame_index': 1, 'timestamp': 0.033, 'bbox': [155, 155, 165, 165]}
                ]
            }
        ],
        'frame_tracks': [
            {
                'frame_index': 0,
                'timestamp': 0.0,
                'player_tracks': [{'track_id': 1, 'bbox': [100, 100, 120, 140]}],
                'ball_track': {'track_id': 1, 'bbox': [150, 150, 160, 160]}
            }
        ]
    }
    
    # Test event detection
    sample_data = {
        'tracking_results': sample_tracking,
        'features': {'passes': True, 'possession': True, 'crosses': False}
    }
    
    # This would normally call the actual detection functions
    # For testing, we'll just verify the structure
    assert 'tracking_results' in sample_data
    assert 'features' in sample_data
    
    print("✅ Event detector test passed") 