"""
Metrics Calculator Block for Football Video Analysis Pipeline
Handles calculation of performance metrics and KPIs
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager

logger = logging.getLogger(__name__)


@transformer
def calculate_metrics(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Calculate performance metrics and KPIs.
    
    This block handles:
    - Processing performance metrics
    - Detection accuracy metrics
    - Event analysis metrics
    - System performance metrics
    - KPI calculations
    
    Args:
        data: Input data containing analysis results
        
    Returns:
        Dict containing calculated metrics
    """
    try:
        events = data.get('events', {})
        tracking_results = data.get('tracking_results', {})
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=65, message="Calculating metrics...")
        
        logger.info("Starting metrics calculation")
        
        # Calculate various metrics
        metrics = {
            'processing_metrics': calculate_processing_metrics(data),
            'detection_metrics': calculate_detection_metrics(data),
            'event_metrics': calculate_event_metrics(events),
            'tracking_metrics': calculate_tracking_metrics(tracking_results),
            'performance_kpis': calculate_performance_kpis(data),
            'timestamp': datetime.now().isoformat()
        }
        
        # Combine results
        result = {
            **data,
            'metrics': metrics
        }
        
        logger.info("Metrics calculation completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in metrics calculator: {e}")
        raise


def calculate_processing_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate processing performance metrics."""
    processing_metrics = {
        'total_processing_time': 0,
        'frames_per_second': 0,
        'processing_efficiency': 0,
        'memory_usage': 0,
        'cpu_usage': 0
    }
    
    # Calculate processing time (if available)
    if 'processing_info' in data:
        processing_info = data['processing_info']
        if 'processed_frames' in processing_info:
            total_frames = processing_info['processed_frames']
            # Estimate processing time (this would be more accurate with actual timing)
            estimated_time = total_frames / 30.0  # Assuming 30 FPS processing
            processing_metrics['total_processing_time'] = estimated_time
            processing_metrics['frames_per_second'] = total_frames / estimated_time if estimated_time > 0 else 0
    
    # Calculate efficiency
    if 'metadata' in data:
        metadata = data['metadata']
        if 'frame_count' in metadata and 'processing_info' in data:
            original_frames = metadata['frame_count']
            processed_frames = data['processing_info'].get('processed_frames', 0)
            processing_metrics['processing_efficiency'] = processed_frames / original_frames if original_frames > 0 else 0
    
    return processing_metrics


def calculate_detection_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate detection performance metrics."""
    detection_metrics = {
        'total_detections': 0,
        'average_confidence': 0,
        'detection_rate': 0,
        'false_positive_rate': 0,
        'miss_rate': 0
    }
    
    if 'detections' in data:
        detections = data['detections']
        
        # Calculate total detections
        total_detections = sum(d.get('detection_count', 0) for d in detections)
        detection_metrics['total_detections'] = total_detections
        
        # Calculate average confidence
        all_confidences = []
        for frame_detection in detections:
            for detection in frame_detection.get('detections', []):
                confidence = detection.get('confidence', 0)
                all_confidences.append(confidence)
        
        if all_confidences:
            detection_metrics['average_confidence'] = np.mean(all_confidences)
        
        # Calculate detection rate
        total_frames = len(detections)
        frames_with_detections = sum(1 for d in detections if d.get('detection_count', 0) > 0)
        detection_metrics['detection_rate'] = frames_with_detections / total_frames if total_frames > 0 else 0
    
    return detection_metrics


def calculate_event_metrics(events: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate event analysis metrics."""
    event_metrics = {
        'total_events': 0,
        'pass_accuracy': 0,
        'possession_balance': 0,
        'event_density': 0,
        'success_rate': 0
    }
    
    # Calculate total events
    total_passes = len(events.get('passes', []))
    total_crosses = len(events.get('crosses', []))
    event_metrics['total_events'] = total_passes + total_crosses
    
    # Calculate pass accuracy
    if total_passes > 0:
        successful_passes = sum(1 for p in events['passes'] if p.get('successful', True))
        event_metrics['pass_accuracy'] = successful_passes / total_passes
    
    # Calculate possession balance
    possession = events.get('possession', {})
    home_possession = possession.get('final_home', 50)
    away_possession = possession.get('final_away', 50)
    event_metrics['possession_balance'] = abs(home_possession - away_possession)
    
    # Calculate event density (events per minute)
    if 'metadata' in events:
        duration_minutes = events['metadata'].get('duration', 0) / 60
        event_metrics['event_density'] = event_metrics['total_events'] / duration_minutes if duration_minutes > 0 else 0
    
    # Calculate overall success rate
    if event_metrics['total_events'] > 0:
        successful_events = sum(1 for p in events.get('passes', []) if p.get('successful', True))
        successful_events += sum(1 for c in events.get('crosses', []) if c.get('successful', True))
        event_metrics['success_rate'] = successful_events / event_metrics['total_events']
    
    return event_metrics


def calculate_tracking_metrics(tracking_results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate tracking performance metrics."""
    tracking_metrics = {
        'total_tracks': 0,
        'average_track_length': 0,
        'track_consistency': 0,
        'tracking_accuracy': 0,
        'track_fragmentation': 0
    }
    
    # Calculate total tracks
    player_tracks = tracking_results.get('players', [])
    ball_tracks = tracking_results.get('ball', [])
    tracking_metrics['total_tracks'] = len(player_tracks) + len(ball_tracks)
    
    # Calculate average track length
    all_tracks = player_tracks + ball_tracks
    if all_tracks:
        track_lengths = [len(track.get('trajectory', [])) for track in all_tracks]
        tracking_metrics['average_track_length'] = np.mean(track_lengths)
        
        # Calculate track consistency
        length_std = np.std(track_lengths)
        length_mean = np.mean(track_lengths)
        tracking_metrics['track_consistency'] = 1 - (length_std / (length_mean + 1e-6))
        tracking_metrics['track_consistency'] = max(0, min(1, tracking_metrics['track_consistency']))
    
    # Calculate tracking accuracy (simplified)
    if all_tracks:
        total_points = sum(len(track.get('trajectory', [])) for track in all_tracks)
        if total_points > 0:
            # Estimate accuracy based on track quality
            quality_scores = []
            for track in all_tracks:
                trajectory = track.get('trajectory', [])
                if trajectory:
                    confidences = [point.get('confidence', 0.5) for point in trajectory]
                    quality_scores.append(np.mean(confidences))
            
            if quality_scores:
                tracking_metrics['tracking_accuracy'] = np.mean(quality_scores)
    
    return tracking_metrics


def calculate_performance_kpis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate key performance indicators."""
    kpis = {
        'overall_quality_score': 0,
        'processing_speed': 0,
        'detection_accuracy': 0,
        'event_detection_rate': 0,
        'system_reliability': 0
    }
    
    # Overall quality score
    if 'quality_metrics' in data:
        quality = data['quality_metrics']
        kpis['overall_quality_score'] = quality.get('quality_score', 0)
    
    # Processing speed (frames per second)
    if 'processing_info' in data:
        processing_info = data['processing_info']
        if 'processed_frames' in processing_info:
            total_frames = processing_info['processed_frames']
            # Estimate processing time
            estimated_time = total_frames / 30.0
            kpis['processing_speed'] = total_frames / estimated_time if estimated_time > 0 else 0
    
    # Detection accuracy
    if 'detections' in data:
        detections = data['detections']
        all_confidences = []
        for frame_detection in detections:
            for detection in frame_detection.get('detections', []):
                all_confidences.append(detection.get('confidence', 0))
        
        if all_confidences:
            kpis['detection_accuracy'] = np.mean(all_confidences)
    
    # Event detection rate
    if 'events' in data:
        events = data['events']
        total_events = len(events.get('passes', [])) + len(events.get('crosses', []))
        
        if 'metadata' in data:
            duration_minutes = data['metadata'].get('duration', 0) / 60
            kpis['event_detection_rate'] = total_events / duration_minutes if duration_minutes > 0 else 0
    
    # System reliability (based on error rates and consistency)
    reliability_factors = []
    
    # Detection consistency
    if 'quality_metrics' in data:
        quality = data['quality_metrics']
        reliability_factors.append(quality.get('detection_consistency', 0))
    
    # Processing efficiency
    if 'processing_info' in data and 'metadata' in data:
        processing_info = data['processing_info']
        metadata = data['metadata']
        if 'frame_count' in metadata and 'processed_frames' in processing_info:
            efficiency = processing_info['processed_frames'] / metadata['frame_count']
            reliability_factors.append(efficiency)
    
    # Track quality
    if 'tracking_results' in data:
        tracking_results = data['tracking_results']
        all_tracks = tracking_results.get('players', []) + tracking_results.get('ball', [])
        if all_tracks:
            track_qualities = []
            for track in all_tracks:
                trajectory = track.get('trajectory', [])
                if trajectory:
                    confidences = [point.get('confidence', 0.5) for point in trajectory]
                    track_qualities.append(np.mean(confidences))
            
            if track_qualities:
                reliability_factors.append(np.mean(track_qualities))
    
    if reliability_factors:
        kpis['system_reliability'] = np.mean(reliability_factors)
    
    return kpis


@test
def test_metrics_calculator():
    """Test the metrics calculator with sample data."""
    # Create sample data
    sample_data = {
        'detections': [
            {
                'frame_index': 0,
                'detection_count': 2,
                'detections': [
                    {'confidence': 0.9, 'bbox': [100, 100, 120, 140]},
                    {'confidence': 0.8, 'bbox': [150, 150, 160, 160]}
                ]
            }
        ],
        'events': {
            'passes': [
                {'successful': True, 'start_time': 0, 'end_time': 1},
                {'successful': False, 'start_time': 2, 'end_time': 3}
            ],
            'possession': {'final_home': 60, 'final_away': 40}
        },
        'tracking_results': {
            'players': [
                {
                    'track_id': 1,
                    'trajectory': [
                        {'confidence': 0.9, 'bbox': [100, 100, 120, 140]},
                        {'confidence': 0.85, 'bbox': [105, 105, 125, 145]}
                    ]
                }
            ],
            'ball': []
        }
    }
    
    # Test metrics calculation
    detection_metrics = calculate_detection_metrics(sample_data)
    event_metrics = calculate_event_metrics(sample_data['events'])
    tracking_metrics = calculate_tracking_metrics(sample_data['tracking_results'])
    
    assert 'total_detections' in detection_metrics
    assert 'pass_accuracy' in event_metrics
    assert 'total_tracks' in tracking_metrics
    
    print("✅ Metrics calculator test passed") 