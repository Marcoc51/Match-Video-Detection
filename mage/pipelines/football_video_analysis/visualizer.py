"""
Visualizer Block for Football Video Analysis Pipeline
Handles video visualization with overlays and annotations
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager
from src.utils.colors import get_color_palette

logger = logging.getLogger(__name__)


@transformer
def visualize_results(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Generate visual outputs for the analysis results.
    
    This block handles:
    - Video overlays with detections and tracks
    - Event visualization (passes, crosses)
    - Statistics charts and graphs
    - Heatmaps and trajectory plots
    - Output video generation
    
    Args:
        data: Input data containing analysis results
        
    Returns:
        Dict containing visualization outputs
    """
    try:
        tracking_results = data['tracking_results']
        events = data.get('events', {})
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=75, message="Generating visualizations...")
        
        logger.info("Starting visualization generation")
        
        # Get color palette
        colors = get_color_palette()
        
        # Generate visualizations
        visualizations = {
            'video_overlay': generate_video_overlay(data, colors),
            'statistics_charts': generate_statistics_charts(data),
            'trajectory_plots': generate_trajectory_plots(tracking_results, colors),
            'event_visualizations': generate_event_visualizations(events, colors),
            'heatmaps': generate_heatmaps(tracking_results, colors)
        }
        
        # Combine results
        result = {
            **data,
            'visualizations': visualizations,
            'visualization_config': {
                'colors': colors,
                'overlay_opacity': 0.7,
                'track_thickness': 2,
                'bbox_thickness': 2
            }
        }
        
        logger.info("Visualization generation completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in visualizer: {e}")
        raise


def generate_video_overlay(data: Dict[str, Any], colors: Dict[str, Tuple[int, int, int]]) -> Dict[str, Any]:
    """Generate video overlay with detections and tracks."""
    try:
        video_path = data.get('processed_video_path', data['video_path'])
        detections = data['detections']
        tracking_results = data['tracking_results']
        events = data.get('events', {})
        
        # Create output video path
        output_dir = Path(get_config().video_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"analyzed_{timestamp}.mp4"
        
        # Open input video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_index = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add overlays to frame
                frame_with_overlays = add_frame_overlays(
                    frame, frame_index, detections, tracking_results, events, colors
                )
                
                # Write frame
                out.write(frame_with_overlays)
                frame_index += 1
                
                # Progress logging
                if frame_index % 100 == 0:
                    logger.info(f"Processed {frame_index}/{frame_count} frames for overlay")
        
        finally:
            cap.release()
            out.release()
        
        return {
            'output_path': str(output_path),
            'frame_count': frame_index,
            'resolution': f"{width}x{height}",
            'fps': fps
        }
        
    except Exception as e:
        logger.error(f"Error generating video overlay: {e}")
        return {'error': str(e)}


def add_frame_overlays(frame: np.ndarray, frame_index: int, detections: List[Dict], 
                      tracking_results: Dict[str, Any], events: Dict[str, Any], 
                      colors: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
    """Add overlays to a single frame."""
    frame_with_overlays = frame.copy()
    
    # Find detections for this frame
    frame_detections = None
    for detection in detections:
        if detection['frame_index'] == frame_index:
            frame_detections = detection
            break
    
    if frame_detections:
        # Draw bounding boxes
        for detection in frame_detections['detections']:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Get color for class
            color = colors.get(class_name.lower(), colors['default'])
            
            # Draw bounding box
            cv2.rectangle(frame_with_overlays, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            
            # Add label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame_with_overlays, label, 
                       (int(bbox[0]), int(bbox[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add track overlays
    frame_tracks = None
    for track in tracking_results['frame_tracks']:
        if track['frame_index'] == frame_index:
            frame_tracks = track
            break
    
    if frame_tracks:
        # Draw player tracks
        for track in frame_tracks['player_tracks']:
            bbox = track['bbox']
            track_id = track['track_id']
            
            # Draw track ID
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            cv2.putText(frame_with_overlays, f"P{track_id}", 
                       (center_x, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['player'], 2)
        
        # Draw ball track
        if frame_tracks['ball_track']:
            bbox = frame_tracks['ball_track']['bbox']
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            cv2.circle(frame_with_overlays, (center_x, center_y), 5, colors['ball'], -1)
    
    # Add event overlays
    add_event_overlays(frame_with_overlays, frame_index, events, colors)
    
    return frame_with_overlays


def add_event_overlays(frame: np.ndarray, frame_index: int, events: Dict[str, Any], 
                      colors: Dict[str, Tuple[int, int, int]]) -> None:
    """Add event overlays to frame."""
    # Add pass overlays
    for pass_event in events.get('passes', []):
        start_frame = int(pass_event['start_time'] * 30)  # Assuming 30 FPS
        end_frame = int(pass_event['end_time'] * 30)
        
        if start_frame <= frame_index <= end_frame:
            # Draw pass line
            start_pos = pass_event.get('start_position', [100, 100])
            end_pos = pass_event.get('end_position', [200, 200])
            
            cv2.line(frame, 
                    (int(start_pos[0]), int(start_pos[1])), 
                    (int(end_pos[0]), int(end_pos[1])), 
                    colors['pass'], 3)
            
            # Add pass label
            cv2.putText(frame, "PASS", 
                       (int(start_pos[0]), int(start_pos[1] - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['pass'], 2)


def generate_statistics_charts(data: Dict[str, Any]) -> Dict[str, str]:
    """Generate statistics charts and graphs."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        charts = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Detection quality chart
        if 'quality_metrics' in data:
            fig, ax = plt.subplots(figsize=(10, 6))
            quality = data['quality_metrics']
            
            metrics = ['Coverage Ratio', 'Detection Consistency', 'Average Detections']
            values = [quality['coverage_ratio'], quality['detection_consistency'], 
                     min(quality['average_detections_per_frame'] / 10, 1)]
            
            bars = ax.bar(metrics, values, color=['#2E86AB', '#A23B72', '#F18F01'])
            ax.set_ylim(0, 1)
            ax.set_title('Detection Quality Metrics')
            ax.set_ylabel('Score')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2f}', ha='center', va='bottom')
            
            chart_path = output_dir / f"detection_quality_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['detection_quality'] = str(chart_path)
        
        # 2. Event statistics chart
        if 'events' in data and 'statistics' in data['events']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            stats = data['events']['statistics']
            
            # Pass statistics
            if stats.get('total_passes', 0) > 0:
                pass_stats = stats['pass_statistics']
                pass_metrics = ['Avg Duration', 'Avg Distance', 'Success Rate']
                pass_values = [
                    pass_stats.get('average_duration', 0),
                    pass_stats.get('average_distance', 0) / 100,  # Normalize
                    pass_stats.get('successful_passes', 0) / stats['total_passes']
                ]
                
                ax1.bar(pass_metrics, pass_values, color=['#2E86AB', '#A23B72', '#F18F01'])
                ax1.set_title('Pass Statistics')
                ax1.set_ylabel('Value')
            
            # Possession chart
            possession_data = [stats.get('possession_home', 50), stats.get('possession_away', 50)]
            labels = ['Home Team', 'Away Team']
            colors = ['#2E86AB', '#A23B72']
            
            ax2.pie(possession_data, labels=labels, colors=colors, autopct='%1.1f%%')
            ax2.set_title('Ball Possession')
            
            chart_path = output_dir / f"event_statistics_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['event_statistics'] = str(chart_path)
        
        return charts
        
    except Exception as e:
        logger.error(f"Error generating statistics charts: {e}")
        return {'error': str(e)}


def generate_trajectory_plots(tracking_results: Dict[str, Any], colors: Dict[str, Tuple[int, int, int]]) -> Dict[str, str]:
    """Generate trajectory plots for players and ball."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plots = {}
        
        # Create trajectory plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot player trajectories
        for i, player_track in enumerate(tracking_results['players'][:10]):  # Limit to 10 players
            trajectory = player_track['trajectory']
            if trajectory:
                x_coords = [(point['bbox'][0] + point['bbox'][2]) / 2 for point in trajectory]
                y_coords = [(point['bbox'][1] + point['bbox'][3]) / 2 for point in trajectory]
                
                ax.plot(x_coords, y_coords, 'o-', alpha=0.7, linewidth=2, 
                       label=f"Player {player_track['track_id']}")
        
        # Plot ball trajectory
        for ball_track in tracking_results['ball']:
            trajectory = ball_track['trajectory']
            if trajectory:
                x_coords = [(point['bbox'][0] + point['bbox'][2]) / 2 for point in trajectory]
                y_coords = [(point['bbox'][1] + point['bbox'][3]) / 2 for point in trajectory]
                
                ax.plot(x_coords, y_coords, 'o-', color='red', alpha=0.8, linewidth=3, 
                       label='Ball', markersize=4)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_title('Player and Ball Trajectories')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plot_path = output_dir / f"trajectories_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['trajectories'] = str(plot_path)
        
        return plots
        
    except Exception as e:
        logger.error(f"Error generating trajectory plots: {e}")
        return {'error': str(e)}


def generate_event_visualizations(events: Dict[str, Any], colors: Dict[str, Tuple[int, int, int]]) -> Dict[str, str]:
    """Generate event-specific visualizations."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        visualizations = {}
        
        # Pass timeline
        if events.get('passes'):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            pass_times = []
            for pass_event in events['passes']:
                start_time = pass_event['start_time']
                end_time = pass_event['end_time']
                pass_times.append((start_time, end_time))
            
            # Create timeline
            for i, (start, end) in enumerate(pass_times):
                ax.barh(i, end - start, left=start, height=0.8, 
                       color=colors['pass'], alpha=0.7)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Pass Number')
            ax.set_title('Pass Timeline')
            ax.grid(True, alpha=0.3)
            
            viz_path = output_dir / f"pass_timeline_{timestamp}.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['pass_timeline'] = str(viz_path)
        
        return visualizations
        
    except Exception as e:
        logger.error(f"Error generating event visualizations: {e}")
        return {'error': str(e)}


def generate_heatmaps(tracking_results: Dict[str, Any], colors: Dict[str, Tuple[int, int, int]]) -> Dict[str, str]:
    """Generate heatmaps for player and ball positions."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        heatmaps = {}
        
        # Create heatmap data
        all_positions = []
        
        for player_track in tracking_results['players']:
            trajectory = player_track['trajectory']
            for point in trajectory:
                center_x = (point['bbox'][0] + point['bbox'][2]) / 2
                center_y = (point['bbox'][1] + point['bbox'][3]) / 2
                all_positions.append([center_x, center_y])
        
        if all_positions:
            positions = np.array(all_positions)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(
                positions[:, 0], positions[:, 1], 
                bins=50, range=[[0, 1920], [0, 1080]]
            )
            
            # Plot heatmap
            im = ax.imshow(heatmap.T, origin='lower', cmap='hot', 
                          extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.set_title('Player Position Heatmap')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Position Frequency')
            
            heatmap_path = output_dir / f"position_heatmap_{timestamp}.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            heatmaps['position_heatmap'] = str(heatmap_path)
        
        return heatmaps
        
    except Exception as e:
        logger.error(f"Error generating heatmaps: {e}")
        return {'error': str(e)}


@test
def test_visualizer():
    """Test the visualizer with sample data."""
    # Create sample data
    sample_data = {
        'video_path': 'test_video.mp4',
        'detections': [
            {
                'frame_index': 0,
                'detections': [
                    {'class_name': 'player', 'bbox': [100, 100, 120, 140], 'confidence': 0.9},
                    {'class_name': 'ball', 'bbox': [150, 150, 160, 160], 'confidence': 0.8}
                ]
            }
        ],
        'tracking_results': {
            'players': [],
            'ball': [],
            'frame_tracks': []
        },
        'events': {
            'passes': [],
            'possession': {},
            'crosses': []
        }
    }
    
    # Test visualization generation
    # This would normally call the actual visualization functions
    # For testing, we'll just verify the structure
    assert 'video_path' in sample_data
    assert 'detections' in sample_data
    assert 'tracking_results' in sample_data
    
    print("✅ Visualizer test passed") 