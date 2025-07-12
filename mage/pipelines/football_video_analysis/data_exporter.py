"""
Data Exporter Block for Football Video Analysis Pipeline
Handles saving results, generating reports, and data export
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List
import json
import csv
import pandas as pd
from datetime import datetime
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager

logger = logging.getLogger(__name__)


@data_exporter
def export_results(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Export analysis results and generate reports.
    
    This block handles:
    - Saving analysis results to files
    - Generating comprehensive reports
    - Exporting data in various formats (JSON, CSV, Excel)
    - Creating summary statistics
    - Archiving results
    
    Args:
        data: Input data containing all analysis results
        
    Returns:
        Dict containing export information and file paths
    """
    try:
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=90, message="Exporting results...")
        
        logger.info("Starting data export")
        
        # Create timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export results
        export_info = {
            'json_export': export_to_json(data, timestamp),
            'csv_export': export_to_csv(data, timestamp),
            'excel_export': export_to_excel(data, timestamp),
            'summary_report': generate_summary_report(data, timestamp),
            'archive': create_archive(data, timestamp)
        }
        
        # Update job with final results
        if job_id:
            job_manager.update_job(job_id, progress=95, message="Finalizing results...")
            
            # Get output video path from visualizations
            video_path = None
            if 'visualizations' in data and 'video_overlay' in data['visualizations']:
                video_path = data['visualizations']['video_overlay'].get('output_path')
            
            # Complete the job
            job_manager.complete_job(
                job_id=job_id,
                result_path=Path(video_path) if video_path else None,
                statistics={
                    'export_info': export_info,
                    'analysis_summary': generate_analysis_summary(data)
                }
            )
        
        logger.info("Data export completed")
        return export_info
        
    except Exception as e:
        logger.error(f"Error in data exporter: {e}")
        raise


def export_to_json(data: Dict[str, Any], timestamp: str) -> Dict[str, str]:
    """Export results to JSON format."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON export
        export_data = {
            'metadata': {
                'timestamp': timestamp,
                'analysis_version': '1.0.0',
                'video_info': {
                    'filename': data.get('filename', 'unknown'),
                    'metadata': data.get('metadata', {}),
                    'features': data.get('features', {})
                }
            },
            'detection_results': {
                'quality_metrics': data.get('quality_metrics', {}),
                'detection_config': data.get('detection_config', {})
            },
            'tracking_results': {
                'tracking_quality': data.get('tracking_quality', {}),
                'trajectory_analysis': data.get('trajectory_analysis', {}),
                'tracking_config': data.get('tracking_config', {})
            },
            'events': data.get('events', {}),
            'visualizations': {
                'output_paths': data.get('visualizations', {})
            }
        }
        
        # Save to JSON file
        json_path = output_dir / f"analysis_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        return {
            'file_path': str(json_path),
            'file_size': json_path.stat().st_size,
            'format': 'json'
        }
        
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        return {'error': str(e)}


def export_to_csv(data: Dict[str, Any], timestamp: str) -> Dict[str, str]:
    """Export results to CSV format."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_exports = {}
        
        # Export detections
        if 'detections' in data:
            detections_data = []
            for frame_detection in data['detections']:
                for detection in frame_detection.get('detections', []):
                    detections_data.append({
                        'frame_index': frame_detection['frame_index'],
                        'timestamp': frame_detection['timestamp'],
                        'class_name': detection['class_name'],
                        'confidence': detection['confidence'],
                        'bbox_x1': detection['bbox'][0],
                        'bbox_y1': detection['bbox'][1],
                        'bbox_x2': detection['bbox'][2],
                        'bbox_y2': detection['bbox'][3]
                    })
            
            if detections_data:
                detections_df = pd.DataFrame(detections_data)
                detections_path = output_dir / f"detections_{timestamp}.csv"
                detections_df.to_csv(detections_path, index=False)
                csv_exports['detections'] = str(detections_path)
        
        # Export passes
        if 'events' in data and 'passes' in data['events']:
            passes_data = []
            for i, pass_event in enumerate(data['events']['passes']):
                passes_data.append({
                    'pass_id': i,
                    'start_time': pass_event.get('start_time'),
                    'end_time': pass_event.get('end_time'),
                    'duration': pass_event.get('end_time', 0) - pass_event.get('start_time', 0),
                    'distance': pass_event.get('distance', 0),
                    'successful': pass_event.get('successful', True)
                })
            
            if passes_data:
                passes_df = pd.DataFrame(passes_data)
                passes_path = output_dir / f"passes_{timestamp}.csv"
                passes_df.to_csv(passes_path, index=False)
                csv_exports['passes'] = str(passes_path)
        
        # Export tracking data
        if 'tracking_results' in data:
            tracking_data = []
            for player_track in data['tracking_results']['players']:
                for point in player_track['trajectory']:
                    tracking_data.append({
                        'track_id': player_track['track_id'],
                        'track_type': 'player',
                        'frame_index': point['frame_index'],
                        'timestamp': point['timestamp'],
                        'bbox_x1': point['bbox'][0],
                        'bbox_y1': point['bbox'][1],
                        'bbox_x2': point['bbox'][2],
                        'bbox_y2': point['bbox'][3],
                        'confidence': point.get('confidence', 0)
                    })
            
            if tracking_data:
                tracking_df = pd.DataFrame(tracking_data)
                tracking_path = output_dir / f"tracking_{timestamp}.csv"
                tracking_df.to_csv(tracking_path, index=False)
                csv_exports['tracking'] = str(tracking_path)
        
        return csv_exports
        
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return {'error': str(e)}


def export_to_excel(data: Dict[str, Any], timestamp: str) -> Dict[str, str]:
    """Export results to Excel format."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        excel_path = output_dir / f"analysis_report_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = generate_summary_data(data)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detections sheet
            if 'detections' in data:
                detections_data = []
                for frame_detection in data['detections']:
                    for detection in frame_detection.get('detections', []):
                        detections_data.append({
                            'Frame': frame_detection['frame_index'],
                            'Time': frame_detection['timestamp'],
                            'Class': detection['class_name'],
                            'Confidence': detection['confidence'],
                            'X1': detection['bbox'][0],
                            'Y1': detection['bbox'][1],
                            'X2': detection['bbox'][2],
                            'Y2': detection['bbox'][3]
                        })
                
                if detections_data:
                    detections_df = pd.DataFrame(detections_data)
                    detections_df.to_excel(writer, sheet_name='Detections', index=False)
            
            # Events sheet
            if 'events' in data:
                events_data = []
                for event_type, events_list in data['events'].items():
                    if isinstance(events_list, list):
                        for event in events_list:
                            events_data.append({
                                'Event_Type': event_type,
                                'Start_Time': event.get('start_time'),
                                'End_Time': event.get('end_time'),
                                'Duration': event.get('end_time', 0) - event.get('start_time', 0),
                                'Distance': event.get('distance', 0),
                                'Successful': event.get('successful', True)
                            })
                
                if events_data:
                    events_df = pd.DataFrame(events_data)
                    events_df.to_excel(writer, sheet_name='Events', index=False)
            
            # Statistics sheet
            if 'events' in data and 'statistics' in data['events']:
                stats = data['events']['statistics']
                stats_data = []
                for key, value in stats.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            stats_data.append({
                                'Category': key,
                                'Metric': sub_key,
                                'Value': sub_value
                            })
                    else:
                        stats_data.append({
                            'Category': 'General',
                            'Metric': key,
                            'Value': value
                        })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        return {
            'file_path': str(excel_path),
            'file_size': excel_path.stat().st_size,
            'format': 'excel',
            'sheets': ['Summary', 'Detections', 'Events', 'Statistics']
        }
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        return {'error': str(e)}


def generate_summary_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate summary data for Excel export."""
    summary_data = []
    
    # Video information
    summary_data.append({
        'Category': 'Video Information',
        'Metric': 'Filename',
        'Value': data.get('filename', 'Unknown')
    })
    
    if 'metadata' in data:
        metadata = data['metadata']
        summary_data.extend([
            {'Category': 'Video Information', 'Metric': 'Duration (seconds)', 'Value': metadata.get('duration', 0)},
            {'Category': 'Video Information', 'Metric': 'FPS', 'Value': metadata.get('fps', 0)},
            {'Category': 'Video Information', 'Metric': 'Resolution', 'Value': f"{metadata.get('width', 0)}x{metadata.get('height', 0)}"}
        ])
    
    # Detection information
    if 'quality_metrics' in data:
        quality = data['quality_metrics']
        summary_data.extend([
            {'Category': 'Detection Quality', 'Metric': 'Quality Score', 'Value': quality.get('quality_score', 0)},
            {'Category': 'Detection Quality', 'Metric': 'Quality Level', 'Value': quality.get('quality_level', 'Unknown')},
            {'Category': 'Detection Quality', 'Metric': 'Total Detections', 'Value': quality.get('total_detections', 0)}
        ])
    
    # Event information
    if 'events' in data and 'statistics' in data['events']:
        stats = data['events']['statistics']
        summary_data.extend([
            {'Category': 'Events', 'Metric': 'Total Passes', 'Value': stats.get('total_passes', 0)},
            {'Category': 'Events', 'Metric': 'Total Crosses', 'Value': stats.get('total_crosses', 0)},
            {'Category': 'Events', 'Metric': 'Home Possession (%)', 'Value': stats.get('possession_home', 0)},
            {'Category': 'Events', 'Metric': 'Away Possession (%)', 'Value': stats.get('possession_away', 0)}
        ])
    
    return summary_data


def generate_summary_report(data: Dict[str, Any], timestamp: str) -> Dict[str, str]:
    """Generate a comprehensive summary report."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / f"summary_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("FOOTBALL VIDEO ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Video information
            f.write("VIDEO INFORMATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Filename: {data.get('filename', 'Unknown')}\n")
            if 'metadata' in data:
                metadata = data['metadata']
                f.write(f"Duration: {metadata.get('duration', 0):.2f} seconds\n")
                f.write(f"FPS: {metadata.get('fps', 0)}\n")
                f.write(f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}\n")
            f.write("\n")
            
            # Detection results
            f.write("DETECTION RESULTS:\n")
            f.write("-" * 20 + "\n")
            if 'quality_metrics' in data:
                quality = data['quality_metrics']
                f.write(f"Quality Score: {quality.get('quality_score', 0):.2f}\n")
                f.write(f"Quality Level: {quality.get('quality_level', 'Unknown')}\n")
                f.write(f"Total Detections: {quality.get('total_detections', 0)}\n")
                f.write(f"Frames with Detections: {quality.get('frames_with_detections', 0)}\n")
            f.write("\n")
            
            # Event results
            f.write("EVENT ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            if 'events' in data and 'statistics' in data['events']:
                stats = data['events']['statistics']
                f.write(f"Total Passes: {stats.get('total_passes', 0)}\n")
                f.write(f"Total Crosses: {stats.get('total_crosses', 0)}\n")
                f.write(f"Home Team Possession: {stats.get('possession_home', 0):.1f}%\n")
                f.write(f"Away Team Possession: {stats.get('possession_away', 0):.1f}%\n")
                f.write(f"Possession Changes: {stats.get('possession_changes', 0)}\n")
            f.write("\n")
            
            # Processing information
            f.write("PROCESSING INFORMATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Analysis Timestamp: {timestamp}\n")
            f.write(f"Features Enabled: {data.get('features', {})}\n")
            f.write("\n")
            
            # Output files
            f.write("OUTPUT FILES:\n")
            f.write("-" * 20 + "\n")
            if 'visualizations' in data:
                viz = data['visualizations']
                if 'video_overlay' in viz:
                    f.write(f"Processed Video: {viz['video_overlay'].get('output_path', 'Not generated')}\n")
                if 'statistics_charts' in viz:
                    f.write(f"Statistics Charts: {len(viz['statistics_charts'])} files\n")
                if 'trajectory_plots' in viz:
                    f.write(f"Trajectory Plots: {len(viz['trajectory_plots'])} files\n")
        
        return {
            'file_path': str(report_path),
            'file_size': report_path.stat().st_size,
            'format': 'text'
        }
        
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")
        return {'error': str(e)}


def create_archive(data: Dict[str, Any], timestamp: str) -> Dict[str, str]:
    """Create an archive of all results."""
    try:
        output_dir = Path(get_config().results_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create archive directory
        archive_dir = output_dir / f"analysis_archive_{timestamp}"
        archive_dir.mkdir(exist_ok=True)
        
        # Copy all generated files
        files_copied = []
        
        # Copy visualization files
        if 'visualizations' in data:
            viz = data['visualizations']
            for viz_type, viz_data in viz.items():
                if isinstance(viz_data, dict) and 'file_path' in viz_data:
                    src_path = Path(viz_data['file_path'])
                    if src_path.exists():
                        dst_path = archive_dir / f"{viz_type}_{src_path.name}"
                        shutil.copy2(src_path, dst_path)
                        files_copied.append(str(dst_path))
        
        # Copy exported files
        export_patterns = [
            f"*{timestamp}.json",
            f"*{timestamp}.csv",
            f"*{timestamp}.xlsx",
            f"*{timestamp}.txt"
        ]
        
        for pattern in export_patterns:
            for file_path in output_dir.glob(pattern):
                dst_path = archive_dir / file_path.name
                shutil.copy2(file_path, dst_path)
                files_copied.append(str(dst_path))
        
        # Create archive info file
        archive_info = {
            'timestamp': timestamp,
            'files_included': files_copied,
            'analysis_summary': generate_analysis_summary(data)
        }
        
        info_path = archive_dir / "archive_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(archive_info, f, indent=2, ensure_ascii=False, default=str)
        
        return {
            'archive_path': str(archive_dir),
            'files_count': len(files_copied),
            'total_size': sum(Path(f).stat().st_size for f in files_copied if Path(f).exists())
        }
        
    except Exception as e:
        logger.error(f"Error creating archive: {e}")
        return {'error': str(e)}


def generate_analysis_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the analysis."""
    summary = {
        'video_info': {
            'filename': data.get('filename', 'Unknown'),
            'duration': data.get('metadata', {}).get('duration', 0),
            'fps': data.get('metadata', {}).get('fps', 0)
        },
        'detection_summary': {
            'quality_score': data.get('quality_metrics', {}).get('quality_score', 0),
            'total_detections': data.get('quality_metrics', {}).get('total_detections', 0)
        },
        'event_summary': {
            'total_passes': data.get('events', {}).get('statistics', {}).get('total_passes', 0),
            'total_crosses': data.get('events', {}).get('statistics', {}).get('total_crosses', 0),
            'possession_home': data.get('events', {}).get('statistics', {}).get('possession_home', 50),
            'possession_away': data.get('events', {}).get('statistics', {}).get('possession_away', 50)
        },
        'processing_info': {
            'features_enabled': data.get('features', {}),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    return summary


@test
def test_data_exporter():
    """Test the data exporter with sample data."""
    # Create sample data
    sample_data = {
        'filename': 'test_video.mp4',
        'metadata': {
            'duration': 120.0,
            'fps': 30.0,
            'width': 1920,
            'height': 1080
        },
        'quality_metrics': {
            'quality_score': 0.85,
            'total_detections': 1500
        },
        'events': {
            'statistics': {
                'total_passes': 25,
                'total_crosses': 5,
                'possession_home': 55.0,
                'possession_away': 45.0
            }
        },
        'features': {
            'passes': True,
            'possession': True,
            'crosses': False
        }
    }
    
    # Test summary generation
    summary = generate_analysis_summary(sample_data)
    
    assert 'video_info' in summary
    assert 'detection_summary' in summary
    assert 'event_summary' in summary
    
    print("✅ Data exporter test passed") 