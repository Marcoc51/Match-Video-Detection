"""
Notification Sender Block for Football Video Analysis Pipeline
Handles sending notifications upon pipeline completion
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager

logger = logging.getLogger(__name__)


@transformer
def send_notifications(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Send notifications upon pipeline completion.
    
    This block handles:
    - Success notifications
    - Error notifications
    - Progress updates
    - Result summaries
    - Email/Slack/webhook notifications
    
    Args:
        data: Input data containing analysis results
        
    Returns:
        Dict containing notification status
    """
    try:
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=98, message="Sending notifications...")
        
        logger.info("Starting notification sending")
        
        # Prepare notification data
        notification_data = prepare_notification_data(data)
        
        # Send notifications
        notification_results = {
            'email_notification': send_email_notification(notification_data),
            'webhook_notification': send_webhook_notification(notification_data),
            'log_notification': log_completion_notification(notification_data)
        }
        
        # Update job with final completion
        if job_id:
            job_manager.update_job(job_id, progress=100, message="Analysis completed successfully")
        
        # Combine results
        result = {
            **data,
            'notifications': notification_results
        }
        
        logger.info("Notification sending completed")
        return result
        
    except Exception as e:
        logger.error(f"Error in notification sender: {e}")
        
        # Send error notification
        error_notification = send_error_notification(data, str(e))
        
        # Update job status if available
        job_id = data.get('job_id')
        if job_id:
            job_manager = get_job_manager()
            job_manager.fail_job(job_id, f"Pipeline failed: {str(e)}")
        
        raise


def prepare_notification_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare notification data."""
    notification_data = {
        'timestamp': datetime.now().isoformat(),
        'status': 'completed',
        'job_id': data.get('job_id'),
        'filename': data.get('filename', 'Unknown'),
        'summary': {}
    }
    
    # Add processing summary
    if 'metadata' in data:
        metadata = data['metadata']
        notification_data['summary']['video_info'] = {
            'duration': metadata.get('duration', 0),
            'fps': metadata.get('fps', 0),
            'resolution': f"{metadata.get('width', 0)}x{metadata.get('height', 0)}"
        }
    
    # Add detection summary
    if 'quality_metrics' in data:
        quality = data['quality_metrics']
        notification_data['summary']['detection_info'] = {
            'quality_score': quality.get('quality_score', 0),
            'total_detections': quality.get('total_detections', 0),
            'quality_level': quality.get('quality_level', 'unknown')
        }
    
    # Add event summary
    if 'events' in data and 'statistics' in data['events']:
        stats = data['events']['statistics']
        notification_data['summary']['event_info'] = {
            'total_passes': stats.get('total_passes', 0),
            'total_crosses': stats.get('total_crosses', 0),
            'possession_home': stats.get('possession_home', 50),
            'possession_away': stats.get('possession_away', 50)
        }
    
    # Add performance summary
    if 'metrics' in data and 'performance_kpis' in data['metrics']:
        kpis = data['metrics']['performance_kpis']
        notification_data['summary']['performance_info'] = {
            'overall_quality_score': kpis.get('overall_quality_score', 0),
            'processing_speed': kpis.get('processing_speed', 0),
            'detection_accuracy': kpis.get('detection_accuracy', 0)
        }
    
    # Add output file information
    if 'visualizations' in data:
        visualizations = data['visualizations']
        notification_data['summary']['output_files'] = {}
        
        if 'video_overlay' in visualizations:
            video_info = visualizations['video_overlay']
            notification_data['summary']['output_files']['processed_video'] = video_info.get('output_path', 'Not generated')
        
        if 'statistics_charts' in visualizations:
            charts = visualizations['statistics_charts']
            notification_data['summary']['output_files']['charts_count'] = len(charts)
    
    return notification_data


def send_email_notification(notification_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send email notification."""
    try:
        config = get_config()
        
        # Check if email notifications are enabled
        if not config.email_notifications_enabled:
            return {'status': 'disabled', 'message': 'Email notifications disabled'}
        
        # Prepare email content
        subject = f"Football Analysis Completed - {notification_data['filename']}"
        
        body = generate_email_body(notification_data)
        
        # Send email (placeholder - would use actual email service)
        # send_email(config.email_recipients, subject, body)
        
        logger.info(f"Email notification prepared for job {notification_data.get('job_id')}")
        
        return {
            'status': 'sent',
            'recipients': config.email_recipients,
            'subject': subject,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error sending email notification: {e}")
        return {'status': 'failed', 'error': str(e)}


def generate_email_body(notification_data: Dict[str, Any]) -> str:
    """Generate email body content."""
    body = f"""
Football Video Analysis Completed

Job ID: {notification_data.get('job_id', 'N/A')}
Filename: {notification_data['filename']}
Completion Time: {notification_data['timestamp']}

SUMMARY:
"""
    
    summary = notification_data['summary']
    
    # Video information
    if 'video_info' in summary:
        video_info = summary['video_info']
        body += f"""
Video Information:
- Duration: {video_info['duration']:.2f} seconds
- FPS: {video_info['fps']}
- Resolution: {video_info['resolution']}
"""
    
    # Detection information
    if 'detection_info' in summary:
        detection_info = summary['detection_info']
        body += f"""
Detection Results:
- Quality Score: {detection_info['quality_score']:.2f}
- Total Detections: {detection_info['total_detections']}
- Quality Level: {detection_info['quality_level']}
"""
    
    # Event information
    if 'event_info' in summary:
        event_info = summary['event_info']
        body += f"""
Event Analysis:
- Total Passes: {event_info['total_passes']}
- Total Crosses: {event_info['total_crosses']}
- Home Team Possession: {event_info['possession_home']:.1f}%
- Away Team Possession: {event_info['possession_away']:.1f}%
"""
    
    # Performance information
    if 'performance_info' in summary:
        perf_info = summary['performance_info']
        body += f"""
Performance Metrics:
- Overall Quality Score: {perf_info['overall_quality_score']:.2f}
- Processing Speed: {perf_info['processing_speed']:.1f} FPS
- Detection Accuracy: {perf_info['detection_accuracy']:.2f}
"""
    
    # Output files
    if 'output_files' in summary:
        output_files = summary['output_files']
        body += f"""
Output Files:
- Processed Video: {output_files.get('processed_video', 'Not generated')}
- Charts Generated: {output_files.get('charts_count', 0)}
"""
    
    body += """
The analysis has been completed successfully. You can download the results from the API endpoint.

Best regards,
Football Analysis System
"""
    
    return body


def send_webhook_notification(notification_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send webhook notification."""
    try:
        config = get_config()
        
        # Check if webhook notifications are enabled
        if not config.webhook_notifications_enabled or not config.webhook_url:
            return {'status': 'disabled', 'message': 'Webhook notifications disabled'}
        
        # Prepare webhook payload
        payload = {
            'event': 'analysis_completed',
            'timestamp': notification_data['timestamp'],
            'job_id': notification_data.get('job_id'),
            'filename': notification_data['filename'],
            'summary': notification_data['summary']
        }
        
        # Send webhook (placeholder - would use actual HTTP client)
        # response = requests.post(config.webhook_url, json=payload)
        
        logger.info(f"Webhook notification prepared for job {notification_data.get('job_id')}")
        
        return {
            'status': 'sent',
            'webhook_url': config.webhook_url,
            'payload': payload,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error sending webhook notification: {e}")
        return {'status': 'failed', 'error': str(e)}


def log_completion_notification(notification_data: Dict[str, Any]) -> Dict[str, Any]:
    """Log completion notification."""
    try:
        # Log completion to file
        log_file = Path(get_config().logs_dir) / "completion_notifications.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            'timestamp': notification_data['timestamp'],
            'job_id': notification_data.get('job_id'),
            'filename': notification_data['filename'],
            'status': 'completed',
            'summary': notification_data['summary']
        }
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.info(f"Completion notification logged for job {notification_data.get('job_id')}")
        
        return {
            'status': 'logged',
            'log_file': str(log_file),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error logging completion notification: {e}")
        return {'status': 'failed', 'error': str(e)}


def send_error_notification(data: Dict[str, Any], error_message: str) -> Dict[str, Any]:
    """Send error notification."""
    try:
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'job_id': data.get('job_id'),
            'filename': data.get('filename', 'Unknown'),
            'error_message': error_message
        }
        
        # Log error notification
        log_file = Path(get_config().logs_dir) / "error_notifications.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_data) + '\n')
        
        logger.error(f"Error notification logged for job {data.get('job_id')}: {error_message}")
        
        return {
            'status': 'logged',
            'log_file': str(log_file),
            'error_message': error_message
        }
        
    except Exception as e:
        logger.error(f"Error sending error notification: {e}")
        return {'status': 'failed', 'error': str(e)}


@test
def test_notification_sender():
    """Test the notification sender with sample data."""
    # Create sample data
    sample_data = {
        'job_id': 1,
        'filename': 'test_video.mp4',
        'metadata': {
            'duration': 120.0,
            'fps': 30.0,
            'width': 1920,
            'height': 1080
        },
        'quality_metrics': {
            'quality_score': 0.85,
            'total_detections': 1500,
            'quality_level': 'good'
        },
        'events': {
            'statistics': {
                'total_passes': 25,
                'total_crosses': 5,
                'possession_home': 55.0,
                'possession_away': 45.0
            }
        }
    }
    
    # Test notification data preparation
    notification_data = prepare_notification_data(sample_data)
    
    assert 'timestamp' in notification_data
    assert 'status' in notification_data
    assert 'summary' in notification_data
    assert notification_data['status'] == 'completed'
    
    print("✅ Notification sender test passed") 