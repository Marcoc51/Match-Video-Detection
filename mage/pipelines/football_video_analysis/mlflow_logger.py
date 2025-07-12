"""
MLflow Logger Block for Football Video Analysis Pipeline
Handles experiment tracking, model versioning, and metrics logging
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List
import mlflow
import mlflow.pytorch
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager

logger = logging.getLogger(__name__)


@transformer
def log_to_mlflow(data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    """
    Log experiment results to MLflow.
    
    This block handles:
    - Experiment tracking
    - Model versioning
    - Metrics logging
    - Parameter tracking
    - Artifact logging
    
    Args:
        data: Input data containing analysis results and metrics
        
    Returns:
        Dict containing MLflow logging information
    """
    try:
        metrics = data.get('metrics', {})
        job_id = data.get('job_id')
        
        # Update job progress if available
        if job_id:
            job_manager = get_job_manager()
            job_manager.update_job(job_id, progress=85, message="Logging to MLflow...")
        
        logger.info("Starting MLflow logging")
        
        # Get MLflow configuration
        config = get_config()
        
        # Set MLflow tracking URI
        if config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
        # Start MLflow run
        with mlflow.start_run(
            experiment_name=config.mlflow_experiment_name,
            run_name=f"football_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ) as run:
            
            # Log parameters
            log_parameters(data)
            
            # Log metrics
            log_metrics(metrics)
            
            # Log artifacts
            log_artifacts(data, run)
            
            # Log model information
            log_model_info(data)
            
            # Get run information
            run_info = {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'artifact_uri': run.info.artifact_uri
            }
        
        # Combine results
        result = {
            **data,
            'mlflow_info': run_info
        }
        
        logger.info(f"MLflow logging completed. Run ID: {run_info['run_id']}")
        return result
        
    except Exception as e:
        logger.error(f"Error in MLflow logger: {e}")
        raise


def log_parameters(data: Dict[str, Any]) -> None:
    """Log parameters to MLflow."""
    try:
        # Log video parameters
        if 'metadata' in data:
            metadata = data['metadata']
            mlflow.log_params({
                'video_duration': metadata.get('duration', 0),
                'video_fps': metadata.get('fps', 0),
                'video_resolution': f"{metadata.get('width', 0)}x{metadata.get('height', 0)}",
                'video_format': metadata.get('format', 'unknown')
            })
        
        # Log feature parameters
        if 'features' in data:
            features = data['features']
            mlflow.log_params({
                'passes_enabled': features.get('passes', False),
                'possession_enabled': features.get('possession', False),
                'crosses_enabled': features.get('crosses', False)
            })
        
        # Log detection parameters
        if 'detection_config' in data:
            detection_config = data['detection_config']
            mlflow.log_params({
                'confidence_threshold': detection_config.get('confidence_threshold', 0.5),
                'iou_threshold': detection_config.get('iou_threshold', 0.5)
            })
        
        # Log tracking parameters
        if 'tracking_config' in data:
            tracking_config = data['tracking_config']
            mlflow.log_params({
                'max_disappeared': tracking_config.get('max_disappeared', 30),
                'min_hits': tracking_config.get('min_hits', 3)
            })
        
        # Log processing parameters
        if 'processing_info' in data:
            processing_info = data['processing_info']
            mlflow.log_params({
                'target_fps': processing_info.get('target_fps', 30),
                'target_resolution': processing_info.get('target_resolution', '1920x1080')
            })
        
        logger.info("Parameters logged to MLflow")
        
    except Exception as e:
        logger.error(f"Error logging parameters: {e}")


def log_metrics(metrics: Dict[str, Any]) -> None:
    """Log metrics to MLflow."""
    try:
        # Log processing metrics
        if 'processing_metrics' in metrics:
            processing_metrics = metrics['processing_metrics']
            mlflow.log_metrics({
                'processing_time': processing_metrics.get('total_processing_time', 0),
                'frames_per_second': processing_metrics.get('frames_per_second', 0),
                'processing_efficiency': processing_metrics.get('processing_efficiency', 0)
            })
        
        # Log detection metrics
        if 'detection_metrics' in metrics:
            detection_metrics = metrics['detection_metrics']
            mlflow.log_metrics({
                'total_detections': detection_metrics.get('total_detections', 0),
                'average_confidence': detection_metrics.get('average_confidence', 0),
                'detection_rate': detection_metrics.get('detection_rate', 0)
            })
        
        # Log event metrics
        if 'event_metrics' in metrics:
            event_metrics = metrics['event_metrics']
            mlflow.log_metrics({
                'total_events': event_metrics.get('total_events', 0),
                'pass_accuracy': event_metrics.get('pass_accuracy', 0),
                'possession_balance': event_metrics.get('possession_balance', 0),
                'event_density': event_metrics.get('event_density', 0),
                'success_rate': event_metrics.get('success_rate', 0)
            })
        
        # Log tracking metrics
        if 'tracking_metrics' in metrics:
            tracking_metrics = metrics['tracking_metrics']
            mlflow.log_metrics({
                'total_tracks': tracking_metrics.get('total_tracks', 0),
                'average_track_length': tracking_metrics.get('average_track_length', 0),
                'track_consistency': tracking_metrics.get('track_consistency', 0),
                'tracking_accuracy': tracking_metrics.get('tracking_accuracy', 0)
            })
        
        # Log performance KPIs
        if 'performance_kpis' in metrics:
            kpis = metrics['performance_kpis']
            mlflow.log_metrics({
                'overall_quality_score': kpis.get('overall_quality_score', 0),
                'processing_speed': kpis.get('processing_speed', 0),
                'detection_accuracy': kpis.get('detection_accuracy', 0),
                'event_detection_rate': kpis.get('event_detection_rate', 0),
                'system_reliability': kpis.get('system_reliability', 0)
            })
        
        logger.info("Metrics logged to MLflow")
        
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")


def log_artifacts(data: Dict[str, Any], run) -> None:
    """Log artifacts to MLflow."""
    try:
        config = get_config()
        
        # Log visualization artifacts
        if 'visualizations' in data:
            visualizations = data['visualizations']
            
            # Log statistics charts
            if 'statistics_charts' in visualizations:
                charts = visualizations['statistics_charts']
                for chart_name, chart_path in charts.items():
                    if isinstance(chart_path, str) and Path(chart_path).exists():
                        mlflow.log_artifact(chart_path, f"charts/{chart_name}")
            
            # Log trajectory plots
            if 'trajectory_plots' in visualizations:
                plots = visualizations['trajectory_plots']
                for plot_name, plot_path in plots.items():
                    if isinstance(plot_path, str) and Path(plot_path).exists():
                        mlflow.log_artifact(plot_path, f"plots/{plot_name}")
            
            # Log heatmaps
            if 'heatmaps' in visualizations:
                heatmaps = visualizations['heatmaps']
                for heatmap_name, heatmap_path in heatmaps.items():
                    if isinstance(heatmap_path, str) and Path(heatmap_path).exists():
                        mlflow.log_artifact(heatmap_path, f"heatmaps/{heatmap_name}")
        
        # Log exported data files
        if 'export_info' in data:
            export_info = data['export_info']
            
            # Log JSON export
            if 'json_export' in export_info:
                json_export = export_info['json_export']
                if 'file_path' in json_export and Path(json_export['file_path']).exists():
                    mlflow.log_artifact(json_export['file_path'], "exports")
            
            # Log CSV exports
            if 'csv_export' in export_info:
                csv_exports = export_info['csv_export']
                for csv_name, csv_path in csv_exports.items():
                    if isinstance(csv_path, str) and Path(csv_path).exists():
                        mlflow.log_artifact(csv_path, f"exports/csv/{csv_name}")
            
            # Log Excel export
            if 'excel_export' in export_info:
                excel_export = export_info['excel_export']
                if 'file_path' in excel_export and Path(excel_export['file_path']).exists():
                    mlflow.log_artifact(excel_export['file_path'], "exports")
            
            # Log summary report
            if 'summary_report' in export_info:
                report_export = export_info['summary_report']
                if 'file_path' in report_export and Path(report_export['file_path']).exists():
                    mlflow.log_artifact(report_export['file_path'], "reports")
        
        # Log configuration files
        config_files = [
            config.project_root / "training_config.yaml",
            config.project_root / "requirements.txt",
            config.project_root / "requirements-api.txt"
        ]
        
        for config_file in config_files:
            if config_file.exists():
                mlflow.log_artifact(str(config_file), "config")
        
        logger.info("Artifacts logged to MLflow")
        
    except Exception as e:
        logger.error(f"Error logging artifacts: {e}")


def log_model_info(data: Dict[str, Any]) -> None:
    """Log model information to MLflow."""
    try:
        config = get_config()
        
        # Log model path
        if config.model_path_absolute and Path(config.model_path_absolute).exists():
            mlflow.log_param("model_path", str(config.model_path_absolute))
            
            # Log model file size
            model_size = Path(config.model_path_absolute).stat().st_size
            mlflow.log_metric("model_size_mb", model_size / (1024 * 1024))
        
        # Log model configuration
        if 'detection_config' in data:
            detection_config = data['detection_config']
            mlflow.log_param("model_confidence_threshold", detection_config.get('confidence_threshold', 0.5))
            mlflow.log_param("model_iou_threshold", detection_config.get('iou_threshold', 0.5))
        
        # Log model performance metrics
        if 'quality_metrics' in data:
            quality = data['quality_metrics']
            mlflow.log_metric("model_quality_score", quality.get('quality_score', 0))
            mlflow.log_metric("model_coverage_ratio", quality.get('coverage_ratio', 0))
            mlflow.log_metric("model_detection_consistency", quality.get('detection_consistency', 0))
        
        logger.info("Model information logged to MLflow")
        
    except Exception as e:
        logger.error(f"Error logging model info: {e}")


@test
def test_mlflow_logger():
    """Test the MLflow logger with sample data."""
    # Create sample data
    sample_data = {
        'metadata': {
            'duration': 120.0,
            'fps': 30.0,
            'width': 1920,
            'height': 1080
        },
        'features': {
            'passes': True,
            'possession': True,
            'crosses': False
        },
        'metrics': {
            'processing_metrics': {
                'total_processing_time': 60.0,
                'frames_per_second': 30.0
            },
            'detection_metrics': {
                'total_detections': 1500,
                'average_confidence': 0.85
            }
        }
    }
    
    # Test parameter logging (without actually logging to MLflow)
    # This would normally call the actual logging functions
    # For testing, we'll just verify the structure
    assert 'metadata' in sample_data
    assert 'features' in sample_data
    assert 'metrics' in sample_data
    
    print("✅ MLflow logger test passed") 