"""
Metrics Collector for Model Monitoring

Collects various metrics for monitoring:
- Model performance metrics
- System resource metrics
- Processing metrics
- Quality metrics
"""

import os
import sys
import logging
import psutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects various metrics for model monitoring.
    
    Collects:
    - Model performance metrics
    - System resource metrics
    - Processing metrics
    - Quality metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metrics collector.
        
        Args:
            config: Metrics collection configuration
        """
        self.config = config
        self.metrics_history: Dict[str, List[float]] = {}
        self.collection_start_time = datetime.now()
        
        logger.info("Metrics collector initialized")
    
    def collect_all_metrics(self) -> Dict[str, float]:
        """
        Collect all available metrics.
        
        Returns:
            Dictionary of current metric values
        """
        metrics = {}
        
        # Collect system metrics
        metrics.update(self._collect_system_metrics())
        
        # Collect model metrics
        metrics.update(self._collect_model_metrics())
        
        # Collect processing metrics
        metrics.update(self._collect_processing_metrics())
        
        # Collect quality metrics
        metrics.update(self._collect_quality_metrics())
        
        # Add timestamp
        metrics['timestamp'] = time.time()
        
        return metrics
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent / 100.0
            
            # Network I/O (if available)
            try:
                network = psutil.net_io_counters()
                network_bytes_sent = network.bytes_sent
                network_bytes_recv = network.bytes_recv
            except:
                network_bytes_sent = 0
                network_bytes_recv = 0
            
            # GPU usage (if available)
            gpu_percent = self._get_gpu_usage()
            
            return {
                'system_cpu_usage': cpu_percent / 100.0,
                'system_memory_usage': memory_percent,
                'system_disk_usage': disk_percent,
                'system_network_sent': network_bytes_sent,
                'system_network_recv': network_bytes_recv,
                'system_gpu_usage': gpu_percent,
                'system_uptime': (datetime.now() - self.collection_start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {
                'system_cpu_usage': 0.0,
                'system_memory_usage': 0.0,
                'system_disk_usage': 0.0,
                'system_network_sent': 0.0,
                'system_network_recv': 0.0,
                'system_gpu_usage': 0.0,
                'system_uptime': 0.0
            }
    
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            # Try to use nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return gpu_util.gpu / 100.0
        except ImportError:
            # Try to use GPUtil if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    return gpus[0].load
                return 0.0
            except ImportError:
                return 0.0
        except Exception:
            return 0.0
    
    def _collect_model_metrics(self) -> Dict[str, float]:
        """Collect model performance metrics."""
        try:
            # These would typically come from the model inference results
            # For now, we'll return placeholder values
            # In a real implementation, these would be updated by the model monitor
            
            return {
                'detection_accuracy': 0.95,  # Placeholder
                'average_confidence': 0.85,  # Placeholder
                'model_latency': 0.1,  # Placeholder - seconds per prediction
                'model_throughput': 30.0,  # Placeholder - predictions per second
                'model_memory_usage': 0.0,  # Placeholder
                'model_error_rate': 0.05  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
            return {
                'detection_accuracy': 0.0,
                'average_confidence': 0.0,
                'model_latency': 0.0,
                'model_throughput': 0.0,
                'model_memory_usage': 0.0,
                'model_error_rate': 1.0
            }
    
    def _collect_processing_metrics(self) -> Dict[str, float]:
        """Collect processing performance metrics."""
        try:
            # These would typically come from the processing pipeline
            # For now, we'll return placeholder values
            
            return {
                'processing_speed': 25.0,  # FPS
                'average_processing_time': 0.04,  # seconds per frame
                'queue_size': 0,  # Number of items in processing queue
                'active_jobs': 0,  # Number of active processing jobs
                'failed_jobs': 0,  # Number of failed jobs
                'success_rate': 0.98  # Success rate of processing
            }
            
        except Exception as e:
            logger.error(f"Error collecting processing metrics: {e}")
            return {
                'processing_speed': 0.0,
                'average_processing_time': 0.0,
                'queue_size': 0,
                'active_jobs': 0,
                'failed_jobs': 0,
                'success_rate': 0.0
            }
    
    def _collect_quality_metrics(self) -> Dict[str, float]:
        """Collect data and output quality metrics."""
        try:
            # These would typically come from quality assessment results
            # For now, we'll return placeholder values
            
            return {
                'data_quality_score': 0.9,  # Overall data quality score
                'detection_coverage': 0.95,  # Percentage of frames with detections
                'tracking_consistency': 0.88,  # Tracking consistency score
                'event_detection_accuracy': 0.92,  # Event detection accuracy
                'output_quality_score': 0.94,  # Output quality score
                'confidence_std': 0.15  # Standard deviation of confidence scores
            }
            
        except Exception as e:
            logger.error(f"Error collecting quality metrics: {e}")
            return {
                'data_quality_score': 0.0,
                'detection_coverage': 0.0,
                'tracking_consistency': 0.0,
                'event_detection_accuracy': 0.0,
                'output_quality_score': 0.0,
                'confidence_std': 0.0
            }
    
    def update_model_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update model metrics with actual values.
        
        Args:
            metrics: Dictionary of model metrics
        """
        # This method would be called by the model monitor with actual values
        # For now, we'll just log the update
        logger.debug(f"Updated model metrics: {metrics}")
    
    def update_processing_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update processing metrics with actual values.
        
        Args:
            metrics: Dictionary of processing metrics
        """
        # This method would be called by the processing pipeline with actual values
        logger.debug(f"Updated processing metrics: {metrics}")
    
    def update_quality_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Update quality metrics with actual values.
        
        Args:
            metrics: Dictionary of quality metrics
        """
        # This method would be called by the quality assessment with actual values
        logger.debug(f"Updated quality metrics: {metrics}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        try:
            current_metrics = self.collect_all_metrics()
            
            # Calculate averages for historical metrics
            averages = {}
            for metric_name, history in self.metrics_history.items():
                if history:
                    averages[f"{metric_name}_average"] = np.mean(history)
                    averages[f"{metric_name}_std"] = np.std(history)
                    averages[f"{metric_name}_min"] = np.min(history)
                    averages[f"{metric_name}_max"] = np.max(history)
            
            return {
                'current_metrics': current_metrics,
                'historical_averages': averages,
                'collection_duration': (datetime.now() - self.collection_start_time).total_seconds(),
                'metrics_count': len(current_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {
                'current_metrics': {},
                'historical_averages': {},
                'collection_duration': 0.0,
                'metrics_count': 0,
                'error': str(e)
            }
    
    def get_metrics_trend(self, metric_name: str, window_size: int = 10) -> List[float]:
        """
        Get trend for a specific metric.
        
        Args:
            metric_name: Name of the metric
            window_size: Size of the trend window
            
        Returns:
            List of recent metric values
        """
        if metric_name in self.metrics_history:
            return self.metrics_history[metric_name][-window_size:]
        return []
    
    def clear_metrics_history(self) -> None:
        """Clear metrics history."""
        self.metrics_history.clear()
        logger.info("Metrics history cleared") 