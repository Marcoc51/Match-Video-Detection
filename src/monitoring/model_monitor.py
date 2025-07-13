"""
Model Monitor for Football Video Analysis System

This module provides comprehensive model monitoring with:
- Real-time metric tracking
- Threshold violation detection
- Automated alerting
- Conditional workflow execution
"""

import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import queue

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager
from .alert_manager import AlertManager
from .threshold_manager import ThresholdManager
from .workflow_orchestrator import WorkflowOrchestrator
from .dashboard_generator import DashboardGenerator
from .metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


from .entities import MonitoringEvent


class ModelMonitor:
    """
    Comprehensive model monitoring system for football video analysis.
    
    This class provides:
    - Real-time metric collection and tracking
    - Threshold violation detection
    - Automated alerting and notifications
    - Conditional workflow execution
    - Performance dashboard generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the model monitor.
        
        Args:
            config_path: Path to monitoring configuration file
        """
        self.config = get_config()
        self.monitoring_config = self._load_monitoring_config(config_path)
        
        # Initialize components
        self.alert_manager = AlertManager(self.monitoring_config.get('alerts', {}))
        self.threshold_manager = ThresholdManager(self.monitoring_config.get('thresholds', {}))
        self.workflow_orchestrator = WorkflowOrchestrator(self.monitoring_config.get('workflows', {}))
        self.dashboard_generator = DashboardGenerator(self.monitoring_config.get('dashboard', {}))
        self.metrics_collector = MetricsCollector(self.monitoring_config.get('metrics', {}))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.event_queue = queue.Queue()
        self.monitoring_events: List[MonitoringEvent] = []
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_confidence': 0.0,
            'average_processing_time': 0.0,
            'model_accuracy': 0.0,
            'system_uptime': 0.0
        }
        
        logger.info("Model monitor initialized")
    
    def _load_monitoring_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'monitoring_interval': 30,  # seconds
            'metrics_retention_days': 30,
            'thresholds': {
                'detection_accuracy': {'min': 0.85, 'max': 1.0},
                'average_confidence': {'min': 0.6, 'max': 1.0},
                'processing_speed': {'min': 20.0, 'max': 100.0},  # FPS
                'error_rate': {'min': 0.0, 'max': 0.1},
                'system_memory_usage': {'min': 0.0, 'max': 0.8},
                'system_cpu_usage': {'min': 0.0, 'max': 0.9}
            },
            'alerts': {
                'email_enabled': True,
                'webhook_enabled': True,
                'slack_enabled': False,
                'notification_channels': ['email', 'webhook']
            },
            'workflows': {
                'auto_retraining': {
                    'enabled': True,
                    'trigger_conditions': ['accuracy_below_threshold', 'confidence_below_threshold'],
                    'min_accuracy_threshold': 0.8,
                    'min_confidence_threshold': 0.6
                },
                'model_switching': {
                    'enabled': True,
                    'fallback_model': 'models/yolo/fallback.pt',
                    'trigger_conditions': ['critical_accuracy_drop', 'model_failure']
                },
                'debugging_dashboard': {
                    'enabled': True,
                    'trigger_conditions': ['performance_degradation', 'error_spike'],
                    'auto_generate': True
                }
            },
            'dashboard': {
                'auto_refresh': True,
                'refresh_interval': 60,  # seconds
                'export_formats': ['html', 'pdf', 'json']
            }
        }
    
    def start_monitoring(self) -> None:
        """Start the monitoring system."""
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Model monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Model monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect current metrics
                current_metrics = self.metrics_collector.collect_all_metrics()
                
                # Update metric history
                self._update_metric_history(current_metrics)
                
                # Check thresholds
                violations = self.threshold_manager.check_thresholds(current_metrics)
                
                # Process violations
                for violation in violations:
                    self._process_threshold_violation(violation)
                
                # Process queued events
                self._process_event_queue()
                
                # Update performance metrics
                self._update_performance_metrics(current_metrics)
                
                # Generate dashboard if needed
                if self.monitoring_config['dashboard']['auto_refresh']:
                    self.dashboard_generator.update_dashboard(current_metrics)
                
                # Wait for next monitoring cycle
                time.sleep(self.monitoring_config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _update_metric_history(self, metrics: Dict[str, float]) -> None:
        """Update metric history with current values."""
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []
            
            self.metric_history[metric_name].append((timestamp, value))
            
            # Clean old metrics
            cutoff_time = timestamp - timedelta(days=self.monitoring_config['metrics_retention_days'])
            self.metric_history[metric_name] = [
                (t, v) for t, v in self.metric_history[metric_name] 
                if t > cutoff_time
            ]
    
    def _process_threshold_violation(self, violation: Dict[str, Any]) -> None:
        """Process a threshold violation."""
        event = MonitoringEvent(
            timestamp=datetime.now(),
            event_type='threshold_violation',
            severity=violation.get('severity', 'medium'),
            metric_name=violation['metric_name'],
            current_value=violation['current_value'],
            threshold_value=violation['threshold_value'],
            message=violation['message'],
            metadata=violation.get('metadata', {})
        )
        
        # Add to event history
        self.monitoring_events.append(event)
        
        # Send alerts
        self.alert_manager.send_alert(event)
        
        # Trigger workflows if conditions are met
        self.workflow_orchestrator.check_and_trigger_workflows(event)
        
        logger.warning(f"Threshold violation: {event.message}")
    
    def _process_event_queue(self) -> None:
        """Process events in the queue."""
        try:
            while not self.event_queue.empty():
                event = self.event_queue.get_nowait()
                self.monitoring_events.append(event)
                
                # Send alerts for queued events
                self.alert_manager.send_alert(event)
                
        except queue.Empty:
            pass
    
    def _update_performance_metrics(self, current_metrics: Dict[str, float]) -> None:
        """Update performance metrics."""
        self.performance_metrics.update({
            'average_confidence': current_metrics.get('average_confidence', 0.0),
            'model_accuracy': current_metrics.get('detection_accuracy', 0.0),
            'average_processing_time': current_metrics.get('average_processing_time', 0.0)
        })
    
    def record_prediction(self, 
                         confidence: float, 
                         processing_time: float, 
                         success: bool,
                         job_id: Optional[str] = None) -> None:
        """
        Record a prediction for monitoring.
        
        Args:
            confidence: Model confidence score
            processing_time: Time taken for prediction
            success: Whether prediction was successful
            job_id: Associated job ID
        """
        self.performance_metrics['total_predictions'] += 1
        
        if success:
            self.performance_metrics['successful_predictions'] += 1
        else:
            self.performance_metrics['failed_predictions'] += 1
        
        # Update average confidence
        total_preds = self.performance_metrics['total_predictions']
        current_avg = self.performance_metrics['average_confidence']
        self.performance_metrics['average_confidence'] = (
            (current_avg * (total_preds - 1) + confidence) / total_preds
        )
        
        # Update average processing time
        current_avg_time = self.performance_metrics['average_processing_time']
        self.performance_metrics['average_processing_time'] = (
            (current_avg_time * (total_preds - 1) + processing_time) / total_preds
        )
        
        # Check for immediate issues
        if confidence < self.monitoring_config['thresholds']['average_confidence']['min']:
            event = MonitoringEvent(
                timestamp=datetime.now(),
                event_type='low_confidence_prediction',
                severity='medium',
                metric_name='prediction_confidence',
                current_value=confidence,
                threshold_value=self.monitoring_config['thresholds']['average_confidence']['min'],
                message=f"Low confidence prediction: {confidence:.3f}",
                job_id=job_id
            )
            self.event_queue.put(event)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'is_monitoring': self.is_monitoring,
            'performance_metrics': self.performance_metrics,
            'recent_events': [
                asdict(event) for event in self.monitoring_events[-10:]  # Last 10 events
            ],
            'threshold_violations': len([
                event for event in self.monitoring_events 
                if event.event_type == 'threshold_violation'
            ]),
            'metric_history_summary': {
                metric: {
                    'count': len(history),
                    'latest_value': history[-1][1] if history else None,
                    'average_value': sum(v for _, v in history) / len(history) if history else None
                }
                for metric, history in self.metric_history.items()
            }
        }
    
    def generate_monitoring_report(self, 
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a comprehensive monitoring report."""
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        # Filter events by time range
        filtered_events = [
            event for event in self.monitoring_events
            if start_time <= event.timestamp <= end_time
        ]
        
        # Filter metric history by time range
        filtered_metrics = {}
        for metric_name, history in self.metric_history.items():
            filtered_history = [
                (t, v) for t, v in history
                if start_time <= t <= end_time
            ]
            if filtered_history:
                filtered_metrics[metric_name] = filtered_history
        
        return {
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'summary': {
                'total_events': len(filtered_events),
                'threshold_violations': len([
                    e for e in filtered_events 
                    if e.event_type == 'threshold_violation'
                ]),
                'critical_events': len([
                    e for e in filtered_events 
                    if e.severity == 'critical'
                ]),
                'total_predictions': self.performance_metrics['total_predictions'],
                'success_rate': (
                    self.performance_metrics['successful_predictions'] / 
                    max(self.performance_metrics['total_predictions'], 1)
                )
            },
            'events': [asdict(event) for event in filtered_events],
            'metrics': filtered_metrics,
            'recommendations': self._generate_recommendations(filtered_events, filtered_metrics)
        }
    
    def _generate_recommendations(self, 
                                events: List[MonitoringEvent],
                                metrics: Dict[str, List[Tuple[datetime, float]]]) -> List[str]:
        """Generate recommendations based on monitoring data."""
        recommendations = []
        
        # Check for frequent threshold violations
        threshold_violations = [e for e in events if e.event_type == 'threshold_violation']
        if len(threshold_violations) > 5:
            recommendations.append(
                "High number of threshold violations detected. Consider adjusting thresholds or investigating model performance."
            )
        
        # Check for accuracy degradation
        if 'detection_accuracy' in metrics:
            accuracy_values = [v for _, v in metrics['detection_accuracy']]
            if len(accuracy_values) > 10:
                recent_avg = sum(accuracy_values[-10:]) / 10
                overall_avg = sum(accuracy_values) / len(accuracy_values)
                if recent_avg < overall_avg * 0.9:  # 10% degradation
                    recommendations.append(
                        "Model accuracy showing degradation trend. Consider retraining or investigating data drift."
                    )
        
        # Check for confidence issues
        if 'average_confidence' in metrics:
            confidence_values = [v for _, v in metrics['average_confidence']]
            if confidence_values and sum(confidence_values) / len(confidence_values) < 0.6:
                recommendations.append(
                    "Low average confidence detected. Consider model retraining or threshold adjustment."
                )
        
        return recommendations
    
    def export_dashboard(self, format: str = 'html') -> str:
        """Export monitoring dashboard."""
        return self.dashboard_generator.export_dashboard(
            self.get_monitoring_status(),
            format=format
        ) 