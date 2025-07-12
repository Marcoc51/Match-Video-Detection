"""
Threshold Manager for Model Monitoring

Handles threshold checking and violation detection for various metrics.
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logger = logging.getLogger(__name__)


class ThresholdManager:
    """
    Manages threshold checking and violation detection.
    
    Supports:
    - Min/max threshold checking
    - Trend-based threshold violations
    - Severity level determination
    - Threshold history tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the threshold manager.
        
        Args:
            config: Threshold configuration dictionary
        """
        self.config = config
        self.threshold_history: List[Dict[str, Any]] = []
        self.violation_count = 0
        
        # Default thresholds if not provided
        self.default_thresholds = {
            'detection_accuracy': {'min': 0.85, 'max': 1.0},
            'average_confidence': {'min': 0.6, 'max': 1.0},
            'processing_speed': {'min': 20.0, 'max': 100.0},
            'error_rate': {'min': 0.0, 'max': 0.1},
            'system_memory_usage': {'min': 0.0, 'max': 0.8},
            'system_cpu_usage': {'min': 0.0, 'max': 0.9},
            'model_latency': {'min': 0.0, 'max': 2.0},
            'data_quality_score': {'min': 0.7, 'max': 1.0}
        }
        
        # Merge with provided config
        self.thresholds = {**self.default_thresholds, **config}
        
        logger.info("Threshold manager initialized")
    
    def check_thresholds(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check metrics against configured thresholds.
        
        Args:
            metrics: Dictionary of current metric values
            
        Returns:
            List of threshold violations
        """
        violations = []
        
        for metric_name, current_value in metrics.items():
            if metric_name in self.thresholds:
                threshold_config = self.thresholds[metric_name]
                violation = self._check_single_threshold(
                    metric_name, current_value, threshold_config
                )
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _check_single_threshold(self, 
                               metric_name: str, 
                               current_value: float, 
                               threshold_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check a single metric against its threshold.
        
        Args:
            metric_name: Name of the metric
            current_value: Current value of the metric
            threshold_config: Threshold configuration
            
        Returns:
            Violation dictionary if threshold is violated, None otherwise
        """
        min_threshold = threshold_config.get('min')
        max_threshold = threshold_config.get('max')
        
        violation = None
        
        # Check minimum threshold
        if min_threshold is not None and current_value < min_threshold:
            violation = {
                'metric_name': metric_name,
                'current_value': current_value,
                'threshold_value': min_threshold,
                'threshold_type': 'min',
                'severity': self._determine_severity(metric_name, current_value, min_threshold, 'min'),
                'message': f"{metric_name} ({current_value:.4f}) below minimum threshold ({min_threshold:.4f})"
            }
        
        # Check maximum threshold
        elif max_threshold is not None and current_value > max_threshold:
            violation = {
                'metric_name': metric_name,
                'current_value': current_value,
                'threshold_value': max_threshold,
                'threshold_type': 'max',
                'severity': self._determine_severity(metric_name, current_value, max_threshold, 'max'),
                'message': f"{metric_name} ({current_value:.4f}) above maximum threshold ({max_threshold:.4f})"
            }
        
        if violation:
            # Add metadata
            violation['timestamp'] = datetime.now().isoformat()
            violation['metadata'] = {
                'threshold_config': threshold_config,
                'violation_count': self._get_violation_count(metric_name)
            }
            
            # Update history
            self.threshold_history.append(violation)
            self.violation_count += 1
            
            # Keep only last 1000 violations
            if len(self.threshold_history) > 1000:
                self.threshold_history = self.threshold_history[-1000:]
        
        return violation
    
    def _determine_severity(self, 
                           metric_name: str, 
                           current_value: float, 
                           threshold_value: float, 
                           threshold_type: str) -> str:
        """
        Determine severity level based on threshold violation.
        
        Args:
            metric_name: Name of the metric
            current_value: Current value
            threshold_value: Threshold value
            threshold_type: Type of threshold ('min' or 'max')
            
        Returns:
            Severity level ('low', 'medium', 'high', 'critical')
        """
        # Calculate deviation percentage
        if threshold_type == 'min':
            if threshold_value == 0:
                deviation = 1.0 if current_value == 0 else float('inf')
            else:
                deviation = (threshold_value - current_value) / threshold_value
        else:  # max
            if threshold_value == 0:
                deviation = 1.0 if current_value == 0 else float('inf')
            else:
                deviation = (current_value - threshold_value) / threshold_value
        
        # Determine severity based on deviation
        if deviation >= 0.5:  # 50% or more deviation
            return 'critical'
        elif deviation >= 0.3:  # 30% or more deviation
            return 'high'
        elif deviation >= 0.1:  # 10% or more deviation
            return 'medium'
        else:
            return 'low'
    
    def _get_violation_count(self, metric_name: str) -> int:
        """Get violation count for a specific metric."""
        return len([
            v for v in self.threshold_history 
            if v['metric_name'] == metric_name
        ])
    
    def update_threshold(self, metric_name: str, threshold_config: Dict[str, Any]) -> None:
        """
        Update threshold for a specific metric.
        
        Args:
            metric_name: Name of the metric
            threshold_config: New threshold configuration
        """
        self.thresholds[metric_name] = threshold_config
        logger.info(f"Updated threshold for {metric_name}: {threshold_config}")
    
    def get_threshold_config(self) -> Dict[str, Any]:
        """Get current threshold configuration."""
        return self.thresholds.copy()
    
    def get_violation_statistics(self, 
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get violation statistics for a time period.
        
        Args:
            start_time: Start time for statistics
            end_time: End time for statistics
            
        Returns:
            Violation statistics
        """
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        # Filter violations by time
        filtered_violations = [
            v for v in self.threshold_history
            if start_time <= datetime.fromisoformat(v['timestamp']) <= end_time
        ]
        
        # Calculate statistics
        violations_by_metric = {}
        violations_by_severity = {}
        
        for violation in filtered_violations:
            metric_name = violation['metric_name']
            severity = violation['severity']
            
            violations_by_metric[metric_name] = violations_by_metric.get(metric_name, 0) + 1
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
        
        return {
            'total_violations': len(filtered_violations),
            'violations_by_metric': violations_by_metric,
            'violations_by_severity': violations_by_severity,
            'period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
        }
    
    def get_recent_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent threshold violations."""
        return self.threshold_history[-limit:] if self.threshold_history else []
    
    def clear_violation_history(self) -> None:
        """Clear violation history."""
        self.threshold_history.clear()
        self.violation_count = 0
        logger.info("Violation history cleared") 