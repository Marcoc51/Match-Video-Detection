"""
Monitoring entities for the football video analysis system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class MonitoringEvent:
    """Represents a monitoring event."""
    timestamp: datetime
    event_type: str  # 'threshold_violation', 'model_degradation', 'system_alert'
    severity: str    # 'low', 'medium', 'high', 'critical'
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    job_id: Optional[str] = None
    model_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None 