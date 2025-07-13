"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class ProcessingStatus(str, Enum):
    """Enum for processing status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FeatureConfig(BaseModel):
    """Configuration for video processing features."""
    passes: bool = Field(default=True, description="Enable pass detection")
    possession: bool = Field(default=True, description="Enable possession tracking")
    crosses: bool = Field(default=False, description="Enable cross detection")
    speed_distance: bool = Field(default=True, description="Enable speed and distance estimation")
    team_assignment: bool = Field(default=True, description="Enable team assignment")


class ProcessingJob(BaseModel):
    """Model for processing job information."""
    job_id: int = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Original video filename")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress: int = Field(ge=0, le=100, description="Processing progress percentage")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    features: Optional[FeatureConfig] = Field(None, description="Processing features configuration")


class ProcessingResult(BaseModel):
    """Model for processing results."""
    job_id: int = Field(..., description="Job identifier")
    filename: str = Field(..., description="Original video filename")
    status: ProcessingStatus = Field(..., description="Processing status")
    features: FeatureConfig = Field(..., description="Features that were processed")
    output_files: Dict[str, str] = Field(..., description="Available output file URLs")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Processing statistics")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    message: str = Field(..., description="Result message")


class VideoStatistics(BaseModel):
    """Model for video processing statistics."""
    total_frames: int = Field(..., description="Total number of frames processed")
    video_duration: float = Field(..., description="Video duration in seconds")
    fps: float = Field(..., description="Video frame rate")
    resolution: str = Field(..., description="Video resolution (width x height)")
    
    # Detection statistics
    players_detected: int = Field(..., description="Number of players detected")
    ball_detections: int = Field(..., description="Number of ball detections")
    
    # Event statistics
    passes_detected: Optional[int] = Field(None, description="Number of passes detected")
    possession_home: Optional[float] = Field(None, description="Home team possession percentage")
    possession_away: Optional[float] = Field(None, description="Away team possession percentage")
    crosses_detected: Optional[int] = Field(None, description="Number of crosses detected")
    
    # Processing metadata
    model_confidence: float = Field(..., description="Average model confidence")
    processing_fps: float = Field(..., description="Processing frame rate")


class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    job_id: Optional[int] = Field(None, description="Job ID if applicable")


class HealthResponse(BaseModel):
    """Model for health check responses."""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Health message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="API version")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")


class APIInfo(BaseModel):
    """Model for API information."""
    name: str = Field(..., description="API name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="API description")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")
    features: List[str] = Field(..., description="Supported features")
    documentation: str = Field(..., description="Documentation URL")


class JobListResponse(BaseModel):
    """Model for job list responses."""
    jobs: List[ProcessingJob] = Field(..., description="List of processing jobs")
    total: int = Field(..., description="Total number of jobs")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Jobs per page")
    has_next: bool = Field(..., description="Whether there are more pages")
    has_prev: bool = Field(..., description="Whether there are previous pages")


class ModelInfo(BaseModel):
    """Model for model information."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type (e.g., YOLO, classification)")
    accuracy: Optional[float] = Field(None, description="Model accuracy")
    last_updated: datetime = Field(..., description="Last model update timestamp")
    status: str = Field(..., description="Model status (active, training, etc.)")


class SystemStatus(BaseModel):
    """Model for system status information."""
    cpu_usage: float = Field(..., description="CPU usage percentage")
    memory_usage: float = Field(..., description="Memory usage percentage")
    gpu_usage: Optional[float] = Field(None, description="GPU usage percentage")
    disk_usage: float = Field(..., description="Disk usage percentage")
    active_jobs: int = Field(..., description="Number of active processing jobs")
    queue_size: int = Field(..., description="Number of jobs in queue")
    uptime: float = Field(..., description="System uptime in seconds") 


class MonitoringStatus(BaseModel):
    """Model for monitoring status information."""
    is_monitoring: bool = Field(..., description="Whether monitoring is currently running")
    performance_metrics: Dict[str, Any] = Field(..., description="Current performance metrics")
    recent_events: List[Dict[str, Any]] = Field(..., description="Recent monitoring events")
    threshold_violations: int = Field(..., description="Number of threshold violations")
    metric_history_summary: Dict[str, Any] = Field(..., description="Summary of metric history")
    timestamp: str = Field(..., description="Timestamp of the status")


class DashboardRequest(BaseModel):
    """Model for dashboard generation request."""
    format: str = Field(default="html", description="Dashboard format (html, pdf, json)")
    start_time: Optional[str] = Field(None, description="Start time for data range (ISO format)")
    end_time: Optional[str] = Field(None, description="End time for data range (ISO format)")
    output_path: Optional[str] = Field(None, description="Custom output path for the dashboard")


class ReportRequest(BaseModel):
    """Model for report generation request."""
    start_time: Optional[str] = Field(None, description="Start time for report period (ISO format)")
    end_time: Optional[str] = Field(None, description="End time for report period (ISO format)")
    export_format: Optional[str] = Field(None, description="Export format (html, pdf, json)")
    output_path: Optional[str] = Field(None, description="Custom output path for the report")


class ThresholdUpdateRequest(BaseModel):
    """Model for threshold update request."""
    metric_name: str = Field(..., description="Name of the metric to update")
    min_value: Optional[float] = Field(None, description="Minimum threshold value")
    max_value: Optional[float] = Field(None, description="Maximum threshold value")


class AlertFilterRequest(BaseModel):
    """Model for alert filtering request."""
    severity: Optional[str] = Field(None, description="Filter by severity level")
    event_type: Optional[str] = Field(None, description="Filter by event type")
    metric_name: Optional[str] = Field(None, description="Filter by metric name")
    limit: int = Field(default=10, description="Number of alerts to return")


class WorkflowTriggerRequest(BaseModel):
    """Model for workflow trigger request."""
    workflow_name: str = Field(..., description="Name of the workflow to trigger")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Workflow parameters")


class MetricsRequest(BaseModel):
    """Model for metrics request."""
    metric_names: Optional[List[str]] = Field(None, description="Specific metrics to retrieve")
    window_size: int = Field(default=10, description="Number of recent values to return")
    include_trends: bool = Field(default=True, description="Include trend analysis")


class MonitoringConfigRequest(BaseModel):
    """Model for monitoring configuration request."""
    monitoring_interval: Optional[int] = Field(None, description="Monitoring interval in seconds")
    metrics_retention_days: Optional[int] = Field(None, description="Metrics retention period in days")
    alerts_enabled: Optional[bool] = Field(None, description="Enable/disable alerts")
    workflows_enabled: Optional[bool] = Field(None, description="Enable/disable workflows")


# Alias classes for backward compatibility with tests
JobStatus = ProcessingStatus
JobRequest = ProcessingJob
JobResponse = ProcessingResult 