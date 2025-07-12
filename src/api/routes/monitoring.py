"""
Monitoring API Routes

Provides endpoints for:
- Monitoring status and metrics
- Dashboard generation and export
- Report generation
- Monitoring control (start/stop)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.models import MonitoringStatus, DashboardRequest, ReportRequest
from src.monitoring import ModelMonitor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Global model monitor instance
_model_monitor: Optional[ModelMonitor] = None


def get_model_monitor() -> ModelMonitor:
    """Get or create the global model monitor instance."""
    global _model_monitor
    if _model_monitor is None:
        config = get_config()
        _model_monitor = ModelMonitor()
    return _model_monitor


@router.get("/status", response_model=MonitoringStatus)
async def get_monitoring_status():
    """
    Get current monitoring status and metrics.
    
    Returns:
        Current monitoring status including performance metrics, recent events, and system health
    """
    try:
        monitor = get_model_monitor()
        status = monitor.get_monitoring_status()
        
        return MonitoringStatus(
            is_monitoring=status['is_monitoring'],
            performance_metrics=status['performance_metrics'],
            recent_events=status['recent_events'],
            threshold_violations=status['threshold_violations'],
            metric_history_summary=status['metric_history_summary'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get monitoring status: {str(e)}")


@router.post("/start")
async def start_monitoring():
    """
    Start the monitoring system.
    
    Returns:
        Success message
    """
    try:
        monitor = get_model_monitor()
        monitor.start_monitoring()
        
        return {
            "status": "success",
            "message": "Monitoring started successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/stop")
async def stop_monitoring():
    """
    Stop the monitoring system.
    
    Returns:
        Success message
    """
    try:
        monitor = get_model_monitor()
        monitor.stop_monitoring()
        
        return {
            "status": "success",
            "message": "Monitoring stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")


@router.get("/dashboard")
async def get_dashboard(
    format: str = Query("html", description="Dashboard format: html, pdf, or json"),
    background_tasks: BackgroundTasks = None
):
    """
    Generate and return monitoring dashboard.
    
    Args:
        format: Output format (html, pdf, json)
        
    Returns:
        Dashboard file or JSON response
    """
    try:
        monitor = get_model_monitor()
        status = monitor.get_monitoring_status()
        
        if format.lower() == "json":
            return JSONResponse(content=status)
        
        # Generate dashboard file
        dashboard_path = monitor.export_dashboard(format=format)
        
        if format.lower() == "html":
            return HTMLResponse(content=open(dashboard_path, 'r', encoding='utf-8').read())
        else:
            return FileResponse(
                path=dashboard_path,
                media_type='application/octet-stream',
                filename=f"monitoring_dashboard.{format}"
            )
            
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")


@router.post("/dashboard/generate")
async def generate_dashboard(request: DashboardRequest):
    """
    Generate a custom dashboard with specific parameters.
    
    Args:
        request: Dashboard generation request
        
    Returns:
        Dashboard generation result
    """
    try:
        monitor = get_model_monitor()
        
        # Get monitoring status for the specified time range
        if request.start_time and request.end_time:
            start_dt = datetime.fromisoformat(request.start_time)
            end_dt = datetime.fromisoformat(request.end_time)
            status = monitor.generate_monitoring_report(start_dt, end_dt)
        else:
            status = monitor.get_monitoring_status()
        
        # Generate dashboard
        dashboard_path = monitor.export_dashboard(
            status, 
            format=request.format, 
            output_path=request.output_path
        )
        
        return {
            "status": "success",
            "dashboard_path": dashboard_path,
            "format": request.format,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating custom dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard: {str(e)}")


@router.post("/report/generate")
async def generate_report(request: ReportRequest):
    """
    Generate a comprehensive monitoring report.
    
    Args:
        request: Report generation request
        
    Returns:
        Report generation result
    """
    try:
        monitor = get_model_monitor()
        
        # Parse time range
        start_time = None
        end_time = None
        
        if request.start_time:
            start_time = datetime.fromisoformat(request.start_time)
        if request.end_time:
            end_time = datetime.fromisoformat(request.end_time)
        
        # Generate report
        report = monitor.generate_monitoring_report(start_time, end_time)
        
        # Export report if requested
        report_path = None
        if request.export_format:
            report_path = monitor.export_dashboard(
                report, 
                format=request.export_format,
                output_path=request.output_path
            )
        
        return {
            "status": "success",
            "report": report,
            "report_path": report_path,
            "format": request.export_format,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/alerts")
async def get_alerts(
    limit: int = Query(10, description="Number of recent alerts to return"),
    severity: Optional[str] = Query(None, description="Filter by severity level")
):
    """
    Get recent monitoring alerts.
    
    Args:
        limit: Number of alerts to return
        severity: Filter by severity level (low, medium, high, critical)
        
    Returns:
        List of recent alerts
    """
    try:
        monitor = get_model_monitor()
        alert_manager = monitor.alert_manager
        
        alerts = alert_manager.get_recent_alerts(limit)
        
        # Filter by severity if specified
        if severity:
            alerts = [alert for alert in alerts if alert['event']['severity'] == severity]
        
        return {
            "alerts": alerts,
            "total_alerts": len(alerts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.get("/alerts/statistics")
async def get_alert_statistics():
    """
    Get alert statistics.
    
    Returns:
        Alert statistics including counts by severity and channel
    """
    try:
        monitor = get_model_monitor()
        alert_manager = monitor.alert_manager
        
        stats = alert_manager.get_alert_statistics()
        
        return {
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting alert statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert statistics: {str(e)}")


@router.get("/workflows")
async def get_workflows():
    """
    Get workflow status and statistics.
    
    Returns:
        Current workflow status and recent workflows
    """
    try:
        monitor = get_model_monitor()
        workflow_orchestrator = monitor.workflow_orchestrator
        
        status = workflow_orchestrator.get_workflow_status()
        
        return {
            "workflow_status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.post("/workflows/{workflow_name}/stop")
async def stop_workflow(workflow_name: str):
    """
    Stop a running workflow.
    
    Args:
        workflow_name: Name of the workflow to stop
        
    Returns:
        Success message
    """
    try:
        monitor = get_model_monitor()
        workflow_orchestrator = monitor.workflow_orchestrator
        
        success = workflow_orchestrator.stop_workflow(workflow_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Workflow {workflow_name} stopped successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail=f"Workflow {workflow_name} not found or not running")
            
    except Exception as e:
        logger.error(f"Error stopping workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop workflow: {str(e)}")


@router.get("/metrics")
async def get_metrics(
    metric_name: Optional[str] = Query(None, description="Specific metric to retrieve"),
    window_size: int = Query(10, description="Number of recent values to return")
):
    """
    Get current metrics or metric trends.
    
    Args:
        metric_name: Specific metric name (if None, returns all metrics)
        window_size: Number of recent values to return for trends
        
    Returns:
        Current metrics or metric trend
    """
    try:
        monitor = get_model_monitor()
        metrics_collector = monitor.metrics_collector
        
        if metric_name:
            # Get specific metric trend
            trend = metrics_collector.get_metrics_trend(metric_name, window_size)
            return {
                "metric_name": metric_name,
                "trend": trend,
                "window_size": window_size,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Get all current metrics
            summary = metrics_collector.get_metrics_summary()
            return {
                "metrics": summary,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.post("/thresholds/update")
async def update_threshold(
    metric_name: str,
    threshold_config: Dict[str, Any]
):
    """
    Update threshold configuration for a metric.
    
    Args:
        metric_name: Name of the metric
        threshold_config: New threshold configuration
        
    Returns:
        Success message
    """
    try:
        monitor = get_model_monitor()
        threshold_manager = monitor.threshold_manager
        
        threshold_manager.update_threshold(metric_name, threshold_config)
        
        return {
            "status": "success",
            "message": f"Threshold updated for {metric_name}",
            "metric_name": metric_name,
            "threshold_config": threshold_config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating threshold: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update threshold: {str(e)}")


@router.get("/thresholds")
async def get_thresholds():
    """
    Get current threshold configurations.
    
    Returns:
        Current threshold configurations for all metrics
    """
    try:
        monitor = get_model_monitor()
        threshold_manager = monitor.threshold_manager
        
        thresholds = threshold_manager.get_threshold_config()
        
        return {
            "thresholds": thresholds,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting thresholds: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get thresholds: {str(e)}")


@router.post("/clear/history")
async def clear_history(
    history_type: str = Query(..., description="Type of history to clear: alerts, workflows, metrics, or all")
):
    """
    Clear monitoring history.
    
    Args:
        history_type: Type of history to clear
        
    Returns:
        Success message
    """
    try:
        monitor = get_model_monitor()
        
        if history_type == "alerts":
            monitor.alert_manager.clear_alert_history()
        elif history_type == "workflows":
            monitor.workflow_orchestrator.clear_workflow_history()
        elif history_type == "metrics":
            monitor.metrics_collector.clear_metrics_history()
        elif history_type == "all":
            monitor.alert_manager.clear_alert_history()
            monitor.workflow_orchestrator.clear_workflow_history()
            monitor.metrics_collector.clear_metrics_history()
        else:
            raise HTTPException(status_code=400, detail=f"Invalid history type: {history_type}")
        
        return {
            "status": "success",
            "message": f"{history_type} history cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}") 