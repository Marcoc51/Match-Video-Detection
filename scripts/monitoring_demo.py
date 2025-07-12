#!/usr/bin/env python3
"""
Monitoring API Demo Script

This script demonstrates how to use the monitoring API endpoints
to check status, generate dashboards, and manage monitoring.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# API Configuration
API_BASE_URL = "http://localhost:8000"
MONITORING_BASE_URL = f"{API_BASE_URL}/monitoring"


def make_request(method: str, endpoint: str, data: dict = None, params: dict = None):
    """Make a request to the monitoring API."""
    url = f"{MONITORING_BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, params=params)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Error making {method} request to {endpoint}: {e}")
        return None


def demo_monitoring_status():
    """Demonstrate getting monitoring status."""
    print("\n=== Monitoring Status ===")
    status = make_request("GET", "/status")
    if status:
        print(f"Monitoring Active: {status['is_monitoring']}")
        print(f"Threshold Violations: {status['threshold_violations']}")
        print(f"Recent Events: {len(status['recent_events'])}")
        print(f"Performance Metrics: {json.dumps(status['performance_metrics'], indent=2)}")


def demo_start_monitoring():
    """Demonstrate starting monitoring."""
    print("\n=== Starting Monitoring ===")
    result = make_request("POST", "/start")
    if result:
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")


def demo_stop_monitoring():
    """Demonstrate stopping monitoring."""
    print("\n=== Stopping Monitoring ===")
    result = make_request("POST", "/stop")
    if result:
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")


def demo_dashboard_generation():
    """Demonstrate dashboard generation."""
    print("\n=== Dashboard Generation ===")
    
    # Generate HTML dashboard
    print("Generating HTML dashboard...")
    dashboard_data = {
        "format": "html",
        "start_time": (datetime.now() - timedelta(days=7)).isoformat(),
        "end_time": datetime.now().isoformat()
    }
    result = make_request("POST", "/dashboard/generate", data=dashboard_data)
    if result:
        print(f"Dashboard generated: {result['dashboard_path']}")
    
    # Generate JSON report
    print("Generating JSON report...")
    report_data = {
        "start_time": (datetime.now() - timedelta(days=1)).isoformat(),
        "end_time": datetime.now().isoformat(),
        "export_format": "json"
    }
    result = make_request("POST", "/report/generate", data=report_data)
    if result:
        print(f"Report generated: {result['report_path']}")


def demo_alerts():
    """Demonstrate alert management."""
    print("\n=== Alert Management ===")
    
    # Get recent alerts
    alerts = make_request("GET", "/alerts", params={"limit": 5})
    if alerts:
        print(f"Total Alerts: {alerts['total_alerts']}")
        for alert in alerts['alerts']:
            event = alert['event']
            print(f"- {event['timestamp']}: {event['severity']} - {event['message']}")
    
    # Get alert statistics
    stats = make_request("GET", "/alerts/statistics")
    if stats:
        print(f"Alert Statistics: {json.dumps(stats['statistics'], indent=2)}")


def demo_workflows():
    """Demonstrate workflow management."""
    print("\n=== Workflow Management ===")
    
    # Get workflow status
    workflows = make_request("GET", "/workflows")
    if workflows:
        status = workflows['workflow_status']
        print(f"Active Workflows: {status['active_workflows']}")
        print(f"Total Workflows: {status['total_workflows']}")
        
        if status['recent_workflows']:
            print("Recent Workflows:")
            for workflow in status['recent_workflows'][-3:]:  # Last 3
                print(f"- {workflow['workflow_name']}: {workflow['status']}")


def demo_metrics():
    """Demonstrate metrics retrieval."""
    print("\n=== Metrics Retrieval ===")
    
    # Get all metrics
    metrics = make_request("GET", "/metrics")
    if metrics:
        print("Current Metrics Summary:")
        summary = metrics['metrics']
        print(f"Collection Duration: {summary['collection_duration']:.2f} seconds")
        print(f"Metrics Count: {summary['metrics_count']}")
    
    # Get specific metric trend
    trend = make_request("GET", "/metrics", params={
        "metric_name": "detection_accuracy",
        "window_size": 5
    })
    if trend:
        print(f"Detection Accuracy Trend: {trend['trend']}")


def demo_thresholds():
    """Demonstrate threshold management."""
    print("\n=== Threshold Management ===")
    
    # Get current thresholds
    thresholds = make_request("GET", "/thresholds")
    if thresholds:
        print("Current Thresholds:")
        for metric, config in thresholds['thresholds'].items():
            print(f"- {metric}: {config}")
    
    # Update a threshold (example)
    print("\nUpdating detection_accuracy threshold...")
    threshold_update = {
        "metric_name": "detection_accuracy",
        "threshold_config": {"min": 0.8, "max": 1.0}
    }
    result = make_request("POST", "/thresholds/update", data=threshold_update)
    if result:
        print(f"Threshold updated: {result['message']}")


def demo_history_management():
    """Demonstrate history management."""
    print("\n=== History Management ===")
    
    # Clear specific history type
    result = make_request("POST", "/clear/history", params={"history_type": "alerts"})
    if result:
        print(f"History cleared: {result['message']}")


def main():
    """Run the monitoring API demo."""
    print("🚀 Football Video Analysis - Monitoring API Demo")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code != 200:
            print(f"❌ API is not responding properly. Status: {response.status_code}")
            return
        print("✅ API is running")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        return
    
    # Run demos
    demo_monitoring_status()
    demo_start_monitoring()
    time.sleep(2)  # Give monitoring time to start
    
    demo_dashboard_generation()
    demo_alerts()
    demo_workflows()
    demo_metrics()
    demo_thresholds()
    demo_history_management()
    
    demo_stop_monitoring()
    
    print("\n" + "=" * 50)
    print("✅ Monitoring API Demo Complete!")
    print("\n📋 Available Endpoints:")
    print("  GET  /monitoring/status              - Get monitoring status")
    print("  POST /monitoring/start               - Start monitoring")
    print("  POST /monitoring/stop                - Stop monitoring")
    print("  GET  /monitoring/dashboard           - Get dashboard (html/pdf/json)")
    print("  POST /monitoring/dashboard/generate  - Generate custom dashboard")
    print("  POST /monitoring/report/generate     - Generate monitoring report")
    print("  GET  /monitoring/alerts              - Get recent alerts")
    print("  GET  /monitoring/alerts/statistics   - Get alert statistics")
    print("  GET  /monitoring/workflows           - Get workflow status")
    print("  GET  /monitoring/metrics             - Get metrics/trends")
    print("  GET  /monitoring/thresholds          - Get threshold config")
    print("  POST /monitoring/thresholds/update   - Update thresholds")
    print("  POST /monitoring/clear/history       - Clear history")


if __name__ == "__main__":
    main() 