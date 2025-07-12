"""
Dashboard Generator for Model Monitoring

Generates monitoring dashboards and reports:
- Live HTML dashboard for model health and alerts
- Export to HTML, PDF, or JSON
- Debugging dashboard for workflow triggers
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import pdfkit  # For PDF export (requires wkhtmltopdf)
except ImportError:
    pdfkit = None

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """
    Generates dashboards and reports for model monitoring.
    
    Supports:
    - Live HTML dashboard
    - Export to HTML, PDF, or JSON
    - Debugging dashboard for workflow triggers
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.last_dashboard_path = None
        logger.info("Dashboard generator initialized")

    def update_dashboard(self, monitoring_status: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate or update the monitoring dashboard (HTML).
        Args:
            monitoring_status: Monitoring status dictionary
            output_path: Path to save the dashboard (default: outputs/monitoring_dashboard.html)
        Returns:
            Path to the generated dashboard
        """
        if not output_path:
            output_path = "outputs/monitoring_dashboard.html"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        html = self._render_html_dashboard(monitoring_status)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        self.last_dashboard_path = output_path
        logger.info(f"Monitoring dashboard updated: {output_path}")
        return output_path

    def export_dashboard(self, monitoring_status: Dict[str, Any], format: str = "html", output_path: Optional[str] = None) -> str:
        """
        Export the dashboard in the specified format (HTML, PDF, JSON).
        Args:
            monitoring_status: Monitoring status dictionary
            format: 'html', 'pdf', or 'json'
            output_path: Path to save the exported dashboard
        Returns:
            Path to the exported dashboard
        """
        if not output_path:
            output_path = f"outputs/monitoring_dashboard.{format}"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if format == "html":
            html = self._render_html_dashboard(monitoring_status)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)
        elif format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(monitoring_status, f, indent=2, default=str)
        elif format == "pdf":
            if pdfkit is None:
                raise ImportError("pdfkit is not installed. Install it with 'pip install pdfkit' and ensure wkhtmltopdf is available.")
            html = self._render_html_dashboard(monitoring_status)
            pdfkit.from_string(html, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        logger.info(f"Dashboard exported: {output_path}")
        return output_path

    def generate_debug_dashboard(self, event: Any, output_dir: str = "outputs/debug_dashboards") -> str:
        """
        Generate a debugging dashboard/report for a triggered workflow event.
        Args:
            event: MonitoringEvent or dict
            output_dir: Directory to save the dashboard
        Returns:
            Path to the generated debug dashboard
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"debug_dashboard_{timestamp}.html"
        html = self._render_debug_dashboard(event)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Debugging dashboard generated: {output_path}")
        return str(output_path)

    def _render_html_dashboard(self, status: Dict[str, Any]) -> str:
        """
        Render the main monitoring dashboard as HTML.
        Args:
            status: Monitoring status dictionary
        Returns:
            HTML string
        """
        def metric_row(name, value):
            return f"<tr><td>{name}</td><td>{value}</td></tr>"
        metrics_html = "".join([metric_row(k, v) for k, v in status.get("performance_metrics", {}).items()])
        events_html = "".join([
            f"<tr><td>{e['timestamp']}</td><td>{e['event_type']}</td><td>{e['metric_name']}</td><td>{e['current_value']}</td><td>{e['severity']}</td><td>{e['message']}</td></tr>"
            for e in status.get("recent_events", [])
        ])
        return f"""
        <html>
        <head>
            <title>Model Monitoring Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
                .critical {{ color: #c0392b; font-weight: bold; }}
                .high {{ color: #e67e22; font-weight: bold; }}
                .medium {{ color: #f1c40f; font-weight: bold; }}
                .low {{ color: #27ae60; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Model Monitoring Dashboard</h1>
            <h2>Status: {'<span style="color:green">Running</span>' if status.get('is_monitoring') else '<span style="color:red">Stopped</span>'}</h2>
            <h3>Performance Metrics</h3>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                {metrics_html}
            </table>
            <h3>Recent Events</h3>
            <table>
                <tr><th>Timestamp</th><th>Event Type</th><th>Metric</th><th>Value</th><th>Severity</th><th>Message</th></tr>
                {events_html}
            </table>
            <h3>Threshold Violations</h3>
            <p>Total: {status.get('threshold_violations', 0)}</p>
            <h3>Metric History Summary</h3>
            <pre>{json.dumps(status.get('metric_history_summary', {}), indent=2)}</pre>
        </body>
        </html>
        """

    def _render_debug_dashboard(self, event: Any) -> str:
        """
        Render a debugging dashboard for a triggered event.
        Args:
            event: MonitoringEvent or dict
        Returns:
            HTML string
        """
        if hasattr(event, '__dict__'):
            event = event.__dict__
        return f"""
        <html>
        <head>
            <title>Debugging Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #c0392b; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Debugging Dashboard</h1>
            <h2>Triggered Event</h2>
            <table>
                <tr><th>Field</th><th>Value</th></tr>
                {''.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in event.items()])}
            </table>
            <h3>Recommendations</h3>
            <ul>
                <li>Check model performance on recent data</li>
                <li>Verify data quality and preprocessing</li>
                <li>Review threshold configurations</li>
                <li>Consider model retraining if degradation persists</li>
            </ul>
            <p>Generated at {datetime.now().isoformat()}</p>
        </body>
        </html>
        """ 