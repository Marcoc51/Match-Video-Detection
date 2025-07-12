"""
Alert Manager for Model Monitoring

Handles sending alerts and notifications through various channels:
- Email notifications
- Webhook notifications
- Slack notifications
- Log-based alerts
"""

import os
import sys
import logging
import json
import smtplib
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .model_monitor import MonitoringEvent

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages alert sending through various notification channels.
    
    Supports:
    - Email notifications
    - Webhook notifications
    - Slack notifications
    - Log-based alerts
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the alert manager.
        
        Args:
            config: Alert configuration dictionary
        """
        self.config = config
        self.alert_history: List[Dict[str, Any]] = []
        self.sent_alerts_count = 0
        
        # Email configuration
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'smtp_username': os.getenv('SMTP_USERNAME', ''),
            'smtp_password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('FROM_EMAIL', 'alerts@fcmasar.com'),
            'to_emails': os.getenv('TO_EMAILS', '').split(',') if os.getenv('TO_EMAILS') else []
        }
        
        # Webhook configuration
        self.webhook_config = {
            'url': os.getenv('WEBHOOK_URL', ''),
            'headers': {
                'Content-Type': 'application/json',
                'Authorization': f"Bearer {os.getenv('WEBHOOK_TOKEN', '')}"
            }
        }
        
        # Slack configuration
        self.slack_config = {
            'webhook_url': os.getenv('SLACK_WEBHOOK_URL', ''),
            'channel': os.getenv('SLACK_CHANNEL', '#alerts'),
            'username': os.getenv('SLACK_USERNAME', 'Model Monitor')
        }
        
        logger.info("Alert manager initialized")
    
    def send_alert(self, event: MonitoringEvent) -> Dict[str, Any]:
        """
        Send alert for a monitoring event.
        
        Args:
            event: Monitoring event to alert about
            
        Returns:
            Dictionary with alert sending results
        """
        alert_result = {
            'event_id': f"{event.timestamp.isoformat()}_{event.metric_name}",
            'timestamp': datetime.now().isoformat(),
            'event': asdict(event),
            'channels': {}
        }
        
        # Send through configured channels
        if self.config.get('email_enabled', False):
            alert_result['channels']['email'] = self._send_email_alert(event)
        
        if self.config.get('webhook_enabled', False):
            alert_result['channels']['webhook'] = self._send_webhook_alert(event)
        
        if self.config.get('slack_enabled', False):
            alert_result['channels']['slack'] = self._send_slack_alert(event)
        
        # Always log the alert
        alert_result['channels']['log'] = self._log_alert(event)
        
        # Store in history
        self.alert_history.append(alert_result)
        self.sent_alerts_count += 1
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        logger.info(f"Alert sent for {event.event_type}: {event.message}")
        return alert_result
    
    def _send_email_alert(self, event: MonitoringEvent) -> Dict[str, Any]:
        """Send email alert."""
        try:
            if not self.email_config['to_emails']:
                return {'success': False, 'error': 'No recipient emails configured'}
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])
            msg['Subject'] = f"Model Monitor Alert: {event.severity.upper()} - {event.event_type}"
            
            # Create email body
            body = self._create_email_body(event)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                if self.email_config['smtp_username'] and self.email_config['smtp_password']:
                    server.login(self.email_config['smtp_username'], self.email_config['smtp_password'])
                server.send_message(msg)
            
            return {'success': True, 'recipients': self.email_config['to_emails']}
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_email_body(self, event: MonitoringEvent) -> str:
        """Create HTML email body."""
        severity_colors = {
            'low': '#ffeb3b',
            'medium': '#ff9800',
            'high': '#f44336',
            'critical': '#9c27b0'
        }
        
        color = severity_colors.get(event.severity, '#2196f3')
        
        return f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert {{ border-left: 5px solid {color}; padding: 10px; background-color: #f9f9f9; }}
                .severity {{ color: {color}; font-weight: bold; }}
                .metric {{ font-family: monospace; background-color: #f0f0f0; padding: 2px 4px; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="alert">
                <h2>🚨 Model Monitoring Alert</h2>
                <p><strong>Severity:</strong> <span class="severity">{event.severity.upper()}</span></p>
                <p><strong>Event Type:</strong> {event.event_type}</p>
                <p><strong>Metric:</strong> <span class="metric">{event.metric_name}</span></p>
                <p><strong>Current Value:</strong> {event.current_value:.4f}</p>
                <p><strong>Threshold:</strong> {event.threshold_value:.4f}</p>
                <p><strong>Message:</strong> {event.message}</p>
                <p class="timestamp">Timestamp: {event.timestamp.isoformat()}</p>
                {f'<p><strong>Job ID:</strong> {event.job_id}</p>' if event.job_id else ''}
                {f'<p><strong>Model Version:</strong> {event.model_version}</p>' if event.model_version else ''}
            </div>
            <p>This alert was automatically generated by the Football Video Analysis Model Monitor.</p>
        </body>
        </html>
        """
    
    def _send_webhook_alert(self, event: MonitoringEvent) -> Dict[str, Any]:
        """Send webhook alert."""
        try:
            if not self.webhook_config['url']:
                return {'success': False, 'error': 'No webhook URL configured'}
            
            # Prepare payload
            payload = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'severity': event.severity,
                'metric_name': event.metric_name,
                'current_value': event.current_value,
                'threshold_value': event.threshold_value,
                'message': event.message,
                'job_id': event.job_id,
                'model_version': event.model_version,
                'metadata': event.metadata or {}
            }
            
            # Send webhook
            response = requests.post(
                self.webhook_config['url'],
                json=payload,
                headers=self.webhook_config['headers'],
                timeout=10
            )
            
            if response.status_code in [200, 201, 202]:
                return {'success': True, 'status_code': response.status_code}
            else:
                return {
                    'success': False, 
                    'error': f'HTTP {response.status_code}: {response.text}'
                }
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_slack_alert(self, event: MonitoringEvent) -> Dict[str, Any]:
        """Send Slack alert."""
        try:
            if not self.slack_config['webhook_url']:
                return {'success': False, 'error': 'No Slack webhook URL configured'}
            
            # Prepare Slack message
            severity_emoji = {
                'low': '🟡',
                'medium': '🟠',
                'high': '🔴',
                'critical': '🟣'
            }
            
            emoji = severity_emoji.get(event.severity, '🔵')
            
            slack_payload = {
                'channel': self.slack_config['channel'],
                'username': self.slack_config['username'],
                'text': f"{emoji} *Model Monitor Alert*",
                'attachments': [{
                    'color': self._get_slack_color(event.severity),
                    'fields': [
                        {
                            'title': 'Severity',
                            'value': event.severity.upper(),
                            'short': True
                        },
                        {
                            'title': 'Event Type',
                            'value': event.event_type,
                            'short': True
                        },
                        {
                            'title': 'Metric',
                            'value': f"`{event.metric_name}`",
                            'short': True
                        },
                        {
                            'title': 'Current Value',
                            'value': f"{event.current_value:.4f}",
                            'short': True
                        },
                        {
                            'title': 'Threshold',
                            'value': f"{event.threshold_value:.4f}",
                            'short': True
                        },
                        {
                            'title': 'Message',
                            'value': event.message,
                            'short': False
                        }
                    ],
                    'footer': 'Football Video Analysis Model Monitor',
                    'ts': int(event.timestamp.timestamp())
                }]
            }
            
            # Add optional fields
            if event.job_id:
                slack_payload['attachments'][0]['fields'].append({
                    'title': 'Job ID',
                    'value': event.job_id,
                    'short': True
                })
            
            if event.model_version:
                slack_payload['attachments'][0]['fields'].append({
                    'title': 'Model Version',
                    'value': event.model_version,
                    'short': True
                })
            
            # Send to Slack
            response = requests.post(
                self.slack_config['webhook_url'],
                json=slack_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return {'success': True}
            else:
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.text}'
                }
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_slack_color(self, severity: str) -> str:
        """Get Slack color for severity level."""
        colors = {
            'low': '#ffeb3b',
            'medium': '#ff9800',
            'high': '#f44336',
            'critical': '#9c27b0'
        }
        return colors.get(severity, '#2196f3')
    
    def _log_alert(self, event: MonitoringEvent) -> Dict[str, Any]:
        """Log alert to system logs."""
        log_level = {
            'low': 'INFO',
            'medium': 'WARNING',
            'high': 'ERROR',
            'critical': 'CRITICAL'
        }.get(event.severity, 'INFO')
        
        log_message = (
            f"ALERT [{event.severity.upper()}] {event.event_type}: "
            f"{event.metric_name}={event.current_value:.4f} "
            f"(threshold={event.threshold_value:.4f}) - {event.message}"
        )
        
        if event.job_id:
            log_message += f" [Job: {event.job_id}]"
        
        if event.model_version:
            log_message += f" [Model: {event.model_version}]"
        
        # Log with appropriate level
        if log_level == 'CRITICAL':
            logger.critical(log_message)
        elif log_level == 'ERROR':
            logger.error(log_message)
        elif log_level == 'WARNING':
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        return {'success': True, 'log_level': log_level}
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'successful_alerts': 0,
                'failed_alerts': 0,
                'alerts_by_severity': {},
                'alerts_by_channel': {}
            }
        
        # Calculate statistics
        successful_alerts = sum(
            1 for alert in self.alert_history
            if any(channel.get('success', False) for channel in alert['channels'].values())
        )
        
        alerts_by_severity = {}
        alerts_by_channel = {}
        
        for alert in self.alert_history:
            severity = alert['event']['severity']
            alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
            
            for channel_name, channel_result in alert['channels'].items():
                alerts_by_channel[channel_name] = alerts_by_channel.get(channel_name, 0) + 1
        
        return {
            'total_alerts': len(self.alert_history),
            'successful_alerts': successful_alerts,
            'failed_alerts': len(self.alert_history) - successful_alerts,
            'alerts_by_severity': alerts_by_severity,
            'alerts_by_channel': alerts_by_channel,
            'sent_alerts_count': self.sent_alerts_count
        }
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def clear_alert_history(self) -> None:
        """Clear alert history."""
        self.alert_history.clear()
        self.sent_alerts_count = 0
        logger.info("Alert history cleared") 