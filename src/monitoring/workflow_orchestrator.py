"""
Workflow Orchestrator for Model Monitoring

Handles conditional workflow execution based on monitoring events:
- Auto-retraining
- Model switching
- Debugging dashboard generation
- Performance optimization
"""

import os
import sys
import logging
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .entities import MonitoringEvent

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Orchestrates conditional workflow execution based on monitoring events.
    
    Supports:
    - Auto-retraining workflows
    - Model switching workflows
    - Debugging dashboard generation
    - Performance optimization workflows
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the workflow orchestrator.
        
        Args:
            config: Workflow configuration dictionary
        """
        self.config = config
        self.active_workflows: List[Dict[str, Any]] = []
        self.workflow_history: List[Dict[str, Any]] = []
        self.workflow_triggers = {
            'accuracy_below_threshold': self._check_accuracy_threshold,
            'confidence_below_threshold': self._check_confidence_threshold,
            'critical_accuracy_drop': self._check_critical_accuracy_drop,
            'model_failure': self._check_model_failure,
            'performance_degradation': self._check_performance_degradation,
            'error_spike': self._check_error_spike
        }
        
        logger.info("Workflow orchestrator initialized")
    
    def check_and_trigger_workflows(self, event: MonitoringEvent) -> List[Dict[str, Any]]:
        """
        Check if any workflows should be triggered based on the event.
        
        Args:
            event: Monitoring event that might trigger workflows
            
        Returns:
            List of triggered workflows
        """
        triggered_workflows = []
        
        for workflow_name, workflow_config in self.config.items():
            if not workflow_config.get('enabled', False):
                continue
            
            trigger_conditions = workflow_config.get('trigger_conditions', [])
            
            for condition in trigger_conditions:
                if self._should_trigger_workflow(event, condition, workflow_config):
                    workflow_result = self._execute_workflow(workflow_name, workflow_config, event)
                    if workflow_result:
                        triggered_workflows.append(workflow_result)
                    break  # Only trigger once per workflow
        
        return triggered_workflows
    
    def _should_trigger_workflow(self, 
                                event: MonitoringEvent, 
                                condition: str, 
                                workflow_config: Dict[str, Any]) -> bool:
        """
        Check if a workflow should be triggered based on the condition.
        
        Args:
            event: Monitoring event
            condition: Trigger condition
            workflow_config: Workflow configuration
            
        Returns:
            True if workflow should be triggered
        """
        if condition not in self.workflow_triggers:
            logger.warning(f"Unknown trigger condition: {condition}")
            return False
        
        return self.workflow_triggers[condition](event, workflow_config)
    
    def _check_accuracy_threshold(self, event: MonitoringEvent, config: Dict[str, Any]) -> bool:
        """Check if accuracy is below threshold."""
        if event.metric_name != 'detection_accuracy':
            return False
        
        threshold = config.get('min_accuracy_threshold', 0.8)
        return event.current_value < threshold and event.severity in ['high', 'critical']
    
    def _check_confidence_threshold(self, event: MonitoringEvent, config: Dict[str, Any]) -> bool:
        """Check if confidence is below threshold."""
        if event.metric_name != 'average_confidence':
            return False
        
        threshold = config.get('min_confidence_threshold', 0.6)
        return event.current_value < threshold and event.severity in ['high', 'critical']
    
    def _check_critical_accuracy_drop(self, event: MonitoringEvent, config: Dict[str, Any]) -> bool:
        """Check for critical accuracy drop."""
        if event.metric_name != 'detection_accuracy':
            return False
        
        return event.severity == 'critical' and event.current_value < 0.5
    
    def _check_model_failure(self, event: MonitoringEvent, config: Dict[str, Any]) -> bool:
        """Check for model failure."""
        return (event.event_type == 'model_failure' or 
                (event.metric_name == 'error_rate' and event.current_value > 0.5))
    
    def _check_performance_degradation(self, event: MonitoringEvent, config: Dict[str, Any]) -> bool:
        """Check for performance degradation."""
        performance_metrics = ['processing_speed', 'model_latency', 'system_cpu_usage']
        return (event.metric_name in performance_metrics and 
                event.severity in ['high', 'critical'])
    
    def _check_error_spike(self, event: MonitoringEvent, config: Dict[str, Any]) -> bool:
        """Check for error spike."""
        return (event.metric_name == 'error_rate' and 
                event.current_value > 0.2 and 
                event.severity in ['high', 'critical'])
    
    def _execute_workflow(self, 
                         workflow_name: str, 
                         workflow_config: Dict[str, Any], 
                         trigger_event: MonitoringEvent) -> Optional[Dict[str, Any]]:
        """
        Execute a workflow.
        
        Args:
            workflow_name: Name of the workflow
            workflow_config: Workflow configuration
            trigger_event: Event that triggered the workflow
            
        Returns:
            Workflow execution result
        """
        workflow_result = {
            'workflow_name': workflow_name,
            'trigger_event': trigger_event.event_type,
            'trigger_metric': trigger_event.metric_name,
            'trigger_value': trigger_event.current_value,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'result': None,
            'error': None
        }
        
        try:
            logger.info(f"Executing workflow: {workflow_name}")
            
            if workflow_name == 'auto_retraining':
                result = self._execute_auto_retraining(workflow_config, trigger_event)
            elif workflow_name == 'model_switching':
                result = self._execute_model_switching(workflow_config, trigger_event)
            elif workflow_name == 'debugging_dashboard':
                result = self._execute_debugging_dashboard(workflow_config, trigger_event)
            else:
                logger.warning(f"Unknown workflow: {workflow_name}")
                return None
            
            workflow_result['status'] = 'completed'
            workflow_result['result'] = result
            workflow_result['end_time'] = datetime.now().isoformat()
            
            # Add to active workflows
            self.active_workflows.append(workflow_result)
            
            # Add to history
            self.workflow_history.append(workflow_result)
            
            # Keep only last 100 workflows in history
            if len(self.workflow_history) > 100:
                self.workflow_history = self.workflow_history[-100:]
            
            logger.info(f"Workflow {workflow_name} completed successfully")
            return workflow_result
            
        except Exception as e:
            workflow_result['status'] = 'failed'
            workflow_result['error'] = str(e)
            workflow_result['end_time'] = datetime.now().isoformat()
            
            logger.error(f"Workflow {workflow_name} failed: {e}")
            return workflow_result
    
    def _execute_auto_retraining(self, config: Dict[str, Any], event: MonitoringEvent) -> Dict[str, Any]:
        """Execute auto-retraining workflow."""
        logger.info("Starting auto-retraining workflow")
        
        # Prepare retraining command
        retraining_script = config.get('retraining_script', 'train_cross_detection.py')
        retraining_args = config.get('retraining_args', [])
        
        cmd = ['python', retraining_script] + retraining_args
        
        # Execute retraining in background thread
        def run_retraining():
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=config.get('timeout', 3600)  # 1 hour timeout
                )
                
                if result.returncode == 0:
                    logger.info("Auto-retraining completed successfully")
                else:
                    logger.error(f"Auto-retraining failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error("Auto-retraining timed out")
            except Exception as e:
                logger.error(f"Auto-retraining error: {e}")
        
        # Start retraining in background
        retraining_thread = threading.Thread(target=run_retraining, daemon=True)
        retraining_thread.start()
        
        return {
            'type': 'auto_retraining',
            'command': ' '.join(cmd),
            'thread_id': retraining_thread.ident,
            'status': 'started'
        }
    
    def _execute_model_switching(self, config: Dict[str, Any], event: MonitoringEvent) -> Dict[str, Any]:
        """Execute model switching workflow."""
        logger.info("Starting model switching workflow")
        
        fallback_model = config.get('fallback_model', 'models/yolo/fallback.pt')
        current_model = config.get('current_model', 'models/yolo/best.pt')
        
        # Check if fallback model exists
        fallback_path = Path(fallback_model)
        if not fallback_path.exists():
            logger.error(f"Fallback model not found: {fallback_model}")
            return {
                'type': 'model_switching',
                'status': 'failed',
                'error': 'Fallback model not found'
            }
        
        # Create backup of current model
        backup_path = Path(f"{current_model}.backup")
        if Path(current_model).exists():
            import shutil
            shutil.copy2(current_model, backup_path)
        
        # Switch to fallback model
        try:
            import shutil
            shutil.copy2(fallback_model, current_model)
            
            logger.info(f"Switched to fallback model: {fallback_model}")
            
            return {
                'type': 'model_switching',
                'fallback_model': fallback_model,
                'current_model': current_model,
                'backup_created': backup_path.exists(),
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Model switching failed: {e}")
            return {
                'type': 'model_switching',
                'status': 'failed',
                'error': str(e)
            }
    
    def _execute_debugging_dashboard(self, config: Dict[str, Any], event: MonitoringEvent) -> Dict[str, Any]:
        """Execute debugging dashboard generation workflow."""
        logger.info("Starting debugging dashboard generation")
        
        dashboard_script = config.get('dashboard_script', 'scripts/generate_debug_dashboard.py')
        output_dir = config.get('output_dir', 'outputs/debug_dashboards')
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate dashboard
        try:
            # Import dashboard generator if available
            try:
                from .dashboard_generator import DashboardGenerator
                dashboard_gen = DashboardGenerator(config.get('dashboard', {}))
                
                # Generate debugging dashboard
                dashboard_path = dashboard_gen.generate_debug_dashboard(
                    event=event,
                    output_dir=output_dir
                )
                
                logger.info(f"Debugging dashboard generated: {dashboard_path}")
                
                return {
                    'type': 'debugging_dashboard',
                    'dashboard_path': str(dashboard_path),
                    'output_dir': output_dir,
                    'status': 'completed'
                }
                
            except ImportError:
                # Fallback: create simple debug report
                debug_report = self._create_simple_debug_report(event, output_dir)
                
                return {
                    'type': 'debugging_dashboard',
                    'debug_report': debug_report,
                    'output_dir': output_dir,
                    'status': 'completed'
                }
                
        except Exception as e:
            logger.error(f"Debugging dashboard generation failed: {e}")
            return {
                'type': 'debugging_dashboard',
                'status': 'failed',
                'error': str(e)
            }
    
    def _create_simple_debug_report(self, event: MonitoringEvent, output_dir: str) -> str:
        """Create a simple debug report."""
        report_path = Path(output_dir) / f"debug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        debug_data = {
            'timestamp': datetime.now().isoformat(),
            'trigger_event': {
                'event_type': event.event_type,
                'severity': event.severity,
                'metric_name': event.metric_name,
                'current_value': event.current_value,
                'threshold_value': event.threshold_value,
                'message': event.message
            },
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': str(Path.cwd())
            },
            'recommendations': [
                "Check model performance on recent data",
                "Verify data quality and preprocessing",
                "Review threshold configurations",
                "Consider model retraining if degradation persists"
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        return str(report_path)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            'active_workflows': len(self.active_workflows),
            'total_workflows': len(self.workflow_history),
            'recent_workflows': self.workflow_history[-10:] if self.workflow_history else [],
            'workflow_config': self.config
        }
    
    def get_workflow_statistics(self, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get workflow statistics for a time period."""
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        # Filter workflows by time
        filtered_workflows = [
            w for w in self.workflow_history
            if start_time <= datetime.fromisoformat(w['start_time']) <= end_time
        ]
        
        # Calculate statistics
        workflows_by_type = {}
        workflows_by_status = {}
        
        for workflow in filtered_workflows:
            workflow_type = workflow.get('result', {}).get('type', 'unknown')
            status = workflow['status']
            
            workflows_by_type[workflow_type] = workflows_by_type.get(workflow_type, 0) + 1
            workflows_by_status[status] = workflows_by_status.get(status, 0) + 1
        
        return {
            'total_workflows': len(filtered_workflows),
            'workflows_by_type': workflows_by_type,
            'workflows_by_status': workflows_by_status,
            'period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            }
        }
    
    def stop_workflow(self, workflow_name: str) -> bool:
        """Stop a running workflow."""
        for workflow in self.active_workflows:
            if workflow['workflow_name'] == workflow_name and workflow['status'] == 'running':
                workflow['status'] = 'stopped'
                workflow['end_time'] = datetime.now().isoformat()
                logger.info(f"Stopped workflow: {workflow_name}")
                return True
        
        return False
    
    def clear_workflow_history(self) -> None:
        """Clear workflow history."""
        self.workflow_history.clear()
        self.active_workflows.clear()
        logger.info("Workflow history cleared") 