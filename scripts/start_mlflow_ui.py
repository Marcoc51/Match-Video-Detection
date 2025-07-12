#!/usr/bin/env python3
"""
Script to start MLflow UI for viewing experiments and model registry.
"""

import subprocess
import sys
import yaml
from pathlib import Path

def start_mlflow_ui():
    """Start MLflow UI server."""
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "training" / "config" / "training_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get tracking URI
        tracking_uri = config['mlflow']['tracking_uri']
        
        print("Starting MLflow UI...")
        print(f"Tracking URI: {tracking_uri}")
        print("Open browser to: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        
        # Start MLflow UI
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", tracking_uri,
            "--host", "0.0.0.0",
            "--port", "5000"
        ])
        
    except KeyboardInterrupt:
        print("\nMLflow UI stopped.")
    except Exception as e:
        print(f"Error starting MLflow UI: {e}")
        print("Make sure MLflow is installed: pip install mlflow")

if __name__ == "__main__":
    start_mlflow_ui() 