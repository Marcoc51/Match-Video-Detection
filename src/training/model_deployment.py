"""
Model deployment script for cross detection models from MLflow registry.
This script provides utilities to load and deploy trained models.
"""

import mlflow
import mlflow.pytorch
from pathlib import Path
from ultralytics import YOLO
import yaml
import logging
from typing import Optional, Dict, Any
import shutil


class CrossDetectionModelDeployment:
    """
    Handles deployment of cross detection models from MLflow registry.
    """
    
    def __init__(self, config_path: str = "training_config.yaml"):
        """
        Initialize the model deployment handler.
        
        Args:
            config_path: Path to the training configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).resolve().parents[2]
        self.setup_mlflow()
        self.setup_logging()
        
    def setup_mlflow(self):
        """Setup MLflow tracking."""
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def list_registered_models(self) -> list:
        """
        List all registered models in the MLflow registry.
        
        Returns:
            List of registered model names
        """
        try:
            client = mlflow.tracking.MlflowClient()
            models = client.list_registered_models()
            return [model.name for model in models]
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    def get_latest_model_version(self, model_name: str) -> Optional[str]:
        """
        Get the latest version of a registered model.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Latest model version or None if not found
        """
        try:
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None"])
            if latest_version:
                return latest_version[0].version
            return None
        except Exception as e:
            self.logger.error(f"Failed to get latest model version: {e}")
            return None
    
    def load_model_from_registry(self, model_name: str, version: str = None) -> Optional[YOLO]:
        """
        Load a model from MLflow registry.
        
        Args:
            model_name: Name of the registered model
            version: Model version (if None, loads latest)
            
        Returns:
            Loaded YOLO model or None if failed
        """
        try:
            if version is None:
                version = self.get_latest_model_version(model_name)
                if version is None:
                    self.logger.error(f"No versions found for model: {model_name}")
                    return None
            
            # Load model from registry
            model_uri = f"models:/{model_name}/{version}"
            self.logger.info(f"Loading model from: {model_uri}")
            
            # Download model to local path
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(self.project_root / "temp_models")
            )
            
            # Load YOLO model
            model = YOLO(local_path)
            self.logger.info(f"Model loaded successfully: {model_name} v{version}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None
    
    def deploy_model(self, model_name: str, version: str = None, 
                    deployment_path: str = None) -> bool:
        """
        Deploy a model from registry to the production location.
        
        Args:
            model_name: Name of the registered model
            version: Model version (if None, deploys latest)
            deployment_path: Path to deploy the model (if None, uses default)
            
        Returns:
            True if deployment successful, False otherwise
        """
        try:
            # Load model from registry
            model = self.load_model_from_registry(model_name, version)
            if model is None:
                return False
            
            # Set deployment path
            if deployment_path is None:
                deployment_path = self.project_root / "models" / "yolo" / "cross_detection.pt"
            else:
                deployment_path = Path(deployment_path)
            
            # Create deployment directory
            deployment_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model to deployment location
            model.save(str(deployment_path))
            
            self.logger.info(f"Model deployed to: {deployment_path}")
            
            # Update the main model path in config if needed
            self.update_model_config(str(deployment_path))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {e}")
            return False
    
    def update_model_config(self, model_path: str):
        """
        Update the main configuration to use the new model.
        
        Args:
            model_path: Path to the deployed model
        """
        try:
            # Update the base model path in training config
            self.config['model']['base_model'] = \
                str(Path(model_path).relative_to(self.project_root))
            
            with open("training_config.yaml", 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            self.logger.info("Configuration updated with new model path")
            
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")
    
    def compare_models(self, model1_name: str, model2_name: str, 
                      model1_version: str = None, model2_version: str = None) \
                        -> Dict[str, Any]:
        """
        Compare two models from the registry.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            model1_version: Version of first model
            model2_version: Version of second model
            
        Returns:
            Comparison results
        """
        try:
            # Load both models
            model1 = self.load_model_from_registry(model1_name, model1_version)
            model2 = self.load_model_from_registry(model2_name, model2_version)
            
            if model1 is None or model2 is None:
                return {"error": "Failed to load one or both models"}
            
            # Get model information
            comparison = {
                "model1": {
                    "name": model1_name,
                    "version": model1_version or self.get_latest_model_version(model1_name),
                    "parameters": sum(p.numel() for p in model1.model.parameters())
                },
                "model2": {
                    "name": model2_name,
                    "version": model2_version or self.get_latest_model_version(model2_name),
                    "parameters": sum(p.numel() for p in model2.model.parameters())
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare models: {e}")
            return {"error": str(e)}
    
    def get_model_metrics(self, model_name: str, version: str = None) -> Dict[str, Any]:
        """
        Get metrics for a specific model version.
        
        Args:
            model_name: Name of the registered model
            version: Model version (if None, gets latest)
            
        Returns:
            Model metrics
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            if version is None:
                version = self.get_latest_model_version(model_name)
            
            # Get run information
            model_uri = f"models:/{model_name}/{version}"
            run_id = client.get_model_version(model_name, version).run_id
            run = client.get_run(run_id)
            
            metrics = {
                "model_name": model_name,
                "version": version,
                "run_id": run_id,
                "metrics": run.data.metrics,
                "parameters": run.data.params,
                "tags": run.data.tags
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get model metrics: {e}")
            return {"error": str(e)}


def main():
    """Main function for model deployment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy cross detection models")
    parser.add_argument("--action", choices=["list", "deploy", "compare", "metrics"], 
                       required=True, help="Action to perform")
    parser.add_argument("--model-name", help="Name of the model")
    parser.add_argument("--version", help="Model version")
    parser.add_argument("--deployment-path", help="Path to deploy model")
    parser.add_argument("--model2-name", help="Second model name for comparison")
    parser.add_argument("--model2-version", help="Second model version for comparison")
    
    args = parser.parse_args()
    
    deployer = CrossDetectionModelDeployment()
    
    if args.action == "list":
        models = deployer.list_registered_models()
        print("Registered models:")
        for model in models:
            print(f"  - {model}")
    
    elif args.action == "deploy":
        if not args.model_name:
            print("Error: --model-name is required for deployment")
            return
        
        success = deployer.deploy_model(args.model_name, args.version, args.deployment_path)
        if success:
            print("Model deployed successfully!")
        else:
            print("Model deployment failed!")
    
    elif args.action == "compare":
        if not args.model_name or not args.model2_name:
            print("Error: --model-name and --model2-name are required for comparison")
            return
        
        comparison = deployer.compare_models(
            args.model_name, args.model2_name, 
            args.version, args.model2_version
        )
        print("Model comparison:")
        print(comparison)
    
    elif args.action == "metrics":
        if not args.model_name:
            print("Error: --model-name is required for metrics")
            return
        
        metrics = deployer.get_model_metrics(args.model_name, args.version)
        print("Model metrics:")
        print(metrics)


if __name__ == "__main__":
    main() 