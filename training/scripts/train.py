"""
Main training script for cross detection model with MLflow integration.
This script fine-tunes a YOLO model to detect crosses in football videos.
"""

import sys
import yaml
import mlflow
import mlflow.pytorch
from pathlib import Path
from ultralytics import YOLO
import argparse
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.training.data_preparation import CrossDataPreparation


class CrossDetectionTrainer:
    """
    Trainer class for cross detection model with MLflow integration.
    """
    
    def __init__(self, config_path: str = "training_config.yaml"):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to the training configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(__file__).parent.parent.parent
        self.setup_mlflow()
        self.setup_logging()
        
    def setup_mlflow(self):
        """Setup MLflow tracking and registry."""
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        
        # Set experiment
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        # Enable MLflow logging for PyTorch
        mlflow.pytorch.autolog()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self):
        """Prepare training data."""
        self.logger.info("Preparing training data...")
        
        data_prep = CrossDataPreparation("training_config.yaml")
        data_prep.prepare_dataset()
        
        self.dataset_path = data_prep.output_dir / "dataset.yaml"
        self.logger.info(f"Dataset prepared at: {self.dataset_path}")
        
    def train_model(self):
        """Train the cross detection model."""
        self.logger.info("Starting model training...")
        
        # Start MLflow run
        with mlflow.start_run(
            run_name=f"cross-detection-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ):
            
            # Log parameters
            mlflow.log_params({
                'epochs': self.config['training']['epochs'],
                'batch_size': self.config['training']['batch_size'],
                'learning_rate': self.config['training']['learning_rate'],
                'weight_decay': self.config['training']['weight_decay'],
                'image_size': self.config['data']['image_size'],
                'num_classes': self.config['model']['num_classes']
            })
            
            # Log configuration file
            mlflow.log_artifact("training_config.yaml")
            
            # Load base model
            base_model_path = self.project_root / self.config['model']['base_model']
            self.logger.info(f"Loading base model from: {base_model_path}")
            
            model = YOLO(str(base_model_path))
            
            # Train the model
            self.logger.info("Starting training...")
            
            results = model.train(
                data=str(self.dataset_path),
                epochs=self.config['training']['epochs'],
                batch=self.config['training']['batch_size'],
                imgsz=self.config['data']['image_size'],
                lr0=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay'],
                patience=self.config['training']['patience'],
                save_period=self.config['training']['save_period'],
                project=str(self.project_root / "training_outputs"),
                name="cross_detection_model",
                exist_ok=True,
                verbose=self.config['logging']['verbose']
            )
            
            # Log training results
            self.logger.info("Training completed!")
            
            # Log metrics
            if hasattr(results, 'results_dict'):
                for metric_name, metric_value in results.results_dict.items():
                    if isinstance(metric_value, (int, float)):
                        # Clean metric name for MLflow (remove parentheses and special chars)
                        clean_metric_name = metric_name.replace('(', '').replace(')', '').replace('/', '_')
                        mlflow.log_metric(clean_metric_name, metric_value)
            
            # Log the best model
            best_model_path = results.save_dir / "weights" / "best.pt"
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), "model")
                
                # Register model in MLflow Model Registry
                self.register_model(str(best_model_path))
            
            # Log training artifacts
            mlflow.log_artifacts(str(results.save_dir), "training_outputs")
            
            self.logger.info("Training run completed and logged to MLflow!")
            
    def register_model(self, model_path: str):
        """Register the trained model in MLflow Model Registry."""
        try:
            # Register model
            model_name = self.config['mlflow']['model_registry_name']
            
            # Log model with MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model_path,
                artifact_path="model",
                registered_model_name=model_name
            )
            
            self.logger.info(f"Model registered as: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
    
    def evaluate_model(self, model_path: str = None):
        """Evaluate the trained model on test data."""
        self.logger.info("Evaluating model...")
        
        if model_path is None:
            # Use the best model from training
            model_path = self.project_root / "training_outputs" / \
                "cross_detection_model" / "weights" / "best.pt"
        
        if not Path(model_path).exists():
            self.logger.error(f"Model not found at: {model_path}")
            return
        
        # Set dataset path if not already set
        if not hasattr(self, 'dataset_path'):
            self.dataset_path = self.project_root / self.config['data']['output_dir'] / "dataset.yaml"
        
        if not self.dataset_path.exists():
            self.logger.error(f"Dataset not found at: {self.dataset_path}")
            self.logger.error("Please run data preparation first: python train_cross_detection.py --prepare-data-only")
            return
        
        # Load model
        model = YOLO(str(model_path))
        
        # Evaluate on test data
        results = model.val(data=str(self.dataset_path))
        
        # Log evaluation metrics
        with mlflow.start_run(
            run_name=f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ):
            mlflow.log_metrics({
                'mAP50': results.box.map50,
                'mAP50-95': results.box.map,
                'precision': results.box.mp,
                'recall': results.box.mr
            })
            
            self.logger.info("Evaluation completed and logged to MLflow!")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        try:
            # Prepare data
            self.prepare_data()
            
            # Train model
            self.train_model()
            
            # Evaluate model
            self.evaluate_model()
            
            self.logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description="Train cross detection model")
    parser.add_argument("--config", default="training_config.yaml", 
                       help="Path to training configuration file")
    parser.add_argument("--prepare-data-only", action="store_true",
                       help="Only prepare data, don't train")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Only evaluate existing model")
    parser.add_argument("--model-path", help="Path to model for evaluation")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = CrossDetectionTrainer(args.config)
    
    if args.prepare_data_only:
        trainer.prepare_data()
    elif args.evaluate_only:
        trainer.evaluate_model(args.model_path)
    else:
        trainer.run_training_pipeline()


if __name__ == "__main__":
    main() 