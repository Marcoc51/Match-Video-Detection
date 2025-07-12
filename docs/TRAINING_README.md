# Cross Detection Model Training Pipeline

This document explains how to fine-tune the YOLO model to detect crosses in football match videos using MLflow for experiment tracking and model registry.

## 🎯 Overview

The training pipeline consists of:
1. **Data Preparation**: Extract frames from videos and create YOLO annotations
2. **Model Training**: Fine-tune the existing YOLO model for cross detection
3. **Experiment Tracking**: Use MLflow to track experiments and metrics
4. **Model Registry**: Store and version trained models
5. **Model Deployment**: Deploy trained models to production

## 📁 Project Structure

```
Match-Video-Detection/
├── training_config.yaml          # Training configuration
├── train_cross_detection.py      # Main training script
├── src/training/
│   ├── data_preparation.py       # Data preparation utilities
│   └── model_deployment.py       # Model deployment utilities
├── Videos-Input/
│   ├── Crosses-Trained/          # Training videos (6 videos)
│   └── Crosses-Tested/           # Test videos (2 videos)
├── training_outputs/             # Generated training data and models
└── mlruns.db                     # MLflow tracking database
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install additional dependencies
pip install mlflow==2.12.0
```

### 2. Prepare Training Data

```bash
# Extract frames and create annotations
python train_cross_detection.py --prepare-data-only
```

This will:
- Extract frames from videos in `Videos-Input/Crosses-Trained/` and `Videos-Input/Crosses-Tested/`
- Detect crosses using the existing pipeline
- Create YOLO format annotations
- Split data into train/validation sets
- Generate `training_outputs/dataset.yaml`

### 3. Train the Model

```bash
# Run complete training pipeline
python train_cross_detection.py
```

This will:
- Load the base model from `models/yolo/best.pt`
- Fine-tune it for cross detection
- Track experiments with MLflow
- Register the best model in MLflow Model Registry
- Save training outputs to `training_outputs/`

### 4. Evaluate the Model

```bash
# Evaluate the trained model
python train_cross_detection.py --evaluate-only
```

## ⚙️ Configuration

Edit `training_config.yaml` to customize the training process:

### MLflow Configuration
```yaml
mlflow:
  tracking_uri: "sqlite:///mlruns.db"  # Local SQLite database
  experiment_name: "cross-detection-finetuning"
  model_registry_name: "cross-detection-model"
```

### Training Parameters
```yaml
training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.001
  weight_decay: 0.0005
  patience: 20  # Early stopping
```

### Data Configuration
```yaml
data:
  train_videos_dir: "Videos-Input/Crosses-Trained"
  test_videos_dir: "Videos-Input/Crosses-Tested"
  frames_per_second: 1  # Extract 1 frame per second
  image_size: 640       # YOLO input size
```

## 📊 MLflow Integration

### Experiment Tracking

MLflow automatically tracks:
- **Parameters**: Learning rate, batch size, epochs, etc.
- **Metrics**: mAP50, mAP50-95, precision, recall
- **Artifacts**: Model files, training logs, configuration
- **Code**: Git commit hash and code state

### View Experiments

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

### Model Registry

Trained models are automatically registered in MLflow Model Registry:

```bash
# List registered models
python src/training/model_deployment.py --action list

# Get model metrics
python src/training/model_deployment.py --action metrics --model-name cross-detection-model

# Compare models
python src/training/model_deployment.py --action compare --model-name model1 --model2-name model2
```

## 🚀 Model Deployment

### Deploy from Registry

```bash
# Deploy latest model to production
python src/training/model_deployment.py --action deploy --model-name cross-detection-model

# Deploy specific version
python src/training/model_deployment.py --action deploy --model-name cross-detection-model --version 1

# Deploy to custom path
python src/training/model_deployment.py --action deploy --model-name cross-detection-model --deployment-path models/yolo/cross_detection.pt
```

### Use in Production

After deployment, update your main pipeline to use the new model:

```python
# In your main.py or API
model_path = project_root / "models" / "yolo" / "cross_detection.pt"
model = YOLO(str(model_path))
```

## 📈 Monitoring and Evaluation

### Training Metrics

The training process tracks:
- **mAP50**: Mean Average Precision at IoU=0.5
- **mAP50-95**: Mean Average Precision across IoU thresholds
- **Precision**: Precision of cross detection
- **Recall**: Recall of cross detection
- **Training Loss**: Model training loss over epochs

### Validation Metrics

Validation is performed on a held-out test set:
- 20% of data is used for validation
- Metrics are logged to MLflow
- Best model is selected based on validation performance

## 🔧 Advanced Usage

### Custom Data Preparation

Modify `src/training/data_preparation.py` to:
- Change frame extraction rate
- Implement custom cross detection logic
- Add data augmentation
- Modify annotation format

### Hyperparameter Tuning

Use MLflow for hyperparameter optimization:

```python
import mlflow
from itertools import product

# Define parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [8, 16, 32],
    'epochs': [50, 100, 200]
}

# Grid search
for lr, bs, epochs in product(param_grid['learning_rate'], 
                             param_grid['batch_size'], 
                             param_grid['epochs']):
    with mlflow.start_run():
        mlflow.log_params({
            'learning_rate': lr,
            'batch_size': bs,
            'epochs': epochs
        })
        # Run training with these parameters
        # ...
```

### Model Comparison

Compare different model versions:

```bash
# Compare two model versions
python src/training/model_deployment.py --action compare \
    --model-name cross-detection-model \
    --model2-name cross-detection-model \
    --version 1 \
    --model2-version 2
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size in `training_config.yaml`
2. **Slow Training**: Use GPU if available, reduce image size
3. **Poor Performance**: Increase training data, adjust learning rate
4. **MLflow Errors**: Check tracking URI and experiment name

### Debug Mode

Enable verbose logging:

```yaml
logging:
  verbose: true
```

### Check Data Quality

```bash
# Verify data preparation
python src/training/data_preparation.py
```

## 📝 Best Practices

1. **Data Quality**: Ensure training videos contain clear crosses
2. **Regular Evaluation**: Evaluate on test set regularly
3. **Version Control**: Use MLflow to track all experiments
4. **Model Registry**: Register all production models
5. **Documentation**: Document model changes and performance

## 🔄 Continuous Training

Set up automated retraining:

1. **Data Pipeline**: Automatically process new videos
2. **Training Pipeline**: Retrain when new data is available
3. **Evaluation Pipeline**: Automatically evaluate new models
4. **Deployment Pipeline**: Deploy models that meet criteria

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [YOLO Training Guide](https://docs.ultralytics.com/guides/training/)

## 🤝 Contributing

When contributing to the training pipeline:

1. Update configuration files
2. Add new data preparation methods
3. Implement new evaluation metrics
4. Document changes in this README
5. Test on sample data before full training 