# Mage.ai Workflow Orchestration for Football Video Analysis

This directory contains the complete Mage.ai workflow orchestration for the Football Video Analysis system. The workflow provides a fully automated, production-ready pipeline for processing football match videos and extracting comprehensive analytics.

## 🏗️ Architecture Overview

The Mage.ai pipeline orchestrates the entire football video analysis workflow with the following components:

### Pipeline Structure

```
API Trigger → Data Loader → Video Processor → Object Detector → Quality Checker → 
Tracker → Event Detector → Metrics Calculator → Visualizer → MLflow Logger → 
Data Exporter → Notification Sender
```

### Key Features

- **Fully Automated**: End-to-end processing from video upload to result delivery
- **Scalable**: Handles multiple concurrent jobs with resource management
- **Monitored**: Real-time progress tracking and quality assessment
- **Reproducible**: Version-controlled experiments with MLflow integration
- **Production-Ready**: Error handling, notifications, and comprehensive logging

## 📁 Directory Structure

```
mage/
├── pipelines/
│   └── football_video_analysis/
│       ├── pipeline.yaml              # Main pipeline configuration
│       ├── api_trigger.py             # API-triggered execution
│       ├── data_loader.py             # Video data loading
│       ├── video_processor.py         # Video preprocessing
│       ├── object_detector.py         # YOLO-based detection
│       ├── quality_checker.py         # Quality validation
│       ├── tracker.py                 # Object tracking
│       ├── event_detector.py          # Event detection
│       ├── metrics_calculator.py      # Performance metrics
│       ├── visualizer.py              # Visualization generation
│       ├── mlflow_logger.py           # Experiment tracking
│       ├── data_exporter.py           # Result export
│       └── notification_sender.py     # Completion notifications
└── README.md                          # This documentation
```

## 🚀 Getting Started

### Prerequisites

1. **Mage.ai Installation**
   ```bash
   pip install mage-ai
   ```

2. **Project Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-api.txt
   ```

3. **Mage.ai Setup**
   ```bash
   # Start Mage.ai
   mage start
   
   # Access the UI at http://localhost:6789
   ```

### Pipeline Deployment

1. **Import Pipeline**
   - Open Mage.ai UI
   - Navigate to Pipelines
   - Import the `football_video_analysis` pipeline

2. **Configure Environment**
   - Set up environment variables in Mage.ai
   - Configure model paths and output directories
   - Set up MLflow tracking URI

3. **Deploy Pipeline**
   - Enable the pipeline for production
   - Configure triggers (API, schedule, manual)
   - Set up monitoring and alerts

## 🔧 Pipeline Blocks

### 1. API Trigger (`api_trigger.py`)
- **Purpose**: Handles API-triggered pipeline execution
- **Input**: Job ID from API
- **Output**: Pipeline parameters and job information
- **Features**: Job validation, status updates, error handling

### 2. Data Loader (`data_loader.py`)
- **Purpose**: Loads video data from various sources
- **Input**: Job information or file path
- **Output**: Video data and metadata
- **Features**: Multi-source support, metadata extraction, format validation

### 3. Video Processor (`video_processor.py`)
- **Purpose**: Preprocesses videos for optimal analysis
- **Input**: Raw video data
- **Output**: Processed video and extracted frames
- **Features**: Quality enhancement, frame rate standardization, resolution adjustment

### 4. Object Detector (`object_detector.py`)
- **Purpose**: Detects players and ball using YOLO models
- **Input**: Video frames
- **Output**: Detection results with bounding boxes
- **Features**: Multi-class detection, confidence filtering, quality assessment

### 5. Quality Checker (`quality_checker.py`)
- **Purpose**: Validates detection quality and provides recommendations
- **Input**: Detection results
- **Output**: Quality metrics and recommendations
- **Features**: Coverage analysis, confidence assessment, quality scoring

### 6. Tracker (`tracker.py`)
- **Purpose**: Tracks objects across video frames
- **Input**: Detection results
- **Output**: Tracking results and trajectory analysis
- **Features**: Multi-object tracking, trajectory analysis, track quality assessment

### 7. Event Detector (`event_detector.py`)
- **Purpose**: Detects football events (passes, possession, crosses)
- **Input**: Tracking results
- **Output**: Detected events and statistics
- **Features**: Multi-event detection, validation, statistical analysis

### 8. Metrics Calculator (`metrics_calculator.py`)
- **Purpose**: Calculates performance metrics and KPIs
- **Input**: Analysis results
- **Output**: Comprehensive metrics and KPIs
- **Features**: Processing metrics, accuracy metrics, performance indicators

### 9. Visualizer (`visualizer.py`)
- **Purpose**: Generates visual outputs and overlays
- **Input**: Analysis results
- **Output**: Visualizations and annotated videos
- **Features**: Video overlays, charts, heatmaps, trajectory plots

### 10. MLflow Logger (`mlflow_logger.py`)
- **Purpose**: Logs experiments and tracks model performance
- **Input**: Analysis results and metrics
- **Output**: MLflow run information
- **Features**: Experiment tracking, artifact logging, model versioning

### 11. Data Exporter (`data_exporter.py`)
- **Purpose**: Exports results in various formats
- **Input**: Analysis results
- **Output**: Exported files and reports
- **Features**: Multi-format export, report generation, archiving

### 12. Notification Sender (`notification_sender.py`)
- **Purpose**: Sends completion notifications
- **Input**: Analysis results
- **Output**: Notification status
- **Features**: Email notifications, webhook integration, logging

## ⚙️ Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=models/best.pt
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.5

# Output Configuration
VIDEO_OUTPUT_DIR=outputs/videos
RESULTS_OUTPUT_DIR=outputs/results
LOGS_DIR=outputs/logs

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=football_analysis

# Notification Configuration
EMAIL_NOTIFICATIONS_ENABLED=false
WEBHOOK_NOTIFICATIONS_ENABLED=false
WEBHOOK_URL=
```

### Pipeline Parameters

The pipeline accepts the following parameters:

- **`job_id`**: Job identifier for API-triggered execution
- **`video_path`**: Path to video file for direct processing
- **`features`**: Dictionary of enabled features (passes, possession, crosses)
- **`confidence_threshold`**: Detection confidence threshold
- **`iou_threshold`**: IoU threshold for NMS

## 📊 Monitoring and Observability

### Progress Tracking
- Real-time job progress updates (0-100%)
- Detailed status messages for each stage
- Performance metrics and timing information

### Quality Metrics
- Detection quality scores
- Coverage ratios and consistency metrics
- Confidence distributions and recommendations

### Performance KPIs
- Processing speed (frames per second)
- Detection accuracy and reliability
- Event detection rates and success rates

### MLflow Integration
- Experiment tracking with parameters and metrics
- Model versioning and artifact logging
- Performance comparison across runs

## 🔄 Workflow Execution

### API-Triggered Execution
1. **API Call**: Submit video through REST API
2. **Job Creation**: System creates job with unique ID
3. **Pipeline Trigger**: Mage.ai triggers pipeline with job ID
4. **Processing**: Pipeline executes all blocks sequentially
5. **Result Delivery**: Results available via API endpoints

### Scheduled Execution
- Configure pipeline to run on schedule
- Process videos from designated directories
- Automated batch processing capabilities

### Manual Execution
- Trigger pipeline manually through Mage.ai UI
- Test and debug pipeline components
- Development and experimentation

## 🛠️ Development and Testing

### Local Development
```bash
# Set up development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start development server
python scripts/start_api.py
```

### Pipeline Testing
Each block includes comprehensive tests:
```bash
# Test individual blocks
python mage/pipelines/football_video_analysis/data_loader.py
python mage/pipelines/football_video_analysis/object_detector.py
# ... etc
```

### Integration Testing
```bash
# Test complete pipeline
python -m pytest tests/integration/test_pipeline.py
```

## 📈 Performance Optimization

### Resource Management
- **Memory**: Efficient frame processing and cleanup
- **CPU**: Multi-threading for parallel processing
- **GPU**: CUDA acceleration for YOLO inference
- **Storage**: Temporary file management and cleanup

### Scalability Features
- **Concurrent Jobs**: Multiple pipeline instances
- **Resource Limits**: Configurable memory and CPU limits
- **Queue Management**: Job queuing and prioritization
- **Load Balancing**: Distributed processing capabilities

## 🔒 Security and Reliability

### Error Handling
- Comprehensive exception handling in all blocks
- Graceful degradation and recovery
- Detailed error logging and reporting
- Automatic retry mechanisms

### Data Security
- Secure file handling and cleanup
- Access control and authentication
- Data encryption for sensitive information
- Audit logging and compliance

### Monitoring and Alerting
- Real-time pipeline monitoring
- Performance alerts and notifications
- Error detection and reporting
- Health checks and diagnostics

## 🚀 Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "scripts/start_api.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: football-analysis-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: football-analysis
  template:
    metadata:
      labels:
        app: football-analysis
    spec:
      containers:
      - name: pipeline
        image: football-analysis:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/best.pt"
```

### Cloud Deployment
- **AWS**: ECS/EKS deployment with S3 storage
- **GCP**: GKE deployment with Cloud Storage
- **Azure**: AKS deployment with Blob Storage

## 📚 Additional Resources

### Documentation
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Model Training Guide](TRAINING_README.md)
- [Configuration Reference](docs/CONFIGURATION.md)

### Examples
- [Sample Pipeline Execution](examples/pipeline_execution.py)
- [Custom Block Development](examples/custom_block.py)
- [Integration Examples](examples/integration/)

### Support
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [FAQ](docs/FAQ.md)
- [Contact Information](docs/CONTACT.md)

## 🤝 Contributing

We welcome contributions to improve the pipeline:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 coding standards
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This Mage.ai workflow provides a production-ready, scalable solution for football video analysis. The pipeline is designed to handle real-world scenarios with comprehensive error handling, monitoring, and optimization features. 