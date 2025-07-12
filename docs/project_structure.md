# Match-Video-Detection - Proposed Project Structure

## 🎯 **New Organized Structure**

```
Match-Video-Detection/
├── README.md                          # Main project documentation
├── requirements.txt                   # Dependencies
├── .gitignore                         # Git ignore rules
├── .env                               # Environment variables template
│
├── src/                               # Main application source code
│   ├── __init__.py
│   ├── api/                           # API related code
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application
│   │   ├── routes/                    # API routes
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   ├── predict.py
│   │   │   └── download.py
│   │   └── middleware/                # API middleware
│   │       ├── __init__.py
│   │       └── cors.py
│   │
│   ├── core/                          # Core application logic
│   │   ├── __init__.py
│   │   ├── main.py                    # Main detection pipeline
│   │   ├── config.py                  # Configuration management
│   │   └── exceptions.py              # Custom exceptions
│   │
│   ├── detection/                     # Object detection
│   │   ├── __init__.py
│   │   └── yolo.py                    # YOLO model interface
│   │
│   ├── tracking/                      # Object tracking
│   │   ├── __init__.py
│   │   └── tracker.py
│   │
│   ├── events/                        # Event detection
│   │   ├── __init__.py
│   │   ├── entities.py
│   │   ├── pass_event.py
│   │   ├── possession_tracker.py
│   │   ├── cross_detector.py
│   │   └── pass_detector.py
│   │
│   ├── assignment/                    # Object assignment logic
│   │   ├── __init__.py
│   │   ├── team_assigner.py
│   │   ├── player_ball_assigner.py
│   │   ├── camera_movement_estimator.py
│   │   ├── speed_distance_estimator.py
│   │   └── view_transformer.py
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── bbox_utils.py
│       ├── colors.py
│       ├── video_utils.py
│       └── io.py
│
├── training/                          # Training pipeline
│   ├── __init__.py
│   ├── config/                        # Training configurations
│   │   ├── training_config.yaml
│   │   └── model_config.yaml
│   ├── data/                          # Data preparation
│   │   ├── __init__.py
│   │   ├── preparation.py
│   │   ├── augmentation.py
│   │   └── validation.py
│   ├── models/                        # Model management
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── deployment.py
│   ├── scripts/                       # Training scripts
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── test_model.py
│   │   └── deploy_model.py
│   └── utils/                         # Training utilities
│       ├── __init__.py
│       ├── mlflow_utils.py
│       └── visualization.py
│
├── mlflow/                            # MLflow configuration
│   ├── tracking_uri.txt
│   └── experiments/
│
├── data/                              # Data directories
│   ├── raw/                           # Raw input videos
│   ├── processed/                     # Processed data
│   ├── training/                      # Training data
│   │   ├── images/
│   │   └── labels/
│   └── validation/                    # Validation data
│       ├── images/
│       └── labels/
│
├── models/                            # Model storage
│   ├── yolo/                          # YOLO models
│   │   ├── best.pt
│   │   ├── last.pt
│   │   └── ball.pt
│   ├── trained/                       # Trained models
│   └── deployed/                      # Deployed models
│
├── outputs/                           # Output files
│   ├── videos/                        # Processed videos
│   ├── results/                       # Analysis results
│   └── logs/                          # Log files
│
├── tests/                             # Test files
│   ├── __init__.py
│   ├── test_detection.py
│   ├── test_tracking.py
│   └── test_events.py
│
├── docs/                              # Documentation
│   ├── api.md
│   ├── training.md
│   ├── deployment.md
│   └── troubleshooting.md
│
├── scripts/                           # Utility scripts
│   ├── setup.py                       # Project setup
│   ├── start_api.py                   # Start API server
│   ├── start_mlflow.py                # Start MLflow UI
│   └── check_dataset.py               # Dataset validation
│
└──
```

## 🚀 **Benefits of New Structure**

### **1. Clear Separation of Concerns**
- **`src/`**: Main application code
- **`training/`**: All training-related code
- **`data/`**: Data management
- **`models/`**: Model storage
- **`outputs/`**: Results and logs

### **2. Better Maintainability**
- Related files grouped together
- Easy to find specific functionality
- Clear import paths

### **3. Professional Organization**
- Follows Python project standards
- Easy for new developers to understand
- Scalable for future features

### **4. Improved Workflow**
- Clear development vs production separation
- Easy testing and deployment
- Better version control

## 🔧 **Migration Plan**

Would you like me to:

1. **Create the new directory structure**
2. **Move existing files** to their new locations
3. **Update import paths** in all files
4. **Create new organized files** (like separate API routes)
5. **Update documentation** to reflect new structure

This will make your project much more professional and easier to maintain!

**Should I proceed with the reorganization?** 