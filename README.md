# FC Masar MLOps Project: Cross Detection from Football Match Videos

## 🎯 Project Overview
This project aims to detect and count football crosses from match videos using machine learning and computer vision. It is designed as an end-to-end MLOps pipeline aligned with the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) guidelines, while also serving as a working production system for FC Masar.

We use object detection (YOLOv8), tracking, and rule-based or ML-based event detection to process match videos and extract key tactical metrics.

---

## 📦 Project Structure

```
fc-masar-mlops/
├── docker-compose.yml          # Docker setup for development
├── Dockerfile                  # Image with Python + CUDA + tools
├── README.md                   # Project description and usage
├── Makefile                    # CLI commands (build, run, test, etc.)

├── src/                        # Core pipeline logic
│   ├── __init__.py
│   ├── extract_frames.py       # Convert video to frames
│   ├── detect_objects.py       # YOLOv8 inference
│   ├── track_objects.py        # Player & ball tracking
│   ├── detect_crosses.py       # Cross detection logic
│   └── utils.py

├── training/                   # Model training and evaluation
│   ├── train.py
│   ├── evaluate.py
│   └── params.yaml             # Config file for training

├── service/                    # API deployment (FastAPI)
│   ├── main.py
│   ├── model.py
│   └── utils.py

├── monitoring/                 # Performance and drift monitoring
│   ├── log_metrics.py
│   └── evidently_report.py

├── notebooks/                  # Jupyter notebooks for testing/EDA
│   ├── eda.ipynb
│   └── test_yolo_detection.ipynb

├── data/                       # Project data (gitignore large files)
│   ├── raw/                    # Raw video input
│   ├── frames/                 # Extracted frames
│   ├── labeled/                # Labeled data (if needed)
│   └── predictions/            # Prediction outputs

├── mlruns/                     # MLflow experiment tracking logs

├── outputs/                    # Inference outputs and metrics
│   ├── results.json
│   └── metrics.csv

├── tests/                      # Unit and integration tests
│   ├── test_pipeline.py
│   └── test_utils.py

└── .github/
    └── workflows/
        └── main.yml            # GitHub Actions for CI/CD
```

---

## ⚙️ Key Components
- **Object Detection**: YOLOv8 for identifying players and ball in frames
- **Object Tracking**: DeepSORT or ByteTrack for tracking players
- **Cross Detection**: Rule-based or ML classifier based on trajectory
- **Experiment Tracking**: MLflow for model metrics and parameters
- **Orchestration**: Airflow or Prefect for batch automation
- **Deployment**: FastAPI with Docker
- **Monitoring**: Evidently for model & data drift tracking

---

## 🚀 How to Run the Project
```bash
# Build and start Jupyter with Docker Compose
docker compose build
docker compose up

# Access Jupyter in browser:
# http://localhost:8888
```

---

## 🧠 Next Steps
- [ ] Add sample data and test YOLOv8 detection
- [ ] Implement frame extraction logic
- [ ] Build tracking and detection logic
- [ ] Set up MLflow and Evidently

---

## 👥 Contributors
- **Marc Sanad** – MLOps Zoomcamp Student & FC Masar Data Engineer

---

## 📜 License
MIT License (to be confirmed)
