# Match Video Detection API Documentation

## Overview

The Match Video Detection API provides a RESTful interface for processing football match videos and detecting various events including passes, possession, crosses, and more. The API is built with FastAPI and provides comprehensive documentation through Swagger UI.

## Quick Start

### Starting the API

```bash
# Using the start script
python scripts/start_api.py

# Or directly with uvicorn
python -m src.api.main

# Or with uvicorn command
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the API

- **API Base URL**: `http://localhost:8000`
- **Interactive Documentation**: `http://localhost:8000/docs`
- **Alternative Documentation**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

## API Endpoints

### Health and System Information

#### GET `/`
Get API information and available endpoints.

**Response:**
```json
{
  "name": "Match Video Detection API",
  "version": "1.0.0",
  "description": "API for detecting events in football match videos",
  "endpoints": {
    "health": "/health",
    "system_status": "/system/status",
    "predict": "/predict",
    "jobs": "/jobs",
    "job_status": "/jobs/{job_id}",
    "cancel_job": "/jobs/{job_id}/cancel",
    "download_video": "/download/{job_id}/video",
    "download_stats": "/download/{job_id}/stats",
    "models": "/models",
    "docs": "/docs",
    "redoc": "/redoc"
  },
  "features": [
    "Player and ball detection",
    "Pass detection and visualization",
    "Possession tracking",
    "Cross detection",
    "Team assignment",
    "Speed and distance estimation",
    "Video processing and analysis"
  ],
  "documentation": "/docs"
}
```

#### GET `/health`
Basic health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "message": "Match Video Detection API is running",
  "timestamp": "2024-01-01T12:00:00",
  "version": "1.0.0",
  "uptime": 3600
}
```

#### GET `/system/status`
Get system resource usage and job statistics.

**Response:**
```json
{
  "cpu_usage": 45.2,
  "memory_usage": 67.8,
  "gpu_usage": null,
  "disk_usage": 23.4,
  "active_jobs": 1,
  "queue_size": 2,
  "uptime": 3600
}
```

#### GET `/health/detailed`
Detailed health check with configuration validation.

### Video Processing

#### POST `/predict`
Process a football match video and detect events.

**Parameters:**
- `video` (file): Video file to process (supports .mp4, .avi, .mov, .mkv, .webm)
- `passes` (boolean, optional): Enable pass detection (default: true)
- `possession` (boolean, optional): Enable possession tracking (default: true)
- `crosses` (boolean, optional): Enable cross detection (default: false)
- `speed_distance` (boolean, optional): Enable speed and distance estimation (default: true)
- `team_assignment` (boolean, optional): Enable team assignment (default: true)

**Response:**
```json
{
  "job_id": 1,
  "filename": "match_video.mp4",
  "status": "pending",
  "features": {
    "passes": true,
    "possession": true,
    "crosses": false,
    "speed_distance": true,
    "team_assignment": true
  },
  "output_files": {
    "processed_video": "/download/1/video",
    "statistics": "/download/1/stats"
  },
  "message": "Job created successfully. Processing will start shortly."
}
```

### Job Management

#### GET `/jobs`
List processing jobs with optional filtering.

**Query Parameters:**
- `status` (string, optional): Filter by job status (pending, processing, completed, failed, cancelled)
- `limit` (integer, optional): Number of jobs to return (default: 50, max: 100)
- `offset` (integer, optional): Number of jobs to skip (default: 0)

**Response:**
```json
{
  "jobs": [
    {
      "job_id": 1,
      "filename": "match_video.mp4",
      "status": "completed",
      "progress": 100,
      "message": "Processing completed successfully",
      "created_at": "2024-01-01T12:00:00",
      "updated_at": "2024-01-01T12:05:00",
      "features": {
        "passes": true,
        "possession": true,
        "crosses": false,
        "speed_distance": true,
        "team_assignment": true
      },
      "processing_time": 300.5
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 50,
  "has_next": false,
  "has_prev": false
}
```

#### GET `/jobs/{job_id}`
Get the status of a specific job.

**Response:**
```json
{
  "job_id": 1,
  "filename": "match_video.mp4",
  "status": "completed",
  "progress": 100,
  "message": "Processing completed successfully",
  "created_at": "2024-01-01T12:00:00",
  "updated_at": "2024-01-01T12:05:00",
  "features": {
    "passes": true,
    "possession": true,
    "crosses": false,
    "speed_distance": true,
    "team_assignment": true
  },
  "processing_time": 300.5,
  "error_message": null,
  "statistics": {
    "processing_time": 300.5,
    "result": {
      "tracks": {...},
      "possession": {...},
      "passes": [...]
    }
  }
}
```

#### POST `/jobs/{job_id}/cancel`
Cancel a processing job.

**Response:**
```json
{
  "job_id": 1,
  "status": "cancelled",
  "message": "Job cancelled successfully"
}
```

### Download Results

#### GET `/download/{job_id}/video`
Download the processed video for a completed job.

**Response:** Video file (MP4 format)

#### GET `/download/{job_id}/stats`
Get processing statistics for a completed job.

**Response:**
```json
{
  "total_frames": 5400,
  "video_duration": 180.0,
  "fps": 30.0,
  "resolution": "1920x1080",
  "players_detected": 22,
  "ball_detections": 5400,
  "passes_detected": 45,
  "possession_home": 52.3,
  "possession_away": 47.7,
  "crosses_detected": 8,
  "model_confidence": 0.85,
  "processing_fps": 30.0
}
```

#### GET `/download/{job_id}/json`
Download complete processing results as JSON.

#### GET `/download/{job_id}/summary`
Get a summary of processing results.

**Response:**
```json
{
  "job_id": 1,
  "filename": "match_video.mp4",
  "processing_time_seconds": 300.5,
  "features_enabled": {
    "passes": true,
    "possession": true,
    "crosses": false,
    "speed_distance": true,
    "team_assignment": true
  },
  "analysis_summary": {
    "total_passes": 45,
    "possession_home_percentage": 52.3,
    "possession_away_percentage": 47.7,
    "video_processed": true,
    "output_video_available": true
  },
  "download_links": {
    "processed_video": "/download/1/video",
    "statistics": "/download/1/stats",
    "complete_results": "/download/1/json"
  }
}
```

### Model Information

#### GET `/models`
List available models and their status.

**Response:**
```json
{
  "available_models": [
    {
      "name": "best.pt",
      "path": "models/yolo/best.pt",
      "size_mb": 45.2,
      "type": ".pt",
      "status": "available"
    }
  ],
  "default_model": "models/yolo/best.pt",
  "models_directory": "/path/to/models",
  "default_model_status": {
    "exists": true,
    "path": "/path/to/models/yolo/best.pt",
    "size_mb": 45.2
  }
}
```

#### GET `/models/{model_name}`
Get detailed information about a specific model.

#### GET `/models/status`
Get overall status of models.

## Error Handling

The API uses standard HTTP status codes and returns error responses in the following format:

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters or file format
- `404 Not Found`: Job or model not found
- `500 Internal Server Error`: Server-side processing error
- `503 Service Unavailable`: Maximum concurrent jobs reached

## Configuration

The API can be configured using environment variables or a `.env` file. Key configuration options:

- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `API_MODEL_PATH`: Path to YOLO model (default: models/yolo/best.pt)
- `API_MAX_CONCURRENT_JOBS`: Maximum concurrent processing jobs (default: 3)
- `API_MAX_FILE_SIZE`: Maximum file size in bytes (default: 500MB)

## Usage Examples

### Python Client Example

```python
import requests

# Upload and process video
with open('match_video.mp4', 'rb') as f:
    files = {'video': f}
    data = {
        'passes': True,
        'possession': True,
        'crosses': False
    }
    response = requests.post('http://localhost:8000/predict', files=files, data=data)
    job = response.json()
    job_id = job['job_id']

# Check job status
status_response = requests.get(f'http://localhost:8000/jobs/{job_id}')
status = status_response.json()

# Download results when completed
if status['status'] == 'completed':
    # Download video
    video_response = requests.get(f'http://localhost:8000/download/{job_id}/video')
    with open('processed_video.mp4', 'wb') as f:
        f.write(video_response.content)
    
    # Get statistics
    stats_response = requests.get(f'http://localhost:8000/download/{job_id}/stats')
    stats = stats_response.json()
    print(f"Detected {stats['passes_detected']} passes")
```

### cURL Examples

```bash
# Upload video for processing
curl -X POST "http://localhost:8000/predict" \
  -F "video=@match_video.mp4" \
  -F "passes=true" \
  -F "possession=true" \
  -F "crosses=false"

# Check job status
curl "http://localhost:8000/jobs/1"

# Download processed video
curl "http://localhost:8000/download/1/video" -o processed_video.mp4

# Get statistics
curl "http://localhost:8000/download/1/stats"
```

## Development

### Running in Development Mode

```bash
# Install dependencies
pip install -r requirements-api.txt

# Start with auto-reload
python scripts/start_api.py
```

### Testing the API

1. Start the API server
2. Open `http://localhost:8000/docs` in your browser
3. Use the interactive documentation to test endpoints
4. Upload a test video file and monitor processing

### Logs

API logs are written to `outputs/logs/api.log` by default. Log level can be configured via `API_LOG_LEVEL` environment variable.

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the YOLO model file exists at the configured path
2. **File upload errors**: Check file size limits and supported formats
3. **Processing failures**: Check logs for detailed error messages
4. **Memory issues**: Reduce `API_MAX_CONCURRENT_JOBS` if running out of memory

### Performance Optimization

- Use GPU acceleration if available
- Adjust `API_MAX_CONCURRENT_JOBS` based on system resources
- Monitor system resources using `/system/status` endpoint
- Consider using smaller video files for testing

## Support

For issues and questions:
1. Check the API logs at `outputs/logs/api.log`
2. Use the `/health/detailed` endpoint for system diagnostics
3. Review the interactive documentation at `/docs`
4. Check the project README for additional information 