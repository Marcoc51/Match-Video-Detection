from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
from pathlib import Path
import shutil
from pathlib import Path
from main import main

# FastAPI app
app = FastAPI(
    title="Match Video Detection API",
    description="API for detecting events in football match videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store processing status
processing_status = {}
current_job_id = 0

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok", 
        "message": "Match Video Detection API is running"
    }

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Match Video Detection API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}",
            "docs": "/docs"
        }
    }

# Predict endpoint
@app.post("/predict")
async def predict(
    video: UploadFile = File(...),
    passes: bool = True,
    possession: bool = True,
    crosses: bool = False
):
    """
    Process a football match video and detect passes, possession, and crosses.
    
    Args:
        video: The video file to process
        passes: Whether to detect and visualize passes (default: True)
        possession: Whether to track and visualize possession (default: True)
        crosses: Whether to detect and visualize crosses (default: False)
    
    Returns:
        JSON response with processing results and job ID
    """
    global current_job_id
    
    # Validate file type
    if not video.filename or \
    not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a video file (.mp4, .avi, .mov, .mkv)"
        )
    
    # Generate job ID
    job_id = current_job_id
    current_job_id += 1
    
    # Initialize processing status
    processing_status[job_id] = {
        "status": "processing",
        "filename": video.filename,
        "progress": 0,
        "message": "Starting video processing..."
    }
    
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded video to temporary location
            video_path = temp_path / video.filename
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            
            # Update status
            processing_status[job_id]["progress"] = 10
            processing_status[job_id]["message"] = "Video uploaded successfully"
            
            # Set project root (works for both Docker and local development)
            if Path("/app").exists():
                # Docker environment
                project_root = Path("/app")
            else:
                # Local development environment
                project_root = Path(__file__).parent
            
            # Run detection pipeline
            processing_status[job_id]["progress"] = 20
            processing_status[job_id]["message"] = "Running detection pipeline..."
            
            # Call your main detection function
            main(
                video_path=video_path,
                project_root=project_root,
                passes=passes,
                possession=possession,
                crosses=crosses
            )
            
            # Update status to completed
            processing_status[job_id]["status"] = "completed"
            processing_status[job_id]["progress"] = 100
            processing_status[job_id]["message"] = "Processing completed successfully"
            
            # Prepare response
            result = {
                "job_id": job_id,
                "filename": video.filename,
                "status": "completed",
                "features": {
                    "passes": passes,
                    "possession": possession,
                    "crosses": crosses
                },
                "output_files": {
                    "processed_video": f"/download/{job_id}/video",
                    "statistics": f"/download/{job_id}/stats"
                },
                "message": "Video processed successfully. \
                    Use the download endpoints to retrieve results."
            }
            
            return JSONResponse(content=result, status_code=200)
            
    except Exception as e:
        # Update status to failed
        processing_status[job_id]["status"] = "failed"
        processing_status[job_id]["message"] = f"Processing failed: {str(e)}"
        
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )

# Job Status endpoint
@app.get("/status/{job_id}")
def get_status(job_id: int):
    """
    Get the processing status of a job.
    
    Args:
        job_id: The job ID returned from the predict endpoint
    
    Returns:
        JSON response with current processing status
    """
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]

# Download video endpoint
@app.get("/download/{job_id}/video")
def download_video(job_id: int):
    """
    Download the processed video for a completed job.
    
    Args:
        job_id: The job ID returned from the predict endpoint
    
    Returns:
        The processed video file
    """
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if processing_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Path to the output video (works for both Docker and local development)
    if Path("/app").exists():
        # Docker environment
        output_path = Path("/app/outputs/new_match_output.avi")
    else:
        # Local development environment
        output_path = Path(__file__).parent / "outputs" / "new_match_output.avi"
    
    if not output_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Output video not found at: {output_path}"
        )
    
    return FileResponse(
        path=output_path,
        filename=f"processed_{processing_status[job_id]['filename']}",
        media_type="video/avi"
    )

# Download stats endpoint
@app.get("/download/{job_id}/stats")
def download_stats(job_id: int):
    """
    Download the processing statistics for a completed job.
    
    Args:
        job_id: The job ID returned from the predict endpoint
    
    Returns:
        JSON file with processing statistics
    """
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if processing_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Create basic statistics (you can enhance this based on your main.py output)
    stats = {
        "job_id": job_id,
        "filename": processing_status[job_id]["filename"],
        "processing_time": "N/A",
        "video_processed": True,
        "features_detected": {
            "passes": True,
            "possession": True,
            "crosses": True
        }
    }
    
    return JSONResponse(content=stats)

# For local development: run with `python src/api.py`
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)