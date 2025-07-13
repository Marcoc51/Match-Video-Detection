"""
Prediction routes for video processing.
"""

import asyncio
import threading
import tempfile
from pathlib import Path
import shutil
import time
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from ..models import ProcessingResult, FeatureConfig, ErrorResponse, ProcessingStatus
from ..config import get_config
from ..job_manager import get_job_manager, JobState
from ...core.main import main

router = APIRouter()

@router.post("/predict", response_model=ProcessingResult)
async def predict(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(..., description="Video file to process"),
    passes: bool = Query(default=True, description="Enable pass detection"),
    possession: bool = Query(default=True, description="Enable possession tracking"),
    crosses: bool = Query(default=False, description="Enable cross detection"),
    speed_distance: bool = Query(default=True, description="Enable speed and distance estimation"),
    team_assignment: bool = Query(default=True, description="Enable team assignment")
):
    """
    Process a football match video and detect events.
    
    This endpoint accepts a video file and processes it to detect various football events
    including passes, possession, crosses, and more. The processing is done asynchronously
    and returns a job ID that can be used to track progress and download results.
    
    Args:
        video: The video file to process (supports .mp4, .avi, .mov, .mkv, .webm)
        passes: Whether to detect and visualize passes
        possession: Whether to track and visualize possession
        crosses: Whether to detect and visualize crosses
        speed_distance: Whether to estimate speed and distance
        team_assignment: Whether to assign players to teams
    
    Returns:
        ProcessingResult: Job information with processing status and download URLs
    
    Raises:
        HTTPException: If file validation fails or processing cannot be started
    """
    config = get_config()
    job_manager = get_job_manager()
    
    # Validate file
    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    # Check file extension
    file_extension = Path(video.filename).suffix.lower()
    if file_extension not in config.allowed_video_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed formats: {', '.join(config.allowed_video_formats)}"
        )
    
    # Check if we can start a new job
    if not job_manager.can_start_job():
        raise HTTPException(
            status_code=503,
            detail="Maximum number of concurrent jobs reached. Please try again later."
        )
    
    # Create feature configuration
    features = FeatureConfig(
        passes=passes,
        possession=possession,
        crosses=crosses,
        speed_distance=speed_distance,
        team_assignment=team_assignment
    )
    
    # Create job
    job_id = job_manager.create_job(video.filename, features)
    
    # Read video content before passing to background task
    video_content = video.file.read()
    
    # Start processing in background
    background_tasks.add_task(
        process_video_job,
        job_id,
        video_content,
        video.filename,
        features
    )
    
    # Prepare response
    output_files = {
        "processed_video": f"/download/{job_id}/video",
        "statistics": f"/download/{job_id}/stats"
    }
    
    return ProcessingResult(
        job_id=job_id,
        filename=video.filename,
        status=ProcessingStatus.PENDING,
        features=features,
        output_files=output_files,
        message="Job created successfully. Processing will start shortly."
    )


async def process_video_job(job_id: int, video_content: bytes, filename: str, features: FeatureConfig):
    """
    Process a video job in the background.
    
    Args:
        job_id: The job ID
        video_content: The video file content as bytes
        filename: The original filename
        features: Feature configuration
    """
    config = get_config()
    job_manager = get_job_manager()
    
    try:
        # Start the job
        if not job_manager.start_job(job_id):
            job_manager.fail_job(job_id, "Could not start job processing")
            return
        
        # Update progress
        job_manager.update_job(job_id, progress=10, message="Saving uploaded video...")
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded video
            video_path = temp_path / filename
            with open(video_path, "wb") as buffer:
                buffer.write(video_content)
            
            # Update progress
            job_manager.update_job(job_id, progress=20, message="Video saved, starting analysis...")
            
            # Get project root
            project_root = config.project_root
            
            # Update progress
            job_manager.update_job(job_id, progress=30, message="Running detection pipeline...")
            
            # Run the main detection pipeline
            start_time = time.time()
            
            result = main(
                video_path=video_path,
                project_root=project_root,
                passes=features.passes,
                possession=features.possession,
                crosses=features.crosses
            )
            
            processing_time = time.time() - start_time
            
            # Update progress
            job_manager.update_job(job_id, progress=90, message="Analysis complete, preparing results...")
            
            # Determine output file path
            output_filename = f"analyzed_{Path(filename).stem}.mp4"
            output_path = Path(config.video_output_dir) / output_filename
            
            # Complete the job
            job_manager.complete_job(
                job_id=job_id,
                result_path=output_path,
                statistics={
                    "processing_time": processing_time,
                    "result": result
                }
            )
            
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        
        # Mark job as failed
        job_manager.fail_job(job_id, str(e))


@router.get("/jobs", response_model=dict)
def list_jobs(
    status: str = Query(None, description="Filter by job status"),
    limit: int = Query(50, ge=1, le=100, description="Number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip")
):
    """
    List processing jobs with optional filtering.
    
    Args:
        status: Filter jobs by status (pending, processing, completed, failed, cancelled)
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip for pagination
    
    Returns:
        dict: List of jobs with pagination information
    """
    job_manager = get_job_manager()
    
    # Convert status string to JobState enum
    job_state = None
    if status:
        try:
            job_state = JobState(status.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid values: pending, processing, completed, failed, cancelled"
            )
    
    # Get jobs
    jobs = job_manager.get_jobs(status=job_state, limit=limit, offset=offset)
    total = job_manager.get_job_count(status=job_state)
    
    # Convert to response format
    job_list = []
    for job in jobs:
        job_list.append({
            "job_id": job.job_id,
            "filename": job.filename,
            "status": job.status.value,
            "progress": job.progress,
            "message": job.message,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "features": job.features.dict() if job.features else None,
            "processing_time": job.processing_time
        })
    
    return {
        "jobs": job_list,
        "total": total,
        "page": offset // limit + 1,
        "per_page": limit,
        "has_next": offset + limit < total,
        "has_prev": offset > 0
    }


@router.get("/jobs/{job_id}", response_model=dict)
def get_job_status(job_id: int):
    """
    Get the status of a specific job.
    
    Args:
        job_id: The job ID
    
    Returns:
        dict: Job status information
    
    Raises:
        HTTPException: If job is not found
    """
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "created_at": job.created_at.isoformat(),
        "updated_at": job.updated_at.isoformat(),
        "features": job.features.dict() if job.features else None,
        "processing_time": job.processing_time,
        "error_message": job.error_message,
        "statistics": job.statistics
    }


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    """
    Cancel a processing job.
    
    Args:
        job_id: The job ID to cancel
    
    Returns:
        dict: Cancellation result
    
    Raises:
        HTTPException: If job is not found or cannot be cancelled
    """
    job_manager = get_job_manager()
    
    if not job_manager.cancel_job(job_id):
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} cannot be cancelled or does not exist"
        )
    
    return {
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancelled successfully"
    } 