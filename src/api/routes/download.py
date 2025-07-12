"""
Download routes for processed video files and statistics.
"""

from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

from ..models import VideoStatistics, ErrorResponse
from ..config import get_config
from ..job_manager import get_job_manager, JobState

router = APIRouter()

@router.get("/download/{job_id}/video")
def download_video(job_id: int):
    """
    Download the processed video for a completed job.
    
    Args:
        job_id: The job ID returned from the predict endpoint
    
    Returns:
        FileResponse: The processed video file
    
    Raises:
        HTTPException: If job is not found or not completed
    """
    config = get_config()
    job_manager = get_job_manager()
    
    # Get job information
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    # Check if job is completed
    if job.status != JobState.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed. Current status: {job.status.value}"
        )
    
    # Check if result file exists
    if not job.result_path or not job.result_path.exists():
        # Try to find the file in the default output directory
        output_filename = f"analyzed_{Path(job.filename).stem}.mp4"
        output_path = Path(config.video_output_dir) / output_filename
        
        if not output_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Processed video file not found for job {job_id}"
            )
    else:
        output_path = job.result_path
    
    # Return the file
    return FileResponse(
        path=output_path,
        filename=f"processed_{job.filename}",
        media_type="video/mp4"
    )

@router.get("/download/{job_id}/stats", response_model=VideoStatistics)
def download_stats(job_id: int):
    """
    Get the processing statistics for a completed job.
    
    Args:
        job_id: The job ID returned from the predict endpoint
    
    Returns:
        VideoStatistics: Processing statistics and analysis results
    
    Raises:
        HTTPException: If job is not found or not completed
    """
    job_manager = get_job_manager()
    
    # Get job information
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    # Check if job is completed
    if job.status != JobState.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed. Current status: {job.status.value}"
        )
    
    # Extract statistics from job result
    if not job.statistics:
        raise HTTPException(
            status_code=404,
            detail=f"No statistics available for job {job_id}"
        )
    
    # Get the result from the main function
    result = job.statistics.get("result", {})
    
    # Extract video information
    tracks = result.get("tracks", {})
    possession = result.get("possession", {})
    passes = result.get("passes", [])
    
    # Calculate basic statistics
    total_frames = len(tracks.get("players", [])) if tracks.get("players") else 0
    video_duration = total_frames / 30.0 if total_frames > 0 else 0  # Assuming 30 FPS
    fps = 30.0  # Default FPS
    
    # Count detections
    players_detected = 0
    ball_detections = 0
    
    if tracks.get("players"):
        for frame_players in tracks["players"]:
            players_detected = max(players_detected, len(frame_players))
    
    if tracks.get("ball"):
        ball_detections = sum(1 for frame_ball in tracks["ball"] if frame_ball)
    
    # Extract possession statistics
    possession_home = possession.get("final_home", 0) if possession else 0
    possession_away = possession.get("final_away", 0) if possession else 0
    
    # Count passes
    passes_detected = len(passes) if passes else 0
    
    # Create statistics object
    stats = VideoStatistics(
        total_frames=total_frames,
        video_duration=video_duration,
        fps=fps,
        resolution="1920x1080",  # Default resolution
        players_detected=players_detected,
        ball_detections=ball_detections,
        passes_detected=passes_detected,
        possession_home=possession_home,
        possession_away=possession_away,
        crosses_detected=0,  # TODO: Extract from result
        model_confidence=0.85,  # Default confidence
        processing_fps=30.0  # Default processing FPS
    )
    
    return stats

@router.get("/download/{job_id}/json")
def download_json_stats(job_id: int):
    """
    Download the complete processing results as JSON.
    
    Args:
        job_id: The job ID returned from the predict endpoint
    
    Returns:
        JSONResponse: Complete processing results
    
    Raises:
        HTTPException: If job is not found or not completed
    """
    job_manager = get_job_manager()
    
    # Get job information
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    # Check if job is completed
    if job.status != JobState.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed. Current status: {job.status.value}"
        )
    
    # Prepare complete results
    results = {
        "job_id": job.job_id,
        "filename": job.filename,
        "processing_time": job.processing_time,
        "features": job.features.dict() if job.features else None,
        "statistics": job.statistics,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.updated_at.isoformat()
    }
    
    return JSONResponse(content=results)

@router.get("/download/{job_id}/summary")
def download_summary(job_id: int):
    """
    Download a summary of the processing results.
    
    Args:
        job_id: The job ID returned from the predict endpoint
    
    Returns:
        JSONResponse: Summary of processing results
    
    Raises:
        HTTPException: If job is not found or not completed
    """
    job_manager = get_job_manager()
    
    # Get job information
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )
    
    # Check if job is completed
    if job.status != JobState.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed. Current status: {job.status.value}"
        )
    
    # Extract summary from statistics
    result = job.statistics.get("result", {}) if job.statistics else {}
    possession = result.get("possession", {})
    passes = result.get("passes", [])
    
    # Create summary
    summary = {
        "job_id": job.job_id,
        "filename": job.filename,
        "processing_time_seconds": job.processing_time,
        "features_enabled": job.features.dict() if job.features else None,
        "analysis_summary": {
            "total_passes": len(passes) if passes else 0,
            "possession_home_percentage": possession.get("final_home", 0) if possession else 0,
            "possession_away_percentage": possession.get("final_away", 0) if possession else 0,
            "video_processed": True,
            "output_video_available": job.result_path.exists() if job.result_path else False
        },
        "download_links": {
            "processed_video": f"/download/{job_id}/video",
            "statistics": f"/download/{job_id}/stats",
            "complete_results": f"/download/{job_id}/json"
        }
    }
    
    return JSONResponse(content=summary) 