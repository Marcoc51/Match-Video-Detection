"""
Job management system for handling video processing jobs.
"""

import asyncio
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from .models import ProcessingStatus, ProcessingJob, FeatureConfig
from .config import get_config


class JobState(Enum):
    """Job state enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Job information container."""
    job_id: int
    filename: str
    status: JobState
    progress: int = 0
    message: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    features: Optional[FeatureConfig] = None
    result_path: Optional[Path] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    statistics: Optional[Dict] = None


class JobManager:
    """Manages video processing jobs."""
    
    def __init__(self):
        self.jobs: Dict[int, JobInfo] = {}
        self.next_job_id = 0
        self.active_jobs = 0
        self.max_concurrent_jobs = get_config().max_concurrent_jobs
        self.job_timeout = get_config().job_timeout
        self.cleanup_interval = get_config().cleanup_interval
        
        # Threading
        self.lock = threading.Lock()
        self.cleanup_thread = None
        self.running = False
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the job manager."""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        self.logger.info("Job manager started")
    
    def stop(self):
        """Stop the job manager."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        self.logger.info("Job manager stopped")
    
    def create_job(self, filename: str, features: Optional[FeatureConfig] = None) -> int:
        """Create a new job and return its ID."""
        with self.lock:
            job_id = self.next_job_id
            self.next_job_id += 1
            
            job = JobInfo(
                job_id=job_id,
                filename=filename,
                status=JobState.PENDING,
                message="Job created, waiting to start",
                features=features
            )
            
            self.jobs[job_id] = job
            self.logger.info(f"Created job {job_id} for file {filename}")
            
            return job_id
    
    def get_job(self, job_id: int) -> Optional[JobInfo]:
        """Get job information by ID."""
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job(self, job_id: int, **kwargs) -> bool:
        """Update job information."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            job.updated_at = datetime.now()
            
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            self.logger.debug(f"Updated job {job_id}: {kwargs}")
            return True
    
    def start_job(self, job_id: int) -> bool:
        """Start processing a job."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            if job.status != JobState.PENDING:
                return False
            
            if self.active_jobs >= self.max_concurrent_jobs:
                return False
            
            job.status = JobState.PROCESSING
            job.message = "Processing started"
            job.updated_at = datetime.now()
            self.active_jobs += 1
            
            self.logger.info(f"Started job {job_id}")
            return True
    
    def complete_job(self, job_id: int, result_path: Optional[Path] = None, 
                    statistics: Optional[Dict] = None) -> bool:
        """Mark a job as completed."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            job.status = JobState.COMPLETED
            job.progress = 100
            job.message = "Processing completed successfully"
            job.result_path = result_path
            job.statistics = statistics
            job.processing_time = (datetime.now() - job.created_at).total_seconds()
            job.updated_at = datetime.now()
            
            if job.status == JobState.PROCESSING:
                self.active_jobs -= 1
            
            self.logger.info(f"Completed job {job_id}")
            return True
    
    def fail_job(self, job_id: int, error_message: str) -> bool:
        """Mark a job as failed."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            job.status = JobState.FAILED
            job.message = f"Processing failed: {error_message}"
            job.error_message = error_message
            job.updated_at = datetime.now()
            
            if job.status == JobState.PROCESSING:
                self.active_jobs -= 1
            
            self.logger.error(f"Failed job {job_id}: {error_message}")
            return True
    
    def cancel_job(self, job_id: int) -> bool:
        """Cancel a job."""
        with self.lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            if job.status in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
                return False
            
            job.status = JobState.CANCELLED
            job.message = "Job cancelled by user"
            job.updated_at = datetime.now()
            
            if job.status == JobState.PROCESSING:
                self.active_jobs -= 1
            
            self.logger.info(f"Cancelled job {job_id}")
            return True
    
    def get_jobs(self, status: Optional[JobState] = None, 
                limit: int = 50, offset: int = 0) -> List[JobInfo]:
        """Get list of jobs with optional filtering."""
        with self.lock:
            jobs = list(self.jobs.values())
            
            if status:
                jobs = [job for job in jobs if job.status == status]
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            
            return jobs[offset:offset + limit]
    
    def get_job_count(self, status: Optional[JobState] = None) -> int:
        """Get count of jobs with optional status filter."""
        with self.lock:
            if status:
                return sum(1 for job in self.jobs.values() if job.status == status)
            return len(self.jobs)
    
    def can_start_job(self) -> bool:
        """Check if a new job can be started."""
        with self.lock:
            return self.active_jobs < self.max_concurrent_jobs
    
    def _cleanup_worker(self):
        """Background worker for cleaning up old jobs."""
        while self.running:
            try:
                self._cleanup_old_jobs()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                self.logger.error(f"Error in cleanup worker: {e}")
    
    def _cleanup_old_jobs(self):
        """Clean up old completed/failed jobs."""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep jobs for 24 hours
        
        with self.lock:
            jobs_to_remove = []
            
            for job_id, job in self.jobs.items():
                if (job.status in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED] and
                    job.updated_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                self.logger.info(f"Cleaned up old job {job_id}")
    
    def get_system_status(self) -> Dict:
        """Get system status information."""
        with self.lock:
            return {
                "total_jobs": len(self.jobs),
                "active_jobs": self.active_jobs,
                "pending_jobs": self.get_job_count(JobState.PENDING),
                "completed_jobs": self.get_job_count(JobState.COMPLETED),
                "failed_jobs": self.get_job_count(JobState.FAILED),
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "can_start_job": self.can_start_job()
            }


# Global job manager instance
job_manager = JobManager()


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    return job_manager


def start_job_manager():
    """Start the global job manager."""
    job_manager.start()


def stop_job_manager():
    """Stop the global job manager."""
    job_manager.stop() 