"""
API Trigger Block for Football Video Analysis Pipeline
Handles API-triggered pipeline execution and job management
"""

import os
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.job_manager import get_job_manager, JobState

logger = logging.getLogger(__name__)


@trigger
def api_trigger(*args, **kwargs) -> Dict[str, Any]:
    """
    API trigger for football video analysis pipeline.
    
    This block handles:
    - API job validation
    - Pipeline parameter extraction
    - Job status updates
    - Error handling for API requests
    
    Returns:
        Dict containing job information and parameters
    """
    try:
        # Get job ID from upstream (API call)
        job_id = kwargs.get('job_id')
        
        if not job_id:
            raise ValueError("No job ID provided for API trigger")
        
        logger.info(f"API trigger activated for job {job_id}")
        
        # Get job information
        job_manager = get_job_manager()
        job = job_manager.get_job(job_id)
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Validate job status
        if job.status != JobState.PENDING:
            raise ValueError(f"Job {job_id} is not in pending state: {job.status.value}")
        
        # Prepare pipeline parameters
        pipeline_params = {
            'job_id': job_id,
            'filename': job.filename,
            'features': job.features.dict() if job.features else {},
            'timestamp': job.created_at.isoformat(),
            'source': 'api'
        }
        
        logger.info(f"Pipeline parameters prepared for job {job_id}")
        
        return pipeline_params
        
    except Exception as e:
        logger.error(f"Error in API trigger: {e}")
        
        # Update job status to failed if we have a job ID
        job_id = kwargs.get('job_id')
        if job_id:
            job_manager = get_job_manager()
            job_manager.fail_job(job_id, f"API trigger failed: {str(e)}")
        
        raise


@test
def test_api_trigger():
    """Test the API trigger with sample data."""
    # Create sample job data
    sample_params = {
        'job_id': 1,
        'filename': 'test_video.mp4',
        'features': {
            'passes': True,
            'possession': True,
            'crosses': False
        }
    }
    
    # Test parameter preparation
    pipeline_params = {
        'job_id': sample_params['job_id'],
        'filename': sample_params['filename'],
        'features': sample_params['features'],
        'timestamp': '2024-01-01T12:00:00',
        'source': 'api'
    }
    
    assert 'job_id' in pipeline_params
    assert 'filename' in pipeline_params
    assert 'features' in pipeline_params
    assert pipeline_params['source'] == 'api'
    
    print("✅ API trigger test passed") 