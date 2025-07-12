"""
Integration tests for API endpoints.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from src.api.main import app
from src.api.models import JobRequest, JobResponse, JobStatus


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_job_manager():
    """Mock job manager."""
    with patch('src.api.routes.predict.get_job_manager') as mock:
        job_manager = Mock()
        mock.return_value = job_manager
        yield job_manager


@pytest.fixture
def mock_config():
    """Mock configuration."""
    with patch('src.api.routes.predict.get_config') as mock:
        config = Mock()
        config.temp_dir = "/tmp"
        config.output_dir = "outputs"
        mock.return_value = config
        yield config


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_health_detailed(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert "services" in data


class TestPredictEndpoints:
    """Test prediction endpoints."""
    
    def test_predict_video_success(self, client, mock_job_manager, mock_config):
        """Test successful video prediction."""
        # Mock job creation
        mock_job = Mock()
        mock_job.job_id = 1
        mock_job.status = JobStatus.PENDING
        mock_job.message = "Job created successfully"
        mock_job_manager.create_job.return_value = mock_job
        
        # Create test video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            video_path = f.name
        
        try:
            with open(video_path, "rb") as video_file:
                response = client.post(
                    "/predict/video",
                    files={"file": ("test_video.mp4", video_file, "video/mp4")},
                    data={
                        "features": json.dumps({
                            "passes": True,
                            "possession": True,
                            "crosses": False
                        })
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == 1
            assert data["status"] == "pending"
            assert "message" in data
            assert "created_at" in data
            
            # Verify job was created
            mock_job_manager.create_job.assert_called_once()
            
        finally:
            Path(video_path).unlink(missing_ok=True)
    
    def test_predict_video_invalid_file(self, client):
        """Test prediction with invalid file."""
        response = client.post(
            "/predict/video",
            files={"file": ("test.txt", b"not a video", "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_predict_video_missing_file(self, client):
        """Test prediction without file."""
        response = client.post("/predict/video")
        
        assert response.status_code == 422  # Validation error
    
    def test_get_job_status_success(self, client, mock_job_manager):
        """Test getting job status."""
        # Mock job
        mock_job = Mock()
        mock_job.job_id = 1
        mock_job.status = JobStatus.COMPLETED
        mock_job.progress = 100
        mock_job.message = "Job completed successfully"
        mock_job_manager.get_job.return_value = mock_job
        
        response = client.get("/predict/job/1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == 1
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert "message" in data
    
    def test_get_job_status_not_found(self, client, mock_job_manager):
        """Test getting non-existent job status."""
        mock_job_manager.get_job.return_value = None
        
        response = client.get("/predict/job/999")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
    
    def test_get_all_jobs(self, client, mock_job_manager):
        """Test getting all jobs."""
        # Mock jobs
        mock_job1 = Mock()
        mock_job1.job_id = 1
        mock_job1.status = JobStatus.COMPLETED
        mock_job1.filename = "video1.mp4"
        
        mock_job2 = Mock()
        mock_job2.job_id = 2
        mock_job2.status = JobStatus.PENDING
        mock_job2.filename = "video2.mp4"
        
        mock_job_manager.get_all_jobs.return_value = [mock_job1, mock_job2]
        
        response = client.get("/predict/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["jobs"][0]["job_id"] == 1
        assert data["jobs"][1]["job_id"] == 2


class TestDownloadEndpoints:
    """Test download endpoints."""
    
    def test_download_result_success(self, client, mock_config):
        """Test successful result download."""
        # Create mock result file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result_data = {"passes": 10, "possession": {"home": 60, "away": 40}}
            f.write(json.dumps(result_data).encode())
            result_path = f.name
        
        try:
            with patch('src.api.routes.download.get_config', return_value=mock_config):
                mock_config.output_dir = str(Path(result_path).parent)
                
                response = client.get("/download/result/1")
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "application/json"
                assert "attachment" in response.headers["content-disposition"]
                
        finally:
            Path(result_path).unlink(missing_ok=True)
    
    def test_download_result_not_found(self, client, mock_config):
        """Test downloading non-existent result."""
        with patch('src.api.routes.download.get_config', return_value=mock_config):
            mock_config.output_dir = "/nonexistent"
            
            response = client.get("/download/result/999")
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
    
    def test_download_video_success(self, client, mock_config):
        """Test successful video download."""
        # Create mock video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            video_path = f.name
        
        try:
            with patch('src.api.routes.download.get_config', return_value=mock_config):
                mock_config.output_dir = str(Path(video_path).parent)
                
                response = client.get("/download/video/1")
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "video/mp4"
                assert "attachment" in response.headers["content-disposition"]
                
        finally:
            Path(video_path).unlink(missing_ok=True)


class TestMonitoringEndpoints:
    """Test monitoring endpoints."""
    
    def test_monitoring_status(self, client):
        """Test monitoring status endpoint."""
        response = client.get("/monitoring/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "metrics" in data
    
    def test_monitoring_dashboard(self, client):
        """Test monitoring dashboard endpoint."""
        response = client.get("/monitoring/dashboard")
        
        assert response.status_code == 200
        data = response.json()
        assert "dashboard" in data
        assert "metrics" in data
        assert "alerts" in data
    
    def test_monitoring_alerts(self, client):
        """Test monitoring alerts endpoint."""
        response = client.get("/monitoring/alerts")
        
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert isinstance(data["alerts"], list)
    
    def test_monitoring_metrics(self, client):
        """Test monitoring metrics endpoint."""
        response = client.get("/monitoring/metrics")
        
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)
    
    def test_monitoring_thresholds(self, client):
        """Test monitoring thresholds endpoint."""
        response = client.get("/monitoring/thresholds")
        
        assert response.status_code == 200
        data = response.json()
        assert "thresholds" in data
        assert isinstance(data["thresholds"], dict)
    
    def test_monitoring_workflows(self, client):
        """Test monitoring workflows endpoint."""
        response = client.get("/monitoring/workflows")
        
        assert response.status_code == 200
        data = response.json()
        assert "workflows" in data
        assert isinstance(data["workflows"], list)


class TestAPIErrorHandling:
    """Test API error handling."""
    
    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON in request."""
        response = client.post(
            "/predict/video",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post("/predict/video")
        
        assert response.status_code == 422
    
    def test_large_file_upload(self, client, mock_job_manager):
        """Test handling of large file upload."""
        # Create a large file (simulate)
        large_content = b"x" * (10 * 1024 * 1024)  # 10MB
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(large_content)
            video_path = f.name
        
        try:
            with open(video_path, "rb") as video_file:
                response = client.post(
                    "/predict/video",
                    files={"file": ("large_video.mp4", video_file, "video/mp4")}
                )
            
            # Should handle large files gracefully
            assert response.status_code in [200, 413]  # 413 = Payload Too Large
            
        finally:
            Path(video_path).unlink(missing_ok=True)
    
    def test_concurrent_requests(self, client, mock_job_manager):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                results.append(f"error: {e}")
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(result == 200 for result in results)


class TestAPIIntegration:
    """Integration tests for complete API workflow."""
    
    def test_complete_video_analysis_workflow(self, client, mock_job_manager, mock_config):
        """Test complete video analysis workflow through API."""
        # Step 1: Upload video and create job
        mock_job = Mock()
        mock_job.job_id = 1
        mock_job.status = JobStatus.PENDING
        mock_job_manager.create_job.return_value = mock_job
        
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            video_path = f.name
        
        try:
            with open(video_path, "rb") as video_file:
                response = client.post(
                    "/predict/video",
                    files={"file": ("test_video.mp4", video_file, "video/mp4")},
                    data={"features": json.dumps({"passes": True, "possession": True})}
                )
            
            assert response.status_code == 200
            job_data = response.json()
            job_id = job_data["job_id"]
            
            # Step 2: Check job status
            mock_job.status = JobStatus.PROCESSING
            mock_job.progress = 50
            mock_job_manager.get_job.return_value = mock_job
            
            response = client.get(f"/predict/job/{job_id}")
            assert response.status_code == 200
            status_data = response.json()
            assert status_data["status"] == "processing"
            
            # Step 3: Check monitoring status
            response = client.get("/monitoring/status")
            assert response.status_code == 200
            
            # Step 4: Get all jobs
            mock_job_manager.get_all_jobs.return_value = [mock_job]
            response = client.get("/predict/jobs")
            assert response.status_code == 200
            jobs_data = response.json()
            assert len(jobs_data["jobs"]) == 1
            
        finally:
            Path(video_path).unlink(missing_ok=True) 