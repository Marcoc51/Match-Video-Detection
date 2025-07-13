"""
Pytest configuration and shared fixtures for Match Video Detection tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.events.entities import Player, Team, Ball
from src.api.models import ProcessingJob, ProcessingStatus
from src.utils.colors import RED, BLUE


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a sample video file for testing."""
    video_path = temp_dir / "sample_video.mp4"
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    
    # Write 10 frames
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    
    out.release()
    return video_path


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection_result():
    """Create a sample detection result for testing."""
    from src.detection.detection_result import Detection, DetectionResult
    
    detections = [
        Detection(
            bbox=[100, 100, 200, 300],
            confidence=0.95,
            class_id=0,
            class_name="player"
        ),
        Detection(
            bbox=[300, 200, 320, 220],
            confidence=0.88,
            class_id=1,
            class_name="ball"
        )
    ]
    
    return DetectionResult(detections)


@pytest.fixture
def sample_tracks():
    """Create sample tracking data for testing."""
    return {
        "players": [
            {
                1: {"bbox": [100, 100, 200, 300], "position": (150, 200)},
                2: {"bbox": [300, 150, 400, 350], "position": (350, 250)}
            },
            {
                1: {"bbox": [110, 110, 210, 310], "position": (160, 210)},
                2: {"bbox": [310, 160, 410, 360], "position": (360, 260)}
            }
        ],
        "ball": [
            {
                1: {"bbox": [300, 200, 320, 220], "position": (310, 210)}
            },
            {
                1: {"bbox": [305, 205, 325, 225], "position": (315, 215)}
            }
        ],
        "referees": [
            {
                1: {"bbox": [500, 100, 550, 200], "position": (525, 150)}
            },
            {
                1: {"bbox": [505, 105, 555, 205], "position": (530, 155)}
            }
        ]
    }


@pytest.fixture
def sample_teams():
    """Create sample teams for testing."""
    home_team = Team(name="Home", abbreviation="HOM", color=RED)
    away_team = Team(name="Away", abbreviation="AWY", color=BLUE)
    return [home_team, away_team]


@pytest.fixture
def sample_players(sample_teams):
    """Create sample players for testing."""
    player_data1 = {
        'bbox': [100, 100, 200, 300],
        'confidence': 0.95,
        'team': 1,
        'has_ball': False
    }
    
    player_data2 = {
        'bbox': [300, 150, 400, 350],
        'confidence': 0.88,
        'team': 2,
        'has_ball': True
    }
    
    player1 = Player(player_id=1, data=player_data1, team=sample_teams[0])
    player2 = Player(player_id=2, data=player_data2, team=sample_teams[1])
    
    return [player1, player2]


@pytest.fixture
def sample_ball():
    """Create a sample ball for testing."""
    bbox = [300, 200, 320, 220]
    return Ball(bbox)


@pytest.fixture
def mock_yolo_detector():
    """Create a mock YOLO detector for testing."""
    with patch('src.detection.yolo_detector.YOLODetector') as mock:
        detector_instance = Mock()
        mock.return_value = detector_instance
        
        # Mock detection result
        mock_result = Mock()
        mock_result.boxes.xyxy = np.array([[100, 100, 200, 300], [300, 200, 320, 220]])
        mock_result.boxes.conf = np.array([0.95, 0.88])
        mock_result.boxes.cls = np.array([0, 1])
        mock_result.names = {0: "player", 1: "ball"}
        
        detector_instance.detect.return_value = mock_result
        detector_instance.detect_frames.return_value = [mock_result] * 5
        
        yield detector_instance


@pytest.fixture
def mock_tracker():
    """Create a mock tracker for testing."""
    with patch('src.tracking.tracker.Tracker') as mock:
        tracker_instance = Mock()
        mock.return_value = tracker_instance
        
        # Mock tracking results
        mock_tracks = {
            "players": [
                {1: {"bbox": [100, 100, 200, 300], "position": (150, 200)}},
                {1: {"bbox": [110, 110, 210, 310], "position": (160, 210)}}
            ],
            "ball": [
                {1: {"bbox": [300, 200, 320, 220], "position": (310, 210)}},
                {1: {"bbox": [305, 205, 325, 225], "position": (315, 215)}}
            ]
        }
        
        tracker_instance.get_object_tracks.return_value = mock_tracks
        tracker_instance.draw_annotations.return_value = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)] * 2
        
        yield tracker_instance


@pytest.fixture
def mock_team_assigner():
    """Create a mock team assigner for testing."""
    with patch('src.assignment.team_assigner.TeamAssigner') as mock:
        assigner_instance = Mock()
        mock.return_value = assigner_instance
        
        # Mock team assignment
        assigner_instance.get_player_team.return_value = 1
        assigner_instance.get_team_color.return_value = RED
        
        yield assigner_instance


@pytest.fixture
def sample_processing_job():
    """Create a sample processing job for testing."""
    return ProcessingJob(
        job_id="test_job_123",
        video_path="test_video.mp4",
        status=ProcessingStatus.PENDING,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z"
    )


@pytest.fixture
def mock_api_client():
    """Create a mock API client for testing."""
    with patch('requests.post') as mock_post, \
         patch('requests.get') as mock_get:
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "job_id": "test_job_123",
            "status": "completed",
            "result_url": "http://localhost:8000/results/test_job_123"
        }
        
        mock_post.return_value = mock_response
        mock_get.return_value = mock_response
        
        yield {
            'post': mock_post,
            'get': mock_get
        }


@pytest.fixture
def sample_camera_movement():
    """Create sample camera movement data for testing."""
    return [
        [0, 0],      # Frame 0: no movement
        [2, 1],      # Frame 1: slight movement
        [-1, 3],     # Frame 2: movement in opposite direction
        [0, 0],      # Frame 3: no movement
        [1, -2]      # Frame 4: movement
    ]


@pytest.fixture
def sample_speed_data():
    """Create sample speed data for testing."""
    return {
        "players": [
            {
                1: {"speed": 10.5, "distance": 5.2},
                2: {"speed": 8.3, "distance": 4.1}
            },
            {
                1: {"speed": 12.1, "distance": 6.0},
                2: {"speed": 9.7, "distance": 4.8}
            }
        ],
        "ball": [
            {
                1: {"speed": 25.0, "distance": 12.5}
            },
            {
                1: {"speed": 22.3, "distance": 11.1}
            }
        ]
    }


@pytest.fixture
def sample_pass_events():
    """Create sample pass events for testing."""
    from src.events.pass_event import PassEvent
    
    passes = [
        PassEvent(
            start_ball_bbox=[100, 100, 120, 120],
            end_ball_bbox=[200, 200, 220, 220],
            start_player_bbox=[90, 90, 130, 130],
            end_player_bbox=[190, 190, 230, 230],
            start_frame=10,
            end_frame=15
        ),
        PassEvent(
            start_ball_bbox=[300, 300, 320, 320],
            end_ball_bbox=[400, 400, 420, 420],
            start_player_bbox=[290, 290, 330, 330],
            end_player_bbox=[390, 390, 430, 430],
            start_frame=25,
            end_frame=30
        )
    ]
    
    return passes


@pytest.fixture
def mock_video_reader():
    """Create a mock video reader for testing."""
    def mock_read_video(video_path):
        """Mock video reading function."""
        return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)] * 10
    
    with patch('src.utils.video_utils.read_video', side_effect=mock_read_video):
        yield mock_read_video


@pytest.fixture
def mock_video_writer():
    """Create a mock video writer for testing."""
    def mock_save_video(frames, output_path, fps=30):
        """Mock video writing function."""
        # Create a dummy file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return True
    
    with patch('src.utils.video_utils.save_video', side_effect=mock_save_video):
        yield mock_save_video


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    mock_config = Mock()
    mock_config.model_path = Path("models/yolo/best.pt")
    mock_config.confidence_threshold = 0.5
    mock_config.iou_threshold = 0.45
    mock_config.YOLO_MODEL_PATH = Path("models/yolo/best.pt")
    mock_config.CONFIDENCE_THRESHOLD = 0.5
    mock_config.IOU_THRESHOLD = 0.45
    return mock_config


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model for testing."""
    mock_model = Mock()
    
    # Mock prediction result
    mock_result = Mock()
    mock_result.boxes.xyxy = np.array([[100, 100, 200, 300], [300, 200, 320, 220]])
    mock_result.boxes.conf = np.array([0.95, 0.88])
    mock_result.boxes.cls = np.array([0, 1])
    mock_result.names = {0: "player", 1: "ball"}
    
    mock_model.predict.return_value = [mock_result]
    
    return mock_model 