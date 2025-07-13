"""
Unit tests for detection modules.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.detection.yolo_detector import YOLODetector
from src.detection.detection_result import DetectionResult, Detection
from src.detection.detection_utils import filter_detections_by_confidence


@pytest.mark.unit
class TestYOLODetector:
    """Test YOLO detector functionality."""
    
    def test_yolo_detector_creation(self, mock_config):
        """Test YOLO detector creation."""
        with patch('src.detection.yolo_detector.YOLO') as mock_yolo:
            detector = YOLODetector(
                model_path=mock_config.model_path,
                confidence_threshold=mock_config.confidence_threshold,
                iou_threshold=mock_config.iou_threshold
            )
            
            assert detector.model_path == mock_config.model_path
            assert detector.confidence_threshold == mock_config.confidence_threshold
            assert detector.iou_threshold == mock_config.iou_threshold
            mock_yolo.assert_called_once_with(mock_config.model_path)
    
    def test_yolo_detector_detect_success(self, sample_frame, mock_yolo_model):
        """Test successful object detection."""
        with patch('src.detection.yolo_detector.YOLO', return_value=mock_yolo_model):
            detector = YOLODetector(
                model_path=Path("models/yolo/best.pt"),
                confidence_threshold=0.5,
                iou_threshold=0.45
            )
            
            detections = detector.detect_frame(sample_frame)
            
            assert isinstance(detections, DetectionResult)
            assert len(detections.detections) > 0
            assert all(isinstance(det, Detection) for det in detections.detections)
            assert all(hasattr(det, 'class_name') for det in detections.detections)
            assert all(hasattr(det, 'confidence') for det in detections.detections)
            assert all(hasattr(det, 'bbox') for det in detections.detections)
    
    def test_yolo_detector_detect_no_objects(self, sample_frame):
        """Test detection with no objects found."""
        mock_model = Mock()
        mock_model.predict.return_value = [Mock(boxes=None)]
        
        with patch('src.detection.yolo_detector.YOLO', return_value=mock_model):
            detector = YOLODetector(
                model_path=Path("models/yolo/best.pt"),
                confidence_threshold=0.5,
                iou_threshold=0.45
            )
            
            detections = detector.detect_frame(sample_frame)
            
            assert isinstance(detections, DetectionResult)
            assert len(detections.detections) == 0
    
    def test_yolo_detector_detect_with_filtering(self, sample_frame, mock_yolo_model):
        """Test detection with confidence filtering."""
        with patch('src.detection.yolo_detector.YOLO', return_value=mock_yolo_model):
            detector = YOLODetector(
                model_path=Path("models/yolo/best.pt"),
                confidence_threshold=0.9,  # High threshold
                iou_threshold=0.45
            )
            
            detections = detector.detect_frame(sample_frame)
            
            # Should filter out low confidence detections
            assert all(det.confidence >= 0.9 for det in detections.detections)
    
    def test_yolo_detector_detect_error_handling(self, sample_frame):
        """Test detection error handling."""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")
        
        with patch('src.detection.yolo_detector.YOLO', return_value=mock_model):
            detector = YOLODetector(
                model_path=Path("models/yolo/best.pt"),
                confidence_threshold=0.5,
                iou_threshold=0.45
            )
            
            with pytest.raises(Exception):
                detector.detect_frame(sample_frame)
    
    def test_yolo_detector_invalid_model_path(self):
        """Test detector creation with invalid model path."""
        with pytest.raises(FileNotFoundError):
            YOLODetector(
                model_path=Path("nonexistent_model.pt"),
                confidence_threshold=0.5,
                iou_threshold=0.45
            )


@pytest.mark.unit
class TestDetectionResult:
    """Test detection result class."""
    
    def test_detection_result_creation(self):
        """Test detection result creation."""
        from src.detection.detection_result import Detection
        
        bbox = [100.0, 100.0, 200.0, 200.0]
        detection = Detection(
            bbox=bbox,
            confidence=0.95,
            class_id=0,
            class_name="player"
        )
        result = DetectionResult([detection])
        
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "player"
        assert result.detections[0].confidence == 0.95
        assert result.detections[0].bbox == bbox
        assert result.detections[0].class_id == 0
    
    def test_detection_result_to_dict(self):
        """Test conversion to dictionary."""
        from src.detection.detection_result import Detection
        
        bbox = [100.0, 100.0, 200.0, 200.0]
        detection = Detection(
            bbox=bbox,
            confidence=0.88,
            class_id=1,
            class_name="ball"
        )
        result = DetectionResult([detection])
        
        result_dict = result.to_dict()
        
        assert len(result_dict['detections']) == 1
        assert result_dict['detections'][0]['class_name'] == "ball"
        assert result_dict['detections'][0]['confidence'] == 0.88
        assert result_dict['detections'][0]['bbox'] == bbox
        assert result_dict['detections'][0]['class_id'] == 1
    
    def test_detection_result_equality(self):
        """Test detection result equality."""
        from src.detection.detection_result import Detection
        
        bbox1 = [100.0, 100.0, 200.0, 200.0]
        bbox2 = [100.0, 100.0, 200.0, 200.0]
        bbox3 = [300.0, 300.0, 400.0, 400.0]
        
        detection1 = Detection(bbox=bbox1, confidence=0.95, class_id=0, class_name="player")
        detection2 = Detection(bbox=bbox2, confidence=0.95, class_id=0, class_name="player")
        detection3 = Detection(bbox=bbox3, confidence=0.95, class_id=0, class_name="player")
        
        result1 = DetectionResult([detection1])
        result2 = DetectionResult([detection2])
        result3 = DetectionResult([detection3])
        
        assert result1.detections[0] == result2.detections[0]
        assert result1.detections[0] != result3.detections[0]
    
    def test_detection_result_str_representation(self):
        """Test string representation."""
        from src.detection.detection_result import Detection
        
        bbox = [100.0, 100.0, 200.0, 200.0]
        detection = Detection(bbox=bbox, confidence=0.95, class_id=0, class_name="player")
        result = DetectionResult([detection])
        
        str_repr = str(result)
        assert "player" in str_repr
        assert "0.95" in str_repr
        assert "bbox" in str_repr


@pytest.mark.unit
class TestDetectionUtils:
    """Test detection utility functions."""
    
    def test_filter_detections_by_confidence(self):
        """Test filtering detections by confidence threshold."""
        detections = [
            {'class_name': 'player', 'confidence': 0.95, 'bbox': [100, 100, 200, 200]},
            {'class_name': 'ball', 'confidence': 0.88, 'bbox': [150, 150, 170, 170]},
            {'class_name': 'player', 'confidence': 0.45, 'bbox': [300, 300, 400, 400]},
            {'class_name': 'referee', 'confidence': 0.75, 'bbox': [500, 500, 600, 600]}
        ]
        
        filtered = filter_detections_by_confidence(detections, threshold=0.5)
        
        assert len(filtered) == 3  # Should filter out the 0.45 confidence detection
        assert all(det['confidence'] >= 0.5 for det in filtered)
    
    def test_filter_detections_by_confidence_empty(self):
        """Test filtering empty detection list."""
        detections = []
        filtered = filter_detections_by_confidence(detections, threshold=0.5)
        
        assert len(filtered) == 0
    
    def test_filter_detections_by_confidence_all_filtered(self):
        """Test filtering when all detections are below threshold."""
        detections = [
            {'class_name': 'player', 'confidence': 0.3, 'bbox': [100, 100, 200, 200]},
            {'class_name': 'ball', 'confidence': 0.2, 'bbox': [150, 150, 170, 170]}
        ]
        
        filtered = filter_detections_by_confidence(detections, threshold=0.5)
        
        assert len(filtered) == 0
    
    def test_filter_detections_by_confidence_none_filtered(self):
        """Test filtering when no detections are below threshold."""
        detections = [
            {'class_name': 'player', 'confidence': 0.95, 'bbox': [100, 100, 200, 200]},
            {'class_name': 'ball', 'confidence': 0.88, 'bbox': [150, 150, 170, 170]}
        ]
        
        filtered = filter_detections_by_confidence(detections, threshold=0.5)
        
        assert len(filtered) == 2
        assert filtered == detections


@pytest.mark.unit
class TestDetectionIntegration:
    """Integration tests for detection functionality."""
    
    def test_detection_pipeline(self, sample_frame, mock_yolo_model):
        """Test complete detection pipeline."""
        with patch('src.detection.yolo_detector.YOLO', return_value=mock_yolo_model):
            # Create detector
            detector = YOLODetector(
                model_path=Path("models/yolo/best.pt"),
                confidence_threshold=0.5,
                iou_threshold=0.45
            )
            
            # Detect objects
            detections = detector.detect_frame(sample_frame)
            
            # Filter by confidence
            filtered_detections = detections.get_detections_by_confidence(0.5)
            
            # Convert to DetectionResult objects
            detection_results = []
            for det in filtered_detections:
                result = DetectionResult([det])
                detection_results.append(result)
            
            # Verify results
            assert len(detection_results) > 0
            assert all(isinstance(result, DetectionResult) for result in detection_results)
            assert all(result.confidence >= 0.5 for result in detection_results)
    
    def test_detection_with_different_thresholds(self, sample_frame, mock_yolo_model):
        """Test detection with different confidence thresholds."""
        with patch('src.detection.yolo_detector.YOLO', return_value=mock_yolo_model):
            detector = YOLODetector(
                model_path=Path("models/yolo/best.pt"),
                confidence_threshold=0.5,
                iou_threshold=0.45
            )
            
            # Low threshold
            detections_low = detector.detect_frame(sample_frame)
            filtered_low = detections_low.get_detections_by_confidence(0.3)
            
            # High threshold
            filtered_high = detections_low.get_detections_by_confidence(0.8)
            
            # Should have more detections with lower threshold
            assert len(filtered_low) >= len(filtered_high) 