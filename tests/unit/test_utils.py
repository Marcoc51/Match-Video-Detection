"""
Unit tests for utility functions.
"""

import pytest
import numpy as np
from pathlib import Path
import cv2

from src.utils.bbox_utils import get_center_of_bbox, calculate_iou, calculate_distance
from src.utils.colors import RED, BLUE, GREEN, YELLOW, WHITE, BLACK
from src.utils.video_utils import read_video, save_video, draw_bezier_pass


class TestBboxUtils:
    """Test bounding box utility functions."""
    
    def test_get_center_of_bbox(self):
        """Test center calculation of bounding box."""
        bbox = [100, 100, 200, 300]
        center = get_center_of_bbox(bbox)
        
        expected_center = (150, 200)  # (x1+x2)/2, (y1+y2)/2
        assert center == expected_center
    
    def test_get_center_of_bbox_float(self):
        """Test center calculation with float coordinates."""
        bbox = [100.5, 100.5, 200.5, 300.5]
        center = get_center_of_bbox(bbox)
        
        expected_center = (150.5, 200.5)
        assert center == expected_center
    
    def test_get_center_of_bbox_invalid(self):
        """Test center calculation with invalid bbox."""
        with pytest.raises(ValueError):
            get_center_of_bbox([100, 100, 200])  # Missing coordinate
    
    def test_calculate_iou_same_box(self):
        """Test IoU calculation for identical boxes."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [100, 100, 200, 200]
        
        iou = calculate_iou(bbox1, bbox2)
        assert iou == 1.0
    
    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation for non-overlapping boxes."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [300, 300, 400, 400]
        
        iou = calculate_iou(bbox1, bbox2)
        assert iou == 0.0
    
    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation for partially overlapping boxes."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [150, 150, 250, 250]
        
        iou = calculate_iou(bbox1, bbox2)
        # Expected IoU = intersection_area / union_area
        # intersection = 50 * 50 = 2500
        # union = 10000 + 10000 - 2500 = 17500
        # IoU = 2500 / 17500 ≈ 0.143
        assert 0.14 < iou < 0.15
    
    def test_calculate_iou_invalid_bbox(self):
        """Test IoU calculation with invalid bbox."""
        with pytest.raises(ValueError):
            calculate_iou([100, 100, 200], [100, 100, 200, 200])
    
    def test_calculate_distance(self):
        """Test distance calculation between two points."""
        point1 = (100, 100)
        point2 = (200, 200)
        
        distance = calculate_distance(point1, point2)
        expected_distance = np.sqrt(100**2 + 100**2)  # ≈ 141.42
        assert abs(distance - expected_distance) < 0.01
    
    def test_calculate_distance_same_point(self):
        """Test distance calculation for same point."""
        point1 = (100, 100)
        point2 = (100, 100)
        
        distance = calculate_distance(point1, point2)
        assert distance == 0.0
    
    def test_calculate_distance_invalid_points(self):
        """Test distance calculation with invalid points."""
        with pytest.raises(ValueError):
            calculate_distance((100,), (200, 200))


class TestColors:
    """Test color constants."""
    
    def test_color_values(self):
        """Test that color constants have correct BGR values."""
        assert RED == (0, 0, 255)      # BGR format
        assert BLUE == (255, 0, 0)     # BGR format
        assert GREEN == (0, 255, 0)    # BGR format
        assert YELLOW == (0, 255, 255) # BGR format
        assert WHITE == (255, 255, 255) # BGR format
        assert BLACK == (0, 0, 0)      # BGR format
    
    def test_color_types(self):
        """Test that colors are tuples of integers."""
        colors = [RED, BLUE, GREEN, YELLOW, WHITE, BLACK]
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(isinstance(c, int) for c in color)
            assert all(0 <= c <= 255 for c in color)


class TestVideoUtils:
    """Test video utility functions."""
    
    def test_read_video_success(self, sample_video_path):
        """Test successful video reading."""
        frames = read_video(sample_video_path)
        
        assert isinstance(frames, list)
        assert len(frames) > 0
        assert all(isinstance(frame, np.ndarray) for frame in frames)
        assert all(frame.shape[2] == 3 for frame in frames)  # 3 channels
    
    def test_read_video_nonexistent(self):
        """Test video reading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_video("nonexistent_video.mp4")
    
    def test_save_video_success(self, sample_frame, temp_dir):
        """Test successful video saving."""
        output_path = temp_dir / "output_video.mp4"
        frames = [sample_frame] * 10  # 10 frames
        
        save_video(frames, output_path, fps=30)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    
    def test_save_video_empty_frames(self, temp_dir):
        """Test video saving with empty frame list."""
        output_path = temp_dir / "empty_video.mp4"
        
        with pytest.raises(ValueError):
            save_video([], output_path, fps=30)
    
    def test_draw_bezier_pass(self, sample_frame):
        """Test drawing Bézier curve for pass visualization."""
        start_point = (100, 100)
        end_point = (200, 200)
        
        result_frame = draw_bezier_pass(
            sample_frame.copy(), 
            start_point, 
            end_point, 
            color=GREEN,
            thickness=2
        )
        
        assert isinstance(result_frame, np.ndarray)
        assert result_frame.shape == sample_frame.shape
        assert result_frame.dtype == sample_frame.dtype
    
    def test_draw_bezier_pass_invalid_points(self, sample_frame):
        """Test drawing Bézier curve with invalid points."""
        # Test with negative coordinates
        result_frame = draw_bezier_pass(
            sample_frame.copy(),
            (-10, -10),
            (200, 200),
            color=GREEN,
            thickness=2
        )
        
        assert isinstance(result_frame, np.ndarray)
        assert result_frame.shape == sample_frame.shape
    
    def test_draw_bezier_pass_none_frame(self):
        """Test drawing Bézier curve with None frame."""
        with pytest.raises(ValueError):
            draw_bezier_pass(
                None,
                (100, 100),
                (200, 200),
                color=GREEN
            ) 