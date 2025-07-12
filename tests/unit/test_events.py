"""
Unit tests for event detection modules.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.events.pass_event import PassEvent
from src.events.possession_tracker import PossessionTracker
from src.events.cross_detector import CrossDetector
from src.events.entities import Player, Team, Ball


class TestPassEvent:
    """Test pass event detection."""
    
    def test_pass_event_creation(self, sample_teams):
        """Test pass event creation."""
        pass_event = PassEvent()
        
        assert pass_event.current_player is None
        assert pass_event.pass_start_frame is None
        assert pass_event.pass_start_ball_bbox is None
        assert pass_event.pass_threshold == 10  # Default threshold
    
    def test_pass_event_update_with_player(self, sample_teams, sample_ball):
        """Test updating pass event with a player."""
        pass_event = PassEvent()
        
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        pass_event.update(player, sample_ball)
        
        assert pass_event.current_player == player
        assert pass_event.pass_start_frame is not None
        assert pass_event.pass_start_ball_bbox == sample_ball.bbox
    
    def test_pass_event_update_same_player(self, sample_teams, sample_ball):
        """Test updating pass event with same player."""
        pass_event = PassEvent()
        
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        # First update
        pass_event.update(player, sample_ball)
        start_frame = pass_event.pass_start_frame
        
        # Second update with same player
        pass_event.update(player, sample_ball)
        
        assert pass_event.pass_start_frame == start_frame  # Should not change
    
    def test_pass_event_update_different_player(self, sample_teams, sample_ball):
        """Test updating pass event with different player."""
        pass_event = PassEvent()
        
        player1_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player1 = Player(player_id=1, data=player1_data, team=sample_teams[0])
        
        player2_data = {
            'bbox': [300, 300, 400, 500],
            'confidence': 0.92,
            'team': 2,
            'has_ball': True
        }
        player2 = Player(player_id=2, data=player2_data, team=sample_teams[1])
        
        # First player
        pass_event.update(player1, sample_ball)
        start_frame1 = pass_event.pass_start_frame
        
        # Different player
        pass_event.update(player2, sample_ball)
        
        assert pass_event.current_player == player2
        assert pass_event.pass_start_frame != start_frame1  # Should change
    
    def test_pass_event_process_pass_success(self, sample_teams, sample_ball):
        """Test successful pass processing."""
        pass_event = PassEvent()
        
        player1_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player1 = Player(player_id=1, data=player1_data, team=sample_teams[0])
        
        player2_data = {
            'bbox': [300, 300, 400, 500],
            'confidence': 0.92,
            'team': 2,
            'has_ball': True
        }
        player2 = Player(player_id=2, data=player2_data, team=sample_teams[1])
        
        # Start with player1
        pass_event.update(player1, sample_ball)
        
        # Switch to player2 (different team)
        pass_event.update(player2, sample_ball)
        
        # Process pass
        pass_event.process_pass(sample_teams, frame_idx=30)
        
        # Check that pass was added to player1's team
        assert len(sample_teams[0].passes) == 1
        assert len(sample_teams[1].passes) == 0
    
    def test_pass_event_process_pass_same_team(self, sample_teams, sample_ball):
        """Test pass processing within same team."""
        pass_event = PassEvent()
        
        player1_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player1 = Player(player_id=1, data=player1_data, team=sample_teams[0])
        
        player2_data = {
            'bbox': [300, 300, 400, 500],
            'confidence': 0.92,
            'team': 1,  # Same team
            'has_ball': True
        }
        player2 = Player(player_id=2, data=player2_data, team=sample_teams[0])
        
        # Start with player1
        pass_event.update(player1, sample_ball)
        
        # Switch to player2 (same team)
        pass_event.update(player2, sample_ball)
        
        # Process pass
        pass_event.process_pass(sample_teams, frame_idx=30)
        
        # Check that pass was added to the team
        assert len(sample_teams[0].passes) == 1
        assert len(sample_teams[1].passes) == 0


class TestPossessionTracker:
    """Test possession tracking."""
    
    def test_possession_tracker_creation(self):
        """Test possession tracker creation."""
        tracker = PossessionTracker(team_ids=["HOM", "AWY"], fps=30)
        
        assert tracker.team_ids == ["HOM", "AWY"]
        assert tracker.fps == 30
        assert tracker.team_possession_frames == {"HOM": 0, "AWY": 0}
        assert tracker.duration == 0
    
    def test_possession_tracker_update(self, sample_teams, sample_ball):
        """Test possession tracker update."""
        tracker = PossessionTracker(team_ids=["HOM", "AWY"], fps=30)
        
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        tracker.update(player, sample_ball)
        
        assert tracker.team_possession_frames["HOM"] == 1
        assert tracker.team_possession_frames["AWY"] == 0
        assert tracker.duration == 1
    
    def test_possession_tracker_multiple_updates(self, sample_teams, sample_ball):
        """Test multiple possession tracker updates."""
        tracker = PossessionTracker(team_ids=["HOM", "AWY"], fps=30)
        
        player1_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player1 = Player(player_id=1, data=player1_data, team=sample_teams[0])
        
        player2_data = {
            'bbox': [300, 300, 400, 500],
            'confidence': 0.92,
            'team': 2,
            'has_ball': True
        }
        player2 = Player(player_id=2, data=player2_data, team=sample_teams[1])
        
        # Update with player1 (HOM team)
        for _ in range(5):
            tracker.update(player1, sample_ball)
        
        # Update with player2 (AWY team)
        for _ in range(3):
            tracker.update(player2, sample_ball)
        
        assert tracker.team_possession_frames["HOM"] == 5
        assert tracker.team_possession_frames["AWY"] == 3
        assert tracker.duration == 8
    
    def test_possession_tracker_percentage_calculation(self, sample_teams, sample_ball):
        """Test possession percentage calculation."""
        tracker = PossessionTracker(team_ids=["HOM", "AWY"], fps=30)
        
        player1_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player1 = Player(player_id=1, data=player1_data, team=sample_teams[0])
        
        player2_data = {
            'bbox': [300, 300, 400, 500],
            'confidence': 0.92,
            'team': 2,
            'has_ball': True
        }
        player2 = Player(player_id=2, data=player2_data, team=sample_teams[1])
        
        # 6 frames for HOM, 4 frames for AWY
        for _ in range(6):
            tracker.update(player1, sample_ball)
        for _ in range(4):
            tracker.update(player2, sample_ball)
        
        hom_percentage = tracker.get_percentage_possession("HOM")
        awy_percentage = tracker.get_percentage_possession("AWY")
        
        assert hom_percentage == 0.6  # 6/10
        assert awy_percentage == 0.4  # 4/10
    
    def test_possession_tracker_get_all_percentages(self, sample_teams, sample_ball):
        """Test getting all possession percentages."""
        tracker = PossessionTracker(team_ids=["HOM", "AWY"], fps=30)
        
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        # 5 frames for HOM
        for _ in range(5):
            tracker.update(player, sample_ball)
        
        percentages = tracker.get_all_percentages()
        
        assert percentages["HOM"] == 1.0
        assert percentages["AWY"] == 0.0
    
    def test_possession_tracker_get_all_times(self, sample_teams, sample_ball):
        """Test getting all possession times."""
        tracker = PossessionTracker(team_ids=["HOM", "AWY"], fps=30)
        
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        # 30 frames = 1 second at 30 FPS
        for _ in range(30):
            tracker.update(player, sample_ball)
        
        times = tracker.get_all_times()
        
        assert times["HOM"] == 1.0  # 30 frames / 30 fps = 1 second
        assert times["AWY"] == 0.0


class TestCrossDetector:
    """Test cross detection."""
    
    def test_cross_detector_creation(self):
        """Test cross detector creation."""
        detector = CrossDetector()
        
        assert detector.cross_threshold == 0.5  # Default threshold
        assert len(detector.detected_crosses) == 0
    
    def test_cross_detector_with_custom_threshold(self):
        """Test cross detector with custom threshold."""
        detector = CrossDetector(cross_threshold=0.7)
        
        assert detector.cross_threshold == 0.7
    
    def test_cross_detector_detect_cross(self):
        """Test cross detection."""
        detector = CrossDetector()
        
        # Mock detection result with high confidence
        detection_result = {
            'class_name': 'cross',
            'confidence': 0.85,
            'bbox': [100, 100, 200, 200]
        }
        
        crosses = detector.detect_cross([detection_result])
        
        assert len(crosses) == 1
        assert crosses[0]['confidence'] == 0.85
        assert crosses[0]['bbox'] == [100, 100, 200, 200]
    
    def test_cross_detector_no_cross(self):
        """Test cross detection with no cross."""
        detector = CrossDetector()
        
        # Mock detection result with low confidence
        detection_result = {
            'class_name': 'cross',
            'confidence': 0.3,  # Below threshold
            'bbox': [100, 100, 200, 200]
        }
        
        crosses = detector.detect_cross([detection_result])
        
        assert len(crosses) == 0
    
    def test_cross_detector_wrong_class(self):
        """Test cross detection with wrong class."""
        detector = CrossDetector()
        
        # Mock detection result with wrong class
        detection_result = {
            'class_name': 'player',  # Not a cross
            'confidence': 0.95,
            'bbox': [100, 100, 200, 200]
        }
        
        crosses = detector.detect_cross([detection_result])
        
        assert len(crosses) == 0
    
    def test_cross_detector_multiple_detections(self):
        """Test cross detection with multiple detections."""
        detector = CrossDetector()
        
        detection_results = [
            {
                'class_name': 'cross',
                'confidence': 0.85,
                'bbox': [100, 100, 200, 200]
            },
            {
                'class_name': 'cross',
                'confidence': 0.75,
                'bbox': [300, 300, 400, 400]
            },
            {
                'class_name': 'player',
                'confidence': 0.95,
                'bbox': [500, 500, 600, 600]
            }
        ]
        
        crosses = detector.detect_cross(detection_results)
        
        assert len(crosses) == 2  # Only the cross detections
        assert crosses[0]['confidence'] == 0.85
        assert crosses[1]['confidence'] == 0.75 