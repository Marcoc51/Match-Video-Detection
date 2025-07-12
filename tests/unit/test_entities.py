"""
Unit tests for entity classes (Player, Team, Ball).
"""

import pytest
import numpy as np

from src.events.entities import Player, Team, Ball
from src.utils.colors import RED, BLUE, YELLOW


class TestTeam:
    """Test Team class."""
    
    def test_team_creation(self):
        """Test team creation with basic parameters."""
        team = Team(name="Home Team", abbreviation="HOM", color=RED)
        
        assert team.name == "Home Team"
        assert team.abbreviation == "HOM"
        assert team.color == RED
        assert len(team.passes) == 0
        assert team.possession_time == 0
    
    def test_team_default_values(self):
        """Test team creation with default values."""
        team = Team(name="Away Team")
        
        assert team.name == "Away Team"
        assert team.abbreviation == "AT"  # Default abbreviation
        assert team.color == BLUE  # Default blue color
        assert len(team.passes) == 0
        assert team.possession_time == 0
    
    def test_team_add_pass(self):
        """Test adding a pass to a team."""
        team = Team(name="Test Team", abbreviation="TST", color=YELLOW)
        
        # Create a mock pass
        mock_pass = type('Pass', (), {
            'start_ball_bbox': [100, 100, 120, 120],
            'end_ball_bbox': [200, 200, 220, 220]
        })()
        
        team.passes.append(mock_pass)
        
        assert len(team.passes) == 1
        assert team.passes[0] == mock_pass
    
    def test_team_update_possession(self):
        """Test updating team possession time."""
        team = Team(name="Test Team")
        
        team.possession_time += 10.5
        assert team.possession_time == 10.5
        
        team.possession_time += 5.2
        assert team.possession_time == 15.7
    
    def test_team_str_representation(self):
        """Test string representation of team."""
        team = Team(name="Home Team", abbreviation="HOM", color=RED)
        
        str_repr = str(team)
        assert "Home Team" in str_repr
        assert "HOM" in str_repr


class TestPlayer:
    """Test Player class."""
    
    def test_player_creation(self, sample_teams):
        """Test player creation with team."""
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': False
        }
        
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        assert player.player_id == 1
        assert player.bbox == [100, 100, 200, 300]
        assert player.confidence == 0.95
        assert player.team == sample_teams[0]
        assert player.has_ball == False
    
    def test_player_creation_without_team(self):
        """Test player creation without team."""
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': True
        }
        
        player = Player(player_id=2, data=player_data)
        
        assert player.player_id == 2
        assert player.bbox == [100, 100, 200, 300]
        assert player.confidence == 0.95
        assert player.team is None
        assert player.has_ball == True
    
    def test_player_center_calculation(self, sample_teams):
        """Test player center calculation."""
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': False
        }
        
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        center = player.center
        expected_center = (150, 200)  # (x1+x2)/2, (y1+y2)/2
        assert center == expected_center
    
    def test_player_area_calculation(self, sample_teams):
        """Test player area calculation."""
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': False
        }
        
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        area = player.area
        expected_area = 100 * 200  # width * height
        assert area == expected_area
    
    def test_player_str_representation(self, sample_teams):
        """Test string representation of player."""
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': False
        }
        
        player = Player(player_id=1, data=player_data, team=sample_teams[0])
        
        str_repr = str(player)
        assert "Player 1" in str_repr
        assert "Home" in str_repr  # Team name
    
    def test_player_equality(self, sample_teams):
        """Test player equality comparison."""
        player_data = {
            'bbox': [100, 100, 200, 300],
            'confidence': 0.95,
            'team': 1,
            'has_ball': False
        }
        
        player1 = Player(player_id=1, data=player_data, team=sample_teams[0])
        player2 = Player(player_id=1, data=player_data, team=sample_teams[0])
        player3 = Player(player_id=2, data=player_data, team=sample_teams[0])
        
        assert player1 == player2
        assert player1 != player3


class TestBall:
    """Test Ball class."""
    
    def test_ball_creation(self):
        """Test ball creation with bbox."""
        bbox = [100, 100, 120, 120]
        ball = Ball(bbox)
        
        assert ball.bbox == bbox
        assert ball.center == (110, 110)  # (x1+x2)/2, (y1+y2)/2
    
    def test_ball_creation_none(self):
        """Test ball creation with None bbox."""
        ball = Ball(None)
        
        assert ball.bbox is None
        assert ball.center is None
    
    def test_ball_center_calculation(self):
        """Test ball center calculation."""
        bbox = [50, 75, 70, 95]
        ball = Ball(bbox)
        
        center = ball.center
        expected_center = (60, 85)  # (x1+x2)/2, (y1+y2)/2
        assert center == expected_center
    
    def test_ball_area_calculation(self):
        """Test ball area calculation."""
        bbox = [100, 100, 120, 120]
        ball = Ball(bbox)
        
        area = ball.area
        expected_area = 20 * 20  # width * height
        assert area == expected_area
    
    def test_ball_area_none_bbox(self):
        """Test ball area calculation with None bbox."""
        ball = Ball(None)
        assert ball.area == 0.0
    
    def test_ball_str_representation(self):
        """Test string representation of ball."""
        bbox = [100, 100, 120, 120]
        ball = Ball(bbox)
        
        str_repr = str(ball)
        assert "Ball at center" in str_repr
        assert "(110, 110)" in str_repr
    
    def test_ball_str_representation_none(self):
        """Test string representation of ball with None bbox."""
        ball = Ball(None)
        
        str_repr = str(ball)
        assert "Ball (no position)" in str_repr
    
    def test_ball_equality(self):
        """Test ball equality comparison."""
        bbox1 = [100, 100, 120, 120]
        bbox2 = [100, 100, 120, 120]
        bbox3 = [200, 200, 220, 220]
        
        ball1 = Ball(bbox1)
        ball2 = Ball(bbox2)
        ball3 = Ball(bbox3)
        ball4 = Ball(None)
        ball5 = Ball(None)
        
        assert ball1 == ball2
        assert ball1 != ball3
        assert ball4 == ball5
        assert ball1 != ball4 