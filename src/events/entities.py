"""Entity classes for football video analysis."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from src.utils.colors import BLUE


class Player:
    """Represents a player in the football match."""
    
    def __init__(self, player_id: int, data: Dict[str, Any], team=None):
        """Initialize a Player object.
        
        Args:
            player_id: Unique identifier for the player
            data: Dictionary containing player data (bbox, confidence, etc.)
            team: Team object the player belongs to
        """
        self.player_id = player_id
        self.bbox = data.get('bbox', [0, 0, 0, 0])
        self.confidence = data.get('confidence', 0.0)
        self.team = team
        self.has_ball = data.get('has_ball', False)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Calculate the center point of the player's bounding box."""
        if not self.bbox or len(self.bbox) != 4:
            return (0.0, 0.0)
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Calculate the area of the player's bounding box."""
        if not self.bbox or len(self.bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def __eq__(self, other) -> bool:
        """Check if two players are equal based on player_id."""
        return isinstance(other, Player) and self.player_id == other.player_id
    
    def __str__(self) -> str:
        """String representation of the player."""
        team_name = self.team.name if self.team else "Unknown"
        return f"Player {self.player_id} ({team_name})"
    
    @staticmethod
    def have_same_id(player1: Optional['Player'], player2: Optional['Player']) -> bool:
        """Check if two players have the same ID.
        
        Args:
            player1: First player
            player2: Second player
            
        Returns:
            True if players have the same ID, False otherwise
        """
        if player1 is None or player2 is None:
            return False
        return player1.player_id == player2.player_id


class Team:
    """Represents a team in the football match."""
    
    def __init__(self, name: str, abbreviation: str = "AT", color: Tuple[int, int, int] = BLUE):
        """Initialize a Team object.
        
        Args:
            name: Team name
            abbreviation: Team abbreviation (default: "AT")
            color: Team color in BGR format (default: blue)
        """
        self.name = name
        self.abbreviation = abbreviation
        self.color = color
        self.passes = []
        self.possession_time = 0.0
    
    def __eq__(self, other) -> bool:
        """Check if two teams are equal based on name."""
        return isinstance(other, Team) and self.name == other.name
    
    def __str__(self) -> str:
        """String representation of the team."""
        return f"{self.name} ({self.abbreviation})"


class Ball:
    """Represents the ball in the football match."""
    
    def __init__(self, bbox: Optional[List[float]]):
        """Initialize a Ball object.
        
        Args:
            bbox: Bounding box of the ball [x1, y1, x2, y2] or None
        """
        self.bbox = bbox
        self.center = self._get_center(bbox) if bbox else None
    
    def _get_center(self, bbox: List[float]) -> Optional[Tuple[float, float]]:
        """Calculate the center point of the ball's bounding box.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Center point (x, y) or None if invalid bbox
        """
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Calculate the area of the ball's bounding box."""
        if not self.bbox or len(self.bbox) != 4:
            return 0.0
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def __eq__(self, other) -> bool:
        """Check if two balls are equal based on bbox."""
        if not isinstance(other, Ball):
            return False
        if self.bbox is None and other.bbox is None:
            return True
        if self.bbox is None or other.bbox is None:
            return False
        return self.bbox == other.bbox
    
    def __str__(self) -> str:
        """String representation of the ball."""
        if self.center:
            return f"Ball at center {self.center}"
        return "Ball (no position)"

