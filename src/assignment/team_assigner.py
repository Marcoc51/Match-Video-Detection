"""Team assignment for football video analysis."""

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from src.utils.colors import WHITE, RED, BLUE


class TeamAssigner:
    """Assigns team colors to players based on jersey colors."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """Initialize the team assigner.
        
        Args:
            confidence_threshold: Confidence threshold for team assignment
        """
        self.confidence_threshold = confidence_threshold
        self.team_colors = {
            1: RED,    # Red for Home team
            2: BLUE    # Blue for Away team
        }
    
    def assign_team_color(self, frame: np.ndarray, player_tracks: Dict[int, Dict]) -> None:
        """Assign team colors to players based on jersey analysis.
        
        Args:
            frame: Reference frame for color analysis
            player_tracks: Dictionary of player tracks
        """
        for player_id, track_data in player_tracks.items():
            bbox = track_data.get('bbox', [])
            if not bbox or len(bbox) != 4:
                continue
            
            # Extract player region
            player_region = self._extract_player_region(frame, bbox)
            if player_region is None:
                continue
            
            # Analyze jersey color
            team_id = self._analyze_jersey_color(player_region)
            if team_id is not None:
                track_data['team'] = team_id
    
    def get_player_team(
        self, 
        frame: np.ndarray, 
        bbox: List[float], 
        player_id: int
    ) -> Optional[int]:
        """Get team assignment for a specific player.
        
        Args:
            frame: Current frame
            bbox: Player bounding box
            player_id: Player ID
            
        Returns:
            Team ID (1 for home, 2 for away) or None if uncertain
        """
        if not bbox or len(bbox) != 4:
            return None
        
        # Extract player region
        player_region = self._extract_player_region(frame, bbox)
        if player_region is None:
            return None
        
        # Analyze jersey color
        return self._analyze_jersey_color(player_region)
    
    def _extract_player_region(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract the player region from the frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Player region as numpy array or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            height, width = frame.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Extract region
            player_region = frame[y1:y2, x1:x2]
            
            # Check if region is valid
            if player_region.size == 0:
                return None
            
            return player_region
            
        except Exception:
            return None
    
    def _analyze_jersey_color(self, player_region: np.ndarray) -> Optional[int]:
        """Analyze jersey color to determine team.
        
        Args:
            player_region: Player region image
            
        Returns:
            Team ID (1 for home, 2 for away) or None if uncertain
        """
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for red and blue
            # Red wraps around in HSV, so we need two ranges
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([170, 50, 50])
            red_upper2 = np.array([180, 255, 255])
            
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            
            # Create masks for each color
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = red_mask1 | red_mask2
            
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            # Calculate color percentages
            total_pixels = hsv.shape[0] * hsv.shape[1]
            red_pixels = np.sum(red_mask > 0)
            blue_pixels = np.sum(blue_mask > 0)
            
            red_percentage = red_pixels / total_pixels
            blue_percentage = blue_pixels / total_pixels
            
            # Determine team based on dominant color
            if red_percentage > self.confidence_threshold and red_percentage > blue_percentage:
                return 1  # Home team (red)
            elif blue_percentage > self.confidence_threshold and blue_percentage > red_percentage:
                return 2  # Away team (blue)
            else:
                return None  # Uncertain
                
        except Exception:
            return None
    
    def get_team_color(self, team_id: int) -> Tuple[int, int, int]:
        """Get the color for a specific team.
        
        Args:
            team_id: Team ID (1 for home, 2 for away)
            
        Returns:
            BGR color tuple
        """
        return self.team_colors.get(team_id, WHITE)  # Default white
    
    def update_team_colors(self, new_colors: Dict[int, Tuple[int, int, int]]) -> None:
        """Update team color definitions.
        
        Args:
            new_colors: Dictionary mapping team IDs to BGR colors
        """
        self.team_colors.update(new_colors)
    
    def get_team_statistics(self, tracks: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Get team assignment statistics.
        
        Args:
            tracks: Dictionary containing object tracks
            
        Returns:
            Dictionary containing team statistics
        """
        if "players" not in tracks:
            return {}
        
        team_counts = {1: 0, 2: 0, 0: 0}  # 0 for unassigned
        
        for frame_tracks in tracks["players"]:
            for track_data in frame_tracks.values():
                team_id = track_data.get('team', 0)
                team_counts[team_id] += 1
        
        total_players = sum(team_counts.values())
        
        return {
            "home_team_players": team_counts[1],
            "away_team_players": team_counts[2],
            "unassigned_players": team_counts[0],
            "total_players": total_players,
            "home_team_percentage": (team_counts[1] / total_players * 100) if total_players > 0 else 0,
            "away_team_percentage": (team_counts[2] / total_players * 100) if total_players > 0 else 0
        }
    
    def validate_team_assignments(self, tracks: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """Validate team assignments for consistency.
        
        Args:
            tracks: Dictionary containing object tracks
            
        Returns:
            List of validation issues
        """
        issues = []
        
        if "players" not in tracks:
            return issues
        
        # Check for players without team assignments
        for frame_num, frame_tracks in enumerate(tracks["players"]):
            for player_id, track_data in frame_tracks.items():
                if 'team' not in track_data or track_data['team'] is None:
                    issues.append({
                        "type": "missing_team",
                        "frame": frame_num,
                        "player_id": player_id,
                        "message": f"Player {player_id} has no team assignment in frame {frame_num}"
                    })
        
        # Check for inconsistent team assignments across frames
        player_teams = {}
        for frame_num, frame_tracks in enumerate(tracks["players"]):
            for player_id, track_data in frame_tracks.items():
                if 'team' in track_data and track_data['team'] is not None:
                    if player_id in player_teams:
                        if player_teams[player_id] != track_data['team']:
                            issues.append({
                                "type": "inconsistent_team",
                                "frame": frame_num,
                                "player_id": player_id,
                                "expected_team": player_teams[player_id],
                                "actual_team": track_data['team'],
                                "message": f"Player {player_id} team changed from {player_teams[player_id]} to {track_data['team']} in frame {frame_num}"
                            })
                    else:
                        player_teams[player_id] = track_data['team']
        
        return issues
    
    