"""
Player-ball assignment functionality for football match analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ..utils.bbox_utils import get_center_of_bbox
from ..core.config import Config


class PlayerBallAssigner:
    """
    Assigns ball possession to players based on proximity.
    
    This class determines which player has possession of the ball
    by calculating distances between players and the ball.
    """
    
    def __init__(self, max_distance_threshold: float = 70.0):
        """
        Initialize the player-ball assigner.
        
        Args:
            max_distance_threshold: Maximum distance for ball assignment
        """
        self.max_distance_threshold = max_distance_threshold
        self.assignment_history = []
    
    def calculate_distance(
        self, 
        point1: Tuple[float, float], 
        point2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            Euclidean distance
        """
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def assign_ball_to_player(
        self, 
        players: Dict[int, Dict[str, Any]], 
        ball_bbox: Tuple[float, float, float, float]
    ) -> int:
        """
        Assign ball possession to the closest player within threshold.
        
        Args:
            players: Dictionary of player detections
            ball_bbox: Ball bounding box (x1, y1, x2, y2)
            
        Returns:
            Player ID with ball possession, or -1 if no player is close enough
        """
        if not players or not ball_bbox:
            return -1
        
        ball_center = get_center_of_bbox(ball_bbox)
        minimum_distance = float('inf')
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']
            
            # Calculate distance from player's foot position to ball
            player_center = get_center_of_bbox(player_bbox)
            
            # Use foot position (bottom center of player)
            player_foot_x = player_center[0]
            player_foot_y = player_bbox[3]  # Bottom of bounding box
            
            # Calculate distance from player's foot to ball center
            distance = self.calculate_distance(
                (player_foot_x, player_foot_y), 
                ball_center
            )
            
            # Check if player is within threshold and closest
            if distance < self.max_distance_threshold and distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id
        
        # Record assignment
        if assigned_player != -1:
            self.assignment_history.append({
                'frame': len(self.assignment_history),
                'player_id': assigned_player,
                'distance': minimum_distance,
                'ball_center': ball_center
            })
        
        return assigned_player
    
    def assign_ball_to_player_advanced(
        self, 
        players: Dict[int, Dict[str, Any]], 
        ball_bbox: Tuple[float, float, float, float],
        ball_velocity: Optional[Tuple[float, float]] = None,
        player_velocities: Optional[Dict[int, Tuple[float, float]]] = None
    ) -> int:
        """
        Advanced ball assignment considering velocity and multiple factors.
        
        Args:
            players: Dictionary of player detections
            ball_bbox: Ball bounding box
            ball_velocity: Ball velocity vector (optional)
            player_velocities: Player velocity vectors (optional)
            
        Returns:
            Player ID with ball possession, or -1 if no player is close enough
        """
        if not players or not ball_bbox:
            return -1
        
        ball_center = get_center_of_bbox(ball_bbox)
        best_score = float('inf')
        assigned_player = -1
        
        for player_id, player in players.items():
            player_bbox = player['bbox']
            player_center = get_center_of_bbox(player_bbox)
            
            # Calculate distance score
            distance = self.calculate_distance(player_center, ball_center)
            
            if distance > self.max_distance_threshold:
                continue
            
            # Initialize score with distance
            score = distance
            
            # Add velocity consideration if available
            if ball_velocity and player_velocities and player_id in player_velocities:
                player_velocity = player_velocities[player_id]
                
                # Calculate velocity alignment (dot product)
                velocity_alignment = (
                    ball_velocity[0] * player_velocity[0] + 
                    ball_velocity[1] * player_velocity[1]
                )
                
                # Add velocity penalty (negative alignment increases score)
                score += max(0, -velocity_alignment) * 0.1
            
            # Add team consideration (prefer players from the same team)
            if 'team' in player:
                # This could be enhanced with team-based logic
                pass
            
            # Update best assignment
            if score < best_score:
                best_score = score
                assigned_player = player_id
        
        # Record assignment
        if assigned_player != -1:
            self.assignment_history.append({
                'frame': len(self.assignment_history),
                'player_id': assigned_player,
                'score': best_score,
                'ball_center': ball_center,
                'method': 'advanced'
            })
        
        return assigned_player
    
    def get_ball_possession_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of ball possession assignments.
        
        Returns:
            List of assignment records
        """
        return self.assignment_history.copy()
    
    def get_player_possession_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get possession statistics for each player.
        
        Returns:
            Dictionary with player possession statistics
        """
        player_stats = {}
        
        for assignment in self.assignment_history:
            player_id = assignment['player_id']
            
            if player_id not in player_stats:
                player_stats[player_id] = {
                    'total_possessions': 0,
                    'total_distance': 0.0,
                    'avg_distance': 0.0,
                    'min_distance': float('inf'),
                    'max_distance': 0.0
                }
            
            stats = player_stats[player_id]
            stats['total_possessions'] += 1
            
            distance = assignment.get('distance', 0.0)
            stats['total_distance'] += distance
            stats['min_distance'] = min(stats['min_distance'], distance)
            stats['max_distance'] = max(stats['max_distance'], distance)
        
        # Calculate averages
        for player_id, stats in player_stats.items():
            if stats['total_possessions'] > 0:
                stats['avg_distance'] = stats['total_distance'] / stats['total_possessions']
            if stats['min_distance'] == float('inf'):
                stats['min_distance'] = 0.0
        
        return player_stats
    
    def get_team_possession_stats(self, team_assignments: Dict[int, int]) -> Dict[int, Dict[str, Any]]:
        """
        Get possession statistics for each team.
        
        Args:
            team_assignments: Dictionary mapping player IDs to team IDs
            
        Returns:
            Dictionary with team possession statistics
        """
        team_stats = {}
        
        for assignment in self.assignment_history:
            player_id = assignment['player_id']
            
            if player_id not in team_assignments:
                continue
            
            team_id = team_assignments[player_id]
            
            if team_id not in team_stats:
                team_stats[team_id] = {
                    'total_possessions': 0,
                    'total_distance': 0.0,
                    'avg_distance': 0.0,
                    'players': set()
                }
            
            stats = team_stats[team_id]
            stats['total_possessions'] += 1
            stats['total_distance'] += assignment.get('distance', 0.0)
            stats['players'].add(player_id)
        
        # Calculate averages and convert sets to lists
        for team_id, stats in team_stats.items():
            if stats['total_possessions'] > 0:
                stats['avg_distance'] = stats['total_distance'] / stats['total_possessions']
            stats['players'] = list(stats['players'])
            stats['unique_players'] = len(stats['players'])
        
        return team_stats
    
    def update_distance_threshold(self, new_threshold: float) -> None:
        """
        Update the maximum distance threshold for ball assignment.
        
        Args:
            new_threshold: New distance threshold
        """
        self.max_distance_threshold = new_threshold
        print(f"Updated ball assignment distance threshold to {new_threshold}")
    
    def reset_history(self) -> None:
        """Reset the assignment history."""
        self.assignment_history = []
        print("Ball assignment history reset")
    
    def get_assignment_confidence(
        self, 
        player_bbox: Tuple[float, float, float, float], 
        ball_bbox: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculate confidence score for ball assignment.
        
        Args:
            player_bbox: Player bounding box
            ball_bbox: Ball bounding box
            
        Returns:
            Confidence score between 0 and 1
        """
        player_center = get_center_of_bbox(player_bbox)
        ball_center = get_center_of_bbox(ball_bbox)
        
        distance = self.calculate_distance(player_center, ball_center)
        
        # Convert distance to confidence (closer = higher confidence)
        if distance <= self.max_distance_threshold:
            confidence = 1.0 - (distance / self.max_distance_threshold)
            return max(0.0, confidence)
        else:
            return 0.0
