"""
Utility functions for assignment processing and analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..core.config import Config


class AssignmentUtils:
    """
    Utility class for assignment processing and analysis.
    
    This class provides static methods for common assignment tasks
    such as validation, analysis, and data processing.
    """
    
    @staticmethod
    def validate_team_assignment(
        team_assignments: Dict[int, int],
        expected_teams: List[int] = [1, 2]
    ) -> Dict[str, Any]:
        """
        Validate team assignments for consistency.
        
        Args:
            team_assignments: Dictionary mapping player IDs to team IDs
            expected_teams: List of expected team IDs
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            "is_valid": True,
            "total_players": len(team_assignments),
            "team_counts": {},
            "missing_teams": [],
            "unexpected_teams": [],
            "errors": []
        }
        
        # Count players per team
        for player_id, team_id in team_assignments.items():
            if team_id not in validation_result["team_counts"]:
                validation_result["team_counts"][team_id] = 0
            validation_result["team_counts"][team_id] += 1
        
        # Check for missing teams
        for expected_team in expected_teams:
            if expected_team not in validation_result["team_counts"]:
                validation_result["missing_teams"].append(expected_team)
                validation_result["is_valid"] = False
        
        # Check for unexpected teams
        for team_id in validation_result["team_counts"]:
            if team_id not in expected_teams:
                validation_result["unexpected_teams"].append(team_id)
                validation_result["is_valid"] = False
        
        # Check for reasonable team sizes
        for team_id, count in validation_result["team_counts"].items():
            if count < 5:  # Minimum reasonable team size
                validation_result["errors"].append(f"Team {team_id} has only {count} players")
                validation_result["is_valid"] = False
            elif count > 15:  # Maximum reasonable team size
                validation_result["errors"].append(f"Team {team_id} has {count} players (suspicious)")
        
        return validation_result
    
    @staticmethod
    def analyze_team_distribution(
        team_assignments: Dict[int, int],
        player_positions: Dict[int, Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Analyze the spatial distribution of teams.
        
        Args:
            team_assignments: Dictionary mapping player IDs to team IDs
            player_positions: Dictionary mapping player IDs to positions
            
        Returns:
            Dictionary containing distribution analysis
        """
        analysis = {
            "team_centers": {},
            "team_spreads": {},
            "field_occupancy": {},
            "team_separation": 0.0
        }
        
        # Group players by team
        team_players = {}
        for player_id, team_id in team_assignments.items():
            if player_id in player_positions:
                if team_id not in team_players:
                    team_players[team_id] = []
                team_players[team_id].append(player_positions[player_id])
        
        # Calculate team centers and spreads
        for team_id, positions in team_players.items():
            if not positions:
                continue
            
            # Calculate team center
            center_x = np.mean([pos[0] for pos in positions])
            center_y = np.mean([pos[1] for pos in positions])
            analysis["team_centers"][team_id] = (center_x, center_y)
            
            # Calculate team spread (standard deviation)
            distances_from_center = [
                np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
                for pos in positions
            ]
            analysis["team_spreads"][team_id] = np.std(distances_from_center)
            
            # Calculate field occupancy (area covered by team)
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            analysis["field_occupancy"][team_id] = x_range * y_range
        
        # Calculate team separation
        if len(analysis["team_centers"]) >= 2:
            team_ids = list(analysis["team_centers"].keys())
            center1 = analysis["team_centers"][team_ids[0]]
            center2 = analysis["team_centers"][team_ids[1]]
            analysis["team_separation"] = np.sqrt(
                (center1[0] - center2[0])**2 + (center1[1] - center2[1])**2
            )
        
        return analysis
    
    @staticmethod
    def calculate_possession_metrics(
        possession_history: List[Dict[str, Any]],
        team_assignments: Dict[int, int],
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive possession metrics.
        
        Args:
            possession_history: List of possession records
            team_assignments: Dictionary mapping player IDs to team IDs
            fps: Frames per second
            
        Returns:
            Dictionary containing possession metrics
        """
        metrics = {
            "total_frames": len(possession_history),
            "team_possession": {},
            "possession_changes": 0,
            "avg_possession_duration": 0.0,
            "possession_sequences": []
        }
        
        if not possession_history:
            return metrics
        
        # Initialize team possession counters
        for team_id in set(team_assignments.values()):
            metrics["team_possession"][team_id] = {
                "frames": 0,
                "percentage": 0.0,
                "sequences": 0,
                "avg_sequence_duration": 0.0
            }
        
        # Analyze possession history
        current_team = None
        current_sequence_start = 0
        possession_sequences = []
        
        for frame_idx, possession in enumerate(possession_history):
            player_id = possession.get('player_id', -1)
            
            if player_id in team_assignments:
                team_id = team_assignments[player_id]
                
                # Count possession frames
                if team_id not in metrics["team_possession"]:
                    metrics["team_possession"][team_id] = {
                        "frames": 0,
                        "percentage": 0.0,
                        "sequences": 0,
                        "avg_sequence_duration": 0.0
                    }
                
                metrics["team_possession"][team_id]["frames"] += 1
                
                # Track possession changes
                if current_team is not None and current_team != team_id:
                    metrics["possession_changes"] += 1
                    
                    # Record previous sequence
                    if current_sequence_start < frame_idx:
                        possession_sequences.append({
                            "team": current_team,
                            "start_frame": current_sequence_start,
                            "end_frame": frame_idx - 1,
                            "duration_frames": frame_idx - current_sequence_start,
                            "duration_seconds": (frame_idx - current_sequence_start) / fps
                        })
                
                # Start new sequence if needed
                if current_team != team_id:
                    current_team = team_id
                    current_sequence_start = frame_idx
        
        # Record final sequence
        if current_team is not None and current_sequence_start < len(possession_history):
            possession_sequences.append({
                "team": current_team,
                "start_frame": current_sequence_start,
                "end_frame": len(possession_history) - 1,
                "duration_frames": len(possession_history) - current_sequence_start,
                "duration_seconds": (len(possession_history) - current_sequence_start) / fps
            })
        
        metrics["possession_sequences"] = possession_sequences
        
        # Calculate percentages and averages
        total_frames = len(possession_history)
        for team_id, team_metrics in metrics["team_possession"].items():
            team_metrics["percentage"] = (team_metrics["frames"] / total_frames) * 100
            
            # Count sequences for this team
            team_sequences = [seq for seq in possession_sequences if seq["team"] == team_id]
            team_metrics["sequences"] = len(team_sequences)
            
            if team_sequences:
                avg_duration = np.mean([seq["duration_seconds"] for seq in team_sequences])
                team_metrics["avg_sequence_duration"] = avg_duration
        
        # Calculate overall average possession duration
        if possession_sequences:
            metrics["avg_possession_duration"] = np.mean([
                seq["duration_seconds"] for seq in possession_sequences
            ])
        
        return metrics
    
    @staticmethod
    def create_possession_heatmap(
        possession_history: List[Dict[str, Any]],
        team_assignments: Dict[int, int],
        frame_shape: Tuple[int, int, int],
        blur_radius: int = 15
    ) -> Dict[int, np.ndarray]:
        """
        Create possession heatmaps for each team.
        
        Args:
            possession_history: List of possession records
            team_assignments: Dictionary mapping player IDs to team IDs
            frame_shape: Shape of the frames (height, width, channels)
            blur_radius: Radius for Gaussian blur
            
        Returns:
            Dictionary mapping team IDs to heatmaps
        """
        height, width, _ = frame_shape
        team_heatmaps = {}
        
        # Initialize heatmaps for each team
        for team_id in set(team_assignments.values()):
            team_heatmaps[team_id] = np.zeros((height, width), dtype=np.float32)
        
        # Build heatmaps from possession history
        for possession in possession_history:
            player_id = possession.get('player_id', -1)
            ball_center = possession.get('ball_center', (0, 0))
            
            if player_id in team_assignments:
                team_id = team_assignments[player_id]
                x, y = int(ball_center[0]), int(ball_center[1])
                
                if 0 <= x < width and 0 <= y < height:
                    team_heatmaps[team_id][y, x] += 1
        
        # Normalize and blur heatmaps
        for team_id, heatmap in team_heatmaps.items():
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            # Apply Gaussian blur for smooth visualization
            heatmap = cv2.GaussianBlur(heatmap, (blur_radius, blur_radius), 0)
            team_heatmaps[team_id] = heatmap
        
        return team_heatmaps
    
    @staticmethod
    def export_assignment_analysis(
        team_assignments: Dict[int, int],
        possession_history: List[Dict[str, Any]],
        output_path: Path,
        fps: float = 30.0
    ) -> None:
        """
        Export assignment analysis to CSV.
        
        Args:
            team_assignments: Dictionary mapping player IDs to team IDs
            possession_history: List of possession records
            output_path: Path to save the analysis
            fps: Frames per second
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Write team assignments
            f.write("=== TEAM ASSIGNMENTS ===\n")
            f.write("player_id,team_id\n")
            for player_id, team_id in team_assignments.items():
                f.write(f"{player_id},{team_id}\n")
            
            f.write("\n=== POSSESSION HISTORY ===\n")
            f.write("frame,player_id,team_id,distance,ball_x,ball_y\n")
            
            for frame_idx, possession in enumerate(possession_history):
                player_id = possession.get('player_id', -1)
                team_id = team_assignments.get(player_id, -1)
                distance = possession.get('distance', 0.0)
                ball_center = possession.get('ball_center', (0, 0))
                
                f.write(f"{frame_idx},{player_id},{team_id},{distance:.2f},"
                       f"{ball_center[0]:.2f},{ball_center[1]:.2f}\n")
    
    @staticmethod
    def validate_ball_assignment(
        player_bbox: Tuple[float, float, float, float],
        ball_bbox: Tuple[float, float, float, float],
        max_distance: float = 70.0
    ) -> Dict[str, Any]:
        """
        Validate ball assignment based on spatial relationship.
        
        Args:
            player_bbox: Player bounding box
            ball_bbox: Ball bounding box
            max_distance: Maximum allowed distance
            
        Returns:
            Dictionary containing validation results
        """
        # Calculate centers
        player_center = ((player_bbox[0] + player_bbox[2]) / 2, (player_bbox[1] + player_bbox[3]) / 2)
        ball_center = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)
        
        # Calculate distance
        distance = np.sqrt(
            (player_center[0] - ball_center[0])**2 + 
            (player_center[1] - ball_center[1])**2
        )
        
        # Check overlap
        overlap_x = max(0, min(player_bbox[2], ball_bbox[2]) - max(player_bbox[0], ball_bbox[0]))
        overlap_y = max(0, min(player_bbox[3], ball_bbox[3]) - max(player_bbox[1], ball_bbox[1]))
        overlap_area = overlap_x * overlap_y
        
        # Calculate confidence
        confidence = 1.0 - (distance / max_distance) if distance <= max_distance else 0.0
        
        return {
            "is_valid": distance <= max_distance,
            "distance": distance,
            "overlap_area": overlap_area,
            "confidence": confidence,
            "player_center": player_center,
            "ball_center": ball_center
        }
    
    @staticmethod
    def get_assignment_statistics(
        team_assignments: Dict[int, int],
        possession_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get comprehensive assignment statistics.
        
        Args:
            team_assignments: Dictionary mapping player IDs to team IDs
            possession_history: List of possession records
            
        Returns:
            Dictionary containing assignment statistics
        """
        stats = {
            "total_players": len(team_assignments),
            "total_possessions": len(possession_history),
            "team_counts": {},
            "possession_counts": {},
            "avg_possession_distance": 0.0
        }
        
        # Count players per team
        for player_id, team_id in team_assignments.items():
            stats["team_counts"][team_id] = stats["team_counts"].get(team_id, 0) + 1
        
        # Count possessions per player
        for possession in possession_history:
            player_id = possession.get('player_id', -1)
            stats["possession_counts"][player_id] = stats["possession_counts"].get(player_id, 0) + 1
        
        # Calculate average possession distance
        distances = [possession.get('distance', 0.0) for possession in possession_history]
        if distances:
            stats["avg_possession_distance"] = np.mean(distances)
        
        return stats 