"""
Main application logic for Match Video Detection.

This module contains the core pipeline for analyzing football match videos,
including object detection, tracking, team assignment, and event detection.
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
from typing import Dict, Any

from src.tracking import Tracker

from src.assignment.team_assigner import TeamAssigner
from src.assignment.player_ball_assigner import PlayerBallAssigner
from src.assignment.camera_movement_estimator import CameraMovementEstimator
from src.assignment.speed_distance_estimator import SpeedAndDistanceEstimator

from src.events.entities import Player, Team, Ball
from src.events.pass_event import PassEvent
from src.events.possession_tracker import PossessionTracker
from src.events.cross_detector import CrossDetector

from src.utils.colors import *
from src.utils.video_utils import read_video, save_video

def main(
    video_path: str, 
    project_root: Path, 
    passes: bool = True, 
    possession: bool = True, 
    crosses: bool = False
) -> Dict[str, Any]:
    """Main function for video analysis pipeline.
    
    Args:
        video_path: Path to the input video file
        project_root: Root directory of the project
        passes: Whether to detect passes
        possession: Whether to track possession
        crosses: Whether to detect crosses
        
    Returns:
        Dictionary containing analysis results
    """
    # Read Video
    video_frames = read_video(video_path)
    
    # Get model path
    model_path = project_root / "models" / "yolo" / "best.pt"

    # Initialize tracker 
    tracker = Tracker(model_path=model_path)

    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames)
    
    # Get object positions
    tracker.add_position_to_tracks(tracks)
    
    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames)
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks, camera_movement_per_frame
    )

    # Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator 
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign player teams 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team

    # Force correct team assignment for known goalkeeper IDs
    # The player with the ID (285) is the goalkeeper of the Away team (Team ID 1), but
    # in the output video, the goalkeeper is assigned to the Home team (Team ID 2).
    # And the player with the ID (155) is the goalkeeper of the Home team (Team ID 2), but
    # in the output video, the goalkeeper is assigned to the Away team (Team ID 1).
    goalkeeper_team_map = {285: 2, 155: 1}
    for frame in tracks['players']:
        for player_id, pdata in frame.items():
            if player_id in goalkeeper_team_map:
                pdata['team'] = goalkeeper_team_map[player_id]

    # Assign ball acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # Assign ball to player if found
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)  # or another default value
    team_ball_control = np.array(team_ball_control)

    # --- Pass Detection ---
    
    # Build teams and player objects
    team_assignments = {}
    player_tracks = tracks['players']
    teams = {}
    
    # Create Team objects for each team id found
    for player_id, info in player_tracks[0].items():
        team_id = info.get('team')
        if team_id is not None and team_id not in teams:
            # Used the new Team class
            if team_id == 2:
                team_name = "Away"
                team_color = BLUE
            elif team_id == 1:
                team_name = "Home"
                team_color = RED
            # Create Team object
            teams[team_id] = Team(
                name=team_name, 
                abbreviation=f"T{team_id:02}", 
                color=team_color
            )
        if team_id is not None:
            team_assignments[player_id] = team_id
    team_list = list(teams.values())

    # --- Possession Tracker ---
    if not team_list:
        print("Warning: No teams found. Creating default teams.")
        # Create default teams if none were found
        teams[1] = Team(name="Home", abbreviation="T01", color=RED)
        teams[2] = Team(name="Away", abbreviation="T02", color=BLUE)
        team_list = list(teams.values())
    
    team_abbrs = [team.abbreviation for team in team_list]
    possession_tracker = PossessionTracker(team_ids=team_abbrs, fps=30)
    running_home_pct = []
    running_away_pct = []

    # Initialize pass event
    pass_event = PassEvent()
    
    # For each frame, update pass event with closest player to ball
    for frame_idx, player_dict in enumerate(player_tracks):
        # Find the player with the ball (has_ball==True)
        closest_player = None
        for player_id, pdata in player_dict.items():
            if pdata.get('has_ball', False):
                team_obj = teams.get(pdata.get('team'))
                # Used the new Player class
                closest_player = Player(player_id, pdata, team=team_obj)
                break
        
        # Ball bbox for this frame
        ball_bbox = None
        if 1 in tracks['ball'][frame_idx]:
            ball_bbox = tracks['ball'][frame_idx][1]['bbox']
        ball_obj = Ball(ball_bbox) if ball_bbox is not None else None
        
        # Update possession tracker only if both are not None
        if closest_player is not None and ball_obj is not None:
            possession_tracker.update(closest_player, ball_obj)
        
        # Store running possession percentages
        home_pct = possession_tracker.get_percentage_possession(
            team_list[0].abbreviation
        ) * 100
        away_pct = possession_tracker.get_percentage_possession(
            team_list[1].abbreviation
        ) * 100
        total = home_pct + away_pct
        if total > 0:
            home_display = home_pct / total * 100
            away_display = away_pct / total * 100
        else:
            home_display = away_display = 0
        running_home_pct.append(home_display)
        running_away_pct.append(away_display)
        
        # Pass event logic
        if closest_player and ball_bbox:
            pass_event.update(closest_player, ball_obj)
            pass_event.process_pass(team_list, frame_idx=frame_idx)

    # Gather all passes
    all_passes = []
    for team in team_list:
        for p in team.passes:
            all_passes.append((team.name, p.start_ball_bbox, p.end_ball_bbox))

    print("***********************************")
    print("Total passes detected:", len(all_passes))

    print(f"Home ({team_list[0].name}, {team_list[0].abbreviation}) passes: {len(team_list[0].passes)}")
    
    print(f"Away ({team_list[1].name}, {team_list[1].abbreviation}) passes: {len(team_list[1].passes)}")
    print("***********************************")

    # Print average/total possession per team
    percentages = possession_tracker.get_all_percentages()
    total_frames = possession_tracker.duration
    assigned_frames = sum(possession_tracker.team_possession_frames.values())
    no_possession_frames = total_frames - assigned_frames
    no_possession_pct = (no_possession_frames / total_frames) * 100 if total_frames > 0 else 0
    times = possession_tracker.get_all_times()
    print("Possession Percentages (Total/Average):")
    for team in team_list:
        pct = percentages[team.abbreviation] * 100
        print(f"{team.name} ({team.abbreviation}): {pct:.1f}%")
    if no_possession_pct > 0:
        print(f"No Possession: {no_possession_pct:.1f}%")
    print("***********************************")

    # --- Cross Detection ---
    if crosses:
        cross_detector = CrossDetector()
        # Get frame shape from the first frame
        frame_shape = video_frames[0].shape if video_frames else (1080, 1920, 3)
        # Convert all_passes to the format expected by detect_crosses
        pass_objects = []
        for team_name, start_bbox, end_bbox in all_passes:
            # Create a simple pass object with the required attributes
            class PassObject:
                def __init__(self, start_bbox, end_bbox):
                    self.start_ball_bbox = start_bbox
                    self.end_ball_bbox = end_bbox
            pass_objects.append(PassObject(start_bbox, end_bbox))
        
        cross_events = cross_detector.detect_crosses(pass_objects, frame_shape)
        print(f"Crosses detected: {len(cross_events)}")

    # --- Visualization ---
    # Create output video with annotations
    output_path = project_root / "outputs" / "videos" / f"analyzed_{Path(video_path).stem}.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create visualization frames
    vis_frames = []
    for frame_idx, frame in enumerate(video_frames):
        vis_frame = frame.copy()
        
        # Draw player tracks
        if frame_idx < len(tracks['players']):
            for player_id, player_data in tracks['players'][frame_idx].items():
                bbox = player_data['bbox']
                team_id = player_data.get('team', 0)
                has_ball = player_data.get('has_ball', False)
                
                # Choose color based on team
                if team_id == 1:
                    color = RED
                elif team_id == 2:
                    color = BLUE
                else:
                    color = WHITE
                
                # Draw player rectangle
                cv2.rectangle(vis_frame, 
                             (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), 
                             color, 2)
                
                # Draw player ID
                cv2.putText(vis_frame, str(player_id), 
                           (int(bbox[0]), int(bbox[1]-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Highlight player with ball
                if has_ball:
                    cv2.circle(vis_frame, 
                              (int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)), 
                              15, GREEN, -1)
        
        # Draw ball track
        if frame_idx < len(tracks['ball']) and 1 in tracks['ball'][frame_idx]:
            ball_bbox = tracks['ball'][frame_idx][1]['bbox']
            cv2.rectangle(vis_frame, 
                         (int(ball_bbox[0]), int(ball_bbox[1])), 
                         (int(ball_bbox[2]), int(ball_bbox[3])), 
                         YELLOW, 2)
        
        # Draw possession percentage
        if frame_idx < len(running_home_pct):
            cv2.putText(vis_frame, f"Home: {running_home_pct[frame_idx]:.1f}%", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)
            cv2.putText(vis_frame, f"Away: {running_away_pct[frame_idx]:.1f}%", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)
        
        vis_frames.append(vis_frame)
    
    # Save video
    save_video(vis_frames, str(output_path), fps=30)
    print(f"Analysis video saved to: {output_path}")
    
    return {
        'tracks': tracks,
        'passes': all_passes,
        'possession': {
            'home_pct': running_home_pct,
            'away_pct': running_away_pct,
            'final_home': running_home_pct[-1] if running_home_pct else 0,
            'final_away': running_away_pct[-1] if running_away_pct else 0
        },
        'output_video': str(output_path)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Match Analysis")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--passes", action="store_true", help="Enable pass detection")
    parser.add_argument("--possession", action="store_true", help="Enable possession tracking")
    parser.add_argument("--crosses", action="store_true", help="Enable cross detection")
    
    args = parser.parse_args()
    
    result = main(
        video_path=args.video_path,
        project_root=Path(args.project_root),
        passes=args.passes,
        possession=args.possession,
        crosses=args.crosses
    )
    
    print("Analysis completed successfully!") 