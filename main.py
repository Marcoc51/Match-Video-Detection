import numpy as np
import cv2
from pathlib import Path
import argparse

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
from src.utils.video_utils import read_video, save_video, draw_bezier_pass
from src.utils.bbox_utils import get_center_of_bbox

def main(video_path, project_root, passes=True, possession=True, crosses=False):
    # Read Video
    video_frames = read_video(video_path)
    
    # Get model path
    model_path = project_root / "models" / "yolo" / "best.pt"

    #initialize tracker 
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

    # # view transformer
    # view_transformer = ViewTransformer()
    # view_transformer.add_transformed_position_to_tracks(tracks)

    #interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and Distance Estimator 
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign player teams 
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
    
    for frame_num, player_tracks in enumerate(tracks['players']):
        for player_id, track in player_tracks.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team

    # Force correct team assignment for known goalkeeper IDs
    ### The player with the ID (285) is the goalkeeper of the Away team (Team ID 1), but
    ### in the output video, the goalkeeper is assigned to the Home team (Team ID 2).
    ### And the player with the ID (155) is the goalkeeper of the Home team (Team ID 2), but
    ### in the output video, the goalkeeper is assigned to the Away team (Team ID 1).
    goalkeeper_team_map = {285: 2, 155: 1}
    for frame in tracks['players']:
        for player_id, pdata in frame.items():
            if player_id in goalkeeper_team_map:
                pdata['team'] = goalkeeper_team_map[player_id]

    # Assigne ball aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        # Assign ball to player if found
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        # else:
        #     team_ball_control.append(team_ball_control[-1])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(0)  # or another default value
    team_ball_control = np.array(team_ball_control)

    # --- Pass Detection ---
    
    ## Build teams and player objects
    team_assignments = {}
    player_tracks = tracks['players']
    teams = {}
    
    ## Create Team objects for each team id found
    for player_id, info in player_tracks[0].items():
        team_id = info.get('team')
        if team_id is not None and team_id not in teams:
            ### Used the new Team class
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
    team_abbrs = [team.abbreviation for team in team_list]
    possession_tracker = PossessionTracker(team_ids=team_abbrs, fps=30)
    running_home_pct = []
    running_away_pct = []

    ## Initialize pass event
    pass_event = PassEvent()
    
    ## For each frame, update pass event with closest player to ball
    for frame_idx, player_dict in enumerate(player_tracks):
        # Find the player with the ball (has_ball==True)
        closest_player = None
        for player_id, pdata in player_dict.items():
            if pdata.get('has_ball', False):
                team_obj = teams.get(pdata.get('team'))
                ### Used the new Player class
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

    ## Gather all passes
    all_passes = []
    for team in team_list:
        for p in team.passes:
            all_passes.append((team.name, p.start_ball_bbox, p.end_ball_bbox))

    print("***********************************")
    print("Total passes detected:", len(all_passes))

    print(f"""Home ({team_list[0].name}, {team_list[0].abbreviation}) 
    passes: {len(team_list[0].passes)}""")
    
    print(f"""Away ({team_list[1].name}, {team_list[1].abbreviation}) 
    passes: {len(team_list[1].passes)}""")
    print("***********************************")

    # Print average/total possession per team
    percentages = possession_tracker.get_all_percentages()
    total_frames = possession_tracker.duration
    assigned_frames = sum(possession_tracker.team_possession_frames.values())
    no_possession_frames = total_frames - assigned_frames
    no_possession_pct = (no_possession_frames / total_frames) * 100 \
        if total_frames > 0 else 0
    times = possession_tracker.get_all_times()
    print("Possession Percentages (Total/Average):")
    for team in team_list:
        pct = percentages[team.abbreviation] * 100
        print(f"{team.name} ({team.abbreviation}): {pct:.1f}%")
    if no_possession_pct > 0:
        print(f"No Possession: {no_possession_pct:.1f}%")
    print("***********************************")
    print("Possession Time:")
    for team in team_list:
        print(f"{team.name} ({team.abbreviation}): {times[team.abbreviation]}")
    print("***********************************")

    # --- End Pass Detection ---

    # After pass detection and before drawing output
    # Gather all passes for cross detection
    all_pass_objs = []
    for team in team_list:
        for p in team.passes:
            all_pass_objs.append(p)
    # Detect crosses (for home team only, or all teams as needed)
    crosses = CrossDetector.detect_crosses(all_pass_objs, video_frames[0].shape)

    print("***********************************")
    print(f"Total crosses detected: {len(crosses)}")
    print("***********************************")

    # Draw Output

    ## Copy video frames
    output_video_frames = video_frames.copy()
    
    # Draw object tracks
    output_video_frames = tracker.draw_annotations(
        output_video_frames, tracks, team_ball_control
    )

    ## Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame
    )

    ## Draw speed and distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    ## Visualize passes by drawing lines between players
    for team in team_list:
        # Only draw passes for the Home team
        if team.abbreviation != "T01":
            continue
        
        for p in team.passes:
            # Get the start position of the pass
            start_pos = np.array([
                (p.start_ball_bbox[0]+p.start_ball_bbox[2])/2, 
                (p.start_ball_bbox[1]+p.start_ball_bbox[3])/2
            ]).astype(int)
            
            # Get the end position of the pass
            end_pos = np.array([
                (p.end_ball_bbox[0]+p.end_ball_bbox[2])/2, 
                (p.end_ball_bbox[1]+p.end_ball_bbox[3])/2
            ]).astype(int)

            # Get the color of the pass
            color = tuple(int(x) for x in team.color) \
                if hasattr(team, 'color') else (255,0,0)
            # Draw the pass as a Bézier curve on all frames from its frame_idx to the end
            for frame in output_video_frames[p.frame_idx:]:
                draw_bezier_pass(
                    frame, 
                    tuple(start_pos), 
                    tuple(end_pos), 
                    color, 
                    2, 
                    curve_height=60
                )
            
            # Before saving the video, after drawing passes
            home_passes = [p for p in team.passes]
            pass_counts_per_frame = [0] * len(output_video_frames)
            for p in home_passes:
                for i in range(p.frame_idx, len(output_video_frames)):
                    pass_counts_per_frame[i] += 1

            for i, frame in enumerate(output_video_frames):
                # Copy frame
                frame_copy= frame.copy()
                overlay = frame.copy()
                
                # Draw white rectangle
                cv2.rectangle(overlay, (0, 100), (500, 220), WHITE, -1 )

                # Add transparency to the rectangle
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, frame_copy, 1-alpha, 0, frame_copy)

                # Draw text in the rectangle
                cv2.putText(
                    frame_copy,
                    f"Home Team Passes: {pass_counts_per_frame[i]}",
                    (10, 130),  # position
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    RED,  # color (red)
                    2  # thickness
                )

                # Replace the frame with the overlay
                output_video_frames[i] = frame_copy

    # Draw passes as Bézier curves if enabled
    if passes:
        for team in team_list:
            if team.abbreviation != "T01":
                continue
            for p in team.passes:
                start_pos = np.array([
                    (p.start_ball_bbox[0]+p.start_ball_bbox[2])/2, 
                    (p.start_ball_bbox[1]+p.start_ball_bbox[3])/2
                ]).astype(int)
                end_pos = np.array([
                    (p.end_ball_bbox[0]+p.end_ball_bbox[2])/2, 
                    (p.end_ball_bbox[1]+p.end_ball_bbox[3])/2
                ]).astype(int)
                color = tuple(int(x) for x in team.color) \
                    if hasattr(team, 'color') else (255,0,0)
                for frame in output_video_frames[p.frame_idx:]:
                    draw_bezier_pass(
                        frame, 
                        tuple(start_pos), 
                        tuple(end_pos), 
                        color, 
                        2, 
                        curve_height=60
                    )
    # Draw crosses if enabled
    if crosses:
        for p in crosses:
            start_pos = np.array([
                (p.start_ball_bbox[0]+p.start_ball_bbox[2])/2, 
                (p.start_ball_bbox[1]+p.start_ball_bbox[3])/2
            ]).astype(int)
            end_pos = np.array([
                (p.end_ball_bbox[0]+p.end_ball_bbox[2])/2, 
                (p.end_ball_bbox[1]+p.end_ball_bbox[3])/2
            ]).astype(int)
            # Use a distinct color for crosses (e.g., green)
            color = (0, 255, 0)
            for frame in output_video_frames[p.frame_idx:]:
                draw_bezier_pass(
                    frame, 
                    tuple(start_pos), 
                    tuple(end_pos), 
                    color, 
                    2, 
                    curve_height=80
                )
    # Draw overlays for possession and passes if enabled
    for i, frame in enumerate(output_video_frames):
        if passes:
            # ... draw pass overlays ...
            pass
        if possession:
            cv2.putText(
                frame, 
                f"Home Possession: {running_home_pct[i]:.1f}%", 
                (10, 170), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                RED, 
                2
            )
            cv2.putText(
                frame, 
                f"Away Possession: {running_away_pct[i]:.1f}%", 
                (10, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                BLUE, 
                2
            )
    
    # Save Video
    output_path = project_root / "outputs" / "output_with_new_crosses.avi"
    save_video(output_video_frames, output_path)
    
    print("***********************************")
    print("Visualization video saved as output_with_new_crosses.avi")
    print("***********************************")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--passes', 
        action='store_true', 
        help='Visualize passes'
    )
    parser.add_argument(
        '--possession', 
        action='store_true', 
        help='Visualize possession'
    )
    parser.add_argument(
        '--crosses', 
        action='store_true', 
        help='Visualize crosses'
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2] \
        / "Projects" / "Match-Video-Detection"
    video_path = project_root / "data" / "raw" / "cross-2.mp4"
    main(
        video_path, 
        project_root, 
        show_passes=args.passes, 
        show_possession=args.possession, 
        show_crosses=args.crosses
    )