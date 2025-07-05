import numpy as np
import cv2
from pathlib import Path
from src.tracking import Tracker
from src.assignment.team_assigner import TeamAssigner
from src.assignment.player_ball_assigner import PlayerBallAssigner
from src.assignment.camera_movement_estimator import CameraMovementEstimator
from src.assignment.speed_distance_estimator import SpeedAndDistanceEstimator
from src.events.pass_detector import PassDetector
from src.utils.video_utils import read_video, save_video
from src.utils.bbox_utils import get_center_of_bbox

def main(video_path, project_root):
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

    # Assigne ball aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']

        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # --- Pass Detection ---
    # Extract team assignments from the first frame
    team_assignments = {}
    player_tracks = tracks['players']
    for player_id, info in player_tracks[0].items():
        if 'team' in info:
            team_assignments[player_id] = info['team']

    # Build ball_trajectory as list of (frame_idx, (x, y))
    ball_trajectory = []
    for frame_idx, ball_dict in enumerate(tracks['ball']):
        if 1 in ball_dict:
            bbox = ball_dict[1]['bbox']
            ball_pos = get_center_of_bbox(bbox)
            ball_trajectory.append((frame_idx, ball_pos))

    # Detect passes
    pass_detector = PassDetector(
        player_tracks=player_tracks,
        ball_trajectory=ball_trajectory,
        team_assignments=team_assignments
    )
    passes = pass_detector.detect_passes()

    # Filter passes: only between Home team players
    my_team_passes = [
        (f, f_id, t_id) for (f, f_id, t_id) in passes
        if team_assignments.get(f_id) == 1 and team_assignments.get(t_id) == 1
    ]

    print("***********************************")
    print("Your Team Pass Count:", len(my_team_passes))
    print("***********************************")
    
    print("***********************************")
    print("Detected passes (frame_idx, from_player, to_player):", passes)
    print(f"Total passes detected: {len(passes)}")
    print("***********************************")

    # Copy video frames
    output_video_frames = video_frames.copy()

    # --- End Pass Detection ---

    # Draw Output (original pipeline)
    ## Draw object tracks
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
    # for i, (frame_idx, from_player, to_player) in enumerate(passes):
    #     if frame_idx < len(tracks['players']):
    #         players = tracks['players'][frame_idx]
    #         if from_player in players and to_player in players:
    #             from_pos = get_center_of_bbox(players[from_player]['bbox'])
    #             to_pos = get_center_of_bbox(players[to_player]['bbox'])
    #             cv2.line(output_video_frames[frame_idx], from_pos, to_pos, (255,0,0), 3)
    
    # Save Video (with passes)
    output_path = project_root / "outputs" / "output_with_passes.avi"
    save_video(output_video_frames, output_path)
    
    print("***********************************")
    print("Visualization video saved as output_with_passes.avi")
    print("***********************************")
    
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[2]\
         / "Projects" / "Match-Video-Detection"
    video_path = project_root / "data" / "raw" / "match.mp4"
    main(video_path, project_root)