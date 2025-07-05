from detection.yolo import detect_objects_in_frame
from tracking.tracker import Tracker
from events.cross_detector import CrossDetector
from events.pass_detector import PassDetector
from utils.video_utils import read_video, save_video
from utils.bbox_utils import get_center_of_bbox
import cv2
from pathlib import Path

def main(video_path):
    frames = read_video(video_path)
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "Match-Video-Detection" / "models" / "yolo" / "best.pt"
    tracker = Tracker(model_path=model_path)

    tracks = tracker.get_object_tracks(frames)
    ball_trajectory = []

    for frame_idx, ball_dict in enumerate(tracks['ball']):
        if 1 in ball_dict:
            bbox = ball_dict[1]['bbox']
            ball_pos = get_center_of_bbox(bbox)
            ball_trajectory.append((frame_idx, ball_pos))

    # Field and penalty area dimensions (example values, adjust as needed)
    field_width, field_height = frames[0].shape[1], frames[0].shape[0]
    penalty_area_coords = (
        (field_width//4, field_height//3), (3*field_width//4, 2*field_height//3)
    )

    # cross_detector = CrossDetector(field_width, field_height, penalty_area_coords)
    # crosses = cross_detector.detect_crosses(ball_trajectory)
    
    # print("***********************************")
    # print("Detected crosses (start_frame, end_frame):", crosses)
    # print(f"Total crosses detected: {len(crosses)}")
    # print("***********************************")

    # # Visualization: highlight ball during crosses
    # output_frames = frames.copy()
    # cross_frames = set()
    # for start, end in crosses:
    #     for idx in range(start, end+1):
    #         cross_frames.add(idx)
    # for idx, (frame_idx, ball_pos) in enumerate(ball_trajectory):
    #     if frame_idx in cross_frames:
    #         cv2.circle(output_frames[frame_idx], ball_pos, 15, (0,0,255), 3)
    
    # project_root = Path(__file__).resolve().parents[2]
    # output_path = project_root / "Match-Video-Detection" / \
    #               "outputs" / "output_with_crosses.avi"
    # save_video(output_frames, output_path)
    
    # print("***********************************")
    # print("Visualization video saved as output_with_crosses.avi")
    # print("***********************************")

    # player_tracks: list of dicts, e.g. tracks['players']
    # ball_trajectory: list of (frame_idx, (x, y))

    # Extract team assignments from a single frame
    team_assignments = {}
    player_tracks=tracks['players']

    for player_id, info in player_tracks[0].items():
        if 'team_id' in info:
            team_assignments[player_id] = info['team_id']

    # Use in pass detector
    pass_detector = PassDetector(
        player_tracks=player_tracks,
        ball_trajectory=ball_trajectory,
        team_assignments=team_assignments
    )
    passes = pass_detector.detect_passes()

    # Filter passes: only between team 1 players
    my_team_passes = [
        (f, f_id, t_id) for (f, f_id, t_id) in passes
        if team_assignments.get(f_id) == 1 and team_assignments.get(t_id) == 1
    ]

    print("***********************************")
    print("Your Team Pass Count:", len(my_team_passes))
    print("***********************************")

    # pass_detector = PassDetector(
    #     player_tracks=tracks['players'], ball_trajectory=ball_trajectory
    # )
    # passes = pass_detector.detect_passes()
    
    print("***********************************")
    print("Detected passes (frame_idx, from_player, to_player):", passes)
    print(f"Total passes detected: {len(passes)}")
    print("***********************************")

    # Visualization: highlight passes
    output_frames = frames.copy()
    for i, (frame_idx, from_player, to_player) in enumerate(passes):
        if frame_idx < len(tracks['players']):
            players = tracks['players'][frame_idx]
            if from_player in players and to_player in players:
                from_pos = get_center_of_bbox(players[from_player]['bbox'])
                to_pos = get_center_of_bbox(players[to_player]['bbox'])
                cv2.line(output_frames[frame_idx], from_pos, to_pos, (255,0,0), 3)
    
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "Match-Video-Detection" / \
                  "outputs" / "output_with_passes.avi"
    save_video(output_frames, output_path)
    
    print("***********************************")
    print("Visualization video saved as output_with_passes.avi")
    print("***********************************")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / "Match-Video-Detection" / "data" / "raw" / "match.mp4"
    main(video_path)
