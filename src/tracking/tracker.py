from ultralytics import YOLO
import supervision as sv
import pickle
import os 
import cv2
import numpy as np
import pandas as pd
from src.utils.bbox_utils import get_bbox_width, get_center_of_bbox,get_foot_position

# Colors - RGB (Red, Green, Blue) --> BGR (Blue, Green, Red)
WHITE = (255,255,255)
BLACK = (0,0,0)
BLUE = (255,0,0)
RED = (0,0,255)
YELLOW = (0,255,255)
GREY = (128,128,128)
ORANGE = (0, 165, 255)
PINK = (255,0,255)

class Tracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])

        #interpolate missing values 
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks 

        detections = self.detect_frames(frames)

        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v.lower():k for k,v in cls_name.items()}

            # convert the detection to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            #Convert goolkeeper to player object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_name[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_name_inv['player']

            #track objects 
            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_name_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_name_inv['main referee'] \
                or cls_id == cls_name_inv['side referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_name_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _  = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center = (x_center, y2),
            axes = (int(width), int(0.35*width)),
            angle = 0.0,
            startAngle = -45,
            endAngle=235,
            color = color,
            thickness = 2,
            lineType = cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2)+15
        y2_rect = (y2 + rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rect),int(y2_rect)),
                color,
                cv2.FILLED
            )
            
            x1_text = x1_rect+12
            if track_id > 9 and track_id <= 99:
                x1_text -= 5
            elif track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        # Draw triangle
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        
        # Draw black border around the triangle
        cv2.drawContours(frame, [triangle_points], 0, BLACK, 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        
        # Draw white rectangle
        cv2.rectangle(overlay, (1350,850), (1900, 970), WHITE, -1 )

        # Add transparency to the rectangle
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        # Get the number of frames each team has the ball control
        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # Get the number of time each team has the ball control
        home_team_num_frames = \
            team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        away_team_num_frames = \
            team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        # Get the percentage of time each team has the ball control
        home_team = home_team_num_frames/(home_team_num_frames+away_team_num_frames)

        # Draw the percentage of time Home team has the ball control
        cv2.putText(
            frame, 
            f"Home Team Ball Control: {home_team*100:.2f}%", 
            (1350,900), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            RED, 
            2
        )

        return frame

    def draw_pass_arrow(self, frame, pos1, pos2):
        # Draw a blue arrow between two points
        cv2.arrowedLine(frame, pos1, pos2, BLUE, 2, tipLength=0.2)

    def draw_annotations(self, video_frames, tracks, team_ball_control, passes=None):
        self.passes = passes or []
        output_video_frames = []
        
        for frame_num, frame in enumerate(video_frames):
            # Get the player, referee, and ball dictionaries
            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                # Get only the Home team Players
                if player.get('team', 0) == 1:
                    # Get the color of the players based on their team id
                    ## Assign team color directly (if not already done in preprocessing)
                    color = player.get('team_color', RED)
                    frame = self.draw_ellipse(frame, player['bbox'], color, track_id)

                    # Draw the triangle if the player has the ball
                    if player.get('has_ball', False):
                        frame = self.draw_triangle(frame, player['bbox'], PINK)
                    
            # Draw Referee 
            for track_id, referee in referee_dict.items():
                # Draw the ellipse
                frame = self.draw_ellipse(frame, referee['bbox'], YELLOW)
            
            # Draw balls 
            for track_id, ball in ball_dict.items():
                # Draw the triangle
                frame = self.draw_triangle(frame, ball['bbox'], ORANGE)
            
            # Draw passes if available
            if hasattr(self, 'passes'):
                for frame_idx, from_id, to_id in self.passes:
                    # Draw the pass arrow if the frame index matches the current frame
                    if frame_idx == frame_num:
                        # Get the position of the from and to players
                        from_pos = player_dict.get(from_id, {}).get('position')
                        to_pos = player_dict.get(to_id, {}).get('position')

                        # Draw the pass arrow if the from and to positions are available
                        if from_pos and to_pos:
                            self.draw_pass_arrow(frame, from_pos, to_pos)

            # Draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # Add the frame to the output video frames
            output_video_frames.append(frame)
        
        return output_video_frames

    # Add position to tracks
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position