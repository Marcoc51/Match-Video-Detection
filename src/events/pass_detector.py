import numpy as np
from src.utils.bbox_utils import get_center_of_bbox

class PassDetector:
    def __init__(self, player_tracks, ball_trajectory, \
        team_assignments=None, possession_threshold=50):
        # list of dicts: frame_num -> {player_id: {'bbox': ...}}
        self.player_tracks = player_tracks
        
        # list of (frame_idx, (x, y))
        self.ball_trajectory = ball_trajectory
        
        # dict: player_id -> team_id
        self.team_assignments = team_assignments
        
        # threshold for possession detection
        self.possession_threshold = possession_threshold

    def get_possession(self):
        possession = []
        for (frame_idx, ball_pos) in self.ball_trajectory:
            min_dist = float('inf')
            possessor = None
            if frame_idx < len(self.player_tracks):
                players = self.player_tracks[frame_idx]
                for player_id, player_info in players.items():
                    player_pos = get_center_of_bbox(player_info['bbox'])
                    dist = np.linalg.norm(np.array(ball_pos) - np.array(player_pos))
                    if dist < min_dist and dist < self.possession_threshold:
                        min_dist = dist
                        possessor = player_id
            possession.append((frame_idx, possessor))
        return possession

    def detect_passes(self):
        possession = self.get_possession()
        passes = []
        last_possessor = None
        for frame_idx, possessor in possession:
            if possessor is not None and possessor != last_possessor:
                if last_possessor is not None:
                    if (self.team_assignments is None or
                        self.team_assignments.get(possessor) == self.team_assignments.get(last_possessor)):
                        passes.append((frame_idx, last_possessor, possessor))
                last_possessor = possessor
        return passes 