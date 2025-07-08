import numpy as np
from .entities import Player, Team, Ball

class PossessionTracker:
    def __init__(self, team_ids, fps=30, possession_threshold=20, ball_distance_threshold=45):
        """
        team_ids: list of team IDs (e.g., [1, 2])
        fps: frames per second
        possession_threshold: frames required to switch possession
        ball_distance_threshold: max distance to consider a player in possession
        """
        self.team_ids = team_ids
        self.fps = fps
        self.possession_threshold = possession_threshold
        self.ball_distance_threshold = ball_distance_threshold
        self.team_possession = team_ids[0]  # Start with first team
        self.current_team = team_ids[0]
        self.possession_counter = 0
        self.duration = 0
        self.team_possession_frames = {tid: 0 for tid in team_ids}

    def update(self, closest_player: Player, ball: Ball):
        """
        Call this for each frame with the closest Player and Ball objects.
        """
        if closest_player is None or ball is None or closest_player.team is None:
            self.duration += 1
            return

        # Calculate distance from player to ball (using bbox center and ball center)
        player_center = np.array([
            (closest_player.bbox[0] + closest_player.bbox[2]) / 2,
            (closest_player.bbox[1] + closest_player.bbox[3]) / 2
        ])
        ball_center = ball.center
        ball_distance = np.linalg.norm(player_center - ball_center)
        closest_player_team = closest_player.team

        if ball_distance > self.ball_distance_threshold:
            self.duration += 1
            return

        if closest_player_team != self.current_team:
            self.possession_counter = 0
            self.current_team = closest_player_team
        self.possession_counter += 1

        if self.possession_counter >= self.possession_threshold:
            self.team_possession = self.current_team

        # Increment possession for the team currently in possession
        if hasattr(self.team_possession, 'abbreviation'):
            team_id = self.team_possession.abbreviation if hasattr(self.team_possession, 'abbreviation') else self.team_possession
        else:
            team_id = self.team_possession
        if team_id in self.team_possession_frames:
            self.team_possession_frames[team_id] += 1
        self.duration += 1

    def get_percentage_possession(self, team_id):
        if self.duration == 0:
            return 0.0
        return round(self.team_possession_frames[team_id] / self.duration, 2)

    def get_time_possession(self, team_id):
        seconds = round(self.team_possession_frames[team_id] / self.fps)
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"

    def get_all_percentages(self):
        return {tid: self.get_percentage_possession(tid) for tid in self.team_ids}

    def get_all_times(self):
        return {tid: self.get_time_possession(tid) for tid in self.team_ids}

