import numpy as np
from .entities import Player, Team, Ball

# Pass and PassEvent logic (adapted, no visualization)
class Pass:
    def __init__(self, start_ball_bbox, end_ball_bbox, team, frame_idx):
        self.start_ball_bbox = start_ball_bbox
        self.end_ball_bbox = end_ball_bbox
        self.team = team
        self.frame_idx = frame_idx

class PassEvent:
    def __init__(self):
        self.ball = None
        self.closest_player = None
        self.init_player_with_ball = None
        self.last_player_with_ball = None
        self.player_with_ball_counter = 0
        self.player_with_ball_threshold = 3
        self.player_with_ball_threshold_dif_team = 4
    
    def update(self, closest_player, ball):
        self.ball = ball
        self.closest_player = closest_player
        same_id = Player.have_same_id(self.init_player_with_ball, closest_player)
        if same_id:
            self.player_with_ball_counter += 1
        else:
            self.player_with_ball_counter = 0
        self.init_player_with_ball = closest_player
    
    def validate_pass(self, start_player, end_player):
        if Player.have_same_id(start_player, end_player):
            return False
        if start_player.team != end_player.team:
            return False
        return True
    
    def generate_pass(self, team, start_pass, end_pass, frame_idx):
        return Pass(start_pass, end_pass, team, frame_idx)
    
    def process_pass(self, teams, frame_idx=None):
        if self.player_with_ball_counter >= self.player_with_ball_threshold:
            if self.last_player_with_ball is None:
                self.last_player_with_ball = self.init_player_with_ball
            valid_pass = self.validate_pass(self.last_player_with_ball, self.closest_player)
            if valid_pass and self.closest_player \
                and self.last_player_with_ball and self.ball:
                team = self.closest_player.team
                start_pass = self.last_player_with_ball.bbox
                end_pass = self.ball.bbox
                new_pass = self.generate_pass(team, start_pass, end_pass, frame_idx)
                for t in teams:
                    if t == team:
                        t.passes.append(new_pass)
            self.last_player_with_ball = self.closest_player

