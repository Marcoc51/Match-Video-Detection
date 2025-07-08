import numpy as np

class Player:
    def __init__(self, player_id, player_dict, team=None):
        self.id = player_id
        self.bbox = player_dict['bbox']
        self.team = team
    
    def __eq__(self, other):
        return isinstance(other, Player) and self.id == other.id
    
    @staticmethod
    def have_same_id(player1, player2):
        if player1 is None or player2 is None:
            return False
        return player1.id == player2.id

class Team:
    def __init__(self, name, color=(0,0,0), abbreviation="NNN"):
        self.name = name
        self.color = color
        self.abbreviation = abbreviation
        self.passes = []
    
    def __eq__(self, other):
        return isinstance(other, Team) and self.name == other.name

class Ball:
    def __init__(self, bbox):
        self.bbox = bbox
        self.center = self.get_center(bbox)
    
    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

