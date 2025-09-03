# rewards_bronze.py — immediate reward for Bronze training
import numpy as np

class BronzeReward:
    def __init__(self, team):
        self.team = int(team)
        self._last_ball_y = None
        self._last_blue = 0
        self._last_orange = 0

    def update_score(self, packet):
        blue = packet.teams[0].score
        orange = packet.teams[1].score
        d_blue = blue - self._last_blue
        d_orange = orange - self._last_orange
        self._last_blue, self._last_orange = blue, orange
        # +1 for our goal, -1 for theirs
        if self.team == 0: return (d_blue - d_orange)
        else:              return (d_orange - d_blue)

    def __call__(self, packet):
        ball_y = float(packet.game_ball.physics.location.y)
        # progress: ball toward their goal line
        dir = 1.0 if self.team==0 else -1.0
        dy = 0.0 if self._last_ball_y is None else (ball_y - self._last_ball_y) * dir
        self._last_ball_y = ball_y
        r_prog = 0.001 * np.clip(dy, -200.0, 200.0)  # gentle

        # danger zone penalty
        b = packet.game_ball.physics.location
        in_slot = (abs(b.x) <= 1100 and ((self.team==0 and -4820 <= b.y <= -3120) or (self.team==1 and 3120 <= b.y <= 4820)) and b.z < 1100)
        r_slot = -0.01 if in_slot else 0.0

        # score delta
        r_goal = self.update_score(packet)

        return float(r_prog + r_slot + r_goal)
