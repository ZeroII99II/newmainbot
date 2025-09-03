# obs_bronze.py — ~40D observation for Bronze
import numpy as np


def _v3(p): return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)

def build_obs(packet, index):
    me = packet.game_cars[index]; opp = packet.game_cars[1-index]; ball = packet.game_ball
    gi = packet.game_info
    me_p = _v3(me.physics.location); me_v = _v3(me.physics.velocity)
    opp_p = _v3(opp.physics.location); opp_v = _v3(opp.physics.velocity)
    b_p = _v3(ball.physics.location); b_v = _v3(ball.physics.velocity)

    team = me.team
    # normalize field to ~[-1,1] ranges
    SXY, SZ, SV = 4096.0, 2048.0, 2300.0
    def nxy(v): return np.clip(v[:2] / SXY, -1, 1)
    def nz(z):  return np.clip(np.array([z / SZ], np.float32), -1, 1)
    def nv(v):  return np.clip(v / SV, -1, 1)

    to_ball = b_p - me_p
    dist_b = np.linalg.norm(to_ball[:2]) / 6000.0
    ball_to_goal_dir = 1.0 if (team==0 and b_p[1] > 0) or (team==1 and b_p[1] < 0) else -1.0

    is_kickoff = float(getattr(gi, "is_kickoff_pause", False))
    boost = float(getattr(me, "boost", 33.0)) / 100.0

    # front-of-own-net danger zone
    own_goal_y = -5120.0 if team==0 else 5120.0
    if team==0: y_min, y_max = own_goal_y+300, own_goal_y+2000
    else:       y_min, y_max = own_goal_y-2000, own_goal_y-300
    dz = float(abs(b_p[0]) <= 1100 and (y_min <= b_p[1] <= y_max) and b_p[2] < 1100)

    obs = np.concatenate([
        nxy(me_p), nz(me_p[2]), nv(me_v),
        nxy(opp_p), nz(opp_p[2]), nv(opp_v),
        nxy(b_p), nz(b_p[2]), nv(b_v),
        np.array([dist_b, ball_to_goal_dir, is_kickoff, dz, boost], dtype=np.float32)
    ], dtype=np.float32)
    return obs  # shape ~ (41,)
