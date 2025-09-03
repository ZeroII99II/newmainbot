# awareness_bronze.py — tiny context for Bronze-only logic
import numpy as np

def _v3(p): return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)

def compute_context(packet, index):
    me   = packet.game_cars[index]
    team = me.team
    ball = packet.game_ball

    my_p = _v3(me.physics.location); my_v = _v3(me.physics.velocity)
    b_p  = _v3(ball.physics.location); b_v = _v3(ball.physics.velocity)

    gi = packet.game_info
    is_kickoff = bool(getattr(gi, "is_kickoff_pause", False))

    # crude ETA edge
    car_speed = float(np.linalg.norm(my_v)) + 1.0
    eta_me_ball = float(np.linalg.norm(b_p[:2] - my_p[:2]) / car_speed)

    opp = packet.game_cars[1 - index]
    opp_p = _v3(opp.physics.location); opp_v = _v3(opp.physics.velocity)
    opp_speed = float(np.linalg.norm(opp_v)) + 1.0
    eta_opp_ball = float(np.linalg.norm(b_p[:2] - opp_p[:2]) / opp_speed)

    in_opp_half = (b_p[1] > 0) if team == 0 else (b_p[1] < 0)
    pressure_idx = float(np.clip(0.7 * float(in_opp_half) + 0.3 * float(eta_me_ball + 0.05 < eta_opp_ball), 0, 1))
    threat_idx   = 1.0 - pressure_idx

    # possession if we’re close and earlier
    possession_idx = float( (eta_me_ball + 0.05 < eta_opp_ball) and (np.linalg.norm(b_p[:2]-my_p[:2]) < 900) )

    # front-of-own-net danger zone ("slot")
    own_goal_y = -5120.0 if team == 0 else 5120.0
    if team == 0:
        y_min, y_max = own_goal_y + 300.0, own_goal_y + 2000.0
    else:
        y_min, y_max = own_goal_y - 2000.0, own_goal_y - 300.0
    danger_zone = bool(abs(b_p[0]) <= 1100.0 and (y_min <= b_p[1] <= y_max) and b_p[2] < 1100.0)

    # intent (Bronze-only)
    if is_kickoff:
        intent = "SHOOT"               # front-flip kickoff
    elif danger_zone:
        intent = "CLEAR_CORNER"        # always to the side, not middle
    elif pressure_idx < 0.35:
        intent = "SHADOW"              # goal-side + patience
    elif possession_idx:
        intent = "CONTROL"             # low simple push/carry
    else:
        intent = "CHALLENGE"           # get in the play

    return dict(
        is_kickoff=is_kickoff,
        pressure_idx=pressure_idx, threat_idx=threat_idx,
        possession_idx=possession_idx,
        eta_me_ball=eta_me_ball, eta_opp_ball=eta_opp_ball,
        danger_zone=danger_zone,
        intent=intent
    )
