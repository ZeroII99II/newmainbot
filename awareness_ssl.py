# awareness_ssl.py — Bronze-only awareness (no aerials / finisher / curriculum)
import numpy as np

def _v3(p): return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)

def compute_context(packet, index):
    me = packet.game_cars[index]
    team = me.team
    ball = packet.game_ball

    my_p = _v3(me.physics.location)
    b_p  = _v3(ball.physics.location)
    b_v  = _v3(ball.physics.velocity)

    # Kickoff detect
    gi = packet.game_info
    is_kickoff = bool(getattr(gi, "is_kickoff_pause", False))

    # Simple ETAs (very rough): distance / (car_speed+1)
    my_v = _v3(me.physics.velocity)
    car_speed = float(np.linalg.norm(my_v))
    eta_me_ball = float(np.linalg.norm(b_p[:2] - my_p[:2]) / (car_speed + 1.0))

    # Opp car (1v1)
    opp = packet.game_cars[1-index]
    opp_p = _v3(opp.physics.location)
    opp_v = _v3(opp.physics.velocity)
    opp_speed = float(np.linalg.norm(opp_v))
    eta_opp_ball = float(np.linalg.norm(b_p[:2] - opp_p[:2]) / (opp_speed + 1.0))

    in_opp_half = (b_p[1] > 0) if team == 0 else (b_p[1] < 0)

    # Pressure / threat indices, basic
    pressure_idx = float(in_opp_half) * 0.7 + float(eta_me_ball + 0.05 < eta_opp_ball) * 0.3
    pressure_idx = float(np.clip(pressure_idx, 0.0, 1.0))
    threat_idx   = float((1.0 - pressure_idx))

    # Possession proxy: we get there sooner and are within ~900uu
    possession_idx = float((eta_me_ball + 0.05 < eta_opp_ball) and (np.linalg.norm(b_p[:2]-my_p[:2]) < 900.0))

    # Back-post x for retreat
    back_post_x = -900.0 if b_p[0] > 0 else 900.0

    # Danger Zone rectangle in front of our own net
    own_goal_y = -5120.0 if team == 0 else 5120.0
    if team == 0:
        y_min, y_max = own_goal_y + 300.0, own_goal_y + 2000.0
    else:
        y_min, y_max = own_goal_y - 2000.0, own_goal_y - 300.0
    danger_zone = bool(abs(b_p[0]) <= 1100.0 and (y_min <= b_p[1] <= y_max) and b_p[2] < 1100.0)

    # Intent selection (Bronze-only set)
    # Prefer clear if in danger; otherwise shadow/press; kickoff uses SHOOT (front-flip)
    if is_kickoff:
        intent = "SHOOT"
    elif danger_zone:
        intent = "CLEAR_CORNER"
    elif pressure_idx < 0.35:
        intent = "SHADOW"
    elif possession_idx:
        intent = "CONTROL"
    else:
        intent = "CHALLENGE"

    return {
        "is_kickoff": is_kickoff,
        "pressure_idx": pressure_idx,
        "threat_idx": threat_idx,
        "possession_idx": possession_idx,
        "back_post_x": back_post_x,
        "danger_zone": danger_zone,
        "intent": intent,
        "eta_me_ball": eta_me_ball,
        "eta_opp_ball": eta_opp_ball,
    }
