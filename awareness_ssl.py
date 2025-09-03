import numpy as np, math
from ones_profile import ONES
from boost_pathing import nearest_small_pad_xy, football_lane
from aerial_play import backboard_intercept

SUP = 2300.0  # supersonic speed cap (uu/s) for quick ETAs

# Big boost pad rough world coords (Psyonix standard)
BIG_BOOSTS = [
    np.array([-3072.0, -4096.0, 0.0], dtype=np.float32),
    np.array([ 3072.0, -4096.0, 0.0], dtype=np.float32),
    np.array([-3072.0,  4096.0, 0.0], dtype=np.float32),
    np.array([ 3072.0,  4096.0, 0.0], dtype=np.float32),
]

def _v3(obj):
    return np.array([float(obj.x), float(obj.y), float(obj.z)], dtype=np.float32)

def _norm(v): 
    n = float(np.linalg.norm(v))
    return max(n, 1e-6)

def _eta(p_from, p_to, cur_speed=0.0):
    # Approx ETA: distance / max(supersonic, current+some accel)
    d = _norm(p_to - p_from)
    v = max(1200.0, min(SUP, cur_speed + 800.0))  # simple accel cushion
    return d / v

def _sign_to_opp(team):  # +Y toward orange when we are blue; -Y toward blue when we are orange
    return -1.0 if team == 0 else 1.0

def nearest_big_boost(my_pos, team):
    # prefer our half if close decision
    target = None; best = 1e9
    half_sign = -1.0 if team == 0 else 1.0
    for b in BIG_BOOSTS:
        if math.copysign(1.0, b[1]) == half_sign:  # our half first
            d = _norm(b - my_pos)
            if d < best:
                best, target = d, b
    if target is None:
        for b in BIG_BOOSTS:
            d = _norm(b - my_pos)
            if d < best:
                best, target = d, b
    return target

def compute_context(packet, index):
    """Returns a dict with awareness signals + recommended intent skeleton (string).
       Robust to missing fields; never raises."""
    ctx = {}
    try:
        me = packet.game_cars[index]
        opp = next(c for i,c in enumerate(packet.game_cars) if i != index and c.team != me.team and not c.is_demolished)
    except Exception:
        opp = None
    try:
        ball = packet.game_ball
        gi = packet.game_info
    except Exception:
        return ctx

    my_p = _v3(me.physics.location)
    my_v = _v3(me.physics.velocity); my_s = _norm(my_v)
    my_b = float(getattr(me, "boost", 33.0))
    team = me.team

    b_p = _v3(ball.physics.location)
    b_v = _v3(ball.physics.velocity)
    opp_p = _v3(opp.physics.location) if opp else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    opp_v = _v3(opp.physics.velocity) if opp else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    opp_s = _norm(opp_v)

    # ETAs to ball & to our back-post (rough)
    eta_me_ball = _eta(my_p, b_p, my_s)
    eta_opp_ball = _eta(opp_p, b_p, opp_s) if opp is not None else 9e9

    own_goal = np.array([0.0, -5120.0 if team == 0 else 5120.0, 0.0], dtype=np.float32)
    back_post = np.array([-900.0, own_goal[1], 0.0], dtype=np.float32) if b_p[0] < 0 else np.array([900.0, own_goal[1], 0.0], dtype=np.float32)
    eta_me_back = _eta(my_p, back_post, my_s)

    # Ball movement toward opponent / our goal (cosine)
    to_opp_sign = _sign_to_opp(team)
    b_dir = b_v[:2]; goal_vec_opp = np.array([0.0, 1.0 * to_opp_sign], dtype=np.float32)
    ball_dir_cos_opp = float(np.dot(b_dir[:2], goal_vec_opp) / (_norm(b_dir[:2]) * _norm(goal_vec_opp)))
    ball_dir_cos_opp = 0.0 if math.isnan(ball_dir_cos_opp) else max(-1.0, min(1.0, ball_dir_cos_opp))

    # Simple-half checks
    in_opp_half = (b_p[1] * to_opp_sign) > 0.0
    in_own_half = not in_opp_half

    # Possession proxy: closer ETA and ball slow
    possession = float(eta_me_ball < eta_opp_ball and _norm(b_v) < 1800.0)

    # Pressure & threat
    pressure_raw = 0.0
    if in_opp_half:
        pressure_raw = 0.5 + 0.5 * max(0.0, ball_dir_cos_opp)  # prefer moving toward their goal
        if eta_me_ball < eta_opp_ball + 0.2:
            pressure_raw = min(1.0, pressure_raw + 0.3)
    threat_raw = 0.0
    if in_own_half:
        toward_own = 1.0 if ball_dir_cos_opp < 0 else 0.0
        faster_opp = 1.0 if eta_opp_ball + 0.1 < eta_me_ball else 0.0
        threat_raw = max(toward_own, faster_opp)

    # Overcommit & recovery quality
    try:
        lm_break = 1.0 if ( (b_p[1] * to_opp_sign) < (my_p[1] * to_opp_sign) and my_b < 15.0 and threat_raw > 0.5 ) else 0.0
    except Exception:
        lm_break = 0.0
    recovery_ok = float(eta_me_back < max(0.8, 1.2 * eta_opp_ball) or pressure_raw > 0.7)

    # Risk budget: when behind late or with high boost, allow more risk
    time_left = max(0.0, 300.0 - float(getattr(gi, "seconds_elapsed", 0.0)) % 300.0)
    my_team_score = packet.teams[team].score
    their_score = packet.teams[1 - team].score
    losing = 1.0 if my_team_score + 0.5 < their_score else 0.0
    risk = float(0.2 + 0.5 * (my_b / 100.0) + 0.3 * losing + (0.2 if time_left < 60.0 else 0.0))
    risk = max(0.0, min(1.0, risk))

    # Recommend a high-level intent
    intent = "PRESS"
    if threat_raw > 0.6 and not recovery_ok:
        intent = "SHADOW"
    elif my_b < 20.0 and pressure_raw < 0.5 and eta_me_ball > 0.7:
        intent = "BOOST"
    elif possession:
        intent = "DRIBBLE"   # set up intentional touch/flick
    elif pressure_raw > 0.7 and eta_me_ball < eta_opp_ball:
        intent = "CHALLENGE"
    elif in_opp_half and ball_dir_cos_opp > 0.4:
        intent = "SHOOT"
    elif threat_raw > 0.3:
        intent = "CLEAR"
    # Allow riskier choices when risk high
    if risk > 0.7 and intent in ("DRIBBLE","PRESS") and in_opp_half:
        intent = "FAKE" if np.linalg.norm(b_v) < 900.0 else "CHALLENGE"

    # --- 1s specifics: boost delta, clock/score tempo, small-pad targets, demo line ---
    my_boost = float(getattr(me, "boost", 33.0))
    opp_boost = float(getattr(opp, "boost", 33.0)) if opp is not None else 33.0
    boost_delta = my_boost - opp_boost
    ctx["boost_delta"] = boost_delta

    # Small-pad target suggestion (prefer our half when defending)
    half_sign = -1.0 if team == 0 else 1.0
    pad_def = football_lane(my_p[0], my_p[1], to_defense=(threat_raw > 0.4))
    ctx["pad_target_xy"] = pad_def  # 2D

    # Tempo control: if leading late or with boost edge, bias to CONTROL/DRIBBLE
    time_left_approx = max(0.0, 300.0 - float(getattr(gi, "seconds_elapsed", 0.0)) % 300.0)
    leading = (packet.teams[team].score > packet.teams[1-team].score)
    if (leading and time_left_approx < ONES["tempo_slow_lead_secs"]) or (boost_delta > ONES["tempo_slow_boost_edge"] and in_opp_half):
        if intent in ("PRESS", "CHALLENGE", "SHOOT"):
            intent = "CONTROL"  # slow the game, force mistakes

    # Low-50 window: if we can get under-ball soon in front of opp
    under_lane = (abs(b_p[2] - my_p[2]) < 240 and np.linalg.norm(b_p[:2]-my_p[:2]) < ONES["low50_distance"])
    if possession and under_lane and eta_me_ball < eta_opp_ball:
        intent = "DRIBBLE"  # then heuristic can trigger pop/low-50

    # Back-post defense bias
    ctx["back_post_x"] = back_post[0]

    # Starve: if we have pressure + edge, allow STARVE (steal pads/boost in their half)
    if pressure_raw > 0.6 and boost_delta >= 10.0 and in_opp_half and ONES["starve_when_edge"]:
        intent = "STARVE"

    # Demo line: if supersonic towards opp and path free near ball
    ctx["allow_demo"] = bool( (eta_me_ball < 0.7) and (_norm(my_v) > ONES["demo_min_speed"]) )
    ctx["intent"] = intent

    ctx.update(dict(
        pressure_idx=float(max(0.0, min(1.0, pressure_raw))),
        threat_idx=float(max(0.0, min(1.0, threat_raw))),
        possession_idx=possession,
        recovery_ok=recovery_ok,
        overcommit_flag=lm_break,
        risk_budget=risk,
        eta_me_ball=eta_me_ball,
        eta_opp_ball=eta_opp_ball,
        eta_me_back=eta_me_back,
        intent=intent,
        nearest_big_boost=nearest_big_boost(my_p, team),
    ))
    # Aerial offense window: ball rising near sidewall & we can get there first
    sidewall_near = (abs(b_p[0]) > 1800.0 and b_p[2] > 200.0)
    aerial_window = sidewall_near and (eta_me_ball + 0.08 < eta_opp_ball)

    # Aerial threat against us: predicted backboard intercept exists soon
    bbP, bbT = backboard_intercept(ball, team, max_t=1.6)
    aerial_threat = (bbP is not None and (bbT or 10.0) < 1.2)

    # Intent nudges
    if aerial_threat and threat_raw > 0.3:
        intent = "BACKBOARD_DEFEND"
    elif aerial_window and pressure_raw > 0.5:
        intent = "AIR_DRIBBLE"
    # ----- Exploit window detection -----
    # Open net proxy: opponent behind ball (relative to their goal line) and far from center lane, or demoed/airborne with low boost.
    opp_bad = False
    if opp is not None:
        opp_loc = _v3(opp.physics.location)
        opp_air = (not getattr(opp, "has_wheel_contact", True)) and opp_loc[2] > 70
        opp_low_boost = float(getattr(opp, "boost", 33.0)) < 10.0
        # Opp behind play: compare y along goal direction
        to_opp_sign = _sign_to_opp(team)
        opp_behind = (opp_loc[1] * to_opp_sign) < (b_p[1] * to_opp_sign) - 600.0
        out_of_lane = abs(opp_loc[0]) > 1800.0
        opp_bad = (opp_air or opp_low_boost or (opp_behind and out_of_lane))

    # Ball scorable corridor: ball in opponent half, inside posts width, height manageable
    in_finishing_lane = ( (b_p[1] * _sign_to_opp(team)) > 700.0 and abs(b_p[0]) < 1800.0 and b_p[2] < 1200.0 )

    # Whiff/bad clear proxy: ball slow or bouncing upward near their box with opponent ETA worse than ours
    box_bounce = in_finishing_lane and (b_v[2] > 0 or np.linalg.norm(b_v) < 1200.0)
    eta_edge = eta_me_ball + 0.10 < eta_opp_ball

    exploit_window = bool((in_finishing_lane and (opp_bad or box_bounce) and eta_edge) or
                          (in_opp_half and eta_me_ball + 0.05 < eta_opp_ball and np.linalg.norm(b_v) < 1800.0))

    # If we see a juicy window and we're not in immediate danger, flip to EXPLOIT
    if exploit_window and threat_raw < 0.45:
        intent = "EXPLOIT"

    ctx.update(dict(
        exploit_window=exploit_window,
        in_finishing_lane=in_finishing_lane,
        opp_bad_state=opp_bad,
    ))
    ctx.update(dict(aerial_threat=aerial_threat, aerial_window=aerial_window))

    # ----- Danger Zone (front of own net) detection -----
    own_goal_y = -5120.0 if team == 0 else 5120.0

    # Front-of-net rectangle (the "slot"): centered X, between goal line and ~18 car lengths out
    # Box bounds: |x| <= 1100, y within [goal_y + sign*300, goal_y + sign*2000]
    if team == 0:
        y_min, y_max = own_goal_y + 300.0, own_goal_y + 2000.0   # -4820 .. -3120
    else:
        y_min, y_max = own_goal_y - 2000.0, own_goal_y - 300.0   # 3120 .. 4820

    in_x = abs(b_p[0]) <= 1100.0
    in_y = (y_min <= b_p[1] <= y_max)
    low_enough = b_p[2] < 1100.0

    danger_zone = bool(in_x and in_y and low_enough)
    ctx["danger_zone"] = danger_zone

    # If ball is in our slot, bias into a purposeful corner clear unless an aerial backboard save is mandated
    if danger_zone and threat_raw >= 0.2:
        # If there's a backboard intercept imminent, BACKBOARD_DEFEND takes priority (already handled above if present)
        if ctx.get("intent") not in ("BACKBOARD_DEFEND",):
            intent = "CLEAR_CORNER"

    ctx["intent"] = intent

    return ctx
