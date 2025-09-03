# aerial_play.py — pro 1s aerial brains: wall→air-dribble (offense), backboard defense (saves)
import math, numpy as np
from rlbot.agents.base_agent import SimpleControllerState

GRAV_Z = -650.0  # uu/s^2 Rocket League gravity (approx)

def _clip(x, lo=-1.0, hi=1.0): return float(max(lo, min(hi, x)))
def _v3(p): return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float32)

def predict_ball_pos(ball, t):
    p = _v3(ball.physics.location)
    v = _v3(ball.physics.velocity)
    a = np.array([0.0, 0.0, GRAV_Z], dtype=np.float32)
    return p + v * t + 0.5 * a * (t ** 2)

def backboard_intercept(ball, team, max_t=2.0):
    # Find where ball hits/approaches backboard plane; return (pos, t)
    y_board = -5120.0 if team == 0 else 5120.0
    # scan small steps for first crossing toward our backboard
    last_p = _v3(ball.physics.location); last_t = 0.0
    for i in range(1, 61):
        t = i * (max_t / 60.0)
        p = predict_ball_pos(ball, t)
        if (team == 0 and p[1] < y_board + 200.0) or (team == 1 and p[1] > y_board - 200.0):
            # clamp to backboard face and keep z within arena
            p[1] = y_board
            p[0] = float(max(-1800.0, min(1800.0, p[0])))
            p[2] = float(max(200.0, min(1800.0, p[2])))
            return p, t
        last_p = p; last_t = t
    return None, None

def steer_to(me, target_xy, yaw_gain=2.2, d_gain=0.7):
    yaw = float(me.physics.rotation.yaw)
    yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))
    ang = math.atan2(target_xy[1] - me.physics.location.y, target_xy[0] - me.physics.location.x) - yaw
    while ang > math.pi: ang -= 2*math.pi
    while ang < -math.pi: ang += 2*math.pi
    steer = _clip(yaw_gain * ang - d_gain * yaw_rate)
    return steer, abs(ang)

class AirDribbleBrain:
    """
    Minimal wall→air-dribble routine:
      1) Route to side wall carry.
      2) Pop ball up the wall, jump to chase.
      3) Gentle boost taps + small pitch to keep ball on nose; aim through to goal.
    """
    def __init__(self): self.state = "route"; self.t0 = 0.0

    def reset(self): self.state = "route"; self.t0 = 0.0

    def act(self, packet, index, intent=None):
        me = packet.game_cars[index]
        ball = packet.game_ball
        ctl = SimpleControllerState()
        # choose sidewall: closer x side
        side_x = 3000.0 if abs(me.physics.location.x) < abs(ball.physics.location.x) else (1500.0 if ball.physics.location.x >= 0 else -1500.0)
        target_xy = np.array([side_x, float(ball.physics.location.y)], dtype=np.float32)

        # Phase routing: go to sidewall line behind ball
        steer, ang_abs = steer_to(me, target_xy)
        ctl.steer = steer
        ctl.throttle = 1.0
        ctl.boost = 1.0 if ang_abs < 0.3 else 0.0

        # If near wall & close to ball horizontally, start lift
        dx = abs(me.physics.location.x - side_x)
        dist_xy = float(np.hypot(me.physics.location.x - ball.physics.location.x,
                                 me.physics.location.y - ball.physics.location.y))

        if dx < 450.0 and dist_xy < 800.0:
            # pop: quick jump if ball is slightly above hood and we are aligned
            if 60 < (ball.physics.location.z - me.physics.location.z) < 220 and ang_abs < 0.35:
                ctl.jump = True
            # chase with soft boost + slight pitch up
            ctl.boost = 1.0 if ang_abs < 0.25 else 0.0
            ctl.pitch = -0.15  # nose up slightly to keep ball on car in air
        return ctl

class BackboardDefenseBrain:
    """
    Backboard read & clear:
      - Predict intercept on own backboard plane.
      - Route to that point; when close, fast aerial & clear toward corner.
    """
    def __init__(self): self.active = False; self.t0 = 0.0

    def act(self, packet, index):
        me = packet.game_cars[index]; team = me.team; ball = packet.game_ball
        ctl = SimpleControllerState()
        P, t = backboard_intercept(ball, team, max_t=2.0)
        if P is None:
            # fallback: head to back-post spot
            bx = -900.0 if ball.physics.location.x < 0 else 900.0
            by = -5120.0 if team == 0 else 5120.0
            steer, ang_abs = steer_to(me, (bx, by))
            ctl.steer = steer; ctl.throttle = 1.0; ctl.boost = 1.0 if ang_abs < 0.25 else 0.0
            return ctl

        # ground routing
        steer, ang_abs = steer_to(me, (P[0], P[1]))
        ctl.steer = steer
        ctl.throttle = 1.0
        ctl.boost = 1.0 if ang_abs < 0.3 else 0.0

        # jump/fast aerial if we need height soon
        dz = P[2] - me.physics.location.z
        dist_xy = float(np.hypot(P[0] - me.physics.location.x, P[1] - me.physics.location.y))
        if dz > 200.0 and dist_xy < 1200.0:
            ctl.jump = True
            ctl.pitch = -0.8  # pitch up for aerial
            ctl.boost = 1.0

        # clear toward far corner (diagonal) on contact
        # renderer-free: bias yaw toward far-corner direction mildly
        far_corner_x = -3072.0 if P[0] > 0 else 3072.0
        # y stays same – we want off the backboard toward side
        # Add mild yaw bias
        desired = math.atan2(P[1] - me.physics.location.y, far_corner_x - me.physics.location.x)
        cur = me.physics.rotation.yaw
        dang = desired - cur
        while dang > math.pi: dang -= 2*math.pi
        while dang < -math.pi: dang += 2*math.pi
        ctl.yaw = _clip(0.3 * dang)
        return ctl
