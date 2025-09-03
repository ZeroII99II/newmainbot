# simple_policy.py — baseline 1v1 controller: intercept ball + aim through it toward opponent goal.
import numpy as np, math
from rlbot.agents.base_agent import SimpleControllerState

def _ang_norm(a):
    # normalize to [-pi, pi]
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _vec(x, y): return np.array([float(x), float(y)], dtype=np.float32)


def _yaw(rot): return float(rot.yaw)

def _forward(rot):
    # 2D forward on XY plane
    cy, sy = math.cos(rot.yaw), math.sin(rot.yaw)
    cp = math.cos(rot.pitch)
    return np.array([cp * cy, cp * sy], dtype=np.float32)

def _speed_xy(v): return float(np.hypot(v.x, v.y))

def _sign_to_opp(team):  # +Y towards orange goal, -Y towards blue goal
    return -1.0 if team == 0 else 1.0

def _aim_point(ball_loc, ball_vel, team, approach_dist=120.0):
    # Place a point *behind* the ball along the line to opponent goal so we hit it toward their net.
    goal = np.array([0.0, 5120.0 * _sign_to_opp(1-team)], dtype=np.float32)  # opponent goal center
    b = _vec(ball_loc.x, ball_loc.y)
    dir_to_goal = goal - b
    n = np.linalg.norm(dir_to_goal) + 1e-6
    back = b - dir_to_goal / n * approach_dist
    return back

def _predict_ball(ball_loc, ball_vel, t):
    return _vec(ball_loc.x + ball_vel.x * t, ball_loc.y + ball_vel.y * t)

class HeuristicBrain:
    """
    Produces an 8-dim action vector [steer, throttle, pitch, yaw, roll, jump, boost, handbrake]
    which gets converted to SimpleControllerState by your to_controller().
    """
    def __init__(self):
        # Tunables
        self.kp = 2.2            # steering P gain
        self.kd = 0.7            # steering D gain on yaw rate
        self.handbrake_thresh = 1.2  # rad; use powerslide when angle is large & close
        self.align_boost_thresh = 0.30 # rad; boost when well aligned
        self.flip_jump_speed = 1800.0  # consider a forward flip when slow & far
        self.flip_dist = 2200.0

    def action(self, packet, index) -> np.ndarray:
        me = packet.game_cars[index]
        ball = packet.game_ball
        team = me.team

        my_pos = _vec(me.physics.location.x, me.physics.location.y)
        my_yaw = _yaw(me.physics.rotation)
        my_fwd = _forward(me.physics.rotation)
        my_vel = _vec(me.physics.velocity.x, me.physics.velocity.y)
        my_speed = float(np.linalg.norm(my_vel))
        yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))

        # Simple time-to-intercept guess and aim
        dist = float(np.linalg.norm(_vec(ball.physics.location.x, ball.physics.location.y) - my_pos))
        # Choose a lookahead proportional to distance but capped to keep it sane
        t_look = max(0.05, min(1.2, dist / max(1200.0, my_speed + 400.0)))
        target_ball = _predict_ball(ball.physics.location, ball.physics.velocity, t_look)
        aim = _aim_point(ball.physics.location, ball.physics.velocity, team, approach_dist=140.0)

        # Blend: mostly aim-behind-ball, but slightly toward predicted position for dynamic approach
        target = 0.75 * aim + 0.25 * target_ball

        to_target = target - my_pos
        target_yaw = math.atan2(to_target[1], to_target[0])
        ang_err = _ang_norm(target_yaw - my_yaw)  # desired steering angle
        steer = np.clip(self.kp * ang_err - self.kd * yaw_rate, -1.0, 1.0)

        # Throttle logic
        throttle = 1.0
        reverse = False
        if abs(ang_err) > 2.35:  # ~135 deg wrong way; back up instead of endless circles
            reverse = True
        if reverse:
            throttle = -0.5
        # Handbrake for tight turns when close AND big angle
        handbrake = 1.0 if (abs(ang_err) > self.handbrake_thresh and dist < 1500.0) else 0.0

        # Boost when aligned and not already fast
        boost = 1.0 if (abs(ang_err) < self.align_boost_thresh and my_speed < 2200.0) else 0.0

        # Simple jump / double jump for high balls in front
        jump = 0.0
        if ball.physics.location.z > 420 and abs(ang_err) < 0.35 and dist < 1000.0:
            jump = 1.0

        # Opportunistic forward flip when far and slow but aligned
        if dist > self.flip_dist and my_speed < self.flip_jump_speed and abs(ang_err) < 0.20:
            jump = 1.0  # one-tap; your speedflip code will handle KO; this is mid-field burst

        # Map to 8-dim action
        a = np.array([
            float(steer), float(throttle),
            0.0, 0.0, 0.0,               # pitch,yaw,roll (not used by ground controller)
            float(jump),
            float(boost),
            float(handbrake)
        ], dtype=np.float32)
        return a
