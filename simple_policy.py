# simple_policy.py — baseline 1v1 controller: intercept, carry, flick
import numpy as np, math
from rlbot.agents.base_agent import SimpleControllerState


def _ang_norm(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _vec(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)


def _yaw(rot):
    return float(rot.yaw)


def _forward(rot):
    cy, sy = math.cos(rot.yaw), math.sin(rot.yaw)
    cp = math.cos(rot.pitch)
    return np.array([cp * cy, cp * sy], dtype=np.float32)


class HeuristicBrain:
    def __init__(self):
        self.kp = 2.2
        self.kd = 0.7
        self.handbrake_thresh = 1.2
        self.align_boost_thresh = 0.30
        self.flip_jump_speed = 1800.0
        self.flip_dist = 2200.0
        self._carry_ticks = 0
        self._last_jump_time = 0.0

    def _aim_point(self, ball_loc, team, approach_dist=140.0):
        goal_y = 5120.0 if team == 1 else -5120.0
        goal = np.array([0.0, goal_y], dtype=np.float32)
        b = _vec(ball_loc.x, ball_loc.y)
        d = goal - b
        n = np.linalg.norm(d) + 1e-6
        return b - d / n * approach_dist

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

        ball_pos = _vec(ball.physics.location.x, ball.physics.location.y)
        ball_vel = _vec(ball.physics.velocity.x, ball.physics.velocity.y)
        dist = float(np.linalg.norm(ball_pos - my_pos))

        # carry detection
        under_ball = (
            abs(ball.physics.location.x - me.physics.location.x) < 120
            and abs(ball.physics.location.y - me.physics.location.y) < 120
            and 60 < (ball.physics.location.z - me.physics.location.z) < 220
            and np.linalg.norm(ball_vel) < 1200
            and my_speed < 1700
        )
        self._carry_ticks = self._carry_ticks + 1 if under_ball else 0
        carrying = self._carry_ticks > 10

        # target point
        aim = self._aim_point(ball.physics.location, team, 140.0)
        target = 0.75 * aim + 0.25 * ball_pos

        # steering
        tgt_yaw = math.atan2(target[1] - my_pos[1], target[0] - my_pos[0])
        ang_err = _ang_norm(tgt_yaw - my_yaw)
        steer = np.clip(self.kp * ang_err - self.kd * yaw_rate, -1.0, 1.0)

        # throttle / boost / hb
        throttle = -0.5 if abs(ang_err) > 2.35 else 1.0
        handbrake = 1.0 if (abs(ang_err) > self.handbrake_thresh and dist < 1500.0) else 0.0
        boost = 1.0 if (abs(ang_err) < self.align_boost_thresh and my_speed < 2200.0) else 0.0

        # jump/flick: if carrying and roughly aligned, execute pop
        jump = 0.0
        if carrying and abs(ang_err) < 0.35:
            jump = 1.0

        # long flip when far & slow & aligned
        if (
            dist > self.flip_dist
            and my_speed < self.flip_jump_speed
            and abs(ang_err) < 0.20
            and not carrying
        ):
            jump = 1.0

        return np.array(
            [
                float(steer),
                float(throttle),
                0.0,
                0.0,
                0.0,
                float(jump),
                float(boost),
                float(handbrake),
            ],
            dtype=np.float32,
        )

