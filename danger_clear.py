# danger_clear.py — when ball is in front of our net, send it safely to a corner
import math, numpy as np
from rlbot.agents.base_agent import SimpleControllerState

def _clip(x, lo=-1.0, hi=1.0): return float(max(lo, min(hi, x)))

def steer_to(me, tx, ty, yaw_gain=2.2, d_gain=0.7):
    yaw = float(me.physics.rotation.yaw)
    yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))
    ang = math.atan2(ty - me.physics.location.y, tx - me.physics.location.x) - yaw
    while ang > math.pi: ang -= 2*math.pi
    while ang < -math.pi: ang += 2*math.pi
    steer = _clip(yaw_gain * ang - d_gain * yaw_rate)
    return steer, abs(ang)

def own_corner_target(team, ball_x, own_goal_y):
    # pick side corner nearest ball x to minimize center spills
    cx = 3072.0 if ball_x >= 0 else -3072.0
    # keep y near our back line but slightly into field to avoid own post pinches
    cy = own_goal_y + (500.0 if team == 0 else -500.0)
    return cx, cy

class DangerClearBrain:
    """
    Routes car to hit ball toward the safe corner:
      - Aim at near-side own corner.
      - Use boost when aligned; jump tap when close & low ball to generate lift.
      - Never steer aim through center (reduces own-goal risks).
    """
    def act(self, packet, index):
        me = packet.game_cars[index]
        team = me.team
        ball = packet.game_ball

        own_goal_y = -5120.0 if team == 0 else 5120.0
        cx, cy = own_corner_target(team, ball.physics.location.x, own_goal_y)

        ctl = SimpleControllerState()
        steer, ang = steer_to(me, cx, cy)
        ctl.steer = steer
        ctl.throttle = 1.0
        ctl.boost = 1.0 if ang < 0.35 else 0.0

        # If ball is very low and close, pop it upward slightly to avoid center dribbles
        dist_xy = float(((me.physics.location.x - ball.physics.location.x)**2 + (me.physics.location.y - ball.physics.location.y)**2) ** 0.5)
        if ball.physics.location.z < 200.0 and dist_xy < 450.0 and ang < 0.25:
            ctl.jump = True
            ctl.pitch = -0.2  # mild upward pop toward corner

        # Mild yaw bias further toward the wall to guarantee side exit
        desired = math.atan2(cy - me.physics.location.y, cx - me.physics.location.x)
        cur = float(me.physics.rotation.yaw)
        dang = desired - cur
        while dang > math.pi: dang -= 2*math.pi
        while dang < -math.pi: dang += 2*math.pi
        ctl.yaw = _clip(0.2 * dang)
        return ctl
