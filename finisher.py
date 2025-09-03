# finisher.py — choose and execute a finishing routine when EXPLOITing
import math, numpy as np
from rlbot.agents.base_agent import SimpleControllerState

def _clip(x, lo=-1.0, hi=1.0): return float(max(lo, min(hi, x)))

def _aim_steer(me, tx, ty, yaw_gain=2.2, d_gain=0.7):
    yaw = float(me.physics.rotation.yaw)
    yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))
    ang = math.atan2(ty - me.physics.location.y, tx - me.physics.location.x) - yaw
    while ang > math.pi: ang -= 2*math.pi
    while ang < -math.pi: ang += 2*math.pi
    steer = _clip(yaw_gain*ang - d_gain*yaw_rate)
    return steer, abs(ang)

def _goal_center(team):
    return (0.0, 5120.0 if team==0 else -5120.0)

class FinisherBrain:
    """
    Pick a finish type based on ball height, alignment, ETA edge, and variety:
      - POWER_SHOT (ground power through lane)
      - HOOK/45 FLICK (when carrying low)
      - LOW50 (under-ball vs diving opponent)
      - BUMP (clean demo line)
      - AIR_TAP (wall/air carry quick tap)
      - DOUBLE_TAP (backboard read)
    Returns a SimpleControllerState.
    """
    def __init__(self):
        self._last_choice = "POWER_SHOT"

    def choose(self, packet, index, ctx):
        me = packet.game_cars[index]; team = me.team; ball = packet.game_ball
        bx, by, bz = ball.physics.location.x, ball.physics.location.y, ball.physics.location.z
        meb = float(getattr(me, "boost", 33.0))
        eta_me = ctx.get("eta_me_ball", 0.6); eta_opp = ctx.get("eta_opp_ball", 0.8)
        edge = eta_opp - eta_me
        # Heuristics
        if bz < 200 and edge > 0.15:
            return "POWER_SHOT"
        if bz < 320 and edge > 0.1 and meb > 20 and ctx.get("possession_idx",0.0) > 0.5:
            return "HOOK_FLICK"
        if bz < 250 and edge > 0.05 and ctx.get("possession_idx",0.0) > 0.5:
            return "LOW50"
        if bz > 350 and abs(bx) > 1200 and meb > 40 and edge > 0.05:
            return "AIR_TAP"
        if bz > 700 and abs(bx) < 1600 and edge > 0.05:
            return "DOUBLE_TAP"
        # Rare bump if clean line
        if edge > 0.10 and meb > 30 and ctx.get("allow_demo", False):
            return "BUMP"
        return "POWER_SHOT"

    def act(self, packet, index, ctx):
        me = packet.game_cars[index]; team = me.team; ball = packet.game_ball
        choice = self.choose(packet, index, ctx)
        self._last_choice = choice
        ctl = SimpleControllerState()
        gx, gy = _goal_center(team)
        # Common steering toward a dynamic aim point (slightly inside far post)
        aim_x = float(np.clip(ball.physics.location.x * 0.7, -800, 800))
        aim_y = gy * 0.92
        steer, ang = _aim_steer(me, aim_x, aim_y)
        ctl.steer = steer; ctl.throttle = 1.0; ctl.boost = 1.0 if ang < 0.35 else 0.0

        bz = ball.physics.location.z
        # Routines
        if choice == "POWER_SHOT":
            # No jump unless close & centered; prefer boost-through
            if ang < 0.15 and abs(me.physics.location.x - ball.physics.location.x) < 250 and abs(me.physics.location.y - ball.physics.location.y) < 350:
                ctl.jump = True
        elif choice == "HOOK_FLICK":
            # Catch then pop: quick throttle cut, small jump + pitch forward
            ctl.throttle = 0.6; ctl.boost = 0.0; ctl.jump = True; ctl.pitch = 1.0
        elif choice == "LOW50":
            ctl.throttle = 0.8; ctl.boost = 0.2
            # soft jump to keep ball low on hood
            ctl.jump = True; ctl.pitch = 0.6
        elif choice == "AIR_TAP":
            ctl.jump = True; ctl.pitch = -0.6; ctl.boost = 1.0  # fast aerial pop and tap
        elif choice == "DOUBLE_TAP":
            # Aim for high backboard then follow
            aim_y = gy * 0.98
            steer, _ = _aim_steer(me, aim_x, aim_y)
            ctl.steer = steer; ctl.jump = True; ctl.pitch = -0.9; ctl.boost = 1.0
        elif choice == "BUMP":
            ctl.boost = 1.0; ctl.throttle = 1.0; ctl.jump = False
        return ctl, choice
