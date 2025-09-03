# recovery.py — recovery controller (air-roll to wheels, wave dash, half-flip)
import math, numpy as np
from rlbot.agents.base_agent import SimpleControllerState


def _ang_norm(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


def _speed_xy(v): return float(math.hypot(v.x, v.y))


def _face_target_yaw(me, x, y):
    yaw = float(me.physics.rotation.yaw)
    ang = _ang_norm(math.atan2(y - me.physics.location.y, x - me.physics.location.x) - yaw)
    return float(np.clip(2.2*ang - 0.7*getattr(me.physics.angular_velocity, "z", 0.0), -1.0, 1.0))


class RecoveryBrain:
    """
    Returns (active: bool, action: np.ndarray[8]) when recovery logic should override.
    Implements:
      - air-roll to wheels & nose-down landings (ground / wall)
      - wave dash on landing window for speed
      - half-flip when facing away and need to reverse direction quickly
    """
    def __init__(self):
        self._land_t0 = 0.0
        self._do_wavedash = False
        self._hf_state = "idle"
        self._hf_t0 = 0.0

    def _is_airborne(self, me):
        try:
            return (not me.has_wheel_contact) and me.physics.location.z > 70
        except Exception:
            return me.physics.location.z > 70

    def _near_ground(self, me):
        return me.physics.location.z < 100

    def _half_flip_needed(self, me, ball):
        # Need to turn around fast: ball roughly behind us and our forward speed small
        my = me.physics
        yaw = my.rotation.yaw
        dx = ball.physics.location.x - my.location.x
        dy = ball.physics.location.y - my.location.y
        ang = abs(_ang_norm(math.atan2(dy, dx) - yaw))
        vxy = _speed_xy(my.velocity)
        return ang > 2.2 and vxy < 700  # ~ >126° and we're slow

    def _run_half_flip(self, t):
        """
        Simple half-flip macro timeline (seconds since start):
          0.00-0.08: first jump, pitch back
          0.14-0.22: second jump to cancel, air-roll to flatten
          0.28-0.60: pitch forward to drive away
        Output: 8-dim action
        """
        a = np.zeros(8, dtype=np.float32)
        # steer, throttle, pitch, yaw, roll, jump, boost, handbrake
        a[1] = 1.0
        if 0.00 <= t < 0.08:
            a[5] = 1.0; a[2] = -1.0
        elif 0.14 <= t < 0.22:
            a[5] = 1.0; a[4] = 1.0  # air-roll right to flatten
        elif 0.22 <= t < 0.32:
            a[2] = 1.0
        return a

    def act(self, packet, index, intent=None):
        """
        Return (active, action8). If active is True, caller should use this action.
        """
        me = packet.game_cars[index]
        ball = packet.game_ball
        gi = packet.game_info
        now = float(getattr(gi, "seconds_elapsed", 0.0))

        # --- Half-flip finite state machine
        if self._hf_state == "idle":
            if self._half_flip_needed(me, ball):
                self._hf_state = "do"
                self._hf_t0 = now
        if self._hf_state == "do":
            t = now - self._hf_t0
            a = self._run_half_flip(t)
            if t > 0.60:
                self._hf_state = "idle"
            return True, a

        # --- Airborne recovery
        airborne = self._is_airborne(me)
        if airborne:
            a = np.zeros(8, dtype=np.float32)
            a[6] = 0.0  # no boost while stabilizing
            # Try to orient nose-down (pitch forward) and roll to wheels
            a[2] = 0.6   # pitch forward
            a[4] = 0.4 if me.physics.rotation.roll < 0 else -0.4  # roll toward upright
            # slight yaw toward ball to be pre-aligned on landing
            steer = _face_target_yaw(me, ball.physics.location.x, ball.physics.location.y)
            a[0] = float(np.clip(0.5*steer, -1.0, 1.0))
            # Set up wave-dash flag when near ground & descending
            if self._near_ground(me) and me.physics.velocity.z < -200:
                self._land_t0 = now
                self._do_wavedash = True
            return True, a

        # --- On/near ground: wave dash window
        if self._do_wavedash:
            t = now - self._land_t0
            a = np.zeros(8, dtype=np.float32)
            a[1] = 1.0
            if 0.04 < t < 0.12:       # quick jump
                a[5] = 1.0
            elif 0.12 <= t < 0.24:    # pitch forward to dash
                a[2] = 1.0
            else:
                self._do_wavedash = False
            return True, a

        # No special recovery needed
        return False, np.zeros(8, dtype=np.float32)
