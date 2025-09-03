# bot.py — Destroyer Bronze-only version
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
import time, numpy as np

from awareness_ssl import compute_context
from decision_head import guard_by_intent
from drills_ssl import inject_kickoff_basic, inject_shadow_lane, inject_corner_push, inject_box_clear
from rewards_ssl import SSLReward
try:
    from overlay_pro import draw_overlay
    HAVE_HUD = True
except Exception:
    HAVE_HUD = False

from mechanics_ssl import SkillTelemetry
from simple_policy import HeuristicBrain

FAST_MODE = False

def to_controller(a: np.ndarray) -> SimpleControllerState:
    ctl = SimpleControllerState()
    steer, throttle, pitch, yaw, roll, jump, boost, handbrake = a.tolist()
    ctl.steer = float(np.clip(steer, -1.0, 1.0))
    ctl.throttle = float(np.clip(throttle, -1.0, 1.0))
    ctl.pitch = float(np.clip(pitch, -1.0, 1.0))
    ctl.yaw = float(np.clip(yaw, -1.0, 1.0))
    ctl.roll = float(np.clip(roll, -1.0, 1.0))
    ctl.jump = bool(jump > 0.5)
    ctl.boost = bool(boost > 0.5)
    ctl.handbrake = bool(handbrake > 0.5)
    return ctl

class Destroyer(BaseAgent):
    def initialize_agent(self):
        # Bronze mode hard-lock
        self._bronze_mode = True

        # State-setting probe flags
        self._state_setting_ok = None
        self._state_probe_t0 = 0.0
        self._state_probe_ball_z0 = None

        # Drill probabilities (Bronze only)
        self.drill_probs = {
            "core":  [("kickoff_basic", 0.30), ("shadow_lane", 0.30), ("pad_lane", 0.0), ("box_clear", 0.40)],
            "reads": [("box_clear", 0.60), ("shadow_lane", 0.40)],
            "aerial": []
        }
        self.curriculum_phase = "core"  # static

        self.rewarder = SSLReward()

        self.telemetry = SkillTelemetry()
        self.heur = HeuristicBrain()
        self.heur.agent = self
        self._last_ctx = {}

    def _probe_state_setting(self, packet):
        """
        One-time self-test: nudge ball Z by +50, then verify next tick.
        Sets self._state_setting_ok = True/False and stops probing.
        """
        try:
            gi = packet.game_info
            if gi is None or gi.is_kickoff_pause:
                return
            # Start probe
            if self._state_setting_ok is None and self._state_probe_t0 == 0.0:
                self._state_probe_ball_z0 = packet.game_ball.physics.location.z
                z = self._state_probe_ball_z0 + 50.0
                self.set_game_state(GameState(ball=BallState(physics=Physics(location=Vector3(0, 0, z)))))
                self._state_probe_t0 = gi.seconds_elapsed
                return
            # Verify after ~0.15s
            if self._state_setting_ok is None and self._state_probe_t0 > 0.0:
                if gi.seconds_elapsed - self._state_probe_t0 > 0.15:
                    z_now = packet.game_ball.physics.location.z
                    self._state_setting_ok = bool(abs(z_now - self._state_probe_ball_z0) > 20.0)
                    msg = "ENABLED" if self._state_setting_ok else "DISABLED"
                    print(f"[Destroyer] State setting probe: {msg}")
        except Exception:
            # If anything goes wrong, assume disabled to be safe
            self._state_setting_ok = False

    def _maybe_inject_ssl_drill(self, packet):
        import numpy as _np, random
        if not getattr(self, "_state_setting_ok", False):
            return
        if random.random() > 0.02:  # low frequency to not spam
            return
        plist = list(self.drill_probs.get(self.curriculum_phase, []))
        if not plist:
            return
        names, weights = zip(*plist)
        p = _np.array(weights, dtype=_np.float32)
        if p.sum() <= 0:
            return
        p = p / p.sum()
        choice = _np.random.choice(names, p=p)
        if choice == "kickoff_basic":
            inject_kickoff_basic(self)
        elif choice == "shadow_lane":
            inject_shadow_lane(self)
        elif choice == "corner_push":
            inject_corner_push(self)
        elif choice == "box_clear":
            inject_box_clear(self)

    def get_output(self, packet):
        self._probe_state_setting(packet)
        ctx = compute_context(packet, self.index)
        self._last_ctx = ctx

        info = self.telemetry.update(packet, self.index)
        # ensure keys used by rewards exist
        for k in [
            "kickoff_first_touch","perfect_touch","ball_progress","small_pad_pickup",
            "back_post_ok","corner_clear_success","own_slot_time","bad_center_touch",
            "wasted_boost","reverse_ticks"
        ]:
            info.setdefault(k, 0.0)

        if ctx.get("danger_zone", False):
            # Inline corner-clear: steer to near own corner and pop when close
            me = packet.game_cars[self.index]; ball = packet.game_ball
            own_goal_y = -5120.0 if me.team == 0 else 5120.0
            cx = 3072.0 if ball.physics.location.x >= 0 else -3072.0
            cy = own_goal_y + (500.0 if me.team == 0 else -500.0)

            def _steer_to(tx, ty):
                import math
                yaw = float(me.physics.rotation.yaw)
                yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))
                ang = math.atan2(ty - me.physics.location.y, tx - me.physics.location.x) - yaw
                while ang > math.pi:
                    ang -= 2*math.pi
                while ang < -math.pi:
                    ang += 2*math.pi
                steer = float(np.clip(2.2*ang - 0.7*yaw_rate, -1.0, 1.0))
                return steer, abs(ang)
            steer, ang = _steer_to(cx, cy)
            dist_xy = float(np.hypot(me.physics.location.x - ball.physics.location.x,
                                     me.physics.location.y - ball.physics.location.y))
            jump = 1.0 if (ball.physics.location.z < 200.0 and dist_xy < 450.0 and ang < 0.25) else 0.0
            action = np.array([steer, 1.0, -0.2 if jump else 0.0, 0.0, 0.0,
                               jump, 1.0 if ang < 0.35 else 0.0, 0.0], dtype=np.float32)
        else:
            action = self.heur.action(packet, self.index, intent=ctx["intent"])

        action = guard_by_intent(ctx["intent"], action, ctx)

        if HAVE_HUD and not FAST_MODE and self.renderer is not None:
            reasons = f"P:{float(ctx.get('pressure_idx',0.0)):.2f} T:{float(ctx.get('threat_idx',0.0)):.2f} DZ:{int(bool(ctx.get('danger_zone',False)))}"
            draw_overlay(
                self.renderer,
                bot_name="Destroyer",
                stage="Bronze • Bootcamp",
                progress=0.0,
                intent=ctx["intent"],
                reasons=reasons,
                last_attempt="",
                boost=float(getattr(packet.game_cars[self.index], "boost", 33.0)),
                action8=action,
                exploit=False, danger=bool(ctx.get("danger_zone", False))
            )

        self._maybe_inject_ssl_drill(packet)
        _ = self.rewarder(info)

        return to_controller(action)
