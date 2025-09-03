# bot.py — Destroyer (Bronze-only)
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import GameState, BallState, Physics, Vector3
import numpy as np, math, random, time, configparser, os

from awareness_bronze import compute_context
from decision_bronze import guard_by_intent
from drills_bronze import inject_kickoff_basic, inject_shadow_lane, inject_corner_push, inject_box_clear
from obs_bronze import build_obs
from rewards_bronze import BronzeReward
from trainer import OnlineTrainer  # resolves to NoopTrainer if torch missing

try:
    from overlay_bronze import draw_overlay
    HAVE_HUD = True
except Exception:
    HAVE_HUD = False

FAST_MODE = False  # set True to disable HUD

def _clip(x, lo=-1.0, hi=1.0): return float(max(lo, min(hi, x)))

def _steer_to(me, tx, ty, yaw_gain=2.2, d_gain=0.7):
    yaw = float(me.physics.rotation.yaw)
    yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))
    ang = math.atan2(ty - me.physics.location.y, tx - me.physics.location.x) - yaw
    while ang > math.pi: ang -= 2*math.pi
    while ang < -math.pi: ang += 2*math.pi
    steer = _clip(yaw_gain * ang - d_gain * yaw_rate)
    return steer, abs(ang)

def _goal_y(team): return 5120.0 if team==0 else -5120.0

class Destroyer(BaseAgent):
    def initialize_agent(self):
        self._bronze_mode = True
        self._state_setting_ok = None
        self._probe_t = 0.0
        self._last_ball_z = None

        # Drill weights (low frequency)
        self.drill_probs = [("kickoff",0.30),("shadow",0.30),("corner",0.20),("box",0.20)]
        print("[Destroyer] Bronze-only initialized.")

        # --- Training config & trainer ---
        cfg = configparser.ConfigParser()
        cfg.read("train.cfg")
        tc = dict(cfg.items("Training")) if cfg.has_section("Training") else {}
        self._train_enabled = (tc.get("enabled","true").lower() == "true")
        self._use_model_for_action = (tc.get("use_model_for_action","false").lower() == "true")
        self._stage_tag = tc.get("stage_tag","Bronze")

        # Build a fake obs once to get dim
        try:
            dummy_obs = build_obs(self.get_game_tick_packet(), self.index)
            self._obs_dim = int(dummy_obs.shape[0])
        except Exception:
            self._obs_dim = 41

        self.trainer = OnlineTrainer(obs_dim=self._obs_dim, **{
            "enabled": self._train_enabled,
            "use_model_for_action": self._use_model_for_action,
            "model_dir": tc.get("model_dir","models"),
            "checkpoint_dir": tc.get("checkpoint_dir","checkpoints"),
            "stage_tag": self._stage_tag,
            "save_every_seconds": int(tc.get("save_every_seconds","300")),
            "save_every_steps": int(tc.get("save_every_steps","2000")),
            "learning_rate": float(tc.get("learning_rate","1e-4")),
            "batch_size": int(tc.get("batch_size","256")),
            "replay_capacity": int(tc.get("replay_capacity","50000")),
            "min_replay": int(tc.get("min_replay","2000")),
            "gamma": float(tc.get("gamma","0.995")),
        })

        self._reward = BronzeReward(self.team)
        self._prev_obs = None
        self._steps = 0

    # Probe whether set_game_state works and set flag once
    def _probe_state_setting(self, packet):
        try:
            if self._state_setting_ok is not None:
                return
            t = time.time()
            if self._probe_t == 0.0:
                # nudge ball z slightly
                b = packet.game_ball
                self._last_ball_z = float(b.physics.location.z)
                self.set_game_state(GameState(ball=BallState(physics=Physics(location=Vector3(0,0,self._last_ball_z+1.0)))))
                self._probe_t = t
            elif t - self._probe_t > 0.3:
                z = float(packet.game_ball.physics.location.z)
                self._state_setting_ok = abs(z - self._last_ball_z) > 0.5
                print(f"[Destroyer] State setting OK: {self._state_setting_ok}")
        except Exception:
            self._state_setting_ok = False

    def _maybe_inject(self, packet):
        if not self._state_setting_ok: return
        if random.random() > 0.02: return  # ~2% of ticks
        choice = random.choices([n for n,_ in self.drill_probs], weights=[w for _,w in self.drill_probs])[0]
        if choice == "kickoff": inject_kickoff_basic(self)
        elif choice == "shadow": inject_shadow_lane(self)
        elif choice == "corner": inject_corner_push(self)
        elif choice == "box":    inject_box_clear(self)

    def get_output(self, packet):
        self._probe_state_setting(packet)

        obs = build_obs(packet, self.index)
        model_action = self.trainer.act(obs)  # None if disabled or using heuristic

        me   = packet.game_cars[self.index]
        ball = packet.game_ball
        team = me.team

        ctx = compute_context(packet, self.index)

        # === Choose simple heuristic action ===
        ctl = SimpleControllerState()
        action = np.zeros(8, dtype=np.float32)  # [steer, throttle, pitch, yaw, roll, jump, boost, handbrake]

        # Emergency: front-of-net danger → clear to corner
        if ctx.get("danger_zone", False):
            cx = 3072.0 if ball.physics.location.x >= 0 else -3072.0
            cy = (_goal_y(team) - ( -500.0 if team==0 else 500.0 ))
            steer, ang = _steer_to(me, cx, cy)
            action[0] = steer
            action[1] = 1.0
            action[6] = 1.0 if ang < 0.35 else 0.0
            # pop when close & ball low
            dx = me.physics.location.x - ball.physics.location.x
            dy = me.physics.location.y - ball.physics.location.y
            if ball.physics.location.z < 200.0 and (dx*dx + dy*dy) ** 0.5 < 450.0 and ang < 0.25:
                action[5] = 1.0
                action[2] = -0.2
        else:
            intent = ctx.get("intent","CONTROL")
            gx, gy = 0.0, _goal_y(team)
            bx, by, bz = ball.physics.location.x, ball.physics.location.y, ball.physics.location.z

            if intent == "SHOOT":
                # kickoff / straight power through center
                steer, ang = _steer_to(me, 0.0, 0.0)
                action[0] = steer; action[1] = 1.0; action[6] = 1.0 if ang < 0.2 else 0.0
                # front-flip if close and aligned
                db = float(((me.physics.location.x - bx)**2 + (me.physics.location.y - by)**2) ** 0.5)
                if db < 450 and ang < 0.2:
                    action[5] = 1.0; action[2] = 1.0  # flip forward
            elif intent == "SHADOW":
                # sit goal-side behind ball on same lane
                shy = by - (800 if team==0 else -800)
                steer, ang = _steer_to(me, bx, shy)
                action[0] = steer; action[1] = 0.8; action[6] = 0.0
            elif intent == "CONTROL":
                # gentle push to far post: bias x toward center
                aim_x = float(np.clip(bx * 0.7, -900, 900))
                steer, ang = _steer_to(me, aim_x, gy * 0.92)
                action[0] = steer; action[1] = 0.9; action[6] = 0.3 if ang < 0.25 else 0.0
                # small front pop if close & centered
                db = float(((me.physics.location.x - bx)**2 + (me.physics.location.y - by)**2) ** 0.5)
                if db < 380 and abs(me.physics.location.x - bx) < 180 and bz < 180 and ang < 0.2:
                    action[5] = 1.0; action[2] = 0.6
            elif intent == "CHALLENGE":
                steer, ang = _steer_to(me, bx, by)
                action[0] = steer; action[1] = 1.0; action[6] = 1.0 if ang < 0.3 else 0.0
            else:
                # CLEAR (non-emergency): roll toward side
                side_x = 2200.0 if bx >= 0 else -2200.0
                steer, ang = _steer_to(me, side_x, by)
                action[0] = steer; action[1] = 1.0; action[6] = 0.6 if ang < 0.35 else 0.0

        # Clamp by intent
        action = guard_by_intent(ctx.get("intent","CONTROL"), action, ctx)

        if model_action is not None:
            action = model_action.astype(np.float32)

        # Build controller
        ctl.steer      = float(action[0])
        ctl.throttle   = float(action[1])
        ctl.pitch      = float(action[2])
        ctl.yaw        = float(action[3])
        ctl.roll       = float(action[4])
        ctl.jump       = bool(action[5] > 0.5)
        ctl.boost      = bool(action[6] > 0.5)
        ctl.handbrake  = bool(action[7] > 0.5)

        # HUD
        if HAVE_HUD and not FAST_MODE and self.renderer is not None:
            reasons = f"P:{ctx.get('pressure_idx',0.0):.2f} T:{ctx.get('threat_idx',0.0):.2f}"
            ti = self.trainer.info()
            reasons += f"  | Train:{ti.get('training')} St:{ti.get('steps','-')}"
            try:
                draw_overlay(self.renderer,
                             intent=ctx.get("intent",""),
                             reasons=reasons,
                             boost=float(getattr(me, "boost", 33.0)),
                             action8=action,
                             dz=bool(ctx.get("danger_zone", False)))
            except Exception:
                pass

        # Occasionally inject a drill once state-setting is confirmed
        self._maybe_inject(packet)

        # --- online training step & autosave ---
        r = self._reward(packet)
        done = bool(packet.game_info.is_kickoff_pause)  # treat kickoff pause as boundary
        self.trainer.step(obs, action, r, done)
        self._steps += 1
        self.trainer.autosave_if_needed(self._steps)
        self._prev_obs = obs

        return ctl

# RLBot entry point
def create_agent(agent_name, team, index):
    return Destroyer(agent_name, team, index)
