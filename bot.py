# bot.py — Destroyer: resumes from Necto (working copy), hot-reload, kickoff-after-pause, fallback motion
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
import numpy as np, math, os, time, shutil
from pathlib import Path
from enum import Enum

# ===== Runtime knobs =====
BOT_NAME = "Destroyer"
FAST_MODE = True

# Start conservative; re-enable after motion confirmed
ENABLE_AERIAL_CURRICULUM = True
ENABLE_AERIAL_DRILLS = False     # OFF first; turn ON later
DRILL_INTERVAL_S = 7.0

ENABLE_ONLINE_TRAINER = False    # OFF first; turn ON after motion confirmed
TRAINER_STEP_EVERY = 1.0
TRAINER_BATCH = 512
TRAINER_LR = 3e-4

ALLOWED_INDICES = {0, 1}         # clean 1v1 only

# ==== Model resume settings ====
MODEL_PREFERENCE = ["destroyer.pt", "necto-model.pt", "necto.pt", "NectoModel.pt", "model.pt"]
WORKING_MODEL = "destroyer.pt"    # fine-tune target; created from first found base

# ==== Imports from our local modules ====
from live_policy import LivePolicy
from ring_buffer import RingBuffer
from rewards_ssl import SSLReward, DEFAULT_SSL_W
from kickoff_detector import is_kickoff_pause, detect_spawn_side, Spawn
from speedflip import run_speedflip, SpeedFlipParams
from trainer_online_bc import OnlineBC

# 107-dim obs adapter
try:
    from necto_obs import build_obs  # def build_obs(packet, index) -> np.ndarray(107,)
except Exception:
    build_obs = None

# ===== helpers for reward features =====
def _dot2(ax, ay, bx, by):
    n1 = math.hypot(ax, ay); n2 = math.hypot(bx, by)
    if n1 < 1e-6 or n2 < 1e-6: return 0.0
    return max(-1.0, min(1.0, (ax*bx + ay*by) / (n1*n2)))

def _sign_to_opp(team):  # rough Y-direction toward opponent goal
    return -1.0 if team == 0 else 1.0

def _car_airborne(car):
    try:
        return (not car.has_wheel_contact) and (car.physics.location.z > 80)
    except Exception:
        return car.physics.location.z > 80

def _double_jumped(car):
    return bool(getattr(car, "double_jumped", False))

def _latest_touch(packet):
    try:
        lt = packet.game_ball.latest_touch
        return lt.player_index, lt.time_seconds
    except Exception:
        return None, None

def _car_forward(rot):
    cp, sp = math.cos(rot.pitch), math.sin(rot.pitch)
    cy, sy = math.cos(rot.yaw), math.sin(rot.yaw)
    return (cp*cy, cp*sy, sp)

def _possession_good_boost(boost, progress_cos):
    return 1.0 if (boost > 20 and progress_cos > 0.1) else 0.0

def _waste_boost(boost, progress_cos):
    return 1.0 if (boost < 5 and progress_cos < -0.1) else 0.0

def _shadow_cos(my_loc, ball_loc, my_fwd, team):
    goal_y = -5120 if team == 0 else 5120
    to_goal = (0 - my_loc.x, goal_y - my_loc.y)
    to_ball = (ball_loc.x - my_loc.x, ball_loc.y - my_loc.y)
    return 0.5*_dot2(*to_goal, *to_ball) + 0.5*_dot2(my_fwd[0], my_fwd[1], to_ball[0], to_ball[1])

def _hit_backboard(ball_loc, team):
    return (ball_loc.y * _sign_to_opp(team)) > 4900 and ball_loc.z > 1200

def _flip_reset_heuristic(car, ball, was_airborne, prev_double_jump):
    airborne = _car_airborne(car); z = ball.physics.location.z
    if airborne and z > 800 and not prev_double_jump:
        return 0.5
    return 0.0

def _double_tap_heuristic(now, last_backboard_ping, latest_touch_me):
    if last_backboard_ping > 0 and latest_touch_me and (now - last_backboard_ping) < 1.2:
        return 1.0
    return 0.0

def to_controller(a: np.ndarray) -> SimpleControllerState:
    ctl = SimpleControllerState()
    steer, throttle, pitch, yaw, roll, jump, boost, handbrake = a.tolist()
    ctl.steer = float(np.clip(steer, -1, 1))
    ctl.throttle = float(np.clip(throttle, -1, 1))
    ctl.pitch = float(np.clip(pitch, -1, 1))
    ctl.yaw = float(np.clip(yaw, -1, 1))
    ctl.roll = float(np.clip(roll, -1, 1))
    ctl.jump = bool(jump > 0.5)
    ctl.boost = bool(boost > 0.5)
    ctl.handbrake = bool(handbrake > 0.5)
    return ctl

class Destroyer(BaseAgent):
    def initialize_agent(self):
        if self.index not in ALLOWED_INDICES:
            print(f"[guard] refusing unexpected bot index={self.index}")
            raise SystemExit(0)

        # --- Warm start from existing model(s) ---
        def _pick_existing_model():
            for name in MODEL_PREFERENCE:
                if Path(name).exists():
                    return name
            return None

        base_model = _pick_existing_model()
        if base_model is None:
            print("[Destroyer] WARNING: No base model found next to bot.py. Expected one of:", MODEL_PREFERENCE)
        else:
            if not Path(WORKING_MODEL).exists():
                try:
                    shutil.copy(base_model, WORKING_MODEL)
                    print(f"[Destroyer] Warm-started working model '{WORKING_MODEL}' from '{base_model}'")
                except Exception as e:
                    print(f"[Destroyer] Copy failed ({e}); will try to load base directly")

        # Policy loads working model, with fallbacks to base names
        self.policy = LivePolicy(path=WORKING_MODEL, device="cpu", fallback_paths=MODEL_PREFERENCE)
        self.reward_fn = SSLReward(DEFAULT_SSL_W)
        self.buffer = RingBuffer(capacity=200_000, obs_dim=107, act_dim=8)

        # kickoff tracking + correct timing (after pause)
        self.kick_prepping = False
        self.kick_active = False
        self.kick_t0 = 0.0
        self.spawn_side = Spawn.MID
        self.kick_params = SpeedFlipParams()

        # score trackers for 'done'
        self._last_blue_goals = 0
        self._last_orange_goals = 0

        # aerial heuristics memory
        self._backboard_ping_time = 0.0
        self._last_car_airborne = False
        self._last_double_jump = False
        self._last_drill_time = 0.0

        # optional online BC trainer (OFF by default; turn on after motion confirmed)
        self._trainer = None
        if ENABLE_ONLINE_TRAINER:
            try:
                self._trainer = OnlineBC(self.buffer, self.policy,
                                         out_path=WORKING_MODEL,
                                         step_every=TRAINER_STEP_EVERY,
                                         batch=TRAINER_BATCH,
                                         lr=TRAINER_LR)
                self._trainer.start()
            except Exception as e:
                print(f"[Destroyer] Trainer did not start: {e}")
                self._trainer = None

        print(f"{BOT_NAME} Ready - Index: {self.index}")

    def retire(self):
        if self._trainer:
            self._trainer.stop()

    def _info_from_packet(self, packet) -> dict:
        info = {}
        ball = packet.game_ball
        my = packet.game_cars[self.index]
        team = self.team

        bv = ball.physics.velocity
        progress_cos = _dot2(bv.x, bv.y, 0.0, _sign_to_opp(team))
        info["ball_to_opp_goal_cos"] = progress_cos
        info["ball_speed_gain_norm"] = float(min(1.0, np.linalg.norm([bv.x, bv.y, bv.z]) / 4000.0))

        my_fwd = _car_forward(my.physics.rotation)
        to_ball = (ball.physics.location.x - my.physics.location.x,
                   ball.physics.location.y - my.physics.location.y)
        facing_ball = _dot2(my_fwd[0], my_fwd[1], to_ball[0], to_ball[1])
        aerial_align = 0.5*facing_ball + 0.5*(1.0 - abs(my.physics.rotation.pitch))
        info["aerial_alignment_cos"] = float(max(-1.0, min(1.0, aerial_align)))
        info["gta_transition_flag"] = 1.0 if _car_airborne(my) and abs(my.physics.velocity.z) > 100 else 0.0

        boost = float(getattr(my, "boost", 33.0))
        info["boost_use_good"] = _possession_good_boost(boost, progress_cos)
        info["boost_waste"] = _waste_boost(boost, progress_cos)

        info["shadow_angle_cos"] = _shadow_cos(my.physics.location, ball.physics.location, my_fwd, team)
        info["last_man_break_flag"] = 0.0

        lt_idx, _ = _latest_touch(packet)
        now = packet.game_info.seconds_elapsed
        latest_touch_me = (lt_idx == self.index) if lt_idx is not None else False

        if _hit_backboard(ball.physics.location, team):
            self._backboard_ping_time = now

        info["double_tap_attempt"] = _double_tap_heuristic(now, self._backboard_ping_time, latest_touch_me)
        info["flip_reset_attempt"] = _flip_reset_heuristic(my, ball, self._last_car_airborne, self._last_double_jump)

        scores = packet.teams
        scored = 1.0 if (scores[0].score != self._last_blue_goals or scores[1].score != self._last_orange_goals) else 0.0
        conceded = 1.0 if scored and ((team == 0 and scores[1].score > self._last_orange_goals) or
                                      (team == 1 and scores[0].score > self._last_blue_goals)) else 0.0
        info["scored"] = scored
        info["conceded"] = conceded

        info["own_goal_touch"] = 0.0
        info["idle_ticks"] = 0.0
        info["kickoff_score"] = 0.0

        self._last_car_airborne = _car_airborne(my)
        self._last_double_jump = _double_jumped(my)
        return info

    # (Aerial drills exist in earlier versions; left off until motion confirmed)

    def get_output(self, packet) -> SimpleControllerState:
        if packet is None or packet.game_info is None:
            return SimpleControllerState()

        if build_obs is None:
            return SimpleControllerState()
        obs = build_obs(packet, self.index)
        if not isinstance(obs, np.ndarray) or obs.shape[0] != 107:
            print(f"[Destroyer] bad obs shape: {None if not isinstance(obs, np.ndarray) else obs.shape}")
            return SimpleControllerState()

        # --- Policy action
        action = self.policy.act(obs)

        # --- Fallback motion if model not loaded (all zeros)
        if (action == 0).all():
            ball = packet.game_ball; me = packet.game_cars[self.index]
            dx = ball.physics.location.x - me.physics.location.x
            dy = ball.physics.location.y - me.physics.location.y
            steer = np.clip(np.arctan2(dy, dx) / np.pi, -1, 1)
            throttle = 1.0
            hb = 1.0 if abs(steer) > 0.5 else 0.0
            jump = 1.0 if ball.physics.location.z > 400 else 0.0
            action = np.array([steer, throttle, 0, 0, 0, jump, 1.0, hb], dtype=np.float32)

        ctl = to_controller(action)

        # --- Kickoff timing: prep during pause; execute after unpause
        gi = packet.game_info
        if is_kickoff_pause(packet):
            if not self.kick_prepping:
                self.kick_prepping = True
                self.kick_active = False
                self.spawn_side = detect_spawn_side(packet.game_cars[self.index])
        else:
            if self.kick_prepping:
                self.kick_prepping = False
                self.kick_active = True
                self.kick_t0 = gi.seconds_elapsed

        if self.kick_active:
            t_since = gi.seconds_elapsed - self.kick_t0
            ctl = run_speedflip(packet, packet.game_cars[self.index], self.spawn_side, ctl, t_since, SpeedFlipParams())
            if t_since > 1.2:
                self.kick_active = False

        # --- Reward + buffer
        info = self._info_from_packet(packet)
        if packet.game_info.is_kickoff_pause:
            info["kickoff_score"] = 0.2
        reward = self.reward_fn(info)
        done = bool(info["scored"] > 0.5)
        self.buffer.push(obs, action.astype(np.float32), reward, done)

        # update scores
        self._last_blue_goals = packet.teams[0].score
        self._last_orange_goals = packet.teams[1].score

        # Overlay (disabled in FAST_MODE)
        if not FAST_MODE:
            try:
                r = self.renderer
                n = self.buffer.capacity if self.buffer.full else self.buffer.ptr
                r.draw_string_2d(10, 10, 1, 1, f"{BOT_NAME} | Idx {self.index} | Buf {n}/{self.buffer.capacity}", r.white())
            except Exception:
                pass

        return ctl

def create_agent(agent_name, team, index):
    if index not in ALLOWED_INDICES:
        print(f"[guard] refusing unexpected bot index={index}")
        raise SystemExit(0)
    return Destroyer(agent_name, team, index)
