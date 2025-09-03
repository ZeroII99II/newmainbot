# bot.py — unified Necto 1v1 bot with hot-reload + speed-flip + SSL aerial curriculum
# Minimal-change design: preserves 107→8 contract, runs clean in RLBot (1v1), no extra spawns.

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
import numpy as np
import math, os, time, threading
from enum import Enum

# =========================
# Runtime / Training knobs
# =========================
FAST_MODE = True            # Skip overlay text for max FPS
AERIAL_CURRICULUM = True    # Heavier aerial weighting in reward
AERIAL_DRILLS = True        # Offline aerial/double/flip-reset practice between plays
DRILL_INTERVAL_S = 7.0      # Cooldown for new drill scenes

ONLINE_TRAINER_ENABLED = True   # Optional online BC trainer (safe to disable)
ONLINE_TRAINER_STEP_EVERY = 1.0 # seconds between BC updates (higher throughput)
ONLINE_TRAINER_BATCH = 512
ONLINE_TRAINER_LR = 3e-4

ALLOWED_INDICES = {0, 1}    # Force clean 1v1 only

# ============
# Safe Torch
# ============
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# ================================
# Observation adapter (107-dim)
# ================================
# We rely on your existing necto_obs.py implementation.
try:
    from necto_obs import build_obs  # def build_obs(packet, index) -> np.ndarray(107,)
except Exception:
    build_obs = None

# ================================
# Live policy (TorchScript reload)
# ================================
class LivePolicy:
    def __init__(self, path="necto-model.pt", device="cpu"):
        self.path = path
        self.device = device
        self.mtime = 0.0
        self.model = None
        if TORCH_OK:
            self._try_load()

    def _try_load(self):
        if not TORCH_OK or not os.path.exists(self.path):
            return
        try:
            self.mtime = os.path.getmtime(self.path)
            self.model = torch.jit.load(self.path, map_location=self.device).eval()
        except Exception:
            pass  # keep previous model if load fails

    def maybe_reload(self):
        if not TORCH_OK or not os.path.exists(self.path):
            return
        m = os.path.getmtime(self.path)
        if m > self.mtime:
            self._try_load()

    def act(self, obs_np: np.ndarray) -> np.ndarray:
        if not TORCH_OK or self.model is None:
            return np.zeros(8, dtype=np.float32)
        try:
            self.maybe_reload()
            with torch.no_grad():
                x = torch.from_numpy(obs_np).float().unsqueeze(0)
                a = self.model(x)[0].cpu().numpy()
                if a.shape[0] != 8:
                    z = np.zeros(8, dtype=np.float32)
                    z[: min(8, a.shape[0])] = a[: min(8, a.shape[0])]
                    return z
                return a.astype(np.float32)
        except Exception:
            return np.zeros(8, dtype=np.float32)

# =========================
# Simple ring buffer (CPU)
# =========================
class RingBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.o = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def push(self, obs, act, rew, done):
        i = self.ptr
        self.o[i] = obs
        self.a[i] = act
        self.r[i] = rew
        self.d[i] = 1.0 if done else 0.0
        self.ptr = (i + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def snapshot(self):
        n = self.capacity if self.full else self.ptr
        return (self.o[:n].copy(), self.a[:n].copy(), self.r[:n].copy(), self.d[:n].copy())

# ==========================
# SSL reward (aerial-first)
# ==========================
DEFAULT_SSL_W = {
    "ball_progress": 0.45, "touch_quality": 0.30,
    "aerial_ctrl": 0.40, "gta_trans": 0.20,
    "flip_reset": 0.30, "double_tap": 0.30,
    "boost_pos": 0.18, "boost_neg": 0.10,
    "shadow": 0.18, "overcommit": 0.28,
    "kickoff": 0.22, "goal": 1.00, "concede": 1.00,
    "bad_touches": 0.12, "idle": 0.05
}

class SSLReward:
    def __init__(self, w=None):
        self.w = w or DEFAULT_SSL_W

    def __call__(self, info: dict) -> float:
        g = self.w; r = 0.0
        # Offense / control
        r += g["ball_progress"] * info.get("ball_to_opp_goal_cos", 0.0)
        r += g["touch_quality"] * info.get("ball_speed_gain_norm", 0.0)
        # Aerial skill
        r += g["aerial_ctrl"]  * info.get("aerial_alignment_cos", 0.0)
        r += g["gta_trans"]    * info.get("gta_transition_flag", 0.0)
        r += g["flip_reset"]   * info.get("flip_reset_attempt", 0.0)
        r += g["double_tap"]   * info.get("double_tap_attempt", 0.0)
        # Resource & defense
        r += g["boost_pos"]    * info.get("boost_use_good", 0.0)
        r -= g["boost_neg"]    * info.get("boost_waste", 0.0)
        r += g["shadow"]       * info.get("shadow_angle_cos", 0.0)
        r -= g["overcommit"]   * info.get("last_man_break_flag", 0.0)
        # Kickoff & outcomes
        r += g["kickoff"]      * info.get("kickoff_score", 0.0)
        r += g["goal"]         * info.get("scored", 0.0)
        r -= g["concede"]      * info.get("conceded", 0.0)
        # Anti-bad habits
        r -= g["bad_touches"]  * info.get("own_goal_touch", 0.0)
        r -= g["idle"]         * info.get("idle_ticks", 0.0)
        return float(max(-1.0, min(1.0, r)))

# =======================================
# Helpers for reward feature extraction
# =======================================
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

def _double_jump_state(car):
    return bool(getattr(car, "double_jumped", False))

def _latest_touch(packet):
    try:
        lt = packet.game_ball.latest_touch
        return lt.player_index, lt.time_seconds
    except Exception:
        return None, None

def _car_forward_vector(rot):
    cp, sp = math.cos(rot.pitch), math.sin(rot.pitch)
    cy, sy = math.cos(rot.yaw), math.sin(rot.yaw)
    return (cp*cy, cp*sy, sp)  # approximate forward

def _possession_good_boost(boost, progress_cos):  # reward using boost when progressing
    return 1.0 if (boost > 20 and progress_cos > 0.1) else 0.0

def _waste_boost(boost, progress_cos):  # penalize wasting when moving wrong way
    return 1.0 if (boost < 5 and progress_cos < -0.1) else 0.0

def _shadowing_cos(my_loc, ball_loc, my_forward, team):
    goal_y = -5120 if team == 0 else 5120
    to_goal = (0 - my_loc.x, goal_y - my_loc.y)
    to_ball = (ball_loc.x - my_loc.x, ball_loc.y - my_loc.y)
    return 0.5*_dot2(*to_goal, *to_ball) + 0.5*_dot2(my_forward[0], my_forward[1], to_ball[0], to_ball[1])

def _hit_backboard(ball_loc, team):
    return (ball_loc.y * _sign_to_opp(team)) > 4900 and ball_loc.z > 1200

def _flip_reset_heuristic(car, ball, was_airborne, prev_double_jump):
    airborne = _car_airborne(car)
    z = ball.physics.location.z
    if airborne and z > 800 and not prev_double_jump:
        return 0.5  # attempt credit
    return 0.0

def _double_tap_heuristic(now, last_backboard_ping, latest_touch_me):
    if last_backboard_ping > 0 and latest_touch_me and (now - last_backboard_ping) < 1.2:
        return 1.0
    return 0.0

# ==========================
# Kickoff detection/routine
# ==========================
class Spawn(Enum):
    MID=0; DIAG_L=1; DIAG_R=2; BACK_L=3; BACK_R=4

def is_kickoff_pause(packet) -> bool:
    gi = packet.game_info
    return bool(getattr(gi, "is_kickoff_pause", False))

def detect_spawn_side(car) -> Spawn:
    x = car.physics.location.x
    y = car.physics.location.y
    if abs(x) < 300: return Spawn.MID
    if y > 0: return Spawn.DIAG_L if x < 0 else Spawn.DIAG_R
    return Spawn.BACK_L if x < 0 else Spawn.BACK_R

class SpeedFlipParams:
    jump1_time: float = 0.06
    jump2_delay: float = 0.17
    pre_align_deadband_deg: float = 4.0
    bail_frontflip: bool = True

def run_speedflip(packet, my_car, spawn_side: Spawn, ctl: SimpleControllerState,
                  t_since: float, P: SpeedFlipParams):
    # Basic deterministic speed-flip with bailout. Tolerant to FPS.
    ctl.throttle = 1.0
    ctl.boost = True

    if spawn_side in (Spawn.DIAG_L, Spawn.DIAG_R):
        sign = -1.0 if spawn_side == Spawn.DIAG_L else +1.0
        ctl.steer = sign * 0.5
        ctl.yaw = sign * 0.25

    if P.jump1_time <= t_since < P.jump1_time + 0.05:
        ctl.jump = True
        ctl.pitch = 1.0

    if (P.jump1_time + P.jump2_delay) <= t_since < (P.jump1_time + P.jump2_delay + 0.05):
        ctl.jump = True
        if spawn_side in (Spawn.DIAG_L, Spawn.DIAG_R):
            ctl.yaw = (-1.0 if spawn_side == Spawn.DIAG_L else 1.0)
        ctl.pitch = 1.0

    if t_since >= P.jump1_time + P.jump2_delay + 0.18:
        ctl.yaw = 0.0
        ctl.pitch = -0.2  # stabilize

    if P.bail_frontflip and t_since > 0.7:  # miss window: front flip
        ctl.jump = True
        ctl.pitch = 1.0
        ctl.boost = True

    return ctl

# ==========================
# Optional Online BC trainer
# ==========================
class OnlineBC(threading.Thread):
    def __init__(self, buffer: RingBuffer, policy: LivePolicy,
                 out_path="necto-model.pt", step_every=3.0, batch=256, lr=2e-4):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.policy = policy
        self.out_path = out_path
        self.step_every = step_every
        self.batch = batch
        self.lr = lr
        self.stop_flag = False
        self.model = None
        self.opt = None
        self._setup_ok = False

    def run(self):
        if not TORCH_OK:
            return
        self.policy.maybe_reload()
        if self.policy.model is None:
            return
        self.model = self.policy.model
        try:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self._setup_ok = True
        except Exception:
            self._setup_ok = False

        last_save = time.time()
        while not self.stop_flag:
            time.sleep(self.step_every)
            if not self._setup_ok:
                continue
            obs, act, _, _ = self.buffer.snapshot()
            if len(obs) < self.batch:
                continue
            try:
                x = torch.from_numpy(obs).float()
                y = torch.from_numpy(act).float()
                idx = np.random.choice(len(x), size=min(self.batch, len(x)), replace=False)
                xb, yb = x[idx], y[idx]
                pred = self.model(xb)
                if pred.shape[-1] != 8:
                    continue
                loss = ((pred - yb) ** 2).mean()
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
            except Exception:
                continue

            if time.time() - last_save > self.step_every:
                try:
                    torch.jit.save(self.model, self.out_path)
                    last_save = time.time()
                except Exception:
                    pass

    def stop(self):
        self.stop_flag = True

# =====================
# Action → Controller
# =====================
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

# =========
# Agent
# =========
class NectoUnifiedAgent(BaseAgent):
    def initialize_agent(self):
        if self.index not in ALLOWED_INDICES:
            print(f"[guard] refusing unexpected bot index={self.index}")
            raise SystemExit(0)

        self.policy = LivePolicy(path="necto-model.pt", device="cpu")
        self.reward_fn = SSLReward(DEFAULT_SSL_W)
        self.buffer = RingBuffer(capacity=200_000, obs_dim=107, act_dim=8)

        # kickoff tracking
        self.kick_started = False
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

        # optional online BC trainer
        self._trainer = None
        if TORCH_OK and ONLINE_TRAINER_ENABLED:
            self._trainer = OnlineBC(self.buffer, self.policy,
                                     out_path="necto-model.pt",
                                     step_every=ONLINE_TRAINER_STEP_EVERY,
                                     batch=ONLINE_TRAINER_BATCH,
                                     lr=ONLINE_TRAINER_LR)
            self._trainer.start()

        print(f"Necto Ready - Index: {self.index}")

    def retire(self):
        if self._trainer:
            self._trainer.stop()

    # ======== reward info ========
    def _info_from_packet(self, packet) -> dict:
        info = {}
        ball = packet.game_ball
        my = packet.game_cars[self.index]
        team = self.team

        bv = ball.physics.velocity
        progress_cos = _dot2(bv.x, bv.y, 0.0, _sign_to_opp(team))
        info["ball_to_opp_goal_cos"] = progress_cos
        info["ball_speed_gain_norm"] = float(min(1.0, np.linalg.norm([bv.x, bv.y, bv.z]) / 4000.0))

        my_fwd = _car_forward_vector(my.physics.rotation)
        to_ball = (ball.physics.location.x - my.physics.location.x,
                   ball.physics.location.y - my.physics.location.y)
        facing_ball = _dot2(my_fwd[0], my_fwd[1], to_ball[0], to_ball[1])
        aerial_align = 0.5*facing_ball + 0.5*(1.0 - abs(my.physics.rotation.pitch))
        info["aerial_alignment_cos"] = float(max(-1.0, min(1.0, aerial_align)))

        info["gta_transition_flag"] = 1.0 if _car_airborne(my) and abs(my.physics.velocity.z) > 100 else 0.0

        boost = float(getattr(my, "boost", 33.0))
        info["boost_use_good"] = _possession_good_boost(boost, progress_cos)
        info["boost_waste"] = _waste_boost(boost, progress_cos)

        info["shadow_angle_cos"] = _shadowing_cos(my.physics.location, ball.physics.location, my_fwd, team)
        info["last_man_break_flag"] = 0.0  # optional later

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
        self._last_double_jump = _double_jump_state(my)

        return info

    # ======== aerial drills ========
    def _rand(self, a, b):
        return a + np.random.rand() * (b - a)

    def _set_aerial_drill(self, mode="general"):
        try:
            car_z = self._rand(50, 200)
            car_x = self._rand(-1500, 1500)
            car_y = self._rand(-3000, -1000) if self.team == 0 else self._rand(1000, 3000)
            car_y *= -1 if self.team == 1 else 1

            ball_x = self._rand(-1200, 1200)
            ball_y = self._rand(1200, 2200) * (_sign_to_opp(self.team))
            ball_z = self._rand(900, 1800)

            if mode == "double":
                bvx = self._rand(-400, 400)
                bvy = self._rand(1400, 1900) * (_sign_to_opp(self.team))
                bvz = self._rand(400, 800)
            elif mode == "flip":
                bvx = self._rand(-150, 150); bvy = self._rand(-150, 150); bvz = self._rand(-150, -50)
            else:
                bvx = self._rand(-300, 300); bvy = self._rand(800, 1400) * (_sign_to_opp(self.team)); bvz = self._rand(200, 600)

            car_boost = 100
            p = Physics(location=Vector3(car_x, car_y, car_z), rotation=Rotator(0.0, 0.0, 0.0), velocity=Vector3(0, 0, 0))
            car_state = CarState(physics=p, boost_amount=car_boost)
            ball_state = BallState(physics=Physics(location=Vector3(ball_x, ball_y, ball_z),
                                                   velocity=Vector3(bvx, bvy, bvz)))
            gs = GameState(ball=ball_state, cars={self.index: car_state})
            self.set_game_state(gs)
        except Exception:
            pass

    def _maybe_inject_drill(self, packet):
        if not (AERIAL_DRILLS and AERIAL_CURRICULUM):
            return
        now = packet.game_info.seconds_elapsed
        if is_kickoff_pause(packet):
            return
        ball = packet.game_ball
        speed = np.linalg.norm([ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z])
        if speed < 200 and ball.physics.location.z < 120 and (now - self._last_drill_time) > DRILL_INTERVAL_S:
            mode = np.random.choice(["general", "double", "flip"], p=[0.5, 0.25, 0.25])
            self._set_aerial_drill(mode)
            self._last_drill_time = now

    # ======== main control ========
    def get_output(self, packet) -> SimpleControllerState:
        if packet is None or packet.game_info is None:
            return SimpleControllerState()

        if build_obs is None:
            return SimpleControllerState()
        obs = build_obs(packet, self.index)
        if not isinstance(obs, np.ndarray) or obs.shape[0] != 107:
            return SimpleControllerState()

        action = self.policy.act(obs)
        ctl = to_controller(action)

        # kickoff only
        if is_kickoff_pause(packet):
            if not self.kick_started:
                self.kick_started = True
                self.kick_t0 = time.time()
                self.spawn_side = detect_spawn_side(packet.game_cars[self.index])
            t_since = time.time() - self.kick_t0
            ctl = run_speedflip(packet, packet.game_cars[self.index], self.spawn_side, ctl, t_since, SpeedFlipParams())
        else:
            self.kick_started = False

        info = self._info_from_packet(packet)
        if packet.game_info.is_kickoff_pause:
            info["kickoff_score"] = 0.2
        reward = self.reward_fn(info)
        done = bool(info["scored"] > 0.5)
        self.buffer.push(obs, action.astype(np.float32), reward, done)

        # update scores
        self._last_blue_goals = packet.teams[0].score
        self._last_orange_goals = packet.teams[1].score

        # optional drills
        self._maybe_inject_drill(packet)

        # minimal overlay (skip in FAST_MODE)
        if not FAST_MODE:
            try:
                r = self.renderer
                n = self.buffer.capacity if self.buffer.full else self.buffer.ptr
                r.draw_string_2d(10, 10, 1, 1, f"Idx {self.index} | Buf {n}/{self.buffer.capacity}", r.white())
                if TORCH_OK:
                    r.draw_string_2d(10, 28, 1, 1, f"HotReload: {int(self.policy.mtime) if self.policy.mtime else 0}", r.white())
            except Exception:
                pass

        return ctl

# RLBot entrypoint
def create_agent(agent_name, team, index):
    if index not in ALLOWED_INDICES:
        print(f"[guard] refusing unexpected bot index={index}")
        raise SystemExit(0)
    return NectoUnifiedAgent(agent_name, team, index)
