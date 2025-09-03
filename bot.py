# bot.py — "Destroyer" (1v1), SSL/pro curriculum, hot-reload, aerial drills
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator
import numpy as np, math, os, time
from enum import Enum

# ==== Runtime knobs ====
BOT_NAME = "Destroyer"        # printed label only
FAST_MODE = True              # hide overlay to maximize FPS
ENABLE_AERIAL_CURRICULUM = True
ENABLE_AERIAL_DRILLS = True   # offline only; inject aerial/double/flip scenes
DRILL_INTERVAL_S = 7.0

ENABLE_ONLINE_TRAINER = True
TRAINER_STEP_EVERY = 1.0      # seconds
TRAINER_BATCH = 512
TRAINER_LR = 3e-4

ALLOWED_INDICES = {0, 1}      # force clean 1v1

# Torch (safe import checked inside live_policy/trainer)
from live_policy import LivePolicy
from ring_buffer import RingBuffer
from rewards_ssl import SSLReward, DEFAULT_SSL_W
from kickoff_detector import is_kickoff_pause, detect_spawn_side, Spawn
from speedflip import run_speedflip, SpeedFlipParams
from trainer_online_bc import OnlineBC

# 107-dim obs adapter (keep your function signature)
try:
    from necto_obs import build_obs  # def build_obs(packet, index) -> np.ndarray(107,)
except Exception:
    build_obs = None

# ==== helpers for reward features ====
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

def _possession_good_boost(boost, progress_cos):  # reward using boost to progress
    return 1.0 if (boost > 20 and progress_cos > 0.1) else 0.0

def _waste_boost(boost, progress_cos):            # penalize wasting while regressing
    return 1.0 if (boost < 5 and progress_cos < -0.1) else 0.0

def _shadow_cos(my_loc, ball_loc, my_fwd, team):
    goal_y = -5120 if team == 0 else 5120
    to_goal = (0 - my_loc.x, goal_y - my_loc.y)
    to_ball = (ball_loc.x - my_loc.x, ball_loc.y - my_loc.y)
    return 0.5*_dot2(*to_goal, *to_ball) + 0.5*_dot2(my_fwd[0], my_fwd[1], to_ball[0], to_ball[1])

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
        if ENABLE_ONLINE_TRAINER:
            try:
                self._trainer = OnlineBC(self.buffer, self.policy,
                                         out_path="necto-model.pt",
                                         step_every=TRAINER_STEP_EVERY,
                                         batch=TRAINER_BATCH,
                                         lr=TRAINER_LR)
                self._trainer.start()
            except Exception:
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
        self._last_double_jump = _double_jumped(my)
        return info

    # drills
    def _rand(self, a, b): return a + np.random.rand() * (b - a)

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
        if not (ENABLE_AERIAL_DRILLS and ENABLE_AERIAL_CURRICULUM):
            return
        now = packet.game_info.seconds_elapsed
        if is_kickoff_pause(packet):
            return
        ball = packet.game_ball
        speed = np.linalg.norm([ball.physics.velocity.x, ball.physics.velocity.y, ball.physics.velocity.z])
        if speed < 200 and ball.physics.location.z < 120 and (now - self._last_drill_time) > DRILL_INTERVAL_S:
            import numpy as _np
            mode = _np.random.choice(["general", "double", "flip"], p=[0.5, 0.25, 0.25])
            self._set_aerial_drill(mode)
            self._last_drill_time = now

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

        self._maybe_inject_drill(packet)

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
