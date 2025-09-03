# kickoff_strats.py — Pro kickoff strategies + adaptive director for 1v1
from dataclasses import dataclass, field
from enum import Enum
import math, random
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from kickoff_detector import Spawn

# ------------ helpers ------------
def _clip(x, lo=-1.0, hi=1.0): return float(max(lo, min(hi, x)))
def _hyp(x, y): return math.hypot(x, y)

def _ang_norm(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _vec2(x, y):
    return np.array([float(x), float(y)], dtype=np.float32)

def _yaw(rot): return float(rot.yaw)
def _forward2(rot):
    cy, sy = math.cos(rot.yaw), math.sin(rot.yaw)
    cp = math.cos(rot.pitch)
    return np.array([cp * cy, cp * sy], dtype=np.float32)

def _sign_to_opp(team):  # +Y towards orange goal if we're blue (team 0)
    return -1.0 if team == 0 else 1.0

# Corner boost approximate world coords
CORNER_BOOSTS = [
    np.array([-3072.0, -4096.0], dtype=np.float32),
    np.array([ 3072.0, -4096.0], dtype=np.float32),
    np.array([-3072.0,  4096.0], dtype=np.float32),
    np.array([ 3072.0,  4096.0], dtype=np.float32),
]

# ------------- core timings -------------
@dataclass
class SpeedFlipParams:
    jump1_time: float = 0.06
    jump2_delay: float = 0.17
    bail_frontflip: bool = True

# ------------- primitive steering -------------
def steer_towards(current_yaw: float, target: np.ndarray, my_pos: np.ndarray, yaw_rate: float = 0.0):
    kp, kd = 2.2, 0.7
    tgt_yaw = math.atan2(target[1] - my_pos[1], target[0] - my_pos[0])
    ang = _ang_norm(tgt_yaw - current_yaw)
    steer = _clip(kp * ang - kd * yaw_rate)
    return steer, abs(ang)

# ------------- wave dash helper -------------
def maybe_wave_dash(ctl: SimpleControllerState, t_since: float, land_pitch: float = -0.2):
    # tiny landing accel if in the right time window
    if 0.65 < t_since < 0.85:
        ctl.jump = True
        ctl.pitch = 1.0
        ctl.boost = True
    elif t_since >= 0.85:
        ctl.pitch = land_pitch
    return ctl

# ------------- strategies -------------
def strat_speedflip(packet, my_car, spawn: Spawn, ctl: SimpleControllerState, t_since: float, P: SpeedFlipParams):
    ctl.throttle = 1.0; ctl.boost = True
    if spawn in (Spawn.DIAG_L, Spawn.DIAG_R):
        sign = -1.0 if spawn == Spawn.DIAG_L else +1.0
        ctl.steer = sign * 0.55
        ctl.yaw = sign * 0.25
    # jump timings
    if P.jump1_time <= t_since < P.jump1_time + 0.05:
        ctl.jump = True; ctl.pitch = 1.0
    if (P.jump1_time + P.jump2_delay) <= t_since < (P.jump1_time + P.jump2_delay + 0.05):
        ctl.jump = True
        if spawn in (Spawn.DIAG_L, Spawn.DIAG_R):
            ctl.yaw = (-1.0 if spawn == Spawn.DIAG_L else 1.0)
        ctl.pitch = 1.0
    if t_since >= P.jump1_time + P.jump2_delay + 0.18:
        ctl.yaw = 0.0; ctl.pitch = -0.2
    if P.bail_frontflip and t_since > 0.7:
        ctl.jump = True; ctl.pitch = 1.0; ctl.boost = True
    return maybe_wave_dash(ctl, t_since)

def strat_delayed_speedflip(packet, my_car, spawn: Spawn, ctl: SimpleControllerState, t_since: float, delay: float, P: SpeedFlipParams):
    if t_since < delay:
        # hold a beat to mess up mirror speedflips
        ctl.throttle = 0.0; ctl.boost = False; ctl.steer = 0.0
        return ctl
    return strat_speedflip(packet, my_car, spawn, ctl, t_since - delay, P)

def strat_hook(packet, my_car, spawn: Spawn, ctl: SimpleControllerState, t_since: float, hook_dir: int, P: SpeedFlipParams):
    """
    Hook left/right: curve path to hit ball off-center for a favorable pinch.
    hook_dir = -1 (left) or +1 (right) relative to our POV.
    """
    ctl = strat_speedflip(packet, my_car, spawn, ctl, t_since, P)
    # extra steer bias early
    if t_since < 0.5:
        ctl.steer = _clip(ctl.steer + 0.35 * hook_dir)
        ctl.yaw = _clip((ctl.yaw or 0.0) + 0.15 * hook_dir)
    return ctl

def strat_fake(packet, my_car, ctl: SimpleControllerState, t_since: float):
    """
    Fake: wait ~0.65s, turn to nearest corner boost, secure possession.
    """
    me = my_car
    my_pos = _vec2(me.physics.location.x, me.physics.location.y)
    # wait first; show fake
    if t_since < 0.65:
        ctl.throttle = 0.0; ctl.boost = False; ctl.steer = 0.0
        return ctl
    # pick nearest corner in our half
    team = me.team
    half_sign = -1.0 if team == 0 else 1.0
    candidates = [b for b in CORNER_BOOSTS if math.copysign(1.0, b[1]) == half_sign]
    target = min(candidates, key=lambda b: float(np.linalg.norm(b - my_pos)))
    yaw = _yaw(me.physics.rotation)
    yaw_rate = float(getattr(me.physics.angular_velocity, "z", 0.0))
    steer, ang_abs = steer_towards(yaw, target, my_pos, yaw_rate)
    ctl.steer = steer
    ctl.handbrake = 1.0 if (ang_abs > 1.3) else 0.0
    ctl.boost = 1.0 if ang_abs < 0.25 else 0.0
    ctl.throttle = 1.0
    return ctl

# ------------- Adaptive director -------------
@dataclass
class StrategyWeights:
    speedflip: float = 0.60
    delayed: float = 0.15
    hook_left: float = 0.10
    hook_right: float = 0.10
    fake: float = 0.05

@dataclass
class KickoffBook:
    rng_seed: int = 0
    weights_mid: StrategyWeights = field(default_factory=StrategyWeights)
    weights_diag: StrategyWeights = field(default_factory=StrategyWeights)
    weights_back: StrategyWeights = field(default_factory=StrategyWeights)
    last_outcomes: list = field(default_factory=list)  # list of (+1 scored/-1 conceded/0 neutral)
    delay_window: tuple = (0.06, 0.12)  # delayed sf window

    def __post_init__(self):
        random.seed(self.rng_seed)

    def _pick(self, spawn: Spawn) -> str:
        w = self.weights_diag if spawn in (Spawn.DIAG_L, Spawn.DIAG_R) else (self.weights_mid if spawn == Spawn.MID else self.weights_back)
        table = [("speedflip", w.speedflip),
                 ("delayed",   w.delayed),
                 ("hook_left", w.hook_left),
                 ("hook_right",w.hook_right),
                 ("fake",      w.fake)]
        tot = sum(v for _, v in table) or 1.0
        r = random.random() * tot
        acc = 0.0
        for name, v in table:
            acc += v
            if r <= acc: return name
        return "speedflip"

    def _nudge(self, spawn: Spawn, good: bool, name: str):
        # tiny bandit-style nudge to favor what worked
        scale = 0.03 if good else -0.03
        w = self.weights_diag if spawn in (Spawn.DIAG_L, Spawn.DIAG_R) else (self.weights_mid if spawn == Spawn.MID else self.weights_back)
        cur = getattr(w, name if name in ("fake", "delayed", "speedflip") else ("hook_left" if name=="hook_left" else "hook_right"), None)
        if cur is None: return
        newv = max(0.02, min(0.85, cur + scale))
        setattr(w, name if name in ("fake", "delayed", "speedflip") else ("hook_left" if name=="hook_left" else "hook_right"), newv)
        # normalize a bit
        s = w.speedflip + w.delayed + w.hook_left + w.hook_right + w.fake
        w.speedflip /= s; w.delayed /= s; w.hook_left /= s; w.hook_right /= s; w.fake /= s

class KickoffDirector:
    def __init__(self, seed=0):
        self.book = KickoffBook(rng_seed=seed)
        self.active_name = None
        self.delay_chosen = 0.08
        self.P = SpeedFlipParams()
        self.spawn = Spawn.MID

    def start(self, spawn: Spawn):
        self.spawn = spawn
        self.active_name = self.book._pick(spawn)
        if self.active_name == "delayed":
            lo, hi = self.book.delay_window
            self.delay_chosen = random.uniform(lo, hi)

    def run(self, packet, my_car, t_since: float, ctl: SimpleControllerState):
        # dispatch
        if self.active_name == "speedflip":
            return strat_speedflip(packet, my_car, self.spawn, ctl, t_since, self.P)
        if self.active_name == "delayed":
            return strat_delayed_speedflip(packet, my_car, self.spawn, ctl, t_since, self.delay_chosen, self.P)
        if self.active_name == "hook_left":
            return strat_hook(packet, my_car, self.spawn, ctl, t_since, hook_dir=-1, P=self.P)
        if self.active_name == "hook_right":
            return strat_hook(packet, my_car, self.spawn, ctl, t_since, hook_dir=+1, P=self.P)
        if self.active_name == "fake":
            return strat_fake(packet, my_car, ctl, t_since)
        # default
        return strat_speedflip(packet, my_car, self.spawn, ctl, t_since, self.P)

    def report_outcome(self, spawn: Spawn, scored: bool, conceded: bool):
        name = self.active_name or "speedflip"
        # Update weights based on outcome
        if scored and not conceded:
            self.book._nudge(spawn, True, name)
        elif conceded and not scored:
            self.book._nudge(spawn, False, name)
        # else neutral: no change
        self.active_name = None
