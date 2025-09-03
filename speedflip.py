from rlbot.agents.base_agent import SimpleControllerState
from dataclasses import dataclass
from kickoff_detector import Spawn

@dataclass
class SpeedFlipParams:
    jump1_time: float = 0.06
    jump2_delay: float = 0.17
    pre_align_deadband_deg: float = 4.0
    bail_frontflip: bool = True

def run_speedflip(packet, my_car, spawn_side: Spawn, ctl: SimpleControllerState,
                  t_since: float, P: SpeedFlipParams):
    # throttle/boost off the line
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
        ctl.pitch = -0.2  # stabilize for landing

    if P.bail_frontflip and t_since > 0.7:  # missed timing
        ctl.jump = True
        ctl.pitch = 1.0
        ctl.boost = True

    return ctl
