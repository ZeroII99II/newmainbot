from dataclasses import dataclass
from rlbot.utils.structures.quick_chats import QuickChats  # optional, if you want a callout

@dataclass
class SpeedFlipParams:
    jump1_time: float = 0.06
    jump2_delay: float = 0.17
    pre_align_deadband_deg: float = 4.0
    boost_on: bool = True
    bail_frontflip: bool = True

def run_speedflip(packet, spawn_side, ctl, params: SpeedFlipParams, t_since_kickoff: float):
    """
    Mutate `ctl` (SimpleControllerState) through kickoff:
    throttle+boost, jump at ~0.06s, pitch+yaw into flip, 2nd jump after ~0.17s,
    cancel at apex, optional wavedash landing; bail to front-flip if timing lost.
    """
    # Pseudocode sketch; implement exact timings and signs by `spawn_side`.
    # Keep it deterministic and frame-rate tolerant (use t_since_kickoff).
    return ctl
