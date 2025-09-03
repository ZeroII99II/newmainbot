from enum import Enum

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
