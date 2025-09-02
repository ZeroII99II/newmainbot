from enum import Enum
class Spawn(Enum):
    MID=0; DIAG_L=1; DIAG_R=2; BACK_L=3; BACK_R=4

def is_kickoff_pause(packet) -> bool:
    # RLBot: kickoff when packet.game_info.is_round_active == False or kickoff_pause True
    gi = packet.game_info
    return getattr(gi, "is_kickoff_pause", False)

def detect_spawn_side(my_car, field_info) -> Spawn:
    # Use car start pos x/y to classify; sign(x) & distance from center decide side
    x = my_car.physics.location.x
    y = my_car.physics.location.y
    if abs(x) < 300: return Spawn.MID
    if y > 0:
        return Spawn.DIAG_L if x < 0 else Spawn.DIAG_R
    else:
        return Spawn.BACK_L if x < 0 else Spawn.BACK_R
