# boost_pathing.py — tiny small-pad planner for 1s
import numpy as np

# XY coords of key small pads (subset to keep tight)
SMALL_PADS = np.array([
    [-2048, -2560], [0, -2560], [2048, -2560],
    [-2560, 0],     [2560, 0],
    [-2048, 2560],  [0, 2560],  [2048, 2560],
    [-1000, -1000], [1000, -1000], [-1000, 1000], [1000, 1000],
], dtype=np.float32)

def nearest_small_pad_xy(x, y, prefer_half_sign=None):
    p = np.array([x, y], dtype=np.float32)
    pads = SMALL_PADS
    if prefer_half_sign is not None:
        pads = pads[np.sign(pads[:,1]) == prefer_half_sign]
        if pads.size == 0: pads = SMALL_PADS
    i = int(np.argmin(np.linalg.norm(pads - p, axis=1)))
    return pads[i]

def football_lane(x, y, to_defense=False):
    # quick “football route” selection: towards own half if defending, else toward opp half
    p = np.array([x, y], dtype=np.float32)
    if to_defense:
        target = nearest_small_pad_xy(p[0], -4096 if y > 0 else 4096, prefer_half_sign=-np.sign(y) if y != 0 else None)
    else:
        target = nearest_small_pad_xy(p[0],  4096 if y < 0 else -4096, prefer_half_sign=np.sign(-y) if y != 0 else None)
    return target
