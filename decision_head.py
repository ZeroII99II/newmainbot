# decision_head.py — intent guards that shape/control actions
import numpy as np

INTENTS = {"PRESS","CONTROL","CHALLENGE","SHADOW","BOOST","CLEAR","DRIBBLE","SHOOT","FAKE","ROTATE_BACK_POST"}

def guard_by_intent(intent: str, action: np.ndarray, ctx: dict) -> np.ndarray:
    """
    Modify the 8-dim action vector to enforce high-level decisions.
    action = [steer, throttle, pitch, yaw, roll, jump, boost, handbrake]
    """
    a = action.copy().astype(np.float32)
    intent = (intent or "PRESS").upper()
    risk = float(ctx.get("risk_budget", 0.3))

    # Conservative clamps when defending / low recovery quality
    if intent in ("SHADOW","ROTATE_BACK_POST"):
        a[6] = 0.0  # boost off
        a[1] = float(np.clip(a[1], 0.4, 0.9))  # modest throttle forward
        a[7] = 0.0  # no powerslide spam
        a[5] = 0.0  # no jumps
        return a

    # Go grab boost
    if intent == "BOOST":
        a[6] = 1.0            # boost on
        a[5] = 0.0            # no jumps
        a[7] = 0.0            # no drift
        a[1] = 1.0            # full throttle
        return a

    # Challenge = decisive approach (less handbrake), allow boost if aligned
    if intent == "CHALLENGE":
        a[7] = 0.0
        a[6] = max(a[6], 0.7)
        a[1] = 1.0
        return a

    # Control/Dribble = gentle throttle; no random jumps; keep boost mostly off
    if intent in ("CONTROL","DRIBBLE"):
        a[1] = float(np.clip(a[1], 0.2, 0.8))
        a[6] = float(np.clip(a[6], 0.0, 0.3))
        a[5] = float(np.clip(a[5], 0.0, 0.5))
        return a

    # Shoot / Clear = allow more boost; jumps allowed
    if intent in ("SHOOT","CLEAR","PRESS"):
        a[6] = max(a[6], 0.5 if intent=="CLEAR" else 0.8)
        a[1] = 1.0
        return a

    # Fake = kill throttle/boost briefly (deception window handled elsewhere)
    if intent == "FAKE":
        a[1] = min(a[1], 0.2)
        a[6] = 0.0
        a[5] = 0.0
        return a

    return a
