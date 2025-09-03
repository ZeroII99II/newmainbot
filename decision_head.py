# decision_head.py — Bronze-only intent guard
import numpy as np

INTENTS = {"SHADOW","CLEAR","CLEAR_CORNER","CONTROL","BOOST","CHALLENGE","SHOOT"}

def guard_by_intent(intent, a, ctx):
    # a = [steer, throttle, pitch, yaw, roll, jump, boost, handbrake]
    a = np.array(a, dtype=np.float32, copy=True)
    if intent == "CLEAR_CORNER":
        a[1] = 1.0; a[6] = max(a[6], 0.7); a[7] = 0.0
    elif intent == "SHADOW":
        a[1] = min(a[1], 0.8); a[6] = min(a[6], 0.2); a[7] = 0.0
    elif intent == "CONTROL":
        a[1] = min(a[1], 0.9); a[6] = min(a[6], 0.5); a[7] = 0.0
    elif intent == "CHALLENGE":
        a[1] = 1.0; a[6] = max(a[6], 0.6); a[7] = 0.0
    elif intent == "SHOOT":
        a[1] = 1.0; a[6] = max(a[6], 0.5)  # kickoff flip will be added by heuristic
    return a
