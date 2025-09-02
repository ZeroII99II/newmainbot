import numpy as np
from live_policy import LivePolicy

class Agent:
    def __init__(self):
        self.policy = LivePolicy(path="necto-model.pt")

    def act(self, obs, beta=1):
        if isinstance(obs, (list, tuple)):
            flat = np.concatenate([np.asarray(o).flatten() for o in obs])
        else:
            flat = np.asarray(obs).flatten()
        if flat.size < 107:
            flat = np.pad(flat, (0, 107 - flat.size))
        action = self.policy.act(flat[:107])
        return action, None
