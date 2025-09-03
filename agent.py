import os, math
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    # Provide stubs so the rest of the file parses if torch is missing
    class Categorical:
        def __init__(self, logits):
            pass
        def sample(self):
            return None

class Agent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        candidates = [
            os.path.join(cur_dir, "models", "destroyer_Bronze_latest.pt"),
            os.path.join(cur_dir, "destroyer_Bronze_latest.pt"),
            os.path.join(cur_dir, "nexto-model.pt"),
        ]
        self.actor = None
        if TORCH_OK:
            for path in candidates:
                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            self.actor = torch.jit.load(f, map_location="cpu")
                        break
                    except Exception:
                        pass
            if self.actor is None:
                # No model available; we'll run Bronze heuristics instead
                pass
            torch.set_num_threads(1)

        self._lookup_table = self.make_lookup_table()
        self.state = None

    def make_lookup_table(self):
        table = []
        for steer in (-1, 0, 1):
            for throttle in (-1, 0, 1):
                action = np.zeros(8, dtype=np.float32)
                action[0] = steer
                action[1] = throttle
                table.append(action)
        return table

    def act(self, state, beta):
        if not TORCH_OK or self.actor is None:
            return None, None  # tell bot.py to use Bronze heuristics

        state = tuple(torch.from_numpy(s).float() for s in state)
        with torch.no_grad():
            out, weights = self.actor(state)
        self.state = state

        out = (out,)
        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [l if l.shape[-1] == max_shape else F.pad(l, (0, max_shape - l.shape[-1]), value=float("-inf"))],
            dim=1
        )

        if beta == 1:
            actions = np.argmax(logits, axis=-1)
        elif beta == -1:
            actions = np.argmin(logits, axis=-1)
        else:
            if beta == 0:
                logits[torch.isfinite(logits)] = 0
            else:
                logits *= math.log((beta + 1) / (1 - beta), 3)
            dist = Categorical(logits=logits)
            actions = dist.sample()

        parsed = self._lookup_table[actions.numpy().item()]
        return parsed, weights
