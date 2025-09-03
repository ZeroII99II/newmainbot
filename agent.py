import math, os, numpy as np

# Torch is optional: bot must still load without it
try:
    import torch
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    torch = None
    F = None
    class Categorical:
        def __init__(self, logits): pass
        def sample(self): return None


class Agent:
    def __init__(self):
        cur = os.path.dirname(os.path.realpath(__file__))
        # Prefer our own Bronze model, but accept nexto-model.pt too
        candidates = [
            os.path.join(cur, "models", "destroyer_Bronze_latest.pt"),
            os.path.join(cur, "destroyer_Bronze_latest.pt"),
            os.path.join(cur, "nexto-model.pt"),
        ]
        self.actor = None
        if TORCH_OK:
            for p in candidates:
                if os.path.exists(p):
                    try:
                        with open(p, "rb") as f:
                            self.actor = torch.jit.load(f, map_location="cpu")
                        break
                    except Exception:
                        pass
            if self.actor is not None:
                torch.set_num_threads(1)

        self._lookup_table = self.make_lookup_table()  # keep existing implementation
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
        # If we don't have a model (or torch), let bot.py fall back to heuristics
        if (not TORCH_OK) or (self.actor is None):
            return None, None

        state = tuple(torch.from_numpy(s).float() for s in state)
        with torch.no_grad():
            out, weights = self.actor(state)

        out = (out,)
        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [o if o.shape[-1] == max_shape else F.pad(o, (0, max_shape - o.shape[-1]), value=float("-inf"))],
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
                import math as _m
                logits *= _m.log((beta + 1) / (1 - beta), 3)
            dist = Categorical(logits=logits)
            actions = dist.sample()

        parsed = self._lookup_table[actions.numpy().item()]
        return parsed, weights

