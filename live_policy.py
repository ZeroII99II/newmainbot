import os, numpy as np
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

class LivePolicy:
    def __init__(self, path="necto-model.pt", device="cpu"):
        self.path = path
        self.device = device
        self.mtime = 0.0
        self.model = None
        if TORCH_OK:
            self._try_load()

    def _try_load(self):
        if not TORCH_OK or not os.path.exists(self.path):
            return
        try:
            self.mtime = os.path.getmtime(self.path)
            self.model = torch.jit.load(self.path, map_location=self.device).eval()
        except Exception:
            pass

    def maybe_reload(self):
        if not TORCH_OK or not os.path.exists(self.path):
            return
        m = os.path.getmtime(self.path)
        if m > self.mtime:
            self._try_load()

    def act(self, obs_np: np.ndarray) -> np.ndarray:
        if not TORCH_OK or self.model is None:
            return np.zeros(8, dtype=np.float32)
        try:
            self.maybe_reload()
            with torch.no_grad():
                x = torch.from_numpy(obs_np).float().unsqueeze(0)
                a = self.model(x)[0].cpu().numpy()
                if a.shape[0] != 8:
                    z = np.zeros(8, dtype=np.float32)
                    z[: min(8, a.shape[0])] = a[: min(8, a.shape[0])]
                    return z
                return a.astype(np.float32)
        except Exception:
            return np.zeros(8, dtype=np.float32)
