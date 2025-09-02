import os, torch, numpy as np

class LivePolicy:
    def __init__(self, path="necto-model.pt", device="cpu"):
        self.path = path
        self.device = device
        self.mtime = 0.0
        self.model = None
        self._try_load()

    def _try_load(self):
        if not os.path.exists(self.path): return
        self.mtime = os.path.getmtime(self.path)
        self.model = torch.jit.load(self.path, map_location=self.device).eval()

    def maybe_reload(self):
        if not os.path.exists(self.path): return
        m = os.path.getmtime(self.path)
        if m > self.mtime:
            self._try_load()

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        self.maybe_reload()
        if self.model is None:
            return np.zeros(8, dtype=np.float32)
        x = torch.from_numpy(obs_np).float().unsqueeze(0)
        a = self.model(x)[0].cpu().numpy()
        return a
