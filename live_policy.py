import os, time, numpy as np
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

class LivePolicy:
    """
    Loads a TorchScript (jit) model and hot-reloads it when the file changes.
    If primary path fails, tries fallbacks at startup.
    """
    def __init__(self, path="destroyer.pt", device="cpu", fallback_paths=None):
        self.device = device
        self.path = path
        self.fallbacks = [p for p in (fallback_paths or []) if p != path]
        self.mtime = 0.0
        self.model = None
        if TORCH_OK:
            self._try_load(first=True)

    def _attempt_load_file(self, fpath):
        if not os.path.exists(fpath):
            return False
        try:
            m = torch.jit.load(fpath, map_location=self.device).eval()
            self.model = m
            self.mtime = os.path.getmtime(fpath)
            self.path = fpath
            print(f"[Destroyer] Loaded JIT model: {os.path.basename(fpath)} @ {time.ctime(self.mtime)}")
            return True
        except Exception as e:
            print(f"[Destroyer] Skipped non-JIT or incompatible model '{fpath}': {e}")
            return False

    def _try_load(self, first=False):
        if self._attempt_load_file(self.path):
            return
        for cand in self.fallbacks:
            if self._attempt_load_file(cand):
                return
        if first:
            print("[Destroyer] No loadable JIT model found; will use fallback controls.")

    def maybe_reload(self):
        if not TORCH_OK or self.model is None:
            return
        try:
            mtime = os.path.getmtime(self.path)
            if mtime > self.mtime:
                self._attempt_load_file(self.path)
        except Exception:
            pass

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
