# progress_store.py — JSON persistence for stage index & EMAs
import json, time
from pathlib import Path
from typing import Dict

DEFAULT_PATH = Path("academy_state.json")

class EMA:
    def __init__(self, alpha=0.1): self.a=alpha; self.v=None
    def update(self, x):
        self.v = x if self.v is None else (1-self.a)*self.v + self.a*x
        return self.v

class ProgressStore:
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)
        self.stage_idx = 0
        self.stage_started_at = time.time()
        self.metric_ema: Dict[str, EMA] = {}
        self.progress_ema = EMA(0.05)

    def load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                self.stage_idx = int(d.get("stage_idx", 0))
                self.stage_started_at = float(d.get("stage_started_at", time.time()))
            except Exception:
                pass

    def save(self):
        try:
            self.path.write_text(json.dumps({
                "stage_idx": self.stage_idx,
                "stage_started_at": self.stage_started_at
            }, indent=2))
        except Exception:
            pass

    def ema(self, key, alpha=0.1):
        e = self.metric_ema.get(key)
        if e is None:
            e = EMA(alpha)
            self.metric_ema[key] = e
        return e
