# trainer.py — online trainer with checkpoints; works without torch (no-op)
import os, time, json, math
from pathlib import Path
import numpy as np

class NoopTrainer:
    def __init__(self, **kw):
        self.enabled = False
        self.model_dir = Path(kw.get("model_dir","models"))
        self.ckpt_dir = Path(kw.get("checkpoint_dir","checkpoints"))
        self.stage_tag = kw.get("stage_tag","Bronze")
        self.model_dir.mkdir(exist_ok=True)
        self.ckpt_dir.mkdir(exist_ok=True)
        self.last_save_t = time.time()

    def act(self, obs):  # no model; let heuristic act
        return None
    def step(self, obs, action, reward, done): pass
    def autosave_if_needed(self, steps): pass
    def save(self, suffix="manual"): pass
    def info(self): return {"training":"disabled (torch not installed)"}

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

if not TORCH_OK:
    OnlineTrainer = NoopTrainer
else:
    class TinyPolicy(nn.Module):
        def __init__(self, obs_dim, act_dim=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, act_dim), nn.Tanh()  # steer..handbrake in [-1,1]
            )
        def forward(self, x): return self.net(x)

    class Replay:
        def __init__(self, cap=50000):
            self.cap = cap
            self.buf = []
        def push(self, s, a, r, done):
            self.buf.append((s.astype(np.float32), a.astype(np.float32), float(r), bool(done)))
            if len(self.buf) > self.cap: self.buf.pop(0)
        def sample(self, n):
            import random
            n = min(n, len(self.buf))
            batch = random.sample(self.buf, n)
            s,a,r,d = zip(*batch)
            return np.stack(s), np.stack(a), np.array(r, np.float32), np.array(d, np.float32)
        def __len__(self): return len(self.buf)

    class OnlineTrainer:
        def __init__(self, obs_dim, **cfg):
            self.enabled = bool(cfg.get("enabled", True))
            self.use_model_for_action = bool(cfg.get("use_model_for_action", False))
            self.model_dir = Path(cfg.get("model_dir","models")); self.model_dir.mkdir(exist_ok=True)
            self.ckpt_dir = Path(cfg.get("checkpoint_dir","checkpoints")); self.ckpt_dir.mkdir(exist_ok=True)
            self.stage_tag = cfg.get("stage_tag","Bronze")
            self.save_every_seconds = int(cfg.get("save_every_seconds", 300))
            self.save_every_steps   = int(cfg.get("save_every_steps", 2000))
            self.last_save_t = time.time()
            self.step_count = 0

            self.device = torch.device("cpu")
            self.policy = TinyPolicy(obs_dim).to(self.device)
            self.opt = optim.Adam(self.policy.parameters(), lr=float(cfg.get("learning_rate",1e-4)))
            self.batch_size = int(cfg.get("batch_size", 256))
            self.replay = Replay(int(cfg.get("replay_capacity", 50000)))
            self.min_replay = int(cfg.get("min_replay", 2000))
            self.gamma = float(cfg.get("gamma", 0.995))

            self._latest = self.model_dir / f"destroyer_{self.stage_tag}_latest.pt"
            self._latest_opt = self.model_dir / f"destroyer_{self.stage_tag}_latest.opt"
            self._meta = self.model_dir / f"destroyer_{self.stage_tag}_meta.json"

            self._load_if_exists()

        def _load_if_exists(self):
            if self._latest.exists():
                try:
                    ckpt = torch.load(self._latest, map_location=self.device)
                    self.policy.load_state_dict(ckpt["model"])
                    if self._latest_opt.exists():
                        opt = torch.load(self._latest_opt, map_location=self.device)
                        self.opt.load_state_dict(opt["opt"])
                    print(f"[Trainer] Loaded {self._latest.name}")
                except Exception as e:
                    print(f"[Trainer] Failed to load latest: {e}")

        def act(self, obs):
            if not (self.enabled and self.use_model_for_action): return None
            with torch.no_grad():
                x = torch.from_numpy(obs).to(self.device).unsqueeze(0)
                a = self.policy(x)[0].cpu().numpy()
            # convert [-1,1] to our action layout where buttons are >=0.5 considered pressed
            a = a.astype(np.float32)
            a[5:] = (a[5:] + 1.0)/2.0  # buttons to [0,1]
            return a

        def step(self, obs, action, reward, done):
            if not self.enabled: return
            self.step_count += 1
            self.replay.push(obs, action, reward, done)
            if len(self.replay) < self.min_replay: return
            # simple supervised RL-ish: predict action that led to reward (behavior cloning with reward weight)
            s,a,r,d = self.replay.sample(self.batch_size)
            w = (np.maximum(r, 0.0) + 0.1).reshape(-1,1).astype(np.float32)
            s_t = torch.from_numpy(s)
            a_t = torch.from_numpy(a)
            w_t = torch.from_numpy(w)
            pred = self.policy(s_t)
            loss = ((pred - a_t)**2 * w_t).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()

        def autosave_if_needed(self, steps_since_start):
            t = time.time()
            if (t - self.last_save_t) >= self.save_every_seconds or (self.step_count % self.save_every_steps == 0 and self.step_count > 0):
                self.save(suffix=f"step{self.step_count}")
                self.last_save_t = t

        def save(self, suffix="manual"):
            payload = {"model": self.policy.state_dict(), "created": time.time(), "steps": self.step_count}
            latest = self._latest
            torch.save(payload, latest)
            torch.save({"opt": self.opt.state_dict()}, self._latest_opt)
            # metadata
            meta = {"stage": self.stage_tag, "steps": self.step_count, "time": time.time()}
            self._meta.write_text(json.dumps(meta, indent=2))
            # rotate checkpoint
            ts = time.strftime("%Y%m%d-%H%M%S")
            ck = self.ckpt_dir / f"destroyer_{self.stage_tag}_{ts}_{suffix}.pt"
            torch.save(payload, ck)
            print(f"[Trainer] Saved {latest.name} and checkpoint {ck.name}")

        def info(self):
            return {"training":"enabled","steps":self.step_count,"stage":self.stage_tag}
