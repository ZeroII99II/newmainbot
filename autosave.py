# autosave.py — periodic model saver + checkpoint rotation
import os, time, datetime, threading, shutil
from pathlib import Path

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

class ModelAutoSaver:
    def __init__(self, policy, interval_sec=300, checkpoint_dir="checkpoints", max_keep=12):
        """
        policy: LivePolicy instance with .model (TorchScript) and .path (str)
        interval_sec: how often to save (default 5 minutes)
        checkpoint_dir: folder for timestamped backups
        max_keep: how many timestamped backups to keep
        """
        self.policy = policy
        self.interval = max(60, int(interval_sec))
        self.ckpt_dir = Path(checkpoint_dir)
        self.max_keep = max_keep
        self.stop_flag = False
        self._thread = None

    def start(self):
        if not TORCH_OK:
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_flag = True

    def _rotate(self):
        files = sorted(self.ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        while len(files) > self.max_keep:
            try:
                files[0].unlink(missing_ok=True)
            except Exception:
                pass
            files = files[1:]

    def _run(self):
        last_saved_to = None
        while not self.stop_flag:
            time.sleep(self.interval)
            try:
                model = getattr(self.policy, "model", None)
                target = getattr(self.policy, "path", None)
                if not (TORCH_OK and model is not None and target):
                    continue

                # 1) Save/refresh the live working model
                try:
                    torch.jit.save(model, target)
                    print(f"[autosave] saved live model -> {target}")
                    last_saved_to = target
                except Exception as e:
                    print(f"[autosave] live save failed: {e}")

                # 2) Also write a timestamped checkpoint
                ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                ck = self.ckpt_dir / f"destroyer_{ts}.pt"
                try:
                    # Reuse the just-written file for speed (copy) if possible
                    if last_saved_to and Path(last_saved_to).exists():
                        shutil.copy(last_saved_to, ck)
                    else:
                        torch.jit.save(model, ck)
                    print(f"[autosave] checkpoint -> {ck}")
                except Exception as e:
                    print(f"[autosave] checkpoint failed: {e}")

                self._rotate()
            except Exception:
                pass
