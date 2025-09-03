import time, numpy as np
try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

class OnlineBC:
    def __init__(self, buffer, policy, out_path="necto-model.pt", step_every=3.0, batch=256, lr=2e-4):
        self.buffer = buffer
        self.policy = policy
        self.out_path = out_path
        self.step_every = step_every
        self.batch = batch
        self.lr = lr
        self.stop_flag = False

    def start(self):
        import threading
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        if not TORCH_OK:
            return
        self.policy.maybe_reload()
        model = self.policy.model
        if model is None:
            return
        try:
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        except Exception:
            return

        last_save = time.time()
        while not self.stop_flag:
            time.sleep(self.step_every)
            obs, act, _, _ = self.buffer.snapshot()
            if len(obs) < self.batch:
                continue
            try:
                x = torch.from_numpy(obs).float()
                y = torch.from_numpy(act).float()
                idx = np.random.choice(len(x), size=min(self.batch, len(x)), replace=False)
                xb, yb = x[idx], y[idx]
                pred = model(xb)
                if pred.shape[-1] != 8:
                    continue
                loss = ((pred - yb) ** 2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            except Exception:
                continue

            if time.time() - last_save > self.step_every:
                try:
                    torch.jit.save(model, self.out_path)
                    last_save = time.time()
                except Exception:
                    pass

    def stop(self):
        self.stop_flag = True
