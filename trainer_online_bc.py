import time, torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset

def train_step(snapshot, model, opt, batch=256, epochs=2):
    obs, act, _, _ = snapshot
    if len(obs) < batch: return None
    ds = TensorDataset(torch.from_numpy(obs), torch.from_numpy(act))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)
    tot=0.0; n=0
    for _ in range(epochs):
        for o,a in dl:
            pred = model(o)
            loss = ((pred - a)**2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
    return tot/max(n,1)

def online_loop(buffer, model, optimizer, out_path="necto-model.pt", step_every=3.0):
    while True:
        time.sleep(step_every)
        snap = buffer.snapshot()
        loss = train_step(snap, model, optimizer)
        if loss is None: continue
        scripted = torch.jit.trace(model, torch.randn(1, snap[0].shape[1]))
        scripted.save(out_path)
        print(f"[online_bc] loss={loss:.4f} saved {out_path}")
