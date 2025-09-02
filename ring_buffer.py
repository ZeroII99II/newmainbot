import multiprocessing as mp
import numpy as np

class RingBuffer:
    def __init__(self, capacity:int, obs_dim:int, act_dim:int):
        self.capacity = capacity
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.obs = mp.Array('d', capacity*obs_dim, lock=False)
        self.act = mp.Array('d', capacity*act_dim, lock=False)
        self.rew = mp.Array('d', capacity, lock=False)
        self.done = mp.Array('b', capacity, lock=False)
        self.ptr  = mp.Value('i', 0)
        self.full = mp.Value('b', 0)

    def push(self, o,a,r,d):
        i = self.ptr.value
        self.obs[i*self.obs_dim:(i+1)*self.obs_dim] = o
        self.act[i*self.act_dim:(i+1)*self.act_dim] = a
        self.rew[i] = r; self.done[i] = d
        self.ptr.value = (i+1) % self.capacity
        if self.ptr.value == 0: self.full.value = 1

    def snapshot(self):
        n = self.capacity if self.full.value else self.ptr.value
        o = np.frombuffer(self.obs, dtype=np.float64)[:n*self.obs_dim].reshape(n, self.obs_dim)
        a = np.frombuffer(self.act, dtype=np.float64)[:n*self.act_dim].reshape(n, self.act_dim)
        r = np.frombuffer(self.rew, dtype=np.float64)[:n]
        d = np.frombuffer(self.done, dtype=np.int8)[:n].astype(np.float32)
        return o.astype(np.float32), a.astype(np.float32), r.astype(np.float32), d
