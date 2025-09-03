import numpy as np

class RingBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.o = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros((capacity,), dtype=np.float32)
        self.d = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def push(self, obs, act, rew, done):
        i = self.ptr
        self.o[i] = obs
        self.a[i] = act
        self.r[i] = rew
        self.d[i] = 1.0 if done else 0.0
        self.ptr = (i + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def snapshot(self):
        n = self.capacity if self.full else self.ptr
        return (self.o[:n].copy(), self.a[:n].copy(), self.r[:n].copy(), self.d[:n].copy())
