"""ACT-style temporal ensembling of absolute action chunks at control ticks."""
from collections import deque

import numpy as np


class TemporalActions:
    def __init__(self, decay):
        if not 0 < decay <= 1:
            raise ValueError("Temporal ensembling decay must be in (0, 1]")
        self.decay = decay
        self.chunks = deque()
        self.next_step = 0

    def step(self, tick, actions):
        chunk = np.asarray(actions)
        if tick != self.next_step or chunk.shape != (8,8) or not np.isfinite(chunk).all():
            raise ValueError("Temporal ensembling requires one finite 8x8 chunk at every control tick")
        while self.chunks and self.chunks[0][0] + len(self.chunks[0][1]) <= tick:
            self.chunks.popleft()
        self.chunks.append((tick,chunk))
        available = np.stack([values[tick-start] for start,values in self.chunks])
        # ACT ordering: oldest prediction first. A larger decay favors older chunks;
        # it must not be interpreted as favoring the newest observation.
        weights = np.exp(-self.decay*np.arange(len(available)))
        self.next_step += 1
        return np.average(available,axis=0,weights=weights), len(available)
