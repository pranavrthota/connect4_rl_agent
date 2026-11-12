"""Fixed-capacity replay buffer of self-play examples, with augmentation:
each Connect Four position is added along with its horizontal mirror,
since the game is symmetric under left-right flip — free 2x data.
"""
import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add_game(self, examples):
        for state, policy, value in examples:
            self.buffer.append((state, policy, value))
            mirrored_state = state[:, :, ::-1].copy()
            mirrored_policy = policy[::-1].copy()
            self.buffer.append((mirrored_state, mirrored_policy, value))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.stack([b[0] for b in batch])
        policies = np.stack([b[1] for b in batch])
        values = np.array([b[2] for b in batch], dtype=np.float32)
        return (
            torch.tensor(states, dtype=torch.float32, device=device),
            torch.tensor(policies, dtype=torch.float32, device=device),
            torch.tensor(values, dtype=torch.float32, device=device),
        )
