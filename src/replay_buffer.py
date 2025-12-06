import random
from collections import deque
from typing import Deque, Tuple, List

import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer
    transition: (state, action, reward, next_state, done)
    """
    def __init__(self, capacity: int):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),  # done을 float으로 (0/1)
        )

    def __len__(self):
        return len(self.buffer)
