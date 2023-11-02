from collections import deque
import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, s_next):
        experience = (s, a, r, s_next)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            minibatch = random.sample(self.buffer, self.count)
        else:
            minibatch = random.sample(self.buffer, batch_size)
        state = [d[0] for d in minibatch]
        action = [d[1] for d in minibatch]
        reward = [d[2] for d in minibatch]
        next_state = [d[3] for d in minibatch]
        return state, action, reward, next_state

    def clear(self):
        self.buffer.clear()
        self.count = 0
