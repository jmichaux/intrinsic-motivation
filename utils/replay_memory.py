# from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

from collections import namedtuple
import random

class ReplayMemory(object):
    def __init__(self, capacity):
        self.transition = namedtuple('transition',
                                     ('state', 'action', 'reward', 'next_state', 'done'))
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
