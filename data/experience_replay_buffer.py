import random
from collections import deque

class ExperienceReplay:
    def __init__(self, max_size=2000):
        self.memory = deque(maxlen=max_size)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)
