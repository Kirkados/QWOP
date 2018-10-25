"""
Generates and manages the experience replay buffer.

Incorporates priority into the sampling (http://arxiv.org/abs/1511.05952), if requested.

"""

import random
from collections import deque
import numpy as np

from settings import Settings


def ReplayBuffer(prioritized):
    if prioritized:
        return RegularReplayBuffer()
    else:
        return PrioritizedReplayBuffer()
    

class RegularReplayBuffer:
    
    def __init__(self):
        self.buffer = deque(maxlen = Settings.REPLAY_BUFFER_SIZE) # Initialize the buffer
        
    # Query how many entries are in the buffer
    def how_filled(self):
        return len(self.buffer)
    
    # Add new experience to the buffer
    def add(self, experience):
        self.buffer.append(experience)
        
    # Randomly sample data from the buffer
    def sample(self, beta = None):
        batch_size = min(Settings.MINI_BATCH_SIZE, len(self.buffer)) # maybe the buffer doesn't contain the requested number of samples yet
        return random.sample(self.buffer, batch_size)
    
    # Only needed for prioritized replay buffer
    def update(self, index, errors):
        pass
        

class PrioritizedReplayBuffer():
    
    def __init__(self):
        self.buffer = SumTree(capacity = Settings.REPLAY_BUFFER_SIZE) # initialize replay buffer using the SumTree class
        
    # Add experience to the replay buffer
    def add(self, experience):
        self.buffer.add(self.buffer.max(), experience)
        
    # Sample data from replay buffer in a prioritized fashion
    def sample(self, beta):
        data, index, priorities = self.buffer.sample(Settings.MINI_BATCH_SIZE)
        probabilities = priorities / self.buffer.total()
        weights = (self.buffer.n_entries * probabilities) ** -beta
        weights /= np.max(weights)
        
        return data, index, weights
    
    # Update priorities??!?!
    def update(self, index, errors):
        priorities = (np.abs(errors) + 1e-6) ** Settings.ALPHA
        
        for i in range(len(index)):
            self.buffer.update(index[i], priorities[i])