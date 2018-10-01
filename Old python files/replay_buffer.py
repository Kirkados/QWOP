"""
This code supplies the replay buffer functionality needed for experience replay in DDPG
with help from: https://github.com/pemami4911/deep-rl/blob/master/ddpg/replay_buffer.py
@author: Kirk
"""

from collections import deque
import random
import numpy as np

class ReplayBuffer():
    
    def __init__(self, buffer_size, random_seed):
        
        # The right hand side of the double-ended queue contains the most recent experience
        
        self.buffer_size = buffer_size # total size of the buffer
        self.count = 0 # how many entries are in the queue
        self.buffer = deque() # build the double-ended queue
        random.seed(random_seed)
        
    # Adds experience to the buffer and discards old experiences if the buffer is full
    def add(self, state, action, reward, next_state, terminate):
        experience = (state, action, reward, next_state, terminate)
        
        if self.count < self.buffer_size: # if the buffer isn't full yet
            self.buffer.append(experience) # add to the right
            self.count += 1 # increment counter
        else:
            self.buffer.popleft() # remove oldest experience from the left
            self.buffer.append(experience) # add newest experience on the right
        
    # Returns how full the replay buffer is
    def size(self):
        return self.count
    
    # User requests a uniformly random sample of experiences from the batch
    def sample_batch(self, batch_size):
        batch = [] # initializing
        
        
        if self.count < batch_size: # if the buffer doesn't have enough samples as requested ()
            batch = random.sample(self.buffer, self.count)
            print('Don''t sample yet!!! Not enough samples in the buffer.')
        else: # the buffer has enough samples
            batch = random.sample(self.buffer, batch_size)
        
        # Extracting the experiences into workable arrays
        state_batch      = np.array([_[0] for _ in batch])
        action_batch     = np.array([_[1] for _ in batch])
        reward_batch     = np.array([_[2] for _ in batch])
        next_state_batch = np.array([_[3] for _ in batch])
        terminate_batch  = np.array([_[4] for _ in batch])
        
        return np.squeeze(state_batch), np.squeeze(action_batch), reward_batch, np.squeeze(next_state_batch), np.squeeze(terminate_batch)
    
    # If we want to purge the buffer
    def clear(self):
        self.buffer.clear()
        self.count = 0