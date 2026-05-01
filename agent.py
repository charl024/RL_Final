# Agent class

import numpy as np

class Agent():
    def __init__(self, width=40, height=40, n_actions=4):
        # q_table is a 3d matrix
        self.q_table = np.zeros((height, width, n_actions))
        self.n_actions = n_actions
        pass 
        
    # state - position in map, used to index into q_table
    # this may be where we use epsilon to choose exploration vs exploitation
    def choose_action(self, state):
        # explore
        # else exploit
        pass