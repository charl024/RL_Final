# Environment class

class Environment():
    def __init__(self, map_abstraction, target_position):
        self.map_abstraction = map_abstraction
        self.target_position = target_position

        # list of actions an agent can perform in the environment
        self.actions = null
    
    # state - position in map
    # action - some way to choose a specific action from actions (idx into an array? idk)
    # reward_strategy - some reward function we can define 
    def update(self, state, action, reward_strategy):
        pass
        # return next_state, immediate_reward

    # would be helpful to plot it later for testing, adding this for now
    def visualize():
        pass