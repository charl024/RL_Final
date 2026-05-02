# Agent class

import numpy as np
from environment import Environment

class Agent():
    """
    Agent class 

    self.environment - Environment object that the agent interacts with
    self.q_table - 3D numpy array representing the Q-values for each state-action pair
    self.n_actions - number of possible actions in the environment
    self.rng - random number generator for exploration
    """
    def __init__(self, environment, rng):
        # Type checking
        if not isinstance(environment, Environment):
            raise TypeError(f"Expected Environment, got {type(environment).__name__}")
        if not isinstance(rng, np.random._generator.Generator):
            raise TypeError(f"Expected Generator, got {type(rng).__name__}")
        
        
        self.environment = environment
        self.rng = rng

        n_actions = len(environment.actions)

        width = environment.width
        height = environment.height

        # q_table is a 3d matrix
        self.q_table = np.zeros((height, width, n_actions))
        self.n_actions = n_actions
        pass 

    def __repr__(self) -> str:
        # String representation of an Agent
        # TODO: this is not very readable
        string = []
        for y in range(self.q_table.shape[0]):
            for x in range(self.q_table.shape[1]):
                string.append("[")
                for a in range(self.q_table.shape[2]):
                    string.append(f"{self.q_table[y][x][a]: .0f}")
                    if a != self.q_table.shape[2] - 1:
                        string.append(",")
                string.append(f"]")
                if x != self.q_table.shape[1] - 1:
                    string.append(",")
            string.append("\n")
        return "".join(string)
    
    def take_action(self, state, action):
        # state - (x, y) position in map
        # action - integer value representing an action to take
        # RETURN - new state, reward
        new_state, reward = self.environment.update(state, action)

        x, y = state

        # Update q_table based on reward received and new state
        self.q_table[y, x, action] += reward

        return new_state, reward

    def explore(self, state):
        # randomly choose an action to perform
        # state - (x, y) position in map
        # RETURN - new state, reward
        action = self.rng.integers(self.n_actions)
        
        return self.take_action(state, action)
        
    def exploit(self, state):
        # choose the action with the highest q value for the given state
        # state - (x, y) position in map
        # RETURN - new state, reward
        action = np.argmax(self.q_table[state[1], state[0]])

        return self.take_action(state, action)

# import map_abstraction
# import reward_strategy

# strat = reward_strategy.reward_strategy_simple
# env = Environment(map_abstraction=map_abstraction.load_bmp_to_map("./map_bmps/map1.bmp"), 
#                   target_position=(25,5), 
#                   reward_strategy=strat,
#                   )

# agent = Agent(environment=env, rng=np.random.default_rng(seed=123))

# # exploit
# print("Exploiting from (0,0)")
# print(agent.exploit((0,0)))
# print(agent.exploit((0,0)))
# print(agent.exploit((0,0)))
# print(agent.exploit((0,0)))
# print(agent.exploit((0,0)))
# print(agent.exploit((0,0)))
# print(agent.exploit((0,0)))
# print(agent.exploit((0,0)))

# print("Exploiting from (26, 5)")
# print(agent.exploit((26,5)))
# print(agent.exploit((26,5)))
# print(agent.exploit((26,5)))
# print(agent.exploit((26,5)))
# print(agent.exploit((26,5)))
# print(agent.exploit((26,5)))

# # explore
# print("Exploring from (25,4)")
# print(agent.explore((25,4)))
# print(agent.explore((25,4)))
# print(agent.explore((25,4)))    
# print(agent.explore((25,4)))    
# print(agent.explore((25,4)))    
# print(agent.explore((25,4)))    
# print(agent.explore((25,4)))   

# #exploit again to see how q_table has changed
# print("Exploiting from (25,4)")
# print(agent.exploit((25,4)))
# print(agent.exploit((25,4)))
# print(agent.exploit((25,4)))
# print(agent.exploit((25,4)))

# print(agent)