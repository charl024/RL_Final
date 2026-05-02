# Environment class

import matplotlib.pyplot as plt

class Environment():
    # map_abstraction - 2d array
    # target position - position in map, tuple
    # reward_strategy - some reward function
    def __init__(self, map_abstraction, target_position, reward_strategy):
        self.map = map_abstraction
        self.target_position = target_position
        self.reward_strategy = reward_strategy

        # list of actions an agent can perform in the environment
        self.actions = [
            # (x, y)
            (0, 1),  # South, 0, Down
            (1, 0),  # East,  1, Right
            (0, -1), # North, 2, Up
            (-1, 0), # West,  3, Left
        ]
    def __repr__(self) -> str:
        # String representation of an Environment
        string = []
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if (x, y) == self.target_position:
                    string.append("T")
                elif self.map[y][x] == 0:
                    string.append("X")
                else:
                    string.append(".")
            string.append("\n")
        return "".join(string)
    
    # state - position in map
    # action - some way to choose a specific action from actions (integer value, idx)
    # reward_strategy - some reward function we can define 
    def update(self, state, action):

        if (state[0] < 0 or 
            state[0] >= self.map.shape[1] or 
            state[1] < 0 or 
            state[1] >= self.map.shape[0]):
            raise ValueError(f"Invalid state: {state} is out of bounds for map of shape {self.map.shape}")
        if (action < 0 or action >= len(self.actions)):
            raise ValueError(f"Invalid action: {action} not in range [0, {len(self.actions) - 1}]")

        # define new state w action
        dx, dy = self.actions[action]
        nx = state[0] + dx
        ny = state[1] + dy
        
        if (nx < 0 or nx >= self.map.shape[1] or ny < 0 or ny >= self.map.shape[0]):
            # out of bounds, return current state and some negative reward
            reward = self.reward_strategy("boundary")
        elif (self.map[ny][nx] == 0):
            # check obstacle: 0 value in map means obstacle
            reward = self.reward_strategy("obstacle")
        elif (nx, ny) == self.target_position:
            # check target reached
            reward = self.reward_strategy("target")
        else:
            reward = self.reward_strategy("move")
        
        return (nx, ny), reward

    def plot(self):
        plt.imshow(self.map, cmap="gray_r")

        target_x, target_y = self.target_position
        plt.scatter(target_x, target_y, c="red", label="target")

        plt.title("Environment Map")
        plt.show()

import map_abstraction
import reward_strategy
env = Environment(map_abstraction=map_abstraction.load_bmp_to_map("./map_bmps/map1.bmp"), target_position=(25,5), reward_strategy=reward_strategy.reward_strategy_simple)
print(env)
# env.plot()

# (x, y)
# (0, 1),  # South, 0, Down
# (1, 0),  # East,  1, Right
# (0, -1), # North, 2, Up
# (-1, 0), # West,  3, Left

# "barier"
print("barrier")
print(env.update((0,0),  3)) # Left out of bound
print(env.update((0,0),  2)) # Up out of bound
print(env.update((39,39),  1)) # Right out of bound
print(env.update((39,39),  0)) # Bottom out of bounds
print(env.update((4,1), 0)) # obstacle

# "move"
print("move")
print(env.update((0,0),  1))
print(env.update((25,4), 1))

# "target"
print("target")
print(env.update((24,5), 1))
print(env.update((25,6), 2))

# Error cases
print("error cases")

# invalid actions
try:    
    print(env.update((0,0), 4)) 
except ValueError as e: 
    print(e)
try:    
    print(env.update((0,0), -1)) 
except ValueError as e: 
    print(e)


# invalid start states
try:    
    print(env.update((40,39), 0))  
except ValueError as e: 
    print(e)
try:    
    print(env.update((39,40), 0))  
except ValueError as e: 
    print(e)
try:    
    print(env.update((0,-1),  0))    
except ValueError as e: 
    print(e)
try:    
    print(env.update((-1,0),  0))    
except ValueError as e: 
    print(e)
  