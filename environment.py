# Environment class

import matplotlib.pyplot as plt
import numpy as np

class Environment():
    # map_abstraction - 2d array
    # target position - position in map, tuple
    # reward_strategy - some reward function
    def __init__(self, map_abstraction, target_position, reward_strategy):
        self.map = map_abstraction
        self.target_position = target_position
        self.reward_strategy = reward_strategy
        self.num_obstacles = np.sum(map_abstraction == 0)

        # print(self.num_obstacles)

        self.height = self.map.shape[0]
        self.width  = self.map.shape[1]

        # An array of all valid starting positions
        valid_states = []
        for y in range(self.height):
            for x in range(self.width):
                if (((x, y) == self.target_position) or (self.map[y][x] == 0)):
                    #invalid starting position
                    continue
                else:
                    valid_states.append((x, y))
        self.valid_states = valid_states

        self.visited_map = np.zeros(shape=(self.height, self.width))

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

    def in_boundary(self, state):
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, state):
        x, y = state
        return self.map[y][x] == 0
    
    def is_visited(self, state):
        x, y = state
        return self.visited_map[y][x] == 1

    # state - position in map
    # action - some way to choose a specific action from actions (integer value, idx)
    # reward_strategy - some reward function we can define 
    def update(self, state, action):
        x, y = state
        if (x < 0 or 
            x >= self.width or 
            y < 0 or 
            y >= self.height):
            # raise ValueError(f"Invalid state: {state} is out of bounds for map of shape {self.map.shape}")
            return None, None
        if (action < 0 or action >= len(self.actions)):
            # raise ValueError(f"Invalid action: {action} not in range [0, {len(self.actions) - 1}]")
            return None, None

        # define new state w action
        dx, dy = self.actions[action]
        nx = x + dx
        ny = y + dy

        self.visited_map[y][x] = 1
        
        if (not self.in_boundary((nx, ny))):
            # out of bounds, return current state and some negative reward
            reward = self.reward_strategy("boundary")
            # print("boundary hit")
            return (x, y), reward

        elif (self.is_obstacle((nx, ny))):
            # check obstacle: 0 value in map means obstacle
            reward = self.reward_strategy("obstacle")
            # print("obstacle hit")
            return (x, y), reward

        elif (nx, ny) == self.target_position:
            # check target reached
            reward = self.reward_strategy("target")
            # print("target hit")
        
        elif (self.is_visited((nx, ny))):
            reward = self.reward_strategy("visited")

        else:
            tx, ty = self.target_position
            dist = abs(nx - tx) + abs(ny - ty)
            reward = self.reward_strategy(dist)
        
        return (nx, ny), reward

    def reset_visited(self):
        self.visited_map = np.zeros(shape=(self.height, self.width))

    def plot(self):
        plt.imshow(self.map, cmap="gray_r")

        target_x, target_y = self.target_position
        plt.scatter(target_x, target_y, c="red", label="target")

        plt.title("Environment Map")
        plt.show()

# import map_abstraction
# import reward_strategy
# import numpy as np
# env = Environment(map_abstraction=map_abstraction.load_bmp_to_map("./map_bmps/map1.bmp"), target_position=(25,5), reward_strategy=reward_strategy.reward_strategy_simple)
# # env = Environment(map_abstraction=np.zeros((41, 40)), target_position=(25,5), reward_strategy=reward_strategy.reward_strategy_simple)
# print(env)
# print(env.valid_states)
# # env.plot()

# # (x, y)
# # (0, 1),  # South, 0, Down
# # (1, 0),  # East,  1, Right
# # (0, -1), # North, 2, Up
# # (-1, 0), # West,  3, Left

# # "barier"
# print("barrier")
# print(env.update((0,0),  3)) # Left out of bound
# print(env.update((0,0),  2)) # Up out of bound
# print(env.update((39,39),  1)) # Right out of bound
# print(env.update((39,39),  0)) # Bottom out of bounds
# print(env.update((4,1), 0)) # obstacle

# # "move"
# print("move")
# print(env.update((0,0),  1))
# print(env.update((25,4), 1))

# # "target"
# print("target")
# print(env.update((24,5), 1))
# print(env.update((25,6), 2))

# # Error cases
# print("error cases")

# # invalid actions
# try:    
#     print(env.update((0,0), 4)) 
# except ValueError as e: 
#     print(e)
# try:    
#     print(env.update((0,0), -1)) 
# except ValueError as e: 
#     print(e)


# # invalid start states
# try:    
#     print(env.update((40,39), 0))  
# except ValueError as e: 
#     print(e)
# try:    
#     print(env.update((39,40), 0))  
# except ValueError as e: 
#     print(e)
# try:    
#     print(env.update((0,-1),  0))    
# except ValueError as e: 
#     print(e)
# try:    
#     print(env.update((-1,0),  0))    
# except ValueError as e: 
#     print(e)
  