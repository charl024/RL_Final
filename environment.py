# Environment class

import matplotlib.pyplot as plt

class Environment():
    # map_abstraction - 2d array
    # target position - position in map, tuple
    def __init__(self, map_abstraction, target_position):
        self.map = map_abstraction
        self.target_position = target_position

        # list of actions an agent can perform in the environment
        self.actions = [
            # (x, y)
            (0, 1),  # North, 0, Up
            (1, 0),  # East,  1, Right
            (0, -1), # South, 2, Down
            (-1, 0), # West,  3, Left
        ]
    
    # state - position in map
    # action - some way to choose a specific action from actions (integer value, idx)
    # reward_strategy - some reward function we can define 
    def update(self, state, action, reward_strategy):

        # define new state w action
        dx, dy = self.actions[action]
        nx = state[0] + dx
        ny = state[1] + dy
        
        if (nx < 0 or nx >= self.map.shape[1] or ny < 0 or ny >= self.map.shape[0]):
            # out of bounds, return current state and some negative reward
            reward = reward_strategy("boundary")
        elif (self.map[ny][nx] == 0):
            # check obstacle: 0 value in map means obstacle
            reward = reward_strategy("obstacle")
        elif (nx, ny) == self.target_position:
            # check target reached
            reward = reward_strategy("target")
        else:
            reward = reward_strategy("move")
        
        return (nx, ny), reward

    def plot(self):
        plt.imshow(self.map, cmap="gray_r")

        target_x, target_y = self.target_position
        plt.scatter(target_x, target_y, c="red", label="target")

        plt.title("Environment Map")
        plt.show()

# import map_abstraction
# import reward_strategy
# env = Environment(map_abstraction=map_abstraction.load_bmp_to_map("./map_bmps/map1.bmp"), target_position=(25,5))
# env.plot()
# print(env.update((0,0), 1,reward_strategy.reward_strategy_simple))
# print(env.update((25,4), 1,reward_strategy.reward_strategy_simple))
# print(env.update((24,5), 1,reward_strategy.reward_strategy_simple))
# print(env.update((25,6), 2,reward_strategy.reward_strategy_simple))