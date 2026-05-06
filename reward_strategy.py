# define reward strategies here, which return a "reward" value

VISITED_REWARD  = -25
TARGET_REWARD   = 100
OBSTACLE_REWARD = -100
BOUNDARY_REWARD = -100
DEFAULT_REWARD  = -1

DISTANCE_SCALING_FACTOR = 0.5

def reward_strategy_simple(event):
    if event == "target":
        return TARGET_REWARD
    if event in ["obstacle", "boundary"]:
        return OBSTACLE_REWARD
    return DEFAULT_REWARD

def reward_strategy_distance_based(event):
    if event == "target":
        return TARGET_REWARD
    if event in ["obstacle", "boundary"]:
        return OBSTACLE_REWARD

    # manhattan distance
    return -event

def reward_strategy_distance_based_scaled(event):
    if event == "target":
        return TARGET_REWARD
    if event in ["obstacle", "boundary"]:
        return OBSTACLE_REWARD

    # manhattan distance
    return DEFAULT_REWARD - DISTANCE_SCALING_FACTOR * event

def reward_strategy_visited(event):
    if event == "target":
        return TARGET_REWARD
    if event in ["obstacle", "boundary"]:
        return OBSTACLE_REWARD
    if event == "visited":
        return VISITED_REWARD
    return DEFAULT_REWARD

def reward_strategy_harsh_visited(event):
    if event == "target":
        return TARGET_REWARD
    if event in ["obstacle", "boundary"]:
        return OBSTACLE_REWARD
    if event == "visited":
        return VISITED_REWARD * 10
    return DEFAULT_REWARD

reward_strategy_visited.uses_visited = True
reward_strategy_harsh_visited.uses_visited = True