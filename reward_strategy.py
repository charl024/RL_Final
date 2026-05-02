# define reward strategies here, which return a "reward" value

def reward_strategy_simple(event):
    if event == "target":
        return 100
    if event in ["obstacle", "boundary"]:
        return -100
    return -1

def reward_strategy_distance_based(event):
    if event == "target":
        return 100
    if event in ["obstacle", "boundary"]:
        return -100
    # negative reward based on distance to target (manhattan distance? euclidean distance?)
    return -event