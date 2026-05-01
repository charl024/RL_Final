# define reward strategies here, which return a "reward" value

def reward_strategy_simple(event):
    if event == "target":
        return 100
    if event in ["obstacle", "boundary"]:
        return -100
    return -1