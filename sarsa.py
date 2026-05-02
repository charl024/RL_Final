# put sarsa related functions here
import numpy as np

def explore_or_exploit(agent, state, epsilon, rng):
    # explore with probability epsilon, otherwise exploit
    if rng.random() < epsilon:
        print(f"Exploring: Chose random action for state {state}")
        return agent.explore(state)
    else:
        return agent.exploit(state)
    
def train_sarsa(environment, agent, reward_strategy, episodes):
    pass