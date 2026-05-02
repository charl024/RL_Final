# put sarsa related functions here
import numpy as np

from environment import Environment
from agent import Agent

def explore_or_exploit(agent, state, epsilon, rng):
    # explore with probability epsilon, otherwise exploit
    if rng.random() < epsilon:
        # print(f"Exploring: Chose random action for state {state}")
        return agent.explore(state)
    else:
        return agent.exploit(state)
    
def train_sarsa(
    agent, 
    episodes, 
    rng,
    epsilon=0.5, 
    gamma=0.5,
    alpha=0.1,
    initial_state=(0,0),
    max_steps=200):
    
    for episode in range(episodes):
        agent.environment.reset_visited()      
        # init S  
        state = initial_state
        # choose A from S
        _, _, action = explore_or_exploit(agent=agent, 
                                          state=state, 
                                          epsilon=epsilon, 
                                          rng=rng)

        # iterate through steps
        for step in range(max_steps):

            new_state, reward, _ = agent.take_action(state, action)

            if new_state is None:
                continue

            # choose A' using S'
            _, _, new_action = explore_or_exploit(
                                          agent=agent, 
                                          state=new_state, 
                                          epsilon=epsilon, 
                                          rng=rng)

            x, y = state
            xp, yp = new_state
            qsa = agent.q_table[y, x, action]
            qsap = agent.q_table[yp, xp, new_action]
            agent.q_table[y, x, action] = qsa + alpha * (reward + gamma * qsap - qsa)

            state = new_state
            action = new_action

            if new_state == agent.environment.target_position:
                break
            

# from map_abstraction import load_bmp_to_map
# from reward_strategy import *

# target_position = (30, 30)

# initial_state = (1, 1)

# rng = np.random.default_rng(seed=1)

# env = Environment(map_abstraction=load_bmp_to_map("./map_bmps/map1.bmp"),
#                   target_position=target_position,
#                   reward_strategy=reward_strategy_distance_based)

# agent = Agent(env, rng=rng)

# train_sarsa(agent=agent, 
#                  episodes=1000,
#                  max_steps=200,
#                  rng=rng,
#                  initial_state=initial_state,
#                  epsilon=0.5,
#                  gamma=0.5)

# def run_episode(agent, initial_state, max_steps=200):
#     state = initial_state

#     for step in range(max_steps):
#         _, _, action = agent.exploit(state)
#         new_state, reward = agent.environment.update(state, action)

#         if new_state is None:
#             continue

#         print(f"step {step}: {state} -> {new_state}")
#         state = new_state

#         if state == agent.environment.target_position:
#             print("target reached!")
#             break

# run_episode(agent=agent, initial_state=initial_state)