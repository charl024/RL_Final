# put q_learning related functions here

import numpy as np

from environment import Environment
from agent import Agent

def explore_or_exploit(agent, state, epsilon, rng):
    # explore with probability epsilon, otherwise exploit
    if rng.random() < epsilon:
        return agent.explore()
    return agent.exploit(state)


# reward strategy, environment passed in implicitly through agent object
def train_q_learning(
    agent, 
    episodes, 
    rng,
    epsilon=0.5, 
    gamma=0.5,
    alpha=0.1,
    initial_state=(0,0),
    max_steps=200):

    step_count = 0
    
    for episode in range(episodes):
        agent.environment.reset_visited()           
        state = initial_state

        # iterate through steps
        for step in range(max_steps):
            step_count += 1
            # compute R, S'
            action = explore_or_exploit(
                agent=agent, 
                state=state, 
                epsilon=epsilon, 
                rng=rng
            )

            new_state, reward = agent.take_action(state, action)

            if (new_state is None and reward is None):
                break
            
            # parse S', and S
            x, y = state
            xp, yp = new_state
            # get Q(S, A)
            qsa = agent.q_table[y, x, action]
            # get max_a(S', a)
            max_a = np.max(agent.q_table[yp, xp])
            # update Q(S, A)
            agent.q_table[y, x, action] = qsa + alpha * (reward + gamma * max_a - qsa)
            # S <- S'
            state = new_state

            if new_state == agent.environment.target_position:
                break
    return step_count

# from map_abstraction import load_bmp_to_map
# from reward_strategy import reward_strategy_simple, reward_strategy_distance_based

# target_position = (30, 30)

# initial_state = (1, 1)

# rng = np.random.default_rng(seed=1)

# env = Environment(map_abstraction=load_bmp_to_map("./map_bmps/map1.bmp"),
#                   target_position=target_position,
#                   reward_strategy=reward_strategy_simple)

# agent = Agent(env, rng=rng)

# train_q_learning(agent=agent, 
#                  episodes=10000, 
#                  rng=rng,
#                  initial_state=initial_state)

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