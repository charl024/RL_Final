# do testing stuff here
"""
TODO: Thoughts for extra cred...
- test for alpha
- Tests for max number of episodes
- Tests for max number of iterations per episode
- More measurements of performance
- More map tests?
- Visualization of testing? 
"""

import time
import numpy as np
import data as data_saver

from environment import Environment
from agent import Agent
from map_abstraction import load_bmp_to_map
from reward_strategy import *
from q_learning import train_q_learning
from sarsa import train_sarsa

TARGET_POSITION = (21, 23)

# test_exploration_rate
EPISODES_EXPL_RATE = 10000
MAX_STEPS_EXPL_RATE = 200
# test_discount_value
EPISODES_DISCOUNT_VALUE = 10000
MAX_STEPS_DISCOUNT_VALUE = 200
# test_map_complexity
EPISODES_MAP_COMP = 10000
MAX_STEPS_MAP_COMP = 200
# test_reward_strategy, both sarsa and q-learning
EPISODES_REWARD_STRAT_QLEARN = 10000
MAX_STEPS_REWARD_STRAT_QLEARN = 200
EPISODES_REWARD_STRAT_SARSA = 10000
MAX_STEPS_REWARD_STRAT_SARSA = 200

# list of reward strategies
strategies = [reward_strategy_simple, reward_strategy_distance_based, reward_strategy_distance_based_scaled,  reward_strategy_harsh_visited]

def path_stats(agent, environment,  max_path_steps=500):
    """
    Test the accuracy of the agent's learned policy by generating a path from 
    each possible initial position and checking if it reaches the target without 
    hitting obstacles.

    RETURN - (test accuracy, minimum path distance, Q1, Q2, Q3, maximum path distance) 
           - test accuracy - percentage of valid paths
           - minimum path distance - shortest path
           - Q1 - upper quartile of paths
           - Q2 - average path length
           - Q3 - lower quartile of paths
           - maximum path distance - longest path
    """
    valid_paths = 0
    total_paths = 0
    path_lengths = np.zeros(environment.height * environment.width)

    for y in range(environment.height):
        for x in range(environment.width):
            state = (x, y)

            if state == environment.target_position:
                continue

            if not environment.in_boundary(state):
                continue

            if environment.is_obstacle(state):
                continue

            total_paths += 1
            path_valid = False
            path_length = 0

            environment.reset_visited()

            for _ in range(max_path_steps):
                if state == environment.target_position:
                    path_valid = True
                    break

                action = agent.exploit(state)
                new_state, reward = environment.update(state, action)

                if new_state is None:
                    break

                # print(new_state)
                # print(reward)

                if reward <= -100:  # hit obstacle or boundary
                    #TODO: is this the best way to be doing this? 
                    path_valid = False
                    break

                state = new_state
                path_length += 1

            if path_valid:
                path_lengths[valid_paths] = path_length
                valid_paths += 1 

    path_lengths = path_lengths[:valid_paths] # Only valid paths

    test_acc = valid_paths / total_paths

    if (path_lengths.size == 0):
        # path_lengths emtpy, do not compute min (will throw error)
        return (test_acc, 0, 0, 0, 0, 0, 0)
    avg_dist = np.mean(path_lengths)
    min_dist = min(path_lengths)
    q1_dist, med_dist, q3_dist = np.percentile(path_lengths, [25, 50, 75])
    max_dist = max(path_lengths)

    return (test_acc, avg_dist, min_dist, q1_dist, med_dist, q3_dist, max_dist)


def test_acc(agent, environment,  max_path_steps=500):
    """
    Test the accuracy of the agent's learned policy by generating a path from 
    each possible initial position and checking if it reaches the target without 
    hitting obstacles.

    THIS IS NOT USED CURRENTLY (using path_stats() instead)

    RETURN - test accuracy (percentage of valid paths)
    """
    valid_paths = 0
    total_paths = 0

    for y in range(environment.height):
        for x in range(environment.width):
            state = (x, y)

            if state == environment.target_position:
                continue

            if not environment.in_boundary(state):
                continue

            if environment.is_obstacle(state):
                continue

            total_paths += 1
            path_valid = False

            environment.reset_visited()

            for _ in range(max_path_steps):
                if state == environment.target_position:
                    path_valid = True
                    break

                action = agent.exploit(state)
                new_state, reward = environment.update(state, action)

                if new_state is None:
                    break

                # print(new_state)
                # print(reward)

                if reward <= -100:  # hit obstacle or boundary
                    #TODO: is this the best way to be doing this? 
                    path_valid = False
                    break

                state = new_state

            if path_valid:
                valid_paths += 1
    
    return valid_paths / total_paths

def eval_performance(environment, training_funct, kwargs):
    """
    Evaluating:
    - The time cost of a learning task.
    - The number of episodes investigated in a learning process.
    - The test accuracy that is evaluated by considering all possible initial 
      positions and generating a path from each of them using the obtained 
      feedback policy. Such a path is valid when it reaches the target position 
      without hitting any obstacle, otherwise it is invalid. Then the test 
      accuracy is the percentage of the valid paths.

    training_funct - the function to call to train the model (e.g. train_sarsa, train_q_learning)
    kwargs - the keyword arguments to pass to the function (e.g. environment, agent, ...)

    RETURN - (time cost, number of episodes, test accuracy)
    TODO: ideas for other crriteria to evaluate:
    - length of valid path / manhattan distance to target (averaged over all initial positions)
    """
    # Random number generator with fixed seed for reproducibility
    rng = np.random.default_rng(seed=123)

    # New agent
    agent = Agent(environment=environment, rng=rng)

    start_time = time.perf_counter()
    steps = training_funct(agent=agent, **kwargs)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    path_stats_tuple = path_stats(agent, environment)

    # acc = test_acc(agent, environment)

    return (total_time, steps, *path_stats_tuple)

def dummy_training_function(agent,
                            episodes, 
                            rng,
                            epsilon=0.5, 
                            gamma=0.5,
                            alpha=0.1,
                            max_steps=200,
                            ):
    return 1

def create_environments(maps, strategy=reward_strategy_simple, root_bmp_path="./map_bmps/"):
    target_position = TARGET_POSITION

    map_abstractions = [load_bmp_to_map(root_bmp_path + m + ".bmp") for m in maps]

    environments = [Environment(map_abstraction=m, 
                                reward_strategy=strategy,
                                target_position=target_position) 
                    for m in map_abstractions]
    
    # for env in environments:
    #     env.plot()
    return environments


"""
PROBLEM 6 PART 1
"""
def test_map_complexity(maps=["map1", "map2", "map3", "map4"]):
    """
    compare the performance of SARSA & Q-learning with the same hyperparameters 
    on the 4 abstractions

    This comparison shows the capability of the two 
    learning processes in handling maps of different complexities.

    maps - those maps to test
    RETURN - (sarsa_dict, q_learn_dict) where each dict maps map name to (time 
             cost, number of episodes, test accuracy)
    """

    # hyperparameters for both algorithms
    kwargs = {
        "episodes": EPISODES_MAP_COMP, # max number of episodes
        "rng": np.random.default_rng(seed=123),
        "epsilon": 0.5,
        "gamma": 0.5,
        "alpha": 0.5,
        "max_steps": MAX_STEPS_MAP_COMP, # max size of an episode
    }

    environments = create_environments(maps=maps)

    sarsa_dict = {}
    q_learn_dict = {}
    
    for i, env in enumerate(environments):
        map = maps[i]

        print(f"Testing {map}...")

        # Track performance of SARSA on this map
        sarsa_dict[map] = eval_performance(
            environment=env,
            training_funct=train_sarsa,
            kwargs=kwargs
        )

        # Track performance of Q-Learning on this map
        q_learn_dict[map] = eval_performance(
            environment=env,
            training_funct=train_q_learning,
            kwargs=kwargs,
        )

    return sarsa_dict, q_learn_dict

"""
PROBLEM 6 PART 2
"""
def test_exploration_rate(map=["map4"]):
    """
    Exploration Rate. Let us choose the abstraction of Map 4 and evaluate the
    performance of SARSA with 3 exploration probabilities: ε = 0, 0.5, 1. Do the
    same comparison for Q-learning. You may use gamma = 0.5.

    maps - those maps to test
    RETURN - (sarsa_dict, q_learn_dict) where each dict maps (map, exploration) 
             rate to (time cost, number of episodes, test accuracy)
    """
    # hyperparameters for both algorithms
    kwargs = {
        "episodes": EPISODES_EXPL_RATE, # max number of episodes
        "rng": np.random.default_rng(seed=123),
        "gamma": 0.5,
        "alpha": 0.5,
        "max_steps": MAX_STEPS_EXPL_RATE, # max size of an episode
    }

    epsilon_values = [0, 0.5, 1]

    environments = create_environments(maps=map)

    sarsa_dict = {}
    q_learn_dict = {}

    if (len(environments) > 1):
        raise ValueError("test exploration meant for a single map")
    for env in environments:
        for epsilon in epsilon_values:
            kwargs["epsilon"] = epsilon

            # Track performance of SARSA on this map
            sarsa_dict[epsilon] = eval_performance(
                environment=env,
                training_funct=train_sarsa,
                kwargs=kwargs,
            )

            # Track performance of Q-Learning on this map
            q_learn_dict[epsilon] = eval_performance(
                environment=env,
                training_funct=train_q_learning,
                kwargs=kwargs,
                )

    return sarsa_dict, q_learn_dict

"""
PROBLEM 6 PART 3
"""
def test_discount_value(maps=["map4"]):
    """
    Discount Value. We still consider the abstraction of Map 4 and evaluate the 
    performance of SARSA with 3 discount values: gamma = 0.1, 0.5, 1. Do the same 
    comparison for Q-learning. You may use ε = 0.5.

    maps - those maps to test
    RETURN - (sarsa_dict, q_learn_dict) where each dict maps (map, gamma) 
             rate to (time cost, number of episodes, test accuracy)
    """
    # hyperparameters for both algorithms
    kwargs = {
        "episodes": EPISODES_DISCOUNT_VALUE, # max number of episodes
        "rng": np.random.default_rng(seed=123),
        "epsilon": 0.5,
        "alpha": 0.5,
        "max_steps": MAX_STEPS_DISCOUNT_VALUE, # max size of an episode
    }

    gamma_values = [0.1, 0.5, 1]

    environments = create_environments(maps=maps)

    sarsa_dict = {}
    q_learn_dict = {}
    
    if (len(environments) > 1):
        raise ValueError("test exploration meant for a single map")
    for env in environments:
        for gamma in gamma_values:
            kwargs["gamma"] = gamma

            # Track performance of SARSA on this map
            sarsa_dict[gamma] = eval_performance(
                environment=env,
                training_funct=train_sarsa,
                kwargs=kwargs,
            )

            # Track performance of Q-Learning on this map
            q_learn_dict[gamma] = eval_performance(
                environment=env,
                training_funct=train_q_learning,
                kwargs=kwargs,
                )

    return sarsa_dict, q_learn_dict

"""
PROBLEM 6 PART 4
"""
def test_reward_strategy(epsilons, gammas, maps=["map4", "hi", "spiral"]):
    """
    Reward Strategy. For each learning process, please use the values of ε and 
    gamma that have the best performance in the previous comparisons. Then, 
    compare the performance of the learning process based on S1 and S2.

    epsilon - best performing epsilon value
    gamma - best performing gamma value
    maps - those maps to test

    RETURN - (sarsa_dict, q_learn_dict) where each dict maps (map, strategy) 
             rate to (time cost, number of episodes, test accuracy)
    """
    # hyperparameters for both algorithms
    kwargs_sarsa = {
        "episodes": EPISODES_REWARD_STRAT_SARSA, # max number of episodes
        "rng": np.random.default_rng(seed=123),
        "epsilon": epsilons[0],
        "alpha": 0.5,
        "gamma": gammas[0],
        "max_steps": MAX_STEPS_REWARD_STRAT_SARSA, # max size of an episode
    }

    kwargs_q = {
        "episodes": EPISODES_REWARD_STRAT_QLEARN, # max number of episodes
        "rng": np.random.default_rng(seed=123),
        "epsilon": epsilons[1],
        "alpha": 0.5,
        "gamma": gammas[1],
        "max_steps": MAX_STEPS_REWARD_STRAT_QLEARN, # max size of an episode
    }

    sarsa_dict = {}
    q_learn_dict = {}

    for strat in strategies:            
        environments = create_environments(maps=maps, strategy=strat)
        for i, env in enumerate(environments):
            map_name = maps[i]
            key = (strat.__name__, map_name)

            # Track performance of SARSA on this map
            sarsa_dict[key] = eval_performance(
                environment=env,
                training_funct=train_sarsa,
                kwargs=kwargs_sarsa,
            )

            # Track performance of Q-Learning on this map
            q_learn_dict[key] = eval_performance(
                environment=env,
                training_funct=train_q_learning,
                kwargs=kwargs_q,
                )

    return sarsa_dict, q_learn_dict

def get_best_acc(dict):
    """
    Get the best performing hyperparameter (for tests 2 and 3)
    """
    best_acc = 0
    best_param_val = 0

    for key, value in dict.items():
        acc = value[2]
        if acc > best_acc:
            best_acc = acc
            best_param_val = key

    return (best_param_val, best_acc)

def print_results(title, sarsa_dict, q_dict):
    print(title)
    print("-" * 70)

    print("SARSA:")
    for key, value in sarsa_dict.items():
        time_cost, steps, acc, avg_dist, min_dist, q1_dist, med_dist, q3_dist, max_dist = value
        print(f"""\t\t{key}: ({time_cost:.4f}s, {steps} steps, {acc * 100:.3f}%, 
              \t    avg path distance: {avg_dist:.4f}, 
              \t    min: {min_dist}, q1: {q1_dist}, median: {med_dist}, q3: {q3_dist}, max: {max_dist})""")

    print()

    print("Q-Learning:")
    for key, value in q_dict.items():
        time_cost, steps, acc, avg_dist, min_dist, q1_dist, med_dist, q3_dist, max_dist = value
        print(f"""\t\t{key}: ({time_cost:.4f}s, {steps} steps, {acc * 100:.3f}%, 
              \t    avg path distance: {avg_dist:.4f}, 
              \t    min: {min_dist}, q1: {q1_dist}, median: {med_dist}, q3: {q3_dist}, max: {max_dist})""")

    print()

    sarsa_best_key, sarsa_best_acc = get_best_acc(sarsa_dict)
    q_best_key, q_best_acc = get_best_acc(q_dict)

    print()
    print(f"\tSARSA best: ({sarsa_best_key}, {sarsa_best_acc * 100:.3f}%)")
    print(f"\tQ-Learning best: ({q_best_key}, {q_best_acc * 100:.3f}%)")
    print()
    print("-" * 70)

if __name__ == "__main__":
    print("Starting!")
    #TODO: progress bar

    maps = ["map1", "map2", "map3", "map4", "hi", "spiral"]

    print("Testing Map Complexity...")
    sarsa_map_dict, q_map_dict = test_map_complexity(maps=maps)
    print_results("Map Complexity Results", sarsa_map_dict, q_map_dict)

    print("Testing Exploration Rate...")
    sarsa_epsilon_dict, q_epsilon_dict = test_exploration_rate()
    sars_exp_rate, _ = get_best_acc(sarsa_epsilon_dict)
    q_exp_rate, _ = get_best_acc(q_epsilon_dict)
    print_results("Exploration Rate Results", sarsa_epsilon_dict, q_epsilon_dict)

    print("Testing Discount Value...")
    sarsa_gamma_dict, q_gamma_dict = test_discount_value()
    sars_disc, _ = get_best_acc(sarsa_gamma_dict)
    q_disc, _ = get_best_acc(q_gamma_dict)
    print_results("Discount Value Results", sarsa_gamma_dict, q_gamma_dict)

    print("Testing Reward Strategy...")
    sarsa_strat_dict, q_strat_dict = test_reward_strategy(
        epsilons=(sars_exp_rate, q_exp_rate),
        gammas=(sars_disc, q_disc),
    )
    print_results("Reward Strategy Results", sarsa_strat_dict, q_strat_dict)

    data_saver.save_data(sarsa_map_dict=sarsa_map_dict,
                         q_map_dict=q_map_dict,
                         sarsa_epsilon_dict=sarsa_epsilon_dict,
                         q_epsilon_dict=q_epsilon_dict,
                         sarsa_gamma_dict=sarsa_gamma_dict,
                         q_gamma_dict=q_gamma_dict,
                         sarsa_strat_dict=sarsa_strat_dict,
                         q_strat_dict=q_strat_dict)
    
    # for dict in data_saver.load_data():
    #     print(dict)