# do testing stuff here

import time
import numpy as np

from environment import Environment
from agent import Agent
from map_abstraction import load_bmp_to_map
from reward_strategy import reward_strategy_simple

TARGET_POSITION = (21, 23)

def test_acc(agent, environment):
    """
    Test the accuracy of the agent's learned policy by generating a path from 
    each possible initial position and checking if it reaches the target without 
    hitting obstacles.

    RETURN - test accuracy (percentage of valid paths)
    """
    valid_paths = 0
    total_paths = environment.width * environment.height

    for y in range(environment.height):
        for x in range(environment.width):
            state = (x, y)
            if (state == environment.target_position or 
                environment.in_boundary(state)):
                # skip target position and obstacles
                continue
            path_valid = True

            while state != environment.target_position:
                new_state, reward, _ = agent.exploit(state)

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
    num_episodes = training_funct(agent=agent, **kwargs)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    acc = test_acc(agent, environment)

    return total_time, num_episodes, acc

def dummy_training_function(agent,
                            episodes, 
                            rng,
                            epsilon=0.5, 
                            gamma=0.5,
                            alpha=0.1,
                            max_steps=200,
                            ):
    return 1

def create_environments(maps, root_bmp_path="./map_bmps/"):
    target_position = TARGET_POSITION

    map_abstractions = [load_bmp_to_map(root_bmp_path + m + ".bmp") for m in maps]

    environments = [Environment(map_abstraction=m, 
                                reward_strategy=reward_strategy_simple, #TODO: are we happy with this reward strat?
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
        "episodes": 1000, # max number of episodes
        "rng": np.random.default_rng(seed=123),
        "epsilon": 0.5,
        "gamma": 0.5,
        "alpha": 0.5,
        "max_steps": 100, # max size of an episode
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
            training_funct=dummy_training_function, #TODO: replace with train_sarsa
            kwargs=kwargs
        )

        # Track performance of Q-Learning on this map
        q_learn_dict[map] = eval_performance(
            environment=env,
            training_funct=dummy_training_function, #TODO: replace with train_q_learning
            kwargs=kwargs,
        )

    return sarsa_dict, q_learn_dict

"""
PROBLEM 6 PART 2
"""
def test_exploration_rate(maps=["map1", "map2", "map3", "map4"]):
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
        "episodes": 1000, # max number of episodes
        "rng": np.random.default_rng(seed=123),
        "gamma": 0.5,
        "alpha": 0.5,
        "max_steps": 100, # max size of an episode
    }

    epsilon_values = [0, 0.5, 1]

    environments = create_environments(maps=maps)

    sarsa_dict = {}
    q_learn_dict = {}
    
    for i, env in enumerate(environments):
        for epsilon in epsilon_values:
            kwargs["epsilon"] = epsilon
            map = maps[i]
            key = (map, epsilon)

            # Track performance of SARSA on this map
            sarsa_dict[key] = eval_performance(
                environment=env,
                training_funct=dummy_training_function, #TODO: replace with train_sarsa
                kwargs=kwargs,
            )

            # Track performance of Q-Learning on this map
            q_learn_dict[key] = eval_performance(
                environment=env,
                training_funct=dummy_training_function, #TODO: replace with train_q_learning
                kwargs=kwargs,
                )

    return sarsa_dict, q_learn_dict

"""
PROBLEM 6 PART 3
"""
def test_discount_value(maps=["map1", "map2", "map3", "map4"]):
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
        "episodes": 1000, # max number of episodes
        "rng": np.random.default_rng(seed=123),
        "epsilon": 0.5,
        "alpha": 0.5,
        "max_steps": 100, # max size of an episode
    }

    gamma_values = [0.1, 0.5, 1]

    environments = create_environments(maps=maps)

    sarsa_dict = {}
    q_learn_dict = {}
    
    for i, env in enumerate(environments):
        for gamma in gamma_values:
            kwargs["gamma"] = gamma
            map = maps[i]
            key = (map, gamma)

            # Track performance of SARSA on this map
            sarsa_dict[key] = eval_performance(
                environment=env,
                training_funct=dummy_training_function, #TODO: replace with train_sarsa
                kwargs=kwargs,
            )

            # Track performance of Q-Learning on this map
            q_learn_dict[key] = eval_performance(
                environment=env,
                training_funct=dummy_training_function, #TODO: replace with train_q_learning
                kwargs=kwargs,
                )

    return sarsa_dict, q_learn_dict

def run_q_learning():
    pass

def run_sarsa():
    pass

if __name__ == "__main__":
    print("Starting!")
    #TODO: progress bar
    root_bmp_path = "./map_bmps/"
    bmp_path_one = f"{root_bmp_path}/SOMETHINGHERE"

    maps = ["map1", "map2", "map3", "map4", "hi", "spiral"]

    print("Testing Map Complexity...")
    print(test_map_complexity(maps=maps))

    print("Testing Exploration Rate...")
    print(test_exploration_rate(maps=maps))

    print("Testing Discount Value...")
    print(test_discount_value(maps=maps))
