# do testing stuff here

import time

from environment import Environment
from agent import Agent
from map_abstraction import load_bmp_to_map
from reward_strategy import reward_strategy_simple

TARGET_POSITION = (21, 23)

def eval_performance(training_funct, kwargs):
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
    """
    
    start_time = time.perf_counter()
    training_funct(**kwargs)
    end_time = time.perf_counter()

    total_time = end_time - start_time

    return total_time, 0, 0

def dummy_training_function(agent, 
                            reward_strategy, 
                            max_episodes, 
                            episode_length, 
                            epsilon, 
                            gamma, 
                            alpha):
    pass


def test_map_complexity(maps=["map1", "map2", "map3", "map4"], root_bmp_path="./map_bmps/"):
    """
    compare the performance of SARSA & Q-learning with the same hyperparameters 
    on the 4 abstractions

    This comparison shows the capability of the two 
    learning processes in handling maps of different complexities.
    """
    epsilon =  0.5
    gamma = 0.5

    target_position = TARGET_POSITION

    map_abstractions = [load_bmp_to_map(root_bmp_path + m + ".bmp") for m in maps]


    environments = [Environment(map_abstraction=m, 
                                reward_strategy=reward_strategy_simple, 
                                target_position=target_position) 
                    for m in map_abstractions]
    
    # agents = [Agent(environment=env, rng=np.random.default_rng(seed=123)) for env in environments]

    performance_dict = {}
    performance = {}
    
    for i, env in enumerate(environments):
        map = maps[i]
        print(f"Testing {map}...")

        performance_dict[map] = eval_performance(
            training_funct=dummy_training_function, 
            kwargs={
                "agent": None,
                "reward_strategy": reward_strategy_simple,
                "max_episodes": 1000,
                "episode_length": 100,
                "epsilon": epsilon,
                "gamma": gamma,
                "alpha": 0.1,
            }
        )

    return performance_dict

def run_q_learning():
    pass

def run_sarsa():
    pass

if __name__ == "__main__":
    print("Starting!")
    #TODO: progress bar
    root_bmp_path = "./map_bmps/"
    bmp_path_one = f"{root_bmp_path}/SOMETHINGHERE"

    print(test_map_complexity(maps=["map1", "map2", "map3", "map4", "hi", "spiral"], root_bmp_path=root_bmp_path))
