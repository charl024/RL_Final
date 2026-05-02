# do testing stuff here

import time

from environment import Environment
from map_abstraction import load_bmp_to_map
from reward_strategy import reward_strategy_simple

TARGET_POSITION = (21, 23)

def test_map_complexity(root_bmp_path="./map_bmps/"):
    """
    compare the performance of SARSA & Q-learning with the same hyperparameters 
    on the 4 abstractions

    This comparison shows the capability of the two 
    learning processes in handling maps of different complexities.
    """
    epsilon =  0.5
    gamma = 0.5

    target_position = TARGET_POSITION

    maps = ["map1", "map2", "map3", "map4", "hi", "spiral"]

    map_abstractions = [load_bmp_to_map(root_bmp_path + m + ".bmp") for m in maps]


    environments = [Environment(map_abstraction=m, 
                                reward_strategy=reward_strategy_simple, 
                                target_position=target_position) 
                    for m in map_abstractions]
    
    for env in environments:
        env.plot()
    # sarsa_time = 0
    # q_learning_time = 0
    
    # for i, env in enumerate(environments):
        
    #     # SARSA TESTING
    #     start_sarsa = time.perf_counter()
    #     #TODO: run SARSA
    #     end_sarsa = time.perf_counter()

    #     sarsa_time = end_sarsa - start_sarsa

    #     # Q-LEARNING TESTING
    #     start_q_learning = time.perf_counter()
    #     #TODO: run Q-learning
    #     end_q_learning = time.perf_counter()

    #     q_learning_time = end_q_learning - start_q_learning

    # return sarsa_time, q_learning_time

def run_q_learning():
    pass

def run_sarsa():
    pass

if __name__ == "__main__":
    print("Starting!")
    #TODO: progress bar
    root_bmp_path = "./map_bmps/"
    bmp_path_one = f"{root_bmp_path}/SOMETHINGHERE"

    print(test_map_complexity(root_bmp_path))
