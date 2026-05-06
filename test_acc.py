import numpy as np

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