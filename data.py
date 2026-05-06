import json

save_path = 'performance_eval/data.json'

def save_data(
    sarsa_map_dict,
    q_map_dict,
    sarsa_epsilon_dict,
    q_epsilon_dict,
    sarsa_gamma_dict,
    q_gamma_dict,
    sarsa_strat_dict,
    q_strat_dict
):
    data = {
        "sarsa_map_dict" : sarsa_map_dict,
        "q_map_dict" : q_map_dict,
        "sarsa_epsilon_dict": sarsa_epsilon_dict,
        "q_epsilon_dict" : q_epsilon_dict,
        "sarsa_gamma_dict": sarsa_gamma_dict,
        "q_gamma_dict": q_gamma_dict,
        "sarsa_strat_dict": sarsa_strat_dict,
        "q_strat_dict": q_strat_dict,
    }
    # Saving to a file
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_data():
    # Saving to a file
    with open(save_path, 'r') as f:
        data = json.load(f)

    data_list = [v for v in data.values()]

    return data_list