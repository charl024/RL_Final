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
    
    temp_dict = {}
    # Preventing tuple typings 
    for key in sarsa_strat_dict:
        str1, str2 = key
        temp_dict[str2 + "_" + str1] = sarsa_strat_dict[key] #Swapping order for readability :D

    sarsa_strat_dict = temp_dict

    temp_dict = {}
    # Preventing tuple typings 
    for key in q_strat_dict:
        str1, str2 = key
        temp_dict[str2 + "_" + str1] = q_strat_dict[key] #Swapping order for readability :D

    q_strat_dict = temp_dict

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