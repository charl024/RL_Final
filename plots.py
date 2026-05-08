import matplotlib.pyplot as plt
from data import load_data
from testing_area import create_environments

def generate_plots(save_fig=False):
    dict = load_data()
    
    for key, value  in dict.items():
        # For box plat statistics
        box_stats = []

        if len(value.items()) > 7:
            # Too many tests for x-axis name, different approach

            for i, (name, stat_list) in enumerate(value.items()):
                if (i % 3 == 0):
                    box_stats = []
                stat_list = stat_list[4:]

                #x-label
                label = name.split("_")
                label = label[0]

                box_stats.append({
                    'label': label,          # Name for the x-axis
                    'whislo': stat_list[0], # Minimum
                    'q1': stat_list[1],     # 1st Quartile
                    'med': stat_list[2],    # Median (Q2)
                    'q3': stat_list[3],     # 3rd Quartile
                    'whishi': stat_list[4], # Maximum
                    'fliers': []            # Outliers (empty if none)
                })
                if ((i + 1) % 3 == 0):  
                    fig, ax = plt.subplots()
                    ax.bxp(box_stats)
                    # Reward Strat String
                    strat_name = name.split("_")
                    strat_name = strat_name[1:]
                    strat_name = "_".join(strat_name)

                    # Set title
                    test_name = key.replace("_", " ")
                    test_name = test_name[:-5] # removing " dict" from test_name
                    test_name += " test"
                    test_name = test_name.title()
                    test_name = test_name.split()
                    test_name[1] = f"{strat_name}"
                    test_name = " ".join(test_name)
                    if test_name[0] == "Q":
                        test_name = "Q-Learning " + test_name[2:]
                    ax.set_title(test_name)
                    # Set x-label
                    test_name_split = test_name.split(" ")
                    x_label = test_name_split[1]
                    ax.set_xlabel("Map")
                    # Y-label
                    ax.set_ylabel("Path Length")
                    if save_fig:
                        plt.savefig(f"box_plots/{test_name.replace(" ","_")}.svg")
                    plt.show()
        else:
            # Include label 
            for name, stat_list in value.items():
                stat_list = stat_list[4:]
                box_stats.append({
                    'label': name,          # Name for the x-axis
                    'whislo': stat_list[0], # Minimum
                    'q1': stat_list[1],     # 1st Quartile
                    'med': stat_list[2],    # Median (Q2)
                    'q3': stat_list[3],     # 3rd Quartile
                    'whishi': stat_list[4], # Maximum
                    'fliers': []            # Outliers (empty if none)
                })
        
            fig, ax = plt.subplots()
            ax.bxp(box_stats)

            # Set title
            test_name = key.replace("_", " ")
            test_name = test_name[:-5] # removing " dict" from test_name
            test_name += " test"
            test_name = test_name.title()
            if test_name[0] == "Q":
                test_name = "Q-Learning " + test_name[2:]
            ax.set_title(test_name)

            # Set x-label
            test_name_split = test_name.split(" ")
            x_label = test_name_split[1]
            ax.set_xlabel(x_label)

            # Y-label
            ax.set_ylabel("Path Length")

            if save_fig:
                plt.savefig(f"box_plots/{test_name.replace(" ","_")}.svg")
            plt.show()

def plot_maps():
    maps = ["hi", "map1", "map2", "map3", "map4", "spiral"]
    envs = create_environments(maps)

    for i, env in enumerate(envs):
        env.plot()

plot_maps()
generate_plots()
