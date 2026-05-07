import matplotlib.pyplot as plt
from data import load_data

def generate_plots():
    dict = load_data()
    
    for key, value  in dict.items():
        # For box plat statistics
        box_stats = []

        if len(value.items()) > 7:
            # Too many tests for x-axis name

            # List of alphabetical characters
            char_list = [chr(i) for i in range(ord('a'), ord('z'))]

            for i, (name, stat_list) in enumerate(value.items()):
                stat_list = stat_list[4:]
                box_stats.append({
                    'label': "(" + char_list[i] + ")",# Name for the x-axis
                    'whislo': stat_list[0],           # Minimum
                    'q1': stat_list[1],               # 1st Quartile
                    'med': stat_list[2],              # Median (Q2)
                    'q3': stat_list[3],               # 3rd Quartile
                    'whishi': stat_list[4],           # Maximum
                    'fliers': []                      # Outliers (empty if none)
                })
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

        plt.savefig(f"box_plots/{test_name.replace(" ","_")}.svg")
        plt.show()
    
# # Your pre-calculated statistics
# stats = [{
#     'label': 'My Dataset',  # Name for the x-axis
#     'whislo': 5,            # Minimum
#     'q1': 10,               # 1st Quartile
#     'med': 20,              # Median (Q2)
#     'q3': 30,               # 3rd Quartile
#     'whishi': 35,           # Maximum
#     'fliers': []            # Outliers (empty if none)
# }]

# fig, ax = plt.subplots()
# ax.bxp(stats)  # Draw the boxplot from pre-calculated stats
# ax.set_title('Boxplot from Summary Statistics')
# plt.show()

generate_plots()