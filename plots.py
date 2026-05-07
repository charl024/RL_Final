import matplotlib.pyplot as plt
from data import load_data

def generate_plots():
    dict_list = load_data()
    
    for dict in dict_list:
        box_stats = []
        for name, stat_list in dict.items():
            stat_list = stat_list[4:]
            box_stats.append({
                'label': name,            # Name for the x-axis
                'whislo': stat_list[0], # Minimum
                'q1': stat_list[1],     # 1st Quartile
                'med': stat_list[2],    # Median (Q2)
                'q3': stat_list[3],     # 3rd Quartile
                'whishi': stat_list[4], # Maximum
                'fliers': []            # Outliers (empty if none)
            })
        
        fig, ax = plt.subplots()
        ax.bxp(box_stats)  # Draw the boxplot from pre-calculated stats
        ax.set_title("")
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