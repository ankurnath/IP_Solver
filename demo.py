import matplotlib.pyplot as plt
import numpy as np
from utils import *




for i,dist in enumerate(['ER','BA']):
    distributions = []
    df = defaultdict(list)
    for n in [20,40,60,100,200,500]:
        distribution = f'{dist}_{n}vertices_weighted'
        distributions.append(f'{dist}_{n}vertices_weighted')
        
        OPT = load_from_pickle(f'../data/testing/{distribution}/optimal')

        for algorithm in [
            'Gurobi',
            'Greedy'
            ]:
            df_ = load_from_pickle(f'data/{algorithm}/{distributions[-1]}/results')
            df['N'].append(n)
            df['algorithm'].append(algorithm)
            df ['Ratio'].append((df_['cut']/df_['OPT']).mean())


    df = pd.DataFrame(df)

    
    plt.figure(dpi=200)
    markers = {'Gurobi': 'p', 'Greedy': 'P', 'TS': 'D'}  # specify markers for each algorithm
    sns.lineplot(x='N', y='Ratio', hue='algorithm', style='algorithm', markers=markers, data=df, markersize=10)


    
    # ax.plot(df['N'], df['Ratio'], label=algorithm)
    # print(df)

plt.legend()
plt.show()





# # Data for plotting
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)

# # Create subplots sharing both x and y axes
# fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

# # Plot on the first subplot
# line1, = ax1.plot(x, y1, label="sin(x)")
# line3, = ax1.plot(x, y2, label="sin(x)")
# ax1.set_title("Sin(x)")

# # Plot on the second subplot
# line2, = ax2.plot(x, y2, label="cos(x)")
# ax2.set_title("Cos(x)")

# # Create a common legend for both subplots
# fig.legend([line1, line2], labels=["sin(x)", "cos(x)"], loc="lower center", ncol=3)

# # Display the plot
# plt.tight_layout()
# plt.show()
