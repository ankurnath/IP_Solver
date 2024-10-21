import matplotlib.pyplot as plt
import numpy as np
from utils import *
import matplotlib.ticker as ticker





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

    #### CPLEX
    size = [20,40,60,100,200,500]
    df['N'] += size
    df['algorithm'] += ['CPLEX*']* len(size)
    if dist =='ER':
        df ['Ratio'] += [1,1,1,0.87,0.46,0.16]
    elif dist == 'BA':
        df ['Ratio'] += [1,1,1,1,0.83,0.17]



    df = pd.DataFrame(df)

    fontsize = 20
    plt.figure(dpi=200)
    markers = {'Gurobi': 'p', 'Greedy': 'P', 'TS': 'D','CPLEX*':'*'}  # specify markers for each algorithm
    sns.lineplot(x='N', y='Ratio', hue='algorithm', style='algorithm', markers=markers, data=df, markersize=20)
    
    plt.xlabel('Graph Size,|V|',fontsize=fontsize)
    plt.ylabel('Approx. Ratio',fontsize=fontsize)

    plt.xticks([20,40,60,100,200,500],fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    

    # plt.locator_params(nbins=6)
    # plt.xticks([20,40,60,100,200,500])
    plt.xscale('log')
    # Set custom xticks on a logarithmic scale using LogLocator
    ax = plt.gca()  # Get current axis
    ax.set_xticks([20, 40, 60, 100, 200, 500])  # Specify the exact tick locations
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())  # Ensure the tick labels are formatted properly
    # plt.xticks([20,40,60,100,200,500])

    # ax = plt.gca()  # Get the current axis
    # ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[0.0,1.0, 2.0, 5.0]))  # Major ticks at log scale

    
    # plt.grid(True,linestyle='--', alpha=0.7)
    plt.legend(frameon=False,fontsize=fontsize)

    plt.savefig(f'{distribution}', bbox_inches='tight')
    plt.savefig(f'{distribution}.pdf', bbox_inches='tight')
    # plt.show()
    
    # ax.plot(df['N'], df['Ratio'], label=algorithm)
    # print(df)








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
