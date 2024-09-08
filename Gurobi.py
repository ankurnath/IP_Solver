import gurobipy as gp
from gurobipy import GRB

from utils import *
import os
import sys



def gurobi_solver(graph,max_time = None,max_threads = None):
    

    model = gp.Model()

    if max_time:
        model.setParam('TimeLimit', max_time)

    if max_threads:
        model.setParam('Threads', max_threads)

    vdict= model.addVars(graph.number_of_nodes(), vtype=GRB.BINARY, name="Build")

    cut = [data['weight']*(vdict[i] + vdict[j] - 2*vdict[i]*vdict[j]) for i,j,data in graph.edges(data=True)]

    model.setObjective(sum(cut), gp.GRB.MAXIMIZE)
    

    model.optimize()
    
    
    
    return model.ObjVal, [key for key in vdict.keys() if abs(vdict[key].x) > 1e-6]



    
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument( "--distribution", type=str, default='torodial_10000vertices_weighted', help="Name of the dataset to be used (default: 'Facebook')" )
    parser.add_argument( "--time_limit", type=float, default= 200, help="Maximum Time Limit" )
    parser.add_argument( "--threads", type=int, default= 20, help="Maximum number of threads" )
  
    args = parser.parse_args()

    distribution = args.distribution
    time_limit = args.time_limit
    threads = args.threads

    sprint(distribution)
    sprint(time_limit)
    sprint(threads)

    # sys.stdout = open('out.dat', 'w')
    test_dataset = GraphDataset(f'../data/testing/{distribution}',ordered=True)


    
    df = defaultdict(list)

    for _ in range(len(test_dataset)):

        graph = test_dataset.get()
        graph = nx.from_numpy_array(graph)
        objVal, solution = gurobi_solver(graph=graph,max_time=time_limit,max_threads=threads)
        df['cut'].append(objVal)
        df['time'].append(time_limit)
        df['threads'].append(threads)
        
        # break

    # df = pd.DataFrame()

    folder_name = f'data/Gurobi/{distribution}'

    os.makedirs(folder_name,exist_ok=True)

    file_path = os.path.join(folder_name,'results') 

    df = pd.DataFrame(df)
    # OPT = load_from_pickle(f'../data/testing/{distribution}/optimal')
    # df['Approx. ratio'] = df['cut']/OPT['OPT'].values
    print(df)

    df.to_pickle(file_path)


        






    # train(dataset=args.dataset,budget=args.budget)