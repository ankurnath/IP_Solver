import gurobipy as gp
from gurobipy import GRB

from utils import *


def maxcut(graph,max_time = None,max_threads = None):
    

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
    parser.add_argument( "--distribution", type=str, default='BA_800vertices_unweighted', help="Name of the dataset to be used (default: 'Facebook')" )
    
  
    args = parser.parse_args()


    test_dataset = GraphDataset(f'../data/testing/{args.distribution}',ordered=True)


    df = {'cut':[]}

    for _ in range(len(test_dataset)):

        graph = test_dataset.get()

        graph = nx.from_numpy_array(graph)

        objVal, solution = maxcut(graph=graph)


        df['cut'].append(objVal)

    df = pd.DataFrame()


        






    # train(dataset=args.dataset,budget=args.budget)