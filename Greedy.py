from  numba import njit
from utils import *


@njit
def flatten_graph(graph):
    """
    Flatten a graph into matrices for adjacency, weights, start indices, and end indices.

    Parameters:
    - graph (adjacency matrix): The input graph to be flattened.

    Returns:
    - numpy.ndarray: Flattened adjacency matrix.
    - numpy.ndarray: Flattened weight matrix.
    - numpy.ndarray: Start indices for nodes in the flattened matrices.
    - numpy.ndarray: End indices for nodes in the flattened matrices.
    """
    flattened_adjacency = []
    flattened_weights = []
    num_nodes = graph.shape[0]
    
    node_start_indices = np.zeros(num_nodes,dtype=np.int64)
    node_end_indices = np.zeros(num_nodes,dtype=np.int64)
    
    for i in range(num_nodes):
        node_start_indices[i] = len(flattened_adjacency)
        for j in range(num_nodes):
            if graph[i, j] != 0:
                flattened_adjacency.append(j)
                flattened_weights.append(graph[i, j])
                
        node_end_indices[i] = len(flattened_adjacency)

    return (
        np.array(flattened_adjacency),
        np.array(flattened_weights),
        node_start_indices,
        node_end_indices
    )



@njit
def standard_greedy(graph):
    adj_matrix, weight_matrix, start_list, end_list=graph
    
    n=len(start_list)
    delta_local_cuts=np.zeros(n)
    spins=np.ones(n)
    
    
    curr_score=0
    for i in range(n):
        for j,weight in zip(adj_matrix[start_list[i]:end_list[i]],
                  weight_matrix[start_list[i]:end_list[i]]):
                
            delta_local_cuts[i]+=weight*(2*spins[i]-1)*(2*spins[j]-1)
            curr_score+=weight*(spins[i]+spins[j]-2*spins[i]*spins[j])

    curr_score/=2    

    
    while True:
        v=np.argmax(delta_local_cuts)

        if delta_local_cuts[v]<=0:
            break
                    
        curr_score+=delta_local_cuts[v]
        delta_local_cuts[v]=-delta_local_cuts[v]
        
        for u,weight in zip(adj_matrix[start_list[v]:end_list[v]],
                                weight_matrix[start_list[v]:end_list[v]]):

            delta_local_cuts[u]+=weight*(2*spins[u]-1)*(2-4*spins[v])

        spins[v] = 1-spins[v]

    return curr_score








from argparse import ArgumentParser

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument( "--distribution", type=str, default='ER_7000vertices_weighted', help="Name of the dataset to be used (default: 'Facebook')" )
    
    args = parser.parse_args()

    distribution = args.distribution

    sprint(distribution)

    test_dataset = GraphDataset(f'../data/testing/{distribution}',ordered=True)
    

    df = defaultdict(list)
    for i in range(len(test_dataset)):

        graph = test_dataset.get()
        start = time.time()
        obj_val = standard_greedy(flatten_graph(graph=graph))
        end = time.time()

        elapesed_time = end -start

        df['cut'].append(obj_val)
        df['Time'].append(elapesed_time)

        # break
    folder_name = f'data/Greedy/{distribution}'

    os.makedirs(folder_name,exist_ok=True)

    file_path = os.path.join(folder_name,'results') 

    df = pd.DataFrame(df)
    
    # OPT = load_from_pickle(f'../data/testing/{distribution}/optimal')
    # df['Approx. ratio'] = df['cut']/OPT['OPT'].values
    print(df)

    df.to_pickle(file_path)



    
    

    
    

















    



