from docplex.mp.model import Model

# mdl = Model(name='buses')
# nbbus40 = mdl.integer_var(name='nbBus40')
# nbbus30 = mdl.integer_var(name='nbBus30')

# mdl.parameters.optimalitytarget=3


# mdl.add_constraint(nbbus40*40 + nbbus30*30 <= 300, 'kids')

# mdl.maximize(nbbus40*nbbus30)

# mdl.solve()

# for v in mdl.iter_integer_vars():
#     print(v," = ",v.solution_value)

# print()
# print("with more constraints")

# option1=mdl.binary_var(name='option1')
# option2=mdl.binary_var(name='option2')


# import cplex
from utils import *

# model  = cplex.Cplex()

model = Model(name='Maximum Cut')

# model.set_problem_name('Maximum Cut')
# model.set_problem_type(model.problem_type.MILP)
# model.parameters.timelimit.set(3600.0)

# model.objective.set_sense(model.objective.sense.maximize)


test_dataset = GraphDataset(f'../data/testing/BA_800vertices_unweighted',ordered=True)
graph = test_dataset.get()
graph = nx.from_numpy_array(graph)

x = {node: model.binary_var(name=f"x_{node}") for node in graph.nodes()}

model.maximize(model.sum(data['weight']*x[u]+data['weight']*x[v]-2*data['weight']*x[u]*x[v] for u,v, data in graph.edges(data=True)))


model.print_information()
model.solve()

print(model._objective_value())

# tms = model.solve()
# assert tms
# tms.display()
# # Add binary variables for each node in the graph
# # Create a dictionary to map the node index to CPLEX variable names
# vdict = {i: f"Build_{i}" for i in graph.nodes()}
# for var_name in vdict.values():
#     model.variables.add(names=[var_name], types=[model.variables.type.binary])

# # Setting the objective function
# objective_terms = []
# for i, j, data in graph.edges(data=True):
#     weight = data.get('weight', 1)  # Default to 1 if 'weight' is not in data
    
#     # Add objective components: weight * (v_i + v_j - 2 * v_i * v_j)
#     # objective_terms.append((vdict[i]+vdict[j] - 2* vdict[i]* vdict[j], weight))
#     objective_terms.append((vdict[j]+'*'+vdict[j], weight))
#     break
#     # objective_terms.append((vdict[i], -2 * weight))
#     # objective_terms.append((vdict[j], -2 * weight))

# # Setting the objective
# # model.objective.set_linear(objective_terms)
# model.objective.set_quadratic(objective_terms)
# # # Set the objective in CPLEX model
# # model.objective.set_linear([(term, 1) for term in objective_terms])

# # objective = model.sum(vdict[i]*data['weight']+vdict[j]*data['weight']-2*data['weight']*vdict[i]*vdict[j] for i, j ,data in graph.edges(data=True))
# # print(model.objective.set)