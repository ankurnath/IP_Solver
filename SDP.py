import cvxpy as cp
from utils import *
from scipy.linalg import sqrtm


test_dataset = GraphDataset(f'../data/testing/BA_800vertices_unweighted',ordered=True)


graph = test_dataset.get()

n = len(graph)

matrix = cp.Variable((n , n ), PSD=True)

cut = .25 * cp.sum(cp.multiply(graph, 1 - matrix))

problem = cp.Problem(cp.Maximize(cut), [cp.diag(matrix) == 1])

problem.solve(verbose=True)


# print(matrix.value)
# x = sqrtm(matrix.value)

# u = np.random.random(n)

# x = np.sign(x @ u)
# print(x)


vectors = matrix.value
random = np.random.normal(size=vectors.shape[1])
random /= np.linalg.norm(random, 2)

spins = np.sign(np.dot(vectors, random))

# print(np.sign(np.dot(vectors, random)))

cut = (1/4) * np.sum( np.multiply( graph, 1 - np.outer(spins, spins) ) )
print(cut)