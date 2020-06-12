import numpy as np
from scipy.linalg import block_diag


def knn_graph(D=None, k=None):

    print(f'Constructing {k}-NN row graph.')

    NN = np.argsort(D, axis=1)
    kNN = NN[:, 1:(k + 1)]

    N, k = np.shape(kNN)

    A = np.zeros((N, N))
    for i in range(N):
        for l in range(k):

            A[i, kNN[i, l]] = 1
            A[kNN[i, l], i] = 1

    np.fill_diagonal(A, 0)
    
    print("Weighting the adjacency matrix.")

    return A * np.exp(-1.0 * D)


def sequential_time_graph(num_nodes, weights=[1]):
    
    print(f'* Constructing sequential time graph with {len(weights)} weights.\n')

    A = np.zeros((num_nodes, num_nodes))
    
    for l in range(1, len(weights) + 1):
        o = weights[l - 1] * np.ones(num_nodes - l)
        A += np.diag(o, l) + np.diag(o, -l)

    np.fill_diagonal(A, 0)

    return A


def grouped_time_graph(num_nodes, num_groups):

    print(f'* Constructing grouped time graph with {num_groups} groups.\n')

    dim = num_nodes // num_groups
    rest = [1] * (num_nodes - dim * num_groups)

    W = block_diag(*[np.ones((dim, dim))] * num_groups, *rest)

    for i in range(0, num_nodes, dim):
        W[i - 1, i] = 1
        W[i, i - 1] = 1

    return W


if __name__ == "__main__":

    A = grouped_time_graph(321, 10)
    
    import networkx as nx

    G = nx.from_numpy_matrix(A)
    print(nx.is_connected(G))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(A, aspect="auto")
    plt.show()
    