import numpy as np
import networkx as nx

def Graph(pop, rc):
    N = pop.shape[0]
    adj_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                dist = np.linalg.norm(pop[i, :2] - pop[j, :2])
                if dist <= rc:
                    adj_matrix[i, j] = dist

    G = nx.from_numpy_array(adj_matrix, create_using=nx.Graph)
    return G