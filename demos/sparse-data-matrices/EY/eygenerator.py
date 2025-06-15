import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite

def generate_erdos_renyi_graph(n, p):
    np.random.seed(0)
    num_possible_edges = n * (n - 1) // 2  # Total edges in an undirected graph without self-loops
    num_edges = int(num_possible_edges * p)  # Estimate the number of edges to create based on probability p
    row_indices = np.random.choice(n, num_edges, replace=True)
    col_indices = np.random.choice(n, num_edges, replace=True)
    mask = row_indices < col_indices  # Ensure no self-loops and undirected edges
    row_indices = row_indices[mask]
    col_indices = col_indices[mask]
    rows = np.concatenate((row_indices, col_indices))
    cols = np.concatenate((col_indices, row_indices))
    vals = np.ones(rows.size)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(n, n))
    A.data = np.minimum(A.data, 1)
    return A

def main(n, p, filename):
    adjacency_matrix = generate_erdos_renyi_graph(n, p)  # Generate the Erdős–Rényi graph
    mmwrite(filename, adjacency_matrix)  # Write the sparse matrix to a Matrix Market file

if __name__ == '__main__':
    # Parameters
    n = 50  # Number of vertices
    p = 0.8  # Probability of edge creation
    filename = 'tiny.mtx'  # Output filename

    main(n, p, filename)
