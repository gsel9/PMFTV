import numpy as np

from scipy.sparse import csgraph

from .adjacency import knn_graph, sequential_time_graph


def row_col_laplacian(row_graph_config, col_graph_config, row_subset_idx=None, col_subset_idx=None):
	
	L_row = row_laplacian(row_graph_config, row_subset_idx)
	L_col = column_laplacian(col_graph_config, col_subset_idx)

	return L_row, L_col


def row_laplacian(row_graph_config, row_subset_idx=None):

	print("Building row graph")
	D = np.load(row_graph_config.path_distance_matrix)
	print(f"Loaded {np.shape(D)} row distance matrix from:")
	print(row_graph_config.path_distance_matrix)

	# NOTE: Slicing A is OK as long as the corresponding slices are 
	# extracted from X. Matrix entries indicate relationships, not the 
	# structure of A.
	_D = D
	if row_subset_idx is not None:
		print(f"Selecting a subset of {len(row_subset_idx)} samples for row graph")
		_D = _D[row_subset_idx, :]
		_D = _D[:, row_subset_idx]

	A_row = knn_graph(D=_D, k=row_graph_config.k)

	return csgraph.laplacian(A_row, normed=True)


def column_laplacian(col_graph_config, col_subset_idx=None):

	print("Building column graph")
	A_col = sequential_time_graph(col_graph_config.num_nodes, weights=col_graph_config.weights)

	return csgraph.laplacian(A_col, normed=True)
