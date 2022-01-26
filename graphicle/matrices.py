import numpy as np

import graphicle as gcl


def knn_adj(
    matrix, self_loop=False, k=8, weighted=False, row=True, dtype=None
):
    """Produce a directed adjacency matrix with outward edges
    towards the k nearest neighbours, determined from the input
    affinity matrix.

    Keyword arguments:
        matrix (2d numpy array): particle affinities
        self_loop (bool): if False will remove self-edges
        k (int): number of nearest neighbours in result
        weighted (bool): if True edges weighted by affinity,
            if False edge is binary
        row (bool): if True outward edges given by rows, if False cols
        dtype (numpy): data type: type of the output array
            note: must be floating point if weighted is True
    """
    axis = 0  # calculate everything row-wise
    if self_loop is False:
        k = k + 1  # for when we get rid of self-neighbours
    knn_idxs = np.argpartition(matrix, kth=k, axis=axis)
    near = knn_idxs[:k]
    edge_weights = 1
    if weighted is True:
        if dtype is None:
            dtype = matrix.dtype.type
        else:
            if not isinstance(dtype(1), np.floating):
                raise ValueError(
                    "Update the dtype parameter passed to this function "
                    + "to a numpy floating point type for weighted output."
                )
        edge_weights = np.take_along_axis(matrix, near, axis=axis)
    else:
        dtype = np.bool_
    adj = np.zeros_like(matrix, dtype=dtype)
    np.put_along_axis(adj, near, edge_weights, axis=axis)
    if row is False:
        adj = adj.T
    if self_loop is False:
        np.fill_diagonal(adj, 0)
    return adj


def fc_adj(num_nodes, self_loop=False, dtype=np.bool_):
    """Create a fully connected adjacency matrix."""
    adj = np.ones((num_nodes, num_nodes), dtype=dtype)
    if self_loop is False:
        np.fill_diagonal(adj, 0)
    return adj


def delta_R(pmu: gcl.MomentumArray) -> np.ndarray:
    """Returns a symmetric matrix of delta R vals from input
    4-momentum array.
    """
    size = len(pmu)
    aff = np.zeros((size, size), dtype=np.float64)
    dR_cols = (pmu[shift].delta_R(pmu[shift:]) for shift in range(size))
    for idx, col in enumerate(dR_cols):
        aff[idx:, idx] = col
        aff[idx, idx:] = col
    return aff
