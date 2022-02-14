import numpy as np
import numpy.typing as npt

import graphicle as gcl
from typicle import Types


_types = Types()


def knn_adj(
    matrix: np.ndarray,
    k: int,
    self_loop: bool = False,
    weighted: bool = False,
    row: bool = True,
    dtype: npt.DTypeLike = None,
) -> np.ndarray:
    """Produce a directed adjacency matrix with outward edges
    towards the k nearest neighbours, determined from the input
    affinity matrix.

    Parameters
    ----------
    matrix : 2d numpy array
        Particle affinities.
    k : int
        Number of nearest neighbours in result.
    weighted : bool
        If True edges weighted by affinity, if False edge is binary.
    self_loop : bool
        If False will remove self-edges.
    row : bool
        If True outward edges given by rows, if False cols.
    dtype : dtype-like
        Type of output. Must be floating point if weighted is True.
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
            type_call = np.dtype(dtype).type
            if not isinstance(type_call(1), np.floating):
                raise ValueError(
                    "Update the dtype parameter passed to this function "
                    + "to a numpy floating point type for weighted output."
                )
        edge_weights = np.take_along_axis(matrix, near, axis=axis)
    else:
        dtype = _types.bool
    adj = np.zeros_like(matrix, dtype=dtype)
    np.put_along_axis(adj, near, edge_weights, axis=axis)
    if row is False:
        adj = adj.T
    if self_loop is False:
        np.fill_diagonal(adj, 0)
    return adj


def fc_adj(num_nodes, self_loop=False, dtype=_types.bool):
    """Create a fully connected adjacency matrix."""
    adj = np.ones((num_nodes, num_nodes), dtype=dtype)
    if self_loop is False:
        np.fill_diagonal(adj, 0)
    return adj


def delta_R_aff(pmu: gcl.MomentumArray) -> np.ndarray:
    """Returns a symmetric matrix of delta R vals from input
    MomentumArray.
    """
    size = len(pmu)
    dtype = pmu.data.dtype.descr[0][1]
    vec = pmu._vector
    aff = np.zeros((size, size), dtype=dtype)
    dR_cols = (vec[shift].deltaR(vec[shift:]) for shift in range(size))
    for idx, col in enumerate(dR_cols):
        aff[idx:, idx] = col
        aff[idx, idx:] = col
    return aff
