from typing import Optional

import numpy as np
import numpy.typing as npt
from typicle import Types

import graphicle as gcl
from . import base


__all__ = ["cut_adj", "knn_adj", "fc_adj", "delta_R_aff"]

_types = Types()


def cut_adj(
    matrix: base.DoubleVector,
    cut: float,
    mode: str = "max",
    self_loop: bool = False,
    weighted: bool = False,
) -> base.DoubleVector:
    """Produce a directed adjacency matrix with outward edges
    towards the neighbours within a cut range, determined from the input
    affinity matrix.

    The cut represents the limiting value of the affinity matrix for
    elements to form edges in the adjacency matrix.

    :group: matrix

    Parameters
    ----------
    matrix : array
        Particle affinities.
    cut : float
        Value beyond which affinities are not sufficient to form edges.
    mode : str
        Sets whether affinities should be above or below cut.
        'max' implies matrix < cut, 'min' implies matrix > cut.
        Default is 'max'.
    self_loop : bool
        If False will remove self-edges. Default is False.
    weighted : bool
        If True edges weighted by affinity, if False edge is binary.
        Default is False.

    Returns
    -------
    adj : array
        Adjacency matrix representing particle connectivity.

    Notes
    -----
    If weighted is False, the returned adjacency matrix will be boolean.
    """
    # form the cut mask
    if mode == "max":
        mask = matrix < cut
    elif mode == "min":
        mask = matrix > cut
    else:
        raise ValueError("mode keyword argument must take either max or min.")
    # set the weights
    if weighted is True:
        weights = matrix
    elif weighted is False:
        weights = np.array(1.0)
    else:
        raise ValueError("weighted keyword argument must take a boolean.")
    # apply the cuts
    adj = np.where(mask, weights, 0.0)
    if self_loop is False:
        np.fill_diagonal(adj, 0.0)
    else:
        np.fill_diagonal(adj, 1.0)
    if weighted is False:
        adj = adj.astype(_types.bool)
    return adj


def knn_adj(
    matrix: base.DoubleVector,
    k: int,
    self_loop: bool = False,
    weighted: bool = False,
    row: bool = True,
    dtype: Optional[npt.DTypeLike] = None,
) -> base.DoubleVector:
    """Produce a directed adjacency matrix with outward edges
    towards the k nearest neighbours, determined from the input
    affinity matrix.

    :group: matrix

    Parameters
    ----------
    matrix : 2d numpy array
        Particle affinities.
    k : int
        Number of nearest neighbours in result.
    weighted : bool
        If True edges weighted by affinity, if False edge is binary.
        Default is False.
    self_loop : bool
        If False will remove self-edges. Default is False.
    row : bool
        If True outward edges given by rows, if False cols. Default is
        True.
    dtype : dtype-like, optional
        Type of output. Must be floating point if weighted is True.

    Returns
    -------
    adj : array
        Adjacency matrix representing particle connectivity.

    Notes
    -----
    If weighted is False, the returned adjacency matrix will be boolean.
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


def fc_adj(
    num_nodes: int, self_loop: bool = False, dtype: npt.DTypeLike = _types.bool
) -> base.DoubleVector:
    """Create a fully connected adjacency matrix.

    :group: matrix

    Parameters
    ----------
    num_nodes : int
        Number of nodes the graph should have.
    self_loop : bool
        Whether to include edges from a node to itself. Default is
        ``False``.
    dtype : dtype-like
        The dtype of the output array. Default is ``np.bool_``.
    """
    adj = np.ones((num_nodes, num_nodes), dtype=dtype)
    if self_loop is False:
        np.fill_diagonal(adj, 0)
    return adj


def delta_R_aff(pmu: gcl.MomentumArray) -> base.DoubleVector:
    """Returns the inter-particle Euclidean distances between particles
    internally within the given MomentumArray.

    :group: matrix

    Parameters
    ----------
    pmu : gcl.MomentumArray
        Four-momenta.

    Returns
    -------
    delta_R_matrix : np.ndarray[double]
        Square symmetric matrix representing the Euclidean distance
        between every pair of particles in the eta-phi plane.

    Notes
    -----
    Infinite values may be encountered if particles are travelling
    parallel to the beam axis, __ie.__ with infinite pseudorapidity.
    """
    return pmu.delta_R(pmu)


def parton_hadron_distance(
    parton_pmu: gcl.MomentumArray,
    hadron_pmu: gcl.MomentumArray,
    pt_exp: float = -0.1,
) -> base.DoubleVector:
    dR = parton_pmu.delta_R(hadron_pmu)
    pt_weight = np.power(parton_pmu.pt, pt_exp)
    return pt_weight[:, np.newaxis] * dR
