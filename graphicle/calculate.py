"""
``graphicle.calculate``
=======================

Algorithms for performing common HEP calculations using graphicle data
structures.
"""
from typing import Tuple, Optional, Set, List
from functools import lru_cache

import numpy as np
from numpy.lib.recfunctions import (
    unstructured_to_structured,
    structured_to_unstructured,
)
from typicle import Types
import networkx as nx
import pandas as pd

import graphicle as gcl


_types = Types()


def jet_mass(
    pmu: gcl.MomentumArray, weight: Optional[np.ndarray] = None
) -> float:
    """Returns the combined jet mass of the particles represented in
    the provided MomentumArray.

    Parameters
    ----------
    pmu : MomentumArray
        Momenta of particles comprising a jet, or an analagous combined
        object.
    weight : array, optional
        Weights for each particle when reconstructing the jet momentum.

    Notes
    -----
    This does not mask the MomentumArray for you. All filters and cuts
    must be applied before passing to this function.
    """
    eps = 1e-8
    data = structured_to_unstructured(pmu.data)
    if weight is not None:
        data = weight[:, np.newaxis] * data
    minkowski = np.array([-1.0, -1.0, -1.0, 1.0])
    return np.sqrt((data.sum(axis=0) ** 2 @ minkowski) + eps)  # type: ignore


@lru_cache(maxsize=None)
def _trace_single(
    nx_graph: nx.DiGraph,
    vertex: int,
    basis: Tuple[int, ...],
    exclusive: bool = False,
) -> np.ndarray:
    dtype = "<f8"
    if vertex in basis:
        color = np.array([base == vertex for base in basis], dtype=dtype)
        if exclusive is True:
            return color
    else:
        color = np.zeros(len(basis), dtype=dtype)
    in_edges = nx_graph.in_edges(vertex, data=True)
    colors_in: List[np.ndarray] = []
    color_weights: List[float] = []
    for edge in in_edges:
        color_weights.append(edge[2]["weight"])
        in_vtx = edge[0]
        colors_in.append(_trace_single(nx_graph, in_vtx, basis, exclusive))
    try:
        color += np.average(colors_in, axis=0, weights=color_weights)
    except ZeroDivisionError:
        # this means no incoming colour contribution, so skip adding
        pass
    return color


def hard_trace(
    graph: gcl.Graphicle,
    mask: gcl.MaskArray,
    prop: np.ndarray,
    exclusive: bool = False,
    target: Optional[Set[int]] = None,
) -> np.ndarray:
    """Performs flow tracing from specified particles in an event, back
    to the hard partons.

    Parameters
    ----------
    graph : Graphicle
        Full particle event, containing hard partons, showering and
        hadronisation.
    mask : MaskArray or MaskGroup
        Boolean mask identifying which particles should have their
        ancestry traced.
    prop : array
        Property to trace back, eg. energy.
        Must be the same shape as arrays stored in graph.
    exclusive : bool
        If True, double counting from descendant particles in the hard
        event will be switched off.
        eg. for event t > b W+, descendants of b will show no
        contribution from t, as b is a subset of t.
        Default is False.
    target : set of ints, optional
        Highlights specific partons in the hard event to decompose
        properties with respect to.
        If left as None, will simply use all partons in hard event,
        except for incoming partons.

    Returns
    -------
    trace_array : array
        Structured array representing the contributions of hard partons
        traced down to the properties of the selected subset of
        particles specified by mask.
    """
    nx_graph = nx.DiGraph()
    graph_dict = graph.adj.to_dicts(edge_data={"weight": prop})
    nx_graph.add_edges_from(graph_dict["edges"])
    hard_mask = graph.hard_mask.copy()
    del hard_mask["incoming"]
    hard_graph = graph[hard_mask.bitwise_or]
    if target:
        target_mask = hard_graph.pdg.mask(
            list(target), blacklist=False, sign_sensitive=True
        )
        hard_graph = hard_graph[target_mask]
    names, vtxs = tuple(hard_graph.pdg.name), tuple(hard_graph.edges["out"])
    focus_pcls = graph.edges[mask]["out"]
    struc_dtype = np.dtype(list(zip(names, ("<f8",) * len(names))))
    trc = np.array([_trace_single(nx_graph, pcl, vtxs) for pcl in focus_pcls])
    _trace_single.cache_clear()
    return unstructured_to_structured(trc, struc_dtype)
