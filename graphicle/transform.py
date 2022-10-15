"""
``graphicle.transform``
=======================

Utilities for manipulating the graph structure of particle data.
"""
import numpy as np
import numpy.typing as npt
import networkx as _nx
from typicle import Types
from typicle.convert import cast_array

import graphicle as gcl


_types = Types()
Vector = npt.NDArray[np.float64]


def particle_as_node(adj_list: gcl.AdjacencyList) -> gcl.AdjacencyList:
    """Converts an adjacency list in which the particles are represented
    as edges, to one in which the particles are the nodes.
    The order of the nodes in the resulting AdjacencyList retains the
    same particle ordering of the initial edge list.

    Parameters
    ----------
    adj_list : AdjacencyList
        The edge-as-particle representation.

    Examples
    --------
    >>> from graphicle import transform
    >>> # restructuring existing graph:
    >>> graph.adj = transform.particle_as_node(graph.adj)
    """
    # create the networkx edge graph
    nx_edge_graph = _nx.MultiDiGraph()
    graph_dicts = adj_list.to_dicts()
    nx_edge_graph.add_edges_from(graph_dicts["edges"])
    # transform into node graph (with edge tuples rep'ing nodes)
    nx_node_graph = _nx.line_graph(G=nx_edge_graph, create_using=_nx.DiGraph)
    # create a node index for each particle
    edges = adj_list.edges
    num_pcls = len(edges)
    node_idxs = np.empty(num_pcls, dtype=_types.int)
    # translate nodes represented with edge tuples into node indices
    edge_node_type = _types.edge.copy()
    edge_node_type.append(("key", _types.int))  # if > 1 pcl between vtxs
    edges_as_nodes = cast_array(
        np.array(tuple(nx_node_graph.nodes)), edge_node_type
    )

    def check_sign(x):
        """Returns -1 if x <= 0, and +1 if x > 0."""
        sign = np.sign(x)
        sign = sign + int(not sign)  # if sign = 0 => sign = +1
        return sign

    # node labels set as particle indices in original edge array
    for i, node_triplet in enumerate(edges_as_nodes):
        key = node_triplet["key"]
        node = node_triplet[["in", "out"]]
        sign = -1 * check_sign(node["in"] * node["out"])
        node_idxs[i] = sign * (np.where(edges == node)[0][key] + 1)
    nx_node_graph = _nx.relabel_nodes(
        nx_node_graph,
        {n: idx for n, idx in zip(nx_node_graph, node_idxs)},
    )
    return gcl.AdjacencyList(np.array(nx_node_graph.edges))


def centre_angle(angle: Vector, pt: Vector) -> Vector:
    """Shifts angles so transverse momentum weighted centroid is at 0.

    Parameters
    ----------
    angle : array
        Angular displacements.
    pt : array
        Transverse momenta.

    Returns
    -------
    centred_angle : array
        Shifted angular displacements, with centroid at 0.
    """
    # convert angles into complex polar positions
    pos = np.exp(1.0j * angle)
    # obtain weighted sum positions ie. un-normalised midpoint
    pos_wt_mid = (pos * pt).sum()
    # convert to U(1) rotation operator e^(-i delta x)
    rot_op = (pos_wt_mid / np.abs(pos_wt_mid)).conjugate()
    # rotate positions so midpoint is at 0
    pos_centred = rot_op * pos
    return np.angle(pos_centred)  # type: ignore


def centre_pseudorapidity(eta: Vector, pt: Vector) -> Vector:
    """Shifts pseudorapidities so pt weighted midpoint is at 0."""
    pt_norm = pt / pt.sum()
    eta_wt_mid = (eta * pt_norm).sum()
    return eta - eta_wt_mid  # type: ignore
