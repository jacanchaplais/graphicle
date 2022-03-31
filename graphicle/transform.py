from typing import Set, Optional

import numpy as np
import pandas as pd
import networkx as _nx

from typicle import Types
from typicle.convert import cast_array

import graphicle as gcl


_types = Types()


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


def find_vertex(
    graph: gcl.Graphicle,
    pdgs_in: Optional[Set[int]] = None,
    pdgs_out: Optional[Set[int]] = None,
) -> np.ndarray:
    """Locate vertices with the inward and outward particles of the
    passed pdg codes.

    Parameters
    ----------
    graph : Graphicle
        Graphicle object, which must contain at least edge and pdg data.
    pdgs_in : set of ints
        Subset of pdg codes to match against the incoming particles.
    pdgs_out : set of ints
        Subset of pdg codes to match against the outgoing particles.

    Returns
    -------
    vertices : array of ints
        List the vertex ids which match the passed incoming and outgoing
        pdg codes.
    """
    # preparing the search sets
    search = dict()
    if (pdgs_in is None) and (pdgs_out is None):
        raise ValueError(
            "Must pass at least one of pdgs_in or pdgs_out a set of integers."
        )
    if pdgs_in is None:
        pdgs_in = set()
    if pdgs_out is None:
        pdgs_out = set()
    search = {"pdg_in": pdgs_in, "pdg_out": pdgs_out}
    # construct dataframe
    df = pd.DataFrame(graph.edges)
    df["pdg"] = graph.pdg.data

    # define vertex dataframe
    def vtx_pdgs(df: pd.DataFrame) -> pd.DataFrame:
        """Dataframe indexed by vertex, showing in and out pdgs."""

        def vtx_pdg_pivot(direction: str):
            """Pivot the vertex dataframe in given direction."""
            return df.pivot_table(
                index=direction,
                values="pdg",
                aggfunc=lambda x: tuple(x.to_list()),
            )

        pcls_in = vtx_pdg_pivot("in")
        pcls_out = vtx_pdg_pivot("out")
        # join in and out vertex pdgs into single dataframe
        vtxs = pcls_out.join(
            pcls_in, how="outer", lsuffix="_in", rsuffix="_out"
        )
        vtxs.sort_index(ascending=False, inplace=True)
        # expand the nested lists into repeated index rows
        vtxs = vtxs.explode("pdg_in").explode("pdg_out")
        # relabel the index
        vtxs.index.name = "vertex"
        return vtxs

    # search the vertices for the ingoing / outgoing particles
    vertices = vtx_pdgs(df)
    # boolean mask if over vertices if user in / out pdgs is subset
    masks = vertices.pivot_table(
        index="vertex",
        values=["pdg_in", "pdg_out"],
        aggfunc=lambda x: bool(search[x.name].issubset(set(x.to_list()))),
    )
    # get the vertex ids
    found = masks.query("pdg_in and pdg_out")
    return np.array(found.index.values)


def vertex_descendants(adj: gcl.AdjacencyList, vertex: int) -> gcl.MaskArray:
    """Return a mask over a graphicle object, identifying which
    particles descend from a given interaction vertex.

    Parameters
    ----------
    adj : AdjacencyList
        Topological structure of the graph.
    vertex : int
        The vertex id from which the descending edges are identified.

    Returns
    -------
    mask : MaskArray
        Boolean mask over the graphicle objects associated with the
        passed AdjacencyList.
    """
    graph_dict = adj.to_dicts()
    vertex = int(vertex)
    nx_graph = _nx.MultiDiGraph()
    _ = nx_graph.add_edges_from(graph_dict["edges"])
    desc_nodes = np.array(list(_nx.descendants(nx_graph, vertex)), dtype="<i4")
    masks = gcl.MaskGroup()
    masks["in"] = np.isin(adj.edges["in"], desc_nodes)
    masks["out"] = np.isin(adj.edges["out"], desc_nodes)
    return gcl.MaskArray(masks.bitwise_or)


def centre_angle(angle: np.ndarray, pt: np.ndarray) -> np.ndarray:
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
    return np.angle(pos_centred)


def centre_pseudorapidity(eta: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Shifts pseudorapidities so pt weighted midpoint is at 0."""
    pt_norm = pt / pt.sum()
    eta_wt_mid = (eta * pt_norm).sum()
    return eta - eta_wt_mid
