"""
``graphicle.select``
====================

Utilities for selecting elements from graph structured particle data.
"""
from typing import Set, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as _nx

import graphicle as gcl
from . import base


def find_vertex(
    graph: gcl.Graphicle,
    pdgs_in: Optional[Set[int]] = None,
    pdgs_out: Optional[Set[int]] = None,
) -> base.IntVector:
    """Locate vertices with the inward and outward particles of the
    passed pdg codes.

    :group: select

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

    :group: select

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
    mask = np.isin(adj.edges["in"], desc_nodes)
    mask[adj.edges["in"] == vertex] = True
    return gcl.MaskArray(mask)


def hard_descendants(
    graph: gcl.Graphicle, target: Set[int], sign_sensitive: bool = False
) -> gcl.MaskGroup:
    """Returns a MaskGroup over the particles in the graph, where True
    indicates a particle descends from a specific hard parton.

    :group: select

    Parameters
    ----------
    graph : Graphicle
        Particle graph containing at least PdgArray and StatusArray.
    target : set
        PDG codes referring to the hard particles for which descendants
        are obtained.
    sign_sensitive : bool
        Indicates whether sign of PDG code should be used when selecting
        hard partons, ie. if set to False both particle and
        anti-particle partons will be masked, whereas if True only the
        partons explicitly matching the target sign will be considered.
        Default is False.
    """
    hard_vtxs = dict()
    target_list = list(target)
    # get the vertices of the hard partons
    for stage in ("intermediate", "outgoing"):
        mask = graph.hard_mask[stage]
        pcls = graph[mask]
        hard_mask = pcls.pdg.mask(
            target_list, blacklist=False, sign_sensitive=sign_sensitive
        )
        hard_pcls = pcls[hard_mask]
        hard_vtxs.update(
            dict(zip(tuple(hard_pcls.pdg.name), tuple(hard_pcls.edges["out"])))
        )
    # find the descendants of those vertices
    masks = gcl.MaskGroup(agg_op=gcl.data.MaskAggOp.OR)
    for pcl_name, vtx in hard_vtxs.items():
        masks[pcl_name] = vertex_descendants(graph.adj, vtx)
    return masks


def hard_edge(graph: gcl.Graphicle, pdg: int) -> Tuple[int, int]:
    hard_graph = graph[graph.hard_mask.bitwise_or]
    parton = hard_graph[
        hard_graph.pdg.mask(target=[pdg], blacklist=False, sign_sensitive=True)
    ]
    return tuple(parton.edges[0])  # type: ignore


def any_overlap(masks: gcl.MaskGroup) -> bool:
    """Given a MaskGroup object, checks if any of the masks overlap
    with each other.

    :group: select

    Parameters
    ----------
    masks : MaskGroup
        Collection of boolean masks to check.

    Returns
    -------
    any_overlap : bool
        True if at least two MaskArrays in MaskGroup have at least one
        True element in the same location.
    """
    combos = combinations(masks.dict.values(), 2)
    pair_checks = map(np.bitwise_and, *zip(*combos))
    overlaps: bool = np.bitwise_or.reduce(tuple(pair_checks), axis=None)
    return overlaps
