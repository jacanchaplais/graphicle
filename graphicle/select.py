"""
``graphicle.select``
====================

Utilities for selecting elements from graph structured particle data.
"""
from typing import Set, Optional, Tuple, Iterable, Dict, Union
from itertools import combinations, compress, chain

import numpy as np
import numpy.typing as npt
import pandas as pd
import networkx as _nx

import graphicle as gcl
from . import base


def find_vertex(
    graph: gcl.Graphicle,
    pdgs_in: Optional[Set[int]] = None,
    pdgs_out: Optional[Set[int]] = None,
) -> npt.NDArray[np.int32]:
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
    masks = gcl.MaskGroup(agg_op=gcl.data.MaskAggOp.OR)
    masks["in"] = np.isin(adj.edges["in"], desc_nodes)
    masks["out"] = np.isin(adj.edges["out"], desc_nodes)
    return gcl.MaskArray(masks.data)


def hadron_vertices(
    adj: gcl.AdjacencyList,
    status: gcl.StatusArray,
) -> npt.NDArray[np.int32]:
    """Locates the hadronisation vertices in the generation DAG.

    Parameters
    ----------
    adj : AdjacencyList
        The adjacency list of the generation DAG.
    status : StatusArray
        The status codes associated with each particle in the event.

    Returns
    -------
    vertex_ids : ndarray[int32]
        Indices of the hadronisation vertices in the generation DAG,
        returned in no particular order.
    """
    return np.unique(adj.edges[status.in_range(80, 90)]["in"])


def _hard_matrix(hard_desc: gcl.MaskGroup) -> base.BoolVector:
    """Returns a square adjacency matrix for the hard process. Rows
    represent each hard parton, and the values indicate whether another
    given hard parton is a descendant of it.
    """
    num_hard = len(hard_desc)
    mat = np.zeros((num_hard, num_hard), dtype="<?")
    for i, mask in enumerate(hard_desc.values()):
        mat[i, :] = mask.data
    return mat


def _composite_matrix(hard_matrix: base.BoolVector) -> base.BoolVector:
    """Similar to ``_hard_matrix()``, but columns represent direct
    children, not grand-children further nested composites.
    """
    pcls_with_hard_children = filter(np.any, hard_matrix)
    asc_parents = sorted(pcls_with_hard_children, key=np.sum)
    for row_i in reversed(asc_parents):
        for row_j in asc_parents:
            if row_i is row_j:
                continue
            row_i[row_j] = False
    return hard_matrix


def _hard_children(
    comp_mat: base.BoolVector,
    names: Tuple[str, ...],
) -> Dict[str, Tuple[str, ...]]:
    parents = filter(lambda parton: np.any(parton[1]), zip(names, comp_mat))
    comp_dict = dict()
    for name, parton in parents:
        comp_dict[name] = tuple(compress(names, parton))
    return comp_dict


def hard_descendants(
    graph: gcl.Graphicle,
    target: Optional[Iterable[int]] = None,
    sign_sensitive: bool = False,
) -> gcl.MaskGroup:
    """Returns a MaskGroup over the particles in the graph, where True
    indicates a particle descends from a specific hard parton.

    :group: select

    Parameters
    ----------
    graph : Graphicle
        Particle graph containing at least PdgArray and StatusArray.
    target : Iterable[int], optional
        PDG codes referring to the hard particles for which descendants
        are obtained. If ``None``, will obtain masks for all
        intermediate and outgoing hard partons. Default is ``None``.
    sign_sensitive : bool
        Indicates whether sign of PDG code should be used when selecting
        hard partons, ie. if set to False both particle and
        anti-particle partons will be masked, whereas if True only the
        partons explicitly matching the target sign will be considered.
        Default is False.
    """
    hard_vtxs = dict()
    # get the vertices of the hard partons
    for stage, mask in graph.hard_mask.items():
        if stage == "incoming":
            continue
        pcls = graph[mask]
        if target is not None:
            hard_mask = pcls.pdg.mask(
                list(target), blacklist=False, sign_sensitive=sign_sensitive
            )
            if not np.any(hard_mask):
                continue
            pcls = pcls[hard_mask]
        hard_vtxs.update(
            dict(zip(tuple(pcls.pdg.name), tuple(pcls.edges["out"])))
        )
    # find the descendants of those vertices
    masks = gcl.MaskGroup(agg_op=gcl.data.MaskAggOp.OR)
    for pcl_name, vtx in hard_vtxs.items():
        masks[pcl_name] = vertex_descendants(graph.adj, vtx)
    return masks


def hierarchy(
    graph: gcl.Graphicle,
    use_pmu: bool = True,
    desc: Optional[gcl.MaskGroup] = None,
) -> gcl.MaskGroup:
    """Composite ``MaskGroup`` of ``MaskGroup`` instances, representing
    the descendants of the hard process. Uses a tree structure, such
    that partons which are descendants of other hard partons are
    accessible, and nested within their parents.

    :group: select

    Parameters
    ----------
    graph : Graphicle
        The event to be parsed. Must include ``status``, ``edges``,
        ``pmu``, and ``pdg`` data.
    use_pmu : bool
        If set to ``True``, when colour singlet particles are outgoing
        from a hadronisation vertex on which more than one hard parton's
        descendants were incident, the outgoing particles will be
        associated with the nearest parton in the pseudorapidity-azimuth
        plane. If set to ``False``, the deepest nested masks may
        overlap, but will be based entirely upon the ancestral structure
        of the DAG. Default is ``True``.
    desc : MaskGroup, optional
        If the masks for all partons in the hard process have already
        been computed, these may be passed here to save performing the
        computation again.

    Returns
    -------
    hierarchy : MaskGroup
        Nested composite of ``MaskGroup`` instances, representing the
        hierarchical structure of the hard process, and the descendants
        of the partons throughout the shower. Nested ``MaskGroup``
        instances additionally contain a ``latent`` ``MaskArray``,
        referring to the descendants of the parent parton which are not
        also descendants of the children partons.

    Examples
    --------
    Generating an event and automatically detecting process structure:

        >>> import showerpipe as shp
        ... import graphicle as gcl
        ...
        ... # generate event using showerpipe
        ... lhe_url = ("https://zenodo.org/record/6034610/"
        ...            "files/unweighted_events.lhe.gz")
        ... gen = shp.generator.PythiaGenerator(
        ...             "pythia-settings.cmnd", lhe_url)
        ... event = next(gen)
        ... # create graphicle objects
        ... graph = gcl.Graphicle.from_event(event)
        ... masks = gcl.select.hierarchy(graph)
        ...
        >>> masks
        MaskGroup(masks=["t", "t~"], agg_op=OR)
        >>> masks["t"]
        MaskGroup(masks=["b", "W+", "latent"], agg_op=OR)
        >>> print(masks)  # view full nested structure
        MaskGroup(agg_op=OR)
        ├── t
        │   ├── b
        │   ├── W+
        │   │   ├── d~
        │   │   ├── u
        │   │   └── latent
        │   └── latent
        └── t~
            ├── b~
            ├── W-
            │   ├── d
            │   ├── u~
            │   └── latent
            └── latent
        >>> # latent contains the descendants not from constituents
        ... graph[masks["t"]["latent"]].pdg.name
        array(['t', 't', 't', 't', 't', 'b', 'W+'], dtype=object)

    Notes
    -----
    Each mask refers to the descendants of the parton it is labelled by.
    This is an exclusive set, ie. it does not include the parton itself.
    """
    hard_mask = graph.hard_mask
    del hard_mask["incoming"]
    pmu = graph.pmu
    hard_pdg = graph.pdg[hard_mask]
    hard_pmu = pmu[hard_mask]
    names = tuple(map(str, hard_pdg.name))
    name_to_pdg = dict(zip(names, map(int, hard_pdg.data)))
    if desc is None:
        desc = hard_descendants(graph)
    hard_desc = desc[hard_mask][list(names)]
    hard = _hard_children(
        _composite_matrix(_hard_matrix(hard_desc)), names  # type: ignore
    )
    keys = set(hard.keys())
    vals = set(chain.from_iterable(hard.values()))
    roots = keys.difference(keys.intersection(vals))

    def make_tree(flat):
        branch = gcl.MaskGroup(agg_op="or")  # type: ignore
        if isinstance(flat, dict):
            for key, nest in flat.items():
                if key not in roots:
                    continue
                branch[key] = make_tree(nest)
                branch[key]["latent"] = (  # type: ignore
                    branch[key].data != desc[key].data
                )
        else:
            no_hard = True
            for parton in flat:
                if parton in hard:
                    no_hard = False
                    branch[parton] = make_tree(hard[parton])
                    branch[parton]["latent"] = (  # type: ignore
                        branch[parton].data != desc[parton].data
                    )
                else:
                    branch[parton] = desc[parton]
            if (use_pmu is True) and (no_hard is True):
                branch_pdgs = tuple(name_to_pdg[k] for k in branch.keys())
                parton_mask = hard_pdg.mask(
                    branch_pdgs, blacklist=False, sign_sensitive=True
                )
                parton_names = tuple(map(str, hard_pdg[parton_mask].name))
                parton_pmu = hard_pmu[parton_mask]
                overlap = branch.bitwise_and
                dR = parton_pmu.delta_R(pmu[overlap])
                alloc = np.argmin(dR, axis=0)
                for num, name in enumerate(parton_names):
                    alloc_mask = alloc == num
                    data = branch[name].data[overlap]
                    data[~alloc_mask] = False
                    branch[name].data[overlap] = data
        return branch

    return make_tree(hard)


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
