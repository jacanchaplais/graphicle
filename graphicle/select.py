"""
``graphicle.select``
====================

Utilities for selecting elements from graph structured particle data.
"""
import typing as ty
import itertools as it
import functools as fn
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as _nx

import graphicle as gcl
from . import base


__all__ = [
    "find_vertex",
    "vertex_descendants",
    "hard_descendants",
    "any_overlap",
    "hierarchy",
    "hadron_vertices",
]


DistFunc = ty.Callable[
    [gcl.MomentumArray, gcl.MomentumArray], base.DoubleVector
]


def find_vertex(
    graph: gcl.Graphicle,
    pdgs_in: ty.Optional[ty.Set[int]] = None,
    pdgs_out: ty.Optional[ty.Set[int]] = None,
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


def hadron_vertices(
    adj: gcl.AdjacencyList,
    status: gcl.StatusArray,
) -> ty.Tuple[int, ...]:
    """Locates the hadronisation vertices in the generation DAG.

    Parameters
    ----------
    adj : AdjacencyList
        The adjacency list of the generation DAG.
    status : StatusArray
        The status codes associated with each particle in the event.

    Returns
    -------
    vertex_ids : tuple[int]
        Indices of the hadronisation vertices in the generation DAG,
        returned in no particular order.
    """
    vtx_arr = np.unique(adj.edges[status.in_range(80, 90)]["in"])
    return tuple(map(int, vtx_arr))


def _hadron_vtx_parton_iter(
    adj: gcl.AdjacencyList,
    status: gcl.StatusArray,
    from_hard: base.BoolVector,
) -> ty.Iterator[ty.Tuple[int, base.BoolVector]]:
    """Locates the hadronisation vertices which have descendants of the
    hard process incident on them.

    Parameters
    ----------
    graph : Graphicle
        Event generation DAG.
    from_hard : ndarray[bool_]
        Mask over whole event, identifying particles descending from the
        hard process.

    Yields
    ------
    hadron_vtx : tuple[int, ndarray[bool_]]
        Hadronisation vertex id, and mask over incident partons, for
        vertices which have radiation from the hard process incident.
    """
    vtxs = hadron_vertices(adj, status)
    in_parton_masks = map(np.equal, vtxs, it.repeat(adj.edges["out"]))
    in_parton_masks, in_parton_masks_ = it.tee(in_parton_masks)
    hard_overlap = map(np.bitwise_and, in_parton_masks_, it.repeat(from_hard))
    has_hard_incident = map(np.any, hard_overlap)
    return it.compress(zip(vtxs, in_parton_masks), has_hard_incident)


def _partition_vertex(
    mask: gcl.MaskArray,
    pcls_in: base.MaskLike,
    vtx_desc: gcl.MaskArray,
    final: gcl.MaskArray,
    pmu: gcl.MomentumArray,
    dist_strat: DistFunc,
) -> gcl.MaskArray:
    """Prunes input ``mask`` representing hard parton descendants based
    on if final state hadrons are closer to them or background,
    according to ``dist_strat()``.

    Parameters
    ----------
    mask : gcl.MaskArray
    """
    mask = mask.copy()
    parton_pmu = pmu[pcls_in]
    final_from_vtx = vtx_desc & final
    hadron_pmu = pmu[final_from_vtx]
    dist = dist_strat(parton_pmu, hadron_pmu)
    alloc = np.argmin(dist, axis=0)
    final_from_hard = np.in1d(alloc, np.flatnonzero(mask[pcls_in]))
    mask.data[final_from_vtx] = final_from_hard
    return mask


def partition_descendants(
    graph: gcl.Graphicle,
    hier: gcl.MaskGroup,
) -> gcl.MaskGroup:
    dist_strat = fn.partial(gcl.matrix.parton_hadron_distance, pt_exp=-0.1)
    hadron_vtxs = _hadron_vtx_parton_iter(
        graph.adj, graph.status, hier.bitwise_or
    )
    hier = hier.copy()
    for vtx_id, pcls_in in hadron_vtxs:
        vtx_desc = vertex_descendants(graph.adj, vtx_id)
        for name, branch in hier.items():
            for _, mask in _leaf_mask_iter(branch, name):
                if vtx_id not in graph.edges["out"][mask.data]:
                    continue
                mask.data = _partition_vertex(
                    mask, pcls_in, vtx_desc, graph.final, graph.pmu, dist_strat
                ).data
    return hier


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
    names: ty.Tuple[str, ...],
) -> ty.Dict[str, ty.Tuple[str, ...]]:
    parents = filter(lambda parton: np.any(parton[1]), zip(names, comp_mat))
    comp_dict = dict()
    for name, parton in parents:
        comp_dict[name] = tuple(it.compress(names, parton))
    return comp_dict


def _flat_hierarchy(
    hard_desc: gcl.MaskGroup,
) -> ty.Dict[str, ty.Tuple[str, ...]]:
    """Produces a flat representation of the hard process hierarchy."""
    names = tuple(hard_desc.keys())
    hard_matrix = _hard_matrix(hard_desc)
    comp_matrix = _composite_matrix(hard_matrix)
    return _hard_children(comp_matrix, names)


def hard_descendants(
    graph: gcl.Graphicle,
    target: ty.Optional[ty.Iterable[int]] = None,
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


@dataclass(frozen=True)
class TreeConf:
    """Encapsulates the data needed in each recursive pass of
    ``make_tree()``.

    Attributes
    ----------
    roots : set[str]
        Identifies ancestors from which all other partons in hard
        process descend (following the collision).
    hard : dict[str, tuple[str, ...]]
        Provides a flat representation of the hard process hierarchy.
    pmu : gcl.MomentumArray
        Momentum of all particles within the full event.
    hard_pmu : gcl.MomentumArray
        Momentum of partons in hard process.
    hard_pdg : gcl.PdgArray
        PDG codes of partons in hard process.
    name_top_pdg : dict[str, int]
        Mapping from string names of hard partons to PDG codes.
    use_pmu : bool
        Setting determines whether to use momentum in parton allocation
        for final state.
    """

    roots: ty.Set[str]
    hard: ty.Dict[str, ty.Tuple[str, ...]]
    pmu: gcl.MomentumArray
    hard_pmu: gcl.MomentumArray
    hard_pdg: gcl.PdgArray
    desc: gcl.MaskGroup
    name_to_pdg: ty.Dict[str, int]
    use_pmu: bool


def hierarchy(
    graph: gcl.Graphicle,
    use_pmu: bool = True,
    desc: ty.Optional[gcl.MaskGroup] = None,
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
        │   │   ├── tau+
        │   │   ├── nu(tau)
        │   │   └── latent
        │   └── latent
        └── t~
            ├── b~
            ├── W-
            │   ├── s
            │   ├── c~
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
    hard_desc: gcl.MaskGroup = desc[hard_mask][list(names)]  # type: ignore
    hard = _flat_hierarchy(hard_desc)
    keys = set(hard.keys())
    vals = set(it.chain.from_iterable(hard.values()))
    roots = keys.difference(keys.intersection(vals))
    tree_conf = TreeConf(
        roots, hard, pmu, hard_pmu, hard_pdg, desc, name_to_pdg, use_pmu
    )
    return _make_tree(hard, tree_conf)


def _make_tree(
    flat: ty.Union[ty.Dict[str, ty.Any], ty.Tuple[str, ...]], conf: TreeConf
) -> gcl.MaskGroup:
    branch = gcl.MaskGroup(agg_op="or")  # type: ignore
    if isinstance(flat, dict):
        for key, nest in flat.items():
            if key not in conf.roots:
                continue
            branch[key] = _make_tree(nest, conf)
            branch[key]["latent"] = (  # type: ignore
                branch[key].data != conf.desc[key].data
            )
    else:
        no_hard = True
        for parton in flat:
            if parton in conf.hard:
                no_hard = False
                branch[parton] = _make_tree(conf.hard[parton], conf)
                branch[parton]["latent"] = (  # type: ignore
                    branch[parton].data != conf.desc[parton].data
                )
            else:
                branch[parton] = conf.desc[parton]
        if (conf.use_pmu is True) and (no_hard is True):
            branch_pdgs = tuple(conf.name_to_pdg[k] for k in branch.keys())
            parton_mask = conf.hard_pdg.mask(
                branch_pdgs, blacklist=False, sign_sensitive=True
            )
            parton_names = tuple(map(str, conf.hard_pdg[parton_mask].name))
            parton_pmu = conf.hard_pmu[parton_mask]
            overlap = branch.bitwise_and
            dR = parton_pmu.delta_R(conf.pmu[overlap])
            alloc = np.argmin(dR, axis=0)
            for num, name in enumerate(parton_names):
                alloc_mask = alloc == num
                data = branch[name].data[overlap]
                data[~alloc_mask] = False
                branch[name].data[overlap] = data
    return branch


def _leaf_mask_iter(
    branch: ty.Union[gcl.MaskGroup, gcl.MaskArray],
    branch_name: str,
) -> ty.Generator[ty.Tuple[str, gcl.MaskArray], None, None]:
    """Recursive function, traversing a branch of the nested mask tree
    structure from ``hierarchy()``, and yielding the leaves.
    """
    if isinstance(branch, gcl.MaskArray):
        yield branch_name, branch
    else:
        for name, mask in branch.items():
            if name == "latent":
                continue
            # TODO: look into contravariant type for this
            yield from _leaf_mask_iter(mask, name)  # type: ignore


def leaf_masks(mask_tree: gcl.MaskGroup) -> gcl.MaskGroup:
    """Find the leaves of the hard process, when organised into a
    hierarchical tree from ``hierarchy()``.

    :group: select

    Parameters
    ----------
    mask_tree : MaskGroup
        Nested masks representing the descedants from partons in the
        hard process.

    Returns
    -------
    leaves : MaskGroup
        Flat ``MaskGroup`` of only the leaves of ``mask_tree``.
    """
    mask_group = gcl.MaskGroup(agg_op="or")  # type: ignore
    for name, branch in mask_tree.items():
        mask_group.update(dict(_leaf_mask_iter(branch, name)))  # type: ignore
    return mask_group


def hard_edge(graph: gcl.Graphicle, pdg: int) -> ty.Tuple[int, int]:
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
    combos = it.combinations(masks.dict.values(), 2)
    pair_checks = map(np.bitwise_and, *zip(*combos))
    overlaps: bool = np.bitwise_or.reduce(tuple(pair_checks), axis=None)
    return overlaps
