"""
``graphicle.select``
====================

Utilities for selecting elements from graph structured particle data.
"""
import collections as cl
import functools as fn
import itertools as it
import operator as op
import typing as ty

import awkward as ak
import fastjet as fj
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.sparse.csgraph import breadth_first_tree as _breadth_first_tree

import graphicle as gcl

from . import base

__all__ = [
    "find_vertex",
    "vertex_descendants",
    "hard_descendants",
    "any_overlap",
    "hierarchy",
    "partition_descendants",
    "hadron_vertices",
    "fastjet_clusters",
    "leaf_masks",
    "centroid_prune",
    "color_singlets",
    "clusters",
    "arg_closest",
    "monte_carlo_tag",
]


DistFunc = ty.Callable[
    [gcl.MomentumArray, gcl.MomentumArray], base.DoubleVector
]
MaskType = ty.Union[gcl.MaskArray, gcl.MaskGroup]
MaskGeneric = ty.TypeVar("MaskGeneric", gcl.MaskGroup, gcl.MaskArray, MaskType)


def _param_check(
    param: ty.Any, name: str, expected: ty.Type
) -> ty.Optional[ty.NoReturn]:
    if not isinstance(param, expected):
        received = type(param)
        raise ValueError(
            f"Expected {name} to be {expected}. Received {received}."
        )


def fastjet_clusters(
    pmu: gcl.MomentumArray,
    radius: float,
    p_val: float,
    pt_cut: ty.Optional[float] = None,
    eta_cut: ty.Optional[float] = None,
    top_k: ty.Optional[int] = None,
) -> ty.List[gcl.MaskArray]:
    """Clusters particles using the **FastJet** implementation of the
    *generalised-kt algorithm*.

    :group: select

    .. versionadded:: 0.2.3
       Migrated from ``graphicle.calculate.cluster_pmu()``.

    Parameters
    ----------
    pmu: MomentumArray
        Four-momenta, :math:`p_\\mu`, of particles in the point cloud.
    radius : float
        The radius, :math:`R`, of the clusters to be produced.
    p_val : float
        The exponent parameter determining the transverse momentum,
        :math:`p_T`, dependence of iterative pseudojet merges. Positive
        values cluster low :math:`p_T` particles first, positive values
        cluster high :math:`p_T` particles first, and a value of zero
        corresponds to no :math:`p_T` dependence.
    pt_cut : float, optional
        Jet transverse momentum threshold. Jets with :math:`p_T` below
        this are discarded.
    eta_cut : float, optional
        Jet pseudorapidity, :math:`\\eta`, threshold. Jets with
        :math:`|\\eta|` above this are discarded.
    top_k : int, optional
        Only return a maximum ``top_k`` number of jets, sorted by
        transverse momentum. *ie.* if ``top_k`` is 3, only 3 jets with
        highest :math:`p_T` will be given. If ``top_k`` exceeds the
        number of jets reconstructed, all of the jets will be included.

    Returns
    -------
    list[MaskArray]
        List containing masks over the input data for each jet
        clustering, in order of descending :math:`p_T`.

    Raises
    ------
    ValueError
        When a negative value is passed to ``eta_cut``, ``pt_cut``, or
        ``radius``. Additionally, when ``top_k`` is passed as either a
        non-integer, or with a value less than one.

    Notes
    -----
    ``p_val`` set to ``-1`` gives **anti-kT**, ``0`` gives
    **Cambridge-Aachen**, and ``1`` gives **kT** clusterings.

    To prevent expensive repeated memory allocations, the underlying
    masks are stored as a single contiguous array, where each row is the
    data for the respective ``MaskArray`` in the output list. This may
    cause undefined behaviour if you apply views on the underlying data
    in a ``MaskArray`` without copying it.
    """
    if pt_cut is None:
        pt_cut = 0.0
    elif pt_cut < 0.0:
        raise ValueError("pt_cut must be non-negative.")
    if radius < 0.0:
        raise ValueError("radius must be non-negative.")
    if (top_k is not None) and ((top_k < 1) or (not isinstance(top_k, int))):
        raise ValueError(
            "top_k must be an integer with a value greater than zero."
        )
    num_pcls = len(pmu)
    pmu_renamed = pmu.data[["e", "x", "y", "z"]].copy()
    pmu_renamed.dtype.names = "E", "px", "py", "pz"
    pmu_ak = ak.from_numpy(pmu_renamed)
    extra_param = tuple()
    if p_val == -1:
        algo = fj.antikt_algorithm
    elif p_val == 0:
        algo = fj.cambridge_algorithm
    elif p_val == 1:
        algo = fj.kt_algorithm
    else:
        algo = fj.genkt_algorithm
        extra_param = (p_val,)
    jet_def = fj.JetDefinition(algo, radius, *extra_param)
    sequence = fj.ClusterSequence(pmu_ak, jet_def)
    jets = sequence.inclusive_jets(pt_cut)
    jet_pmus = gcl.MomentumArray(jets.to_numpy())
    pt_descend_idxs = np.argsort(jet_pmus.pt)[::-1].tolist()
    jet_idxs_ = sequence.constituent_index(pt_cut)[pt_descend_idxs]
    if eta_cut is not None:
        if eta_cut < 0.0:
            raise ValueError("eta_cut must be non-negative.")
        jet_pmus = jet_pmus[pt_descend_idxs]
        jet_idxs_ = jet_idxs_[np.abs(jet_pmus.eta) < eta_cut]
    jet_idxs_ = jet_idxs_[:top_k]
    jet_idxs = jet_idxs_.to_list()
    num_jets = len(jet_idxs)
    cluster_buffer = np.zeros((num_jets, num_pcls), dtype=np.bool_)
    for mask, jet_idx in zip(cluster_buffer, jet_idxs):
        mask[jet_idx] = True
    return list(map(gcl.MaskArray, cluster_buffer))


def find_vertex(
    graph: gcl.Graphicle,
    pdgs_in: ty.Optional[ty.Set[int]] = None,
    pdgs_out: ty.Optional[ty.Set[int]] = None,
) -> base.IntVector:
    """Locate vertices with the inward and outward particles of the
    passed pdg codes.

    :group: select

    .. versionadded:: 0.1.0

    Parameters
    ----------
    graph : Graphicle
        ``Graphicle`` object, which must contain at least edge and pdg
        data.
    pdgs_in : set of ints
        Subset of pdg codes to match against the incoming particles.
    pdgs_out : set of ints
        Subset of pdg codes to match against the outgoing particles.

    Returns
    -------
    ndarray[int32]
        List the vertex ids which match the passed incoming and outgoing
        pdg codes.

    Raises
    ------
    ValueError
        Raised if ``pdgs_in`` and ``pdgs_out`` are both left blank.
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

        pcls_in = vtx_pdg_pivot("src")
        pcls_out = vtx_pdg_pivot("dst")
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
    """Returns a ``MaskArray`` to select edges which descend from a
    given interaction vertex.

    :group: select

    .. versionadded:: 0.1.0

    .. versionchanged:: 0.2.7
       In the edge case of no descendants, now returns a mask with
       identifying edges whose source is the vertex, rather than raising
       an unhandled ``IndexError``.

    Parameters
    ----------
    adj : AdjacencyList
        Topological structure of the graph.
    vertex : int
        The vertex id from which the descending edges are identified.

    Returns
    -------
    MaskArray
        Boolean mask over the graphicle objects associated with the
        passed AdjacencyList.
    """
    vertex_mask = adj.edges["src"] == vertex
    if not np.any(vertex_mask):
        return gcl.MaskArray(adj.edges["dst"] == vertex)
    sparse = adj._sparse_signed
    vertex = sparse.row[vertex_mask][0]
    bft = _breadth_first_tree(adj._sparse_csr, vertex)
    mask = np.isin(sparse.row, bft.indices)
    mask[sparse.row == vertex] = True  # include edges directly from parent
    return gcl.MaskArray(mask)


def edge_descendants(
    adj: gcl.AdjacencyList, edge: gcl.VertexPair
) -> gcl.MaskArray:
    """Returns a ``MaskArray`` to select particles which descend from a
    given edge in the DAG. This mask includes the parent edge.

    :group: select

    .. versionadded:: 0.2.8

    Parameters
    ----------
    adj : AdjacencyList
        Topological structure of the graph.
    edge : VertexPair or tuple[int, int]
        The vertex id from which the descending edges are identified.

    Returns
    -------
    MaskArray
        Boolean mask over the graphicle objects associated with the
        passed AdjacencyList.
    """
    desc = vertex_descendants(adj, edge[1])
    desc[adj == edge] = True
    return desc


def hadron_vertices(
    adj: gcl.AdjacencyList,
    status: gcl.StatusArray,
) -> ty.Tuple[int, ...]:
    """Locates the hadronisation vertices in the generation DAG.

    :group: select

    .. versionadded:: 0.1.11

    Parameters
    ----------
    adj : AdjacencyList
        The adjacency list of the generation DAG.
    status : StatusArray
        The status codes associated with each particle in the event.

    Returns
    -------
    tuple[int, ...]
        Indices of the hadronisation vertices in the generation DAG,
        returned in no particular order.
    """
    vtx_arr = np.unique(adj.edges[status.in_range(80, 90)]["src"])
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
    in_parton_masks = map(np.equal, vtxs, it.repeat(adj.edges["dst"]))
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
    mask : MaskArray
        Hard parton descendants.
    pcls_in : MaskArray or ndarray[bool_]
        The particles entering the hadronisation vertex.
    vtx_desc : MaskArray
        Particles descending from the hadronisation vertex.
    final : MaskArray
        Final state particles.
    pmu : MomentumArray
        Four momenta.
    dist_strat : callable
        Callable which takes two ``MomentumArray`` instances, and
        returns a double array with number of rows and columns equal to
        the lengths of the input momenta, respectively. Output should
        represent pairwise distance between particles incident on the
        hadronisation vertex, and the final state descendants.

    Returns
    -------
    MaskArray
        Input ``MaskArray``, filtered to remove background incident on
        the same hadronisation vertex.
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
    hier: gcl.MaskGroup[MaskGeneric],
    pt_exp: float = -0.1,
) -> gcl.MaskGroup[MaskGeneric]:
    """Partitions the final state descendants with mixed hard partonic
    heritage, by aligning them with their nearest ancestor.

    :group: select

    .. versionadded:: 0.1.11

    Parameters
    ----------
    graph : Graphicle
        The generation DAG of the event.
    hier : MaskGroup
        Nested ``MaskGroup`` tree structure representing the hierarchy
        of the hard process, obtained with ``hierarchy()``.
    pt_exp : float
        Exponent to raise the transverse momentum weighting on the
        distances. Passing ``pt_exp=0.0`` will result in standard
        delta R distances from partons to hadrons. Setting this to a
        negative value ensures infrared safety. High negative values
        will bias high transverse momentum partons to claim all of the
        hadrons. Default is ``-0.1``.

    Returns
    -------
    MaskGroup
        Same nested tree structure as input, but with the final
        state hadrons partitioned to their nearest hard parton ancestor.
    """
    dist_strat = fn.partial(gcl.matrix.parton_hadron_distance, pt_exp=pt_exp)
    hadron_vtxs = _hadron_vtx_parton_iter(
        graph.adj, graph.status, hier.bitwise_or()
    )
    hier = hier.copy()
    for vtx_id, pcls_in in hadron_vtxs:
        vtx_desc = vertex_descendants(graph.adj, vtx_id)
        for name, branch in hier.items():
            for _, mask in _leaf_mask_iter(name, branch):
                if vtx_id not in graph.edges["dst"][mask]:
                    continue
                mask.data = _partition_vertex(
                    mask,
                    pcls_in,
                    vtx_desc,
                    graph.final,
                    graph.pmu,
                    dist_strat,
                ).data
    return hier


def _hard_ancestor_matrix(
    hard_desc: gcl.MaskGroup[MaskType],
) -> base.BoolVector:
    """Returns a directed adjacency matrix for the hard process.

    Parameters
    ----------
    hard_desc : MaskGroup
        Collection of masks indicating descendants from hard partons,
        over the hard process only. Obtain with ``hard_descendants()``
        and then applying ``StatusArray.hard_mask`` (with ``'incoming'``
        removed).

    Returns
    -------
    ndarray[bool_]
        Square directed adjacency matrix, linking partons in the hard
        process to their descendants. Rows represent each hard parton,
        and columns indicate which hard partons descend from it
        (including descendants-of-descendants).
    """
    num_hard = len(hard_desc)
    mat = np.zeros((num_hard, num_hard), dtype="<?")
    for i, mask in enumerate(hard_desc.values()):
        mat[i, :] = mask.data
    np.fill_diagonal(mat, False)  # no self-parenting
    return mat


def _hard_parent_matrix(hard_matrix: base.BoolVector) -> base.BoolVector:
    """Returns an adjacency matrix over the hard process of parents to
    their direct children. Filters out grandchildren *etc*.

    Parameters
    ----------
    hard_matrix : ndarray[bool_]
        Directed adjacency matrix representing the links from ancestors
        to descendants (including descendants-of-descendants *etc*.)
        Obtain with ``_hard_ancestor_matrix()``.

    Returns
    -------
    ndarray[bool_]
        Directed adjacency matrix representing links from parents to
        direct (first generation) children. Same as ``hard_matrix``, but
        with grandchildren *etc.* removed.

    Notes
    -----
    This function has side-effects on the input. In fact, the return
    value is the same object in memory as ``hard_matrix``, so could
    technically be discarded entirely. If this behaviour is problematic,
    pass ``hard_matrix.copy()`` instead.
    """
    pcls_with_hard_parent_dict = filter(np.any, hard_matrix)
    asc_parents = sorted(pcls_with_hard_parent_dict, key=np.sum)
    row_pairs = it.product(reversed(asc_parents), asc_parents)
    for row_i, row_j in filter(lambda pair: op.is_not(*pair), row_pairs):
        row_i[row_j] = False
    return hard_matrix


def _hard_parent_dict(
    comp_mat: base.BoolVector,
    names: ty.Tuple[str, ...],
) -> ty.Dict[str, ty.Tuple[str, ...]]:
    """Creates a mapping between the names of parent partons in the hard
    process to the names of their children.

    Parameters
    ----------
    comp_mat : ndarray[bool_]
        Directed adjacency matrix for the hard process, with edges from
        parent hard partons to their immediate children. Each row
        represents a parent parton, where columns represent the children
        they are linked to. Obtain this with ``_hard_parent_matrix()``.
    names : tuple[str, ...]
        Names of the hard partons, appearing in the same order as the
        corresponding rows / columns of adjacency matrix representing
        the hard process (and therefore same order as ``comp_mat``).

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping from parents to children in the hard process. Keys are
        parent names, and values are tuples of children names.
    """
    parents = filter(lambda parton: np.any(parton[1]), zip(names, comp_mat))
    parents = it.tee(parents, 2)
    parent_names = map(op.itemgetter(0), parents[0])
    children_masks = map(op.itemgetter(1), parents[1])
    children_names = map(it.compress, it.repeat(names), children_masks)
    children_names = map(tuple, children_names)
    return dict(zip(parent_names, children_names))


def _flat_hierarchy(
    hard_desc: gcl.MaskGroup[gcl.MaskArray],
) -> ty.Dict[str, ty.Tuple[str, ...]]:
    """Produces a mapping between partons to their direct children in
    the hard process. This mapping is flat, *ie.* there is no nesting
    of dictionaries in dictionaries. Children of one parton may be
    parents of another, and so may be both included in a child tuple
    and as a parent key. There is no distinction between grandparents
    and parents, *etc*.

    Parameters
    ----------
    hard_desc : MaskGroup
        Collection of masks indicating descendants from hard partons,
        over the hard process only. Obtain with ``hard_descendants()``
        and then applying ``StatusArray.hard_mask`` (with ``'incoming'``
        removed).

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping from parents to children in the hard process. Keys are
        parent names, and values are tuples of children names.

    Notes
    -----
    In rare cases, when several particles with the same PDG code are
    present in the hard process, the particle names will be appended
    with a colon followed by a numerical index, *eg.* ``'b~:0'``. See
    ``_pdgs_to_keys()`` for more information.
    """
    names = tuple(hard_desc.keys())
    ancestor_matrix = _hard_ancestor_matrix(hard_desc)
    parent_matrix = _hard_parent_matrix(ancestor_matrix)
    return _hard_parent_dict(parent_matrix, names)


def _pdgs_to_keys(pdg: gcl.PdgArray) -> ty.Tuple[str, ...]:
    """Converts a ``PdgArray`` into a tuple of unique strings, which
    starts with the particle name, and if a PDG code has duplicates,
    they will be numerically indexed by appending ``:n``, where ``n``
    is the index of each duplicate, starting at ``:0`` for the first
    occurrence.
    """
    names = tuple(map(str, pdg.name))
    pdg_counter = cl.Counter(names)
    pdg_counts = map(pdg_counter.__getitem__, names)
    has_dup = tuple(map(fn.partial(op.lt, 1), pdg_counts))
    if not any(has_dup):
        return names
    counters = cl.defaultdict(it.count)
    nums = map(next, map(counters.__getitem__, names))
    renames = map(":".join, zip(names, map(str, nums)))
    names = it.starmap(
        lambda name, rename, dup: rename if dup else name,
        zip(names, renames, has_dup),
    )
    return tuple(names)


def hard_descendants(
    graph: gcl.Graphicle,
    target: ty.Optional[ty.Iterable[int]] = None,
    sign_sensitive: bool = False,
    strict: bool = True,
) -> gcl.MaskGroup[gcl.MaskArray]:
    """Returns a ``MaskGroup`` instance to select particle descendants
    of ``target`` hard partons (by PDG code).

    :group: select

    .. versionadded:: 0.1.0

    .. versionchanged:: 0.1.11
       ``target`` parameter now optional.

    .. versionchanged:: 0.2.5
       ``strict`` parameter added.

    Parameters
    ----------
    graph : Graphicle
        Particle graph containing at least ``PdgArray`` and
        ``StatusArray``.
    target : Iterable[int], optional
        PDG codes referring to the hard particles for which descendants
        are obtained. If ``None``, will obtain masks for all
        intermediate and outgoing hard partons. Default is ``None``.
    sign_sensitive : bool
        Indicates whether sign of PDG code should be used when selecting
        hard partons, *ie.* if set to ``False`` both particle and
        anti-particle partons will be masked, whereas if ``True`` only
        the partons explicitly matching the target sign will be
        considered. Default is ``False``.
    strict : bool
        If ``True`` all PDGs in ``target`` must be present in the hard
        process. If ``False``, only a subset need be present. Default
        is ``True``.

    Returns
    -------
    MaskGroup[MaskArray]
        Collection of masks over the event indicating the descendants
        of partons from within the hard process.

    Raises
    ------
    ValueError
        If PDG codes required by ``target`` and ``strict`` are absent
        from the partons in the hard process.
    """
    hard_mask = graph.hard_mask
    del hard_mask["incoming"]
    hard_graph = graph[hard_mask]
    if target is not None:
        target_pdgs = set(target) if sign_sensitive else set(map(abs, target))
        target_mask = hard_graph.pdg.mask(
            target=tuple(target_pdgs),
            blacklist=False,
            sign_sensitive=sign_sensitive,
        )
        if not np.any(target_mask):
            raise ValueError("No target PDGs found in hard process.")
        found_pdgs_ = hard_graph.pdg[target_mask]
        if not sign_sensitive:
            found_pdgs_ = np.abs(found_pdgs_)
        found_pdgs = set(found_pdgs_)
        missing_pdgs = target_pdgs - found_pdgs
        if strict and missing_pdgs:
            missing_str = ", ".join(map(str, missing_pdgs))
            raise ValueError(f"Missing PDGs in hard process: {missing_str}.")
        hard_graph = hard_graph[target_mask]
    pdg_keys = _pdgs_to_keys(hard_graph.pdg)
    pcl_out_vtxs = map(int, hard_graph.edges["dst"])
    descs = map(vertex_descendants, it.repeat(graph.adj), pcl_out_vtxs)
    group = gcl.MaskGroup(cl.OrderedDict(zip(pdg_keys, descs)), agg_op="or")
    for key, idx in zip(pdg_keys, np.flatnonzero(hard_mask)):
        group[key][idx] = True
    return group


def hierarchy(
    graph: gcl.Graphicle,
    desc: ty.Optional[gcl.MaskGroup[gcl.MaskArray]] = None,
) -> gcl.MaskGroup[MaskType]:
    """Composite ``MaskGroup`` of ``MaskGroup`` instances, representing
    the partons of the hard process and their descendants. Uses a tree
    structure, such that partons which are descendants of other hard
    partons are accessible, and nested within their parents.

    :group: select

    .. versionadded:: 0.1.11

    .. versionchanged:: 0.2.13
       Patch: works for hard processes with no intermediate states,
       *eg.* ``g g > j j``.

    Parameters
    ----------
    graph : Graphicle
        The event to be parsed. Must include ``status``, ``edges``,
        ``pmu``, and ``pdg`` data.
    desc : MaskGroup[MaskArray], optional
        If the masks for all partons in the hard process have already
        been computed, these may be passed here to save performing the
        computation again.

    Returns
    -------
    MaskGroup[MaskGroup | MaskArray]
        Nested composite of ``MaskGroup`` instances, representing the
        hierarchical structure of the hard process, and the descendants
        of the hard process partons throughout the shower. Nested
        ``MaskGroup`` instances additionally contain a ``latent``
        ``MaskArray``, referring to the parent parton and its
        descendants which are not also descendants of the children
        partons.

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
        >>> # latent contains the mask not from constituents
        ... graph[masks["t"]["latent"]].pdg.name
        array(['t', 't', 't', 't', 't'], dtype=object)

    Notes
    -----
    This function will not separate the mixed heritage when two sibling
    hard partons share ancestry for a given particle. In order to
    partition the resulting structure, use ``partition_descendants()``.
    """
    hard_mask = graph.hard_mask
    del hard_mask["incoming"]
    hard_pdg = graph.pdg[hard_mask]
    names = _pdgs_to_keys(hard_pdg)
    if desc is None:
        desc = hard_descendants(graph)
    else:
        desc = desc.copy()
    if not np.any(hard_mask["intermediate"]):
        return desc
    hard_desc = desc[hard_mask][list(names)]
    hard = _flat_hierarchy(hard_desc)
    keys = set(hard.keys())
    vals = set(it.chain.from_iterable(hard.values()))
    roots = keys.difference(keys.intersection(vals))
    hard_final = hard_mask & graph.final
    if np.any(hard_final):
        hard_final_keys = it.compress(names, hard_final[hard_mask].data)
        hard_final_idxs = np.flatnonzero(hard_final)
        for key, idx in zip(hard_final_keys, hard_final_idxs):
            desc[key].data[idx] = True
    masks = _make_tree(hard, roots, hard, desc, graph.adj)
    return masks


def _make_tree(
    flat: ty.Union[ty.Dict[str, ty.Any], ty.Tuple[str, ...]],
    roots: ty.Set[str],
    hard: ty.Dict[str, ty.Tuple[str, ...]],
    desc: gcl.MaskGroup[gcl.MaskArray],
    adj: gcl.AdjacencyList,
) -> gcl.MaskGroup[MaskType]:
    """Recursive function to convert a flat representation of the hard
    process tree into a nested ``MaskGroup`` of ``MaskGroup`` object.

    Parameters
    ----------
    flat : dict[str, tuple[str, ...]]
        Flat representation of a branch of the hard process tree. At the
        first level before entering recursion, this is the same as the
        value for ``hard``.
    roots : set[str]
        String valued keys at the root of the hard process tree.
    hard : dict[str, tuple[str, ...]]
        Flat representation of the hard process tree. May be obtained
        from ``_flat_hierarchy()``.
    desc : MaskGroup
        The masks descending from the hard process, given by
        ``hard_descendants()``.
    """
    branch = gcl.MaskGroup(agg_op="or")  # type: ignore
    if isinstance(flat, dict):
        for key, nest in filter(lambda item: item[0] in roots, flat.items()):
            branch[key] = _make_tree(nest, roots, hard, desc, adj)
            branch[key]["latent"] = (  # type: ignore
                branch[key].data != desc[key].data
            )
    else:
        for parton in flat:
            if parton in hard:
                branch[parton] = _make_tree(
                    hard[parton], roots, hard, desc, adj
                )
                branch[parton]["latent"] = (  # type: ignore
                    branch[parton].data != desc[parton].data
                )
            else:
                mask = desc[parton]
                in_edge = adj.edges["src"][mask][0]
                initial_parton_mask = adj.edges["dst"] == in_edge
                if np.sum(initial_parton_mask) == 1:
                    mask.data[initial_parton_mask] = True
                branch[parton] = mask
    return branch


def _leaf_mask_iter(
    branch_name: str,
    branch: MaskType,
    exclude_latent: bool = True,
) -> ty.Generator[ty.Tuple[str, gcl.MaskArray], None, None]:
    """Recursive function, traversing a branch of the nested mask tree
    structure from ``hierarchy()``, and yielding the leaves.
    """
    if isinstance(branch, gcl.MaskArray):
        yield branch_name, branch
    else:
        for name, mask in branch.items():
            if exclude_latent and name == "latent":
                continue
            yield from _leaf_mask_iter(name, mask, exclude_latent)


def leaf_masks(mask_tree: gcl.MaskGroup) -> gcl.MaskGroup[gcl.MaskArray]:
    """Find the leaves of the hard process, when organised into a
    hierarchical tree from ``hierarchy()``.

    :group: select

    .. versionadded:: 0.1.11

    .. versionchanged 0.3.7
       Output ``MaskGroup`` matches agg_op of ``mask_tree``.

    Parameters
    ----------
    mask_tree : MaskGroup
        Nested masks representing the descedants from partons in the
        hard process.

    Returns
    -------
    MaskGroup
        Flat ``MaskGroup`` of only the leaves of ``mask_tree``.
    """
    mask_group = gcl.MaskGroup(agg_op=mask_tree.agg_op)
    for name, branch in mask_tree.items():
        mask_group.update(dict(_leaf_mask_iter(name, branch)))  # type: ignore
    return mask_group


def any_overlap(masks: gcl.MaskGroup[MaskType]) -> bool:
    """Given a ``MaskGroup`` object, checks if any of the masks overlap
    with each other.

    :group: select

    .. versionadded:: 0.1.0

    Parameters
    ----------
    masks : MaskGroup
        Collection of boolean masks to check.

    Returns
    -------
    bool
        ``True`` if at least two ``MaskArrays`` in ``masks`` have at
        least one ``True`` element in the same location.
    """
    combos = it.combinations(masks.dict.values(), 2)
    pair_checks = map(np.bitwise_and, *zip(*combos))
    overlaps: bool = np.bitwise_or.reduce(tuple(pair_checks), axis=None)
    return overlaps


def centroid_prune(
    pmu: gcl.MomentumArray,
    radius: float,
    mask: ty.Optional[gcl.MaskArray] = None,
    centre: ty.Optional[ty.Tuple[float, float]] = None,
) -> gcl.MaskArray:
    """For a given ``MomentumArray``, calculate the distance every
    particle is from a centroid location, and return a ``MaskArray`` for
    all of the particles which are within a given ``radius``.
    If ``centre`` is not provided, the transverse momentum weighted
    centroid will be used.

    :group: select

    .. versionadded:: 0.2.4

    Parameters
    ----------
    pmu : MomentumArray
        Four-momenta, :math:`p_\\mu`, of particles in the set.
    radius : float
        Euclidean distance, or Radius :math:`R`, in the
        pseudorapidity-azimuth, :math:`\\eta-\\phi`, plane from the
        centroid, beyond which particles will be filtered out.
    mask : MaskArray, optional
        If provided, will apply the mask to the passed ``pmu``, and
        output ``MaskArray`` will have the same length.
    centre : tuple[float, float]
        Pseudorapidity and azimuth coordinates for a user-defined
        centroid.

    Returns
    -------
    MaskArray
        Mask which retains only the particles within ``radius`` of the
        centroid.
    """
    _param_check(pmu, "pmu", gcl.MomentumArray)
    if mask is not None:
        pmu = pmu[mask]
        event_mask = np.zeros_like(mask, "<?")
    if centre is None:
        eta_mid, phi_mid = gcl.calculate.resultant_coords(pmu, pseudo=True)
    else:
        eta_mid, phi_mid = centre
    phi_mid_ = np.exp(complex(0, phi_mid))
    dist = np.hypot(pmu.eta - eta_mid, np.angle(pmu._xy_pol * phi_mid_.conj()))
    is_within = dist < radius
    if mask is None:
        return gcl.MaskArray(is_within)
    event_mask[mask] = is_within
    return gcl.MaskArray(event_mask)


def color_singlets(
    leaves: gcl.MaskGroup[gcl.MaskArray],
    status: gcl.StatusArray,
    color: gcl.ColorArray,
    invert: bool = False,
) -> ty.List[ty.Tuple[str, ...]]:
    """Identifies groups of outgoing partons from the hard process which
    form color singlets.

    :group: select

    .. versionadded:: 0.2.8

    Parameters
    ----------
    leaves : MaskGroup[MaskArray]
        Innermost nested ``MaskArray`` instances, obtained from calling
        ``hierarchy()`` followed by ``leaf_masks()`` on a ``Graphicle``
        object.
    status : StatusArray
        The status codes for the event.
    color : ColorArray
        The color / anticolor pair values for the event.
    invert : bool
        If ``True``, will invert the operation to find all colored
        particles in the hard process which are not grouped with other
        partons in the hard-process to form a color-singlet. Default is
        ``False``.

    Returns
    -------
    list[tuple[str, ...]]
        Tuples of keys, grouping the color-connected partons in the hard
        process.

    Notes
    -----
    This function checks for color singlets formed from two color
    triplets (quarks), or two color triplets plus a color octet (gluon).
    It does not check for color singlets formed from three colour
    triplets. This functionality *may* be added in future.
    """
    hard_mask = status.hard_mask["outgoing"]
    masks = map(op.and_, it.repeat(hard_mask), leaves.values())
    colors_ = map(it.compress, it.repeat(color), masks)
    colors = tuple(it.chain.from_iterable(colors_))
    colored_mask = tuple(map(op.ne, it.repeat((0, 0)), colors))
    colored_only = it.compress(colors, colored_mask)
    keys = tuple(it.compress(leaves.keys(), colored_mask))
    named_colors = tuple(zip(keys, colored_only))
    combos = it.chain(
        it.combinations(named_colors, 2), it.combinations(named_colors, 3)
    )
    singlets = []
    for names, color_elems in it.starmap(zip, combos):
        color_set = set(map(op.attrgetter("color"), color_elems))
        anticolor_set = set(map(op.attrgetter("anticolor"), color_elems))
        if color_set == anticolor_set:
            singlets.append(names)
    if invert is True:
        members = set(it.chain.from_iterable(singlets))
        outcasts = set(keys).symmetric_difference(members)
        return list(zip(outcasts))
    return singlets


def clusters(
    graph: gcl.Graphicle, radius: float
) -> gcl.MaskGroup[gcl.MaskArray]:
    """Cluster and tag the final state particles in an event represented
    by a ``Graphicle`` object. These clusters are formed by considering
    the topology of the directed acyclic graph (DAG) generating the
    event, tracking descendants of hard partons, and the momenta of the
    hard partons compared against the final state particles.

    :group: select

    .. versionadded:: 0.2.8

    The steps defining this algorithm are summarised:

    * Find descendants of all hard partons within DAG
    * Remove final state radiation from descendants
    * Where descendants of multiple hard partons annihilate color with
      each other, assign exclusive parentage of subsequent color neutral
      particles to closest hard parton in the pseudorapidity-azimuth
      (:math:`\\eta-\\phi`) plane
    * Where background is used to annihilate color of hard parton
      descendants, remove all final state particles beyond a distance
      of ``radius`` from the position of the hard parton in the
      :math:`\\eta-\\phi` plane

    Parameters
    ----------
    graph : Graphicle
        Full event record as a DAG.
    radius : float
        Radius in the :math:`\\eta-\\phi` plane defining the enclosed
        clustering region around hard partons which are color-connected
        to the underlying event. See notes for more information.

    Returns
    -------
    MaskGroup[MaskArray]
        Flat ``MaskGroup``, containing ``MaskArray`` instances which
        reconstruct the hard partons from the final state particles.

    Notes
    -----
    Hard partons may be color connected with each other, or with the
    underlying event. *eg.* for a hierarchical clustering:

    .. code-block:: text

        MaskGroup(agg_op=OR)
        ├── t
        │   ├── b
        │   └── W+
        │       ├── c
        │       └── s~
        └── t~
            ├── b~
            └── W-
                ├── s
                └── c~


    the quarks decaying from the W bosons are color-connected to each
    other, as they form from a color-singlet. This means they will
    almost certainly annihilate their color with each other during
    hadronisation. However, the top quarks are color connected to the
    underlying event, and therefore will almost certainly annihilate
    their color with partons that do not descend from the hard process.
    This is done via proxy of the bottom quark, which inherits its color
    from the top. This results in background radiation in the
    descendants tree of the bottom quark, spread over a wide region of
    the :math:`\\eta-\\phi` plane. This is when the ``radius`` parameter
    is applied, excluding all final state descendants whose distance
    from the hard bottom quark exceeds the value passed, cleaning up the
    signal.
    """
    hier_ = hierarchy(graph)
    hier = partition_descendants(graph, hier_, -0.1)
    leaves = leaf_masks(hier)
    color_keys = color_singlets(leaves, graph.status, graph.color, True)
    colored_leaves = map(
        op.getitem, it.repeat(leaves), it.chain.from_iterable(color_keys)
    )
    colored_leaves, colored_leaves_ = it.tee(colored_leaves, 2)
    hard_mask = graph.hard_mask["outgoing"]
    parton_masks = map(op.and_, it.repeat(hard_mask), colored_leaves_)
    parton_pmus = map(op.getitem, it.repeat(graph.pmu), parton_masks)
    parton_centroids = map(op.attrgetter("eta", "phi"), parton_pmus)
    for leaf, centroid in zip(colored_leaves, parton_centroids):
        centroid = tuple(map(op.methodcaller("item"), centroid))
        leaf.data[...] = centroid_prune(graph.pmu, radius, leaf, centroid).data
    hier.recursive_drop(inplace=True)
    flat_hier = hier.flatten("rise")
    flat_hier_final = map(op.itemgetter(graph.final), flat_hier.values())
    return gcl.MaskGroup(
        cl.OrderedDict(zip(flat_hier.keys(), flat_hier_final)), agg_op="or"
    )


def arg_closest(
    focus: gcl.MomentumArray,
    candidate: gcl.MomentumArray,
    num_threads: int = 1,
) -> ty.List[int]:
    """Assigns four-momenta elements in ``candidate`` to the nearest
    four-momenta elements in ``focus``. Elements in ``candidate`` are
    assigned to one element in ``focus`` only.

    :group: select

    .. versionadded:: 0.2.14

    .. versionchanged:: 0.3.8
       Modified the distance metric to include transverse momentum.

    Parameters
    ----------
    focus : MomentumArray
        Four-momenta of objects to receive assignments to nearest
        ``candidate`` objects.
    candidate : MomentumArray
        Four-momenta of candidate objects to draw from until ``focus``
        objects have each received an assignment.
    num_threads : int
        Number of threads to parallelise the cost matrix computation
        over. Default is 1.

    Returns
    -------
    list[int]
        Indices of elements in ``candidate`` assigned to each respective
        element in ``focus``. This will be the same length as ``focus``.

    See Also
    --------
    monte_carlo_tag : MC truth parton assignment to particle clustering.

    Notes
    -----
    Since only one ``focus`` element can receive a given ``candidate``
    element, this must be regarded a cost-minimisation problem. The
    costs here are the distances in the azimuth-rapidity plane between
    elements. To illustrate, if one ``candidate`` element is the closest
    for two ``focus`` elements, it must be assigned to the smaller
    distance of the two, and the remaining ``focus`` element must be
    assigned the next-nearest ``candidate`` element. This is equivalent
    to the Assignment Problem [1]_ for a complete bipartite graph, and
    uses SciPy's modified *Jonker-Volgenant algorithm* with no
    initialisation ref. [2]_ under the hood.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Assignment_problem

    .. [2] DF Crouse. On implementing 2D rectangular assignment
       algorithms. *IEEE Transactions on Aerospace and Electronic
       Systems*, 52(4):1679-1696, August 2016,
       :doi:`10.1109/TAES.2016.140952`
    """
    with gcl.calculate._thread_scope(num_threads):
        cost_matrix = gcl.calculate._assignment_cost(
            focus.rapidity,
            candidate.rapidity,
            focus._xy_pol,
            candidate._xy_pol,
        )
    del candidate.rapidity  # inplace operation above invalidates cache
    _, idxs = opt.linear_sum_assignment(cost_matrix)
    return idxs.tolist()


def monte_carlo_tag(
    particles: gcl.ParticleSet,
    cluster_masks: ty.Sequence[gcl.MaskArray],
    clustered_pmu: ty.Optional[gcl.MomentumArray] = None,
    intermediate: bool = False,
    outgoing: bool = True,
    sign_sensitive: bool = False,
    blacklist: ty.Optional[ty.Sequence[int]] = None,
    whitelist: ty.Optional[ty.Sequence[int]] = None,
) -> gcl.MaskGroup[gcl.MaskArray]:
    """Assigns clusters to nearest Monte-Carlo truth partons in the hard
    process. Clusters are drawn from ``cluster_masks``, until each
    each parton is assigned.

    :group: select

    .. versionadded:: 0.2.14

    .. versionchanged:: 0.3.6
       Added ``clustered_pmu`` parameter, enabling tagging on clusters
       with detector-level cuts.

    Parameters
    ----------
    particles : ParticleSet
        Monte-Carlo particle data record for the whole event.
    cluster_masks : Sequence[MaskArray]
        Boolean masks identifying which particles belong to each of the
        clusterings. These are defined over the final particles, or
        clustered_pmu, see below.
    clustered_pmu : MomentumArray, optional
        MomentumArray containing the data as passed to the clustering
        algorithm. This is useful for when cuts have been applied before
        clustering. If unset, the final state momentum for the whole
        event will be assumed.
    intermediate : bool
        If ``True`` includes partons from the intermediate stage of the
        hard process. Default is ``False``.
    outgoing : bool
        If ``True`` includes partons from the outgoing stage of the
        hard process. Default is ``True``.
    sign_sensitive : bool
        If ``False``, the sign of PDG codes in the ``blacklist`` are
        ignored. **ie.** Particles and anti-particles are not
        distinguished. Default is ``False``.
    blacklist : Sequence[int], optional
        A sequence of PDG codes, identifying particles in the hard
        process which should not be assigned a cluster.
    whitelist : Sequence[int], optional
        A sequence of PDG codes, identifying particles in the hard
        process which exclusively should be assigned clusters.

    Returns
    -------
    MaskGroup[MaskArray]
        Mapping of the particle names within the hard process to the
        assigned closest clusters, with ``agg_op=OR``.

    Raises
    ------
    ValueError
        If ``intermediate`` and ``outgoing`` are simultaneously set to
        ``False``, or ``blacklist`` and ``whitelist`` are simultaneously
        not ``None``.
    ValueError
        If cluster_masks is empty. Additionally, if the elements have a
        size mismatch with either the number of particles in the final
        state, or the length of clustered_pmu when passed.
    IndexError
        If after applying ``blacklist`` or ``whitelist``, no matching
        partons remain in the hard process.

    See Also
    --------
    arg_closest : indices of closest 4-momenta objects between two sets.

    Examples
    --------
    Boosted top decay, clustered with anti-kt, and tagged with MC truth:

        >>> print(gcl.select.hierarchy(graph))  # hard process tree
        MaskGroup(agg_op=OR)
        ├── W+
        │   ├── e+
        │   ├── nu(e)
        │   └── latent
        └── t~
            ├── W-
            │   ├── d
            │   ├── u~
            │   └── latent
            ├── b~
            └── latent
        >>> final_pmu = graph.pmu[graph.final]
        ... clusters = gcl.select.fastjet_clusters(  # anti-kt clusters
        ...     pmu=final_pmu,
        ...     radius=0.3,
        ...     p_val=-1,
        ...     eta_cut=3.0,
        ...     pt_cut=10.0,
        ...     top_k=10,
        ... )
        ... tagged_clusters = gcl.select.monte_carlo_tag(
        ...     particles=graph.particles,
        ...     cluster_masks=clusters,
        ...     blacklist=[11, 12],
        ... )
        ... tagged_clusters  # subset of anti-kt tagged to MC truth
        MaskGroup(masks=["b~", "d", "u~"], agg_op=OR)
        >>> # calculate combined mass of clusters from top quark
        ... np.sum(final_pmu[tagged_clusters], axis=0).mass
        array([163.33889956])
    """
    if not cluster_masks:
        raise ValueError("cluster_masks is an empty sequence.")
    portions = []
    if outgoing:
        portions.append("outgoing")
    if intermediate:
        portions.append("intermediate")
    if not portions:
        raise ValueError(
            "At least one of either intermediate or outgoing must be True."
        )
    portion_mask = particles.status.hard_mask[portions].data
    hard_pmu = particles.pmu[portion_mask]
    hard_pdg = particles.pdg[portion_mask]
    if blacklist and whitelist:
        raise ValueError("Cannot simultaneously blacklist and whitelist.")
    if list_ := (blacklist or whitelist):
        hard_pdg_ = hard_pdg
        if not sign_sensitive:
            hard_pdg_ = np.abs(hard_pdg_)
            list_ = np.abs(list_)
        hard_mask = np.isin(hard_pdg_, list_, invert=(whitelist is None))
        if not np.any(hard_mask):
            raise IndexError("No partons matching filters found.")
        hard_pmu = hard_pmu[hard_mask]
        hard_pdg = hard_pdg[hard_mask]
    ref_length = "clustered_pmu"
    if clustered_pmu is None:
        clustered_pmu = particles.pmu[particles.final]
        ref_length = "particles.final"
    if len(cluster_masks) < len(hard_pmu):
        raise ValueError(
            f"shape mismatch: only {len(cluster_masks)} clusters "
            f"passed to tag {len(hard_pmu)} partons."
        )
    if len(clustered_pmu) != len(cluster_masks[0]):
        raise ValueError(
            "shape mismatch: length of elements in cluster_masks must be the "
            f"same as the length of {ref_length}."
        )
    jets_pmu = gcl.calculate.aggregate_momenta(clustered_pmu, cluster_masks)
    idxs = arg_closest(hard_pmu, jets_pmu)
    tagged_clusters = op.itemgetter(*idxs)(cluster_masks)
    if len(idxs) == 1:
        tagged_clusters = (tagged_clusters,)
    names = _pdgs_to_keys(hard_pdg)
    return gcl.MaskGroup(dict(zip(names, tagged_clusters)), agg_op="or")
