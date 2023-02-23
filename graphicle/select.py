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

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import pyjet
from pyjet import ClusterSequence, PseudoJet
from scipy.sparse.csgraph import breadth_first_tree

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
]


DistFunc = ty.Callable[
    [gcl.MomentumArray, gcl.MomentumArray], base.DoubleVector
]


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
        The momenta of each particle in the point cloud.
    radius : float
        The radius of the clusters to be produced.
    p_val : float
        The exponent parameter determining the transverse momentum (pt)
        dependence of iterative pseudojet merges. Positive values
        cluster low pt particles first, positive values cluster high pt
        particles first, and a value of zero corresponds to no pt
        dependence.
    pt_cut : float, optional
        Jet transverse momentum threshold, below which jets will be
        discarded.
    eta_cut : float, optional
        Jet pseudorapidity threshold, above which jets will be
        discarded.
    top_k : int, optional
        Only return a maximum ``top_k`` number of jets, sorted by
        transverse momentum. *ie.* if ``top_k`` is 3, only 3 jets with
        highest pt will be given. If ``top_k`` exceeds the number of
        jets reconstructed, all of the jets will be included.

    Returns
    -------
    clusters : list[MaskArray]
        List containing masks over the input data for each jet
        clustering, in order of descending transverse momentum.

    Notes
    -----
    This is a wrapper around FastJet's implementation.

    ``p_val`` set to ``-1`` gives **anti-kT**, ``0`` gives
    **Cambridge-Aachen**, and ``1`` gives **kT** clusterings.
    """
    pmu_pyjet = pmu.data[["e", "x", "y", "z"]]
    pmu_pyjet.dtype.names = "E", "px", "py", "pz"
    pmu_pyjet_idx = rfn.append_fields(
        pmu_pyjet, "idx", np.arange(len(pmu_pyjet))
    )
    sequence: ClusterSequence = pyjet.cluster(
        pmu_pyjet_idx, R=radius, p=p_val, ep=True
    )
    jets: ty.Iterable[PseudoJet] = sequence.inclusive_jets()
    if pt_cut is not None:
        jets = filter(lambda jet: jet.pt > pt_cut, jets)
    if eta_cut is not None:
        jets = filter(lambda jet: abs(jet.eta) < eta_cut, jets)
    if top_k is not None:
        jets = it.islice(jets, top_k)
    jet_idxs = map(lambda j: list(map(op.attrgetter("idx"), j)), jets)
    mask_empty = gcl.MaskArray(np.zeros_like(pmu_pyjet, dtype="<?"))
    clusters: ty.List[gcl.MaskArray] = []
    for jet_idx in jet_idxs:
        mask = mask_empty.copy()
        mask[jet_idx] = True
        clusters.append(mask)
    return clusters


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
    """Returns a ``MaskArray`` to select particles which descend from a
    given interaction vertex.

    :group: select

    .. versionadded:: 0.1.0

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
    sparse = adj._sparse_signed
    vertex = sparse.row[vertex == adj.edges["in"]][0]
    bft = breadth_first_tree(sparse, vertex)
    mask = np.isin(sparse.row, bft.indices)
    mask[sparse.row == vertex] = True  # include parent vertex
    return gcl.MaskArray(mask)


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
    filtered_mask : MaskArray
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
    hier: gcl.MaskGroup,
    pt_exp: float = -0.1,
) -> gcl.MaskGroup:
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
    hier_parted : MaskGroup
        Same nested tree structure as input, but with the final
        state hadrons partitioned to their nearest hard parton ancestor.
    """
    dist_strat = fn.partial(gcl.matrix.parton_hadron_distance, pt_exp=pt_exp)
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
                    mask,
                    pcls_in,
                    vtx_desc,
                    graph.final,
                    graph.pmu,
                    dist_strat,
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
) -> gcl.MaskGroup:
    """Returns a ``MaskGroup`` instance to select particle descendants
    of ``target`` hard partons (by PDG code).

    :group: select

    .. versionadded:: 0.1.0

    .. versionchanged:: 0.1.11
       Target parameter now optional.

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
    """
    hard_vtxs = list()
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
        pdg_keys = _pdgs_to_keys(pcls.pdg)
        pcl_out_vtxs = map(int, pcls.edges["out"])
        hard_vtxs.extend(list(zip(pdg_keys, pcl_out_vtxs)))
    # find the descendants of those vertices
    masks = gcl.MaskGroup(agg_op="or")
    for pdg_key, vtx in hard_vtxs:
        masks[pdg_key] = vertex_descendants(graph.adj, vtx)
    return masks


def hierarchy(
    graph: gcl.Graphicle,
    desc: ty.Optional[gcl.MaskGroup] = None,
) -> gcl.MaskGroup:
    """Composite ``MaskGroup`` of ``MaskGroup`` instances, representing
    the partons of the hard process and their descendants. Uses a tree
    structure, such that partons which are descendants of other hard
    partons are accessible, and nested within their parents.

    :group: select

    .. versionadded:: 0.1.11

    Parameters
    ----------
    graph : Graphicle
        The event to be parsed. Must include ``status``, ``edges``,
        ``pmu``, and ``pdg`` data.
    desc : MaskGroup, optional
        If the masks for all partons in the hard process have already
        been computed, these may be passed here to save performing the
        computation again.

    Returns
    -------
    hierarchy : MaskGroup
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
    hard_desc: gcl.MaskGroup = desc[hard_mask][list(names)]  # type: ignore
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
    for root_name, root in masks.items():
        for name, mask in _leaf_mask_iter(root, root_name, False):
            if name != "latent":
                continue
            mask[mask & graph.hard_mask] = False
    return masks


def _make_tree(
    flat: ty.Union[ty.Dict[str, ty.Any], ty.Tuple[str, ...]],
    roots: ty.Set[str],
    hard: ty.Dict[str, ty.Tuple[str, ...]],
    desc: gcl.MaskGroup,
    adj: gcl.AdjacencyList,
) -> gcl.MaskGroup:
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
        for key, nest in flat.items():
            if key not in roots:
                continue
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
                in_edge = adj.edges["in"][mask][0]
                initial_parton_mask = adj.edges["out"] == in_edge
                if np.sum(initial_parton_mask) == 1:
                    mask.data[initial_parton_mask] = True
                branch[parton] = mask
    return branch


def _leaf_mask_iter(
    branch: ty.Union[gcl.MaskGroup, gcl.MaskArray],
    branch_name: str,
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
            # TODO: look into contravariant type for this
            yield from _leaf_mask_iter(mask, name)  # type: ignore


def leaf_masks(mask_tree: gcl.MaskGroup) -> gcl.MaskGroup:
    """Find the leaves of the hard process, when organised into a
    hierarchical tree from ``hierarchy()``.

    :group: select

    .. versionadded:: 0.1.11

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


def any_overlap(masks: gcl.MaskGroup) -> bool:
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
    any_overlap : bool
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
        Four-momenta for a set of particles.
    radius : float
        Euclidean distance in the azimuth-pseudorapidity plane from the
        centroid, beyond which particles will be filtered out.
    mask : MaskArray, optional
        If provided, will apply the mask to the passed ``pmu``, and
        output ``MaskArray`` will have the same length.
    centre : tuple[float, float]
        Pseudorapidity and azimuth coordinates for a user-defined
        centroid.

    Returns
    -------
    prune_mask : MaskArray
        Mask which retains only the particles within ``radius`` of the
        centroid.
    """
    if mask is not None:
        pmu = pmu[mask]
        event_mask = np.zeros_like(mask, "<?")
    if centre is None:
        eta_mid = (pmu.eta * pmu.pt).sum() / pmu.pt.sum()
        phi_sum_ = (pmu._xy_pol * pmu.pt).sum()
        phi_mid_ = phi_sum_ / np.abs(phi_sum_)
    else:
        eta_mid, phi_mid = centre
        phi_mid_ = np.exp(complex(0, phi_mid))
    dist = np.hypot(pmu.eta - eta_mid, np.angle(pmu._xy_pol * phi_mid_.conj()))
    is_within = dist < radius
    if mask is None:
        return gcl.MaskArray(is_within)
    event_mask[mask] = is_within
    return gcl.MaskArray(event_mask)
