"""
``graphicle.calculate``
=======================

Algorithms for performing common HEP calculations using graphicle data
structures.
"""
import contextlib as ctx
import math
import typing as ty
import warnings
from functools import lru_cache, partial

import deprecation
import networkx as nx
import numba as nb
import numpy as np
import numpy.lib.recfunctions as rfn
import pyjet
from pyjet import ClusterSequence, PseudoJet

from . import base

if ty.TYPE_CHECKING is True:
    from graphicle.data import *

__all__ = [
    "azimuth_centre",
    "pseudorapidity_centre",
    "rapidity_centre",
    "combined_mass",
    "flow_trace",
    "cluster_pmu",
]


def azimuth_centre(pmu: "MomentumArray", pt_weight: bool = True) -> float:
    """Calculates the central point in azimuth for a set of particles.

    :group: calculate

    .. versionadded:: 0.1.7

    Parameters
    ----------
    pmu : MomentumArray
        Four-momenta of the particles.
    pt_weight : bool
        If ``True``, will weight the contributions of each particle by
        transverse momentum. Similar to finding a centre-of-mass, but
        for transverse momentum. Default is ``True``.

    Returns
    -------
    float
        The centre of the particle set in the azimuth dimension.
    """
    pol = pmu._xy_pol
    if pt_weight is True:
        pol = pol * pmu.pt
    return np.angle(pol.sum()).item()


def pseudorapidity_centre(pmu: "MomentumArray") -> float:
    """Calculates the central point in pseudorapidity for a set of
    particles.

    :group: calculate

    .. versionadded:: 0.1.7

    Parameters
    ----------
    pmu : MomentumArray
        Four-momenta of the particles.

    Returns
    -------
    float
        The :math:`p_T` weighted centre of the particle set in the
        pseudorapidity dimension.
    """
    return ((pmu.eta * pmu.pt).sum() / pmu.pt.sum()).item()


def rapidity_centre(pmu: "MomentumArray") -> float:
    """Calculates the central point in rapidity for a set of particles.

    :group: calculate

    .. versionadded:: 0.2.11

    Parameters
    ----------
    pmu : MomentumArray
        Four-momenta of the particles.

    Returns
    -------
    float
        The :math:`p_T` weighted centre of the particle set in the
        rapidity dimension.
    """
    return (pmu.rapidity * pmu.pt).sum() / pmu.pt.sum()


def combined_mass(
    pmu: ty.Union["MomentumArray", base.VoidVector],
    weight: ty.Optional[base.DoubleVector] = None,
) -> float:
    """Returns the combined mass of the particles represented in the
    provided ``MomentumArray``.

    This is done by summing the four momenta, optionally weighting the
    components, and then taking the inner product of the result with
    itself in Minkowski space.

    :group: calculate

    .. versionadded:: 0.1.0

    Parameters
    ----------
    pmu : MomentumArray or ndarray[void]
        Momenta of particles comprising a jet, or an analagous combined
        object. If a numpy array is passed, it must be structured with
        fields with names ``('x', 'y', 'z', 'e')``.
    weight : ndarray[float64], optional
        Weights for each particle when reconstructing the jet momentum.
        May be either structured or unstructured. If unstructured,
        ensure the columns are in the order ``('x', 'y', 'z', 'e')``.

    Notes
    -----
    This does not mask the ``MomentumArray`` for you. All filters and
    cuts must be applied before passing to this function.

    In the event of a squared mass below zero (due to numerical
    fluctuations for very low mass reconstructions), this function will
    simply return ``0.0``.
    """
    # sanitizing and combining the inputs
    if isinstance(pmu, base.ArrayBase):
        pmu = pmu.data
    data = rfn.structured_to_unstructured(pmu)
    if weight is not None:
        if not isinstance(weight, np.ndarray):
            raise ValueError("Weights must be provided as a numpy array.")
        if weight.dtype.names is not None:
            weight = rfn.structured_to_unstructured(weight)
        data = weight * data
    # mass given as minkowski norm of (weighted) momenta
    minkowski = np.array([-1.0, -1.0, -1.0, 1.0])
    mass: float
    with warnings.catch_warnings():  # catch when np.sqrt warns < 0 input
        warnings.filterwarnings("error")
        try:
            mass = np.sqrt((data.sum(axis=0) ** 2) @ minkowski)
        except RuntimeWarning:
            mass = 0.0  # if sqrt(pmu^2) < 0, return mass as 0.0
    return mass


def _diffuse(colors: ty.List[base.AnyVector], feats: ty.List[base.AnyVector]):
    color_shape = colors[0].shape
    av_color = np.zeros((color_shape[0], color_shape[1]), dtype="<f8")
    color_stack = np.dstack(colors)  # len_basis x feat_dim x num_in
    feat_stack = np.vstack(feats).T  # feat_dim x num_in
    feat_sum = np.sum(feat_stack, axis=1)
    nonzero_mask = feat_sum != 0.0
    av_color[:, nonzero_mask] = (
        np.sum(
            color_stack[:, nonzero_mask, :] * feat_stack[nonzero_mask], axis=2
        )
        / feat_sum[nonzero_mask]
    )
    return av_color


@lru_cache(maxsize=None)
def _trace_vector(
    nx_graph: nx.DiGraph,
    vertex: int,
    basis: ty.Tuple[int, ...],
    feat_dim: int,
    is_structured: bool,
    exclusive: bool = False,
) -> base.AnyVector:
    len_basis = len(basis)
    feat_fmt = rfn.structured_to_unstructured if is_structured else lambda x: x
    color = np.zeros((len_basis, feat_dim), dtype="<f8")
    if vertex in basis:
        color[basis.index(vertex)] = 1.0
        if exclusive is True:
            return color
    in_edges = nx_graph.in_edges(vertex, data=True)
    colors_in: list[base.AnyVector] = []
    feats = []
    for edge in in_edges:
        feats.append(feat_fmt(edge[2]["feat"]))
        in_vtx = edge[0]
        colors_in.append(
            _trace_vector(
                nx_graph, in_vtx, basis, feat_dim, is_structured, exclusive
            )
        )
    if colors_in:
        color += _diffuse(colors_in, feats)
    return color


def flow_trace(
    graph: "Graphicle",
    mask: ty.Union[base.MaskBase, base.BoolVector],
    prop: ty.Union[base.ArrayBase, base.AnyVector],
    exclusive: bool = False,
    target: ty.Optional[ty.Set[int]] = None,
) -> ty.Dict[str, base.DoubleVector]:
    """Performs flow tracing from specified particles in an event, back
    to the hard partons.

    :group: calculate

    .. versionadded:: 0.1.0

    Parameters
    ----------
    graph : Graphicle
        Full particle event, containing hard partons, showering and
        hadronisation.
    mask : MaskArray or MaskGroup or ndarray[bool_]
        Boolean mask identifying which particles should have their
        ancestry traced.
    prop : ArrayBase or ndarray[any]
        Property to trace back, *eg.* 4-momentum, charge, *etc*. Must be
        the same shape as arrays stored in graph. Can be structured,
        unstructured, or a graphicle array, though unstructured arrays
        must be 1D.
    exclusive : bool
        If ``True``, double counting from descendant particles in the
        hard event will be switched off. *eg.* for event ``t > b W+``,
        descendants of ``b`` will show no contribution from ``t``, as
        ``b`` is a subset of ``t``. Default is ``False``.
    target : set[int], optional
        Highlights specific partons in the hard event to decompose
        properties with respect to. If unset, will use all partons in
        hard event, except for incoming partons.

    Returns
    -------
    trace_array : dict[str, ndarray]
        Keys are parton names, arrays represent the contributions of
        hard partons traced down to the properties of the selected
        subset of particles specified by mask.
    """
    if isinstance(prop, base.ArrayBase):
        prop = prop.data
    # encoding graph features onto NetworkX
    nx_graph = nx.DiGraph()
    graph_dict = graph.adj.to_dicts(edge_data={"feat": prop})
    example_feat = graph_dict["edges"][0][2]["feat"]
    try:
        feat_dim = len(example_feat)
        dtype = example_feat.dtype
    except TypeError:
        feat_dim = 1
        dtype = np.dtype(type(example_feat))
    is_structured = dtype.names is not None
    nx_graph.add_edges_from(graph_dict["edges"])
    # identify the hard ancestors to which we trace
    hard_mask = graph.hard_mask.copy()
    del hard_mask["incoming"]
    hard_graph = graph[hard_mask.bitwise_or()]
    if target:  # restrict hard partons to user specified pdgs
        target_mask = hard_graph.pdg.mask(
            list(target), blacklist=False, sign_sensitive=True
        )
        hard_graph = hard_graph[target_mask]
    names, vtxs = tuple(hard_graph.pdg.name), tuple(hard_graph.edges["out"])
    # out vertices of user specified particles
    focus_pcls = graph.edges[mask]["out"]
    trc = np.array(
        [
            _trace_vector(
                nx_graph, pcl, vtxs, feat_dim, is_structured, exclusive
            )
            for pcl in focus_pcls
        ]
    )
    _trace_vector.cache_clear()
    traces = dict()
    array_fmt: ty.Callable[[base.AnyVector], base.AnyVector] = (
        partial(rfn.unstructured_to_structured, dtype=dtype)  # type: ignore
        if is_structured
        else lambda x: x.squeeze()
    )

    for i, name in enumerate(names):
        traces[name] = array_fmt(trc[:, i, :])
    return traces


@deprecation.deprecated(
    deprecated_in="0.2.3",
    removed_in="0.3.0",
    details="Use ``graphicle.select.fastjet_clusters()`` instead.",
)
def cluster_pmu(
    pmu: "MomentumArray",
    radius: float,
    p_val: float,
    pt_cut: ty.Optional[float] = None,
    eta_cut: ty.Optional[float] = None,
) -> "MaskGroup":
    """Clusters particles using the generalised-kt algorithm.

    :group: calculate

    .. versionadded:: 0.1.4

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

    Returns
    -------
    clusters : MaskGroup
        MaskGroup object, containing boolean masks over the input data
        for each jet clustering.

    Notes
    -----
    This is a wrapper around FastJet's implementation.

    ``p_val`` set to ``-1`` gives anti-kT, ``0`` gives Cambridge-Aachen,
    and ``1`` gives kT clusterings.
    """
    from graphicle.data import MaskGroup

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
    cluster_mask = MaskGroup()
    cluster_mask.agg_op = "or"
    for i, jet in enumerate(jets):
        mask = np.zeros_like(pmu_pyjet, dtype="<?")
        mask[list(map(lambda pcl: pcl.idx, jet))] = True  # type: ignore
        cluster_mask[f"{i}"] = mask
    return cluster_mask


@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def _root_diff_two_squares(
    x1: base.DoubleUfunc, x2: base.DoubleUfunc
) -> base.DoubleUfunc:
    """Numpy ufunc to calculate the square rooted difference of two
    squares.

    Equivalent to ``sign * sqrt(abs(x1**2 - x2**2))``, element-wise.
    Where `sign` is the +1 or -1 sign associated with the value of the
    squared difference. This means that root negative squared
    differences are permitted, but produce negative, rather than
    imaginary, values.
    If `x1` or `x2` is scalar_like (*ie.* unambiguously cast-able to a
    scalar type), it is broadcast for use with each element of the other
    argument.

    Parameters
    ----------
    x1, x2 : array_like
        Double precision floating point, or sequence thereof.

    Returns
    -------
    ndarray or float
        Root difference of two squares. This is a scalar if both `x1`
        and `x2` are scalars.
    """
    diff = x1 - x2
    sqrt_diff = math.copysign(math.sqrt(abs(diff)), diff)
    sqrt_sum = math.sqrt(x1 + x2)
    return sqrt_diff * sqrt_sum  # type: ignore


@nb.njit(
    nb.float64[:, :](
        nb.float64[:], nb.float64[:], nb.complex128[:], nb.complex128[:]
    ),
    parallel=True,
)
def _delta_R(
    rapidity_1: base.DoubleVector,
    rapidity_2: base.DoubleVector,
    xy_pol_1: base.ComplexVector,
    xy_pol_2: base.ComplexVector,
) -> base.DoubleVector:
    """Numba compiled implementation of Euclidean distance function in
    the (pseudo)rapidity-azimuth plane.

    Parameters
    ----------
    rapidity_1, rapidity_2 : ndarray[float64]
        (Pseudo)rapidities of the particle point clouds being compared.
    xy_pol_1, xy_pol_2 : ndarray[complex128]
        Momentum coordinates in the xy plane of both particle point
        clouds being compared. These are given as complex numbers, where
        x is real, and y is imaginary.

    Returns
    -------
    ndarray[float64]
        Two-dimensional array containing Euclidean distance in the
        plane. Lengths of input point clouds are mapped to the number of
        rows and columns, respectively.
    """
    size_1, size_2 = len(rapidity_1), len(rapidity_2)
    result = np.empty((size_1, size_2), dtype=np.float64)
    for i in nb.prange(size_1):
        for j in range(size_2):
            drap = rapidity_1[i] - rapidity_2[j]
            if np.isnan(drap):
                drap = 0.0
            dphi = np.angle(xy_pol_1[i] * xy_pol_2[j].conjugate())
            result[i, j] = np.hypot(drap, dphi)
    return result


@nb.njit(
    nb.float64[:, :](nb.float64[:], nb.complex128[:]),
    parallel=True,
)
def _delta_R_symmetric(
    rapidity: base.DoubleVector, xy_pol: base.ComplexVector
) -> base.DoubleVector:
    """Secondary implementation of ``_delta_R()``, but for the special
    case when the inter-particle distances within a single point cloud
    are being calculated. This is more efficient than passing the same
    arrays to ``rapidity_1`` and ``rapidity_2``, *etc*.

    Parameters
    ----------
    rapidity : ndarray[float64]
        (Pseudo)rapidities of the particle point cloud.
    xy_pol : ndarray[complex128]
        Momentum coordinates in the xy plane of the particle point
        cloud. These are given as complex numbers, where x is real, and
        y is imaginary.

    Returns
    -------
    ndarray[float64]
        Two-dimensional array representing the square symmetric matrix
        of Euclidean distances between particles in the plane.
    """
    size = len(rapidity)
    result = np.empty((size, size), dtype=np.float64)
    for i in nb.prange(size):
        result[i, i] = 0.0
        for j in range(i + 1, size):
            drap = rapidity[i] - rapidity[j]
            if np.isnan(drap):
                drap = 0.0
            dphi = np.angle(xy_pol[i] * xy_pol[j].conjugate())
            result[i, j] = result[j, i] = np.hypot(drap, dphi)
    return result


@ctx.contextmanager
def _thread_scope(num_threads: int):
    prev_threads = nb.get_num_threads()
    nb.set_num_threads(num_threads)
    try:
        yield None
    finally:
        nb.set_num_threads(prev_threads)
