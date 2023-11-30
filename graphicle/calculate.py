"""
``graphicle.calculate``
=======================

Algorithms for performing common HEP calculations using graphicle data
structures.
"""
import cmath
import contextlib as ctx
import functools as fn
import itertools as it
import math
import operator as op
import typing as ty
import warnings
from functools import lru_cache, partial

import networkx as nx
import numba as nb
import numpy as np
import numpy.lib.recfunctions as rfn
import scipy.optimize as spo
from deprecation import deprecated

from . import base

if ty.TYPE_CHECKING is True:
    from graphicle.data import *

__all__ = [
    "azimuth_centre",
    "pseudorapidity_centre",
    "rapidity_centre",
    "weighted_centroid",
    "resultant_coords",
    "combined_mass",
    "flow_trace",
    "aggregate_momenta",
    "cluster_coeff_distbn",
    "thrust",
    "spherocity",
    "c_parameter",
    "jaccard_distance",
]


PMU_DTYPE = nb.typeof(
    np.empty(1, dtype=np.dtype(list(zip("xyze", ("<f8",) * 4))))
)


@deprecated(
    deprecated_in="0.3.1",
    removed_in="0.4.0",
    details="Use ``weighted_centroid()`` or ``resultant_coords()`` instead.",
)
def azimuth_centre(pmu: "MomentumArray", pt_weight: bool = True) -> float:
    """Calculates the central point in azimuth for a set of particles.

    :group: calculate

    .. versionadded:: 0.1.7

    .. versionchanged:: 0.3.1
       The ``pt_weight`` parameter implementation has been corrected.
       Previous versions resulted in :math:`p_T`-weighting when
       ``False``, and :math:`p_T^2`-weighting when ``True``.

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
    if not pt_weight:
        pol = pol / pmu.pt
    return np.angle(pol.sum()).item()


@deprecated(
    deprecated_in="0.3.1",
    removed_in="0.4.0",
    details="Use ``weighted_centroid()`` or ``resultant_coords()`` instead.",
)
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


@deprecated(
    deprecated_in="0.3.1",
    removed_in="0.4.0",
    details="Use ``weighted_centroid()`` or ``resultant_coords()`` instead.",
)
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


def weighted_centroid(
    pmu: "MomentumArray", pseudo: bool = True
) -> ty.Tuple[float, float]:
    """Calculates the :math:`p_T`-weighted centroid for a set of
    particles in the (pseudo)rapidity-azimuth plane.

    :group: calculate

    .. versionadded:: 0.3.1

    Parameters
    ----------
    pmu : MomentumArray
        Momenta of the particles in the set.
    pseudo : bool
        If ``True``, will use pseudorapidity. If ``False``, will use
        true rapidity. Default is ``True``.

    Returns
    -------
    rapidity_centre, azimuth_centre : float
        The :math:`p_T`-weighted centroid of the particles.

    See Also
    --------
    resultant_coords : Coordinates of the momentum sum.

    Notes
    -----
    The :math:`p_T`-weighted centroid pseudorapidity is given by:

    .. math::
       \\bar \\eta = \\dfrac{\\sum_j p_{T j} \\eta_j} {\\sum_k p_{T k}}

    For azimuth, :math:`\\eta` is swapped for :math:`\\phi`, and the
    result is shifted to remain in the :math:`[-\\pi, +\\pi]` range.
    """
    pt_sum = pmu.pt.sum()
    rap = pmu.eta if pseudo else pmu.rapidity
    rap_mid = ((rap * pmu.pt).sum() / pt_sum).item()
    phi_mid_unbound = ((pmu.phi * pmu.pt).sum() / pt_sum).item()
    phi_mid = cmath.phase(cmath.rect(1, phi_mid_unbound))
    return rap_mid, phi_mid


def resultant_coords(
    pmu: "MomentumArray", pseudo: bool = True
) -> ty.Tuple[float, float]:
    """Returns the resulting (pseudo)rapidity and azimuthal coordinates
    when summing the momenta of a particle set.

    :group: calculate

    .. versionadded:: 0.3.1

    Parameters
    ----------
    pmu : MomentumArray
        Momenta of the particles in the set.
    pseudo : bool
        If ``True``, will use pseudorapidity. If ``False``, will use
        true rapidity. Default is ``True``.

    Returns
    -------
    rapidity_centre, azimuth_centre : float
        The central location on the (pseudo)rapidity-azimuth plane for
        the given particle set.

    See Also
    --------
    weighted_centroid : :math:`p_T`-weighted average coordinates.

    Notes
    -----
    While this does provide central coordinates, which are effectively
    weighted by :math:`p_T`, this is **not** mathematically equivalent
    to the :math:`p_T`-weighted centroid, see ``weighted_centroid()``.
    This function takes a sum of the underlying momenta, and gives the
    the coordinates of the reconstructed object. This is often more
    desirable, as the calculation is implicitly consistent with the
    detector geometry, whereas ``weighted_centroid()`` treats the
    (pseudo)rapidity-azimuth plane as Cartesian.
    """
    pmu_sum: "MomentumArray" = np.sum(pmu, axis=0)
    rap = (pmu_sum.eta if pseudo else pmu_sum.rapidity).item()
    phi = pmu_sum.phi.item()
    return rap, phi


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
    names, vtxs = tuple(hard_graph.pdg.name), tuple(hard_graph.edges["dst"])
    # out vertices of user specified particles
    focus_pcls = graph.edges[mask]["dst"]
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


@nb.njit("float64[:](float64[:], float64[:], float64)", cache=True)
def _rapidity(
    energy: base.DoubleVector, z: base.DoubleVector, zero_tol: float
) -> base.DoubleVector:
    """Calculate the rapidity of a set of particles using energy
    and longitudinal components of their four-momenta.

    Parameters
    ----------
    energy, z : ndarray[float64]
        Components of the particles' four-momenta.
    zero_tol : float
        Absolute tolerance for energy values to be considered close to
        zero.

    Returns
    -------
    ndarray[float64]
        Rapidity of the particles.
    """
    rap = np.empty_like(energy)
    for idx, (e_val, z_val) in enumerate(zip(energy, z)):
        z_val_abs = abs(z_val)
        diff = e_val - z_val_abs
        if abs(diff) > zero_tol:
            rap_val = 0.5 * math.log((e_val + z_val_abs) / diff)
        else:
            rap_val = math.inf
        rap[idx] = math.copysign(rap_val, z_val)
    return rap


@nb.vectorize("float64(float64, float64)", cache=True)
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
    x1_, x2_ = abs(x1), abs(x2)
    diff = x1_ - x2_
    sqrt_diff = math.copysign(math.sqrt(abs(diff)), diff)
    sqrt_sum = math.sqrt(x1_ + x2_)
    return sqrt_diff * sqrt_sum  # type: ignore


@nb.njit(
    "float64[:, :](float64[:], float64[:], complex128[:], complex128[:])",
    parallel=True,
    cache=True,
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


@nb.njit("float64[:, :](float64[:], complex128[:])", parallel=True, cache=True)
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


@nb.njit("float32[:](bool_[:, :])", parallel=True, cache=True)
def _clust_coeffs(adj: base.BoolVector) -> base.FloatVector:
    num_nodes = adj.shape[0]
    coefs = np.empty(num_nodes, dtype=np.float32)
    for node_idx in nb.prange(num_nodes):
        neibs = np.flatnonzero(adj[node_idx, :])
        num_neibs = neibs.shape[0]
        if num_neibs < 2:
            coefs[node_idx] = 0.0
            continue
        num_interlinks_x2 = np.sum(adj[neibs, :][:, neibs], dtype=np.float32)
        coefs[node_idx] = num_interlinks_x2 / (num_neibs * (num_neibs - 1))
    return coefs


def cluster_coeff_distbn(
    pmu: "MomentumArray",
    radius: float,
    pseudo: bool = True,
    threads: int = 1,
) -> base.FloatVector:
    """A measure of clustering for a particle point-cloud. Transforms
    point-cloud into a graph, where node neighbourhood is determined by
    which particles fall within a distance on the :math:`\\eta-\\phi`
    plane below ``radius``. The clustering coefficients of these nodes
    are then computed.

    :group: calculate

    .. versionadded:: 0.3.4

    Parameters
    ----------
    pmu : MomentumArray
        Four-momenta of particles in the point-cloud.
    radius : float
        Pairwise distance below which nodes are considered adjacent.
    pseudo : bool
        If True, will use pseudorapidity, rather than true rapidity.
        Default is True.
    threads : int
        Number of threads to use in the parallel portions of the
        calculation.

    Returns
    -------
    ndarray[float32]
        Clustering coefficients of the particles in the point-cloud.
    """
    adj = pmu.delta_R(pmu, pseudo, threads) < radius
    np.fill_diagonal(adj, False)
    with _thread_scope(threads):
        return _clust_coeffs(adj)


@ctx.contextmanager
def _thread_scope(num_threads: int):
    prev_threads = nb.get_num_threads()
    nb.set_num_threads(num_threads)
    try:
        yield None
    finally:
        nb.set_num_threads(prev_threads)


def aggregate_momenta(
    pmu: "MomentumArray", cluster_masks: ty.Iterable["MaskArray"]
) -> "MomentumArray":
    """Calculates the aggregate momenta for a sequence of clusters over
    the same point cloud of particles.

    :group: calculate

    .. versionadded:: 0.2.14

    Parameters
    ----------
    pmu : MomentumArray
        Four-momenta of particles in the point cloud.
    cluster_masks : Iterable[MaskArray]
        Iterable of boolean masks identifying which particles belong to
        each of the clusterings.

    Returns
    -------
    MomentumArray
        Where each element represents the sum aggregate four-momenta of
        each cluster, in the same order passed via ``clusters``.
    """
    momentum_class = type(pmu)
    pmus = map(fn.partial(op.getitem, pmu), cluster_masks)
    pmu_sums = map(fn.partial(np.sum, axis=0), pmus)
    return momentum_class(list(it.chain.from_iterable(pmu_sums)))


@nb.njit("float64(float64, float64, float64)", cache=True)
def _three_norm(x: float, y: float, z: float) -> float:
    max_component = max(abs(x), abs(y), abs(z))
    max_recip = 1.0 / max_component
    x *= max_recip
    y *= max_recip
    z *= max_recip
    return max_component * np.sqrt(x * x + y * y + z * z)


@nb.njit("float64[:](float64[:])", cache=True)
def _angles_to_axis(axis_angles: base.DoubleVector) -> base.DoubleVector:
    """Given the azimuth and inclination angles, return the Cartesian
    coordinates a unit vector.

    Parameters
    ----------
    axis_angles : ndarray[float64]
        Array of shape (2,), containing azimuth and inclination angles
        of thrust axis.

    Returns
    -------
    axis : ndarray[float64]
        Array of shape (3,) containing the x, y, and z components of the
        unit vector, respectively.
    """
    azimuth, inclination = axis_angles[0], axis_angles[1]
    sin_theta, cos_theta = math.sin(inclination), math.cos(inclination)
    sin_phi, cos_phi = math.sin(azimuth), math.cos(azimuth)
    return np.array(
        [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta], dtype=np.float64
    )


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[:]))(nb.float64[:], PMU_DTYPE),
    cache=True,
)
def _thrust_with_grad(
    axis_angles: base.DoubleVector, momenta: base.VoidVector
) -> ty.Tuple[float, base.DoubleVector]:
    """Computes the thrust for a thrust axis given by a pair of azimuth
    and inclination angles, as well as the gradient of the thrust with
    respect to these angles.

    Parameters
    ----------
    axis_angles : ndarray[float64]
        Array of shape (2,), containing azimuth and inclination angles
        of thrust axis.
    momenta : ndarray[void]
        Structured array, containing fields labeled 'x', 'y', 'z',
        of 64-bit floating point numbers for the components of a
        momentum point cloud.

    Returns
    -------
    thrust_val : float
        The (negated) value of thrust, with the given thrust axis.
    thrust_grad : ndarray[float64]
        Array of shape (2,) containing the (negated) gradient of the
        thrust with respect to azimuth and inclination angles of the
        thrust axis, respectively.

    Notes
    -----
    This routine is defined to provide an objective for optimizers.
    Specifically, the thrust must be maximized by finding a suitable
    thrust axis, so the thrust and gradient are negated, so that they
    can be passed to a minimizer.
    """
    azimuth, inclination = axis_angles[0], axis_angles[1]
    sin_theta, cos_theta = math.sin(inclination), math.cos(inclination)
    sin_phi, cos_phi = math.sin(azimuth), math.cos(azimuth)
    ax_x, ax_y, ax_z = sin_theta * cos_phi, sin_theta * sin_phi, cos_theta
    abs_dot_sum = norm_sum = grad_phi_sum = grad_theta_sum = 0.0
    for momentum in momenta:
        x, y, z = momentum["x"], momentum["y"], momentum["z"]
        grad_phi = (y * sin_theta * cos_phi) - (x * sin_theta * sin_phi)
        grad_theta = (
            (x * cos_theta * cos_phi)
            + (y * cos_theta * sin_phi)
            - (z * sin_theta)
        )
        dot_prod = (ax_x * x) + (ax_y * y) + (ax_z * z)
        dot_prod_sign = math.copysign(1.0, dot_prod)
        abs_dot_sum += abs(dot_prod)
        grad_phi_sum += dot_prod_sign * grad_phi
        grad_theta_sum += dot_prod_sign * grad_theta
        norm_sum += _three_norm(x, y, z)
    func_val = abs_dot_sum / norm_sum
    grad_vec = (
        np.array([grad_phi_sum, grad_theta_sum], dtype=np.float64) / norm_sum
    )
    return -func_val, -grad_vec


@ty.overload
def thrust(
    pmu: "MomentumArray",
    return_axis: ty.Literal[False],
    rng_seed: ty.Optional[int],
) -> float:
    ...


@ty.overload
def thrust(pmu: "MomentumArray", return_axis: ty.Literal[False]) -> float:
    ...


@ty.overload
def thrust(pmu: "MomentumArray", rng_seed: ty.Optional[int]) -> float:
    ...


@ty.overload
def thrust(pmu: "MomentumArray") -> float:
    ...


@ty.overload
def thrust(
    pmu: "MomentumArray",
    return_axis: ty.Literal[True],
    rng_seed: ty.Optional[int],
) -> ty.Tuple[float, base.DoubleVector]:
    ...


@ty.overload
def thrust(
    pmu: "MomentumArray",
    return_axis: ty.Literal[True],
) -> ty.Tuple[float, base.DoubleVector]:
    ...


def thrust(
    pmu: "MomentumArray",
    return_axis: bool = False,
    rng_seed: ty.Optional[int] = None,
):
    """Computes the thrust of an event, from the final-state momenta.

    :group: calculate

    .. versionadded:: 0.3.8

    Parameters
    ----------
    pmu : MomentumArray
        Momentum of hadronised particles in the final state of the event
        record.
    return_axis : bool
        If ``True``, will return a tuple with the thrust, and the axis
        unit vector which was found to maximise thrust.
    rng_seed : int, optional
        Initial guess for the axis unit vector is sampled from a uniform
        random distribution, over the surface of a sphere. If passed,
        will initialise the random number generator with the provided
        seed, enabling reproducible results.

    Returns
    -------
    thrust : float
        The thrust of the event.
    axis : ndarray[float64], optional
        The x, y, and z components of the thrust axis, respectively.

    See Also
    --------
    spherocity : Computes the spherocity from final state momenta.
    c_parameter : Computes the C-parameter from the final state momenta.
    """
    half_pi = 0.5 * math.pi
    domain = (-half_pi, half_pi)
    rng = np.random.default_rng(seed=rng_seed)
    guess = rng.uniform(*domain, size=2)
    optim = spo.minimize(
        fun=_thrust_with_grad,
        x0=guess,
        bounds=(domain, domain),
        method="L-BFGS-B",
        jac=True,
        args=(pmu.data,),
    )
    thrust_val = -optim.fun
    if return_axis:
        return thrust_val, _angles_to_axis(optim.x)
    return thrust_val


@nb.njit(
    nb.types.Tuple((nb.float64, nb.float64[:]))(nb.float64[:], PMU_DTYPE),
    cache=True,
)
def _spherocity_with_grad(
    axis_angles: base.DoubleVector, momenta: base.VoidVector
) -> ty.Tuple[float, base.DoubleVector]:
    """Computes the spherocity for a unit vector given by a pair of
    azimuth and inclination angles, as well as the gradient of the
    spherocity with respect to these angles.

    Parameters
    ----------
    axis_angles : ndarray[float64]
        Array of shape (2,), containing azimuth and inclination angles
        of the unit vector.
    momenta : ndarray[void]
        Structured array, containing fields labeled 'x', 'y', 'z',
        of 64-bit floating point numbers for the components of a
        momentum point cloud.

    Returns
    -------
    spherocity_val : float
        The value of spherocity, with the given unit vector.
    spherocity_grad : ndarray[float64]
        Array of shape (2,) containing the gradient of the spherocity
        with respect to azimuth and inclination angles of the unit
        vector, respectively.
    """
    phi, theta = axis_angles[0], axis_angles[1]
    # pre-compute trig ratios and spherocity axis
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)
    cos_2phi = 2.0 * cos_phi * cos_phi - 1.0
    sin_2phi = 2.0 * cos_phi * sin_phi
    cos_2theta = 2.0 * cos_theta * cos_theta - 1.0
    sin_2theta = 2.0 * cos_theta * sin_theta
    ax_x, ax_y, ax_z = sin_theta * cos_phi, sin_theta * sin_phi, cos_theta
    # initialising accumulators
    cross_norm_sum = norm_sum = grad_theta_sum = grad_phi_sum = 0.0
    for momentum in momenta:
        x, y, z = momentum["x"], momentum["y"], momentum["z"]
        norm_sum += _three_norm(x, y, z)
        # magnitude of the cross product
        cross_x = y * ax_z - z * ax_y
        cross_y = z * ax_x - x * ax_z
        cross_z = x * ax_y - y * ax_x
        cross_norm = _three_norm(cross_x, cross_y, cross_z)
        cross_norm_sum += cross_norm
        # accumulate gradient contributions
        x_sq, y_sq = x * x, y * y
        cross_norm_recip = 1.0 / cross_norm
        grad_phi_sum += cross_norm_recip * (
            0.5
            * (cos_2theta - 1.0)
            * ((2.0 * x * y * cos_2phi) - ((x_sq - y_sq) * sin_2phi))
            - sin_2theta * ((y * z * cos_phi) - (x * z * sin_phi))
        )
        grad_theta_sum += cross_norm_recip * (
            -2.0 * cos_2theta * ((x * y * cos_phi) + (y * z * sin_phi))
            - 0.5
            * sin_2theta
            * (
                (x_sq + y_sq)
                + ((x_sq - y_sq) * cos_2phi)
                - (2.0 * z * z)
                + (2.0 * x * y * sin_2phi)
            )
        )
    scale_factor = (4.0 / (math.pi * norm_sum)) ** 2
    spherocity = scale_factor * cross_norm_sum * cross_norm_sum
    grad_phi = scale_factor * cross_norm_sum * grad_phi_sum
    grad_theta = scale_factor * cross_norm_sum * grad_theta_sum
    return spherocity, np.array([grad_phi, grad_theta], dtype=np.float64)


@ty.overload
def spherocity(
    pmu: "MomentumArray",
    return_axis: ty.Literal[False],
    rng_seed: ty.Optional[int],
) -> float:
    ...


@ty.overload
def spherocity(pmu: "MomentumArray", return_axis: ty.Literal[False]) -> float:
    ...


@ty.overload
def spherocity(pmu: "MomentumArray", rng_seed: ty.Optional[int]) -> float:
    ...


@ty.overload
def spherocity(pmu: "MomentumArray") -> float:
    ...


@ty.overload
def spherocity(
    pmu: "MomentumArray",
    return_axis: ty.Literal[True],
    rng_seed: ty.Optional[int],
) -> ty.Tuple[float, base.DoubleVector]:
    ...


@ty.overload
def spherocity(
    pmu: "MomentumArray",
    return_axis: ty.Literal[True],
) -> ty.Tuple[float, base.DoubleVector]:
    ...


def spherocity(
    pmu: "MomentumArray",
    return_axis: bool = False,
    rng_seed: ty.Optional[int] = None,
):
    """Computes the spherocity of an event, from final-state momenta.

    :group: calculate

    .. versionadded:: 0.3.8

    Parameters
    ----------
    pmu : MomentumArray
        Momentum of hadronised particles in the final state of the event
        record.
    return_axis : bool
        If ``True``, will return a tuple with the spherocity, and the
        axis unit vector which was found to minimize spherocity.
    rng_seed : int, optional
        Initial guess for the axis unit vector is sampled from a uniform
        random distribution, over the surface of a sphere. If passed,
        will initialise the random number generator with the provided
        seed, enabling reproducible results.

    Returns
    -------
    spherocity : float
        The spherocity of the event.
    axis : ndarray[float64], optional
        The x, y, and z components of the spherocity axis, respectively.

    See Also
    --------
    thrust : Computes the thrust from final state momenta.
    c_parameter : Computes the C-parameter from the final state momenta.
    """
    half_pi = 0.5 * math.pi
    domain = (-half_pi, half_pi)
    rng = np.random.default_rng(seed=rng_seed)
    guess = rng.uniform(*domain, size=2)
    optim = spo.minimize(
        fun=_spherocity_with_grad,
        x0=guess,
        bounds=(domain, domain),
        jac=True,
        args=(pmu.data,),
    )
    sph_val = optim.fun
    if return_axis:
        return sph_val, _angles_to_axis(optim.x)
    return sph_val


@nb.njit(nb.float64(PMU_DTYPE), cache=True)
def _c_parameter(momenta: base.VoidVector) -> float:
    output = norm_sum = 0.0
    for idx_i, pmu_i in enumerate(momenta):
        x_i, y_i, z_i = pmu_i["x"], pmu_i["y"], pmu_i["z"]
        norm_i = _three_norm(x_i, y_i, z_i)
        norm_sum += norm_i
        for pmu_j in momenta[idx_i:]:
            x_j, y_j, z_j = pmu_j["x"], pmu_j["y"], pmu_j["z"]
            norm_j = _three_norm(x_j, y_j, z_j)
            norm_prod = norm_i * norm_j
            i_dot_j = (x_i * x_j) + (y_i * y_j) + (z_i * z_j)
            output += 2.0 * (norm_prod - (i_dot_j * i_dot_j / norm_prod))
    return 1.5 * output / (norm_sum * norm_sum)


def c_parameter(pmu: "MomentumArray") -> float:
    """Computes the C-parameter for the event, from the final state
    momenta.

    :group: calculate

    .. versionadded:: 0.3.8

    Parameters
    ----------
    pmu : MomentumArray
        Momentum of hadronised particles in the final state of the event
        record.

    Returns
    -------
    float
        The C-parameter of the event.

    See Also
    --------
    thrust : Computes the thrust from final state momenta.
    spherocity : Computes the spherocity from final state momenta.
    """
    return _c_parameter(pmu.data)


@nb.njit(
    [
        "float64(bool_[:], bool_[:], Optional(float64[:]))",
        "float64(bool_[:], bool_[:], Omitted(None))",
    ],
    cache=True,
)
def _jaccard_distance(
    u: base.BoolVector,
    v: base.BoolVector,
    w: ty.Optional[base.DoubleVector] = None,
) -> float:
    """Computes the Jaccard distance for two sets.

    Parameters
    ----------
    u, v : ndarray[bool_]
        Boolean masks, identifying which elements belong
        respective sets.
    w : ndarray[float64]
        Weights associated with each element. If not passed,
        will assume weights are 1.

    Returns
    -------
    float
        Jaccard distance between sets.
    """
    difference_sum = 0.0
    union_sum = 0.0
    if w is None:
        w = np.ones(u.shape, dtype=np.float64)
    for u_elem, v_elem, weight in zip(u, v, w):
        nonzero = u_elem | v_elem
        unequal_nonzero = nonzero & (u_elem != v_elem)
        union_sum += weight * nonzero
        difference_sum += weight * unequal_nonzero
    if union_sum == 0.0:
        return 0.0
    return difference_sum / union_sum


def jaccard_distance(
    mask_1: ty.Union[base.MaskBase, base.BoolVector],
    mask_2: ty.Union[base.MaskBase, base.BoolVector],
    weights: ty.Optional[base.DoubleVector] = None,
) -> float:
    """Computes the Jaccard distance between two sets.

    :group: calculate

    .. versionadded:: 0.3.8

    Parameters
    ----------
    mask_1, mask_2 : ndarray[bool_] or MaskBase
        Boolean masks, identifying which elements belong to the
        respective sets.
    weights : ndarray[float64]
        Weights associated with each element. If not passed,
        will assume weights are 1.

    Returns
    -------
    float
        Jaccard distance between sets.
    """
    if not isinstance(mask_1, np.ndarray):
        mask_1 = mask_1.data
    if not isinstance(mask_2, np.ndarray):
        mask_2 = mask_2.data
    return _jaccard_distance(mask_1, mask_2, weights)


@nb.njit(
    "float64[:, :](float64[:], float64[:], complex128[:], complex128[:])",
    parallel=True,
    cache=True,
)
def _assignment_cost(
    rapidity_1: base.DoubleVector,
    rapidity_2: base.DoubleVector,
    xy_pol_1: base.ComplexVector,
    xy_pol_2: base.ComplexVector,
) -> base.DoubleVector:
    dist_matrix = _delta_R(rapidity_1, rapidity_2, xy_pol_1, xy_pol_2)
    pt_2 = rapidity_2  # recycle memory buffer for transverse momenta
    for pol_idx, pol_val in enumerate(xy_pol_2):
        pt_2[pol_idx] = abs(pol_val)
    var_pt_recip = 1.0 / np.var(pt_2)
    num_partons = dist_matrix.shape[0]
    for parton_idx in nb.prange(num_partons):
        row = dist_matrix[parton_idx, :]
        pt_1_val = abs(xy_pol_1[parton_idx])
        var_dR_recip = 1.0 / np.var(row)
        for jet_idx, (dR_val, pt_2_val) in enumerate(zip(row, pt_2)):
            dpt = pt_1_val - pt_2_val
            dR_cost = math.expm1(-0.5 * var_dR_recip * dR_val * dR_val)
            pt_cost = math.expm1(-0.5 * var_pt_recip * dpt * dpt)
            row[jet_idx] = -(dR_cost + pt_cost)
    return dist_matrix
