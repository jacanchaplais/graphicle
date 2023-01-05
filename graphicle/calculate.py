"""
``graphicle.calculate``
=======================

Algorithms for performing common HEP calculations using graphicle data
structures.
"""
from typing import (
    Tuple,
    Optional,
    Set,
    List,
    Dict,
    Callable,
    Union,
    Iterable,
)
from functools import lru_cache, partial
import warnings

import numpy as np
import numpy.lib.recfunctions as rfn
from typicle import Types
import networkx as nx
import pyjet
from pyjet import ClusterSequence, PseudoJet

import graphicle as gcl
from . import base


__all__ = [
    "azimuth_centre",
    "combined_mass",
    "flow_trace",
    "cluster_pmu",
]

_types = Types()


def azimuth_centre(pmu: gcl.MomentumArray, pt_weight: bool = True) -> float:
    """Calculates the central point in azimuth for a set of particles.

    :group: calculate

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
    azimuth : float
        The centre of the particle set in the azimuth dimension.
    """
    pol = pmu._xy_pol
    if pt_weight is True:
        pol = pol * pmu.pt
    return float(np.angle(pol.sum()))


def combined_mass(
    pmu: Union[gcl.MomentumArray, np.ndarray],
    weight: Optional[base.DoubleVector] = None,
) -> float:
    """Returns the combined mass of the particles represented in the
    provided MomentumArray.

    This is done by summing the four momenta, optionally weighting the
    components, and then taking the inner product of the result with
    itself in Minkowski space.

    :group: calculate

    Parameters
    ----------
    pmu : MomentumArray, ndarray
        Momenta of particles comprising a jet, or an analagous combined
        object. If a numpy array is passed, it must be structured with
        fields (x, y, z, e).
    weight : array, optional
        Weights for each particle when reconstructing the jet momentum.
        May be either structured or unstructured. If unstructured,
        ensure the columns are in the order (x, y, z, e).

    Notes
    -----
    This does not mask the MomentumArray for you. All filters and cuts
    must be applied before passing to this function.

    In the event of a squared mass below zero (due to numerical
    fluctuations for very low mass reconstructions), this function will
    simply return 0.0.
    """
    # sanitizing and combining the inputs
    if isinstance(pmu, gcl.MomentumArray):
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


def _diffuse(colors: List[np.ndarray], feats: List[np.ndarray]):
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
    basis: Tuple[int, ...],
    feat_dim: int,
    is_structured: bool,
    exclusive: bool = False,
) -> np.ndarray:
    len_basis = len(basis)
    feat_fmt = rfn.structured_to_unstructured if is_structured else lambda x: x
    color = np.zeros((len_basis, feat_dim), dtype=_types.double)
    if vertex in basis:
        color[basis.index(vertex)] = 1.0
        if exclusive is True:
            return color
    in_edges = nx_graph.in_edges(vertex, data=True)
    colors_in: List[np.ndarray] = []
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
    graph: gcl.Graphicle,
    mask: Union[base.MaskBase, np.ndarray],
    prop: Union[base.ArrayBase, np.ndarray],
    exclusive: bool = False,
    target: Optional[Set[int]] = None,
) -> Dict[str, np.ndarray]:
    """Performs flow tracing from specified particles in an event, back
    to the hard partons.

    :group: calculate

    Parameters
    ----------
    graph : Graphicle
        Full particle event, containing hard partons, showering and
        hadronisation.
    mask : MaskArray, MaskGroup, ndarray
        Boolean mask identifying which particles should have their
        ancestry traced.
    prop : ArrayBase, ndarray
        Property to trace back, eg. 4-momentum, charge.
        Must be the same shape as arrays stored in graph.
        Can be structured, unstructured, or a graphicle array, though
        unstructured arrays must be 1d.
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
    trace_array : Dict of arrays
        Dictionary of arrays. Keys are parton names, arrays represent
        the contributions of hard partons traced down to the properties
        of the selected subset of particles specified by mask.
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
    hard_graph = graph[hard_mask.bitwise_or]
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
    array_fmt: Callable[[np.ndarray], np.ndarray] = (
        partial(rfn.unstructured_to_structured, dtype=dtype)  # type: ignore
        if is_structured
        else lambda x: x.squeeze()
    )

    for i, name in enumerate(names):
        traces[name] = array_fmt(trc[:, i, :])
    return traces


def cluster_pmu(
    pmu: gcl.MomentumArray,
    radius: float,
    p_val: float,
    pt_cut: Optional[float] = None,
    eta_cut: Optional[float] = None,
) -> gcl.MaskGroup:
    """Clusters particles using the generalised-kt algorithm.

    :group: calculate

    Parameters
    ----------
    pmu: gcl.MomentumArray
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
    clusters : gcl.MaskGroup
        MaskGroup object, containing boolean masks over the input data
        for each jet clustering.

    Notes
    -----
    This is a wrapper around FastJet's implementation.

    Standard settings:
        kt algorithm: p_val = 1
        Cambridge/Aachen algorithm: p_val = 0
        anti-kt algorithm: p_val = -1
    """
    pmu_pyjet = pmu.data[["e", "x", "y", "z"]]
    pmu_pyjet.dtype.names = "E", "px", "py", "pz"
    pmu_pyjet_idx = rfn.append_fields(
        pmu_pyjet, "idx", np.arange(len(pmu_pyjet))
    )
    sequence: ClusterSequence = pyjet.cluster(
        pmu_pyjet_idx, R=radius, p=p_val, ep=True
    )
    jets: Iterable[PseudoJet] = sequence.inclusive_jets()
    if pt_cut is not None:
        jets = filter(lambda jet: jet.pt > pt_cut, jets)
    if eta_cut is not None:
        jets = filter(lambda jet: abs(jet.eta) < eta_cut, jets)
    cluster_mask = gcl.MaskGroup()
    cluster_mask.agg_op = gcl.data.MaskAggOp.OR
    for i, jet in enumerate(jets):
        mask = np.zeros_like(pmu_pyjet, dtype="<?")
        mask[list(map(lambda pcl: pcl.idx, jet))] = True  # type: ignore
        cluster_mask[f"{i}"] = mask
    return cluster_mask
