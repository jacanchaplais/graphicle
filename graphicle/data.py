"""
``graphicle.data``
==================

Data structures to encapsulate particle physics data, and provide
convenient methods to aid in analysis.

Classes for storing and manipulating data are listed in the table below.

|---------------+----------------------------------+-----------|
| Name          | Used for                         | Composite |
|---------------+----------------------------------+-----------|
| MaskArray     | Masking                          | No        |
| MaskGroup     | Masking                          | Yes       |
| PdgArray      | HEP data manipulation            | No        |
| MomentumArray | HEP data manipulation            | No        |
| ColorArray    | HEP data manipulation            | No        |
| HelicityArray | HEP data manipulation            | No        |
| StatusArray   | HEP data manipulation            | No        |
| ParticleSet   | HEP data manipulation            | Yes       |
| AdjacencyList | Graph connectivity               | No        |
| Graphicle     | Graph representation of HEP data | Yes       |
|---------------+----------------------------------+-----------|

All data structures are subscriptable, and may be masked and sliced,
using Python native methods, numpy arrays, or the MaskArray / MaskGroup
objects.

The composite data structures wrap one or more of the data structures
in this module to bring their behaviour together in useful ways.
These components are accessible via object attributes.

The most generically useful object, which encapsulates all others, is
the Graphicle. This brings together all particle physics data and
connectivity into a single graph representation of HEP data.

For more details, see individual docstrings.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Tuple, List, Dict, Optional, Any, TypedDict, Union

from itertools import zip_longest
from functools import partial
from copy import deepcopy

from attr import define, field, Factory, cmp_using, setters  # type: ignore
import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions as rfn
from typicle import Types
from typicle.convert import cast_array

from ._base import ParticleBase, AdjacencyBase, MaskBase, ArrayBase


###########################################
# SET UP ARRAY ATTRIBUTES FOR DATACLASSES #
###########################################
_types = Types()


def array_field(type_name):
    """Prepares a field for attrs dataclass with typicle input."""
    types = Types()
    dtype = getattr(types, type_name)
    default = Factory(lambda: np.array([], dtype=dtype))
    equality_comparison = cmp_using(np.array_equal)
    converter = partial(cast_array, cast_type=dtype)
    return field(
        default=default,
        eq=equality_comparison,
        converter=converter,
        on_setattr=setters.convert,
    )


##################################
# COMPOSITE MASK DATA STRUCTURES #
##################################
@define
class MaskArray(MaskBase, ArrayBase):
    """Data structure for containing masks over particle data."""

    data: np.ndarray = array_field("bool")

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self):
        return len(self.data)


if TYPE_CHECKING:
    _IN_MASK_DICT = Dict[str, Union[MaskArray, np.ndarray]]
    _MASK_DICT = Dict[str, MaskArray]


def _mask_dict_convert(masks: _IN_MASK_DICT) -> _MASK_DICT:
    out_masks = dict()
    for key, val in masks.items():
        if isinstance(val, MaskArray):
            mask = val
        else:
            mask = MaskArray(val)
        out_masks[key] = mask
    return out_masks


@define
class MaskGroup(MaskBase):
    """Data structure to compose groups of masks over particle arrays.
    Can be nested to form complex hierarchies.

    Parameters
    ----------
    mask_arrays : dict[str] -> MaskArray
        Dictionary of MaskArray objects to be composed.

    Attributes
    ----------
    data : np.ndarray
        Combination of all masks in group via bitwise AND reduction.
    """

    _mask_arrays: _MASK_DICT = field(
        repr=False, factory=dict, converter=_mask_dict_convert
    )

    def __repr__(self):
        keys = ", ".join(self.names)
        return f"MaskGroup(mask_arrays=[{keys}])"

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError("Key must be string.")
        return self._mask_arrays[key]

    def __setitem__(self, key, mask):
        """Add a new MaskArray to the group, with given key."""
        if not isinstance(key, str):
            raise KeyError("Key must be string.")
        if not isinstance(mask, MaskBase):
            mask = MaskArray(mask)
        self._mask_arrays.update({key: mask})

    def __delitem__(self, key):
        """Remove a MaskArray from the group, using given key."""
        self._mask_arrays.pop(key)

    def copy(self):
        return deepcopy(self)

    @property
    def names(self) -> List[str]:
        return list(self._mask_arrays.keys())

    @property
    def bitwise_or(self) -> np.ndarray:
        return np.bitwise_or.reduce(  # type: ignore
            [child.data for child in self._mask_arrays.values()]
        )

    @property
    def bitwise_and(self) -> np.ndarray:
        return np.bitwise_and.reduce(  # type: ignore
            [child.data for child in self._mask_arrays.values()]
        )

    @property
    def data(self) -> np.ndarray:
        """Same as MaskGroup.bitwise_and."""
        return self.bitwise_and

    @property
    def dict(self) -> Dict[str, np.ndarray]:
        return {key: val.data for key, val in self._mask_arrays.items()}


############################
# PDG STORAGE AND QUERYING #
############################
@define
class PdgArray(ArrayBase):
    """Returns data structure containing PDG integer codes for particle
    data.

    Attributes
    ----------
    name : ndarray
        String representation of particle names.
    charge : ndarray
        Charge for each particle.
    mass : ndarray
        Mass for each particle.
    mass_bounds : ndarray
        Mass upper and lower bounds for each particle.
    quarks : ndarray
        String representation of quark composition for each particle.
    width : ndarray
        Width for each particle.
    width_bounds : ndarray
        Width upper and lower bounds for each particle.
    isospin : ndarray
        Isospin for each particle.
    g_parity : ndarray
        G-parity for each particle.
    space_parity : ndarray
        Spatial parity for each particle.
    charge_parity : ndarray
        Charge parity for each particle.
    """

    from mcpid.lookup import PdgRecords as __PdgRecords

    data: np.ndarray = array_field("int")
    __lookup_table: __PdgRecords = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.__lookup_table = self.__PdgRecords()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def copy(self):
        return deepcopy(self)

    def mask(
        self,
        target: npt.ArrayLike,
        blacklist: bool = True,
        sign_sensitive: bool = False,
    ) -> MaskArray:
        """Provide a mask over particle evaluated against the passed
        target pdg codes. May be used to select or filter out particles
        with given pdg.

        Parameters
        ----------
        target : array_like of int
            List of pdg codes for the mask to target.
        blacklist : bool
            Flag indicates whether black- or white- listing the pdgs
            in target. If False, mask selects only target. If True, mask
            removes target. Default is True.
        sign_sensitive : bool
            Indicates whether the passed target is sign sensitive.
            If False will check for positive and negative pdgs
            simultaenously. Default is False.

        Returns
        -------
        mask : MaskArray
            Boolean mask over data, with blacklisted pdgs marked as
            False. Same shape as pdg array stored in parent object.
        """
        target = np.array(target, dtype=_types.int)
        data = self.data
        if sign_sensitive is False:
            data = np.abs(data, dtype=_types.int)
        return MaskArray(
            np.isin(data, target, assume_unique=False, invert=blacklist)
        )

    def __get_prop(self, field: str) -> np.ndarray:
        props = self.__lookup_table.properties(self.data, [field])[field]
        return props  # type: ignore

    def __get_prop_range(self, field: str) -> np.ndarray:
        field_range = [field + "lower", field + "upper"]
        props = self.__lookup_table.properties(self.data, field_range)
        return props  # type: ignore

    @property
    def name(self) -> np.ndarray:
        return self.__get_prop("name")

    @property
    def charge(self) -> np.ndarray:
        return self.__get_prop("charge")

    @property
    def mass(self) -> np.ndarray:
        return self.__get_prop("mass")

    @property
    def mass_bounds(self) -> np.ndarray:
        return self.__get_prop_range("mass")

    @property
    def quarks(self) -> np.ndarray:
        return self.__get_prop("quarks")

    @property
    def width(self) -> np.ndarray:
        return self.__get_prop("width")

    @property
    def width_bounds(self) -> np.ndarray:
        return self.__get_prop_range("width")

    @property
    def isospin(self) -> np.ndarray:
        return self.__get_prop("i")

    @property
    def g_parity(self) -> np.ndarray:
        return self.__get_prop("g")

    @property
    def space_parity(self) -> np.ndarray:
        return self.__get_prop("p")

    @property
    def charge_parity(self) -> np.ndarray:
        return self.__get_prop("c")


########################################
# MOMENTUM STORAGE AND TRANSFORMATIONS #
########################################
@define
class MomentumArray(ArrayBase):
    """Returns data structure containing four-momentum of particle
    list.

    Attributes
    ----------
    data : ndarray
        Structured array containing four momenta.
    pt : ndarray
        Transverse component of particle momenta.
    eta : ndarray
        Pseudorapidity component of particle momenta.
    phi : ndarray
        Azimuthal component of particle momenta.
    """

    data: np.ndarray = array_field("pmu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def copy(self):
        return deepcopy(self)

    @property
    def _vector(self):
        from vector import MomentumNumpy4D

        dtype = deepcopy(self.data.dtype)
        dtype.names = ("x", "y", "z", "t")
        vec = self.data.view(dtype).view(MomentumNumpy4D)
        return vec

    @property
    def pt(self) -> np.ndarray:
        return self._vector.pt  # type: ignore

    @property
    def eta(self) -> np.ndarray:
        return self._vector.eta  # type: ignore

    @property
    def phi(self) -> np.ndarray:
        return self._vector.phi  # type: ignore

    def delta_R(self, other_pmu: "MomentumArray") -> np.ndarray:
        return self._vector.deltaR(other_pmu._vector)  # type: ignore


#################
# COLOR STORAGE #
#################
@define
class ColorArray(ArrayBase):
    """Returns data structure of color / anti-color pairs for particle
    shower.

    Attributes
    ----------
    data : ndarray
        Structured array containing color / anti-color pairs.
    """

    data: np.ndarray = array_field("color")

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self):
        return len(self.data)


####################
# HELICITY STORAGE #
####################
@define
class HelicityArray(ArrayBase):
    data: np.ndarray = array_field("helicity")

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self):
        return len(self.data)


####################################
# STATUS CODE STORAGE AND QUERYING #
####################################
@define
class StatusArray(ArrayBase):
    data: np.ndarray = array_field("h_int")

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self):
        return len(self.data)

    def in_range(
        self, min_status: int, max_status: int, sign_sensitive: bool = False
    ) -> MaskArray:
        """Returns a boolean mask over particles with status codes
        sitting within passed (inclusive) range.

        Parameters
        ----------
        min_status : int
            Minimum value for status codes.
        max_status : int
            Maximum value for status codes.
        sign_sensitive : bool
            Whether or not to take signs into account during the
            comparison. (Default is False.)

        Returns
        -------
        mask_out : MaskArray
            Boolean mask over the particle dataset which selects
            data where min_status <= status <= max_status.
        """
        array = self.data
        if sign_sensitive is False:
            array = np.abs(array)
        elif sign_sensitive is not True:
            raise ValueError("sign_sensitive must be boolean valued.")
        more_than = array >= min_status  # type: ignore
        less_than = array <= max_status  # type: ignore
        return MaskArray(np.bitwise_and(more_than, less_than))

    @property
    def hard_mask(self) -> MaskGroup:
        data = np.abs(self.data)
        masks = MaskGroup(
            {
                "incoming": data == 21,
                "intermediate": data == 22,
                "outgoing": data == 23,
                "outgoing_nonperturbative_diffraction": data == 24,
            }
        )
        return masks


#########################################
# COMPOSITE OF PARTICLE DATA STRUCTURES #
#########################################
@define
class ParticleSet(ParticleBase):
    """Combines rich particle description.

    Attributes
    ----------
    data : ndarray
        Structured array containing color / anti-color pairs.
    """

    pdg: PdgArray = PdgArray()
    pmu: MomentumArray = MomentumArray()
    color: ColorArray = ColorArray()
    helicity: HelicityArray = HelicityArray()
    status: StatusArray = StatusArray()
    final: MaskArray = MaskArray()

    @property
    def __dsets(self):
        for dset_name in tuple(self.__annotations__.keys()):
            yield {"name": dset_name, "data": getattr(self, dset_name)}

    def __getitem__(self, key):
        kwargs = dict()
        for dset in self.__dsets:
            name = dset["name"]
            data = dset["data"]
            if len(data) > 0:
                kwargs.update({name: data[key]})
        return self.__class__(**kwargs)

    def __repr__(self):
        dset_repr = (repr(dset["data"]) for dset in self.__dsets)
        dset_str = ",\n".join(dset_repr)
        return f"ParticleSet(\n{dset_str}\n)"

    def __len__(self):
        filled_dsets = filter(lambda dset: len(dset["data"]) > 0, self.__dsets)
        dset = next(filled_dsets)
        return len(dset)

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_numpy(
        cls,
        pdg: Optional[np.ndarray] = None,
        pmu: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        helicity: Optional[np.ndarray] = None,
        status: Optional[np.ndarray] = None,
        final: Optional[np.ndarray] = None,
    ):
        def optional(data_class, data: Optional[np.ndarray]):
            return data_class(data) if data is not None else data_class()

        return cls(
            pdg=optional(PdgArray, pdg),
            pmu=optional(MomentumArray, pmu),
            color=optional(ColorArray, color),
            helicity=optional(HelicityArray, helicity),
            status=optional(StatusArray, status),
            final=optional(MaskArray, final),
        )


#############################################
# CONNECTIVITY INFORMATION AS COO EDGE LIST #
#############################################
if TYPE_CHECKING:

    class _AdjDict(TypedDict):
        edges: Tuple[int, int, Dict[str, Any]]
        nodes: Tuple[int, Dict[str, Any]]


@define
class AdjacencyList(AdjacencyBase):
    _data: np.ndarray = array_field("edge")
    weights: np.ndarray = array_field("double")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self._data[key]))

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_matrix(cls, adj_matrix: np.ndarray, weighted: bool = False):
        """Construct an AdjacencyList object from an optionally weighted
        adjacency matrix.

        Parameters
        ----------
        adj_matrix : array_like
            2 dimensional numpy array representing the adjacency or
            affinity matrix of a graph.
        weighted : bool
            Whether or not to propogate the numerical values in the
            elements of the adjacency matrix as the edge weights
            (default: False).
        """
        edges = np.where(adj_matrix)
        kwargs = {"data": np.array(edges).T}
        if weighted is True:
            kwargs["weights"] = adj_matrix[edges]
        return cls(**kwargs)

    @property
    def matrix(self) -> np.ndarray:
        size = len(self.nodes)
        if len(self.weights) > 0:
            weights = self.weights
            dtype = self.weights.dtype
        else:
            weights = np.array(1)
            dtype = _types.int
        adj = np.zeros((size, size), dtype=dtype)
        adj[self.edges["in"], self.edges["out"]] = weights
        return adj

    @property
    def edges(self) -> np.ndarray:
        return self._data

    @property
    def nodes(self) -> np.ndarray:
        """Nodes are extracted from the edge list, and put in
        ascending order of magnitude, regardless of sign.
        Positive sign conventionally means final state particle.
        """
        # extract nodes from edge list
        unstruc_edges = rfn.structured_to_unstructured(self._data)
        unsort_nodes = np.unique(unstruc_edges)
        sort_idxs = np.argsort(np.abs(unsort_nodes))
        return unsort_nodes[sort_idxs]  # type: ignore

    def to_dicts(
        self,
        edge_data: Optional[Dict[str, ArrayBase]] = None,
        node_data: Optional[Dict[str, ArrayBase]] = None,
    ) -> _AdjDict:
        if edge_data is None:
            edge_data = dict()
        if node_data is None:
            node_data = dict()

        def make_data_dicts(orig: Tuple[Any, ...], data: Dict[str, ArrayBase]):
            data_arrays = (array.data for array in data.values())
            data_rows = zip(*data_arrays)
            dicts = (dict(zip(data.keys(), row)) for row in data_rows)
            combo = zip_longest(*orig, dicts, fillvalue=dict())
            return tuple(combo)  # type: ignore

        # form edges with data for easier ancestry tracking
        edges = make_data_dicts(
            orig=(self.edges["in"], self.edges["out"]),
            data=edge_data,
        )
        nodes = make_data_dicts(
            orig=(self.nodes,),
            data=node_data,
        )
        return dict(
            edges=edges,
            nodes=nodes,
        )


#####################################################
# COMPOSITE OF GRAPH CONNECTIVITY AND PARTICLE DATA #
#####################################################
@define
class Graphicle:
    particles: ParticleSet = ParticleSet()
    adj: AdjacencyList = AdjacencyList()

    @property
    def __attr_names(self):
        return tuple(self.__annotations__.keys())

    def __getitem__(self, key):
        kwargs = dict()
        for name in self.__attr_names:
            data = getattr(self, name)
            kwargs.update({name: data[key]})
        return self.__class__(**kwargs)

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_numpy(
        cls,
        pdg: Optional[np.ndarray] = None,
        pmu: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        helicity: Optional[np.ndarray] = None,
        status: Optional[np.ndarray] = None,
        final: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ):
        particles = ParticleSet.from_numpy(
            pdg=pdg,
            pmu=pmu,
            color=color,
            helicity=helicity,
            status=status,
            final=final,
        )
        if edges is not None:
            kwargs = {"data": edges}
            if weights is not None:
                kwargs["weights"] = weights
            adj_list = AdjacencyList(**kwargs)
        else:
            adj_list = AdjacencyList()
        return cls(particles=particles, adj=adj_list)

    @property
    def pdg(self) -> PdgArray:
        return self.particles.pdg

    @property
    def pmu(self) -> MomentumArray:
        return self.particles.pmu

    @property
    def color(self) -> ColorArray:
        return self.particles.color

    @property
    def helicity(self) -> HelicityArray:
        return self.particles.helicity

    @property
    def status(self) -> StatusArray:
        return self.particles.status

    @property
    def hard_mask(self) -> MaskBase:
        return self.particles.status.hard_mask

    @property
    def final(self) -> MaskBase:
        return self.particles.final

    @property
    def edges(self) -> np.ndarray:
        return self.adj.edges

    @property
    def nodes(self) -> np.ndarray:
        return self.adj.nodes

    def _need_attr(self, attr_name: str, task: str) -> None:
        if len(getattr(self, attr_name)) == 0:
            raise AttributeError(
                f"Graphicle object needs '{attr_name}' attribute to {task}."
            )

    @property
    def hard_vertex(self) -> int:
        """Id of vertex at which hard interaction occurs."""
        for prop in ("status", "edges"):
            self._need_attr(attr_name=prop, task="infer hard vertex")
        hard_edges = self.edges[self.status.hard_mask.bitwise_or]
        vertex_array = np.intersect1d(hard_edges["in"], hard_edges["out"])
        central = vertex_array[np.argmin(np.abs(vertex_array))]
        return int(central)
