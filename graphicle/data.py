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
from itertools import zip_longest
from copy import deepcopy
from enum import Enum
from functools import cached_property
import warnings
from collections import abc
from typing import TYPE_CHECKING
from typing import (
    Tuple,
    List,
    Dict,
    Optional,
    Any,
    TypedDict,
    Union,
    TypeVar,
    Type,
    Iterator,
)

from attr import define, field, Factory, cmp_using, setters  # type: ignore
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array
from numpy.lib import recfunctions as rfn
from typicle import Types
from typicle.convert import cast_array

from ._base import ParticleBase, AdjacencyBase, MaskBase, ArrayBase


###########################################
# SET UP ARRAY ATTRIBUTES FOR DATACLASSES #
###########################################
_types = Types()
DoubleVector = npt.NDArray[np.float64]
ComplexVector = npt.NDArray[np.complex128]
BoolVector = npt.NDArray[np.bool_]
IntVector = npt.NDArray[np.int32]
HalfIntVector = npt.NDArray[np.int16]
ObjVector = npt.NDArray[np.object_]
AnyVector = npt.NDArray[Any]
DataType = TypeVar("DataType", bound=ArrayBase)

DHUGE = np.finfo(np.dtype("<f8")).max * 0.1


def _is_np_structured(array: AnyVector) -> bool:
    return array.dtype.names is not None


def array_field(type_name: str):
    """Prepares a field for attrs dataclass with typicle input."""
    types = Types()
    dtype = getattr(types, type_name)
    default = Factory(lambda: np.array([], dtype=dtype))
    equality_comparison = cmp_using(np.array_equal)

    def converter(values: npt.ArrayLike) -> AnyVector:
        data = np.array(values)
        out_array: AnyVector = cast_array(data, cast_type=dtype)
        return out_array

    return field(
        default=default,
        eq=equality_comparison,
        converter=converter,
        on_setattr=setters.convert,
    )


##################################
# COMPOSITE MASK DATA STRUCTURES #
##################################
class MaskAggOp(Enum):
    AND = "and"
    OR = "or"
    NONE = None


@define
class MaskArray(MaskBase, ArrayBase):
    """Data structure for containing masks over particle data."""

    data: BoolVector = array_field("bool")

    def copy(self) -> "MaskArray":
        return deepcopy(self)

    def __getitem__(self, key) -> "MaskArray":
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self) -> BoolVector:
        return self.data

    def __and__(self, other: Union[MaskBase, BoolVector]) -> MaskArray:
        if isinstance(other, MaskBase):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other
        else:
            raise ValueError(
                "Bitwise operation only supported for graphicle "
                "or numpy arrays."
            )
        return self.__class__(np.bitwise_and(self.data, other_data))

    def __or__(self, other: Union[MaskBase, BoolVector]) -> MaskArray:
        if isinstance(other, MaskBase):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other
        else:
            raise ValueError(
                "Bitwise operation only supported for graphicle "
                "or numpy arrays."
            )
        return self.__class__(np.bitwise_or(self.data, other_data))

    def __invert__(self) -> MaskArray:
        return self.__class__(~self.data)


if TYPE_CHECKING:
    _IN_MASK_DICT = Dict[str, Union[MaskArray, BoolVector]]
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
class MaskGroup(MaskBase, abc.MutableMapping[str, MaskBase]):
    """Data structure to compose groups of masks over particle arrays.
    Can be nested to form complex hierarchies.

    Parameters
    ----------
    mask_arrays : dict of MaskArrays or array-like objects
        Dictionary of MaskArray objects to be composed.
    agg_op : str or MaskAggOp
        Defines the aggregation operation when accessing the `data`
        attribute. Options include "and" and "or".
        Default is "and".

    Attributes
    ----------
    data : np.ndarray
        Combination of all masks in group via bitwise AND reduction.
    """

    _mask_arrays: _MASK_DICT = field(
        repr=False, factory=dict, converter=_mask_dict_convert
    )
    agg_op: MaskAggOp = field(converter=MaskAggOp, default=MaskAggOp.AND)

    @classmethod
    def from_numpy_structured(cls, arr: np.ndarray) -> "MaskGroup":
        return cls(dict(map(lambda name: (name, arr[name]), arr.dtype.names)))

    def __repr__(self) -> str:
        keys = ", ".join(map(lambda name: '"' + name + '"', self.names))
        return f"MaskGroup(mask_arrays=[{keys}], agg_op={self.agg_op.name})"

    def __iter__(self) -> Iterator[str]:
        return iter(self._mask_arrays)

    def __getitem__(self, key) -> Union[MaskArray, "MaskGroup"]:
        if not isinstance(key, str):
            return self.__class__(
                dict(
                    map(
                        lambda name_arr: (name_arr[0], name_arr[1][key]),
                        self._mask_arrays.items(),
                    )
                )
            )

        return self._mask_arrays[key]

    def __setitem__(self, key, mask) -> None:
        """Add a new MaskArray to the group, with given key."""
        if not isinstance(key, str):
            raise KeyError("Key must be string.")
        if not isinstance(mask, MaskBase):
            mask = MaskArray(mask)
        self._mask_arrays.update({key: mask})

    def __len__(self) -> int:
        return len(self.names)

    def __delitem__(self, key) -> None:
        """Remove a MaskArray from the group, using given key."""
        self._mask_arrays.pop(key)

    def __array__(self) -> BoolVector:
        return self.data

    def __and__(self, other: Union[MaskBase, BoolVector]) -> MaskArray:
        if isinstance(other, MaskBase):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other
        else:
            raise ValueError(
                "Bitwise operation only supported for graphicle "
                "or numpy arrays."
            )
        return MaskArray(np.bitwise_and(self.data, other_data))

    def __or__(self, other: Union[MaskBase, BoolVector]) -> MaskArray:
        if isinstance(other, MaskBase):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other
        else:
            raise ValueError(
                "Bitwise operation only supported for graphicle "
                "or numpy arrays."
            )
        return MaskArray(np.bitwise_or(self.data, other_data))

    def __invert__(self) -> MaskArray:
        return MaskArray(~self.data)

    def copy(self):
        return deepcopy(self)

    @property
    def names(self) -> List[str]:
        return list(self._mask_arrays.keys())

    @property
    def bitwise_or(self) -> BoolVector:
        return np.bitwise_or.reduce(  # type: ignore
            [child.data for child in self._mask_arrays.values()]
        )

    @property
    def bitwise_and(self) -> BoolVector:
        return np.bitwise_and.reduce(  # type: ignore
            [child.data for child in self._mask_arrays.values()]
        )

    @property
    def data(self) -> BoolVector:
        """Same as MaskGroup.bitwise_and."""
        if self.agg_op is MaskAggOp.AND:
            return self.bitwise_and
        elif self.agg_op is MaskAggOp.OR:
            return self.bitwise_or
        elif self.agg_op is MaskAggOp.NONE:
            raise ValueError(
                "No bitwise aggregation operation set for this MaskGroup."
            )
        else:
            raise NotImplementedError(
                "Aggregation operation over MaskGroup not implemented. "
                "Please contact developers with a bug report."
            )

    @property
    def dict(self) -> Dict[str, BoolVector]:
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
        Charge for each particle in elementary units.
    mass : ndarray
        Mass for each particle in GeV.
    mass_bounds : ndarray
        Mass upper and lower bounds for each particle in GeV.
    quarks : ndarray
        String representation of quark composition for each particle.
    width : ndarray
        Width for each particle in GeV.
    width_bounds : ndarray
        Width upper and lower bounds for each particle in GeV.
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

    data: IntVector = array_field("int")
    __lookup_table: __PdgRecords = field(init=False, repr=False)
    __mega_to_giga: float = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self.__lookup_table = self.__PdgRecords()
        self.__mega_to_giga: float = 1.0e-3

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self) -> IntVector:
        return self.data

    def __getitem__(self, key) -> "PdgArray":
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def copy(self) -> "PdgArray":
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

    def __get_prop(self, field: str) -> AnyVector:
        props = self.__lookup_table.properties(self.data, [field])[field]
        return props  # type: ignore

    def __get_prop_range(self, field: str) -> AnyVector:
        field_range = [field + "lower", field + "upper"]
        props = self.__lookup_table.properties(self.data, field_range)
        return props  # type: ignore

    @property
    def name(self) -> ObjVector:
        return self.__get_prop("name")

    @property
    def charge(self) -> DoubleVector:
        return self.__get_prop("charge")

    @property
    def mass(self) -> DoubleVector:
        out: DoubleVector = self.__get_prop("mass") * self.__mega_to_giga
        return out

    @property
    def mass_bounds(self) -> AnyVector:
        range_arr = self.__get_prop_range("mass")
        for name in range_arr.dtype.names:
            range_arr[name] *= self.__mega_to_giga
        return range_arr

    @property
    def quarks(self) -> ObjVector:
        return self.__get_prop("quarks")

    @property
    def width(self) -> DoubleVector:
        out: DoubleVector = self.__get_prop("width") * self.__mega_to_giga
        return out

    @property
    def width_bounds(self) -> AnyVector:
        range_arr = self.__get_prop_range("width")
        for name in range_arr.dtype.names:
            range_arr[name] *= self.__mega_to_giga
        return range_arr

    @property
    def isospin(self) -> DoubleVector:
        return self.__get_prop("i")

    @property
    def g_parity(self) -> DoubleVector:
        return self.__get_prop("g")

    @property
    def space_parity(self) -> DoubleVector:
        return self.__get_prop("p")

    @property
    def charge_parity(self) -> DoubleVector:
        return self.__get_prop("c")


########################################
# MOMENTUM STORAGE AND TRANSFORMATIONS #
########################################
@define
class MomentumArray(ArrayBase):
    """Data structure containing four-momentum of particle list.

    Attributes
    ----------
    data : np.ndarray[double]
        Structured array containing four momenta.
    pt : np.ndarray[double]
        Transverse component of particle momenta.
    eta : np.ndarray[double]
        Pseudorapidity component of particle momenta.
    phi : np.ndarray[double]
        Azimuthal component of particle momenta.
    theta : np.ndarray[double]
        Angular displacement from beam axis.
    mass : np.ndarray[double]
        Mass of the particles
    """

    data: AnyVector = array_field("pmu")

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self) -> AnyVector:
        return self.data

    def __getitem__(self, key) -> "MomentumArray":
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def copy(self) -> "MomentumArray":
        return deepcopy(self)

    @cached_property
    def _xy_pol(self) -> ComplexVector:
        return self.data["x"] + 1.0j * self.data["y"]  # type: ignore

    @cached_property
    def _zt_pol(self) -> ComplexVector:
        return self.data["z"] + 1.0j * self.pt  # type: ignore

    @cached_property
    def _spatial_mag(self) -> DoubleVector:
        return np.abs(self._zt_pol)

    @cached_property
    def pt(self) -> DoubleVector:
        """Momentum component transverse to the beam-axis."""
        return np.abs(self._xy_pol)

    @cached_property
    def eta(self) -> DoubleVector:
        """Pseudorapidity of particles.

        Notes
        -----
        Infinite values for particles travelling parallel to beam axis.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = np.arctanh(self.data["z"] / self._spatial_mag)
        return arr  # type: ignore

    @cached_property
    def rapidity(self) -> DoubleVector:
        """Rapidity of particles."""
        e, z = self.data["e"], self.data["z"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rap = 0.5 * np.log((e + z) / (e - z))  # type: ignore
        return rap  # type: ignore

    @cached_property
    def phi(self) -> DoubleVector:
        """Azimuthal angular displacement of particles about beam-axis."""
        return np.angle(self._xy_pol)

    @cached_property
    def theta(self) -> DoubleVector:
        """Angular displacement of particles from positive beam-axis."""
        return np.angle(self._zt_pol)

    @cached_property
    def mass(self) -> DoubleVector:
        """Mass of particles."""
        e: DoubleVector = self.data["e"]
        p = self._spatial_mag
        sq_diff = e * e - p * p
        sign = np.sign(sq_diff)
        return sign * np.sqrt(np.abs(sq_diff))  # type: ignore

    def delta_R(self, other: "MomentumArray") -> DoubleVector:
        """Calculates the Euclidean inter-particle distances in the
        eta-phi plane between this set of particles and a provided
        'other' set. Produces a mxn matrix, where m is number of
        particles in this MomentumArray, and n is the number of
        particles in other.

        Parameters
        ----------
        other : MomentumArray
            Four-momenta of the particle set to compute delta_R against.

        Returns
        -------
        delta_R_matrix : np.ndarray[double]
            Matrix representing the Euclidean distance between the two
            sets of particles in the eta-phi plane. Rows represent
            particles in this particle set, and columns particles in
            the other set.

        Notes
        -----
        Infinite values may be encountered if comparing with particles
        not present on the eta-phi plane, __ie.__ travelling parallel to
        the beam axis.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            deta = self.eta[:, np.newaxis] - other.eta
            deta = np.nan_to_num(deta, nan=0.0, posinf=np.inf, neginf=-np.inf)
        dphi = np.angle(self._xy_pol[:, np.newaxis] * other._xy_pol.conj())
        return np.hypot(deta, dphi)


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

    data: AnyVector = array_field("color")

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key) -> "ColorArray":
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self) -> AnyVector:
        return self.data


####################
# HELICITY STORAGE #
####################
@define
class HelicityArray(ArrayBase):
    """Data structure containing helicity / polarisation values for
    particle set.

    Attributes
    ----------
    data : ndarray
        Helicity values.
    """

    data: HalfIntVector = array_field("helicity")

    def copy(self) -> "HelicityArray":
        """Returns a new StatusArray instance with same data."""
        return deepcopy(self)

    def __getitem__(self, key) -> "HelicityArray":
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self) -> HalfIntVector:
        return self.data


####################################
# STATUS CODE STORAGE AND QUERYING #
####################################
@define
class StatusArray(ArrayBase):
    """Data structure containing status values for particle set.

    Attributes
    ----------
    data : ndarray
        Status codes.

    Notes
    -----
    These codes are specific to the Monte-Carlo event generators which
    produced the data.
    """

    data: HalfIntVector = array_field("h_int")

    def copy(self) -> "StatusArray":
        """Returns a new StatusArray instance with same data."""
        return deepcopy(self)

    def __getitem__(self, key) -> "StatusArray":
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self) -> HalfIntVector:
        return self.data

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
            },
            agg_op=MaskAggOp.OR,
        )
        return masks


#########################################
# COMPOSITE OF PARTICLE DATA STRUCTURES #
#########################################
@define
class ParticleSet(ParticleBase):
    """Composite of data structures containing particle set description.

    Attributes
    ----------
    pdg : PdgArray
        PDG codes.
    pmu : MomentumArray
        Four momenta.
    color : ColorArray
        Color / anti-color pairs.
    helicity : HelicityArray
        Helicity values.
    status : StatusArray
        Status codes from Monte-Carlo event generator.
    final : MaskArray
        Boolean array indicating final state in particle set.
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

    def __getitem__(self, key) -> "ParticleSet":
        kwargs = dict()
        for dset in self.__dsets:
            name = dset["name"]
            data = dset["data"]
            if len(data) > 0:
                kwargs.update({name: data[key]})
        return self.__class__(**kwargs)

    def __repr__(self) -> str:
        dset_repr = (repr(dset["data"]) for dset in self.__dsets)
        dset_str = ",\n".join(dset_repr)
        return f"ParticleSet(\n{dset_str}\n)"

    def __len__(self) -> int:
        filled_dsets = filter(lambda dset: len(dset["data"]) > 0, self.__dsets)
        dset = next(filled_dsets)
        return len(dset)

    def copy(self) -> "ParticleSet":
        return deepcopy(self)

    @classmethod
    def from_numpy(
        cls,
        pdg: Optional[IntVector] = None,
        pmu: Optional[AnyVector] = None,
        color: Optional[AnyVector] = None,
        helicity: Optional[HalfIntVector] = None,
        status: Optional[HalfIntVector] = None,
        final: Optional[BoolVector] = None,
    ) -> "ParticleSet":
        """Creates a ParticleSet instance directly from numpy arrays.

        Parameters
        ----------
        pdg : ndarray, optional
            PDG codes.
        pmu : ndarray, optional
            Four momenta, formatted in columns of (x, y, z, e), or as
            a structured array with those fields.
        color : ndarray, optional
            Color / anti-color pairs, formatted in columns of
            (col, acol), or as a structured array with those fields.
        helicity : ndarray, optional
            Helicity values.
        status : ndarray, optional
            Status codes from Monte-Carlo event generator.
        final : ndarray, optional
            Boolean array indicating which particles are final state.

        Returns
        -------
        particle_set : ParticleSet
            A composite object, wrapping the data provided in Graphicle
            objects, and providing a unified interface to them.
        """

        def optional(
            data_class: Type[DataType], data: Optional[AnyVector]
        ) -> DataType:
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
    """Describes relations between particles in particle set using a
    COO edge list, and provides methods to convert representation.

    Attributes
    ----------
    edges : ndarray
        COO edge list.
    nodes : ndarray
        Vertex ids of each particle with at least one edge.
    weights : ndarray
        Scalar value embedded on each edge.
    matrix : ndarray
        Adjacency matrix representation.
    """

    _data: AnyVector = array_field("edge")
    weights: DoubleVector = array_field("double")

    def __len__(self) -> int:
        return len(self._data)

    def __array__(self) -> AnyVector:
        return self._data

    def __getitem__(self, key) -> "AdjacencyList":
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self._data[key]))

    def __add__(self, other_array: "AdjacencyList") -> "AdjacencyList":
        """Combines two AdjacencyList objects by extending edge and
        weight lists of both arrays.
        If the same edge occurs in both AdjacencyLists, this will lead
        to multigraph connectivity.
        """
        if not isinstance(other_array, self.__class__):
            raise ValueError("Can only add AdjacencyList.")
        this_has_weights = len(self.weights) != 0
        other_has_weights = len(other_array.weights) != 0
        both_weighted = this_has_weights and other_has_weights
        both_unweighted = (not this_has_weights) and (not other_has_weights)
        if not (both_weighted or both_unweighted):
            raise ValueError(
                "Mismatch between weights: both adjacency lists "
                + "must either be weighted, or unweighted."
            )
        return self.__class__(
            data=np.concatenate([self._data, other_array._data]),
            weights=np.concatenate([self.weights, other_array.weights]),
        )

    def copy(self) -> "AdjacencyList":
        return deepcopy(self)

    @classmethod
    def from_matrix(
        cls,
        adj_matrix: Union[DoubleVector, BoolVector, IntVector],
        weighted: bool = False,
        self_loop: bool = False,
    ) -> "AdjacencyList":
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
        self_loop : bool
            If True will add edges from nodes to themselves, with weight
            0 (if applicable). (default: True)
        """
        sps_adj = coo_array(adj_matrix)
        if self_loop is True:
            sps_adj.setdiag(0.0)
        kwargs = {"data": np.vstack((sps_adj.row, sps_adj.col)).T}
        if weighted is True:
            kwargs["weights"] = sps_adj.data
        return cls(**kwargs)

    @property
    def matrix(self) -> Union[DoubleVector, IntVector]:
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
    def edges(self) -> AnyVector:
        return self._data

    @property
    def nodes(self) -> IntVector:
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
        edge_data: Optional[Dict[str, Union[ArrayBase, AnyVector]]] = None,
        node_data: Optional[Dict[str, Union[ArrayBase, AnyVector]]] = None,
    ) -> _AdjDict:
        """Returns data in dictionary format, which is more easily
        parsed by external libraries, such as NetworkX.
        """
        if edge_data is None:
            edge_data = dict()
        if node_data is None:
            node_data = dict()

        def make_data_dicts(
            orig: Tuple[Any, ...],
            data: Dict[str, Union[ArrayBase, AnyVector]],
        ):
            def array_iterator(array_dict):
                for array in array_dict.values():
                    if isinstance(array, ArrayBase):
                        yield array.data
                    elif isinstance(array, np.ndarray):
                        yield array
                    else:
                        raise TypeError("Data structure not supported.")

            data_rows = zip(*array_iterator(data))
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
    """Composite object, combining particle set data with relational
    information between particles.

    Attributes
    ----------
    particles : ParticleSet
        Data describing the particles in the set.
    adj : AdjacencyList
        Connectivity between the particles, to form a graph.
    pdg : PdgArray
        PDG codes.
    pmu : MomentumArray
        Four momenta.
    color : ColorArray
        Color / anti-color pairs.
    helicity : HelicityArray
        Helicity values.
    status : StatusArray
        Status codes from Monte-Carlo event generator.
    final : MaskArray
        Boolean array indicating final state in particle set.
    edges : ndarray
        COO edge list.
    nodes : ndarray
        Vertex ids of each particle with at least one edge.
    hard_mask : MaskGroup
        Identifies which particles participate in the hard process.
        For Pythia, this is split into four categories: incoming,
        intermediate, outgoing, outgoing_nonperturbative_diffraction.
    hard_vertex : int
        Vertex at which the hard process is initiated.
    """

    particles: ParticleSet = ParticleSet()
    adj: AdjacencyList = AdjacencyList()

    @property
    def __attr_names(self) -> Tuple[str, ...]:
        return tuple(self.__annotations__.keys())

    def __getitem__(self, key) -> "Graphicle":
        kwargs = dict()
        for name in self.__attr_names:
            data = getattr(self, name)
            if len(data) != 0:
                kwargs.update({name: data[key]})
        return self.__class__(**kwargs)

    def copy(self) -> "Graphicle":
        return deepcopy(self)

    @classmethod
    def from_numpy(
        cls,
        pdg: Optional[IntVector] = None,
        pmu: Optional[AnyVector] = None,
        color: Optional[AnyVector] = None,
        helicity: Optional[HalfIntVector] = None,
        status: Optional[HalfIntVector] = None,
        final: Optional[BoolVector] = None,
        edges: Optional[AnyVector] = None,
        weights: Optional[DoubleVector] = None,
    ) -> "Graphicle":
        """Instantiates a Graphicle object from an optional collection
        of numpy arrays.

        Parameters
        ----------
        pdg : ndarray, optional
            PDG codes.
        pmu : ndarray, optional
            Four momenta, formatted in columns of (x, y, z, e), or as
            a structured array with those fields.
        color : ndarray, optional
            Color / anti-color pairs, formatted in columns of
            (col, acol), or as a structured array with those fields.
        helicity : ndarray, optional
            Helicity values.
        status : ndarray, optional
            Status codes from Monte-Carlo event generator.
        final : ndarray, optional
            Boolean array indicating which particles are final state.
        edges : ndarray, optional
            COO formatted pairs of vertex ids, of shape (N, 2), where
            N is the number of particles in the graph.
            Alternatively, supply a structured array with fields
            (in, out).
        weights : ndarray, optional
            Weights to be associated with each edge in the COO edge
            list, provided in the same order.
        """

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
    def hard_mask(self) -> MaskGroup:
        return self.particles.status.hard_mask

    @property
    def final(self) -> MaskBase:
        return self.particles.final

    @property
    def edges(self) -> AnyVector:
        return self.adj.edges

    @property
    def nodes(self) -> IntVector:
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
        hard_edges = self.edges[self.status.hard_mask]
        vertex_array = np.intersect1d(hard_edges["in"], hard_edges["out"])
        central = vertex_array[np.argmin(np.abs(vertex_array))]
        return int(central)
