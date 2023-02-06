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

import collections as cl
import collections.abc as cla
import functools as fn
import itertools as it
import numbers as nm
import operator as op
import typing as ty
import warnings
from copy import deepcopy
from enum import Enum

import numpy as np
import numpy.typing as npt
from attr import Factory, cmp_using, define, field, setters
from numpy.lib import recfunctions as rfn
from rich.console import Console
from rich.tree import Tree
from scipy.sparse import coo_array
from typicle import Types

from . import base, calculate

__all__ = [
    "MaskAggOp",
    "MaskGroup",
    "MaskArray",
    "PdgArray",
    "MomentumArray",
    "MomentumElement",
    "ColorArray",
    "HelicityArray",
    "StatusArray",
    "ParticleSet",
    "AdjacencyList",
    "Graphicle",
]

###########################################
# SET UP ARRAY ATTRIBUTES FOR DATACLASSES #
###########################################
_types = Types()
DataType = ty.TypeVar("DataType", bound=base.ArrayBase)
FuncType = ty.TypeVar("FuncType", bound=ty.Callable[..., ty.Any])
EdgeLike = ty.Union[
    base.IntVector, base.VoidVector, ty.Sequence[ty.Tuple[int, int]]
]
DHUGE = np.finfo(np.dtype("<f8")).max * 0.1


def _map_invert(mapping: ty.Dict[str, ty.Set[str]]) -> ty.Dict[str, str]:
    return dict(
        it.chain.from_iterable(
            map(it.product, mapping.values(), mapping.keys())
        )
    )


_MOMENTUM_MAP = _map_invert(
    {
        "x": {"x", "px"},
        "y": {"y", "py"},
        "z": {"z", "pz"},
        "e": {"e", "pe", "tau", "t"},
    }
)
_MOMENTUM_ORDER = tuple("xyze")
_EDGE_MAP = _map_invert({"in": {"in", "src"}, "out": {"out", "dst"}})
_EDGE_ORDER = ("in", "out")


class MomentumElement(ty.NamedTuple):
    """Named tuple container for the momentum of a single particle."""

    x: float
    y: float
    z: float
    e: float


class ColorElement(ty.NamedTuple):
    """Named tuple container for the color / anticolor pair of a single
    particle.
    """

    color: int
    anticolor: int


class VertexPair(ty.NamedTuple):
    """Named tuple container for the color / anticolor pair of a single
    particle.
    """

    src: int
    dst: int


# from https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins
# .NDArrayOperatorsMixin.html
def _array_ufunc(
    instance: DataType,
    ufunc: ty.Callable[..., ty.Any],
    method: str,
    *inputs: ty.Tuple[ty.Any, ...],
    **kwargs: ty.Dict[str, ty.Any],
) -> ty.Optional[
    ty.Union[
        ty.Tuple[ty.Union[DataType, base.AnyVector]],
        ty.Union[DataType, base.AnyVector],
    ]
]:
    out = kwargs.get("out", ())
    class_type = instance.__class__
    for x in inputs + out:  # type: ignore
        # Only support operations with instances of _HANDLED_TYPES.
        # Use ArrayLike instead of type(self) for isinstance to
        # allow subclasses that don't override __array_ufunc__ to
        # handle ArrayLike objects.
        if not isinstance(x, instance._HANDLED_TYPES + (class_type,)):
            return NotImplemented

    # Defer to the implementation of the ufunc on unwrapped values.
    inputs = tuple(x._data if isinstance(x, class_type) else x for x in inputs)
    if out:
        kwargs["out"] = tuple(
            x._data if isinstance(x, class_type) else x for x in out
        )
    result = op.methodcaller(method, *inputs, **kwargs)(ufunc)

    if type(result) is tuple:
        # multiple return values
        return tuple(class_type(x) for x in result)
    elif method == "at":
        # no return value
        return None
    elif isinstance(result, np.ndarray) and (result.dtype == np.bool_):
        return MaskArray(result)
    else:
        # one return value
        if not result.shape or kwargs.get("axis", None) == 1:
            return result
        return class_type(result)


def _array_eq_prep(
    instance: base.ArrayBase, other: ty.Union[base.ArrayBase, base.AnyVector]
) -> ty.Tuple[base.AnyVector, base.AnyVector]:
    other_data = other
    dtype = instance.data.dtype
    if isinstance(other, base.ArrayBase):
        other_data = other.data
    elif (dtype.type == np.void) and not (
        isinstance(other, np.ndarray) and (other.dtype == np.void)
    ):
        dt_set = set(map(op.itemgetter(1), dtype.descr))
        assert len(dt_set) == 1
        if (not isinstance(other, cla.Sequence)) or (
            len(other) != len(dtype.names)  # type: ignore
        ):
            raise ValueError(
                f"Cannot compare {other} against {instance.__class__.__name__}"
                ", incompatible shape or type."
            )
        other_data = np.array(other, dtype=dt_set.pop()).view(dtype)
    return instance.data, other_data  # type: ignore


def _array_eq(
    instance: base.ArrayBase, other: ty.Union[base.ArrayBase, base.AnyVector]
) -> "MaskArray":
    x, y = _array_eq_prep(instance, other)
    return MaskArray(x == y)


def _array_ne(
    instance: base.ArrayBase, other: ty.Union[base.ArrayBase, base.AnyVector]
) -> "MaskArray":
    x, y = _array_eq_prep(instance, other)
    return MaskArray(x != y)


def _array_repr(instance: base.ArrayBase) -> str:
    data_str = str(instance._data)
    data_splits = data_str.split("\n")
    first_str = data_splits.pop(0)
    class_name = instance.__class__.__name__
    idnt_lvl = len(class_name) + 1
    idnt = " " * idnt_lvl
    dtype_str = f"dtype={instance.data.dtype}"
    if not data_splits:
        return f"{class_name}({first_str}, {dtype_str})"
    dtype_str = idnt + dtype_str
    rows = "\n".join(map(op.add, it.repeat(idnt), data_splits))
    return f"{class_name}({first_str}\n{rows},\n{dtype_str})"


def array_field(type_name: str):
    """Prepares a field for attrs dataclass with typicle input."""
    types = Types()
    dtype = np.dtype(getattr(types, type_name))
    default = Factory(lambda: np.array([], dtype=dtype))

    def converter(values: npt.ArrayLike) -> base.AnyVector:
        if not isinstance(values, np.ndarray):
            data = np.array(values)
        else:
            data = values
        if data.dtype != dtype:
            if dtype.names is None:
                return data.astype(dtype)
            return rfn.unstructured_to_structured(data, dtype)
        return data

    return field(
        default=default,
        eq=cmp_using(np.array_equal),
        converter=converter,
        on_setattr=setters.convert,
    )


def _reorder_pmu(array: base.VoidVector) -> base.VoidVector:
    names = array.dtype.names
    assert names is not None
    if names == _MOMENTUM_ORDER:
        return array
    gcl_to_ext = dict(zip(map(_MOMENTUM_MAP.__getitem__, names), names))
    name_reorder = list(map(gcl_to_ext.__getitem__, _MOMENTUM_ORDER))
    return array[name_reorder]


def _array_field(dtype: npt.DTypeLike, num_cols: int = 1, repr: bool = True):
    dtype = np.dtype(dtype)

    def converter(values: npt.ArrayLike) -> base.AnyVector:
        if isinstance(values, np.ndarray) and values.dtype.type == np.void:
            names = values.dtype.names
            assert names is not None
            if num_cols == 4:
                values = _reorder_pmu(values)
            values = rfn.structured_to_unstructured(values)
        array = np.asarray(values, dtype=dtype)
        shape = array.shape
        is_flat = len(shape) == 1
        if is_flat and (num_cols == 1):
            return array
        cols = shape[0] if is_flat else shape[1]
        not_empty = array.shape[0] != 0
        if not_empty and (cols != num_cols):
            raise ValueError(f"Number of columns must be equal to {num_cols}")
        return array

    return field(
        default=Factory(
            fn.partial(np.array, tuple(), dtype=dtype)  # type: ignore
        ),
        converter=converter,
        eq=cmp_using(np.array_equal),
        on_setattr=setters.convert,
        repr=repr,
    )


def _truthy(data: ty.Union[base.ArrayBase, base.AdjacencyBase]) -> bool:
    if len(data) == 0:
        return False
    return True


##################################
# COMPOSITE MASK DATA STRUCTURES #
##################################
class MaskAggOp(Enum):
    AND = "and"
    OR = "or"
    NONE = None


@define
class MaskArray(base.MaskBase, base.ArrayBase):
    """Boolean mask over Graphicle data structures.

    :group: datastructure

    Parameters
    ----------
    data : Sequence[bool]
        Boolean values consituting the mask.

    Attributes
    ----------
    data : ndarray[bool_]
        Numpy representation of the boolean mask.

    Methods
    -------
    copy()
        Provides a deepcopy of the data.

    Examples
    --------
    Instantiating, copying, updating by index, and comparison:

        >>> import graphicle as gcl
        >>> mask1 = gcl.MaskArray([True, True, False])
        >>> mask2 = mask1.copy()
        >>> mask2[1] = False
        >>> mask2
        MaskArray(data=array([ True, False, False]))
        >>> mask1 == mask2
        MaskArray(data=array([ True, False,  True]))
        >>> mask1 != mask2
        MaskArray(data=array([False,  True, False]))
    """

    _data: base.BoolVector = _array_field("<?")
    dtype: np.dtype = field(init=False, repr=False)
    __array_interface__: ty.Dict[str, ty.Any] = field(init=False, repr=False)
    _HANDLED_TYPES: ty.Tuple[ty.Type, ...] = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.dtype = self._data.dtype
        self.__array_interface__ = self._data.__array_interface__
        self._HANDLED_TYPES = (np.ndarray, nm.Number, cla.Sequence)

    def __array__(self) -> base.BoolVector:
        return self.data

    @classmethod
    def __array_wrap__(cls, array: base.BoolVector) -> "MaskArray":
        return cls(array)

    def __iter__(self) -> ty.Iterator[bool]:
        yield from map(bool, self.data)

    def copy(self) -> "MaskArray":
        return deepcopy(self)

    def __repr__(self) -> str:
        return _array_repr(self)

    def __getitem__(self, key) -> "MaskArray":
        if isinstance(key, base.MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __setitem__(self, key, val) -> None:
        if isinstance(key, base.MaskBase):
            key = key.data
        self.data[key] = val

    def __len__(self) -> int:
        return len(self.data)

    def __and__(self, other: base.MaskLike) -> "MaskArray":
        if isinstance(other, base.MaskBase):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other
        else:
            raise ValueError(
                "Bitwise operation only supported for graphicle "
                "or numpy arrays."
            )
        return self.__class__(np.bitwise_and(self.data, other_data))

    def __or__(self, other: base.MaskLike) -> "MaskArray":
        if isinstance(other, base.MaskBase):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other
        else:
            raise ValueError(
                "Bitwise operation only supported for graphicle "
                "or numpy arrays."
            )
        return self.__class__(np.bitwise_or(self.data, other_data))

    def __invert__(self) -> "MaskArray":
        return self.__class__(~self.data)

    def __eq__(self, other: base.MaskLike) -> "MaskArray":
        return _mask_eq(self.data, other)

    def __ne__(self, other: base.MaskLike) -> "MaskArray":
        return _mask_neq(self, other)

    def __bool__(self) -> bool:
        return _truthy(self)

    @property
    def data(self) -> base.BoolVector:
        return self._data

    @data.setter
    def data(
        self, values: ty.Union[base.BoolVector, ty.Sequence[bool]]
    ) -> None:
        self._data = values  # type: ignore


def _mask_compat(*masks: base.MaskLike) -> bool:
    """Check if a collection of masks are compatible, ie. they must be
    either numpy boolean arrays, or ``MaskBase`` instances.
    """
    for mask in masks:
        valid = isinstance(mask, base.MaskBase) or isinstance(mask, np.ndarray)
        if not valid:
            return False
    return True


def _mask_eq(mask1: base.MaskLike, mask2: base.MaskLike) -> MaskArray:
    """Provides a boolean comparison ``==`` operation for ``MaskBase``
    objects against other masks (including numpy boolean arrays).

    Parameters
    ----------
    mask1 : base.MaskBase | numpy.ndarray
        Boolean array to compare against.
    mask2 : base.MaskBase | numpy.ndarray
        The other array.
    """
    if not _mask_compat(mask1, mask2):
        raise ValueError(
            "Bitwise operation supported for graphicle or numpy arrays."
        )
    shape1, shape2 = np.shape(mask1), np.shape(mask2)
    if shape1 != shape2:
        raise ValueError(f"Incompatible mask shapes {shape1} and {shape2}.")
    return MaskArray(np.equal(mask1, mask2))


def _mask_neq(mask1: base.MaskLike, mask2: base.MaskLike) -> MaskArray:
    """Provides a boolean comparison ``!=`` operation for ``MaskBase``
    objects against other masks (including numpy boolean arrays).

    Parameters
    ----------
    mask1 : base.MaskBase | numpy.ndarray
        Boolean array to compare against.
    mask2 : base.MaskBase | numpy.ndarray
        The other array.
    """
    if not _mask_compat(mask1, mask2):
        raise ValueError(
            "Bitwise operation supported for graphicle or numpy arrays."
        )
    shape1, shape2 = np.shape(mask1), np.shape(mask2)
    if shape1 != shape2:
        raise ValueError(f"Incompatible mask shapes {shape1} and {shape2}.")
    return MaskArray(np.not_equal(mask1, mask2))


_IN_MASK_DICT = ty.OrderedDict[str, ty.Union[MaskArray, base.BoolVector]]
_MASK_DICT = ty.OrderedDict[str, MaskArray]


def _mask_dict_convert(masks: _IN_MASK_DICT) -> _MASK_DICT:
    out_masks = cl.OrderedDict()
    for key, val in masks.items():
        if isinstance(val, MaskArray) or isinstance(val, MaskGroup):
            mask = val
        else:
            mask = MaskArray(val)
        out_masks[key] = mask
    return out_masks


@define
class MaskGroup(base.MaskBase, cla.MutableMapping[str, base.MaskBase]):
    """Data structure to compose groups of masks over particle arrays.
    Can be nested to form complex hierarchies.

    :group: datastructure

    Parameters
    ----------
    _mask_arrays : dict of MaskArrays or array-like objects
        Dictionary of MaskArray objects to be composed.
    agg_op : str or MaskAggOp
        Defines the aggregation operation when accessing the `data`
        attribute. Options include "and" and "or".
        Default is "and".

    Attributes
    ----------
    data : np.ndarray
        Combination of all masks in group via bitwise AND reduction.
    names : list[str]
        Provides the string values of the keys to the top-level nested
        ``MaskBase`` objects as a list. Will be deprecated in future.
        ``MaskGroup.keys()`` is preferred.
    bitwise_or : np.ndarray[bool_]
        Bitwise ``OR`` reduction over the nested masks.
    bitwise_or : np.ndarray[bool_]
        Bitwise ``AND`` reduction over the nested masks.
    dict : dict[base.MaskBase]
        Masks nested in a dictionary instead of a ``MaskGroup``.

    Methods
    -------
    from_numpy_structured()
        Converts a structured boolean array into a ``MaskGroup``.
    flatten()
        Removes any nesting of ``MaskGroup``s within ``MaskGroup``s.
    copy()
        Copies the ``MaskGroup`` instance.
    """

    _mask_arrays: _MASK_DICT = field(
        repr=False, factory=dict, converter=_mask_dict_convert
    )
    agg_op: MaskAggOp = field(converter=MaskAggOp, default=MaskAggOp.AND)

    @classmethod
    def from_numpy_structured(cls, arr: np.ndarray) -> "MaskGroup":
        return cls(
            dict(  # type: ignore
                map(lambda name: (name, arr[name]), arr.dtype.names)
            )
        )

    def __repr__(self) -> str:
        keys = ", ".join(map(lambda name: '"' + name + '"', self.names))
        return f"MaskGroup(masks=[{keys}], agg_op={self.agg_op.name})"

    def __rich__(self) -> Tree:
        name = self.__class__.__name__
        agg_op = self.agg_op.name
        tree = Tree(f"{name}(agg_op=[yellow]{agg_op}[default])")

        def make_tree(mask: "MaskGroup", branch: Tree) -> Tree:
            branch_copy = deepcopy(branch)
            for key, val in mask.items():
                sub_branch = Tree(f"{key}")
                if isinstance(val, MaskGroup):
                    sub_branch = sub_branch.add(make_tree(val, sub_branch))
                branch_copy.add(sub_branch)
            return branch_copy

        return make_tree(self, tree)

    def __str__(self) -> str:
        console = Console(color_system=None)
        with console.capture() as capture:
            console.print(self)
        return capture.get()

    def __iter__(self) -> ty.Iterator[str]:
        return iter(self._mask_arrays)

    def __getitem__(self, key) -> ty.Union[MaskArray, "MaskGroup"]:
        """Subscripting for ``MaskGroup`` object.

        Parameters
        ----------
        key : str, list[str], slice, np.ndarray[bool_], base.MaskBase
            If string, will return the ``MaskBase`` object associated
            with a key of the same name. If list of strings, will return
            a new ``MaskGroup`` with only the keys included in the list.
            Otherwise will be treated as an array-like slice, returning
            the ``MaskGroup`` whose components each individually have
            the passed slice applied.
        """
        agg = self.agg_op
        if isinstance(key, list):
            return self.__class__(
                cl.OrderedDict({k: self._mask_arrays[k] for k in key}), agg
            )
        elif not isinstance(key, str):
            masked_data = cl.OrderedDict()
            for dict_key, val in self._mask_arrays.items():
                masked_data[dict_key] = val[key]
            return self.__class__(masked_data, agg)
        else:
            return self._mask_arrays[key]

    def __setitem__(self, key, mask) -> None:
        """Add a new MaskArray to the group, with given key."""
        if not isinstance(key, str):
            raise KeyError("Key must be string.")
        if not isinstance(mask, base.MaskBase):
            mask = MaskArray(mask)
        self._mask_arrays.update({key: mask})

    def __bool__(self) -> bool:
        if np.shape(self.data)[0] == 0:
            return False
        return True

    def __len__(self) -> int:
        return len(self.names)

    def __delitem__(self, key) -> None:
        """Remove a MaskArray from the group, using given key."""
        self._mask_arrays.pop(key)

    def __array__(self) -> base.BoolVector:
        return self.data

    def __and__(self, other: base.MaskLike) -> MaskArray:
        if isinstance(other, base.MaskBase):
            other_data = other.data
        elif isinstance(other, np.ndarray):
            other_data = other
        else:
            raise ValueError(
                "Bitwise operation only supported for graphicle "
                "or numpy arrays."
            )
        return MaskArray(np.bitwise_and(self.data, other_data))

    def __or__(self, other: base.MaskLike) -> MaskArray:
        if isinstance(other, base.MaskBase):
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

    def __eq__(self, other: base.MaskLike) -> "MaskArray":
        return _mask_eq(self, other)

    def __ne__(self, other: base.MaskLike) -> "MaskArray":
        return _mask_neq(self, other)

    def copy(self):
        return deepcopy(self)

    @property
    def names(self) -> ty.List[str]:
        return list(self._mask_arrays.keys())

    @property
    def bitwise_or(self) -> base.BoolVector:
        return np.bitwise_or.reduce(  # type: ignore
            [child.data for child in self._mask_arrays.values()]
        )

    @property
    def bitwise_and(self) -> base.BoolVector:
        return np.bitwise_and.reduce(  # type: ignore
            [child.data for child in self._mask_arrays.values()]
        )

    @property
    def data(self) -> base.BoolVector:
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
    def dict(self) -> ty.Dict[str, base.BoolVector]:
        return {key: val.data for key, val in self._mask_arrays.items()}

    def flatten(self) -> "MaskGroup":
        """Removes nesting such that the ``MaskGroup`` contains only
        ``MaskArray``s, and no other ``MaskGroup``s.

        Returns
        -------
        flat_masks : MaskGroup
            ``MaskGroup`` in which all sub-``MaskGroup``s are aggregated
            and placed at the top level of the outer ``MaskGroup``,
            along with the ``MaskArray``s from the innermost levels.
        """

        def leaves(mask_group: "MaskGroup"):
            for key, val in mask_group.items():
                if key == "latent":
                    continue
                if isinstance(val, type(self)):
                    yield key, val.data
                    yield from leaves(val)
                else:
                    yield key, val

        return self.__class__(dict(leaves(self)), "or")  # type: ignore


############################
# PDG STORAGE AND QUERYING #
############################
@define(eq=False)
class PdgArray(base.ArrayBase):
    """Returns data structure containing PDG integer codes for particle
    data.

    :group: datastructure

    Parameters
    ----------
    data : Sequence[int]
        The PDG codes for each particle in the point cloud.

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

    _data: base.IntVector = _array_field("<i4")
    dtype: np.dtype = field(init=False, repr=False)
    __lookup_table: __PdgRecords = field(init=False, repr=False)
    __mega_to_giga: float = field(init=False, repr=False)
    __array_interface__: ty.Dict[str, ty.Any] = field(init=False, repr=False)
    _HANDLED_TYPES = field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self.__lookup_table = self.__PdgRecords()
        self.__mega_to_giga: float = 1.0e-3
        self.__array_interface__ = self._data.__array_interface__
        self.dtype = self._data.dtype
        self._HANDLED_TYPES = (np.ndarray, nm.Number, cla.Sequence)

    @classmethod
    def __array_wrap__(cls, array: base.IntVector) -> "PdgArray":
        return cls(array)

    def __repr__(self) -> str:
        return _array_repr(self)

    def __iter__(self) -> ty.Iterator[int]:
        yield from map(int, self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self) -> base.IntVector:
        return self.data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def __bool__(self) -> bool:
        return _truthy(self)

    def __eq__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_eq(self, other)

    def __ne__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_ne(self, other)

    def __getitem__(self, key) -> "PdgArray":
        if isinstance(key, base.MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    @property
    def data(self) -> base.IntVector:
        return self._data

    @data.setter
    def data(self, values: npt.ArrayLike) -> None:
        self._data = values  # type: ignore

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

    def __get_prop(self, field: str) -> base.VoidVector:
        props = self.__lookup_table.properties(self.data, [field])[field]
        return props  # type: ignore

    def __get_prop_range(self, field: str) -> base.VoidVector:
        field_range = [field + "lower", field + "upper"]
        props = self.__lookup_table.properties(self.data, field_range)
        return props  # type: ignore

    @property
    def name(self) -> base.ObjVector:
        return self.__get_prop("name")

    @property
    def charge(self) -> base.DoubleVector:
        return self.__get_prop("charge")

    @property
    def mass(self) -> base.DoubleVector:
        out: base.DoubleVector = self.__get_prop("mass") * self.__mega_to_giga
        return out

    @property
    def mass_bounds(self) -> base.VoidVector:
        range_arr = self.__get_prop_range("mass")
        for name in range_arr.dtype.names:
            range_arr[name] *= self.__mega_to_giga
        return range_arr

    @property
    def quarks(self) -> base.ObjVector:
        return self.__get_prop("quarks")

    @property
    def width(self) -> base.DoubleVector:
        out: base.DoubleVector = self.__get_prop("width") * self.__mega_to_giga
        return out

    @property
    def width_bounds(self) -> base.VoidVector:
        range_arr = self.__get_prop_range("width")
        for name in range_arr.dtype.names:
            range_arr[name] *= self.__mega_to_giga
        return range_arr

    @property
    def isospin(self) -> base.DoubleVector:
        return self.__get_prop("i")

    @property
    def g_parity(self) -> base.DoubleVector:
        return self.__get_prop("g")

    @property
    def space_parity(self) -> base.DoubleVector:
        return self.__get_prop("p")

    @property
    def charge_parity(self) -> base.DoubleVector:
        return self.__get_prop("c")


########################################
# MOMENTUM STORAGE AND TRANSFORMATIONS #
########################################
@define(eq=False)
class MomentumArray(base.ArrayBase):
    """Data structure containing four-momentum of particle list.

    :group: datastructure

    Parameters
    ----------
    data : np.ndarray[np.float64]
        Data representing the four-momentum of each particle in the
        point cloud. Given as either a (n, 4)-dimensional numpy array,
        or a structured array, with fields "x", "y", "z", "e".

    Attributes
    ----------
    data : ndarray[float64]
        Structured array containing four momenta.
    pt : ndarray[float64]
        Transverse component of particle momenta.
    rapidity : ndarray[float64]
        Rapidity component of the particle momenta.
    eta : ndarray[float64]
        Pseudorapidity component of particle momenta.
    phi : ndarray[float64]
        Azimuthal component of particle momenta.
    theta : np.ndarray[double]
        Angular displacement from beam axis.
    mass : np.ndarray[double]
        Mass of the particles

    Methods
    -------
    delta_R()
        Calculates interparticle distances with ``other`` MomentumArray.
    """

    # data: base.AnyVector = array_field("pmu")
    _data: base.DoubleVector = _array_field("<f8", num_cols=4, repr=False)
    dtype: np.dtype = field(init=False, repr=False)
    __array_interface__: ty.Dict[str, ty.Any] = field(init=False, repr=False)
    _HANDLED_TYPES: ty.Tuple[ty.Type, ...] = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.dtype = np.dtype(list(zip("xyze", it.repeat("<f8"))))
        self.__array_interface__ = self._data.__array_interface__
        self._HANDLED_TYPES = (np.ndarray, nm.Number, cla.Sequence)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, *inputs, **kwargs)

    def __repr__(self) -> str:
        return _array_repr(self)

    @property
    def data(self) -> base.VoidVector:
        return self._data.view(self.dtype).squeeze()

    @data.setter
    def data(self, values: npt.ArrayLike) -> None:
        self._data = values  # type: ignore

    @classmethod
    def __array_wrap__(cls, array: base.AnyVector) -> "MomentumArray":
        return cls(array)

    def __iter__(self) -> ty.Iterator[MomentumElement]:
        flat_vals = map(float, self._data.flatten())
        elems = zip(*(flat_vals,) * 4, strict=True)  # type: ignore
        yield from it.starmap(MomentumElement, elems)

    def __len__(self) -> int:
        return len(self.data)

    def __array__(self) -> base.VoidVector:
        return self.data

    def __getitem__(self, key) -> "MomentumArray":
        if isinstance(key, base.MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __bool__(self) -> bool:
        return _truthy(self)

    def __eq__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_eq(self, other)

    def __ne__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_ne(self, other)

    def copy(self) -> "MomentumArray":
        return deepcopy(self)

    @property
    def _xy_pol(self) -> base.ComplexVector:
        return self._data[:, :2].view(dtype="<c16").reshape(-1)

    @fn.cached_property
    def _zt_pol(self) -> base.ComplexVector:
        return (
            np.stack((self.data["z"], np.abs(self._xy_pol)), axis=-1)
            .view(dtype="<c16")
            .reshape(-1)
        )

    @fn.cached_property
    def _spatial_mag(self) -> base.DoubleVector:
        return np.abs(self._zt_pol)

    @property
    def pt(self) -> base.DoubleVector:
        """Momentum component transverse to the beam-axis."""
        return self._zt_pol.imag

    @fn.cached_property
    def eta(self) -> base.DoubleVector:
        """Pseudorapidity of particles.

        Notes
        -----
        Infinite values for particles travelling parallel to beam axis.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = np.arctanh(self.data["z"] / self._spatial_mag)
        return arr  # type: ignore

    @fn.cached_property
    def rapidity(self) -> base.DoubleVector:
        """Rapidity of particles."""
        e, z = self.data["e"], self.data["z"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rap = 0.5 * np.log((e + z) / (e - z))  # type: ignore
        return rap  # type: ignore

    @fn.cached_property
    def phi(self) -> base.DoubleVector:
        """Azimuthal angular displacement of particles about beam-axis."""
        return np.angle(self._xy_pol)

    @fn.cached_property
    def theta(self) -> base.DoubleVector:
        """Angular displacement of particles from positive beam-axis."""
        return np.angle(self._zt_pol)

    @fn.cached_property
    def mass(self) -> base.DoubleVector:
        """Mass of particles."""
        return calculate._root_diff_two_squares(
            self.data["e"], self._spatial_mag
        )

    def delta_R(self, other: "MomentumArray") -> base.DoubleVector:
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
@define(eq=False)
class ColorArray(base.ArrayBase):
    """Returns data structure of color / anti-color pairs for particle
    shower.

    :group: datastructure

    Parameters
    ----------
    data : np.ndarray[np.int32]
        Data representing the QCD color charge of each particle in the
        point cloud. Given as either a (n, 2)-dimensional numpy array,
        or a structured array, with fields "color", "anticolor".

    Attributes
    ----------
    data : ndarray
        Structured array containing color / anti-color pairs.
    """

    _data: base.VoidVector = _array_field("<i4", 2)
    __array_interface__: ty.Dict[str, ty.Any] = field(init=False, repr=False)
    dtype: np.dtype = field(init=False, repr=False)
    _HANDLED_TYPES: ty.Tuple[ty.Type, ...] = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.__array_interface__ = self._data.__array_interface__
        self.dtype = np.dtype(
            list(zip(("color", "anticolor"), it.repeat("<i4")))
        )
        self._HANDLED_TYPES = (np.ndarray, nm.Number, cla.Sequence)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, *inputs, **kwargs)

    @classmethod
    def __array_wrap__(cls, array: base.AnyVector) -> "ColorArray":
        return cls(array)

    def __array__(self) -> base.VoidVector:
        return self.data

    def __repr__(self) -> str:
        return _array_repr(self)

    def __iter__(self) -> ty.Iterator[ColorElement]:
        flat_vals = map(int, it.chain.from_iterable(self.data))
        elems = zip(*(flat_vals,) * 2, strict=True)  # type: ignore
        yield from it.starmap(ColorElement, elems)

    @property
    def data(self) -> base.VoidVector:
        return self._data.view(self.dtype).squeeze()

    @data.setter
    def data(self, values: npt.ArrayLike) -> None:
        self._data = values  # type: ignore

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key) -> "ColorArray":
        if isinstance(key, base.MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return _truthy(self)

    def __eq__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_eq(self, other)

    def __ne__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_ne(self, other)


####################
# HELICITY STORAGE #
####################
@define(eq=False)
class HelicityArray(base.ArrayBase):
    """Data structure containing helicity / polarisation values for
    particle set.

    :group: datastructure

    Parameters
    ----------
    data : Sequence[int]
        Data representing the spin polarisation of each particle in the
        point cloud.

    Attributes
    ----------
    data : ndarray[int16]
        Helicity values.
    """

    _data: base.HalfIntVector = _array_field("<i2")
    dtype: np.dtype = field(init=False, repr=False)
    __array_interface__: ty.Dict[str, ty.Any] = field(init=False, repr=False)
    _HANDLED_TYPES: ty.Tuple[ty.Type, ...] = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.__array_interface__ = self._data.__array_interface__
        self.dtype = self._data.dtype
        self._HANDLED_TYPES = (np.ndarray, nm.Number, cla.Sequence)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, *inputs, **kwargs)

    @classmethod
    def __array_wrap__(cls, array: base.HalfIntVector) -> "HelicityArray":
        return cls(array)

    def __array__(self) -> base.HalfIntVector:
        return self.data

    def __repr__(self) -> str:
        return _array_repr(self)

    def __iter__(self) -> ty.Iterator[int]:
        yield from map(int, self.data)

    @property
    def data(self) -> base.HalfIntVector:
        return self._data

    @data.setter
    def data(self, values: npt.ArrayLike) -> None:
        self._data = values  # type: ignore

    def copy(self) -> "HelicityArray":
        """Returns a new HelicityArray instance with same data."""
        return deepcopy(self)

    def __getitem__(self, key) -> "HelicityArray":
        if isinstance(key, base.MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return _truthy(self)

    def __eq__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_eq(self, other)

    def __ne__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_ne(self, other)


####################################
# STATUS CODE STORAGE AND QUERYING #
####################################
@define(eq=False)
class StatusArray(base.ArrayBase):
    """Data structure containing status values for particle set.

    :group: datastructure

    Parameters
    ----------
    data : Sequence[int]
        Data representing the Monte-Carlo event generator's status for
        each particle in the point cloud.

    Attributes
    ----------
    data : ndarray[int16]
        Status codes.

    Notes
    -----
    These codes are specific to the Monte-Carlo event generators which
    produced the data.
    """

    _data: base.HalfIntVector = _array_field("<i2")
    __array_interface__: ty.Dict[str, ty.Any] = field(init=False, repr=False)
    dtype: np.dtype = field(init=False, repr=False)
    _HANDLED_TYPES: ty.Tuple[ty.Type, ...] = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.__array_interface__ = self._data.__array_interface__
        self.dtype = self._data.dtype
        self._HANDLED_TYPES = (np.ndarray, nm.Number, cla.Sequence)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _array_ufunc(self, ufunc, method, *inputs, **kwargs)

    @classmethod
    def __array_wrap__(cls, array: base.HalfIntVector) -> "StatusArray":
        return cls(array)

    def __array__(self) -> base.HalfIntVector:
        return self.data

    def __repr__(self) -> str:
        return _array_repr(self)

    def __iter__(self) -> ty.Iterator[int]:
        yield from map(int, self.data)

    def __getitem__(self, key) -> "StatusArray":
        if isinstance(key, base.MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> bool:
        return _truthy(self)

    def __eq__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_eq(self, other)

    def __ne__(
        self, other: ty.Union[base.ArrayBase, base.AnyVector]
    ) -> MaskArray:
        return _array_ne(self, other)

    @property
    def data(self) -> base.HalfIntVector:
        return self._data

    @data.setter
    def data(self, values: npt.ArrayLike) -> None:
        self._data = values  # type: ignore

    def copy(self) -> "StatusArray":
        """Returns a new StatusArray instance with same data."""
        return deepcopy(self)

    def in_range(
        self,
        min_status: int = 0,
        max_status: ty.Optional[int] = None,
        sign_sensitive: bool = False,
    ) -> MaskArray:
        """Returns a boolean mask over particles with status codes
        sitting within passed (inclusive) range.

        Parameters
        ----------
        min_status : int
            Minimum value for status codes. Default is 0.
        max_status : int, optional
            Maximum value for status codes. Passing ``None`` results in
            unbounded upper range. Default is ``None``.
        sign_sensitive : bool
            Whether or not to take signs into account during the
            comparison. Default is False.

        Returns
        -------
        mask_out : MaskArray
            Boolean mask over the particle dataset which selects
            data where min_status <= status <= max_status.
        """
        array = self.data
        if sign_sensitive is False:
            array = np.abs(array)
        more_than = array >= min_status
        if max_status is None:
            return MaskArray(more_than)
        less_than = array <= max_status
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
class ParticleSet(base.ParticleBase):
    """Composite of data structures containing particle set description.

    :group: datastructure

    Parameters
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

    pdg: PdgArray = field(default=Factory(fn.partial(PdgArray, tuple())))
    pmu: MomentumArray = field(default=Factory(fn.partial(PdgArray, tuple())))
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

    def __bool__(self) -> bool:
        for dset in self.__dsets:
            if dset["data"]:
                return True
        return False

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
        pdg: ty.Optional[base.IntVector] = None,
        pmu: ty.Optional[base.VoidVector] = None,
        color: ty.Optional[base.VoidVector] = None,
        helicity: ty.Optional[base.HalfIntVector] = None,
        status: ty.Optional[base.HalfIntVector] = None,
        final: ty.Optional[base.BoolVector] = None,
    ) -> "ParticleSet":
        """Creates a ParticleSet instance directly from numpy arrays.

        Parameters
        ----------
        pdg : ndarray[int32], optional
            PDG codes.
        pmu : ndarray[float64], optional
            Four momenta, formatted in columns of (x, y, z, e), or as
            a structured array with those fields.
        color : ndarray[int32], optional
            Color / anti-color pairs, formatted in columns of
            (col, acol), or as a structured array with those fields.
        helicity : ndarray[int16], optional
            Helicity values.
        status : ndarray[int32], optional
            Status codes from Monte-Carlo event generator.
        final : ndarray[bool_], optional
            Boolean array indicating which particles are final state.

        Returns
        -------
        particle_set : ParticleSet
            A composite object, wrapping the data provided in Graphicle
            objects, and providing a unified interface to them.
        """

        def optional(
            data_class: ty.Type[DataType], data: ty.Optional[base.AnyVector]
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
class _AdjDict(ty.TypedDict):
    edges: ty.Tuple[int, int, ty.Dict[str, ty.Any]]
    nodes: ty.Tuple[int, ty.Dict[str, ty.Any]]


@define
class AdjacencyList(base.AdjacencyBase):
    """Describes relations between particles in particle set using a
    COO edge list, and provides methods to convert representation.

    :group: datastructure

    Parameters
    ----------
    _data : np.ndarray[np.int32]
        COO formatted edge pairs, either given as a (n-2)-dimensional
        array, or a structured array with fields "in", "out".
    weights : np.ndarray[np.float64]
        Weights attributed to each edge in the COO list.

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

    _data: base.AnyVector = array_field("edge")
    weights: base.DoubleVector = array_field("double")
    __array_interface__: ty.Dict[str, ty.Any] = field(init=False, repr=False)

    def __attrs_post_init__(self):
        self.__array_interface__ = self._data.__array_interface__

    @classmethod
    def __array_wrap__(cls, array: base.AnyVector) -> "AdjacencyList":
        return cls(array)

    def __iter__(self) -> ty.Iterator[VertexPair]:
        flat_vals = map(int, it.chain.from_iterable(self._data))
        elems = zip(*(flat_vals,) * 2, strict=True)  # type: ignore
        yield from it.starmap(VertexPair, elems)

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return _truthy(self)

    def __array__(self) -> base.VoidVector:
        return self._data

    def __getitem__(self, key) -> "AdjacencyList":
        if isinstance(key, base.MaskBase):
            key = key.data
        return self.__class__(np.array(self._data[key]))

    def copy(self) -> "AdjacencyList":
        return deepcopy(self)

    @classmethod
    def from_matrix(
        cls,
        adj_matrix: ty.Union[
            base.DoubleVector, base.BoolVector, base.IntVector
        ],
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
    def matrix(self) -> ty.Union[base.DoubleVector, base.IntVector]:
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
    def edges(self) -> base.AnyVector:
        return self._data

    @property
    def nodes(self) -> base.IntVector:
        """Nodes are extracted from the edge list, and put in
        ascending order of magnitude, regardless of sign.
        Positive sign conventionally means final state particle.
        """
        # extract nodes from edge list
        unstruc_edges = rfn.structured_to_unstructured(self._data)
        unsort_nodes = np.unique(unstruc_edges)
        sort_idxs = np.argsort(np.abs(unsort_nodes))
        return unsort_nodes[sort_idxs]  # type: ignore

    @fn.cached_property
    def _sparse(self) -> coo_array:
        return self.to_sparse()

    def to_sparse(self, data: ty.Optional[base.AnyVector] = None) -> coo_array:
        """Converts the graph structure to a ``scipy.sparse.coo_array``
        instance.

        Parameters
        ----------
        data : ndarray, optional
            Data stored on each edge. If ``None``, these will be boolean
            values indicating whether the outgoing node is a leaf.
            Default is ``None``.

        Returns
        -------
        arr : scipy.sparse.coo_array
            COO-formatted sparse array, where rows are "in" and cols
            are "out" indices for ``AdjacencyList.edges``.
        """
        out = self._data["out"]
        size = np.max(out) + 1
        if data is None:
            data = np.sign(self._data["out"]) == 1
        return coo_array(
            (data, (np.abs(self._data["in"]), np.abs(self._data["out"]))),
            shape=(size, size),
        )

    def to_dicts(
        self,
        edge_data: ty.Optional[
            ty.Dict[str, ty.Union[base.ArrayBase, base.AnyVector]]
        ] = None,
        node_data: ty.Optional[
            ty.Dict[str, ty.Union[base.ArrayBase, base.AnyVector]]
        ] = None,
    ) -> _AdjDict:
        """Returns data in dictionary format, which is more easily
        parsed by external libraries, such as NetworkX.
        """
        if edge_data is None:
            edge_data = dict()
        if node_data is None:
            node_data = dict()

        def make_data_dicts(
            orig: ty.Tuple[ty.Any, ...],
            data: ty.Dict[str, ty.Union[base.ArrayBase, base.AnyVector]],
        ):
            def array_iterator(array_dict):
                for array in array_dict.values():
                    if isinstance(array, base.ArrayBase):
                        yield array.data
                    elif isinstance(array, np.ndarray):
                        yield array
                    else:
                        raise TypeError("Data structure not supported.")

            data_rows = zip(*array_iterator(data))
            dicts = (dict(zip(data.keys(), row)) for row in data_rows)
            combo = it.zip_longest(*orig, dicts, fillvalue=dict())
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

    :group: datastructure

    Parameters
    ----------
    particles : ParticleSet
        The point cloud data for the particles in the dataset.
    adj : AdjacencyList
        The connectivity of the particles in the point cloud, forming a
        graph.

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
    def __attr_names(self) -> ty.Tuple[str, ...]:
        return tuple(self.__annotations__.keys())

    def __getitem__(self, key) -> "Graphicle":
        kwargs = dict()
        for name in self.__attr_names:
            data = getattr(self, name)
            if len(data) != 0:
                kwargs.update({name: data[key]})
        return self.__class__(**kwargs)

    def __bool__(self) -> bool:
        for name in self.__attr_names:
            if getattr(self, name):
                return True
        return False

    def copy(self) -> "Graphicle":
        return deepcopy(self)

    @classmethod
    def from_event(cls, event: base.EventInterface) -> "Graphicle":
        """Instantiates a Graphicle object from a generic event object,
        whose attribute structure is compatible with
        ``base.EventInterface``.

        Parameters
        ----------
        event : base.EventInterface
            Object of any type with a subset of parameters with
            consistent names and values to those defined in the
            interface. ``heparchy`` and ``showerpipe`` event objects
            can be passed for easy instantiation.
        """
        params = dict()
        for attr in base.EventInterface.__dict__:
            if attr[0] == "_":
                continue
            try:
                params[attr] = getattr(event, attr)
            except AttributeError:
                if attr == "final" and hasattr(event, "masks"):
                    params["final"] = event.masks["final"]
        return cls.from_numpy(**params)

    @classmethod
    def from_numpy(
        cls,
        pdg: ty.Optional[base.IntVector] = None,
        pmu: ty.Optional[base.VoidVector] = None,
        color: ty.Optional[base.VoidVector] = None,
        helicity: ty.Optional[base.HalfIntVector] = None,
        status: ty.Optional[base.HalfIntVector] = None,
        final: ty.Optional[base.BoolVector] = None,
        edges: ty.Optional[base.VoidVector] = None,
        weights: ty.Optional[base.DoubleVector] = None,
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
    def final(self) -> base.MaskBase:
        return self.particles.final

    @property
    def edges(self) -> base.VoidVector:
        return self.adj.edges

    @property
    def nodes(self) -> base.IntVector:
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
