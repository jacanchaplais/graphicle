"""
``graphicle.base``
==================

Defines the base classes, types, and interface protocols used by
graphicle's modules.
"""
import collections.abc as cla
import typing as ty
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

__all__ = [
    "DoubleVector",
    "ComplexVector",
    "BoolVector",
    "IntVector",
    "HalfIntVector",
    "ObjVector",
    "AnyVector",
    "VoidVector",
    "DoubleUfunc",
    "EventInterface",
    "ArrayBase",
    "ParticleBase",
    "AdjacencyBase",
    "MaskBase",
    "MaskLike",
    "NumericalStabilityWarning",
]

DoubleVector = npt.NDArray[np.float64]
FloatVector = npt.NDArray[np.float32]
ComplexVector = npt.NDArray[np.complex128]
BoolVector = npt.NDArray[np.bool_]
IntVector = npt.NDArray[np.int32]
HalfIntVector = npt.NDArray[np.int16]
ObjVector = npt.NDArray[np.object_]
AnyVector = npt.NDArray[ty.Any]
VoidVector = npt.NDArray[np.void]
MaskLike = ty.Union["MaskBase", BoolVector]
DoubleUfunc = ty.TypeVar("DoubleUfunc", DoubleVector, np.float64)
DataType = ty.TypeVar("DataType")


class EventInterface(ty.Protocol):
    """Defines the interface for a generic event object expected by
    graphicle's routines. Attributes are stored as numpy arrays, with
    each element corresponding to a particle in an event. Attributes
    with 'fields' are numpy structured arrays.

    :group: base

    .. versionadded:: 0.1.7

    Attributes
    ----------
    pdg : ndarray[int32]
        PDG codes.
    pmu : ndarray[float64], fields ("x", "y", "z", "e")
        Four-momenta.
    color : ndarray[int32], fields ("color", "anticolor")
        QCD color codes.
    helicity : ndarray[int16]
        Spin polarisation eigenvalues.
    status : ndarray[int16]
        Status codes annotating particles describing their generation.
    final : ndarray[bool_]
        Mask identifying final particles in the event's ancestry.
    edges : ndarray[int32], fields ("src", "dst")
        Ancestry of particles in event, encoded as a COO list of
        integers, describing a graph structure.
    """

    @property
    def pdg(self) -> IntVector:
        ...

    @property
    def pmu(self) -> AnyVector:
        ...

    @property
    def color(self) -> AnyVector:
        ...

    @property
    def helicity(self) -> HalfIntVector:
        ...

    @property
    def status(self) -> HalfIntVector:
        ...

    @property
    def final(self) -> BoolVector:
        ...

    @property
    def edges(self) -> AnyVector:
        ...


class ArrayBase(ABC, cla.Sequence, np.lib.mixins.NDArrayOperatorsMixin):
    @abstractmethod
    def __init__(self, data: ty.Optional[AnyVector] = None) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> ty.Iterator[ty.Any]:
        """Iterator exposing contained data as Python native."""

    @abstractmethod
    def __bool__(self) -> bool:
        """Truthy returns ``False`` if no elements, ``True`` otherwise."""

    @abstractmethod
    def __array__(self) -> AnyVector:
        """Numpy array representation of the data."""

    @property
    @abstractmethod
    def dtype(self) -> AnyVector:
        """Numpy array representation of the data."""

    @abstractmethod
    def __eq__(self) -> "MaskBase":
        """Equality comparison."""

    @abstractmethod
    def __ne__(self) -> "MaskBase":
        """Non equality comparison."""

    @property
    @abstractmethod
    def _data(self) -> AnyVector:
        pass

    @property
    @abstractmethod
    def data(self) -> AnyVector:
        pass


class ParticleBase(ABC):
    @abstractmethod
    def __getitem__(self, key) -> "ParticleBase":
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass

    @abstractmethod
    def copy(self) -> "ParticleBase":
        pass

    @property
    @abstractmethod
    def pdg(self) -> ArrayBase:
        pass

    @property
    @abstractmethod
    def pmu(self) -> ArrayBase:
        pass

    @property
    @abstractmethod
    def color(self) -> ArrayBase:
        pass

    @property
    @abstractmethod
    def final(self) -> ArrayBase:
        pass


class AdjacencyBase(ABC, cla.Sequence, np.lib.mixins.NDArrayOperatorsMixin):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __array__(self) -> AnyVector:
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass

    @abstractmethod
    def __getitem__(self, key) -> "AdjacencyBase":
        pass

    @property
    @abstractmethod
    def edges(self) -> AnyVector:
        pass

    @property
    @abstractmethod
    def nodes(self) -> IntVector:
        pass


# ---------------------------
# composite pattern for masks
# ---------------------------
class MaskBase(ABC):
    @property
    @abstractmethod
    def data(self) -> BoolVector:
        pass

    @abstractmethod
    def copy(self) -> "MaskBase":
        pass

    @abstractmethod
    def __array__(self) -> npt.NDArray[ty.Any]:
        """Numpy array representation of the data."""

    @abstractmethod
    def __getitem__(self, key) -> "MaskBase":
        pass

    @abstractmethod
    def __and__(self, other: MaskLike) -> "MaskBase":
        pass

    @abstractmethod
    def __or__(self, other: MaskLike) -> "MaskBase":
        pass

    @abstractmethod
    def __eq__(self, other: MaskLike) -> "MaskBase":
        pass

    @abstractmethod
    def __invert__(self) -> "MaskBase":
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass


class NumericalStabilityWarning(UserWarning):
    """Raised when the result of a calculation may not be numerically
    stable.

    :group: errors_warnings

    .. versionadded:: 0.3.1
    """
