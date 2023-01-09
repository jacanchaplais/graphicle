"""
``graphicle.base``
==================

Defines the base classes, types, and interface protocols used by
graphicle's modules.
"""
from abc import ABC, abstractmethod
from typing import Union, Any, Optional, Protocol

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
    "EventInterface",
    "ArrayBase",
    "ParticleBase",
    "AdjacencyBase",
    "MaskBase",
    "MaskLike",
]

DoubleVector = npt.NDArray[np.float64]
ComplexVector = npt.NDArray[np.complex128]
BoolVector = npt.NDArray[np.bool_]
IntVector = npt.NDArray[np.int32]
HalfIntVector = npt.NDArray[np.int16]
ObjVector = npt.NDArray[np.object_]
AnyVector = npt.NDArray[Any]
MaskLike = Union["MaskBase", BoolVector]


class EventInterface(Protocol):
    """Defines the interface for a generic event object expected by
    graphicle's routines. Attributes are stored as numpy arrays, with
    each element corresponding to a particle in an event. Attributes
    with 'fields' are numpy structured arrays.

    :group: base

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
    edges : ndarray[int32], fields ("in", "out")
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


class ArrayBase(ABC):
    @abstractmethod
    def __init__(self, data: Optional[AnyVector] = None) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of elements in array."""

    @abstractmethod
    def __bool__(self) -> bool:
        """Truthy returns ``False`` if no elements, ``True`` otherwise."""

    @abstractmethod
    def __array__(self) -> npt.NDArray[Any]:
        """Numpy array representation of the data."""

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


class AdjacencyBase(ABC):
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
    def __array__(self) -> npt.NDArray[Any]:
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
