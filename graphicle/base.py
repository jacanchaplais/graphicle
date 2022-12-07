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
]

DoubleVector = npt.NDArray[np.float64]
ComplexVector = npt.NDArray[np.complex128]
BoolVector = npt.NDArray[np.bool_]
IntVector = npt.NDArray[np.int32]
HalfIntVector = npt.NDArray[np.int16]
ObjVector = npt.NDArray[np.object_]
AnyVector = npt.NDArray[Any]


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

    @property
    @abstractmethod
    def data(self) -> AnyVector:
        pass


class ParticleBase(ABC):
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
    @property
    @abstractmethod
    def edges(self) -> AnyVector:
        pass

    @property
    @abstractmethod
    def nodes(self) -> npt.NDArray[np.int32]:
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
    def __getitem__(self, key) -> "MaskBase":
        pass

    @abstractmethod
    def __and__(self, other: Union["MaskBase", BoolVector]) -> "MaskBase":
        pass

    @abstractmethod
    def __or__(self, other: Union["MaskBase", BoolVector]) -> "MaskBase":
        pass
