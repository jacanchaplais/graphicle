from abc import ABC, abstractmethod
from typing import Union, Any, Optional

import numpy as np
import numpy.typing as npt


AnyVector = npt.NDArray[Any]
BoolVector = npt.NDArray[np.bool_]


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


class GraphicleBase(ABC):
    @property
    @abstractmethod
    def edges(self) -> AnyVector:
        pass

    # @abstractmethod
    # def vertex_pdg(self):
    #     pass

    # @abstractmethod
    # def to_networkx(self):
    #     pass

    # @abstractmethod
    # def to_pandas(self):
    #     pass

    # @abstractmethod
    # def signal_mask(self):
    #     pass

    # @abstractmethod
    # def copy(self):
    #     pass


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
