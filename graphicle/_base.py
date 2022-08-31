from abc import ABC, abstractmethod

import numpy as np


class ArrayBase(ABC):
    @property
    @abstractmethod
    def data(self) -> np.ndarray:
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
    def edges(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def nodes(self) -> np.ndarray:
        pass


class GraphicleBase(ABC):
    @property
    @abstractmethod
    def edges(self) -> np.ndarray:
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
    def data(self) -> np.ndarray:
        pass

    @abstractmethod
    def __getitem__(self, key) -> "MaskBase":
        pass

    # @abstractmethod
    # def __and__(self, other) -> "MaskBase":
    #     pass

    # @abstractmethod
    # def __or__(self, other) -> "MaskBase":
    #     pass
