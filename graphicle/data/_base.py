from abc import ABC, abstractmethod


class ParticleBase(ABC):
    @property
    @abstractmethod
    def pdg(self):
        pass

    @property
    @abstractmethod
    def pmu(self):
        pass

    @property
    @abstractmethod
    def color(self):
        pass

    @property
    @abstractmethod
    def final(self):
        pass

class NetworkBase(ABC):
    @property
    @abstractmethod
    def nodes(self):
        pass

    @property
    @abstractmethod
    def edges(self):
        pass

class ParticleNetworkBase(ABC):
    @property
    @abstractmethod
    def edges(self):
        pass

    @property
    @abstractmethod
    def pmu(self):
        pass

    @property
    @abstractmethod
    def color(self):
        pass

    @property
    @abstractmethod
    def pdg(self):
        pass

    @property
    @abstractmethod
    def final(self):
        pass

    @abstractmethod
    def vertex_pdg(self):
        pass

    @abstractmethod
    def to_networkx(self):
        pass

    @abstractmethod
    def to_pandas(self):
        pass

    @abstractmethod
    def signal_mask(self):
        pass

    @abstractmethod
    def copy(self):
        pass
