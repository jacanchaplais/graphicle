from functools import partial

from attr import define, field, Factory, cmp_using
import numpy as np
from typicle import Types

from ._base import ParticleBase, NetworkBase
from .converters import cast_array


def array_field(type_name):
    types = Types()
    dtype = getattr(types, type_name)
    default = Factory(lambda: np.array([], dtype=dtype))
    equality_comparison = cmp_using(np.array_equal)
    converter = partial(cast_array, cast_type=dtype)
    return field(default=default,
                 eq=equality_comparison,
                 converter=converter
                 )

@define
class ParticleCloud(ParticleBase):
    pdg: np.ndarray = array_field('int')
    pmu: np.ndarray = array_field('pmu')
    color: np.ndarray = array_field('color')
    final: np.ndarray = array_field('bool')

@define
class Network(NetworkBase):
    edges: np.ndarray = array_field('edge')

    @property
    def nodes(self):
        return np.unique(self.edges)


def main():
    print(ParticleCloud(pmu=np.arange(100).reshape((-1, 4))))
    print(Network(np.arange(100).reshape((-1, 2))))

if __name__ == '__main__':
    main()
