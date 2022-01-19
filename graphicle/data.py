from functools import partial
from copy import deepcopy
from typing import List, Dict

from attr import define, field, Factory, cmp_using, setters, fields
import numpy as np
from typicle import Types
from typicle.convert import cast_array
from mcpid import lookup

from ._base import ParticleBase, EdgeBase, MaskBase, ArrayBase, GraphicleBase


_types = Types()


def array_field(type_name):
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


@define
class MaskArray(MaskBase, ArrayBase):
    data: np.ndarray = array_field("bool")

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(self.data[key])

    def __len__(self):
        return len(self.data)


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

    _mask_arrays: Dict[str, MaskArray] = field(repr=False, factory=dict)

    def __repr__(self):
        keys = ", ".join(self.children)
        return f"MaskGroup(mask_arrays=[{keys}])"

    @property
    def children(self) -> list:
        return list(self._mask_arrays.keys())

    def add(self, key: str, mask: MaskArray) -> None:
        """Add a new MaskArray to the group, with given key."""
        self._mask_arrays.update({key: mask})

    def remove(self, key: str) -> None:
        """Remove a MaskArray from the group, using given key."""
        self._mask_arrays.pop(key)

    @property
    def data(self) -> np.ndarray:
        return np.bitwise_and.reduce(
            [child.data for child in self._mask_arrays.values()]
        )


@define
class PdgArray(ArrayBase):
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
        return self.__class__(self.data[key])

    def mask(
        self,
        target: list,
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
        if sign_sensitive == False:
            data = np.abs(data, dtype=_types.int)
        return MaskArray(
            np.isin(data, target, assume_unique=False, invert=blacklist)
        )

    def __get_prop(self, field: str):
        return self.__lookup_table.properties(self.data, [field])[field]

    def __get_prop_range(self, field: str):
        fields = [field + "lower", field + "upper"]
        return self.__lookup_table.properties(self.data, fields)

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


@define
class MomentumArray(ArrayBase):
    data: np.ndarray = array_field("pmu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(self.data[key])

    @property
    def __vector(self):
        from vector import MomentumNumpy4D

        dtype = deepcopy(self.data.dtype)
        dtype.names = ("x", "y", "z", "t")
        vec = self.data.view(dtype).view(MomentumNumpy4D)
        return vec

    @property
    def pt(self) -> np.ndarray:
        return self.__vector.pt

    @property
    def eta(self) -> np.ndarray:
        return self.__vector.eta

    @property
    def phi(self) -> np.ndarray:
        return self.__vector.phi


@define
class ColorArray(ArrayBase):
    data: np.ndarray = array_field("color")

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(self.data[key])

    def __len__(self):
        return len(self.data)


@define
class ParticleSet(ParticleBase):
    pdg: PdgArray = PdgArray()
    pmu: MomentumArray = MomentumArray()
    color: ColorArray = ColorArray()
    final: MaskArray = MaskArray()

    @classmethod
    def from_numpy(
        cls,
        pdg: np.ndarray = None,
        pmu: np.ndarray = None,
        color: np.ndarray = None,
        final: np.ndarray = None,
    ):
        def optional(data_class, data: np.ndarray):
            return data_class(data) if data is not None else data_class()

        return cls(
            pdg=optional(PdgArray, pdg),
            pmu=optional(MomentumArray, pmu),
            color=optional(ColorArray, color),
            final=optional(MaskArray, final),
        )

    @property
    def __attr_names(self):
        return tuple(self.__annotations__.keys())

    def __getitem__(self, key):
        kwargs = dict()
        for name in self.__attr_names:
            data = getattr(self, name)
            if len(data) > 0:
                kwargs.update({name: data[key]})
        return self.__class__(**kwargs)

    def __repr__(self):
        attr_repr = (repr(getattr(self, name)) for name in self.__attr_names)
        attr_str = ",\n".join(attr_repr)
        return f"ParticleSet(\n{attr_str}\n)"


@define
class EdgeList(EdgeBase):
    import networkx as __nx

    data: np.ndarray = array_field("edge")

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(self.data[key])

    @property
    def nodes(self):
        return np.unique(self.edges)

    def to_networkx(self, data_dict: Dict[str, ArrayBase] = None):
        """Output directed acyclic graph representation of the shower,
        implemented by NetworkX. Each edge is a particle, and each node
        is an interaction vertex, except for the terminating leaf nodes.

        Parameters
        ----------
        data : dict[str] -> Graphicle.ArrayBase
            Dict of Particle data arrays which subclass ArrayBase, eg.
            PdgArray, PmuArray, MaskArray, etc.

        Returns
        -------
        shower : networkx.DiGraph
            Directed NetworkX graph, with embedded edge data.
        """
        if data_dict is None:
            data_dict = dict()
        # -------------------------------------------------
        # form edges with data for easier ancestry tracking
        # -------------------------------------------------
        data_rows = (array.data for array in data_dict.values())
        # join elements from each array into rows
        data_rows = zip(*data_rows)
        # generator of dicts, with key/val pairs for each row elem
        edge_dicts = (dict(zip(data_dict.keys(), row)) for row in data_rows)
        # attach data dict to list
        edges = zip(self.data["in"], self.data["out"], edge_dicts)
        shower = self.__nx.DiGraph()
        shower.add_edges_from(edges)
        return shower


@define
class Graphicle:
    particles: ParticleSet = ParticleSet()
    edges: EdgeList = EdgeList()

    @classmethod
    def from_numpy(
        cls,
        pdg: np.ndarray = None,
        pmu: np.ndarray = None,
        color: np.ndarray = None,
        final: np.ndarray = None,
        edges: np.ndarray = None,
    ):
        particles = ParticleSet.from_numpy(
            pdg=pdg, pmu=pmu, color=color, final=final
        )
        edges = EdgeList(edges) if edges is not None else EdgeList()
        return cls(particles=particles, edges=edges)

    @property
    def pdg(self):
        return self.particles.pdg

    @property
    def pmu(self):
        return self.particles.pmu

    @property
    def color(self):
        return self.particles.pmu

    @property
    def final(self):
        return self.particles.final

    @property
    def __attr_names(self):
        return tuple(self.__annotations__.keys())

    def __getitem__(self, key):
        kwargs = dict()
        for name in self.__attr_names:
            data = getattr(self, name)
            kwargs.update({name: data[key]})
        return self.__class__(**kwargs)
