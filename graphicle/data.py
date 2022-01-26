from itertools import zip_longest
from functools import partial
from copy import deepcopy
from typing import Tuple, List, Dict, Optional, Any

from attr import define, field, Factory, cmp_using, setters  # type: ignore
import numpy as np
import numpy.typing as npt
from numpy.lib import recfunctions as rfn
from typicle import Types
from typicle.convert import cast_array

from ._base import ParticleBase, AdjacencyBase, MaskBase, ArrayBase


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

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

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

    def copy(self):
        return deepcopy(self)

    @property
    def children(self) -> List[str]:
        return list(self._mask_arrays.keys())

    def add(self, key: str, mask: MaskArray) -> None:
        """Add a new MaskArray to the group, with given key."""
        self._mask_arrays.update({key: mask})

    def remove(self, key: str) -> None:
        """Remove a MaskArray from the group, using given key."""
        self._mask_arrays.pop(key)

    @property
    def data(self):
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
        return self.__class__(np.array(self.data[key]))

    def copy(self):
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

        Returns ------- mask : MaskArray
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

    def __get_prop(self, field: str) -> np.ndarray:
        props = self.__lookup_table.properties(self.data, [field])[field]
        return props  # type: ignore

    def __get_prop_range(self, field: str) -> np.ndarray:
        field_range = [field + "lower", field + "upper"]
        props = self.__lookup_table.properties(self.data, field_range)
        return props  # type: ignore

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
        return self.__class__(np.array(self.data[key]))

    def copy(self):
        return deepcopy(self)

    @property
    def _vector(self):
        from vector import MomentumNumpy4D

        dtype = deepcopy(self.data.dtype)
        dtype.names = ("x", "y", "z", "t")
        vec = self.data.view(dtype).view(MomentumNumpy4D)
        return vec

    @property
    def pt(self) -> np.ndarray:
        return self._vector.pt  # type: ignore

    @property
    def eta(self) -> np.ndarray:
        return self._vector.eta  # type: ignore

    @property
    def phi(self) -> np.ndarray:
        return self._vector.phi  # type: ignore

    def delta_R(self, other_pmu: "MomentumArray") -> np.ndarray:
        return self._vector.deltaR(other_pmu._vector)  # type: ignore


@define
class ColorArray(ArrayBase):
    data: np.ndarray = array_field("color")

    def copy(self):
        return deepcopy(self)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self.data[key]))

    def __len__(self):
        return len(self.data)


@define
class ParticleSet(ParticleBase):
    pdg: PdgArray = PdgArray()
    pmu: MomentumArray = MomentumArray()
    color: ColorArray = ColorArray()
    final: MaskArray = MaskArray()

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

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_numpy(
        cls,
        pdg: Optional[np.ndarray] = None,
        pmu: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        final: Optional[np.ndarray] = None,
    ):
        def optional(data_class, data: Optional[np.ndarray]):
            return data_class(data) if data is not None else data_class()

        return cls(
            pdg=optional(PdgArray, pdg),
            pmu=optional(MomentumArray, pmu),
            color=optional(ColorArray, color),
            final=optional(MaskArray, final),
        )


@define
class AdjacencyList(AdjacencyBase):
    _data: np.ndarray = array_field("edge")
    weights: np.ndarray = array_field("double")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        if isinstance(key, MaskBase):
            key = key.data
        return self.__class__(np.array(self._data[key]))

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_matrix(cls, adj_matrix: np.ndarray, weighted: bool = False):
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
        """
        edges = np.where(adj_matrix)
        kwargs = {"data": np.array(edges).T}
        if weighted is True:
            kwargs["weights"] = adj_matrix[edges]
        return cls(**kwargs)

    @property
    def edges(self):
        return self._data

    @property
    def nodes(self):
        """Nodes are extracted from the edge list, and put in
        ascending order of magnitude, regardless of sign.
        Positive sign conventionally means final state particle.
        """
        # extract nodes from edge list
        unstruc_edges = rfn.structured_to_unstructured(self._data)
        unsort_nodes = np.unique(unstruc_edges)
        sort_idxs = np.argsort(np.abs(unsort_nodes))
        return unsort_nodes[sort_idxs]

    def to_dicts(
        self,
        edge_data: Optional[Dict[str, ArrayBase]] = None,
        node_data: Optional[Dict[str, ArrayBase]] = None,
    ):
        if edge_data is None:
            edge_data = dict()
        if node_data is None:
            node_data = dict()

        def make_data_dicts(orig: Tuple[Any, ...], data: Dict[str, ArrayBase]):
            data_arrays = (array.data for array in data.values())
            data_rows = zip(*data_arrays)
            dicts = (dict(zip(data.keys(), row)) for row in data_rows)
            combo = zip_longest(*orig, dicts, fillvalue=dict())
            return tuple(combo)

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


@define
class Graphicle:
    particles: ParticleSet = ParticleSet()
    adj: AdjacencyList = AdjacencyList()

    @property
    def __attr_names(self):
        return tuple(self.__annotations__.keys())

    def __getitem__(self, key):
        kwargs = dict()
        for name in self.__attr_names:
            data = getattr(self, name)
            kwargs.update({name: data[key]})
        return self.__class__(**kwargs)

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_numpy(
        cls,
        pdg: Optional[np.ndarray] = None,
        pmu: Optional[np.ndarray] = None,
        color: Optional[np.ndarray] = None,
        final: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
    ):
        particles = ParticleSet.from_numpy(
            pdg=pdg, pmu=pmu, color=color, final=final
        )
        if edges is not None:
            adj_list = AdjacencyList(edges)
        else:
            adj_list = AdjacencyList()
        return cls(particles=particles, adj=adj_list)

    @property
    def pdg(self):
        return self.particles.pdg

    @property
    def pmu(self):
        return self.particles.pmu

    @property
    def color(self):
        return self.particles.color

    @property
    def final(self):
        return self.particles.final

    @property
    def edges(self):
        return self.adj.edges

    @property
    def nodes(self):
        return self.adj.nodes
