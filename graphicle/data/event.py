"""
graphicle.data.event
===================

This module defines the dataclasses and methods which construct the
DAG of the event generation.
"""

from copy import deepcopy
from functools import wraps

import attr
import numpy as np

import heparchy.utils as utils
from heparchy import TYPE, REAL_TYPE


def val_elems_int(instance, attribute, value):
    all_ints = set(map(type, value)) == {int} # True if elems all ints
    empty = False
    if not value:
        empty = True
    if not empty and not all_ints:
        raise TypeError(
                f'Inputs must be iterables of integers. Got value={value}')

def val_int_array(instance, attribute, value):
    pass

@attr.s(kw_only=True, frozen=True, slots=True)
class SignalVertex:
    """Data structure to define which of the vertices represent a
    signal in an event, and which of the given vertex's descendants
    are to be followed in the subsequent showering.

    Each of the inputs are to be formatted as an iterable of integers,
    representing the pdgs of the particles.

    Keyword arguments:
        incoming: particles incident on the vertex
        outgoing: particles outbound from the vertex
        follow: outbound particles marked to have their children tracked
    """

    from typing import Set as __Set

    __PdgSet = __Set[int]
    __pdg_kwargs = dict(converter=set, validator=[val_elems_int])
    incoming: __PdgSet = attr.ib(**__pdg_kwargs)
    outgoing: __PdgSet = attr.ib(**__pdg_kwargs)
    follow: __PdgSet = attr.ib(**__pdg_kwargs)

# @attr.s
# class ParticleSet:
#     pmu: np.ndarray = attr.ib(
#             **__array_kwargs)
#     pdg: np.ndarray = attr.ib(
#             **__array_kwargs)
#     __pmu_vector = self.__vector_view()

#     def __vector_view(self):
#         dtype = deepcopy(self.pmu.dtype)
#         dtype.names = ('x', 'y', 'z', 't')
#         pmu_vec = self.pmu.view(dtype).view(self.__vector.MomentumNumpy4D)
#         return pmu_vec

#     @property
#     def pt(self):
#         return self.__pmu_vector.pt

#     @property
#     def eta(self):
#         return self.__pmu_vector.eta

@attr.s(on_setattr=attr.setters.convert)
class ShowerData:
    import pandas as __pd
    import networkx as __nx
    import vector as __vector

    __array_kwargs = dict(
            eq=attr.cmp_using(eq=np.array_equal),
            )
    edges: np.ndarray = attr.ib(
            converter=utils.structure_edges,
            **__array_kwargs)
    pmu: np.ndarray = attr.ib(
            converter=utils.structure_pmu,
            **__array_kwargs)
    pdg: np.ndarray = attr.ib(
            **__array_kwargs)
    final: np.ndarray = attr.ib(
            **__array_kwargs)

    @classmethod
    def empty(cls):
        return cls(
            edges = np.array([[0, 1]], dtype=TYPE['int']),
            pmu=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=REAL_TYPE),
            pdg=np.array([1], dtype=TYPE['int']),
            final=np.array([False], dtype=TYPE['bool'])
            )

    @property
    def __pmu_vector(self):
        dtype = deepcopy(self.pmu.dtype)
        dtype.names = ('x', 'y', 'z', 't')
        pmu_vec = self.pmu.view(dtype).view(self.__vector.MomentumNumpy4D)
        return pmu_vec

    def pt(self, mask=None):
        pmu_subset = self.__pmu_vector[mask].squeeze()
        return pmu_subset.pt

    def eta(self, mask=None):
        pmu_subset = self.__pmu_vector[mask].squeeze()
        return pmu_subset.eta

    def phi(self, mask=None):
        pmu_subset = self.__pmu_vector[mask].squeeze()
        return pmu_subset.phi

    def flush_cache(self):
        """Clears cached private attributes of ShowerData instance.
        These must currently be cleared manually after updating
        public attributes of an instance.

        Notes
        -----
        If looping through many events, performance is vastly improved
        if only one instance of the ShowerData class is created,
        and the attributes of that instance are iteratively updated.
        In this case, it is necessary to use `flush_cache` during each
        iteration, so the next event does not have cached data from the
        previous one.

        In a future release, this will be handled automatically whenever
        an attribute value is updated.
        """
        try:
            self.__shower
        except AttributeError:
            pass
        else:
            del self.__shower

    def to_networkx(self, data=['pdg']):
        """Output directed acyclic graph representation of the shower,
        implemented by NetworkX. Each edge is a particle, and each node
        is an interaction vertex, except for the terminating leaf nodes.

        Parameters
        ----------
        data : iterable of strings
            Specify which of the particle properties should be embedded
            on the edges.
            Valid options are:
                - pdg [default]
                - pmu
                - final
        """
        # form edges with pdg data on for easier ancestry tracking
        names = data
        data_rows = (getattr(self, name) for name in names)
        data_rows = zip(*data_rows)
        edge_dicts = (dict(zip(names, row)) for row in data_rows)
        edges = zip(self.edges['in'], self.edges['out'], edge_dicts)
        shower = self.__nx.DiGraph()
        shower.add_edges_from(edges)
        return shower

    def __requires_shower(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            try:
                self.__shower
            except AttributeError:
                self.__shower = self.to_networkx()
            return func(self, *args, **kwargs)
        return inner

    def to_pandas(self, data=('pdg', 'final')):
        """Return event data in a single pandas dataframe.

        Parameters
        ----------
        data : iterable of strings or True
            The particle properties to include as columns.
            Valid options are:
                - pdg [default]
                - final [default]
                - pmu
                - x, y, z, or e
                - edges
                - pt
                - eta
                - phi
            If set to True instead of iterable, all data are included.

        Notes
        -----
        This is particularly useful if you want to mask the data
        in complex ways involving multiple particle properties at once,
        using the dataframe's `query` method.
        """
        # collect column data into a dict, ready to pass to pandas
        cols = {
            'pdg': self.pdg,
            'final': self.final,
            'edge_in': self.edges['in'],
            'edge_out': self.edges['out'],
            'pt': self.pt(),
            'eta': self.eta(),
            'phi': self.phi(),
            }
        pmu_cols = list(self.pmu.dtype.names)
        cols.update({col_name: self.pmu[col_name] for col_name in pmu_cols})
        # check which data to include
        if data == True: # all of it
            return self.__pd.DataFrame(cols)
        else: # restricted selection
            try:
                iterator = iter(data)
            except TypeError:
                print("data must either be an iterable, or True")
            else:
                # define valid input
                col_names_edge = ['edge_in', 'edge_out']
                valid_col_names = list(cols.keys())
                valid_col_names = set(valid_col_names + col_names_edge
                                      + ['pmu', 'edges'])
                # validate user input
                user_col_names = set(data)
                invalid_col_names = user_col_names.difference(valid_col_names)
                # discard invalid input
                col_names = user_col_names.intersection(valid_col_names)
                col_names = list(col_names)
                if invalid_col_names: # let user know of invalid input
                    print(f'Warning: {len(invalid_col_names)} invalid column '
                          + f'name(s) provided, {invalid_col_names}. '
                          + 'Omitting.')
                if 'edges' in col_names: # convert abbreviations
                    col_names.remove('edges')
                    col_names += col_names_edge
                if 'pmu' in col_names:
                    col_names.remove('pmu')
                    col_names += pmu_cols
                return self.__pd.DataFrame( # populate dataframe with selection
                        {key: cols[key] for key in col_names})

    @__requires_shower
    def vertex_pdg(self):
        """Returns a generator object which loops over all interaction
        vertices in the event, yielding the vertex id, and pdg codes
        of incoming and outgoing particles to the vertex, respectively.
        """
        shower = self.__shower
        for vertex in shower:
            incoming = shower.in_edges(vertex, data='pdg')
            outgoing = shower.out_edges(vertex, data='pdg')
            if incoming and outgoing:
                vtxs_in, _, pdgs_in = zip(*incoming)
                _, vtxs_out, pdgs_out = zip(*outgoing)
                yield vertex, set(pdgs_in), set(pdgs_out)
    
    @__requires_shower
    def signal_mask(self, signal_vertices):
        """Locates given vertices, based on their incoming and outgoing
        particles.

        Parameters
        ----------
        signal_vertices : iterable of `heparchy.SignalVertex` objects
            Specify multiple vertices to locate, and which of their
            descendants to follow.
            See `heparchy.SignalVertex` for information.

        Returns
        -------
        signal_masks : list of dicts containing 1d boolean numpy arrays
            Masks over all particles in the event, identifying which
            descend from the followed particle, produced at the
            chosen vertex.
            The output for each vertex is returned, in the same order as
            provided, as an element in a list.
            These elements are dictionaries, whose keys refer to the pdg
            codes of the followed particles, and whose values are the
            boolean arrays which identify their descendants.
        """
        # --- LOCATE SIGNAL VERTICES IN SHOWER --- #
        is_signal = []
        # mark where the signal vertices are in the shower:
        for vertex, pdgs_in, pdgs_out in self.vertex_pdg():
            is_signal.append(tuple(
                [vertex] + [sig_vtx.incoming.issubset(pdgs_in)
                            and sig_vtx.outgoing.issubset(pdgs_out)
                            for sig_vtx in signal_vertices]
                ))
        # split into two iterables of vertex ids and booleans, respectively
        is_signal = list(zip(*is_signal))
        vertices = is_signal[0]
        signal_lists = is_signal[1:]
        # --- POPULATE A LIST OF DICTIONARIES WITH THE SIGNAL MASKS --- #
        signal_masks = []
        # iterate through id and corresponding SignalVertex object together
        for signal_list, signal_vertex in zip(signal_lists, signal_vertices):
            follow_masks = dict() # to hold the mask, keyed by pdg
            if True in signal_list:
                # find the vertex id of the signal vertices:
                signal_id = vertices[signal_list.index(True)]
                # identify which vertex products need to be followed:
                edges_out_info = self.__shower.out_edges(signal_id, data='pdg')
                follow_edges = filter(
                        lambda edge: edge[-1] in signal_vertex.follow,
                        edges_out_info
                        )
                # store incident vtx of followed pcl as follow_id
                _, follow_ids, follow_pdgs = zip(*follow_edges)
                # find the descendants of follow_id vtxs for each followed pcl
                for follow_id, follow_pdg in zip(follow_ids, follow_pdgs):
                    desc_vtxs = self.__nx.descendants(self.__shower, follow_id)
                    # convert into structured np array for set comparison
                    desc_edges = list(self.__shower.edges(nbunch=desc_vtxs))
                    desc_edges = utils.structure_edges(np.array(desc_edges))
                    # create mask identifying edges / pcls desc from signal
                    mask = np.isin(self.edges, desc_edges, assume_unique=False)
                    follow_masks.update({follow_pdg: mask}) # store to dict
            else: # if vertex not located, fill with all False masks
                mask = np.full_like(self.final, fill_value=False)
                follow_masks.update(
                    {follow_pdg: mask for follow_pdg in signal_vertex.follow})
            signal_masks += [follow_masks]
        return signal_masks

    def copy(self):
        """Returns a copy of the ShowerData instance as a new object
        in memory.
        """
        return deepcopy(self)

