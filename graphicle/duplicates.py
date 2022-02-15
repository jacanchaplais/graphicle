import numpy as np
import numpy.lib.recfunctions as rfn

from typicle import Types

from typing import TYPE_CHECKING
import graphicle as gcl


_types = Types()


def duplicate_mask(adj: gcl.AdjacencyList) -> gcl.MaskArray:
    """

    Notes
    -----
    Duplicate particles in full event graphs are identified easily.
    Two particles are duplicates if they enter and leave the same
    interaction vertex, one-edge-in and one-edge-out.
    This is because they physically are unchanged unless they radiate
    or decay.

    This function simply identifies vertices in particle-as-edge
    generation graphs with in-degree and out-degree of one.

    To do this, a set is constructed of the vertex ids which only turn
    up once in the `in` and `out` fields of the edge list.
    The intersection of the `in` and `out` sets is found, leaving only
    vertices with both and in-degree and out-degree of one.
    Finally, a mask is constructed highlighting the edges belonging to
    those vertices.
    """
    edges = adj.edges
    mask = np.zeros_like(edges, dtype=_types.bool)

    def unique_array(direction: str):
        """Returns a structured array of the unique vertices in the
        edge list's given direction.
        Fields are vertex id, array index, and frequency count.
        """
        unique_dtype = [("vtx", "<i4"), ("index", "<i4"), ("count", "<i4")]
        unique_dtype = np.dtype(unique_dtype)  # type: ignore
        return rfn.unstructured_to_structured(
            np.dstack(  # join the output arrays of unique together
                np.unique(
                    edges[direction], return_index=True, return_counts=True
                )
            ).squeeze(),  # remove dimension nesting
            unique_dtype,
        )

    # get indices and counts for each vertex id
    in_unique = unique_array("in")
    out_unique = unique_array("out")

    # select where in and out degree of vertices is 1
    def singles(unique):
        """Returns a structured unique array, masked such that only
        elements with a count of 1 remain.
        """
        mask = unique["count"] == 1
        return unique[mask]

    in_singles = singles(in_unique)
    out_singles = singles(out_unique)
    # find indices for which vertices have both in and out degrees of 1
    dup_vtxs, in_dup_idxs, out_dup_idxs = np.intersect1d(
        in_singles["vtx"], out_singles["vtx"], return_indices=True
    )
    dups = np.union1d(in_singles[in_dup_idxs], out_singles[out_dup_idxs])
    # mark these vertices as duplicates
    mask[dups["index"]] = True
    return gcl.MaskArray(mask)
