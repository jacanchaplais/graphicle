cimport cython
import numpy as np
from libc.math cimport hypot


DTYPE = np.double

@cython.boundscheck(False)
@cython.wraparound(False)
def delta_R_aff(double[:] eta, double[:] phi):
    """Returns a 2D array of Euclidean distances between particles in
    eta-phi space.

    Parameters
    ----------
    eta : ndarray
        1d array of pseudorapidity values for each particle.
    phi : ndarray
        1d array of azimuthal angles for each particle.
    """
    cdef Py_ssize_t length = eta.shape[0]
    assert tuple(eta.shape) == tuple(phi.shape)

    result = np.zeros((length, length), dtype=DTYPE)
    cdef double[:, ::1] result_view = result

    cdef Py_ssize_t x, y
    for y in range(1, length):
        for x in range(y):
            result_view[y, x] = hypot(eta[y] - eta[x], phi[y] - phi[x])
            result_view[x, y] = result_view[y, x]
    return result
