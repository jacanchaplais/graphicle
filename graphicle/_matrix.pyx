cimport cython
import numpy as np
from libc.math cimport hypot


DTYPE = np.double

cdef double delta_R(double eta_1, double eta_2, double phi_1, double phi_2):
    return hypot(eta_2 - eta_1, phi_2 - phi_1)

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

    # TODO: take advantage of fact matrix is symmetric
    for y in range(1, length):
        for x in range(y):
            result_view[y, x] = delta_R(eta[y], eta[x], phi[y], phi[x])

    result = result + result.T
    return result
