"""
``graphicle.transform``
=======================

Utilities for manipulating the graph structure of particle data.

"""
import cmath
import operator as op
import typing as ty

import numpy as np
import numpy.typing as npt

import graphicle as gcl

_complex_unpack = op.attrgetter("real", "imag")


def _cos_sin(angle: float) -> ty.Tuple[float, float]:
    """Returns a tuple containing the sine and cosine of an angle."""
    return _complex_unpack(cmath.rect(1.0, angle))


class SphericalAngle(ty.NamedTuple):
    """Pair of inclination and azimuthal angles, respectively."""

    theta: float
    phi: float


def soft_hard_axis(momenta: gcl.MomentumArray) -> SphericalAngle:
    """Calculates the axis defined by the plane swept out between the
    hardest and softest particles in ``momenta``.

    Parameters
    ----------
    momenta : MomentumArray
        Point cloud of particle four-momenta.

    Returns
    -------
    SphericalAngle
        Normal axis to the soft-hard plane, defined in terms of
        inclination and azimuthal angles.
    """
    data = momenta.data.view(np.float64).reshape(-1, 4)
    softest = data[momenta.pt.argmin()]
    hardest = data[momenta.pt.argmax()]
    axis = np.cross(softest[:3], hardest[:3])
    phi_polar = axis[:2].view(np.complex128).item()
    pt, phi = cmath.polar(phi_polar)
    theta_polar = complex(pt, axis[2].item())
    theta = cmath.phase(theta_polar)
    return SphericalAngle(theta=theta, phi=phi)


def rotation_matrix(
    angle: float, axis: SphericalAngle
) -> npt.NDArray[np.float64]:
    """Computes the matrix operator to rotate a 3D vector with respect
    to an arbitrary ``axis`` by a given ``angle``.

    Parameters
    ----------
    angle : float
        Desired angular displacement after matrix multiplication.
    axis : SphericalAngle
        Inclination and azimuthal angles, defining the axis about which
        the rotation is to be performed.

    Returns
    -------
    ndarray[float64]
        A 3x3 matrix which, when acting to the left of a 3D vector, will
        rotate it about the provided ``axis`` by the given ``angle``.

    Notes
    -----
    This is a matrix implementation of Rodrigues' rotation formula [1]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    cos_theta, sin_theta = _cos_sin(axis.theta)
    cos_phi, sin_phi = _cos_sin(axis.phi)
    u_x, u_y, u_z = (sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)
    skew_sym = np.zeros((3, 3), dtype=np.float64)
    upper_idxs = np.triu_indices(3, 1)
    skew_sym[upper_idxs] = -u_z, u_y, -u_x
    skew_sym[np.tril_indices(3, -1)] = -skew_sym[upper_idxs]
    cos_alpha, sin_alpha = _cos_sin(angle)
    rot = sin_alpha * skew_sym + (1.0 - cos_alpha) * (skew_sym @ skew_sym)
    np.fill_diagonal(rot, rot.diagonal() + 1.0)
    return rot


def split_hardest(
    momenta: gcl.MomentumArray, z: float, angle: float
) -> gcl.MomentumArray:
    """Splits the momentum of the hardest particle into two momenta.
    Energy and 3-momentum is conserved over the whole MomentumArray.
    Hardness and collinearity of the split are determined by function
    parameters.

    Parameters
    ----------
    momenta : MomentumArray
        Set of four-momenta, representing the point cloud of particles,
        prior to splitting.
    z : float
        Energy fraction retained by the first child after the split.
        Must be in range ``0.0 < z <= 0.5``.
    angle : float
        Angular displacement of the first child after the split. Will be
        rotated in the plane swept out by the hardest and softest
        particles in the particle set.

    Returns
    -------
    MomentumArray
        Set of four-momenta after splitting, such that the length is
        increased by one. The highest transverse momentum element has
        been removed from the set, and replaced with two momenta
        elements. The first and second children of the split are the
        penultimate and final elements of the MomentumArray,
        respectively.

    Notes
    -----
    This function is intended to check the IRC safety of our GNN jet
    clustering algorithms. It is implemented from the description given
    in a jet tagging paper [1]_, which defined the IRC safe message
    passing procedure used in this work.

    References
    ----------
    .. [1] https://doi.org/10.48550/arXiv.2109.14636
    """
    if not (0.0 < z <= 0.5):
        raise ValueError("z must be in range (0, 0.5]")
    data = momenta.data.view(np.float64).reshape(-1, 4)
    hard_idx = momenta.pt.argmax()
    out = np.empty((len(momenta) + 1, 4), dtype=np.float64)
    out[:hard_idx] = data[:hard_idx]
    out[hard_idx:-2] = data[(hard_idx + 1) :]
    parent = data[hard_idx]
    child_1 = z * parent
    child_1[:3] @= rotation_matrix(angle, soft_hard_axis(momenta)).T
    child_2 = parent - child_1
    out[-2] = child_1[...]
    out[-1] = child_2[...]
    return gcl.MomentumArray(out)
