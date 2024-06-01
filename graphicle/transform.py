"""
``graphicle.transform``
=======================

Utilities for mutating particle data.
"""
import cmath
import operator as op
import typing as ty

import numpy as np
import numpy.typing as npt

import graphicle as gcl

_complex_unpack = op.attrgetter("real", "imag")


class SphericalAngle(ty.NamedTuple):
    """Pair of inclination and azimuthal angles, respectively.

    :group: transform

    .. versionadded:: 0.4.0
    """

    theta: float
    phi: float


class SphericalAxis(ty.NamedTuple):
    """Axis vector in 3D cartesian coordinates.

    :group: transform

    .. versionadded:: 0.4.0
    """

    x: float
    y: float
    z: float


def _angles_to_axis(angles: SphericalAngle) -> SphericalAxis:
    axis = gcl.calculate._angles_to_axis(np.array([angles.phi, angles.theta]))
    return SphericalAxis(*axis.tolist())


def _axis_to_angles(axis: SphericalAxis) -> SphericalAngle:
    phi_polar = complex(axis.x, axis.y)
    pt, phi = cmath.polar(phi_polar)
    theta_polar = complex(pt, axis.z)
    theta = cmath.phase(theta_polar)
    return SphericalAngle(theta=theta, phi=phi)


def _momentum_to_numpy(momenta: gcl.MomentumArray) -> gcl.base.DoubleVector:
    return momenta.data.view(np.float64).reshape(-1, 4)


def _cos_sin(angle: float) -> ty.Tuple[float, float]:
    """Returns a tuple containing the sine and cosine of an angle."""
    return _complex_unpack(cmath.rect(1.0, angle))


def soft_hard_axis(momenta: gcl.MomentumArray) -> SphericalAngle:
    """Calculates the axis defined by the plane swept out between the
    hardest and softest particles in ``momenta``.

    :group: transform

    .. versionadded:: 0.4.0

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
    return _axis_to_angles(SphericalAxis(*axis.tolist()))


def rotation_matrix(
    angle: float, axis: SphericalAngle
) -> npt.NDArray[np.float64]:
    """Computes the matrix operator to rotate a 3D vector with respect
    to an arbitrary ``axis`` by a given ``angle``.

    :group: transform

    .. versionadded:: 0.4.0

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
    axis_3d = _angles_to_axis(axis)
    skew_sym = np.zeros((3, 3), dtype=np.float64)
    upper_idxs = np.triu_indices(3, 1)
    skew_sym[upper_idxs] = -axis_3d.z, axis_3d.y, -axis_3d.x
    skew_sym[np.tril_indices(3, -1)] = -skew_sym[upper_idxs]
    cos_alpha, sin_alpha = _cos_sin(angle)
    rot = sin_alpha * skew_sym + (1.0 - cos_alpha) * (skew_sym @ skew_sym)
    np.fill_diagonal(rot, rot.diagonal() + 1.0)
    return rot


def split_momentum(
    momentum: gcl.MomentumArray,
    z: float,
    angle: float,
    axis: ty.Union[ty.Tuple[float, float], SphericalAngle],
) -> gcl.MomentumArray:
    """Splits the momentum of the given particle into two momenta.
    Energy and 3-momentum is conserved. Hardness and collinearity of the
    split are determined by ``z`` and ``angle``.

    :group: transform

    .. versionadded:: 0.4.0

    Parameters
    ----------
    momentum : MomentumArray
        Four-momentum prior to splitting. Must contain only one element.
    z : float
        Energy fraction retained by the first child after the split.
        Must be in range ``0.0 < z <= 0.5``.
    angle : float
        Angular displacement of the first child after the split.
    axis : SphericalAngle or tuple[float, float], optional
        The theta and phi values of the axis vector about which to
        rotate the momentum. If ``None``, will choose the axis normal to
        the plane swept out by the hardest and softest momentum
        constituents.

    Returns
    -------
    MomentumArray
        Four-momenta of two particles produced by splitting.

    See Also
    --------
    soft_hard_axis : Axis of plane swept by softest and hardest momenta.
    rotation_matrix : Matrix to rotate 3-vectors about a given axis.
    """
    if len(momentum) > 1:
        raise ValueError("momentum must have only one element.")
    if not isinstance(axis, SphericalAngle):
        axis = SphericalAngle(*axis)
    parent = _momentum_to_numpy(momentum)
    children = np.tile(parent, (2, 1))
    children[0, :] *= z
    # backwards compatibility, switched inplace syntax for explicit ufunc
    # children[0, :3] @= rotation_matrix(angle, axis).T
    np.matmul(
        children[0, :3], rotation_matrix(angle, axis).T, out=children[0, :3]
    )
    children[1, :] -= children[0, :]
    return gcl.MomentumArray(children)


def split_hardest(
    momenta: gcl.MomentumArray,
    z: float,
    angle: float,
    axis: ty.Optional[ty.Union[ty.Tuple[float, float], SphericalAngle]] = None,
) -> gcl.MomentumArray:
    """Splits the momentum of the hardest particle into two momenta.
    Energy and 3-momentum is conserved over the whole MomentumArray.
    Hardness and collinearity of the split are determined by function
    parameters.

    :group: transform

    .. versionadded:: 0.4.0

    Parameters
    ----------
    momenta : MomentumArray
        Set of four-momenta, representing the point cloud of particles,
        prior to splitting.
    z : float
        Energy fraction retained by the first child after the split.
        Must be in range ``0.0 < z <= 0.5``.
    angle : float
        Angular displacement of the first child after the split.
    axis : SphericalAngle or tuple[float, float], optional
        The theta and phi values of the axis vector about which to
        rotate the momenta. If ``None``, will choose the axis normal to
        the plane swept out by the hardest and softest momentum
        constituents.

    Returns
    -------
    MomentumArray
        Set of four-momenta after splitting, such that the length is
        increased by one. The highest transverse momentum element has
        been removed from the set, and replaced with two momenta
        elements. The first and second children of the split are the
        penultimate and final elements of the MomentumArray,
        respectively.

    See Also
    --------
    soft_hard_axis : Axis of plane swept by softest and hardest momenta.
    rotation_matrix : Matrix to rotate 3-vectors about a given axis.

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
    if axis is None:
        if len(momenta) < 2:
            raise ValueError(
                "If axis is not provided, there must be at least two elements "
                "in momenta."
            )
        axis = soft_hard_axis(momenta)
    data = _momentum_to_numpy(momenta)
    hard_idx = momenta.pt.argmax()
    out = np.empty((len(momenta) + 1, 4), dtype=np.float64)
    out[:hard_idx] = data[:hard_idx]
    out[hard_idx:-2] = data[(hard_idx + 1) :]
    children = split_momentum(momenta[hard_idx], z, angle, axis)
    out[-2:, ...] = _momentum_to_numpy(children)[...]
    return gcl.MomentumArray(out)
