"""
``test_transform``
==================

Unit tests for the data structures, probing their attributes and
methods.
"""
import math

import numpy as np
import pytest

import graphicle as gcl

DoubleVector = gcl.base.DoubleVector


def unit_vec_about_axis(vec: DoubleVector, axis: DoubleVector) -> DoubleVector:
    """Normalises ``vec``, such that its closest distance from ``axis``
    is unity. This enables us to consider it as a point along a 2D unit
    circle about ``axis``, albeit embedded in 3D space.
    """
    if not math.isclose(np.linalg.norm(axis), 1.0):
        raise ValueError("axis must be unit length.")
    mag = np.linalg.norm(vec)
    axis_angle = np.arccos(np.dot(vec, axis) / mag)
    radius = mag * np.sin(axis_angle)
    return vec / radius


def angular_distance_about_axis(
    vec_1: DoubleVector, vec_2: DoubleVector, axis: DoubleVector
) -> float:
    """Computes the angle subtended by the chord connecting projections
    of ``vec_1`` and ``vec_2`` in the plane of the unit circle about
    ``axis``.

    Will always give a positive number, as direction is not considered.
    """
    chord = unit_vec_about_axis(vec_2, axis) - unit_vec_about_axis(vec_1, axis)
    chord_length = np.linalg.norm(chord)
    return abs(2.0 * np.arcsin(0.5 * chord_length).item())


def test_momentum_split():
    rng = np.random.default_rng()
    for _ in range(100):
        energy_fraction = rng.uniform(1.0e-4, 0.5)
        angles = rng.uniform(-1.0, 1.0, 3)
        angles[0] = np.arccos(angles[0])
        angles[1:] *= np.pi
        rotation_angle = angles[2].item()
        rotation_axis = gcl.transform.SphericalAngle(*angles[:2].tolist())
        parent = gcl.MomentumArray.from_spherical_uniform(1, 100.0)
        children = gcl.transform.split_momentum(
            parent, energy_fraction, rotation_angle, rotation_axis
        )
        child_sum = np.sum(children, axis=0)
        assert np.allclose(parent, child_sum)
        first_child = children[0].copy()
        actual_angle = angular_distance_about_axis(
            parent._data[0, :3],
            first_child._data[0, :3],
            np.array(gcl.transform._angles_to_axis(rotation_axis)),
        )
        assert math.isclose(abs(rotation_angle), actual_angle)
        # invert the split of the first child:
        first_child_inv = children[0].copy()
        first_child_inv /= energy_fraction
        np.matmul(
            first_child_inv._data[0, :3],
            gcl.transform.rotation_matrix(
                -rotation_angle,
                rotation_axis,
            ).T,
            out=first_child_inv._data[0, :3],
        )
        assert np.allclose(parent, first_child_inv)


if __name__ == "__main__":
    pytest.main()
