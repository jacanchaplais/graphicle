import math

import numba as nb
import numpy as np

import graphicle as gcl


@nb.njit("float64[:, :](int32, float64)")
def _fibonacci_sphere_momenta(
    num_points: int,
    energy: float,
) -> gcl.base.DoubleVector:
    pmu = np.empty((num_points, 4), dtype=np.float64)
    phi = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians
    for i in range(num_points):
        y = 1.0 - (i / (num_points - 1.0)) * 2.0  # y goes from 1 to -1
        radius = math.sqrt(1.0 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        pmu[i, 0] = math.cos(theta) * radius
        pmu[i, 1] = y
        pmu[i, 2] = math.sin(theta) * radius
        pmu[i, 3] = energy
    return pmu


def spherical_momenta(
    num_points: int, energy: float = 2.0
) -> gcl.MomentumArray:
    return gcl.MomentumArray(_fibonacci_sphere_momenta(num_points, energy))


def pencil_momenta() -> gcl.MomentumArray:
    return gcl.MomentumArray([[1.0, 0.0, 0.0, 2.0], [-1.0, 0.0, 0.0, 2.0]])


def test_thrust_spherical() -> None:
    pmu = spherical_momenta(10_000)
    thrust = gcl.calculate.thrust(pmu)
    msg = "Thrust for spherical event is not 0.5."
    assert math.isclose(thrust, 0.5, rel_tol=1.0e-3), msg


def test_thrust_pencil() -> None:
    pmu = pencil_momenta()
    thrust = gcl.calculate.thrust(pmu)
    msg = "Thrust for pencil-like event is not 1.0."
    assert math.isclose(thrust, 1.0, abs_tol=1.0e-6), msg


def test_spherocity_spherical() -> None:
    pmu = spherical_momenta(10_000)
    spherocity = gcl.calculate.spherocity(pmu)
    msg = "Spherocity for spherical event is not 1.0."
    assert math.isclose(spherocity, 1.0, rel_tol=1.0e-3), msg


def test_spherocity_pencil() -> None:
    pmu = pencil_momenta()
    spherocity = gcl.calculate.spherocity(pmu)
    msg = "Spherocity for pencil-like event is not 0.0."
    assert math.isclose(spherocity, 0.0, abs_tol=1.0e-6), msg


def test_c_parameter_spherical() -> None:
    pmu = spherical_momenta(10_000)
    c_param = gcl.calculate.c_parameter(pmu)
    msg = "C-parameter for spherical event is not 1.0."
    assert math.isclose(c_param, 1.0, rel_tol=1.0e-3), msg


def test_c_parameter_pencil() -> None:
    pmu = pencil_momenta()
    c_param = gcl.calculate.c_parameter(pmu)
    msg = "C-parameter for pencil-like event is not 0.0."
    assert math.isclose(c_param, 0.0, abs_tol=1.0e-6), msg
