"""
``test_data``
=============

Unit tests for the data structures, probing their attributes and
methods.
"""
import cmath
import dataclasses as dc
import math
import random

import numpy as np
import pytest

import graphicle as gcl

ZERO_TOL = 1.0e-10  # absolute tolerance for detecting zero-values


@dc.dataclass
class MomentumExample:
    """Dataclass for representing the four-momentum of a particle, based
    on its cylindrical beamline coordinates.

    Parameters
    ----------
    pt, phi, pz, energy : float
        Components of the four-momentum in cylindrical coordinates.
        Note that ``phi`` is given in units of pi.
    """

    pt: float = 3.0
    phi: float = 0.75
    pz: float = 4.0
    energy: float = 5.0

    @property
    def _polar(self) -> complex:
        return self.pt * cmath.exp(complex(0, self.phi * math.pi))

    @property
    def px(self) -> float:
        """x-component of four momentum."""
        return self._polar.real

    @property
    def py(self) -> float:
        """y-component of four momentum."""
        return self._polar.imag

    def to_momentum_array(self) -> gcl.MomentumArray:
        """Formats the components in a ``MomentumArray``."""
        return gcl.MomentumArray([(self.px, self.py, self.pz, self.energy)])


def test_pdg_quark_names():
    """Tests that the PDG names are correcly identified for the quarks."""
    pdg_vals = np.arange(1, 7, dtype=np.int32)
    pdgs = gcl.PdgArray(pdg_vals)
    assert pdgs.name.tolist() == ["d", "u", "s", "c", "b", "t"]


def test_pmu_coords() -> None:
    """Tests that the components of the momentum are correctly stored
    and calculated.
    """
    momentum = MomentumExample()
    pmu = momentum.to_momentum_array()
    correct_pt = math.isclose(pmu.pt.item(), momentum.pt)
    assert correct_pt, "Incorrect pT."
    correct_phi = math.isclose(pmu.phi.item(), momentum.phi * math.pi)
    assert correct_phi, "Incorrect phi."
    correct_mass = math.isclose(pmu.mass.item(), 0.0, abs_tol=ZERO_TOL)
    assert correct_mass, "Nonzero mass."
    correct_theta = math.isclose(
        pmu.theta.item(), math.atan(momentum.pt / momentum.pz)
    )
    assert correct_theta, "Incorrect theta."


def test_pmu_transform_invertible() -> None:
    """Tests that the ``MomentumArray`` transforms are invertible."""
    momentum = MomentumExample()
    pmu = momentum.to_momentum_array()
    shift = random.uniform(0.0, math.tau)
    phi_invertible = np.allclose(pmu, pmu.shift_phi(shift).shift_phi(-shift))
    assert phi_invertible, "Azimuth shift is not invertible."
    rap_invertible = np.allclose(
        pmu, pmu.shift_rapidity(shift).shift_rapidity(-shift)
    )
    assert rap_invertible, "Rapidity shift is not invertible."
    eta_invertible = np.allclose(pmu, pmu.shift_eta(shift).shift_eta(-shift))
    assert eta_invertible, "Pseudorapidity shift is not invertible."


def test_pmu_zero_pt() -> None:
    """Tests that when antiparallel momenta in the xy plane are added,
    they have the correct properties, and the azimuth is flagged as
    invalid.
    """
    momentum = MomentumExample()
    pmu = momentum.to_momentum_array()
    pmu_zero_pt = pmu.shift_phi(math.pi) + pmu
    zero_pt = math.isclose(0.0, pmu_zero_pt.pt.item(), abs_tol=ZERO_TOL)
    assert zero_pt, "Transverse momentum not properly cancelled"
    correct_mass = math.isclose(pmu_zero_pt.mass.item(), 6.0)
    assert correct_mass, "Mass generated is incorrect."
    eta_inf = math.isinf(pmu_zero_pt.eta.item())
    assert eta_inf, "Pseudorapidity is not infinite when longitudinal."
    with pytest.warns(gcl.base.NumericalStabilityWarning):
        phi_invalid = math.isnan(pmu_zero_pt.phi.item())
    assert phi_invalid, "Azimuth is not NaN when pT is low"
