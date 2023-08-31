import cmath
import dataclasses as dc
import math

import numpy as np

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


def test_pdgs():
    pdg_vals = np.arange(1, 7, dtype=np.int32)
    pdgs = gcl.PdgArray(pdg_vals)
    assert pdgs.name.tolist() == ["d", "u", "s", "c", "b", "t"]


def test_pmu_coords() -> None:
    momentum = MomentumExample()
    pmu = momentum.to_momentum_array()
    assert math.isclose(pmu.pt.item(), momentum.pt)
    assert math.isclose(pmu.phi.item(), momentum.phi * math.pi)
    assert math.isclose(pmu.mass.item(), 0.0, abs_tol=ZERO_TOL)
    assert math.isclose(pmu.theta.item(), math.atan(momentum.pt / momentum.pz))


def test_pmu_transform() -> None:
    momentum = MomentumExample()
    pmu = momentum.to_momentum_array()
    zero_transverse = pmu.shift_phi(math.pi) + pmu
    assert math.isclose(0.0, zero_transverse.pt.item(), abs_tol=ZERO_TOL)
    assert math.isclose(zero_transverse.mass.item(), 6.0)
