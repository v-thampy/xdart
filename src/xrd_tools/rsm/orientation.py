"""A SPEC source's UB recomputed in the instrument's own model (AUD-9 stage 2b decision 2).

SPEC stores no circle directions: its psic convention lives in its code, and the ``#G3`` UB in a scan header was
computed in it. What the header does store is the lattice and the two orientation reflections SPEC computed
``#G3`` from (``#G1``). RSM maps with the instrument record's circle directions and the calibration's zero
offsets, so it recomputes the UB from those reflections in that same model, by Busing and Levy (1967): B from the
reciprocal lattice, U from the two reflections' scattering vectors in the sample frame. ``#G3`` becomes a
cross-check: in psic's standard directions the two agree to the header's printed digits.

Headless and Qt-free; the scattering vectors come from a caller's ``point`` (``PixelQMap.orientation_ub``, the
same xrayutilities model the area mapping uses).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: SPEC psic's motor order for a reflection's angles in ``#G1`` (as the ``#O0`` header lists them).
G1_ANGLE_ORDER = ("del", "eta", "chi", "phi", "nu", "mu")
#: xrayutilities' circle order for RSM: the sample circles, then the detector circles.
ROLE_ORDER = ("mu", "eta", "chi", "phi", "nu", "del")
#: A recomputed UB within this relative difference of ``#G3`` is ``#G3`` itself: ``#G1`` prints about ten
#: significant digits (scan 43: 5.8e-10).
ROUNDING = 1e-6


@dataclass(frozen=True)
class OrientationReflection:
    hkl: tuple[float, float, float]
    angles: tuple[float, ...]  # by ROLE_ORDER
    wavelength_A: float


@dataclass(frozen=True)
class SpecOrientation:
    """A scan's ``#G1``: the lattice (Å, degrees), its reciprocal (2π/Å, degrees) and the two reflections."""

    lattice: tuple[float, ...]
    reciprocal: tuple[float, ...]
    reflections: tuple[OrientationReflection, OrientationReflection]

    def b_matrix(self) -> np.ndarray:
        """Busing and Levy's B, from ``#G1``'s own reciprocal lattice (2π included, as SPEC's ``#G3``)."""
        a_star, b_star, c_star = self.reciprocal[:3]
        alpha_star, beta_star, gamma_star = np.radians(self.reciprocal[3:6])
        alpha = np.radians(self.lattice[3])
        return np.array([
            [a_star, b_star * np.cos(gamma_star), c_star * np.cos(beta_star)],
            [0.0, b_star * np.sin(gamma_star), -c_star * np.sin(beta_star) * np.cos(alpha)],
            [0.0, 0.0, 2.0 * np.pi / self.lattice[2]],
        ], dtype=np.float64)


def spec_orientation(values) -> SpecOrientation | None:
    """``#G1``'s values as a ``SpecOrientation``; None when they hold no two usable orientation reflections (too
    few values, a non-finite one, a zero or parallel HKL pair, a non-positive wavelength or lattice)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 32 or not np.all(np.isfinite(array[:32])):
        return None
    lattice, reciprocal = tuple(array[0:6]), tuple(array[6:12])
    hkl0, hkl1 = array[12:15], array[15:18]
    wavelengths = array[30:32]
    if (min(lattice[:3]) <= 0 or min(reciprocal[:3]) <= 0 or np.any(wavelengths <= 0)
            or np.linalg.norm(np.cross(hkl0, hkl1)) == 0.0):
        return None
    reflections = []
    for hkl, start, wavelength in ((hkl0, 18, wavelengths[0]), (hkl1, 24, wavelengths[1])):
        by_name = dict(zip(G1_ANGLE_ORDER, array[start:start + 6]))
        reflections.append(OrientationReflection(
            tuple(float(value) for value in hkl), tuple(float(by_name[role]) for role in ROLE_ORDER),
            float(wavelength)))
    return SpecOrientation(lattice, reciprocal, (reflections[0], reflections[1]))


def _triad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    t1 = first / np.linalg.norm(first)
    t3 = np.cross(first, second)
    t3 = t3 / np.linalg.norm(t3)
    return np.column_stack((t1, np.cross(t3, t1), t3))


def busing_levy_ub(
    point: Callable[[tuple[float, ...], float], np.ndarray],
    orientation: SpecOrientation,
) -> np.ndarray:
    """The UB the two reflections give in ``point``'s model: ``point(angles, wavelength)`` is a reflection's
    scattering vector in the sample frame (xrayutilities ``Ang2Q.point`` with the identity UB). U turns the
    reflections' crystal triad onto their sample-frame triad; UB = U·B."""
    b = orientation.b_matrix()
    vectors = [np.asarray(point(item.angles, item.wavelength_A), dtype=np.float64)
               for item in orientation.reflections]
    if any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors):
        raise ValueError("an orientation reflection has no finite scattering vector")
    if np.linalg.norm(np.cross(*vectors)) == 0.0:
        raise ValueError("the orientation reflections are parallel in this model")
    crystal = [b @ np.asarray(item.hkl, dtype=np.float64) for item in orientation.reflections]
    return _triad(*vectors) @ _triad(*crystal).T @ b


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    """The largest element difference over the reference's largest element."""
    right = np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - right)) / np.max(np.abs(right)))
