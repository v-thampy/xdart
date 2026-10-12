"""Phase analysis: PhaseModel (CIF-driven), PeakData, PowerXRDWrapper."""
from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

__all__ = ["PeakData", "PhaseModel", "PowerXRDWrapper", "HAS_POWERXRD", "HAS_PYMATGEN"]

logger = logging.getLogger(__name__)

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False

try:
    import powerxrd as xrd
    HAS_POWERXRD = True
except ImportError:
    HAS_POWERXRD = False


@dataclass
class PeakData:
    """Single Bragg reflection.

    Attributes
    ----------
    q : float
        Scattering vector magnitude (Å⁻¹).
    intensity : float
        Relative intensity (0–100 scale from pymatgen, or arbitrary).
    hkl : tuple of int
        Miller indices.
    d_spacing : float
        d-spacing (Å).
    hkls : tuple of tuple of int
        Every hkl family of this line, ``hkl`` first: pymatgen merges the families sharing one 2θ (decision D3), and a
        lattice position keeps every family sharing it.
    """
    q: float
    intensity: float
    hkl: tuple[int, int, int]
    d_spacing: float
    hkls: tuple[tuple[int, int, int], ...] = ()


def _hkl_families(allowed, rotations):
    """The symmetry-distinct families among one lattice position's allowed hkl, Friedel pairs and the group's
    rotations being equivalent; each is named by its largest member, the largest family first."""
    remaining, families = set(allowed), []
    while remaining:
        representative = max(remaining)
        images = np.rint(np.einsum("nji,j->ni", rotations, np.asarray(representative))).astype(int)
        orbit = {tuple(int(index) for index in image) for image in images}
        orbit |= {tuple(-index for index in member) for member in orbit}
        families.append(representative)
        remaining -= orbit
        remaining.discard(representative)
    return tuple(families)


class PhaseModel:
    """Wrapper for a structural phase derived from a CIF file or manual lattice parameters."""
    
    def __init__(self, name: str, structure: "Structure" = None) -> None:
        self.name = name
        self.structure = structure
        self.lattice = None if structure is None else structure.lattice
        self.space_group = None
        self.source_kind = "cif" if structure is not None else "peaks"
        self.wavelength = 1.5406
        self.peaks: list[PeakData] = []
        if self.structure:
            self.calculate_peaks()
        
    @classmethod
    def from_cif(cls, path: Path | str, name: str | None = None) -> "PhaseModel":
        if not HAS_PYMATGEN:
            raise ImportError("pymatgen is required for CIF parsing.")

        path = Path(path)
        name = name or path.stem
        structure = Structure.from_file(str(path))
        return cls(name=name, structure=structure)

    @classmethod
    def from_lattice(
        cls,
        name: str,
        *,
        a: float,
        b: float | None = None,
        c: float | None = None,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        space_group: int | str = 1,
        species: str | list[str] = "H",
    ) -> "PhaseModel":
        """Build a PhaseModel from explicit lattice parameters and a space group.

        Lattice positions use the space group's systematic absences only.
        No dummy atoms or structure factors are invented. Numeric equal
        starting heights support a free-height fit; they are not physical
        relative intensities and cannot determine phase fractions.

        Parameters
        ----------
        name : str
            Name used to identify the phase in fits and reports.
        a : float
            Lattice parameter ``a`` in Ångström.  Required.
        b, c : float, optional
            Lattice parameters ``b`` and ``c``.  Default to ``a`` (cubic).
        alpha, beta, gamma : float
            Lattice angles in degrees.  Default 90° (orthogonal).
        space_group : int or str
            Space group number (1–230) or Hermann–Mauguin symbol
            (e.g. ``"Pca2_1"``, ``"P4_2/nmc"``).  Defaults to ``1`` (P1)
            which simply enumerates all lattice-allowed reflections.
        species : str or list of str
            Retained for call compatibility; lattice-only phases have no atoms.

        Returns
        -------
        PhaseModel

        Examples
        --------
        >>> # Orthorhombic HfO2 (Pca2_1)
        >>> ortho = PhaseModel.from_lattice(
        ...     "Ortho", a=5.06, b=5.23, c=5.07, space_group="Pca2_1",
        ... )
        >>> # Tetragonal HfO2 (P4_2/nmc)
        >>> tetra = PhaseModel.from_lattice(
        ...     "Tetra", a=5.08, c=5.19, space_group="P4_2/nmc",
        ... )
        """
        if not HAS_PYMATGEN:
            raise ImportError("pymatgen is required for PhaseModel.from_lattice.")

        from pymatgen.symmetry.groups import SpaceGroup

        _b = b if b is not None else a
        _c = c if c is not None else a

        lattice = Lattice.from_parameters(a, _b, _c, alpha, beta, gamma)

        # Accept int, numeric string, or H–M symbol
        if isinstance(space_group, str) and space_group.strip().isdigit():
            space_group = int(space_group.strip())
        try:
            sg = SpaceGroup(space_group) if isinstance(space_group, str) else SpaceGroup.from_int_number(int(space_group))
        except Exception as exc:
            raise ValueError(
                f"Unknown space group {space_group!r}. "
                "Pass an integer (1–230) or a Hermann–Mauguin symbol like 'Pca2_1'."
            ) from exc

        phase = cls(name=name)
        phase.lattice, phase.space_group, phase.source_kind = lattice, sg, "lattice"
        phase.calculate_peaks()
        return phase
        
    def calculate_peaks(self, wavelength: float = 1.5406, *,
                        two_theta_range: tuple[float, float] = (0, 180),
                        q_range: tuple[float, float] | None = None,
                        max_reflections: int | None = None) -> None:
        """List only the reachable range, normalized inside that range.

        This is the shared listing owner for fitting and reference overlays.
        A forced wider backend listing is filtered before normalization.
        Kept as pymatgen gives them (maintainer decisions D2 and D3, October 11):
        its XRDCalculator omits reflections under 0.1 % of the strongest, so
        that relative cut applies even at a floor of 0; and reflections sharing
        one 2θ are one line, whose ``hkls`` list every merged family.
        """
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be positive and finite")
        angles = np.asarray(two_theta_range, dtype=float)
        if angles.shape != (2,) or not np.isfinite(angles).all() or not 0 <= angles[0] <= angles[1] <= 180:
            raise ValueError("invalid reachable 2theta range")
        lower, upper = 4*np.pi*np.sin(np.radians(angles/2))/wavelength
        if q_range is not None:
            requested = np.asarray(q_range, dtype=float)
            if requested.shape != (2,) or not np.isfinite(requested).all() or not 0 <= requested[0] <= requested[1]:
                raise ValueError("invalid reachable q range")
            lower, upper = max(lower, requested[0]), min(upper, requested[1])
            angles = np.degrees(2*np.arcsin(np.clip(np.array([lower, upper])*wavelength/(4*np.pi), 0, 1)))
        self.wavelength = float(wavelength)
        self.listing_q_range = (float(lower), float(upper))
        self.peaks = []
        if lower > upper:
            return
        if self.source_kind == "lattice":
            self._calculate_lattice_peaks(lower, upper, max_reflections)
            return
        if not self.structure:
            return
        self.lattice = self.structure.lattice
        xrd_calc = XRDCalculator(wavelength=wavelength)
        pattern = xrd_calc.get_pattern(self.structure, scaled=False, two_theta_range=tuple(angles))
        
        self.peaks = []
        for two_theta, intensity, hkls, d_hkl in zip(pattern.x, pattern.y, pattern.hkls, pattern.d_hkls):
            # Convert 2-theta to q (assuming wavelength)
            # q = 4 * pi * sin(theta) / lambda
            theta_rad = np.radians(two_theta / 2.0)
            q = (4 * np.pi * np.sin(theta_rad)) / wavelength
            
            # Every family pymatgen merged into this line, in its order; the first names the line.
            families = []
            for family in hkls:
                hkl = family["hkl"] if isinstance(family, dict) else family
                # Convert 4-index Miller-Bravais (h,k,i,l) to 3-index (h,k,l)
                # for trigonal/hexagonal space groups where i = -(h+k)
                if len(hkl) == 4:
                    hkl = (hkl[0], hkl[1], hkl[3])
                families.append(tuple(int(index) for index in hkl))
            if lower - 1e-12 <= q <= upper + 1e-12:
                self.peaks.append(PeakData(q=q, intensity=float(intensity), hkl=families[0], d_spacing=d_hkl,
                                           hkls=tuple(families)))
        maximum = max((p.intensity for p in self.peaks), default=0.)
        if maximum > 0:
            for peak in self.peaks:
                peak.intensity = 100.*peak.intensity/maximum
        if max_reflections is not None and len(self.peaks) > max_reflections:
            raise ValueError("per-phase reflection limit exceeded")

    def _calculate_lattice_peaks(self, lower, upper, limit):
        operations = tuple(self.space_group.symmetry_ops)
        rotations = np.array([op.rotation_matrix for op in operations])
        translations = np.array([op.translation_vector for op in operations])
        points = self.lattice.reciprocal_lattice_crystallographic.get_points_in_sphere(
            [[0, 0, 0]], [0, 0, 0], upper/(2*np.pi),
        )
        positions = {}  # each allowed position's q and its allowed hkl, in distance order
        for point in sorted(points, key=lambda item: item[1]):
            hkl = np.rint(point[0]).astype(int)
            q = float(2*np.pi*point[1])
            if q <= 1e-12 or q < lower-1e-12 or q > upper+1e-12:
                continue
            # An operation stabilizing h imposes F(h)=exp(2πih.t)F(h).
            # A non-unit phase therefore forces a systematic absence.
            transformed = np.einsum("nji,j->ni", rotations, hkl)
            stabilizers = np.all(np.isclose(transformed, hkl, atol=1e-8, rtol=0), axis=1)
            phases = translations[stabilizers] @ hkl
            if np.any(np.abs(phases-np.rint(phases)) > 1e-7):
                continue
            key = round(q, 10)
            if key not in positions:
                positions[key] = (q, [])
                if limit is not None and len(positions) > limit:
                    raise ValueError("per-phase reflection limit exceeded")
            positions[key][1].append(tuple(map(int, hkl)))
        for q, allowed in positions.values():
            families = _hkl_families(allowed, rotations)
            self.peaks.append(PeakData(q, 100., families[0], 2*np.pi/q, families))

    def update_lattice(self, a: float=None, b: float=None, c: float=None,
                       alpha: float=None, beta: float=None, gamma: float=None) -> None:
        """Update lattice parameters and recalculate peaks dynamically.

        Raises
        ------
        ValueError
            If no crystal structure has been set on this PhaseModel.
        """
        if not self.structure and self.lattice is None:
            raise ValueError(
                f"Cannot update lattice for phase {self.name!r}: "
                "no lattice loaded. Use from_cif() or from_lattice() first."
            )
            
        current_lattice = self.structure.lattice if self.structure else self.lattice
        _a = a if a is not None else current_lattice.a
        _b = b if b is not None else current_lattice.b
        _c = c if c is not None else current_lattice.c
        _alpha = alpha if alpha is not None else current_lattice.alpha
        _beta = beta if beta is not None else current_lattice.beta
        _gamma = gamma if gamma is not None else current_lattice.gamma
        
        # Build new lattice; safely bypasses read-only lock
        new_lattice = Lattice.from_parameters(_a, _b, _c, _alpha, _beta, _gamma)
        self.lattice = new_lattice
        if self.structure:
            self.structure.lattice = new_lattice
        self.calculate_peaks(wavelength=self.wavelength,
                             q_range=getattr(self, "listing_q_range", None))


class PowerXRDWrapper:
    """Basic wrapper showcasing how to use PowerXRD for simplistic pattern phase analysis."""
    
    @staticmethod
    def load_data(xy_path: str | Path):
        """Loads data into PowerXRD Data format."""
        if not HAS_POWERXRD:
            raise ImportError("powerxrd is required.")
        return xrd.Data(str(xy_path)).importfile()

    @staticmethod
    def calc_scherrer(data_tuple, xrange: list[float], show: bool = False):
        """Calculate Scherrer crystallite size using PowerXRD's backend."""
        if not HAS_POWERXRD:
            raise ImportError("powerxrd is required.")
        chart = xrd.Chart(*data_tuple)
        return chart.SchPeak(xrange=xrange, show=show)
