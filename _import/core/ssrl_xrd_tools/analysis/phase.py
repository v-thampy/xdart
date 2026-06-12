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
    """
    q: float
    intensity: float
    hkl: tuple[int, int, int]
    d_spacing: float


class PhaseModel:
    """Wrapper for a structural phase derived from a CIF file or manual lattice parameters."""
    
    def __init__(self, name: str, structure: "Structure" = None) -> None:
        self.name = name
        self.structure = structure
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

        This is the CIF-less entry point: it constructs a pymatgen
        :class:`Structure` with a single dummy atom at the origin and uses
        the given space group to generate the symmetry-allowed reflections
        via :class:`XRDCalculator`. The computed peak positions depend only
        on the lattice and space group (atomic form factors do scale the
        intensities, but for phase-fraction fitting what matters is the
        relative *pattern*, which :class:`PhaseFitter` normalises and scales
        anyway).

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
            Dummy species placed at the origin.  Only affects structure
            factors, not peak *positions*.  Default is hydrogen.

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

        # Single dummy atom at the origin; symmetry will populate equivalents
        species_list = [species] if isinstance(species, str) else list(species)
        coords = [[0.0, 0.0, 0.0]] * len(species_list)
        structure = Structure.from_spacegroup(
            sg.symbol, lattice, species_list, coords,
        )
        return cls(name=name, structure=structure)
        
    def calculate_peaks(self, wavelength: float = 1.5406) -> None:
        """Calculate peak positions based on the current structure lattice."""
        if not self.structure:
            return
            
        xrd_calc = XRDCalculator(wavelength=wavelength)
        pattern = xrd_calc.get_pattern(self.structure, scaled=True, two_theta_range=(0, 180))
        
        self.peaks = []
        for two_theta, intensity, hkls, d_hkl in zip(pattern.x, pattern.y, pattern.hkls, pattern.d_hkls):
            # Convert 2-theta to q (assuming wavelength)
            # q = 4 * pi * sin(theta) / lambda
            theta_rad = np.radians(two_theta / 2.0)
            q = (4 * np.pi * np.sin(theta_rad)) / wavelength
            
            # Select first HKL representation for simplicity
            hkl = hkls[0]["hkl"] if isinstance(hkls[0], dict) else hkls[0]
            # Convert 4-index Miller-Bravais (h,k,i,l) to 3-index (h,k,l)
            # for trigonal/hexagonal space groups where i = -(h+k)
            if len(hkl) == 4:
                hkl = (hkl[0], hkl[1], hkl[3])
            self.peaks.append(PeakData(q=q, intensity=intensity, hkl=hkl, d_spacing=d_hkl))

    def update_lattice(self, a: float=None, b: float=None, c: float=None,
                       alpha: float=None, beta: float=None, gamma: float=None) -> None:
        """Update lattice parameters and recalculate peaks dynamically.

        Raises
        ------
        ValueError
            If no crystal structure has been set on this PhaseModel.
        """
        if not self.structure:
            raise ValueError(
                f"Cannot update lattice for phase {self.name!r}: "
                "no crystal structure loaded. Use from_cif() or set structure first."
            )
            
        current_lattice = self.structure.lattice
        _a = a if a is not None else current_lattice.a
        _b = b if b is not None else current_lattice.b
        _c = c if c is not None else current_lattice.c
        _alpha = alpha if alpha is not None else current_lattice.alpha
        _beta = beta if beta is not None else current_lattice.beta
        _gamma = gamma if gamma is not None else current_lattice.gamma
        
        # Build new lattice; safely bypasses read-only lock
        new_lattice = Lattice.from_parameters(_a, _b, _c, _alpha, _beta, _gamma)
        self.structure.modify_lattice(new_lattice)
        self.calculate_peaks()


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
