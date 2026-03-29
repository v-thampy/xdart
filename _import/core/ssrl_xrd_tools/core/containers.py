# ssrl_xrd_tools/core/containers.py
"""
Shared data containers for calibration and azimuthal integration results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PONI key mapping
# ---------------------------------------------------------------------------

# Mapping from pyFAI capitalised .poni keys → PONI field names.
_PYFAI_KEY_MAP: dict[str, str] = {
    "Distance": "dist",
    "Poni1": "poni1",
    "Poni2": "poni2",
    "Rot1": "rot1",
    "Rot2": "rot2",
    "Rot3": "rot3",
    "Wavelength": "wavelength",
    "Detector": "detector",
}

# ---------------------------------------------------------------------------
# Unit-conversion helpers (inline math keeps core/ dep-free)
# ---------------------------------------------------------------------------

_HC_KEV_A: float = 12.398  # hc in keV·Å


def _tth_deg_to_q_a(tth_deg: np.ndarray, wavelength_A: float) -> np.ndarray:
    return (4.0 * np.pi / wavelength_A) * np.sin(np.deg2rad(tth_deg) / 2.0)


def _q_a_to_tth_deg(q: np.ndarray, wavelength_A: float) -> np.ndarray:
    return 2.0 * np.rad2deg(np.arcsin(np.clip(q * wavelength_A / (4.0 * np.pi), -1.0, 1.0)))


def _convert_radial_axis(
    radial: np.ndarray,
    from_unit: str,
    to_unit: str,
    wavelength_A: float | None = None,
) -> np.ndarray:
    """
    Convert a radial axis array between pyFAI unit representations.

    Parameters
    ----------
    radial : ndarray
        Input axis values.
    from_unit, to_unit : str
        pyFAI unit strings (e.g. ``"q_A^-1"``, ``"2th_deg"``).
    wavelength_A : float or None
        Wavelength in Angstroms; required for 2theta ↔ q conversions.

    Returns
    -------
    ndarray
        Converted axis.

    Raises
    ------
    ValueError
        If the conversion is not supported, or wavelength is missing when
        required, or a cross-axis GI conversion is attempted.
    """
    if from_unit == to_unit:
        return radial.copy()

    # --- Q-type units: detect base and suffix for _A^-1 / _nm^-1
    _Q_SFXS = ("_A^-1", "_nm^-1")
    from_q_base, from_q_sfx = None, None
    to_q_base, to_q_sfx = None, None
    for _s in _Q_SFXS:
        if from_unit.endswith(_s):
            from_q_base, from_q_sfx = from_unit[: -len(_s)], _s
        if to_unit.endswith(_s):
            to_q_base, to_q_sfx = to_unit[: -len(_s)], _s

    if from_q_base is not None and to_q_base is not None:
        if from_q_base != to_q_base:
            raise ValueError(
                f"Cross-axis GI conversion not supported: '{from_unit}' → '{to_unit}'. "
                "Only conversions between _A^-1 and _nm^-1 of the same axis are allowed."
            )
        if from_q_sfx == "_A^-1" and to_q_sfx == "_nm^-1":
            return radial * 10.0
        if from_q_sfx == "_nm^-1" and to_q_sfx == "_A^-1":
            return radial / 10.0
        # Same base and same suffix → same unit, already handled above
        return radial.copy()

    # --- 2theta degree ↔ radian
    if from_unit == "2th_deg" and to_unit == "2th_rad":
        return np.deg2rad(radial)
    if from_unit == "2th_rad" and to_unit == "2th_deg":
        return np.rad2deg(radial)

    # --- 2theta ↔ q (requires wavelength)
    _need_wl = (
        f"Conversion '{from_unit}' → '{to_unit}' requires wavelength in Angstroms."
    )
    if wavelength_A is None:
        raise ValueError(_need_wl)
    wl = float(wavelength_A)

    # Normalise input to q_A^-1
    if from_unit == "2th_deg":
        q = _tth_deg_to_q_a(radial, wl)
    elif from_unit == "2th_rad":
        q = _tth_deg_to_q_a(np.rad2deg(radial), wl)
    elif from_unit == "q_nm^-1":
        q = radial / 10.0
    elif from_unit == "q_A^-1":
        q = radial.copy()
    else:
        raise ValueError(
            f"Cannot convert from unit '{from_unit}' to '{to_unit}'. "
            "Unsupported source unit."
        )

    # Convert q_A^-1 to target
    if to_unit == "2th_deg":
        return _q_a_to_tth_deg(q, wl)
    if to_unit == "2th_rad":
        return np.deg2rad(_q_a_to_tth_deg(q, wl))
    if to_unit == "q_nm^-1":
        return q * 10.0
    if to_unit == "q_A^-1":
        return q

    raise ValueError(
        f"Cannot convert from unit '{from_unit}' to '{to_unit}'. "
        "Unsupported target unit."
    )


def _convert_angular_axis(
    arr: np.ndarray,
    from_unit: str,
    to_unit: str,
) -> np.ndarray:
    """
    Convert an angular axis array between ``*_deg`` / ``*_rad`` unit variants.

    Parameters
    ----------
    arr : ndarray
        Input axis values.
    from_unit, to_unit : str
        Must share the same base (e.g. ``"chi_deg"`` / ``"chi_rad"``).

    Returns
    -------
    ndarray
        Converted axis.

    Raises
    ------
    ValueError
        If the conversion is not supported.
    """
    if from_unit == to_unit:
        return arr.copy()
    for sfx_from, sfx_to, fn in (
        ("_deg", "_rad", np.deg2rad),
        ("_rad", "_deg", np.rad2deg),
    ):
        if from_unit.endswith(sfx_from) and to_unit.endswith(sfx_to):
            base_f = from_unit[: -len(sfx_from)]
            base_t = to_unit[: -len(sfx_to)]
            if base_f == base_t:
                return fn(arr)
            raise ValueError(
                f"Cannot convert azimuthal unit '{from_unit}' → '{to_unit}': "
                "different axis bases."
            )
    raise ValueError(
        f"Cannot convert azimuthal unit '{from_unit}' → '{to_unit}'. "
        "Only deg ↔ rad conversions of the same axis are supported."
    )


def _sigma_add(
    s1: np.ndarray | None,
    s2: np.ndarray | None,
) -> np.ndarray | None:
    """Propagate uncertainty through addition/subtraction: σ = √(σ₁² + σ₂²)."""
    if s1 is None or s2 is None:
        return None
    return np.sqrt(s1 ** 2 + s2 ** 2)


# ---------------------------------------------------------------------------
# NeXus unit mapping
# ---------------------------------------------------------------------------

_UNIT_TO_NEXUS: dict[str, tuple[str, str]] = {
    "2th_deg": ("degrees", "2Theta"),
    "2th_rad": ("radians", "2Theta"),
    "q_A^-1": ("angstrom^-1", "Q"),
    "q_nm^-1": ("nm^-1", "Q"),
    "d*2_A^-2": ("angstrom^-2", "d*^2"),
    "r_mm": ("mm", "r"),
    "chi_deg": ("degrees", "Chi"),
    "chi_rad": ("radians", "Chi"),
    "qip_A^-1": ("angstrom^-1", "Q_ip"),
    "qip_nm^-1": ("nm^-1", "Q_ip"),
    "qoop_A^-1": ("angstrom^-1", "Q_oop"),
    "qoop_nm^-1": ("nm^-1", "Q_oop"),
    "qtot_A^-1": ("angstrom^-1", "Q_total"),
    "qtot_nm^-1": ("nm^-1", "Q_total"),
    "chigi_deg": ("degrees", "Chi_GI"),
    "chigi_rad": ("radians", "Chi_GI"),
}


def _pyfai_unit_to_nexus(unit_str: str) -> tuple[str, str]:
    """
    Map a pyFAI unit string to ``(nexus_units, long_name)``.

    Parameters
    ----------
    unit_str : str
        pyFAI unit string (e.g. ``"q_A^-1"``, ``"chi_deg"``).

    Returns
    -------
    tuple of (str, str)
        ``(nexus_units, long_name)`` — e.g. ``("angstrom^-1", "Q")``.
        Falls back to ``("a.u.", unit_str)`` for unknown units.
    """
    return _UNIT_TO_NEXUS.get(unit_str, ("a.u.", unit_str))


# ---------------------------------------------------------------------------
# HDF5 helpers (lazy import wrapper)
# ---------------------------------------------------------------------------

def _h5_replace(grp: h5py.Group, name: str, data: np.ndarray, **kwargs) -> None:
    """Delete-and-recreate an HDF5 dataset.

    Note: lzf compression causes bus errors on ARM64 macOS with certain
    h5py builds.  We replace it with gzip which is universally safe.
    """
    if name in grp:
        del grp[name]
    # lzf crashes on ARM64 macOS — use gzip instead
    if kwargs.get('compression') == 'lzf':
        kwargs['compression'] = 'gzip'
        kwargs['compression_opts'] = 1  # fastest gzip level
    grp.create_dataset(name, data=data, **kwargs)


# ---------------------------------------------------------------------------
# PONI
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PONI:
    """
    pyFAI point-of-normal-incidence calibration geometry.

    Parameters
    ----------
    dist : float
        Sample-to-detector distance (m).
    poni1, poni2 : float
        PONI coordinates in detector plane (m).
    rot1, rot2, rot3 : float, optional
        Detector rotations (rad).
    wavelength : float, optional
        Wavelength (m); 0 if unknown.
    detector : str, optional
        pyFAI detector name or identifier.
    """

    dist: float
    poni1: float
    poni2: float
    rot1: float = 0.0
    rot2: float = 0.0
    rot3: float = 0.0
    wavelength: float = 0.0
    detector: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a pyFAI-style dictionary representation.

        Returns
        -------
        dict
            Keys: ``dist``, ``poni1``, ``poni2``, ``rot1``, ``rot2``,
            ``rot3``, ``wavelength``, ``detector``.
        """
        return {
            "dist": self.dist,
            "poni1": self.poni1,
            "poni2": self.poni2,
            "rot1": self.rot1,
            "rot2": self.rot2,
            "rot3": self.rot3,
            "wavelength": self.wavelength,
            "detector": self.detector,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PONI:
        """
        Create a :class:`PONI` from a dictionary.

        Accepts both pyFAI capitalised keys (``Distance``, ``Poni1``, …)
        and lowercase keys (``dist``, ``poni1``, …).  Unknown keys are
        silently ignored.

        Parameters
        ----------
        d : dict
            Mapping of calibration values.

        Returns
        -------
        PONI
        """
        _float_fields = {"dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"}
        values: dict[str, float | str] = {}
        for k, v in d.items():
            field = _PYFAI_KEY_MAP.get(k) or (
                k if k in _float_fields or k == "detector" else None
            )
            if field is None:
                continue
            if field == "wavelength" and isinstance(v, str):
                try:
                    v = float(eval(v))  # pyFAI sometimes writes wavelength as "6.2e-11"
                except Exception:
                    logger.warning("Could not parse wavelength %r; using 0.0", v)
                    v = 0.0
            if field == "detector":
                values["detector"] = str(v) if v is not None else ""
            else:
                values[field] = float(v) if v is not None else 0.0
        return cls(
            dist=values.get("dist", 0.0),
            poni1=values.get("poni1", 0.0),
            poni2=values.get("poni2", 0.0),
            rot1=values.get("rot1", 0.0),
            rot2=values.get("rot2", 0.0),
            rot3=values.get("rot3", 0.0),
            wavelength=values.get("wavelength", 0.0),
            detector=values.get("detector", ""),
        )

    @classmethod
    def from_poni_file(cls, path: Path | str) -> PONI:
        """
        Load a ``.poni`` calibration file (pyFAI YAML format).

        Parameters
        ----------
        path : Path or str
            Path to the ``.poni`` file.

        Returns
        -------
        PONI

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the file does not contain a YAML mapping.
        """
        import yaml

        p = Path(path)
        with p.open() as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{p}: expected a YAML mapping, got {type(data).__name__}")
        return cls.from_dict(data)

    def to_poni_file(self, path: Path | str) -> None:
        """
        Write calibration geometry to a ``.poni`` file (pyFAI YAML format).

        Parameters
        ----------
        path : Path or str
            Destination path.  Parent directories are created if needed.
        """
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "poni_version": 2,
            "Detector": self.detector or "Detector",
            "Detector_config": {},
            "Distance": self.dist,
            "Poni1": self.poni1,
            "Poni2": self.poni2,
            "Rot1": self.rot1,
            "Rot2": self.rot2,
            "Rot3": self.rot3,
            "Wavelength": self.wavelength,
        }
        with p.open("w") as fh:
            yaml.dump(data, fh, default_flow_style=False)


# ---------------------------------------------------------------------------
# IntegrationResult1D
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IntegrationResult1D:
    """
    Result of 1D azimuthal or radial integration.

    Works for standard integration (intensity vs q or 2theta) and GI
    integration (intensity vs qip, qoop, qtotal, exit angle, etc).
    The ``unit`` field identifies what the radial axis represents.

    Parameters
    ----------
    radial : ndarray
        Radial axis values (q, 2theta, qip, qoop, …).
    intensity : ndarray
        Integrated intensity; same length as ``radial``.
    sigma : ndarray or None, optional
        Per-bin uncertainty.
    unit : str, optional
        pyFAI unit string for the radial axis (e.g. ``"q_A^-1"``,
        ``"2th_deg"``, ``"qoop_A^-1"``).
    """

    radial: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None = None
    unit: str = "2th_deg"

    def __post_init__(self) -> None:
        self.radial = np.asarray(self.radial, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float)
        if self.radial.shape != self.intensity.shape:
            raise ValueError(
                f"radial shape {self.radial.shape} != intensity shape "
                f"{self.intensity.shape}"
            )
        if self.sigma is not None and self.sigma.shape != self.radial.shape:
            raise ValueError(
                f"sigma shape {self.sigma.shape} != radial shape "
                f"{self.radial.shape}"
            )

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def to_unit(
        self,
        target_unit: str,
        wavelength: float | None = None,
    ) -> IntegrationResult1D:
        """
        Convert the radial axis to a different unit.

        Returns a *new* :class:`IntegrationResult1D`; the original is
        unchanged.

        Supported conversions (no wavelength required):

        - ``q_A^-1`` ↔ ``q_nm^-1`` (and similarly for qip, qoop, qtot)
        - ``2th_deg`` ↔ ``2th_rad``

        Supported conversions (requires *wavelength* in Angstroms):

        - ``2th_deg`` / ``2th_rad`` ↔ ``q_A^-1`` / ``q_nm^-1``

        Cross-axis GI conversions (e.g. ``qip_A^-1`` → ``qoop_A^-1``)
        raise :exc:`ValueError`.

        Parameters
        ----------
        target_unit : str
            pyFAI unit string to convert to.
        wavelength : float or None, optional
            Wavelength in **Angstroms**.  Required for 2theta ↔ q
            conversions.

        Returns
        -------
        IntegrationResult1D
            New instance with converted radial axis and updated unit.

        Raises
        ------
        ValueError
            If the conversion is not supported or wavelength is missing.
        """
        new_radial = _convert_radial_axis(
            self.radial, self.unit, target_unit, wavelength
        )
        return IntegrationResult1D(
            radial=new_radial,
            intensity=self.intensity.copy(),
            sigma=self.sigma.copy() if self.sigma is not None else None,
            unit=target_unit,
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _check_compatible(self, other: IntegrationResult1D) -> None:
        if self.unit != other.unit:
            raise ValueError(
                f"Unit mismatch: '{self.unit}' vs '{other.unit}'. "
                "Convert both to the same unit before combining."
            )
        if self.radial.shape != other.radial.shape:
            raise ValueError(
                f"Radial axis shape mismatch: {self.radial.shape} vs "
                f"{other.radial.shape}."
            )

    def __add__(self, other: IntegrationResult1D) -> IntegrationResult1D:
        """
        Add intensities element-wise.  Propagates sigma if both have it.

        Raises
        ------
        ValueError
            If units or radial axis shapes differ.
        """
        self._check_compatible(other)
        return IntegrationResult1D(
            radial=self.radial.copy(),
            intensity=self.intensity + other.intensity,
            sigma=_sigma_add(self.sigma, other.sigma),
            unit=self.unit,
        )

    def __sub__(self, other: IntegrationResult1D) -> IntegrationResult1D:
        """
        Subtract intensities element-wise.  Propagates sigma if both have it.

        Raises
        ------
        ValueError
            If units or radial axis shapes differ.
        """
        self._check_compatible(other)
        return IntegrationResult1D(
            radial=self.radial.copy(),
            intensity=self.intensity - other.intensity,
            sigma=_sigma_add(self.sigma, other.sigma),
            unit=self.unit,
        )

    def __mul__(self, scalar: float) -> IntegrationResult1D:
        """
        Scale intensity (and sigma) by *scalar*.

        Parameters
        ----------
        scalar : float
            Multiplicative scale factor.

        Returns
        -------
        IntegrationResult1D
        """
        s = float(scalar)
        new_sigma = abs(s) * self.sigma if self.sigma is not None else None
        return IntegrationResult1D(
            radial=self.radial.copy(),
            intensity=s * self.intensity,
            sigma=new_sigma,
            unit=self.unit,
        )

    def __rmul__(self, scalar: float) -> IntegrationResult1D:
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pyfai(
        cls,
        result: object,
        unit: str | None = None,
    ) -> IntegrationResult1D:
        """
        Create from a pyFAI ``integrate1d`` result.

        Parameters
        ----------
        result : pyFAI integrate1d namedtuple
            Must have ``.radial``, ``.intensity`` attributes, and
            optionally ``.sigma`` and ``.unit``.
        unit : str or None, optional
            Override the unit from the result object.

        Returns
        -------
        IntegrationResult1D
        """
        raw_unit = getattr(result, "unit", None)
        if unit is None:
            unit = str(raw_unit) if raw_unit is not None else "2th_deg"
        sigma_raw = getattr(result, "sigma", None)
        sigma = np.asarray(sigma_raw, dtype=float) if sigma_raw is not None else None
        return cls(
            radial=np.asarray(result.radial, dtype=float),  # type: ignore[attr-defined]
            intensity=np.asarray(result.intensity, dtype=float),  # type: ignore[attr-defined]
            sigma=sigma,
            unit=unit,
        )

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    def to_hdf5(self, grp: h5py.Group, compression: str = "lzf") -> None:
        """
        Write to an HDF5 group.

        Creates datasets ``radial``, ``intensity``, and (if present)
        ``sigma``.  Stores ``unit`` as a group attribute.

        Parameters
        ----------
        grp : h5py.Group
            Destination group.
        compression : str, optional
            HDF5 compression filter (default ``"lzf"``).
        """
        ck = {"compression": compression} if compression else {}
        _h5_replace(grp, "radial", self.radial, **ck)
        _h5_replace(grp, "intensity", self.intensity, **ck)
        if "sigma" in grp:
            del grp["sigma"]
        if self.sigma is not None:
            _h5_replace(grp, "sigma", self.sigma, **ck)
        grp.attrs["unit"] = self.unit

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> IntegrationResult1D:
        """
        Read from an HDF5 group written by :meth:`to_hdf5`.

        Parameters
        ----------
        grp : h5py.Group
            Source group.

        Returns
        -------
        IntegrationResult1D
        """
        radial = np.asarray(grp["radial"])
        intensity = np.asarray(grp["intensity"])
        sigma = np.asarray(grp["sigma"]) if "sigma" in grp else None
        unit = str(grp.attrs.get("unit", "2th_deg"))
        return cls(radial=radial, intensity=intensity, sigma=sigma, unit=unit)

    # ------------------------------------------------------------------
    # NeXus output
    # ------------------------------------------------------------------

    def to_nexus(
        self,
        grp: h5py.Group,
        signal_name: str = "intensity",
    ) -> None:
        """
        Write as an NXdata group with NeXus-compliant attributes.

        Layout::

            grp/          (NXdata)
              @signal = "<signal_name>"
              @axes = ["radial"]
              @radial_indices = [0]
              radial        @units, @long_name
              <signal_name> @long_name = "Intensity"
              sigma         @long_name = "Uncertainty"  (optional)

        Parameters
        ----------
        grp : h5py.Group
            Destination group (will be annotated as NXdata).
        signal_name : str, optional
            Dataset name for intensity (default ``"intensity"``).
        """
        grp.attrs["NX_class"] = "NXdata"
        grp.attrs["signal"] = signal_name
        grp.attrs["axes"] = ["radial"]
        grp.attrs["radial_indices"] = [0]

        nexus_units, long_name = _pyfai_unit_to_nexus(self.unit)

        for name in ("radial", signal_name, "sigma"):
            if name in grp:
                del grp[name]

        ds_r = grp.create_dataset("radial", data=self.radial)
        ds_r.attrs["units"] = nexus_units
        ds_r.attrs["long_name"] = long_name

        ds_i = grp.create_dataset(signal_name, data=self.intensity)
        ds_i.attrs["long_name"] = "Intensity"

        if self.sigma is not None:
            ds_s = grp.create_dataset("sigma", data=self.sigma)
            ds_s.attrs["long_name"] = "Uncertainty"


# ---------------------------------------------------------------------------
# IntegrationResult2D
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IntegrationResult2D:
    """
    Result of 2D integration (cake, GI reciprocal space map, polar map, etc).

    The ``unit`` / ``azimuthal_unit`` fields identify what each axis represents,
    making this container generic across all pyFAI integration modes:

    - Standard cake:   radial = q or 2theta,  azimuthal = chi
    - GI qip/qoop map: radial = qip,          azimuthal = qoop
    - GI polar map:    radial = qtotal,        azimuthal = chi_gi
    - GI exit angles:  radial = horiz_exit,    azimuthal = vert_exit

    Parameters
    ----------
    radial : ndarray
        1D radial axis.
    azimuthal : ndarray
        1D azimuthal axis.
    intensity : ndarray
        2D array of shape ``(len(radial), len(azimuthal))``.
    sigma : ndarray or None, optional
        Per-pixel uncertainty, same shape as ``intensity``.
    unit : str, optional
        pyFAI unit string for the radial axis.
    azimuthal_unit : str, optional
        pyFAI unit string for the azimuthal axis.
    """

    radial: np.ndarray
    azimuthal: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray | None = None
    unit: str = "2th_deg"
    azimuthal_unit: str = "chi_deg"

    def __post_init__(self) -> None:
        self.radial = np.asarray(self.radial, dtype=float)
        self.azimuthal = np.asarray(self.azimuthal, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.sigma is not None:
            self.sigma = np.asarray(self.sigma, dtype=float)
        if self.intensity.ndim != 2:
            raise ValueError("intensity must be a 2D array")
        nr, naz = len(self.radial), len(self.azimuthal)
        expected = (nr, naz)
        if self.intensity.shape != expected:
            raise ValueError(
                f"intensity shape {self.intensity.shape} != {expected} "
                f"(radial × azimuthal)"
            )
        if self.sigma is not None and self.sigma.shape != self.intensity.shape:
            raise ValueError(
                f"sigma shape {self.sigma.shape} != intensity shape "
                f"{self.intensity.shape}"
            )

    # ------------------------------------------------------------------
    # Unit conversion
    # ------------------------------------------------------------------

    def to_unit(
        self,
        target_unit: str,
        wavelength: float | None = None,
    ) -> IntegrationResult2D:
        """
        Convert the **radial** axis to a different unit.

        See :meth:`IntegrationResult1D.to_unit` for supported conversions
        and *wavelength* semantics.

        Parameters
        ----------
        target_unit : str
            Target pyFAI unit string.
        wavelength : float or None, optional
            Wavelength in Angstroms; required for 2theta ↔ q.

        Returns
        -------
        IntegrationResult2D
            New instance with converted radial axis; azimuthal axis unchanged.
        """
        new_radial = _convert_radial_axis(
            self.radial, self.unit, target_unit, wavelength
        )
        return IntegrationResult2D(
            radial=new_radial,
            azimuthal=self.azimuthal.copy(),
            intensity=self.intensity.copy(),
            sigma=self.sigma.copy() if self.sigma is not None else None,
            unit=target_unit,
            azimuthal_unit=self.azimuthal_unit,
        )

    def to_azimuthal_unit(self, target_unit: str) -> IntegrationResult2D:
        """
        Convert the **azimuthal** axis to a different unit.

        Supports ``*_deg`` ↔ ``*_rad`` conversions of the same axis
        (e.g. ``"chi_deg"`` ↔ ``"chi_rad"``).

        Parameters
        ----------
        target_unit : str
            Target unit string for the azimuthal axis.

        Returns
        -------
        IntegrationResult2D
            New instance with converted azimuthal axis; radial axis unchanged.
        """
        # Q-type azimuthal axes (GI: qoop_A^-1, etc.) use the radial converter
        # for Å^-1 ↔ nm^-1 scaling; angular axes use the angular converter.
        _Q_SFXS = ("_A^-1", "_nm^-1")
        if any(self.azimuthal_unit.endswith(s) for s in _Q_SFXS):
            new_azimuthal = _convert_radial_axis(
                self.azimuthal, self.azimuthal_unit, target_unit
            )
        else:
            new_azimuthal = _convert_angular_axis(
                self.azimuthal, self.azimuthal_unit, target_unit
            )
        return IntegrationResult2D(
            radial=self.radial.copy(),
            azimuthal=new_azimuthal,
            intensity=self.intensity.copy(),
            sigma=self.sigma.copy() if self.sigma is not None else None,
            unit=self.unit,
            azimuthal_unit=target_unit,
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def _check_compatible(self, other: IntegrationResult2D) -> None:
        if self.unit != other.unit:
            raise ValueError(
                f"Unit mismatch: '{self.unit}' vs '{other.unit}'."
            )
        if self.azimuthal_unit != other.azimuthal_unit:
            raise ValueError(
                f"Azimuthal unit mismatch: '{self.azimuthal_unit}' vs "
                f"'{other.azimuthal_unit}'."
            )
        if self.intensity.shape != other.intensity.shape:
            raise ValueError(
                f"Intensity shape mismatch: {self.intensity.shape} vs "
                f"{other.intensity.shape}."
            )

    def __add__(self, other: IntegrationResult2D) -> IntegrationResult2D:
        """Add intensities element-wise.  Propagates sigma if both have it."""
        self._check_compatible(other)
        return IntegrationResult2D(
            radial=self.radial.copy(),
            azimuthal=self.azimuthal.copy(),
            intensity=self.intensity + other.intensity,
            sigma=_sigma_add(self.sigma, other.sigma),
            unit=self.unit,
            azimuthal_unit=self.azimuthal_unit,
        )

    def __sub__(self, other: IntegrationResult2D) -> IntegrationResult2D:
        """Subtract intensities element-wise.  Propagates sigma if both have it."""
        self._check_compatible(other)
        return IntegrationResult2D(
            radial=self.radial.copy(),
            azimuthal=self.azimuthal.copy(),
            intensity=self.intensity - other.intensity,
            sigma=_sigma_add(self.sigma, other.sigma),
            unit=self.unit,
            azimuthal_unit=self.azimuthal_unit,
        )

    def __mul__(self, scalar: float) -> IntegrationResult2D:
        """Scale intensity (and sigma) by *scalar*."""
        s = float(scalar)
        new_sigma = abs(s) * self.sigma if self.sigma is not None else None
        return IntegrationResult2D(
            radial=self.radial.copy(),
            azimuthal=self.azimuthal.copy(),
            intensity=s * self.intensity,
            sigma=new_sigma,
            unit=self.unit,
            azimuthal_unit=self.azimuthal_unit,
        )

    def __rmul__(self, scalar: float) -> IntegrationResult2D:
        return self.__mul__(scalar)

    # ------------------------------------------------------------------
    # Line-cut extraction
    # ------------------------------------------------------------------

    def extract_1d(
        self,
        axis: str = "radial",
        index: int | None = None,
        range_: tuple[float, float] | None = None,
    ) -> IntegrationResult1D:
        """
        Extract a 1D line cut from the 2D result.

        Parameters
        ----------
        axis : {'radial', 'azimuthal'}
            Which output axis to produce:

            - ``'radial'``    → I(radial) by summing/slicing over azimuthal.
            - ``'azimuthal'`` → I(azimuthal) by summing/slicing over radial.

        index : int or None, optional
            If given, extract the single row/column at this position along
            the *other* axis.
        range_ : tuple of (float, float) or None, optional
            If given, sum all rows/columns whose axis value falls within
            ``[lo, hi]`` (inclusive).
        Neither index nor range_ given
            Sum over the entire other axis.

        Returns
        -------
        IntegrationResult1D

        Raises
        ------
        ValueError
            If *axis* is not ``'radial'`` or ``'azimuthal'``.
        """
        if axis == "radial":
            out_radial = self.radial.copy()
            out_unit = self.unit
            if index is not None:
                out_intensity = self.intensity[:, index]
                out_sigma = (
                    self.sigma[:, index] if self.sigma is not None else None
                )
            elif range_ is not None:
                lo, hi = range_
                mask = (self.azimuthal >= lo) & (self.azimuthal <= hi)
                out_intensity = self.intensity[:, mask].sum(axis=1)
                out_sigma = (
                    np.sqrt((self.sigma[:, mask] ** 2).sum(axis=1))
                    if self.sigma is not None
                    else None
                )
            else:
                out_intensity = self.intensity.sum(axis=1)
                out_sigma = (
                    np.sqrt((self.sigma ** 2).sum(axis=1))
                    if self.sigma is not None
                    else None
                )

        elif axis == "azimuthal":
            out_radial = self.azimuthal.copy()
            out_unit = self.azimuthal_unit
            if index is not None:
                out_intensity = self.intensity[index, :]
                out_sigma = (
                    self.sigma[index, :] if self.sigma is not None else None
                )
            elif range_ is not None:
                lo, hi = range_
                mask = (self.radial >= lo) & (self.radial <= hi)
                out_intensity = self.intensity[mask, :].sum(axis=0)
                out_sigma = (
                    np.sqrt((self.sigma[mask, :] ** 2).sum(axis=0))
                    if self.sigma is not None
                    else None
                )
            else:
                out_intensity = self.intensity.sum(axis=0)
                out_sigma = (
                    np.sqrt((self.sigma ** 2).sum(axis=0))
                    if self.sigma is not None
                    else None
                )

        else:
            raise ValueError(
                f"axis must be 'radial' or 'azimuthal', got '{axis}'."
            )

        return IntegrationResult1D(
            radial=out_radial,
            intensity=out_intensity,
            sigma=out_sigma,
            unit=out_unit,
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pyfai(
        cls,
        result: object,
        unit: str | None = None,
        azimuthal_unit: str | None = None,
    ) -> IntegrationResult2D:
        """
        Create from a pyFAI ``integrate2d`` result.

        Handles both standard ``AzimuthalIntegrator.integrate2d`` results
        (attributes ``.radial``, ``.azimuthal``) and ``FiberIntegrator``
        results (attributes ``.inplane``, ``.outofplane``).

        pyFAI returns ``intensity`` with shape ``(npt_azim, npt_rad)``;
        this factory transposes to ``(npt_rad, npt_azim)`` to match the
        project convention.

        Parameters
        ----------
        result : pyFAI integrate2d result
            Must have ``.intensity``, and either (``.radial``,
            ``.azimuthal``) or (``.inplane``, ``.outofplane``).
        unit : str or None, optional
            Override radial unit.
        azimuthal_unit : str or None, optional
            Override azimuthal unit.

        Returns
        -------
        IntegrationResult2D
        """
        # Axis arrays (standard vs FiberIntegrator result)
        if hasattr(result, "radial"):
            radial = np.asarray(result.radial, dtype=float)  # type: ignore[attr-defined]
            azimuthal = np.asarray(result.azimuthal, dtype=float)  # type: ignore[attr-defined]
        elif hasattr(result, "inplane"):
            radial = np.asarray(result.inplane, dtype=float)  # type: ignore[attr-defined]
            azimuthal = np.asarray(result.outofplane, dtype=float)  # type: ignore[attr-defined]
        else:
            raise ValueError(
                "Cannot parse pyFAI result: expected .radial/.azimuthal "
                "or .inplane/.outofplane attributes."
            )

        # Unit strings
        raw_unit = getattr(result, "unit", None)
        if unit is None:
            if isinstance(raw_unit, tuple):
                unit = str(raw_unit[0])
            elif raw_unit is not None:
                unit = str(raw_unit)
            elif hasattr(result, "ip_unit") and result.ip_unit is not None:  # type: ignore[attr-defined]
                unit = str(result.ip_unit)  # type: ignore[attr-defined]
            else:
                unit = "2th_deg"

        if azimuthal_unit is None:
            if isinstance(raw_unit, tuple) and len(raw_unit) > 1:
                azimuthal_unit = str(raw_unit[1])
            else:
                oop = getattr(result, "oop_unit", None)
                azimuthal_unit = str(oop) if oop is not None else "chi_deg"

        # pyFAI: (npt_azim, npt_rad) → transpose to (npt_rad, npt_azim)
        intensity = np.asarray(result.intensity, dtype=float).T  # type: ignore[attr-defined]
        sigma_raw = getattr(result, "sigma", None)
        sigma = (
            np.asarray(sigma_raw, dtype=float).T
            if sigma_raw is not None
            else None
        )

        return cls(
            radial=radial,
            azimuthal=azimuthal,
            intensity=intensity,
            sigma=sigma,
            unit=unit,
            azimuthal_unit=azimuthal_unit,
        )

    # ------------------------------------------------------------------
    # HDF5 I/O
    # ------------------------------------------------------------------

    def to_hdf5(self, grp: h5py.Group, compression: str = "lzf") -> None:
        """
        Write to an HDF5 group.

        Creates datasets ``radial``, ``azimuthal``, ``intensity``, and
        (if present) ``sigma``.  Stores ``unit`` and ``azimuthal_unit``
        as group attributes.

        Parameters
        ----------
        grp : h5py.Group
            Destination group.
        compression : str, optional
            HDF5 compression filter (default ``"lzf"``).
        """
        ck = {"compression": compression} if compression else {}
        _h5_replace(grp, "radial", self.radial, **ck)
        _h5_replace(grp, "azimuthal", self.azimuthal, **ck)
        chunks = (self.intensity.shape[0], min(self.intensity.shape[1], 64))
        _h5_replace(grp, "intensity", self.intensity, chunks=chunks, **ck)
        if "sigma" in grp:
            del grp["sigma"]
        if self.sigma is not None:
            _h5_replace(grp, "sigma", self.sigma, chunks=chunks, **ck)
        grp.attrs["unit"] = self.unit
        grp.attrs["azimuthal_unit"] = self.azimuthal_unit

    @classmethod
    def from_hdf5(cls, grp: h5py.Group) -> IntegrationResult2D:
        """
        Read from an HDF5 group written by :meth:`to_hdf5`.

        Parameters
        ----------
        grp : h5py.Group
            Source group.

        Returns
        -------
        IntegrationResult2D
        """
        radial = np.asarray(grp["radial"])
        azimuthal = np.asarray(grp["azimuthal"])
        intensity = np.asarray(grp["intensity"])
        sigma = np.asarray(grp["sigma"]) if "sigma" in grp else None
        unit = str(grp.attrs.get("unit", "2th_deg"))
        azimuthal_unit = str(grp.attrs.get("azimuthal_unit", "chi_deg"))
        return cls(
            radial=radial,
            azimuthal=azimuthal,
            intensity=intensity,
            sigma=sigma,
            unit=unit,
            azimuthal_unit=azimuthal_unit,
        )

    # ------------------------------------------------------------------
    # NeXus output
    # ------------------------------------------------------------------

    def to_nexus(
        self,
        grp: h5py.Group,
        signal_name: str = "intensity",
    ) -> None:
        """
        Write as an NXdata group with NeXus-compliant attributes.

        Layout::

            grp/           (NXdata)
              @signal = "<signal_name>"
              @axes = ["radial", "azimuthal"]
              @radial_indices = [0]
              @azimuthal_indices = [1]
              radial         @units, @long_name
              azimuthal      @units, @long_name
              <signal_name>  @long_name = "Intensity"
              sigma          @long_name = "Uncertainty"  (optional)

        Parameters
        ----------
        grp : h5py.Group
            Destination group (will be annotated as NXdata).
        signal_name : str, optional
            Dataset name for intensity (default ``"intensity"``).
        """
        grp.attrs["NX_class"] = "NXdata"
        grp.attrs["signal"] = signal_name
        grp.attrs["axes"] = ["radial", "azimuthal"]
        grp.attrs["radial_indices"] = [0]
        grp.attrs["azimuthal_indices"] = [1]

        r_units, r_long = _pyfai_unit_to_nexus(self.unit)
        az_units, az_long = _pyfai_unit_to_nexus(self.azimuthal_unit)

        for name in ("radial", "azimuthal", signal_name, "sigma"):
            if name in grp:
                del grp[name]

        ds_r = grp.create_dataset("radial", data=self.radial)
        ds_r.attrs["units"] = r_units
        ds_r.attrs["long_name"] = r_long

        ds_az = grp.create_dataset("azimuthal", data=self.azimuthal)
        ds_az.attrs["units"] = az_units
        ds_az.attrs["long_name"] = az_long

        ds_i = grp.create_dataset(signal_name, data=self.intensity)
        ds_i.attrs["long_name"] = "Intensity"

        if self.sigma is not None:
            ds_s = grp.create_dataset("sigma", data=self.sigma)
            ds_s.attrs["long_name"] = "Uncertainty"
