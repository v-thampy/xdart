"""Per-pixel q-space mapping for reciprocal-space mapping (RSM).

Bridges :class:`DiffractometerConfig` (xrayutilities convention + camera
orientation) and the new :class:`DetectorHeader` (beam center, pixel size,
distance, detector dimensions) into a single :class:`PixelQMap` that maps
per-frame angle arrays to per-pixel ``(qx, qy, qz)`` arrays.

This is the *per-pixel* geometry layer.  The *per-frame, scalar* layer
(motor → pyFAI rotation / GI incidence) lives in
:mod:`ssrl_xrd_tools.core.geometry.diffractometer`.

Typical usage::

    from ssrl_xrd_tools.core.geometry import (
        DiffractometerConfig, DetectorHeader, PixelQMap,
    )

    diff = DiffractometerConfig(sample_rot=("x+", "z-", "y+", "z-"),
                                detector_rot=("x+", "z-"),
                                init_area_detrot="x-",
                                init_area_tiltazimuth="z-")
    header = DetectorHeader(cch1=290, cch2=450,
                            pwidth1=0.075, pwidth2=0.075,
                            distance=830, Nch1=514, Nch2=1030)
    mapper = PixelQMap(diff, header)

    qx, qy, qz = mapper.pixel_q(angles, energy=11205, UB=UB)
    # qx, qy, qz each have shape (N_frame, 514, 1030)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from ssrl_xrd_tools.core.geometry.diffractometer import DiffractometerConfig


# ---------------------------------------------------------------------------
# DetectorHeader
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DetectorHeader:
    """Detector geometry parameters for ``xu.HXRD.Ang2Q.init_area``.

    Holds the per-detector quantities the xrayutilities area-detector
    Q-conversion needs.  The *camera orientation* (``init_area_detrot``,
    ``init_area_tiltazimuth``) lives on :class:`DiffractometerConfig`
    because it is a property of the diffractometer convention, not the
    detector itself.

    Units follow xrayutilities convention: ``pwidth1``/``pwidth2`` and
    ``distance`` are in **mm**; ``cch1``/``cch2`` and ``Nch1``/``Nch2``
    are in **pixels**.

    Attributes
    ----------
    cch1, cch2 : float
        Beam centre in detector pixel coordinates (axis 1, axis 2).
    pwidth1, pwidth2 : float
        Pixel pitch in mm.
    distance : float
        Sample–detector distance in mm.
    Nch1, Nch2 : int
        Detector size in pixels.  May be overridden at call time via
        ``with_image_shape`` or the ``image_shape=`` argument to
        :meth:`PixelQMap.pixel_q`.
    """

    cch1: float
    cch2: float
    pwidth1: float
    pwidth2: float
    distance: float
    Nch1: int
    Nch2: int

    def with_roi(self, roi: tuple[int, int, int, int]) -> "DetectorHeader":
        """Return a new header adjusted for a ``(r0, r1, c0, c1)`` ROI crop.

        ``r0``/``c0`` shift the beam centre; ``r1``/``c1`` set the new
        detector size.  Negative indices are interpreted Python-style
        relative to the original ``Nch1``/``Nch2``.
        """
        r0, r1, c0, c1 = roi
        new_n1 = (self.Nch1 + r1) - r0 if r1 < 0 else r1 - r0
        new_n2 = (self.Nch2 + c1) - c0 if c1 < 0 else c1 - c0
        return replace(
            self,
            cch1=self.cch1 - r0,
            cch2=self.cch2 - c0,
            Nch1=int(new_n1),
            Nch2=int(new_n2),
        )

    def with_image_shape(self, shape: tuple[int, ...]) -> "DetectorHeader":
        """Return a new header with ``Nch1``/``Nch2`` taken from an image shape.

        ``shape[-2:]`` is used, so 2D ``(H, W)`` and stacked ``(N, H, W)``
        shapes are both accepted.
        """
        if len(shape) < 2:
            raise ValueError(f"image shape must have >= 2 dims, got {shape!r}")
        return replace(self, Nch1=int(shape[-2]), Nch2=int(shape[-1]))

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "DetectorHeader":
        return cls(**json.loads(s))

    @classmethod
    def from_poni(
        cls,
        poni: Any,
        *,
        pixel1: float | None = None,
        pixel2: float | None = None,
        detector: str | None = None,
        image_shape: tuple[int, ...] | None = None,
    ) -> "DetectorHeader":
        """Build a :class:`DetectorHeader` from a pyFAI PONI calibration.

        Bridges the xdart-side detector calibration (PONI, in SI units) to
        the xrayutilities-side header (mm + pixel units).  Useful when
        plumbing an :class:`~xdart.modules.ewald.EwaldSphere` into the RSM
        pipeline — `arch.poni` plus the detector's pixel size are
        everything ``DetectorHeader`` needs apart from ``Nch1`` / ``Nch2``.

        Parameters
        ----------
        poni : PONI (or compatible)
            Anything exposing ``dist``, ``poni1``, ``poni2`` (in metres)
            and optionally ``detector`` (a pyFAI detector name).  The
            :class:`ssrl_xrd_tools.core.containers.PONI` dataclass fits.
        pixel1, pixel2 : float, optional
            Detector pixel pitch in **metres**.  Required if neither
            ``detector`` nor ``poni.detector`` resolves via pyFAI's
            registry.  If only one is given the other defaults to the
            same value.
        detector : str, optional
            pyFAI detector name (e.g. ``"Pilatus300k"``, ``"Eiger1M"``).
            Looked up via ``pyFAI.detectors.detector_factory`` to obtain
            pixel sizes when ``pixel1``/``pixel2`` are absent.  Overrides
            ``poni.detector`` when both are present.
        image_shape : tuple, optional
            ``shape[-2:]`` populates ``Nch1`` / ``Nch2``.  If omitted the
            header is returned with placeholder zeros — pass them at
            call time via :meth:`with_image_shape` or via the
            ``image_shape=`` argument to :meth:`PixelQMap.pixel_q`.

        Returns
        -------
        DetectorHeader

        Raises
        ------
        ValueError
            If pixel sizes cannot be resolved from any of the three
            sources (explicit args, ``detector=`` arg, ``poni.detector``).
        """
        # 1) Resolve pixel sizes ---------------------------------------
        if pixel1 is None or pixel2 is None:
            det_name = detector or getattr(poni, "detector", "") or ""
            if not det_name:
                raise ValueError(
                    "DetectorHeader.from_poni: pixel1/pixel2 are required "
                    "when neither detector= nor poni.detector is set."
                )
            try:
                import pyFAI.detectors as _det  # noqa: PLC0415 — lazy
                det = _det.detector_factory(det_name)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"DetectorHeader.from_poni: pyFAI cannot resolve "
                    f"detector {det_name!r}; supply pixel1/pixel2 "
                    f"explicitly (underlying error: {exc})"
                ) from exc
            if pixel1 is None:
                pixel1 = float(det.pixel1)
            if pixel2 is None:
                pixel2 = float(det.pixel2)
        pixel1 = float(pixel1)
        pixel2 = float(pixel2)
        if pixel1 <= 0 or pixel2 <= 0:
            raise ValueError(
                f"DetectorHeader.from_poni: pixel sizes must be > 0; "
                f"got pixel1={pixel1}, pixel2={pixel2}"
            )

        # 2) Detector size --------------------------------------------
        nch1 = nch2 = 0
        if image_shape is not None:
            if len(image_shape) < 2:
                raise ValueError(
                    f"image_shape must have >= 2 dims, got {image_shape!r}"
                )
            nch1, nch2 = int(image_shape[-2]), int(image_shape[-1])

        # 3) Build the header -----------------------------------------
        return cls(
            cch1=float(poni.poni1) / pixel1,
            cch2=float(poni.poni2) / pixel2,
            pwidth1=pixel1 * 1000.0,     # m → mm
            pwidth2=pixel2 * 1000.0,
            distance=float(poni.dist) * 1000.0,
            Nch1=nch1,
            Nch2=nch2,
        )


# ---------------------------------------------------------------------------
# PixelQMap
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PixelQMap:
    """Map per-frame angle arrays to per-pixel ``(qx, qy, qz)`` arrays.

    Bundles a :class:`DiffractometerConfig` (axis convention + camera
    orientation) and a :class:`DetectorHeader` (beam centre, pixel size,
    distance, detector size) into the minimal state needed to call
    ``xu.HXRD.Ang2Q.init_area`` + ``Ang2Q.area``.

    The mapper is intentionally pure geometry: it knows nothing about
    data sources (SPEC files, NeXus spheres, image files).  Pulling
    angles + energy from a v2 NeXus scan and feeding them in lives in
    :mod:`ssrl_xrd_tools.rsm.pipeline`.
    """

    diff_config: DiffractometerConfig
    header: DetectorHeader

    def pixel_q(
        self,
        angles: list[np.ndarray] | tuple[np.ndarray, ...],
        energy: float,
        *,
        UB: np.ndarray | None = None,
        roi: tuple[int, int, int, int] | None = None,
        image_shape: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-pixel ``(qx, qy, qz)`` arrays for a frame stack.

        Parameters
        ----------
        angles : list of array-like
            Per-frame angle arrays, ordered as ``xu.QConversion`` expects
            (sample axes first, then detector axes).  Each array is
            length-N where N is the number of frames.
        energy : float
            X-ray energy in eV.
        UB : (3, 3) ndarray, optional
            Sample orientation matrix.  ``None`` leaves the output in
            raw lab-frame q-space (equivalent to UB = identity).
        roi : (r0, r1, c0, c1), optional
            Crop ROI applied to the header before mapping.  Shifts the
            beam centre and reduces ``Nch1``/``Nch2`` accordingly.
        image_shape : tuple, optional
            Override ``Nch1``/``Nch2`` from an image array's actual
            shape (``shape[-2:]``).  Applied *after* ``roi``.  Useful
            when feeding a stack whose dimensions might not match the
            configured header (e.g. detector binning).

        Returns
        -------
        qx, qy, qz : ndarray
            Each shaped ``(N_frame, Nch1, Nch2)``.
        """
        # NOTE: ``xrayutilities`` is imported lazily inside
        # ``DiffractometerConfig.make_hxrd``; we don't import it here so
        # that header-validation errors fire before any import cost.
        header = self.header
        if roi is not None:
            header = header.with_roi(roi)
        if image_shape is not None:
            header = header.with_image_shape(image_shape)
        if header.Nch1 <= 0 or header.Nch2 <= 0:
            raise ValueError(
                f"Detector size not set (Nch1={header.Nch1}, "
                f"Nch2={header.Nch2}); pass image_shape= or fill in "
                f"DetectorHeader.Nch1/Nch2 explicitly."
            )

        hxrd = self.diff_config.make_hxrd(energy)
        hxrd.Ang2Q.init_area(
            self.diff_config.init_area_detrot,
            self.diff_config.init_area_tiltazimuth,
            cch1=float(header.cch1),
            cch2=float(header.cch2),
            pwidth1=float(header.pwidth1),
            pwidth2=float(header.pwidth2),
            distance=float(header.distance),
            Nch1=int(header.Nch1),
            Nch2=int(header.Nch2),
        )
        qx, qy, qz = hxrd.Ang2Q.area(
            *angles,
            UB=UB,
            **self.diff_config.ang2q_kwargs,
        )
        return qx, qy, qz


__all__ = [
    "DetectorHeader",
    "PixelQMap",
]
