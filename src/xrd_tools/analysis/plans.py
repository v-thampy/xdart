"""Typed headless analysis plans built on the public notebook APIs."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D, PONI
from xrd_tools.core.roi import RoiSpec, invalid_pixel_mask, roi_reduce
from xrd_tools.core.scan import FrameSource, MaskSpec
from xrd_tools.sources import ensure_frame_source

if TYPE_CHECKING:
    from xrd_tools.analysis.fitting import FitConfig, PhaseFitter


def _default_fit_config():
    from xrd_tools.analysis.fitting import FitConfig

    return FitConfig()


def stitch_images(*args: Any, **kwargs: Any) -> Any:
    from xrd_tools.integrate.multi import stitch_images as _impl

    return _impl(*args, **kwargs)


def process_scan_from_nexus(*args: Any, **kwargs: Any) -> Any:
    from xrd_tools.rsm.pipeline import process_scan_from_nexus as _impl

    return _impl(*args, **kwargs)


def grid_scans_streaming(*args: Any, **kwargs: Any) -> Any:
    from xrd_tools.rsm.pipeline import grid_scans_streaming as _impl

    return _impl(*args, **kwargs)


def fit_peaks(*args: Any, **kwargs: Any) -> Any:
    from xrd_tools.analysis.fitting import fit_peaks as _impl

    return _impl(*args, **kwargs)


def fit_sequence(*args: Any, **kwargs: Any) -> Any:
    from xrd_tools.analysis.fitting import fit_sequence as _impl

    return _impl(*args, **kwargs)


def sin2psi_analysis(*args: Any, **kwargs: Any) -> Any:
    from xrd_tools.analysis.strain import sin2psi_analysis as _impl

    return _impl(*args, **kwargs)


@dataclass(slots=True)
class AnalysisResult:
    """Small JSON-friendly envelope around an analysis payload."""

    kind: str
    payload: Any
    provenance: dict[str, Any] = field(default_factory=dict)
    #: per-frame source records for a grouped/single Stitch/RSM result — the
    #: raw-popup enabler the whole-result writer persists via
    #: ``write_stitched(frame_records=…)`` / ``write_rsm(frame_records=…)``.  Not
    #: provenance (it can carry binary thumbnails); excluded from to_dict/to_json.
    frame_records: list[dict[str, Any]] | None = field(
        default=None, repr=False, compare=False)

    def to_dict(self, *, include_payload: bool = True) -> dict[str, Any]:
        data = {
            "kind": self.kind,
            "payload_type": type(self.payload).__name__,
            "provenance": _json_safe(self.provenance),
        }
        if include_payload:
            data["payload"] = _json_safe(self.payload)
        return data

    def to_json(self, *, include_payload: bool = True, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(include_payload=include_payload), **kwargs)


@dataclass(frozen=True, slots=True)
class StitchPlan:
    """Plan for MultiGeometry stitching of a frame source."""

    base_poni: PONI | None = None
    #: a calibrated/preset Diffractometer (closes GAP A: per-frame rotations from
    #: its fitted scales, not a deg2rad hardwire). When set, it drives the stitch
    #: geometry; the rot1_key/rot2_key path below is the no-calibration fallback.
    diffractometer: Any = None
    #: merge engine: "multigeometry" (pyFAI MG), "pyfai_hist" (pyFAI q/chi maps →
    #: the streaming Σraw/Σnorm histogram), or "xu_hist" (xu q-provider; not built
    #: yet). Histogram backends require the geometry path (a Diffractometer).
    backend: str = "multigeometry"
    #: per-pixel CorrectionStack for the histogram backends (solid-angle /
    #: polarization …); None = unit weight. Ignored by the multigeometry backend
    #: (which carries pyFAI's own correctSolidAngle + polarization_factor).
    corrections: Any = None
    #: grazing-incidence settings (a GISettings: the GICorrectionStack +
    #: incident_angle_deg / sample_orientation / tilt). pyfai_hist backend only.
    #: αi from gi.incident_angle_deg, else the Diffractometer's incident_angle
    #: mapping. Absolute GI correctness is pending real-data validation (grazing.py).
    gi: Any = None
    rot1_key: str = "rot1"
    rot2_key: str | None = None
    monitor_key: str | None = None
    mode: str = "1d"
    npt_1d: int = 2000
    npt_rad_2d: int = 1500
    npt_azim_2d: int = 720
    unit: str = "q_A^-1"
    method: str = "BBox"
    radial_range: tuple[float, float] | None = None
    azimuth_range: tuple[float, float] | None = None
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    max_eager_bytes: int | None = 2 * 1024 * 1024 * 1024
    extra: dict[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        """A JSON-safe record of this stitch for persistence (pass to
        ``write_stitched(provenance=…)``).

        The merge parameters + the applied corrections — NOT the binary ``mask``
        nor the full Diffractometer blob (that round-trips separately under
        ``/entry/diffractometer``; only its preset/convention tag is noted here).
        """
        def _ser(obj):
            to_dict = getattr(obj, "to_dict", None)
            return to_dict() if callable(to_dict) else None

        prov: dict[str, Any] = {
            "backend": self.backend,
            "mode": self.mode,
            "unit": self.unit,
            "npt_1d": self.npt_1d,
            "npt_rad_2d": self.npt_rad_2d,
            "npt_azim_2d": self.npt_azim_2d,
            "radial_range": list(self.radial_range) if self.radial_range else None,
            "azimuth_range": (list(self.azimuth_range)
                              if self.azimuth_range else None),
            "monitor_key": self.monitor_key,
            "corrections": _ser(self.corrections),
            "gi": _ser(self.gi),
        }
        diff = self.diffractometer
        if diff is not None:
            prov["diffractometer"] = (getattr(diff, "preset", None)
                                      or getattr(diff, "convention", None))
        return prov

    @classmethod
    def from_provenance(
        cls,
        prov: "dict[str, Any]",
        *,
        diffractometer: Any = None,
        base_poni: "PONI | None" = None,
        mask: "np.ndarray | None" = None,
    ) -> "StitchPlan":
        """Rebuild a StitchPlan from a :meth:`provenance` dict — the reload half of
        the round-trip (e.g. to repopulate the GUI processing panel from a saved
        ``.nxs``).

        Only the **processing options** round-trip through provenance; the
        geometry (``diffractometer``/``base_poni``) + the binary ``mask`` are NOT
        in it (they persist separately, e.g. under ``/entry/diffractometer``) —
        pass them in to reattach.
        """
        from xrd_tools.corrections.grazing import GISettings  # noqa: PLC0415
        from xrd_tools.corrections.stack import CorrectionStack  # noqa: PLC0415
        corr = prov.get("corrections")
        gi = prov.get("gi")
        rr = prov.get("radial_range")
        ar = prov.get("azimuth_range")
        return cls(
            diffractometer=diffractometer,
            base_poni=base_poni,
            mask=mask,
            backend=prov.get("backend", "multigeometry"),
            mode=prov.get("mode", "1d"),
            unit=prov.get("unit", "q_A^-1"),
            npt_1d=int(prov.get("npt_1d", 2000)),
            npt_rad_2d=int(prov.get("npt_rad_2d", 1500)),
            npt_azim_2d=int(prov.get("npt_azim_2d", 720)),
            radial_range=tuple(rr) if rr else None,
            azimuth_range=tuple(ar) if ar else None,
            monitor_key=prov.get("monitor_key"),
            corrections=CorrectionStack.from_dict(corr) if corr else None,
            gi=GISettings.from_dict(gi) if gi else None,
        )


def _harvest_frame_records(source, scan_labels, *, selected_labels=None):
    """Fail-soft wrapper around :func:`harvest_frame_records` — the per-frame raw
    pointers are an enhancement (the raw popup), never core to the reduction, so a
    harvest failure logs and yields ``None`` rather than killing the run."""
    try:
        from xrd_tools.io.nexus_record import harvest_frame_records  # noqa: PLC0415
        return harvest_frame_records(source, scan_labels=scan_labels,
                                     selected_labels=selected_labels)
    except Exception:  # noqa: BLE001
        logger.debug("frame-record harvest failed; raw popup will be unavailable",
                     exc_info=True)
        return None


def run_stitch(
    plan: StitchPlan,
    source: FrameSource | Sequence[FrameSource],
    *,
    frame_indices: Sequence[int] | None = None,
    scan_labels: Sequence[Any] | None = None,
) -> AnalysisResult:
    """Run MultiGeometry stitching over one headless frame source or a GROUP.

    A group (a sequence of sources — a multi-scan stitch) is concatenated into one
    :class:`~xrd_tools.sources.composite.CompositeFrameSource` and merged as a
    single output, frames re-indexed ``0..N-1`` (the documented grouping path).
    ``scan_labels`` tags the grouped contributing frames for the raw-popup records
    (default ``1..N``; pass the real scan numbers, e.g. ``[5, 7, 8]``).
    """

    # A group (sequence of sources) → one CompositeFrameSource; a single source
    # passes through unchanged.  Harvest reads ``src`` directly: a composite
    # exposes both its members (for scan-tagging) and its ``_map`` (to translate a
    # subselected GLOBAL frame back to the right member+local label).
    grouped = isinstance(source, Sequence) and not isinstance(source, (str, bytes))
    if grouped:
        from xrd_tools.sources.composite import CompositeFrameSource  # noqa: PLC0415
        members = [ensure_frame_source(s) for s in source]
        if not members:
            raise ValueError("run_stitch received an empty source group")
        src = CompositeFrameSource(members) if len(members) > 1 else members[0]
    else:
        src = ensure_frame_source(source)
    labels = [int(i) for i in (frame_indices or src.frame_indices)]
    if not labels:
        raise ValueError("run_stitch requires at least one frame")
    # The multigeometry backend uses pyFAI's OWN correctSolidAngle/polarization,
    # NOT the shared CorrectionStack/GISettings — surface the silent drop so a
    # caller (and the GUI) doesn't believe it applied the shared pre-weight.
    if plan.backend == "multigeometry" and (
            plan.corrections is not None or plan.gi is not None):
        logger.warning(
            "StitchPlan.corrections/gi are IGNORED by the 'multigeometry' backend "
            "(it uses pyFAI's own correctSolidAngle/polarization). Use the "
            "'pyfai_hist' backend to apply the shared CorrectionStack/GI weight.")
    base_poni = plan.base_poni or getattr(src, "poni", None)
    # base_poni is required UNLESS a calibrated Diffractometer supplies the base
    # geometry itself (its DetectorCalibration carries dist/poni/Detector_config).
    _diff = plan.diffractometer or getattr(src, "diffractometer", None)
    if base_poni is None and getattr(_diff, "calibration", None) is None:
        raise ValueError(
            "StitchPlan.base_poni or source.poni is required (or a Diffractometer "
            "carrying a DetectorCalibration)")

    images: list[np.ndarray] = []
    eager_bytes = 0
    for label in labels:
        image = np.asarray(src.load_frame(label), dtype=float)
        eager_bytes += int(image.nbytes)
        if plan.max_eager_bytes is not None and eager_bytes > int(plan.max_eager_bytes):
            raise MemoryError(
                "run_stitch currently materializes all selected images before "
                "calling pyFAI MultiGeometry. Selected frames require at least "
                f"{eager_bytes / (1024 ** 3):.2f} GiB, exceeding "
                f"StitchPlan.max_eager_bytes={plan.max_eager_bytes}. "
                "Use fewer frames, raise the limit intentionally, or migrate "
                "this call to the future streaming StitchPlan backend."
            )
        images.append(image)
    normalization = (
        _metadata_series(src, labels, plan.monitor_key)
        if plan.monitor_key is not None else None
    )
    if normalization is not None and not np.all(np.isfinite(normalization)):
        # Same multi-source footgun as the geometry path: a composite NaN-pads the
        # monitor for a member that lacks it → NaN normalization → those frames
        # silently vanish from the merge. Fail loud instead.
        bad = np.flatnonzero(~np.isfinite(np.asarray(normalization))).tolist()
        raise ValueError(
            f"stitch monitor {plan.monitor_key!r} has non-finite value(s) at frame "
            f"position(s) {bad[:10]}{' …' if len(bad) > 10 else ''} — a "
            f"grouped/composite source is NaN-padding a member that lacks this "
            f"monitor. Every member of a multi-source stitch must provide it.")

    diffractometer = plan.diffractometer or getattr(src, "diffractometer", None)
    if diffractometer is not None:
        # GAP-A path: per-frame rotations from the calibrated Diffractometer.
        from xrd_tools.integrate.multi import (  # noqa: PLC0415
            create_multigeometry_integrators_from_geometry,
            stitch_1d, stitch_2d,
        )
        motors: dict[str, np.ndarray] = {}
        for m in diffractometer.all_referenced_motors():
            try:
                motors[m] = _metadata_series(src, labels, m)
            except Exception:  # noqa: BLE001 — motor not in this source's metadata
                continue
        base_cal = getattr(diffractometer, "calibration", None)
        if base_cal is not None and "detector_config" in plan.extra:
            raise ValueError(
                "ambiguous detector_config: the Diffractometer already carries a "
                "DetectorCalibration (with its own Detector_config); remove "
                "detector_config from StitchPlan.extra")
        if base_cal is None:
            from xrd_tools.core.geometry import DetectorCalibration  # noqa: PLC0415
            base_cal = DetectorCalibration(
                poni=base_poni, detector_config=dict(plan.extra.get("detector_config", {})))
        integrators = create_multigeometry_integrators_from_geometry(
            diffractometer, motors, base_calibration=base_cal)
        extra = {k: v for k, v in plan.extra.items() if k != "detector_config"}
        if plan.gi is not None and plan.backend != "pyfai_hist":
            raise ValueError(
                "GI stitching (StitchPlan.gi) is only available on the 'pyfai_hist' "
                f"backend; got backend={plan.backend!r}.")
        if plan.backend == "xu_hist":
            raise NotImplementedError(
                "the 'xu_hist' stitch backend (xu Ang2Q q-provider) is not built "
                "yet; use 'multigeometry' or 'pyfai_hist'")
        if plan.backend == "pyfai_hist":
            from xrd_tools.integrate.stitch_hist import (  # noqa: PLC0415
                pyfai_gi_q_frames, pyfai_q_frames, stitch_q_grid)
            # the pyfai_hist provider emits |q| in Å⁻¹ ONLY — a non-q unit would
            # silently mislabel a q-axis as 2θ/r (fail loud instead).
            if plan.unit != "q_A^-1":
                raise ValueError(
                    f"the 'pyfai_hist' stitch backend only emits q in Å⁻¹ "
                    f"(unit='q_A^-1'); got unit={plan.unit!r}. Use the "
                    "'multigeometry' backend for other units.")
            # the histogram merge takes its parameters directly from the plan; it
            # cannot forward pyFAI integrate kwargs. Surface anything it would drop.
            if extra:
                raise ValueError(
                    f"the 'pyfai_hist' stitch backend cannot consume pyFAI "
                    f"integrate kwargs {sorted(extra)} (StitchPlan.extra); they "
                    "apply only to the 'multigeometry' backend.")
            if plan.method != "BBox":
                logger.warning(
                    "StitchPlan.method=%r is ignored by the 'pyfai_hist' backend "
                    "(the histogram merge does its own pixel splitting).",
                    plan.method)
            # 2D uses the 2D radial bin count; 1D uses the 1D count.
            npt_rad = plan.npt_rad_2d if plan.mode == "2d" else plan.npt_1d
            if plan.gi is not None:
                # The GI optical constants use gi.corrections.energy_eV — warn if
                # it diverges from the canonical calibration wavelength (one
                # energy, the calibration's, is the source of truth).
                _gc = getattr(plan.gi, "corrections", None)
                _cal = getattr(diffractometer, "calibration", None)
                _wl = getattr(getattr(_cal, "poni", None), "wavelength", None)
                if _gc is not None and _wl:
                    from xrd_tools.core.energy import (  # noqa: PLC0415
                        check_energy_consistency, wavelength_m_to_energy_eV)
                    check_energy_consistency(
                        getattr(_gc, "energy_eV", None),
                        wavelength_m_to_energy_eV(_wl),
                        what_a="GICorrectionStack.energy_eV",
                        what_b="calibration wavelength")
                # per-frame αi: explicit fixed angle (gi.incident_angle_deg), else
                # the Diffractometer's incident_angle mapping (degrees).
                if plan.gi.incident_angle_deg is not None:
                    inc = np.full(len(images), float(plan.gi.incident_angle_deg))
                else:
                    per = diffractometer.to_pyfai_per_frame(motors)
                    inc = np.atleast_1d(np.asarray(per["incident_angle"], dtype=float))
                    if inc.shape[0] == 1 and len(images) > 1:
                        inc = np.full(len(images), float(inc[0]))
                    if not np.any(inc != 0.0):
                        raise ValueError(
                            "GI stitching: the incident angle is 0 for every frame — "
                            "set GISettings.incident_angle_deg or activate the "
                            "Diffractometer's incident_angle mapping (an incidence "
                            "motor in the scan).")
                frames = pyfai_gi_q_frames(
                    images, integrators, gi=plan.gi.corrections,
                    incident_angles_deg=inc,
                    sample_orientation=plan.gi.sample_orientation,
                    tilt_deg=plan.gi.tilt_deg, corrections=plan.corrections,
                    mask=plan.mask, normalization=normalization)
            else:
                frames = pyfai_q_frames(
                    images, integrators, corrections=plan.corrections,
                    mask=plan.mask, normalization=normalization)
            payload = stitch_q_grid(
                frames,
                mode=plan.mode, npt=npt_rad, npt_azim=plan.npt_azim_2d,
                unit=plan.unit, radial_range=plan.radial_range,
                azimuth_range=plan.azimuth_range)
        elif plan.mode == "2d":
            payload = stitch_2d(
                images, integrators, npt_rad=plan.npt_rad_2d,
                npt_azim=plan.npt_azim_2d, unit=plan.unit, method=plan.method,
                radial_range=plan.radial_range, azimuth_range=plan.azimuth_range,
                mask=plan.mask, normalization=normalization, **extra)
        else:
            payload = stitch_1d(
                images, integrators, npt=plan.npt_1d, unit=plan.unit,
                method=plan.method, radial_range=plan.radial_range,
                mask=plan.mask, normalization=normalization, **extra)
    else:
        if plan.gi is not None:
            raise ValueError(
                "GI stitching (StitchPlan.gi) requires the geometry path — set "
                "StitchPlan.diffractometer (the per-frame incident angle αi is read "
                "from its incident_angle mapping).")
        rot1 = _metadata_series(src, labels, plan.rot1_key)
        rot2 = (
            _metadata_series(src, labels, plan.rot2_key)
            if plan.rot2_key is not None else None
        )
        payload = stitch_images(
            images,
            base_poni,
            rot1_angles=rot1,
            rot2_angles=rot2,
            mode=plan.mode,
            npt_1d=plan.npt_1d,
            npt_rad_2d=plan.npt_rad_2d,
            npt_azim_2d=plan.npt_azim_2d,
            unit=plan.unit,
            method=plan.method,
            radial_range=plan.radial_range,
            azimuth_range=plan.azimuth_range,
            mask=plan.mask,
            normalization=normalization,
            **plan.extra,
        )
    return AnalysisResult(
        kind="stitch",
        payload=payload,
        provenance={
            "plan": _plan_dict(plan),
            "source": getattr(src, "name", type(src).__name__),
            "frame_indices": labels,
        },
        # only the frames that actually contributed (the subselected `labels`),
        # in src's own label space (composite GLOBAL index, else the source's own)
        frame_records=_harvest_frame_records(
            src, scan_labels,
            selected_labels=(None if frame_indices is None else labels)),
    )


@dataclass(frozen=True, slots=True)
class RSMPlan:
    """Plan for streaming reciprocal-space map gridding."""

    mapper: Any = field(repr=False, compare=False)
    diff_motors: tuple[str, ...] = ()
    bins: tuple[int, int, int] = (101, 101, 101)
    UB: np.ndarray | None = field(default=None, repr=False, compare=False)
    energy: float | None = None
    chunk_size: int = 8
    q_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] | None = None
    roi: tuple[int, int, int, int] | None = None
    static_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    scout_pad: float = 0.0
    #: per-pixel CorrectionStack (solid-angle / polarization …) folded into the
    #: Σ(raw·w)/Σ(w) grid as the SAME weight stitching uses; None = unit weight
    #: (the count-mean). Geometry-static (fixed lab detector).
    corrections: Any = None
    #: grazing-incidence settings (a GISettings) — adds the GI intensity weight
    #: (footprint/Fresnel/absorption) to the grid. Requires gi.incident_angle_deg
    #: (a FIXED αi). Refraction + per-frame-varying αi + the absolute GI signs are
    #: the real-data-gated tail (see rsm/corrections.gi_grid_weight).
    gi: Any = None

    def provenance(self) -> dict[str, Any]:
        """A JSON-safe record of this RSM run for persistence (pass to
        ``write_rsm(provenance=…)``).

        The grid parameters + the applied corrections — NOT the binary mask/UB
        nor the PixelQMap; the diffractometer geometry round-trips separately
        under ``/entry/diffractometer`` (its preset tag is noted here).
        """
        def _ser(obj):
            to_dict = getattr(obj, "to_dict", None)
            return to_dict() if callable(to_dict) else None

        diff = getattr(self.mapper, "diff_config", None)
        return {
            "kind": "rsm",
            "bins": list(self.bins),
            "diff_motors": list(self.diff_motors),
            "energy": self.energy,
            "q_bounds": ([list(r) for r in self.q_bounds]
                         if self.q_bounds is not None else None),
            "roi": list(self.roi) if self.roi is not None else None,
            "chunk_size": self.chunk_size,
            "corrections": _ser(self.corrections),
            "gi": _ser(self.gi),
            "diffractometer": (getattr(diff, "preset", None)
                               or getattr(diff, "convention", None)),
        }

    @classmethod
    def from_provenance(
        cls,
        prov: "dict[str, Any]",
        *,
        mapper: Any = None,
        UB: "np.ndarray | None" = None,
        static_mask: "np.ndarray | None" = None,
    ) -> "RSMPlan":
        """Rebuild an RSMPlan from a :meth:`provenance` dict — the reload half of
        the round-trip.  The ``mapper`` (PixelQMap/geometry), ``UB``, and the
        binary ``static_mask`` are NOT in provenance (they come from the scan /
        the persisted geometry) — pass them in to reattach.
        """
        from xrd_tools.corrections.grazing import GISettings  # noqa: PLC0415
        from xrd_tools.corrections.stack import CorrectionStack  # noqa: PLC0415
        corr = prov.get("corrections")
        gi = prov.get("gi")
        qb = prov.get("q_bounds")
        roi = prov.get("roi")
        return cls(
            mapper=mapper,
            UB=UB,
            static_mask=static_mask,
            bins=tuple(prov.get("bins", (101, 101, 101))),
            diff_motors=tuple(prov.get("diff_motors", ())),
            energy=prov.get("energy"),
            q_bounds=(tuple(tuple(r) for r in qb) if qb is not None else None),
            roi=tuple(roi) if roi else None,
            chunk_size=int(prov.get("chunk_size", 8)),
            corrections=CorrectionStack.from_dict(corr) if corr else None,
            gi=GISettings.from_dict(gi) if gi else None,
        )


def run_rsm(
    plan: RSMPlan,
    source: FrameSource | Sequence[FrameSource],
    *,
    scan_labels: Sequence[Any] | None = None,
) -> AnalysisResult:
    """Run the streaming RSM pipeline for one source or a list of sources.

    ``scan_labels`` tags grouped contributing frames for the raw-popup records
    (default ``1..N``; pass the real scan numbers, e.g. ``[5, 7, 8]``).
    """

    grouped = isinstance(source, Sequence) and not isinstance(source, (str, bytes))
    if grouped:
        from xrd_tools.rsm.pipeline import ScanInput

        members = [ensure_frame_source(scan) for scan in source]
        inputs = [
            ScanInput(scan=m, energy=plan.energy, UB=plan.UB, roi=plan.roi)
            for m in members
        ]
        payload = grid_scans_streaming(
            plan.mapper,
            inputs,
            plan.diff_motors,
            plan.bins,
            chunk_size=plan.chunk_size,
            q_bounds=plan.q_bounds,
            static_mask=plan.static_mask,
            scout_pad=plan.scout_pad,
            corrections=plan.corrections,
            gi=plan.gi,
        )
        n_sources = len(inputs)
        _harvest = members
    else:
        src = ensure_frame_source(source)  # type: ignore[arg-type]
        _harvest = src
        payload = process_scan_from_nexus(
            src,
            plan.mapper,
            plan.diff_motors,
            plan.bins,
            UB=plan.UB,
            energy=plan.energy,
            chunk_size=plan.chunk_size,
            q_bounds=plan.q_bounds,
            roi=plan.roi,
            static_mask=plan.static_mask,
            scout_pad=plan.scout_pad,
            corrections=plan.corrections,
            gi=plan.gi,
        )
        n_sources = 1
    return AnalysisResult(
        kind="rsm",
        payload=payload,
        provenance={"plan": _plan_dict(plan), "n_sources": n_sources},
        frame_records=_harvest_frame_records(_harvest, scan_labels),
    )


@dataclass(frozen=True, slots=True)
class PeakFitPlan:
    positions: tuple[float, ...] | None = None
    model: str = "pseudovoigt"
    n_peaks: int | None = None
    background: str = "linear"
    sigma_init: float | Sequence[float] | None = None
    sigma_bounds: tuple[float, float] | None = None
    amplitude_init: float | Sequence[float] | None = None
    amplitude_bounds: tuple[float, float] | None = None
    center_bounds_delta: float | None = None
    fraction_init: float = 0.5
    fit_kwargs: dict[str, Any] = field(default_factory=dict)


def run_peak_fit(plan: PeakFitPlan, x: np.ndarray, y: np.ndarray) -> AnalysisResult:
    payload = fit_peaks(
        x,
        y,
        positions=None if plan.positions is None else list(plan.positions),
        model=plan.model,
        n_peaks=plan.n_peaks,
        background=plan.background,
        sigma_init=plan.sigma_init,
        sigma_bounds=plan.sigma_bounds,
        amplitude_init=plan.amplitude_init,
        amplitude_bounds=plan.amplitude_bounds,
        center_bounds_delta=plan.center_bounds_delta,
        fraction_init=plan.fraction_init,
        **plan.fit_kwargs,
    )
    return AnalysisResult(kind="peak_fit", payload=payload, provenance={"plan": _plan_dict(plan)})


@dataclass(frozen=True, slots=True)
class PhaseFitPlan:
    """Plan wrapper for phase-aware fitting."""

    config: "FitConfig" = field(default_factory=_default_fit_config)
    sequential: bool = False


def run_phase_fit(
    plan: PhaseFitPlan,
    patterns: Sequence[
        tuple[np.ndarray, np.ndarray]
        | tuple[np.ndarray, np.ndarray, np.ndarray | None]
        | IntegrationResult1D
    ],
    phases: list[Any],
    *,
    labels: Sequence[str] | None = None,
    progress_callback=None,
    fit_background_template: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> AnalysisResult:
    normalized = [
        (p.radial, p.intensity, p.sigma)
        if isinstance(p, IntegrationResult1D) else p
        for p in patterns
    ]
    payload = fit_sequence(
        normalized,
        phases,
        plan.config,
        sequential=plan.sequential,
        labels=labels,
        progress_callback=progress_callback,
        fit_background_template=fit_background_template,
    )
    return AnalysisResult(kind="phase_fit", payload=payload, provenance={"plan": _plan_dict(plan)})


@dataclass(frozen=True, slots=True)
class Sin2PsiPlan:
    q_range: tuple[float, float]
    chi_centers: tuple[float, ...] | None = None
    chi_width: float = 5.0
    n_sectors: int | None = None
    chi_range: tuple[float, float] | None = None
    model: str = "pseudovoigt"
    background: str = "linear"
    sigma_init: float | None = None
    sigma_bounds: tuple[float, float] | None = None
    center_bounds_delta: float | None = None
    E: float | None = None
    nu: float | None = None


def run_sin2psi(plan: Sin2PsiPlan, result2d: IntegrationResult2D) -> AnalysisResult:
    payload = sin2psi_analysis(
        result2d,
        q_range=plan.q_range,
        chi_centers=None if plan.chi_centers is None else list(plan.chi_centers),
        chi_width=plan.chi_width,
        n_sectors=plan.n_sectors,
        chi_range=plan.chi_range,
        model=plan.model,
        background=plan.background,
        sigma_init=plan.sigma_init,
        sigma_bounds=plan.sigma_bounds,
        center_bounds_delta=plan.center_bounds_delta,
        E=plan.E,
        nu=plan.nu,
    )
    return AnalysisResult(kind="sin2psi", payload=payload, provenance={"plan": _plan_dict(plan)})


def make_phase_fitter(
    result: IntegrationResult1D | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray | None],
    **kwargs: Any,
) -> PhaseFitter:
    """Convenience factory that accepts either an IntegrationResult1D or arrays."""

    from xrd_tools.analysis.fitting import PhaseFitter

    if isinstance(result, IntegrationResult1D):
        return PhaseFitter(result.radial, result.intensity, result.sigma, **kwargs)
    return PhaseFitter(*result, **kwargs)


def _metadata_series(source: FrameSource, labels: Sequence[int], key: str | None) -> np.ndarray:
    if key is None:
        raise ValueError("metadata key must not be None")
    motors = getattr(source, "motors", None)
    if motors and key in motors:
        values = np.asarray(motors[key], dtype=float)
        if values.shape[0] == len(getattr(source, "frame_indices")):
            row_of = {int(label): i for i, label in enumerate(source.frame_indices)}
            return np.asarray([values[row_of[int(label)]] for label in labels], dtype=float)
    frame_for = getattr(source, "frame_for", None)
    metadata_for = getattr(source, "metadata_for", None)
    out: list[float] = []
    for label in labels:
        metadata = None
        if callable(metadata_for):
            metadata = metadata_for(int(label))
        elif callable(frame_for):
            metadata = frame_for(int(label)).metadata
        if metadata is None:
            metadata = {}
        value = _metadata_get_case_insensitive(metadata, key)
        if value is None:
            raise KeyError(f"frame {label} has no metadata key {key!r}")
        out.append(float(value))
    return np.asarray(out, dtype=float)


def _metadata_get_case_insensitive(metadata: Any, key: str) -> Any:
    if key in metadata:
        return metadata[key]
    key_lower = str(key).lower()
    for candidate, value in dict(metadata).items():
        if str(candidate).lower() == key_lower:
            return value
    return None


def _plan_dict(plan: Any) -> dict[str, Any]:
    if is_dataclass(plan):
        return _json_safe(asdict(plan))
    return _json_safe(plan)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


# ---------------------------------------------------------------------------
# ROI statistics (rectangular ROIs reduced over a scan's RAW frames)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoiStatsPlan:
    """Reduce one or more :class:`RoiSpec` rectangles over a scan's raw frames.

    ``background`` (optional) is combined with each signal ROI per
    ``background_op`` off the background *density* (mean per valid bg pixel), so
    both ``"subtract"`` and ``"divide"`` stay area-scaled for the ``sum`` reducer.
    ``x_key`` picks a scan_data/metadata column for the x-axis (else frame
    label).  ``mask`` is a static detector mask (beamstop / bad module,
    :class:`~xrd_tools.core.scan.MaskSpec` or a bool/flat-index array) excluded
    from every ROI — the SAME mask the reducer applies, so ROI stats and the
    reduction agree (§6.3).  ``mask_saturation`` opts into the dtype-ceiling
    saturation mask (the uint32 dummy + non-finite are always excluded)."""

    rois: tuple[RoiSpec, ...] = ()
    background: RoiSpec | None = None
    background_op: str = "subtract"        # "subtract" | "divide"
    reducer: str = "mean"                  # mean | sum | max | min | std
    x_key: str | None = None
    mask: Any = None                       # static detector MaskSpec / array
    mask_saturation: bool = False
    frame_indices: tuple[int, ...] | None = None


@dataclass
class RoiStatsResult:
    """Per-frame ROI series — one ``{frame -> value}`` column per ROI."""

    x: np.ndarray
    x_label: str
    series: dict[str, np.ndarray]          # roi name -> stat per frame
    frames: np.ndarray
    valid_counts: dict[str, np.ndarray]    # roi name -> valid-pixel count per frame
    diagnostics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RoiSignal:
    """One signal ROI carrying its OWN reducer + optional paired background.

    The per-ROI generalization of :class:`RoiStatsPlan` (which shares a single
    reducer/background across every ROI): each signal owns its column ``name``,
    its ``reducer``, and an optional ``background`` ROI combined per
    ``background_op`` off the background density (``"subtract"`` and ``"divide"``
    are both area-scaled for ``sum``).  Drive a sequence of these with
    :func:`run_roi_signals`."""

    roi: RoiSpec
    reducer: str = "mean"                  # mean | sum | max | min | std
    background: RoiSpec | None = None
    background_op: str = "subtract"        # "subtract" | "divide"
    name: str = ""


def _combine_background(value, n_valid, reducer, background_op, bkg_mean):
    """Combine a signal ROI value with its background, both ops keyed off the
    background **density** ``bkg_mean`` (mean per valid bg pixel) so they are
    area-consistent: for the ``sum`` reducer the background is scaled by the
    signal ROI's valid-pixel count (a small bg box and a large one give the same
    per-pixel correction).  ``subtract`` → ``value - scaled``; ``divide`` →
    ``value / scaled``."""
    if bkg_mean is None or not np.isfinite(bkg_mean):
        # No usable background: subtract is a no-op; divide is undefined.
        return value if background_op == "subtract" else float("nan")
    scaled = bkg_mean * n_valid if reducer == "sum" else bkg_mean
    if background_op == "divide":
        return value / scaled if scaled != 0 else float("nan")
    return value - scaled


def _dedup_names(names):
    """Disambiguate colliding resolved ROI names (append ``_2``, ``_3``, …) so
    the per-ROI series dict never collapses two signals onto one key — which
    would interleave their values into one wrong-length array.  Mirrors the
    GUI's ``_unique_col_name`` so a headless caller is protected too."""
    seen = set()
    out = []
    for name in names:
        candidate = name
        k = 2
        while candidate in seen:
            candidate = f"{name}_{k}"
            k += 1
        seen.add(candidate)
        out.append(candidate)
    return out


def _resolve_static_mask(mask, shape):
    """Resolve a static detector mask (``MaskSpec`` / bool 2-D / flat-index
    array) to a bool array of ``shape`` (True = exclude), or ``None`` if absent /
    unresolvable.  Warn-don't-crash: a shape mismatch logs and yields ``None`` so
    a bad mask never aborts the whole scan."""
    if mask is None:
        return None
    try:
        spec = mask if hasattr(mask, "to_bool") else MaskSpec(mask)
        return np.asarray(spec.to_bool(tuple(shape)), dtype=bool)
    except Exception:
        logger.warning("ROI stats: could not resolve the static mask to %s; "
                       "ignoring it", tuple(shape), exc_info=True)
        return None


def _reduce_signal(img, mask, signal):
    """Reduce one :class:`RoiSignal` over a single (mask-computed) image →
    ``(value, n_valid)``, applying its optional background.  Both subtract and
    divide use the background DENSITY (mean over valid bg pixels), so a sum
    reducer stays area-scaled for either op."""
    val, n_valid = roi_reduce(img, signal.roi, mask=mask, reducer=signal.reducer)
    if signal.background is None:
        return val, n_valid
    bkg_mean, _ = roi_reduce(img, signal.background, mask=mask, reducer="mean")
    val = _combine_background(val, n_valid, signal.reducer,
                              signal.background_op, bkg_mean)
    return val, n_valid


def run_roi_signals(
    signals,
    source,
    *,
    x_key: str | None = None,
    mask: Any = None,
    mask_saturation: bool = False,
    frame_indices=None,
    on_progress=None,
    on_frame=None,
    should_cancel=None,
) -> AnalysisResult:
    """Reduce a sequence of :class:`RoiSignal`s (each with its OWN reducer +
    optional background) over each raw frame of ``source`` into per-frame
    series — the per-ROI generalization of :func:`run_roi_stats`, with optional
    streaming hooks mirroring :func:`xrd_tools.analysis.runner.run_batch`.

    One pass over the frames (the dominant I/O cost) reduces every signal, so N
    ROIs do not load each frame N times.  ``on_frame(frame_index, {name:
    value})`` fires after each frame; ``on_progress(done, total)`` after each;
    ``should_cancel()`` is polled BEFORE each frame (returning True stops early —
    the result then covers only the frames processed so far).  All default
    ``None`` ⇒ a plain blocking run.  An unreadable raw frame records NaN + a
    diagnostic (warn, never crash).  Returns ``AnalysisResult(kind="roi_stats",
    payload=RoiStatsResult)``."""
    src = ensure_frame_source(source)
    signals = tuple(signals) or (RoiSignal(RoiSpec.full_frame()),)
    names = _dedup_names(
        [sig.name or sig.roi.name or f"roi{i}" for i, sig in enumerate(signals)])
    all_frames = [int(f) for f in
                  (src.frame_indices if frame_indices is None else frame_indices)]
    series: dict[str, list] = {n: [] for n in names}
    counts: dict[str, list] = {n: [] for n in names}
    no_raw: list[int] = []
    done_frames: list[int] = []
    total = len(all_frames)
    cancelled = False
    static_mask = None          # resolved lazily on the first frame (uniform shape)

    for done, f in enumerate(all_frames, start=1):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        try:
            img = np.asarray(src.load_frame(f))
        except Exception:
            row = {}
            for n in names:
                series[n].append(float("nan"))
                counts[n].append(0)
                row[n] = float("nan")
            no_raw.append(f)
        else:
            if mask is not None and static_mask is None:
                static_mask = _resolve_static_mask(mask, img.shape)
            frame_mask = invalid_pixel_mask(img, mask_saturation=mask_saturation)
            if static_mask is not None and static_mask.shape == img.shape:
                frame_mask = frame_mask | static_mask
            row = {}
            for sig, n in zip(signals, names):
                val, n_valid = _reduce_signal(img, frame_mask, sig)
                series[n].append(val)
                counts[n].append(n_valid)
                row[n] = val
        done_frames.append(f)
        if on_frame is not None:
            on_frame(f, row)
        if on_progress is not None:
            on_progress(done, total)

    x_label = "frame"
    x = np.asarray(done_frames, dtype=float)
    if x_key:
        try:
            x = _metadata_series(src, done_frames, x_key)
            x_label = x_key
        except (KeyError, ValueError, TypeError):
            pass  # a frame lacked the key -> fall back to frame labels

    payload = RoiStatsResult(
        x=np.asarray(x, dtype=float), x_label=x_label,
        series={n: np.asarray(v, dtype=float) for n, v in series.items()},
        frames=np.asarray(done_frames),
        valid_counts={n: np.asarray(v) for n, v in counts.items()},
        diagnostics={"no_raw_frames": no_raw, "cancelled": cancelled},
    )
    return AnalysisResult(
        kind="roi_stats", payload=payload,
        provenance={"signals": [_json_safe(asdict(s)) for s in signals]})


def run_roi_stats(plan: RoiStatsPlan, source) -> AnalysisResult:
    """Reduce ``plan.rois`` over each raw frame of ``source`` into per-frame
    series.  An unreadable raw frame records NaN + a diagnostic (warn, never
    crash).  ``x`` is ``plan.x_key`` aligned to the frames, else the frame
    labels.  Returns ``AnalysisResult(kind="roi_stats", payload=RoiStatsResult)``.

    A thin wrapper that maps the plan's SHARED reducer/background onto per-ROI
    :class:`RoiSignal`s and delegates to :func:`run_roi_signals`."""
    rois = plan.rois or (RoiSpec.full_frame(),)
    signals = tuple(
        RoiSignal(roi=roi, reducer=plan.reducer, background=plan.background,
                  background_op=plan.background_op, name=roi.name or f"roi{i}")
        for i, roi in enumerate(rois))
    result = run_roi_signals(
        signals, source, x_key=plan.x_key, mask=plan.mask,
        mask_saturation=plan.mask_saturation, frame_indices=plan.frame_indices)
    # Preserve the original provenance (the plan dict) for back-compat.
    return AnalysisResult(kind=result.kind, payload=result.payload,
                          provenance={"plan": _plan_dict(plan)})


__all__ = [
    "AnalysisResult",
    "PeakFitPlan",
    "PhaseFitPlan",
    "RSMPlan",
    "RoiSignal",
    "RoiStatsPlan",
    "RoiStatsResult",
    "Sin2PsiPlan",
    "StitchPlan",
    "make_phase_fitter",
    "run_peak_fit",
    "run_phase_fit",
    "run_roi_signals",
    "run_roi_stats",
    "run_rsm",
    "run_sin2psi",
    "run_stitch",
]
