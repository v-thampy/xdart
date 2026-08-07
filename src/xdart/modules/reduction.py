"""Adapters from xdart live objects to ssrl scan/frame APIs.

This module is the migration boundary for the thin-GUI refactor.  The rest of
xdart may still import the transitional ``Ewald*`` aliases for now, but new
headless reduction work should cross into ``xrd_tools`` as ``Frame`` /
``Scan`` / ``ReductionPlan`` objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Any, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)


class DynamicXyeReceiptBoundaryRequired(TypeError):
    """A dynamic XYE target lacks the accepted transaction receipt owner."""

from dataclasses import fields as _dc_fields

if TYPE_CHECKING:
    from xrd_tools.session import FrameRecordStore

from xrd_tools.core.metadata import (
    IncidenceAngleUnresolved,
    resolve_incident_angle,
    resolve_monitor_norm,
)
from xrd_tools.reduction import (
    Frame,
    FrameReduction,
    GIMode,
    Integration1DPlan,
    Integration2DPlan,
    MaskSpec,
    ReductionPlan,
    ReductionResult,
    ReductionSession,
    Scan,
    StrictPolicy,
    NexusSink,
    run_reduction,
    supports_durable_xye_receipts,
)
from xrd_tools.reduction.masks import _flat_mask_as_bool, _mask_for_plan
from xdart.modules.wavelength import wavelength_m_to_angstrom

# S4: GI-only scan kwargs that must NOT flow through to the standard
# pyFAI integrator path.  Derived from :class:`GIMode` (so adding a GIMode
# field automatically excludes that name) plus a small set of legacy
# xdart-only names that the GI widgets emit but :class:`GIMode` doesn't
# own.  Anything not in this set rides through ``Integration*Plan.extra``.
_GI_ONLY_ARGS: frozenset[str] = frozenset(
    {field.name for field in _dc_fields(GIMode)}
    | {
        "gi_mode_1d",
        "gi_mode_2d",
        "npt_oop",
        "npt_ip",
        "x_range",
        "y_range",
    }
)


def frame_from_live_frame(
    live_frame: Any,
    *,
    include_image: bool = True,
    include_background: bool = True,
) -> Frame:
    """Build an ``xrd_tools.reduction.Frame`` from a ``LiveFrame``."""
    image = getattr(live_frame, "map_raw", None) if include_image else None
    source_path = _source_path(live_frame)
    mask = _live_frame_mask_for_frame(live_frame)
    metadata = dict(getattr(live_frame, "scan_info", {}) or {})
    bg_raw = getattr(live_frame, "bg_raw", None) if include_background else None
    if bg_raw is not None and np.ndim(bg_raw) == 0:
        metadata.setdefault("bg_raw", float(bg_raw))

    return Frame(
        index=int(getattr(live_frame, "idx", 0) or 0),
        image=image,
        metadata=metadata,
        source_path=source_path,
        source_frame_index=int(getattr(live_frame, "source_frame_idx", 0) or 0),
        background=bg_raw,
        mask=mask,
    )


def scan_from_live_scan(
    live_scan: Any,
    *,
    frame_indices: Iterable[int] | None = None,
    pinned_frames: Iterable[Any] | None = None,
    include_images: bool = True,
    include_backgrounds: bool | None = None,
) -> Scan:
    """Build an ``xrd_tools.reduction.Scan`` from a ``LiveScan``."""
    if include_backgrounds is None:
        include_backgrounds = include_images
    pins = None if pinned_frames is None else list(pinned_frames)
    indices = (
        [int(getattr(frame, "idx", 0) or 0) for frame in pins]
        if pins is not None
        else (list(frame_indices) if frame_indices is not None
              else list(live_scan.frames.index))
    )
    scan_data = getattr(live_scan, "scan_data", None)
    frames = []
    for position, idx in enumerate(indices):
        live_frame = pins[position] if pins is not None else live_scan.frames[int(idx)]
        frame = frame_from_live_frame(
            live_frame,
            include_image=include_images,
            include_background=include_backgrounds,
        )
        frame.metadata.update(
            {
                key: value
                for key, value in _scan_data_row(scan_data, int(idx)).items()
                if key not in frame.metadata
            }
        )
        frames.append(frame)
    first_frame = (
        pins[0] if pins
        else (live_scan.frames[int(indices[0])] if indices else None)
    )
    poni = getattr(first_frame, "poni", None) if first_frame is not None else None
    wavelength_A = wavelength_m_to_angstrom(
        getattr(live_scan, "_persisted_wavelength_m", None),
        allow_default_sentinel=True,
    )
    if wavelength_A is None:
        wavelength_A = wavelength_m_to_angstrom(
            (getattr(live_scan, "mg_args", {}) or {}).get("wavelength", None)
        )

    # Per-COLUMN numeric coercion: one string column (sample name, file
    # path — common in scan_data) must skip itself, not nuke every numeric
    # motor column with it.
    motors = {}
    if scan_data is not None and hasattr(scan_data, "columns"):
        for col in scan_data.columns:
            try:
                motors[str(col)] = np.asarray(scan_data[col].values,
                                              dtype=float)
            except (TypeError, ValueError):
                continue

    calibration = {}
    source_snapshots = {}
    for live_frame in (
        pins if pins is not None
        else [live_scan.frames[int(index)] for index in indices]
    ):
        path = _source_path(live_frame)
        snapshot = getattr(live_frame, "source_snapshot", None)
        if path is not None and snapshot:
            source_snapshots[str(path)] = dict(snapshot)
    poni = getattr(first_frame, "poni", None) if first_frame is not None else None
    if poni is None:
        poni = getattr(live_scan, "_cached_poni", None)
    if poni is not None:
        for name in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
            value = getattr(poni, name, None)
            if value is not None:
                calibration[name] = value
        detector_name = getattr(poni, "detector", None)
        if detector_name:
            calibration["detector_name"] = detector_name
    integrator = getattr(live_scan, "_cached_integrator", None)
    detector = getattr(integrator, "detector", None)
    if detector is not None:
        if getattr(detector, "pixel2", None) is not None:
            calibration["x_pixel_size"] = detector.pixel2
        if getattr(detector, "pixel1", None) is not None:
            calibration["y_pixel_size"] = detector.pixel1

    return Scan(
        name=str(getattr(live_scan, "name", "scan")),
        frames=frames,
        poni=poni,
        wavelength=wavelength_A,
        # so a headless NexusSink derives /entry/per_frame_geometry at
        # finish, matching what the GUI writer produces for the same scan
        geometry=getattr(live_scan, "geometry", None),
        motors=motors,
        output_path=getattr(live_scan, "data_file", None),
        extra={
            "source": "xdart.LiveScan",
            "detector_calibration": calibration,
            "global_mask": getattr(live_scan, "global_mask", None),
            "detector_shape": getattr(live_scan, "detector_shape", None),
            "stitched_1d": getattr(live_scan, "stitched_1d", None),
            "stitched_2d": getattr(live_scan, "stitched_2d", None),
            "source_snapshots": source_snapshots,
        },
    )


class LiveScanNexusSession:
    """One canonical writer/lease/cursor lifetime for one LiveScan output."""

    def __copy__(self):
        raise TypeError("LiveScan Nexus session authority cannot be copied")

    def __deepcopy__(self, _memo):
        raise TypeError("LiveScan Nexus session authority cannot be copied")

    def __init__(
        self,
        live_scan: Any,
        *,
        entry: str = "entry",
        replace: bool = False,
        file_lock=None,
        append_preflight=None,
        accounting=None,
    ) -> None:
        self.live_scan = live_scan
        self.plan = plan_from_live_scan(live_scan)
        self.same_run_intent = getattr(live_scan, "_same_run_intent", None)
        self.sink = NexusSink(
            live_scan.data_file,
            entry=str(entry),
            overwrite=bool(replace),
            flush_every=None,
            atomic=False,
            source_base=getattr(live_scan, "source_base", None),
            file_lock=(file_lock if file_lock is not None
                       else getattr(live_scan, "file_lock", None)),
            append_preflight=append_preflight,
            same_run_intent=(None if append_preflight is not None
                             else self.same_run_intent),
            allow_unbound_same_run=bool(
                replace and append_preflight is None
                and self.same_run_intent is None
            ),
            incremental_finalization=True,
            run_configuration_provenance=getattr(
                live_scan, "run_configuration_provenance", None,
            ),
            source_execution_provenance=getattr(
                live_scan, "source_execution_provenance", None,
            ),
            source_snapshots_provenance=getattr(
                live_scan, "source_snapshots_provenance", None,
            ),
        )
        self._begun = False
        self._closed = False
        self._products: dict[int, FrameReduction] = {}
        self._written: list[int] = []
        self._accounting = None
        self._accounting_owner_token = object()
        self._terminal_accounting_identity = object()
        self._accounting_epoch_seal = None
        self._accounting_epoch_anchor = None
        self._accounting_committed_epoch_anchor = None
        self._accounting_epoch_dirty = False
        self._accounting_finish_seal = None
        self._accounting_finish_disposition = None
        self._terminal_result = None
        self._terminal_finalize = None
        self._sink_finished = False
        self._sink_aborted = False
        if accounting is not None:
            self.bind_accounting(accounting)

    def bind_accounting(self, facade) -> None:
        """Bind the one run ledger before the canonical writer begins.

        Exact replay of the same facade is harmless. A different authority may
        never replace it, and no first binding is accepted after ``begin()``.
        """
        if self._accounting is facade:
            return
        if self._accounting is not None:
            raise RuntimeError("LiveScan Nexus accounting authority cannot change")
        if self._begun:
            raise RuntimeError("LiveScan Nexus accounting must bind before begin")
        self.sink.bind_session(facade)
        self._accounting = facade
        bind_live = getattr(facade, "bind_live_session", None)
        if callable(bind_live):
            bind_live(self, self._accounting_owner_token)

    def extend(self, intent):
        """Extend this exact open owner; never re-admit or reopen the target."""
        if (
            self._accounting_epoch_seal is not None
            or self._accounting_finish_seal is not None
        ):
            raise RuntimeError("sealed LiveScan lineage must retry or abort first")
        if self.sink.append_preflight is not None:
            decision = self.sink.append_preflight.extend(intent)
            self.same_run_intent = self.live_scan._same_run_intent = intent
            return decision
        if not self._begun:
            self.same_run_intent = self.sink.same_run_intent = intent
            self.live_scan._same_run_intent = intent
            return None
        decision = self.sink.extend_live(self.sink.extension_owner, intent)
        self.same_run_intent = self.live_scan._same_run_intent = intent
        return decision

    def _select(self, replace_frame_indices=None) -> list[int]:
        all_labels = [int(value) for value in self.live_scan.frames.index]
        if replace_frame_indices is not None:
            selected = [int(value) for value in replace_frame_indices]
        elif self.sink.overwrite and not self._begun:
            selected = all_labels
        else:
            persisted = set(getattr(self.live_scan.frames, "_persisted", set()))
            selected = [label for label in all_labels if label not in persisted]
        if self.sink.append_preflight is not None:
            allowed = set(self.sink.append_preflight.snapshot.write_labels)
            selected = [label for label in selected if label in allowed]
        return selected

    def _terminal(self, live) -> bool:
        if self.plan.integration_1d is not None and getattr(live, "int_1d", None) is None:
            return False
        if (self.plan.integration_2d is not None
                and not getattr(self.live_scan, "skip_2d", False)
                and getattr(live, "int_2d", None) is None):
            return False
        return all(
            value is not None
            for mapping in (
                getattr(live, "gi_1d", None) or {},
                getattr(live, "gi_2d", None) or {},
            )
            for value in mapping.values()
        )

    def flush(self, *, replace_frame_indices=None, force: bool = False):
        if self._closed:
            raise RuntimeError("LiveScan Nexus session is terminal")
        if (
            self._accounting_epoch_seal is not None
            or self._accounting_finish_seal is not None
        ):
            raise RuntimeError(
                "sealed LiveScan lineage must retry commit/finish or abort first"
            )
        intent = getattr(self.live_scan, "_same_run_intent", None)
        if intent is not None and intent != self.same_run_intent:
            self.extend(intent)
        labels = self._select(replace_frame_indices)
        if self.sink.append_preflight is not None or self.same_run_intent is not None:
            labels = [label for label in labels
                      if self._terminal(self.live_scan.frames[label])]
        pins = [self.live_scan.frames[label] for label in labels]
        if not pins:
            if self._begun:
                self.sink.flush(force=force)
            elif (force and self.sink.overwrite
                  and self.sink.append_preflight is None):
                scan = scan_from_live_scan(
                    self.live_scan,
                    pinned_frames=(),
                    include_images=False,
                    include_backgrounds=False,
                )
                self.sink.begin(scan, self.plan)
                self._begun = True
                self.sink.flush(force=True)
            return {}
        scan = scan_from_live_scan(
            self.live_scan,
            pinned_frames=pins,
            include_images=False,
            include_backgrounds=False,
        )
        # From this point a public flush may mutate a new H23 epoch.  A prior
        # accounting-confirmed anchor must no longer authorize an unsealed
        # committed phase until commit_epoch() positively publishes the new
        # exact seal.
        self._accounting_epoch_dirty = True
        if not self._begun:
            self.sink.begin(scan, self.plan)
            self._begun = True
        else:
            self.sink._scan.extra.update(scan.extra)
        from xrd_tools.io.nexus_record import (
            legacy_to_canonical_1d,
            legacy_to_canonical_2d,
        )

        terminal = []
        for frame, live in zip(scan.frames, pins):
            metadata = dict(getattr(live, "scan_info", {}) or {})
            reduction = FrameReduction(
                frame_index=int(frame.index),
                result_1d=getattr(live, "int_1d", None),
                result_2d=getattr(live, "int_2d", None),
                metadata=metadata,
                thumbnail=getattr(live, "thumbnail", None),
            )
            self._products[int(frame.index)] = reduction
            self.sink.write(frame, reduction)
            for key, result in (getattr(live, "gi_1d", None) or {}).items():
                if result is not reduction.result_1d:
                    self.sink.write(frame, FrameReduction(
                        int(frame.index), result_1d=result,
                        mode_1d=legacy_to_canonical_1d(key),
                        metadata=metadata,
                        write_frame_record=False,
                    ))
            for key, result in (getattr(live, "gi_2d", None) or {}).items():
                if result is not reduction.result_2d:
                    self.sink.write(frame, FrameReduction(
                        int(frame.index), result_2d=result,
                        mode_2d=legacy_to_canonical_2d(key),
                        metadata=metadata,
                        write_frame_record=False,
                    ))
            if self._terminal(live):
                terminal.append(int(frame.index))
            if int(frame.index) not in self._written:
                self._written.append(int(frame.index))
        self.sink.flush(force=force)
        mark = getattr(self.live_scan.frames, "mark_persisted", None)
        if callable(mark):
            mark(terminal)
        return {}

    def _prepare_accounting_epoch(self, *, finishing: bool, stopped: bool = False):
        if self._accounting is None:
            return None
        name = "prepare_session_finish" if finishing else "prepare_epoch_commit"
        prepare = getattr(self._accounting, name, None)
        if not callable(prepare):
            return None
        if finishing:
            return prepare(
                self,
                self._accounting_owner_token,
                stopped=bool(stopped),
            )
        return prepare(self, self._accounting_owner_token)

    def _transaction_phase(self):
        transaction = getattr(self.sink, "_transaction", None)
        if transaction is None:
            return None
        phase = transaction.snapshot().phase
        return getattr(phase, "value", str(phase))

    def _require_terminal_transaction_phase(self):
        phase = self._transaction_phase()
        if (
            self._accounting is not None
            and self._begun
            and phase not in {"committed", "aborted"}
        ):
            raise RuntimeError(
                "bound dynamic accounting requires an exact terminal "
                f"transaction phase; got {phase!r}"
            )
        return phase

    def _notify_accounting_finish(self) -> None:
        seal = self._accounting_finish_seal
        if seal is None:
            return
        if self._accounting_finish_disposition == "stopped-discard":
            notify = getattr(self._accounting, "session_stopped", None)
            if not callable(notify):
                raise RuntimeError("stopped accounting boundary has no terminal callback")
            notify(
                self,
                self._accounting_owner_token,
                seal,
                "LiveScan stopped before any canonical epoch was committed",
            )
        else:
            notify = getattr(self._accounting, "session_finished", None)
            if not callable(notify):
                raise RuntimeError("finished accounting boundary has no terminal callback")
            notify(
                self,
                self._accounting_owner_token,
                seal,
                self._terminal_accounting_identity,
            )
        self._accounting_finish_seal = None

    def finish(self, *, finalize: bool = True, commit_empty: bool = False):
        if self._closed:
            return {}
        if self._sink_finished:
            self._notify_accounting_finish()
            self._closed = True
            return {}
        if self._accounting_finish_seal is not None:
            if (bool(finalize), bool(commit_empty)) != self._terminal_finalize:
                raise RuntimeError("LiveScan finish retry changed frozen terminal values")
            result = self._terminal_result
            self.sink.finish(result)
            phase = self._require_terminal_transaction_phase()
            self._accounting_finish_disposition = (
                "stopped-discard" if phase == "aborted" else "committed"
            )
            self._sink_finished = True
            self._notify_accounting_finish()
            self._closed = True
            return {}
        if not self._begun:
            self._accounting_finish_seal = self._prepare_accounting_epoch(
                finishing=True, stopped=not finalize,
            )
            self._terminal_finalize = (bool(finalize), bool(commit_empty))
            if self.sink.append_preflight is not None:
                if self.sink.append_preflight.snapshot.disposition.value == "skip":
                    self.sink.append_preflight.complete_noop()
                else:
                    self.sink.append_preflight.abort()
            self._sink_finished = True
            self._accounting_finish_disposition = (
                "stopped-discard" if not finalize else "committed"
            )
            self._notify_accounting_finish()
            self._closed = True
            return {}
        self.flush(force=True)
        if not finalize and self._written:
            self.sink.truncate_epoch(tuple(self._written))
        if not finalize and self.sink._scan is not None:
            self.sink._scan.extra["stitched_1d"] = None
            self.sink._scan.extra["stitched_2d"] = None
        result = ReductionResult(
            str(getattr(self.live_scan, "name", "scan")),
            self._products,
            len(self._products),
            cancelled=bool(
                not finalize
                and not self._written
                and not commit_empty
                and self._transaction_phase() not in {
                    "epoch-committed", "committed",
                }
            ),
        )
        self._accounting_finish_seal = self._prepare_accounting_epoch(
            finishing=True, stopped=not finalize,
        )
        self._terminal_finalize = (bool(finalize), bool(commit_empty))
        self._terminal_result = result
        self.sink.finish(result)
        phase = self._require_terminal_transaction_phase()
        self._accounting_finish_disposition = (
            "stopped-discard" if phase == "aborted" else "committed"
        )
        self._sink_finished = True
        self._notify_accounting_finish()
        self._closed = True
        return {}

    def commit_epoch(self):
        """Finalize a durable lineage epoch without releasing this session."""
        if self._closed or not self._begun:
            raise RuntimeError("lineage epoch commit requires an active session")
        if self._accounting_finish_seal is not None:
            raise RuntimeError("terminal finish is already sealed")
        if self._accounting_epoch_seal is None:
            self.flush(force=True)
            self._accounting_epoch_seal = self._prepare_accounting_epoch(
                finishing=False,
            )
        if self._accounting_epoch_anchor is None:
            result = ReductionResult(
                str(getattr(self.live_scan, "name", "scan")),
                self._products,
                len(self._products),
            )
            self._accounting_epoch_anchor = self.sink.commit_epoch(result)
        anchor = self._accounting_epoch_anchor
        notify = getattr(self._accounting, "epoch_committed", None)
        if callable(notify) and self._accounting_epoch_seal is not None:
            notify(
                self,
                self._accounting_owner_token,
                self._accounting_epoch_seal,
                anchor,
            )
            self._accounting_committed_epoch_anchor = anchor
            self._accounting_epoch_dirty = False
        self._accounting_epoch_seal = None
        self._accounting_epoch_anchor = None
        self._written.clear()
        return anchor

    def abort(self):
        if self._closed:
            return
        if self._sink_finished:
            self._notify_accounting_finish()
            self._closed = True
            return
        if not self._sink_aborted:
            if self._begun:
                self.sink.abort(ReductionResult(
                    str(getattr(self.live_scan, "name", "scan")),
                    self._products,
                    len(self._products),
                    failed=True,
                ))
            elif self.sink.append_preflight is not None:
                self.sink.append_preflight.abort()
            self._sink_aborted = True
        phase = self._require_terminal_transaction_phase()
        if self._accounting_finish_seal is not None and phase == "committed":
            self._sink_finished = True
            self._accounting_finish_disposition = "committed"
            self._notify_accounting_finish()
            self._closed = True
            return
        if self._accounting_epoch_seal is not None and phase == "committed":
            anchor = self._accounting_epoch_anchor or getattr(
                self.sink, "_epoch_decision", None,
            )
            if anchor is None:
                raise RuntimeError("canonical H23 epoch has no lineage anchor")
            committed = getattr(self._accounting, "epoch_committed", None)
            if callable(committed):
                committed(
                    self,
                    self._accounting_owner_token,
                    self._accounting_epoch_seal,
                    anchor,
                )
            self._accounting_epoch_seal = None
            self._accounting_epoch_anchor = None
        elif (
            self._accounting is not None
            and self._begun
            and phase == "committed"
            and (
                self._accounting_epoch_dirty
                or self._accounting_committed_epoch_anchor is None
            )
        ):
            raise RuntimeError(
                "committed dynamic transaction has no exact accounting seal"
            )
        notify = getattr(self._accounting, "epoch_aborted", None)
        if callable(notify):
            notify(
                self,
                self._accounting_owner_token,
                "LiveScan Nexus session abort",
            )
        self._accounting_epoch_seal = None
        self._accounting_epoch_anchor = None
        self._accounting_finish_seal = None
        self._closed = True


def open_live_scan_nexus_session(live_scan: Any, **kwargs) -> LiveScanNexusSession:
    return LiveScanNexusSession(live_scan, **kwargs)


def write_live_scan_to_nexus(
    live_scan: Any,
    *,
    entry: str = "entry",
    replace: bool = False,
    finalize: bool = False,
    replace_frame_indices=None,
    file_lock=None,
    append_preflight=None,
    accounting=None,
) -> dict[str, list[int]]:
    """One-shot compatibility call using the same bounded shared session."""
    session = open_live_scan_nexus_session(
        live_scan,
        entry=entry,
        replace=replace,
        file_lock=file_lock,
        append_preflight=append_preflight,
        accounting=accounting,
    )
    try:
        session.flush(replace_frame_indices=replace_frame_indices, force=True)
        return session.finish(finalize=finalize, commit_empty=True)
    except BaseException as primary:
        try:
            session.abort()
        except BaseException as cleanup:
            raise primary from cleanup
        raise


@dataclass
class ThresholdSaturationConfig:
    """The wrangler's per-frame pixel-rejection policy, in plan terms.

    Carries the GUI's *current* Intensity-Threshold + Mask-Saturated settings so
    a re-integration can apply the same per-frame pixel rejection a live run
    does — but via the headless ReductionPlan fields (``threshold_min`` /
    ``threshold_max`` / ``mask_saturation``, applied by the reducer after
    ``load_image``) rather than the wrangler's image-preprocessing path, which
    the reintegrate route does not go through.
    """
    apply_threshold: bool = False
    threshold_min: float | None = None
    threshold_max: float | None = None
    mask_saturation: bool = False


def apply_threshold_saturation_to_plan(plan, cfg):
    """Return a plan with the threshold/saturation fields set from ``cfg``.

    No-op (identity preserved, so a cached session can be reused) when ``cfg``
    is None or when it would not change the plan.  When the intensity threshold
    is off, the band collapses to ``None`` (parity with the wrangler's
    ``_apply_threshold_inline`` no-op).  Applied AFTER the plan cache .get
    because the plan-cache / session keys don't fingerprint these fields.
    """
    if plan is None or cfg is None:
        return plan
    tmin = cfg.threshold_min if cfg.apply_threshold else None
    tmax = cfg.threshold_max if cfg.apply_threshold else None
    msat = bool(cfg.mask_saturation)
    if (getattr(plan, "threshold_min", None) == tmin
            and getattr(plan, "threshold_max", None) == tmax
            and bool(getattr(plan, "mask_saturation", False)) == msat):
        return plan
    return replace(plan, threshold_min=tmin, threshold_max=tmax,
                   mask_saturation=msat)


def bad_pixel_counts(raw_image) -> dict[str, int]:
    """Diagnostics for one raw detector frame: how many pixels are unambiguous
    invalids (negatives + the uint32 dead/hot sentinel) and how many sit at the
    fraction-guarded detector-saturation ceiling.  Pure counting, never raises —
    for ``[REINT-MASK]`` logging so a reintegrate can SHOW what it rejected."""
    from xrd_tools.core.invalid import (
        UINT32_CEILING, integer_saturation_ceiling, saturation_pixels,
    )
    out = {"size": 0, "negative": 0, "uint32_dummy": 0, "saturation": 0}
    try:
        arr0 = np.asarray(raw_image)
        flat = arr0.astype(float).flatten()
    except (TypeError, ValueError):
        return out
    if flat.size == 0:
        return out
    out["size"] = int(flat.size)
    out["negative"] = int((flat < 0).sum())
    out["uint32_dummy"] = int((flat >= UINT32_CEILING).sum())
    out["saturation"] = int(
        saturation_pixels(flat, ceiling=integer_saturation_ceiling(arr0)).sum())
    return out


_CEILING_AUTO = object()


def compute_bad_pixel_mask(raw_image, *, mask_saturation: bool = True,
                           saturation_ceiling=_CEILING_AUTO):
    """Flat-index "bad pixel" mask for one raw detector frame — the SINGLE
    implementation shared by the live wrangler (``_resolve_frame_mask``) and the
    reintegrate path, so they mask the same pixels on the same frame.

    ``mask_saturation`` (the GUI "Auto Mask Saturated" toggle) is the
    AUTHORITATIVE on/off for ALL masking here:

    * **False -> mask NOTHING** (returns ``None``).  The raw frame is integrated
      as-is, including the highest-value pixels — the uint32 dead/hot sentinel
      (:data:`~xrd_tools.core.invalid.UINT32_CEILING` = 4294967295, Eiger
      masters) AND genuinely-saturated pixels.  This is deliberate: strong Bragg
      peaks legitimately saturate, and the user must be able to KEEP them rather
      than have them auto-masked.  (Yes, the unmasked uint32 sentinel then
      dominates radial bins -> the high-Q spike; that is the user's choice when
      the toggle is off.  See [[mask-saturated-toggle-authoritative]].)
    * **True -> mask** negatives + the uint32 sentinel + the fraction-guarded
      detector-saturation ceiling (:func:`~xrd_tools.core.invalid.saturation_pixels`
      — uint16 65535 etc., only when a whole module sits there).

    NOTE: the ``xrd_tools.core.invalid`` policy that the uint32 dummy is "masked
    always, not gated" is the HEADLESS/core stance for non-GUI callers; the xdart
    GUI gates everything on the toggle here.  ``saturation_ceiling`` selects the
    ceiling for the fraction-guarded part: the default derives it from the integer
    dtype (``None`` for float — core never hardcodes 65535); GUI callers pass the
    display policy's ceiling (its 65535 float fallback) so live and reintegrate
    agree on a float-typed raw too.

    Returns a flat ``int`` index array (the pyFAI / :class:`MaskSpec` format),
    or ``None`` when nothing is masked or the input is unusable (never raises)."""
    if not mask_saturation:
        return None
    from xrd_tools.core.invalid import (
        UINT32_CEILING, integer_saturation_ceiling, saturation_pixels,
    )
    try:
        arr0 = np.asarray(raw_image)
        flat = arr0.astype(float).flatten()
    except (TypeError, ValueError):
        return None
    if flat.size == 0:
        return None
    ceil = (integer_saturation_ceiling(arr0)
            if saturation_ceiling is _CEILING_AUTO else saturation_ceiling)
    bad = (flat < 0) | (flat >= UINT32_CEILING) | saturation_pixels(flat, ceiling=ceil)
    idx = np.flatnonzero(bad)
    return idx if idx.size else None


def apply_frozen_run_configuration(live_scan: Any, run_configuration: Any) -> Any:
    """Project one accepted immutable run configuration onto its LiveScan."""
    scan = live_scan
    scan.skip_2d = bool(run_configuration.skip_2d)
    scan.gi_config = run_configuration.gi.scan_config()
    scan.sample_orientation = int(run_configuration.gi.sample_orientation)
    scan.tilt_angle = float(run_configuration.gi.tilt_angle)
    scan.th_mtr = run_configuration.gi.scan_incidence_motor
    scan.max_cores = int(run_configuration.max_cores)
    threshold = run_configuration.threshold
    scan.apply_threshold = bool(threshold.apply_threshold)
    scan.threshold_min = threshold.threshold_min
    scan.threshold_max = threshold.threshold_max
    scan.mask_sentinel = bool(threshold.mask_saturation)
    scan.run_configuration = run_configuration
    scan.run_configuration_generation = int(run_configuration.generation)
    scan.run_configuration_fingerprint = run_configuration.fingerprint
    scan.run_configuration_provenance = run_configuration.as_provenance()
    return scan


def plan_from_live_scan(
    live_scan: Any,
    *,
    integrate_1d: bool = True,
    integrate_2d: bool | None = None,
    gi_incident_angle: float | None = None,
    run_configuration: Any = None,
) -> ReductionPlan:
    """Create a ``ReductionPlan`` using xdart's current live scan settings.

    Note: ``chunk_size`` and other execution-policy knobs live on
    :func:`run_reduction` (and on :func:`reduce_live_frame` by way of
    the single-frame call here), not on the plan itself.
    """
    frozen = run_configuration
    if frozen is None:
        frozen = getattr(live_scan, "run_configuration", None)
    if frozen is not None and not hasattr(frozen, "processing_mapping"):
        frozen = None
    if integrate_2d is None:
        integrate_2d = not bool(
            frozen.skip_2d if frozen is not None
            else getattr(live_scan, "skip_2d", False)
        )

    if frozen is not None:
        args_1d = dict(frozen.bai_1d_args)
        args_2d = dict(frozen.bai_2d_args)
    else:
        args_1d = dict(getattr(live_scan, "bai_1d_args", {}) or {})
        args_2d = dict(getattr(live_scan, "bai_2d_args", {}) or {})
    unit_1d = _pop_first(args_1d, ("unit",), None)
    unit_2d = _pop_first(args_2d, ("unit",), None)
    method_1d = str(_pop_first(args_1d, ("method",), "csr"))
    method_2d = str(_pop_first(args_2d, ("method",), "csr"))
    npt_1d = int(_pop_first(args_1d, ("npt", "numpoints", "npt_rad"), 1000))
    # Azimuthal Mode A (unit='chi_deg') band sampling.  A DISTINCT key from
    # 'npt_rad' (which is aliased to the chi-bin npt above) so the two never
    # collide; absent -> the pyFAI default of 1000.
    npt_rad_1d = int(_pop_first(args_1d, ("chi_npt_rad",), 1000))
    npt_rad_2d, npt_azim_2d = _npt_2d(args_2d)
    radial_range = _pop_first(args_1d, ("radial_range",), None)
    radial_range_2d = _pop_first(args_2d, ("radial_range",), None)
    azimuth_range_1d = _pop_first(args_1d, ("azimuth_range",), None)
    azimuth_range_2d = _pop_first(args_2d, ("azimuth_range",), None)
    monitor_key = _pop_first(args_1d, ("monitor",), None)
    monitor_key_2d = _pop_first(args_2d, ("monitor",), None)
    chi_offset_1d = _pop_first(args_1d, ("chi_offset",), 0.0)
    chi_offset_2d = _pop_first(args_2d, ("chi_offset",), 0.0)
    # S-4 (ported to the legacy env fallback): carry chi_offset as the 1D plan's
    # azimuth_offset -- the reduction shifts the input by -offset and re-adds it
    # at OUTPUT (mirroring the 2D) -- instead of pre-shifting the input range here
    # and leaving the written 1D chi axis in the raw pyFAI frame 90deg off the 2D.
    # GI keeps offset 0 (its chi goes to FiberIntegrator's own polar convention
    # unshifted).  Without this, flipping XDART_CONTROLS_PANEL_V2 /
    # ..._NATIVE_RUN_PLAN to "0" silently changed written 1D chi data.
    azimuth_offset_1d = (
        float(chi_offset_1d)
        if chi_offset_1d and not bool(getattr(live_scan, "gi", False))
        else 0.0)
    error_model = _pop_first(args_1d, ("error_model",), None)
    error_model_2d = _pop_first(args_2d, ("error_model",), None)
    polarization_factor = _pop_first(args_1d, ("polarization_factor",), None)
    polarization_factor_2d = _pop_first(args_2d, ("polarization_factor",), None)
    _pop_first(args_1d, ("normalization_factor",), None)
    _pop_first(args_2d, ("normalization_factor",), None)

    is_gi = bool(
        frozen.gi.enabled if frozen is not None
        else getattr(live_scan, "gi", False)
    )
    # Reintegrate-on-reload: a .nxs-reloaded scan carries its GI geometry only in
    # ``scan.gi_config`` — a live run sets the direct attrs via
    # ``sync_live_scan_gi_settings``, but a reload restores only the dict.  Fall
    # back to it so reintegrate uses the SAME sample_orientation / tilt as live;
    # otherwise sample_orientation silently defaults to 1 and the GI out-of-plane
    # (Q_oop) axis flips sign vs the live run.
    _gi_cfg = dict(
        frozen.gi.scan_config() if frozen is not None
        else (getattr(live_scan, "gi_config", {}) or {})
    )

    def _gi_geom(attr, default):
        if frozen is not None:
            v = _gi_cfg.get(attr)
            return default if v is None else v
        v = getattr(live_scan, attr, None)
        if v is None:
            v = _gi_cfg.get(attr)
        return default if v is None else v

    gi_mode_1d = _pop_first(args_1d, ("gi_mode_1d",), "q_total")
    gi_mode_2d = _pop_first(args_2d, ("gi_mode_2d",), "qip_qoop")
    npt_oop = _pop_first(args_1d, ("npt_oop",), None)
    if npt_oop is None:
        npt_oop = _pop_first(args_2d, ("npt_oop",), None)
    gi_method = _pop_first(args_1d, ("gi_method_1d",), None)
    if gi_method is None:
        gi_method = _pop_first(args_2d, ("gi_method_2d",), "cython")

    if not is_gi:
        _strip_nonstandard_args(args_1d)
        _strip_nonstandard_args(args_2d)
    if is_gi and gi_incident_angle is None:
        gi_incident_angle = getattr(live_scan, "_cached_fiber_integrator_angle", None)
    incidence_motor = (
        frozen.gi.scan_incidence_motor if frozen is not None
        else getattr(live_scan, "incidence_motor", None)
    )
    if is_gi and gi_incident_angle is None and incidence_motor is not None:
        try:
            gi_incident_angle = resolve_incident_angle({}, incidence_motor)
            incidence_motor = None
        except IncidenceAngleUnresolved:
            pass
    if is_gi and gi_incident_angle is None and not incidence_motor:
        raise ValueError(
            "Cannot build a GI ReductionPlan without gi_incident_angle or incidence_motor."
        )

    # The flat ``global_mask`` indexes the FULL-RES detector, so PREFER the
    # detector shape (persisted + restored for exactly this).  ``frames[0]
    # .map_raw.shape`` is only sound for a LIVE/full-res frame — on a reloaded
    # scan it can be a THUMBNAIL, whose smaller shape made the full-res flat
    # indices fall out of bounds so ``_mask_for_plan`` silently DROPPED the mask
    # and reintegrate ran UNMASKED.  detector_shape (present on reload) avoids
    # that; the frame-shape fallback still covers live + older files that lack it.
    mask_shape = None
    _det_shape = getattr(live_scan, "detector_shape", None)
    if _det_shape is not None:
        try:
            mask_shape = (int(_det_shape[0]), int(_det_shape[1]))
        except (TypeError, ValueError, IndexError):
            mask_shape = None
    if mask_shape is None:
        try:
            first_idx = live_scan.frames.index[0]
            first_img = getattr(live_scan.frames[int(first_idx)], "map_raw", None)
            mask_shape = getattr(first_img, "shape", None)
        except Exception:
            mask_shape = None

    gi_mode = (
        GIMode(
            incident_angle=(float(gi_incident_angle) if gi_incident_angle is not None else None),
            incidence_motor=str(incidence_motor) if incidence_motor else None,
            tilt_angle=float(_gi_geom("tilt_angle", 0.0) or 0.0),
            sample_orientation=int(_gi_geom("sample_orientation", 1) or 1),
            method=str(gi_method),
            mode_1d=str(gi_mode_1d),
            mode_2d=str(gi_mode_2d),
            npt_oop=(int(npt_oop) if npt_oop is not None else None),
        )
        if is_gi else None
    )
    unit_1d = _gi_1d_unit_default(unit_1d, str(gi_mode_1d), is_gi=is_gi)
    unit_2d = _gi_2d_unit_default(unit_2d, str(gi_mode_2d), is_gi=is_gi)

    if is_gi and gi_mode is not None:
        # Diagnostic for the GI-1D live-vs-reintegrate divergence hunt (now
        # resolved): kept at DEBUG so it's available for a future hunt without
        # spamming the normal run log.
        logger.debug(
            "[GI-PLAN] incident_angle=%s incidence_motor=%s sample_orientation=%s "
            "tilt=%s gi_mode_1d=%s gi_method=%s npt_1d=%s npt_oop=%s "
            "radial_range_1d=%s azimuth_range_1d=%s unit_1d=%s",
            gi_mode.incident_angle, gi_mode.incidence_motor,
            gi_mode.sample_orientation, gi_mode.tilt_angle, gi_mode_1d,
            gi_method, npt_1d, npt_oop, radial_range, azimuth_range_1d, unit_1d,
        )

    return ReductionPlan(
        integration_1d=(
            Integration1DPlan(
                npt=npt_1d,
                npt_rad=npt_rad_1d,
                unit=unit_1d,
                method=method_1d,
                radial_range=radial_range,
                azimuth_range=azimuth_range_1d,
                azimuth_offset=azimuth_offset_1d,
                monitor_key=monitor_key,
                error_model=error_model,
                polarization_factor=polarization_factor,
                extra=args_1d,
            )
            if integrate_1d else None
        ),
        integration_2d=(
            Integration2DPlan(
                npt_rad=npt_rad_2d,
                npt_azim=npt_azim_2d,
                unit=unit_2d,
                method=method_2d,
                radial_range=radial_range_2d,
                azimuth_range=azimuth_range_2d,
                azimuth_offset=float(chi_offset_2d or 0.0),
                monitor_key=monitor_key_2d,
                error_model=error_model_2d,
                polarization_factor=polarization_factor_2d,
                extra=args_2d,
            )
            if integrate_2d else None
        ),
        gi=gi_mode,
        mask=_mask_for_plan(getattr(live_scan, "global_mask", None), mask_shape),
    )


def reduce_live_frame(
    live_frame: Any,
    plan: ReductionPlan,
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
) -> Any:
    """Reduce one ``LiveFrame`` through ``xrd_tools.reduction``.

    The returned object is the same ``live_frame`` instance, populated with
    ``int_1d`` / ``int_2d`` so existing xdart display and writer code can
    continue to operate while the computation crosses the new Scan/Frame API.
    """
    frame = frame_from_live_frame(live_frame)
    plan = _plan_with_mask_for_live_frame(plan, global_mask, live_frame)
    scan = Scan(
        name=scan_name,
        frames=[frame],
        poni=getattr(live_frame, "poni", None),
        integrator=integrator if integrator is not None else getattr(live_frame, "integrator", None),
    )
    # GUI never aborts a save on a per-frame degradation — opt INTO graceful
    # (the headless default is loud).
    result = run_reduction(plan, scan, strict=StrictPolicy.graceful())
    reduction = result.frames[int(live_frame.idx)]
    # Only overwrite the dimension the plan actually computed.  A 1D-only or
    # 2D-only plan (e.g. a GI 2D reintegrate, which sets integrate_1d=False so it
    # doesn't recompute a wrong full-range 1D) must PRESERVE the other dimension's
    # existing result instead of nulling/clobbering it with an absent result.
    if plan.integration_1d is not None:
        live_frame.int_1d = reduction.result_1d
    if plan.integration_2d is not None:
        live_frame.int_2d = reduction.result_2d
    live_frame.map_norm = _frame_norm(frame, plan)
    return live_frame


def reduce_live_frames(
    live_frames: Iterable[Any],
    plan: ReductionPlan,
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
    poni: Any = None,
    executor: Any = None,
    session: ReductionSession | None = None,
    cancel_token: Any = None,
    chunk_size: int | None = None,
    gi_freeze_mode: str | None = None,
) -> list[Any]:
    """Reduce a batch of ``LiveFrame`` objects through one headless run."""

    frames = list(live_frames)
    if not frames:
        return []
    headless_frames = [frame_from_live_frame(frame) for frame in frames]
    if session is None:
        # Only the one-shot path needs its own Scan + masked plan; a supplied
        # session already owns both (it was opened from the first chunk), so
        # building them here would be throwaway work.
        plan = _plan_with_mask_for_live_frame(plan, global_mask, frames[0])
        scan = Scan(
            name=scan_name,
            frames=headless_frames,
            poni=poni if poni is not None else getattr(frames[0], "poni", None),
            integrator=integrator if integrator is not None else getattr(frames[0], "integrator", None),
        )
        result = run_reduction(
            plan,
            scan,
            executor=executor,
            cancel_token=cancel_token,
            chunk_size=chunk_size or (len(frames) if executor is not None else 1),
            gi_freeze_mode=gi_freeze_mode,
            strict=StrictPolicy.graceful(),   # GUI never aborts a save
        )
        result_frames = result.frames
        active_plan = plan
    else:
        if session.execution == "streaming":
            for headless_frame in headless_frames:
                session.submit(headless_frame)
            session.drain()
        else:
            session.process(headless_frames)
        result_frames = session.frames
        active_plan = session.plan
    by_index = {int(frame.idx): frame for frame in frames}
    reduced_frames = []
    for headless_frame in headless_frames:
        live_frame = by_index[int(headless_frame.index)]
        reduction = result_frames.get(int(headless_frame.index))
        if reduction is None:
            continue
        # Only overwrite the dimension the plan computed (see reduce_live_frame):
        # a 2D-only GI reintegrate preserves the existing clean int_1d instead of
        # clobbering it with an absent/wrong 1D result.
        if active_plan.integration_1d is not None:
            live_frame.int_1d = reduction.result_1d
        if active_plan.integration_2d is not None:
            live_frame.int_2d = reduction.result_2d
        live_frame.map_norm = _frame_norm(headless_frame, active_plan)
        reduced_frames.append(live_frame)
    if session is not None:
        # S2 (serial flavor): a PERSISTENT session reused across a long
        # true-live watch run retains every harvested FrameReduction (full 2D
        # arrays) in session.frames — release the ones this call just copied
        # onto the LiveFrames so the session stays O(chunk), not O(scan).
        # getattr: older ssrl builds lack release_products (capability probe
        # covers the floor; harmless to skip there).
        release = getattr(session, "release_products", None)
        if callable(release):
            release(int(f.index) for f in headless_frames)
    # Same PERF-3 reasoning for the SINK-LESS paths (true-live serial,
    # reintegration, GI scouts): the session's registered Frames pin the raw
    # arrays; results are already copied onto the LiveFrames above, so drop
    # the session-side references now.
    for headless_frame in headless_frames:
        headless_frame.image = None
        headless_frame.background = None   # full bg images pin like raws
    return reduced_frames


def _build_live_scan_and_plan(
    live_frames: Iterable[Any],
    plan: ReductionPlan,
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
    poni: Any = None,
) -> tuple[Scan, ReductionPlan, int]:
    """Shared construction for the live session openers: materialize the live
    frames into a headless :class:`Scan` and fold the global mask into the plan.
    Returns ``(scan, plan, n_frames)``."""
    frames = list(live_frames)
    if not frames:
        raise ValueError("cannot open a session without frames")
    plan = _plan_with_mask_for_live_frame(plan, global_mask, frames[0])
    headless_frames = [frame_from_live_frame(frame) for frame in frames]
    scan = Scan(
        name=scan_name,
        frames=headless_frames,
        poni=poni if poni is not None else getattr(frames[0], "poni", None),
        integrator=(integrator if integrator is not None
                    else getattr(frames[0], "integrator", None)),
    )
    return scan, plan, len(frames)

def live_target_maps(plan, *, nexus_target=None, xye_target=None):
    """Per-mode applicability; XYE-only gives 1-D {xye} with NO store target."""
    from xrd_tools.session import required_result_modes
    targets, store = {}, {}
    for mode in required_result_modes(plan):
        applicable = []
        if nexus_target:
            applicable.append(nexus_target)
        if xye_target and mode.kind == "1d":
            applicable.append(xye_target)
        if not applicable:
            return None, None
        targets[mode] = tuple(applicable)
        store[mode] = (nexus_target,) if nexus_target else ()
    return (targets, store) if targets else (None, None)

def open_live_scan_session(
    live_frames: Iterable[Any],
    plan: ReductionPlan,
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
    poni: Any = None,
    executor: Any = None,
    cancel_token: Any = None,
    gi_freeze_mode: str | None = None,
    sink: Any = None,
    inflight_max: int | None = None,
    record_store: FrameRecordStore | None = None,
    record_store_persisted_on_write: bool = False,
    nexus_target: str | None = None,
    xye_target: str | None = None,
    accounting=None,
    xye_receipt_boundary=None,
):
    """Open a public :class:`xrd_tools.session.ScanSession` over xdart live
    frames (4f-bridge).

    Same Scan/plan construction as :func:`open_live_reduction_session`, but
    returns the headless commands-in / events-out ``ScanSession`` (which builds
    + arms its own streaming ``ReductionSession`` internally) instead of a raw
    ``ReductionSession``.  Streaming-only — the GUI live/batch write path.
    ``clear_frame_images=True`` preserves xdart's PERF-3 raw-nulling.
    """
    from xrd_tools.session import DynamicRunAccounting, ScanSession

    if type(accounting) is DynamicRunAccounting:
        if xye_target and xye_receipt_boundary is None:
            raise DynamicXyeReceiptBoundaryRequired(
                "dynamic XYE requires a transaction-qualified durable "
                "receipt boundary before session mutation"
            )
        if (xye_receipt_boundary is not None
                and not supports_durable_xye_receipts(xye_receipt_boundary)):
            raise DynamicXyeReceiptBoundaryRequired(
                "dynamic XYE receipt boundary must be the exact shared "
                "durable receipt owner"
            )
        accounting = accounting.ledger

    scan, plan, _n = _build_live_scan_and_plan(
        live_frames, plan, scan_name=scan_name, global_mask=global_mask,
        integrator=integrator, poni=poni)
    targets_by_mode, store_targets_by_mode = live_target_maps(
        plan, nexus_target=nexus_target, xye_target=xye_target)
    return ScanSession(
        plan,
        scan,
        sink=sink,
        executor=executor,
        inflight_max=inflight_max,
        gi_freeze_mode=gi_freeze_mode,
        cancel_token=cancel_token,
        clear_frame_images=True,
        record_store=record_store,
        record_store_persisted_on_write=record_store_persisted_on_write,
        targets_by_mode=targets_by_mode,
        store_targets_by_mode=store_targets_by_mode,
        accounting=accounting,
        # GUI never aborts a save (loud is the headless default).  Without this, the
        # streaming live/batch write path ran loud and a single degraded frame
        # (dead monitor / all-dummy 2D) aborted the whole-scan save (B-1 regression).
        strict=StrictPolicy.graceful(),
    )


def open_live_reduction_session(
    live_frames: Iterable[Any],
    plan: ReductionPlan,
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
    poni: Any = None,
    executor: Any = None,
    cancel_token: Any = None,
    chunk_size: int | None = None,
    gi_freeze_mode: str | None = None,
    sink: Any = None,
    execution: str = "chunked",
    inflight_max: int | None = None,
    accept_cb=None,
    outcome_cb=None,
) -> ReductionSession:
    """Open a persistent headless reducer for xdart live-frame chunks.

    The returned session owns the worker pool and per-thread pyFAI
    integrators.  Callers feed subsequent chunks with
    :func:`reduce_live_frames(..., session=session)` (chunked) or
    ``session.submit(frame)`` (``execution="streaming"``) and close it at the
    end of the scan/run.  Pass a ``sink`` (e.g. xdart's ``QtNexusSink``) to have
    the session drive the write itself instead of copying results back.
    """

    scan, plan, n_frames = _build_live_scan_and_plan(
        live_frames, plan, scan_name=scan_name, global_mask=global_mask,
        integrator=integrator, poni=poni)
    return ReductionSession(
        plan,
        scan,
        sink=sink,
        executor=executor,
        cancel_token=cancel_token,
        chunk_size=chunk_size or (n_frames if executor is not None else 1),
        gi_freeze_mode=gi_freeze_mode,
        execution=execution,
        inflight_max=inflight_max,
        accept_cb=accept_cb,
        outcome_cb=outcome_cb,
        strict=StrictPolicy.graceful(),   # GUI never aborts a save (loud is the
        # headless default; the GUI drops a bad frame per-frame and keeps going)

        # S2: streaming sink-driven sessions (the GUI batch/live path) consume
        # results through the sink (QtNexusSink hydrates LiveFrames + writes
        # the .nxs per frame) and never read result.frames — retaining every
        # FrameReduction (full 2D arrays) for the session's life was ~14 GB
        # on a 10k-frame 2D batch.  Chunked sessions KEEP retention: their
        # callers read results back via reduce_live_frames(session=...) →
        # session.frames (serial live, reintegration, GI scouts).
        retain_products=not (execution == "streaming" and sink is not None),
        # PERF-3 completion (pre-ship sweep): _register_process_frames keeps
        # every submitted Frame on session.scan for the session's life, and
        # Frame.image references the SAME array as LiveFrame.map_raw -- so
        # freeing the LiveFrame side (free_raw) never released the raw
        # (~18 MB/frame Eiger) while the session-side reference lived on.
        # The writer loop nulls frame.image post-write with this flag; any
        # later consumer reloads from the source path via Frame.load_image.
        clear_frame_images=True,
    )


def freeze_live_scan_gi_ranges(
    live_scan: Any,
    live_frames: Iterable[Any],
    *,
    scan_name: str = "scan",
    global_mask: Any = None,
    integrator: Any = None,
    poni: Any = None,
    integrate_1d: bool = True,
    integrate_2d: bool = True,
    gi_freeze_mode: str = "scout_union",
) -> ReductionPlan:
    """Freeze missing GI output ranges through the headless reducer.

    This is the xdart boundary adapter for the old GI scout step.  The actual
    scout integrations and common-grid calculation live in
    :class:`xrd_tools.reduction.ReductionSession`; xdart only mirrors the
    frozen plan ranges back into ``bai_1d_args`` / ``bai_2d_args`` so the
    existing writer and display state stay coherent.
    """

    frames = list(live_frames)
    if not frames:
        return plan_from_live_scan(
            live_scan,
            integrate_1d=integrate_1d,
            integrate_2d=integrate_2d,
        )
    plan = plan_from_live_scan(
        live_scan,
        integrate_1d=integrate_1d,
        integrate_2d=integrate_2d,
    )
    session = open_live_reduction_session(
        frames,
        plan,
        scan_name=scan_name,
        global_mask=global_mask,
        integrator=integrator,
        poni=poni,
        executor=None,
        chunk_size=len(frames),
        gi_freeze_mode=gi_freeze_mode,
    )
    try:
        frozen = session.plan
    finally:
        # Freeze-only session (no write sink) — close for cleanup; a GI scout
        # failure already surfaces as GIFreezeError, so don't fail-loud here.
        session.finish(raise_on_failure=False)
    _copy_frozen_gi_ranges_to_live_scan(live_scan, frozen)
    return frozen


def _copy_frozen_gi_ranges_to_live_scan(
    live_scan: Any,
    plan: ReductionPlan,
) -> None:
    if plan.gi is None:
        return
    if plan.integration_1d is not None:
        args_1d = getattr(live_scan, "bai_1d_args", None)
        if isinstance(args_1d, dict):
            from xrd_tools.integrate.gid import gi_1d_output_axis_key

            key = gi_1d_output_axis_key(plan.gi.mode_1d.value)
            value = getattr(plan.integration_1d, key, None)
            if value is not None and args_1d.get(key) is None:
                args_1d[key] = tuple(map(float, value))

    if plan.integration_2d is not None:
        args_2d = getattr(live_scan, "bai_2d_args", None)
        if not isinstance(args_2d, dict):
            return
        p2d = plan.integration_2d
        if plan.gi.mode_2d.value == "qip_qoop":
            ranges = {
                "x_range": p2d.extra.get("x_range"),
                "y_range": p2d.extra.get("y_range"),
            }
        else:
            ranges = {
                "radial_range": p2d.radial_range,
                "azimuth_range": p2d.azimuth_range,
            }
        for key, value in ranges.items():
            if value is not None and args_2d.get(key) is None:
                args_2d[key] = tuple(map(float, value))


# ---------------------------------------------------------------------------
# S3 + C1 helpers — used by every wrangler so the GI-vs-standard dispatch
# and the per-scan plan cache live in exactly one place.
# ---------------------------------------------------------------------------

_UNSET = object()  # sentinel: "mask_sig not supplied" (distinct from None)


def _mask_signature(mask: Any) -> Any:
    """Content digest of a detector mask (shape + dtype + size + sum +
    head/tail for numeric masks).  This is the O(N) part — it touches the
    whole array via ``np.sum`` — so callers in per-frame hot loops should
    memoize it by mask identity rather than recompute it every frame
    (see :meth:`StandardPlanCache._mask_sig_for`)."""
    if mask is None:
        return None
    arr = np.asarray(mask)
    flat = arr.ravel()
    if np.issubdtype(arr.dtype, np.number) and flat.size:
        mask_sum = float(np.sum(flat, dtype=np.float64))
        head = tuple(flat[:8].tolist())
        tail = tuple(flat[-8:].tolist())
    else:
        mask_sum = None
        head = ()
        tail = ()
    return (arr.shape, str(arr.dtype), int(arr.size), mask_sum, head, tail)


def _plan_signature(
    live_scan: Any,
    integrate_1d: bool,
    integrate_2d: bool,
    *,
    mask_sig: Any = _UNSET,
) -> tuple:
    """Hashable signature of the inputs that ``plan_from_live_scan`` reads.

    Used by :class:`StandardPlanCache` to skip plan rebuilds when nothing
    relevant on the scan has changed.  Covers the bai_*_args dicts
    (sorted) and a digest of ``global_mask``.

    ``mask_sig`` lets the caller pass an already-computed mask digest so
    the O(N) :func:`_mask_signature` isn't recomputed on every per-frame
    call; when omitted it's derived from ``live_scan.global_mask``.
    """
    def _items(args: Any) -> tuple:
        return tuple(
            sorted((str(key), repr(value)) for key, value in (args or {}).items())
        )

    if mask_sig is _UNSET:
        mask_sig = _mask_signature(getattr(live_scan, "global_mask", None))

    return (
        id(live_scan),
        bool(integrate_1d),
        bool(integrate_2d),
        bool(getattr(live_scan, "gi", False)),
        repr(getattr(live_scan, "incidence_motor", None)),
        repr(getattr(live_scan, "tilt_angle", None)),
        repr(getattr(live_scan, "sample_orientation", None)),
        _items(getattr(live_scan, "bai_1d_args", {})),
        _items(getattr(live_scan, "bai_2d_args", {})),
        mask_sig,
    )


class StandardPlanCache:
    """Per-owner cache for the standard (non-GI) :class:`ReductionPlan`.

    Wrappers (wranglers, integrator threads) keep one instance for the
    lifetime of a scan; the cached plan is rebuilt only when one of the
    scan settings ``_plan_signature`` covers actually changes.

    GI scans now get real headless plans too; callers may still pass
    ``None`` explicitly to the dispatch helper as an escape hatch for a
    known-legacy site, but the cache no longer forces that fork.
    """

    __slots__ = ("_plan", "_key", "_mask_obj", "_mask_sig", "_plan_builder")

    def __init__(self, plan_builder: Any = None) -> None:
        self._plan: ReductionPlan | None = None
        self._key: tuple | None = None
        self._plan_builder: Any = plan_builder
        # Memoized mask digest, keyed by the mask *object* (see below).
        self._mask_obj: Any = _UNSET
        self._mask_sig: Any = None

    @property
    def plan_builder(self) -> Any:
        return self._plan_builder

    @plan_builder.setter
    def plan_builder(self, builder: Any) -> None:
        if builder is self._plan_builder:
            return
        self._plan_builder = builder
        self.invalidate()

    def _mask_sig_for(self, mask: Any) -> Any:
        """Return the mask digest, recomputing the O(N) part only when the
        mask object itself changes.

        ``global_mask`` is built once per scan (detector mask + user mask)
        and *replaced* — not mutated in place — when the user swaps the
        mask file, so object identity is a sound proxy for "contents
        unchanged".  Holding a reference in ``_mask_obj`` also pins the id
        so it can't be reused by a later array.  This keeps the per-frame
        ``get()`` off the full-array ``np.sum`` that dominated mask digest
        cost on large detectors.
        """
        if mask is self._mask_obj:
            return self._mask_sig
        self._mask_obj = mask
        self._mask_sig = _mask_signature(mask)
        return self._mask_sig

    def get(
        self,
        live_scan: Any,
        *,
        integrate_1d: bool = True,
        integrate_2d: bool = True,
    ) -> ReductionPlan | None:
        builder = self._plan_builder or plan_from_live_scan
        prepare_scan = getattr(builder, "prepare_scan", None)
        if callable(prepare_scan):
            prepare_scan(live_scan)
        mask_sig = self._mask_sig_for(getattr(live_scan, "global_mask", None))
        builder_key = getattr(builder, "plan_cache_key", _UNSET)
        if builder_key is _UNSET:
            builder_key = id(builder)
        elif callable(builder_key):
            builder_key = builder_key()
        key = _plan_signature(
            live_scan, integrate_1d, integrate_2d, mask_sig=mask_sig,
        ) + (builder_key,)
        if self._plan is None or self._key != key:
            self._plan = builder(
                live_scan,
                integrate_1d=integrate_1d,
                integrate_2d=integrate_2d,
            )
            self._key = key
        return self._plan

    def invalidate(self) -> None:
        self._plan = None
        self._key = None
        self._mask_obj = _UNSET
        self._mask_sig = None


def sync_live_scan_gi_settings(
    live_scan: Any,
    *,
    incidence_motor: Any = None,
    sample_orientation: Any = None,
    tilt_angle: Any = None,
) -> None:
    """Mirror wrangler-thread GI settings onto a live scan before planning."""

    if not bool(getattr(live_scan, "gi", False)):
        return
    if incidence_motor is not None:
        live_scan.incidence_motor = incidence_motor
        live_scan.th_mtr = incidence_motor
    if sample_orientation is not None:
        live_scan.sample_orientation = sample_orientation
    if tilt_angle is not None:
        live_scan.tilt_angle = tilt_angle


def _source_path(frame: Any) -> Path | None:
    resolver = getattr(frame, "_resolved_source_path", None)
    path = resolver() if callable(resolver) else getattr(frame, "source_file", "")
    return Path(path) if path else None


def _incidence_available(live_scan: Any, incidence_motor: Any) -> bool:
    if incidence_motor is None:
        return False
    try:
        float(incidence_motor)
        return True
    except (TypeError, ValueError):
        pass
    key = str(incidence_motor).lower()
    scan_data = getattr(live_scan, "scan_data", None)
    if scan_data is not None and hasattr(scan_data, "columns"):
        if any(str(col).lower() == key for col in scan_data.columns):
            return True
    frames = getattr(live_scan, "frames", None)
    for idx in list(getattr(frames, "index", []) or []):
        try:
            info = getattr(frames[int(idx)], "scan_info", {}) or {}
        except Exception:
            continue
        if any(str(candidate).lower() == key for candidate in info):
            return True
    return False


def _scan_data_row(scan_data: Any, idx: int) -> dict[str, Any]:
    if scan_data is None or not hasattr(scan_data, "loc"):
        return {}
    try:
        row = scan_data.loc[int(idx)]
    except (KeyError, TypeError, ValueError):
        return {}
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]
    try:
        return {
            str(key): value
            for key, value in row.to_dict().items()
        }
    except AttributeError:
        return {}


def _frame_norm(frame: Frame, plan: ReductionPlan) -> float:
    if frame.normalization_factor is not None:
        return float(frame.normalization_factor)
    integration = plan.integration_1d or plan.integration_2d
    if integration and integration.monitor_key:
        norm = resolve_monitor_norm(frame.metadata, integration.monitor_key)
        return norm if norm is not None else 1.0
    return 1.0


def _plan_with_mask_for_live_frame(
    plan: ReductionPlan,
    global_mask: Any,
    live_frame: Any,
) -> ReductionPlan:
    shape = getattr(getattr(live_frame, "map_raw", None), "shape", None)
    gmask = _flat_mask_as_bool(global_mask, shape)
    plan_mask = _flat_mask_as_bool(plan.mask, shape)
    if plan_mask is None:
        return replace(plan, mask=gmask)
    if gmask is None:
        return replace(plan, mask=plan_mask)
    return replace(plan, mask=plan_mask | gmask)


def _live_frame_mask_for_frame(live_frame: Any) -> np.ndarray | MaskSpec | None:
    mask = getattr(live_frame, "mask", None)
    image = getattr(live_frame, "map_raw", None)
    shape = getattr(image, "shape", None)
    return _mask_for_frame(mask, shape)


def _mask_for_frame(mask: Any, shape: tuple[int, int] | None) -> np.ndarray | MaskSpec | None:
    if mask is None:
        return None
    if isinstance(mask, MaskSpec):
        if shape is None:
            return mask
        try:
            mask.to_bool(shape)
        except ValueError as exc:
            logger.warning("Ignoring mask: %s", exc)
            return None
        return mask
    arr = np.asarray(mask)
    if arr.ndim == 1:
        if shape is None:
            return MaskSpec(arr)
        n_pixels = int(np.prod(shape))
        if arr.dtype == bool:
            if arr.size != n_pixels:
                logger.warning(
                    "Ignoring boolean mask: length %d does not match image shape %s.",
                    arr.size, shape,
                )
                return None
            return MaskSpec(arr)
        flat = np.asarray(arr, dtype=int).ravel()
        if flat.size and (flat.min() < 0 or flat.max() >= n_pixels):
            logger.warning(
                "Ignoring mask: flat indices out of bounds for image shape %s "
                "(index range [%d, %d], image has %d pixels).",
                shape, int(flat.min()), int(flat.max()), n_pixels,
            )
            return None
        return MaskSpec(arr)
    return _flat_mask_as_bool(mask, shape)


def _npt_2d(args_2d: dict[str, Any]) -> tuple[int, int]:
    npt = args_2d.pop("npt", None)
    if isinstance(npt, (tuple, list)) and len(npt) == 2:
        return int(npt[0]), int(npt[1])
    npt_rad = args_2d.pop("npt_rad", None)
    npt_azim = args_2d.pop("npt_azim", None)
    if npt_rad is None:
        npt_rad = npt if npt is not None else 1000
    if npt_azim is None:
        npt_azim = 360
    return int(npt_rad), int(npt_azim)


def _pop_first(args: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for key in keys:
        if key in args:
            return args.pop(key)
    return default


def _strip_nonstandard_args(args: dict[str, Any]) -> None:
    for key in _GI_ONLY_ARGS:
        args.pop(key, None)


def _gi_1d_unit_default(unit: Any, mode: str, *, is_gi: bool) -> str:
    if not is_gi:
        return str(unit or "q_A^-1")
    if mode == "q_ip":
        return "qip_A^-1"
    if mode == "q_oop":
        return "qoop_A^-1"
    return str(unit or "q_A^-1")


def _gi_2d_unit_default(unit: Any, mode: str, *, is_gi: bool) -> str:
    text = str(unit or "").strip()
    if not is_gi:
        return text or "q_A^-1"
    if mode == "qip_qoop":
        return text if text.startswith("qip_") else "qip_A^-1"
    return text or "q_A^-1"


__all__ = [
    "DynamicXyeReceiptBoundaryRequired",
    "LiveScanNexusSession",
    "StandardPlanCache",
    "ThresholdSaturationConfig",
    "apply_threshold_saturation_to_plan",
    "bad_pixel_counts",
    "compute_bad_pixel_mask",
    "apply_frozen_run_configuration",
    "frame_from_live_frame",
    "scan_from_live_scan",
    "write_live_scan_to_nexus",
    "plan_from_live_scan",
    "reduce_live_frame",
    "reduce_live_frames",
    "open_live_reduction_session",
    "open_live_scan_nexus_session",
    "live_target_maps",
    "open_live_scan_session",
    "freeze_live_scan_gi_ranges",
    "sync_live_scan_gi_settings",
]
