# -*- coding: utf-8 -*-
"""Silent-by-default structured logging for run-configuration ownership.

R4-D diagnostics deliberately inspect every mutable carrier involved in a run
boundary.  They are gated by ``XDART_RUN_CONFIG_DEBUG`` so the normal GUI does
not traverse Qt widgets, parameter trees, or scan state for logging.

O-2 adds the two decisions the R4-D stock-take found uninstrumented, on THIS
channel (one gate, one owner — a second diagnostics owner is forbidden):

* :func:`display_context_transition_log` — the ownership snapshot at each of the
  five display-context boundaries (:data:`DISPLAY_CONTEXT_PHASES`).  It carries
  the TARGET-object identity, so a trace shows directly whether a browse builds
  its own scan or repoints the acquisition singleton.
* :func:`fail_closed_rejection_log` — every context-qualified projection or
  availability rejection that blanks a panel, with the reason and the expected
  versus found owner identity.  Before this, a blank panel was silent.

Both emit identity only: object ids, keys, paths, shapes, dtypes, counts and
generations.  No array, no live Qt object, and no unbounded sequence is ever
serialized, so the channel stays safe to leave on for a whole beamline session.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass


def run_config_debug_enabled() -> bool:
    value = os.environ.get("XDART_RUN_CONFIG_DEBUG", "")
    return value.lower() not in ("", "0", "false", "no", "off")


def bump_run_config_debug_generation(owner, kind: str) -> int | None:
    """Advance a diagnostic-only run/config generation when tracing is on."""
    if not run_config_debug_enabled() or owner is None:
        return None
    attr = f"_run_config_debug_{str(kind)}_generation"
    try:
        generation = int(getattr(owner, attr, 0) or 0) + 1
        setattr(owner, attr, generation)
        return generation
    except Exception:
        return None


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(val) for val in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        try:
            return [_jsonable(val) for val in value]
        except TypeError:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _safe_attr(owner, name, default=None):
    try:
        return getattr(owner, name, default)
    except Exception:
        return default


def _parameter_value(parameters, *paths):
    for path in paths:
        try:
            return parameters.child(*path).value()
        except Exception:
            continue
    return None


def _object_identity(value) -> dict:
    if value is None:
        return {"present": False, "object_id": None, "type": None}
    shape = _safe_attr(value, "shape")
    try:
        shape = list(shape) if shape is not None else None
    except TypeError:
        shape = str(shape)
    return {
        "present": True,
        "object_id": hex(id(value)),
        "type": type(value).__name__,
        "shape": shape,
        "dtype": (
            str(_safe_attr(value, "dtype"))
            if _safe_attr(value, "dtype") is not None
            else None
        ),
    }


def _path_identity(value) -> dict:
    text = "" if value is None else str(value)
    return {
        "configured": text,
        "absolute": os.path.abspath(os.path.expanduser(text)) if text else "",
    }


def _scan_key(scan):
    name = _safe_attr(scan, "name")
    if name not in (None, "", "null_main"):
        return name
    data_file = _safe_attr(scan, "data_file")
    return data_file or None


def _scan_summary(scan) -> dict:
    if scan is None:
        return {"present": False}
    a1 = dict(_safe_attr(scan, "bai_1d_args", {}) or {})
    a2 = dict(_safe_attr(scan, "bai_2d_args", {}) or {})
    return {
        "present": True,
        "object_id": hex(id(scan)),
        "scan_key": _scan_key(scan),
        "name": _safe_attr(scan, "name"),
        "data_file": _path_identity(_safe_attr(scan, "data_file")),
        "gi": bool(_safe_attr(scan, "gi", False)),
        "gi_config": dict(_safe_attr(scan, "gi_config", {}) or {}),
        "incidence_motor": _safe_attr(scan, "incidence_motor"),
        "th_mtr": _safe_attr(scan, "th_mtr"),
        "sample_orientation": _safe_attr(scan, "sample_orientation"),
        "tilt_angle": _safe_attr(scan, "tilt_angle"),
        "gi_mode_1d": a1.get("gi_mode_1d"),
        "gi_mode_2d": a2.get("gi_mode_2d"),
        "unit_1d": a1.get("unit"),
        "unit_2d": a2.get("unit"),
        "poni": _object_identity(_safe_attr(scan, "_cached_poni")),
        "global_mask": _object_identity(_safe_attr(scan, "global_mask")),
    }


def _controls_summary(widget) -> dict:
    if widget is None:
        return {"present": False}
    panel = _safe_attr(widget, "controls_v2")
    gi_values = []
    if panel is not None:
        try:
            gi_values = [
                edit.value
                for edit in panel.current_form_edits()
                if tuple(edit.path) == ("GI", "Grazing")
            ]
        except Exception:
            gi_values = []

    state = _safe_attr(widget, "_controls_v2_threshold_state")
    threshold = dict(state) if isinstance(state, dict) else None
    poni_path = ""
    get_poni_path = _safe_attr(widget, "_controls_v2_poni_path")
    if callable(get_poni_path):
        try:
            poni_path = get_poni_path()
        except Exception:
            poni_path = ""
    wrangler = _safe_attr(widget, "wrangler")
    parameters = _safe_attr(wrangler, "parameters")
    return {
        "present": panel is not None,
        "gi_visible_values": gi_values,
        "threshold": threshold,
        "poni_file": _path_identity(poni_path),
        "mask_file": _path_identity(
            _parameter_value(parameters, ("Signal", "mask_file"))
            if parameters is not None
            else ""
        ),
    }


def _carrier_summary(owner) -> dict:
    if owner is None:
        return {"present": False}
    return {
        "present": True,
        "object_id": hex(id(owner)),
        "gi": _safe_attr(owner, "gi"),
        "incidence_motor": _safe_attr(owner, "incidence_motor"),
        "sample_orientation": _safe_attr(owner, "sample_orientation"),
        "tilt_angle": _safe_attr(owner, "tilt_angle"),
        "threshold": {
            "apply": _safe_attr(owner, "apply_threshold"),
            "min": _safe_attr(owner, "threshold_min"),
            "max": _safe_attr(owner, "threshold_max"),
            "mask_saturation": _safe_attr(owner, "mask_sentinel"),
        },
        "poni_file": _path_identity(_safe_attr(owner, "poni_file")),
        "poni": _object_identity(_safe_attr(owner, "poni")),
        "mask_file": _path_identity(_safe_attr(owner, "mask_file")),
        "scan": _scan_summary(_safe_attr(owner, "scan")),
    }


def _wrangler_summary(wrangler) -> dict:
    if wrangler is None:
        return {"present": False}
    parameters = _safe_attr(wrangler, "parameters")
    hidden = {
        "gi": _parameter_value(parameters, ("GI", "Grazing")),
        "incidence_motor": _parameter_value(parameters, ("GI", "th_motor")),
        "sample_orientation": _parameter_value(
            parameters, ("GI", "sample_orientation")
        ),
        "tilt_angle": _parameter_value(parameters, ("GI", "tilt_angle")),
        "threshold": {
            "apply": _parameter_value(parameters, ("Mask", "Threshold")),
            "min": _parameter_value(parameters, ("Mask", "min")),
            "max": _parameter_value(parameters, ("Mask", "max")),
            "mask_saturation": _parameter_value(
                parameters, ("MaskSat", "mask_sentinel")
            ),
        },
        "poni_file": _path_identity(
            _parameter_value(
                parameters,
                ("Signal", "poni_file"),
                ("Calibration", "poni_file"),
            )
        ),
        "mask_file": _path_identity(
            _parameter_value(parameters, ("Signal", "mask_file"))
        ),
    }
    summary = _carrier_summary(wrangler)
    summary["hidden_parameters"] = hidden
    summary["worker"] = _carrier_summary(_safe_attr(wrangler, "thread"))
    return summary


def _generations(widget) -> dict:
    if widget is None:
        return {}
    display = _safe_attr(widget, "displayframe")
    publication = _safe_attr(widget, "publication_store")
    viewer = _safe_attr(widget, "h5viewer")
    return {
        "diagnostic_run": _safe_attr(
            widget, "_run_config_debug_run_generation"
        ),
        "diagnostic_config": _safe_attr(
            widget, "_run_config_debug_config_generation"
        ),
        "runend": _safe_attr(widget, "_runend_generation"),
        "display": _safe_attr(display, "display_generation"),
        "publication": _safe_attr(publication, "generation"),
        "h5_load": _safe_attr(viewer, "_load_generation"),
    }


def parameter_change_summary(changes) -> list[dict]:
    """Return compact pyqtgraph parameter-tree changes for R4-D traces."""
    result = []
    for change in changes or ():
        try:
            param, change_type, value = change[:3]
        except (TypeError, ValueError):
            result.append({"change": str(change)})
            continue
        path = []
        current = param
        for _ in range(12):
            if current is None:
                break
            try:
                path.append(str(current.name()))
            except Exception:
                break
            try:
                current = current.parent()
            except Exception:
                break
        result.append({
            "path": list(reversed(path)),
            "change_type": change_type,
            "value": _jsonable(value),
        })
    return result


def run_config_debug_log(
    logger,
    event: str,
    *,
    widget=None,
    wrangler=None,
    origin: str = "",
    level: str = "info",
    **fields,
) -> None:
    """Log one ownership snapshot, doing no inspection when tracing is off."""
    if not run_config_debug_enabled():
        return
    wrangler = wrangler if wrangler is not None else _safe_attr(widget, "wrangler")
    scan = _safe_attr(widget, "scan")
    payload = {
        "event": event,
        "origin": origin,
        "t": round(time.monotonic(), 6),
        "run_active": bool(_safe_attr(widget, "_run_active", False)),
        "generations": _generations(widget),
        "controls": _controls_summary(widget),
        "shared_scan": _scan_summary(scan),
        "wrangler": _wrangler_summary(wrangler),
    }
    payload.update({str(key): _jsonable(val) for key, val in fields.items()})
    message = "RUN_CONFIG_DEBUG " + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    log = getattr(logger, level, None)
    if not callable(log):
        log = logger.info
    log(message)


# --------------------------------------------------------------------------- #
# O-2: display-context transitions and fail-closed rejections
# --------------------------------------------------------------------------- #

#: The five boundaries a display context can move across.  ``rescope`` is the
#: frame-driven scan-boundary reset (the site that clears the publication
#: store), not a user gesture.
DISPLAY_CONTEXT_PHASES = (
    "pause",
    "browse_load_start",
    "browse_load_finish",
    "resume",
    "rescope",
)

#: Attribute a live run's ``FrameRecordStore`` carries to declare its scan.
#: Duplicated from ``frame_projection_adapter.STORE_SCAN_KEY_ATTR`` so this
#: module stays import-free of the GUI display stack (it is imported by the
#: worker-side ``scan_threads`` too).
_STORE_SCAN_KEY_ATTR = "_xdart_scan_key"

#: Diagnostic-only event names, so a trace consumer can filter without
#: string-matching prose.
DISPLAY_CONTEXT_TRANSITION_EVENT = "display_context_transition"
FAIL_CLOSED_REJECTION_EVENT = "fail_closed_rejection"

#: The canonical fail-closed decision names.  O-2.1 (§61.4 C): the taxonomy has
#: to be TRUTHFUL, so "no publication exists" is a different decision from "a
#: publication exists under the wrong owner", and a record-store candidate that
#: was skipped while the publication fallback still served is NOT a blanking
#: event.  Every record carries ``blanks_panel`` so a reader never has to infer
#: that from the decision name.
DECISION_RECORD_STORE_SKIPPED = "record_store_skipped_owner_mismatch"
DECISION_PUBLICATION_ABSENT = "publication_absent"
DECISION_PUBLICATION_OWNER_MISMATCH = "publication_owner_mismatch"
DECISION_PROJECTION_STORE_ABSENT = "projection_store_absent"
DECISION_PROJECTION_SUPERSEDED = "projection_superseded"
DECISION_CAPABILITY_FORCES_CLEAR = "capability_forces_clear"

#: The ONE name a context-qualified hydration rejection must use.  Nothing emits
#: it at this tip: hydration completions carry a generation and no owner, which
#: is the defect O-2's reproducer pins and O-3 fixes.  It is named here so the
#: reproducer can require the exact decision instead of matching a permissive
#: set, and so O-3 has no naming latitude when it starts emitting it.
DECISION_HYDRATION_CONTEXT_MISMATCH = "hydration_context_mismatch"

FAIL_CLOSED_DECISIONS = (
    DECISION_RECORD_STORE_SKIPPED,
    DECISION_PUBLICATION_ABSENT,
    DECISION_PUBLICATION_OWNER_MISMATCH,
    DECISION_PROJECTION_STORE_ABSENT,
    DECISION_PROJECTION_SUPERSEDED,
    DECISION_CAPABILITY_FORCES_CLEAR,
    DECISION_HYDRATION_CONTEXT_MISMATCH,
)

#: Process-unique operation counter.  A browse load crosses a thread boundary,
#: so start and finish can only be paired by a token that is minted once and
#: echoed — wall-clock ordering does not survive a queued task.
_operation_counter = itertools.count(1)


@dataclass(frozen=True, slots=True)
class DiagnosticRunIdentity:
    """A DETACHED snapshot of which run/config a display operation happened under.

    Strings and integers only.  The browse chain needs the accepted run identity
    to qualify its transitions, and the static widget is not reachable from
    there; handing the viewer this frozen tuple keeps the qualification without
    giving a background seam a live widget, store, scan or callback to hold
    (§61.4 B).
    """

    run_generation: int | None = None
    #: The ADMITTED ``FrozenRunConfiguration``'s own generation and content
    #: fingerprint, read from the wrapper's admission LEDGER -- O-2.2R
    #: (§65.3.6).  Never read from a public carrier, so a foreign object
    #: assigned there cannot have its identity published as accepted.
    config_generation: int | None = None
    config_fingerprint: str = ""
    #: True only when all FOUR references -- wrapper carrier, wrapper ledger,
    #: worker carrier, worker ledger -- are the same object.  Comparing the two
    #: carriers alone reported an equal-valued foreign configuration as
    #: consistent, because equality here is by content, not by admission.
    config_consistent: bool = False
    runend_generation: int | None = None
    display_generation: int | None = None
    run_active: bool = False
    acquisition_key: str = ""
    acquisition_object_id: str = ""

    def as_fields(self) -> dict:
        return {
            "run_generation": self.run_generation,
            "config_generation": self.config_generation,
            "config_fingerprint": self.config_fingerprint,
            "config_consistent": self.config_consistent,
            "runend_generation": self.runend_generation,
            "display_generation": self.display_generation,
            "run_active": self.run_active,
            "acquisition_key": self.acquisition_key,
            "acquisition_object_id": self.acquisition_object_id,
        }


def capture_diagnostic_run_identity(widget) -> DiagnosticRunIdentity | None:
    """Freeze the widget's current run/config identity, or ``None`` when off."""
    if not run_config_debug_enabled() or widget is None:
        return None
    try:
        display = _safe_attr(widget, "displayframe")
        acquisition = _safe_attr(widget, "_x1_run_scan_capture")
        wrangler = _safe_attr(widget, "wrangler")
        worker = _safe_attr(wrangler, "thread")
        # O-2.2R (§65.3.6): the ADMISSION LEDGER is the authority, and all four
        # references must be the same object.  ``FrozenRunConfiguration``
        # equality is by content, so an equal-valued foreign object assigned to
        # the public carriers is indistinguishable from the admitted one by
        # value -- only identity against the ledgers proves it passed the gate.
        accepted = _safe_attr(wrangler, "_admitted_run_configuration")
        config_generation, config_fingerprint = None, ""
        if accepted is not None:
            try:
                config_generation, config_fingerprint = accepted.identity
            except Exception:
                config_generation, config_fingerprint = None, ""
        # Deliberately published from the LEDGER, never from a carrier: a
        # foreign carrier must not be able to have its generation/fingerprint
        # reported as the accepted identity.
        config_consistent = bool(
            accepted is not None
            and _safe_attr(wrangler, "run_configuration") is accepted
            and _safe_attr(worker, "run_configuration") is accepted
            and _safe_attr(worker, "_admitted_run_configuration") is accepted)
        return DiagnosticRunIdentity(
            run_generation=_safe_attr(
                widget, "_run_config_debug_run_generation"),
            config_generation=config_generation,
            config_fingerprint=str(config_fingerprint or ""),
            config_consistent=config_consistent,
            runend_generation=_safe_attr(widget, "_runend_generation"),
            display_generation=_safe_attr(display, "display_generation"),
            run_active=bool(_safe_attr(widget, "_run_active", False)),
            acquisition_key=_owner_identity(_scan_key(acquisition)),
            acquisition_object_id=(
                "" if acquisition is None else hex(id(acquisition))),
        )
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class DisplayContextOperation:
    """One browse load's immutable diagnostic identity.

    Minted at ``browse_load_start`` and echoed verbatim at
    ``browse_load_finish``, so a reader can pair the two halves of an
    asynchronous load without guessing from timestamps.  ``kind`` separates the
    operator's own browse from the application's internal run-output rewiring:
    both are real transitions, but only the first is a user browse, and the
    O-2.1 acceptance assertion must not be satisfiable by an internal reload
    (§61.4 B).
    """

    token: str
    kind: str
    requested_path: str
    load_generation: int | None
    identity: DiagnosticRunIdentity | None
    previous_scan_key: str = ""
    previous_data_file: str = ""

    def as_fields(self) -> dict:
        fields = {
            "token": self.token,
            "kind": self.kind,
            "requested_path": self.requested_path,
            "load_generation": self.load_generation,
            "previous_scan_key": self.previous_scan_key,
            "previous_data_file": self.previous_data_file,
        }
        fields["identity"] = (
            None if self.identity is None else self.identity.as_fields())
        return fields


def new_display_context_operation(
    *,
    kind: str,
    requested_path,
    load_generation=None,
    identity: DiagnosticRunIdentity | None = None,
    previous_scan=None,
) -> DisplayContextOperation:
    """Mint one browse-load operation identity.  Callers gate on the channel."""
    return DisplayContextOperation(
        token=f"{os.getpid():x}-{next(_operation_counter):x}",
        kind=str(kind),
        requested_path=("" if requested_path is None else str(requested_path)),
        load_generation=load_generation,
        identity=identity,
        previous_scan_key=_owner_identity(_scan_key(previous_scan)),
        previous_data_file=_owner_identity(
            _safe_attr(previous_scan, "data_file")),
    )


def _owner_identity(value) -> str:
    """One short, comparable spelling of an ownership key.

    Never raises and never returns a live object: a rejection record has to be
    readable next to another record's owner, and nothing more.
    """
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return "<unprintable>"


def _store_identity(store, kind: str) -> dict:
    """Identity, declared owner and generation of one frame store.

    Counts are read through ``len`` only — the records themselves are never
    touched, so this cannot hydrate anything or serialize a payload.
    """
    if store is None:
        return {"kind": kind, "present": False}
    items = _safe_attr(store, "_items")
    if items is None:
        items = _safe_attr(store, "_records")
    try:
        count = len(items) if items is not None else None
    except Exception:
        count = None
    return {
        "kind": kind,
        "present": True,
        "object_id": hex(id(store)),
        "type": type(store).__name__,
        "owner": _owner_identity(_safe_attr(store, _STORE_SCAN_KEY_ATTR)),
        "generation": _safe_attr(store, "generation"),
        "count": count,
    }


def _context_scan_identity(scan, role: str) -> dict:
    """The ownership facts one scan object carries at a context boundary.

    This is deliberately NARROWER than :func:`_scan_summary`: it is the set a
    reviewer needs to answer "is this the same owner as the previous event, and
    does it still hold its own calibration?" — object identity, canonical key,
    file path, GI identity, and the PONI/mask identities (never their arrays).
    """
    if scan is None:
        return {"role": role, "present": False}
    a1 = dict(_safe_attr(scan, "bai_1d_args", {}) or {})
    a2 = dict(_safe_attr(scan, "bai_2d_args", {}) or {})
    return {
        "role": role,
        "present": True,
        "object_id": hex(id(scan)),
        "scan_key": _owner_identity(_scan_key(scan)),
        "name": _safe_attr(scan, "name"),
        "data_file": _path_identity(_safe_attr(scan, "data_file")),
        "gi": {
            "enabled": bool(_safe_attr(scan, "gi", False)),
            "incidence_motor": _safe_attr(scan, "incidence_motor"),
            "sample_orientation": _safe_attr(scan, "sample_orientation"),
            "tilt_angle": _safe_attr(scan, "tilt_angle"),
            "gi_mode_1d": a1.get("gi_mode_1d"),
            "gi_mode_2d": a2.get("gi_mode_2d"),
        },
        "poni": _object_identity(_safe_attr(scan, "_cached_poni")),
        "global_mask": _object_identity(_safe_attr(scan, "global_mask")),
        "cached_data_mask": _object_identity(
            _safe_attr(scan, "_cached_data_mask")),
    }


def display_context_summary(
    widget=None,
    *,
    target=None,
    scan=None,
    record_store=None,
    publication_store=None,
) -> dict:
    """Bounded ownership snapshot for one display-context transition.

    ``target`` is the object the transition is about to write to (the file
    thread's scan on a browse load, for instance).  Logging it beside the
    acquisition/display/viewer roles is the whole point: when one ``object_id``
    appears under every role, the trace has recorded a singleton mutation
    rather than a context switch.

    Tolerates ``widget=None`` so the browse chain — the viewer and the file
    thread, neither of which owns the static widget — emits the same shape;
    those callers pass their own ``scan``/``target``/store handles instead.
    """
    display = _safe_attr(widget, "displayframe")
    viewer = _safe_attr(widget, "h5viewer")
    acquisition = _safe_attr(widget, "_x1_run_scan_capture")
    if record_store is None:
        record_store = _safe_attr(widget, "_frame_record_store")
    if record_store is None:
        thread = _safe_attr(_safe_attr(widget, "wrangler"), "thread")
        record_store = _safe_attr(thread, "_streaming_record_store")
    if publication_store is None:
        publication_store = _safe_attr(widget, "publication_store")
    if publication_store is None:
        publication_store = _safe_attr(display, "publication_store")
    return {
        "run_active": bool(_safe_attr(widget, "_run_active", False)),
        "run_generation": _safe_attr(
            widget, "_run_config_debug_run_generation"),
        "config_generation": _safe_attr(
            widget, "_run_config_debug_config_generation"),
        "runend_generation": _safe_attr(widget, "_runend_generation"),
        "display_generation": _safe_attr(display, "display_generation"),
        "load_generation": _safe_attr(viewer, "_load_generation"),
        "acquisition": _context_scan_identity(acquisition, "acquisition"),
        "shared": _context_scan_identity(
            scan if scan is not None else _safe_attr(widget, "scan"), "shared"),
        "display_scan": _context_scan_identity(
            _safe_attr(display, "scan"), "display"),
        "viewer_scan": _context_scan_identity(
            _safe_attr(viewer, "scan"), "viewer"),
        "target": _context_scan_identity(target, "target"),
        "record_store": _store_identity(record_store, "frame_record_store"),
        "publication_store": _store_identity(
            publication_store, "publication_store"),
    }


def display_context_transition_log(
    logger,
    phase: str,
    *,
    widget=None,
    origin: str = "",
    target=None,
    scan=None,
    record_store=None,
    publication_store=None,
    operation: DisplayContextOperation | None = None,
    level: str = "info",
    **fields,
) -> None:
    """Record one DISPLAY-CONTEXT-TRANSITION on the run-config channel.

    ``phase`` is one of :data:`DISPLAY_CONTEXT_PHASES`.  Emitting an unknown
    phase is not an error (a diagnostic must never break a boundary), but it is
    flagged in the payload so a malformed trace is visible rather than silent.

    ``operation`` carries the browse-load pair's immutable identity.  When the
    caller has no widget (the browse chain), its ``identity`` snapshot is also
    folded into the context block, so a browse record is generation-qualified
    instead of reporting nulls for every generation (§61.4 B).
    """
    if not run_config_debug_enabled():
        return
    try:
        context = display_context_summary(
            widget, target=target, scan=scan,
            record_store=record_store,
            publication_store=publication_store)
        if widget is None and operation is not None and operation.identity:
            for key, value in operation.identity.as_fields().items():
                if key in context and context[key] is None:
                    context[key] = value
                elif key not in context:
                    context[key] = value
            if context.get("load_generation") is None:
                context["load_generation"] = operation.load_generation
        payload = {
            "phase": str(phase),
            "phase_known": str(phase) in DISPLAY_CONTEXT_PHASES,
            "context": context,
            "operation": (
                None if operation is None else operation.as_fields()),
        }
        payload.update(fields)
        run_config_debug_log(
            logger,
            DISPLAY_CONTEXT_TRANSITION_EVENT,
            widget=widget,
            origin=origin,
            level=level,
            **payload,
        )
    except Exception:
        # A boundary must survive its own instrumentation.
        pass


def fail_closed_rejection_log(
    logger,
    decision: str,
    *,
    reason: str,
    outcome: str,
    blanks_panel: bool,
    expected=None,
    found=None,
    origin: str = "",
    level: str = "info",
    **fields,
) -> None:
    """Record one FAIL-CLOSED REJECTION.

    ``decision`` must be one of :data:`FAIL_CLOSED_DECISIONS`; ``expected`` and
    ``found`` are the owner identities the gate compared.  ``outcome`` and
    ``blanks_panel`` are REQUIRED (§61.4 C): a record-store candidate that was
    skipped while the publication fallback still served is a real rejection but
    blanks nothing, and calling both cases the same event made the taxonomy
    untruthful.  Deliberately does NOT take a widget: these fire inside
    per-frame lookup paths, so the record stays a flat identity tuple with no
    traversal behind it.
    """
    if not run_config_debug_enabled():
        return
    try:
        payload = {
            "event": FAIL_CLOSED_REJECTION_EVENT,
            "decision": str(decision),
            "decision_known": str(decision) in FAIL_CLOSED_DECISIONS,
            "reason": str(reason),
            "outcome": str(outcome),
            "blanks_panel": bool(blanks_panel),
            "expected_owner": _owner_identity(expected),
            "found_owner": _owner_identity(found),
            "origin": origin,
            "t": round(time.monotonic(), 6),
        }
        payload.update(
            {str(key): _jsonable(val) for key, val in fields.items()})
        message = "RUN_CONFIG_DEBUG " + json.dumps(
            payload, sort_keys=True, separators=(",", ":"))
        log = getattr(logger, level, None)
        if not callable(log):
            log = logger.info
        log(message)
    except Exception:
        pass
