# -*- coding: utf-8 -*-
"""
@author: walroth, thampy
"""

# Standard library imports
import logging
from queue import Queue
import threading
import copy
import os
import math
import time
import types
from pathlib import Path
from collections import OrderedDict
import gc
import imageio
import pyFAI

from .browse_debug import browse_debug_enabled, browse_debug_log, sequence_summary
from .run_config_debug import (
    bump_run_config_debug_generation,
    run_config_debug_log,
)

logger = logging.getLogger(__name__)
_ORPHANED_STITCH_THREADS = []
_XYE_REFRESH_COALESCE_MS = 50
_XYE_REFRESH_RETRY_MS = (150, 500, 1500, 3000, 7000)


def _retain_orphaned_close_thread(thread) -> None:
    """Keep a slow close-time QThread wrapper alive until Qt finishes it."""
    if thread in _ORPHANED_STITCH_THREADS:
        return
    _ORPHANED_STITCH_THREADS.append(thread)

    def _forget_thread():
        try:
            _ORPHANED_STITCH_THREADS.remove(thread)
        except ValueError:
            pass

    try:
        thread.finished.connect(_forget_thread)
    except Exception:
        pass


def _retain_orphaned_stitch_thread(thread) -> None:
    """Keep a slow stitch QThread wrapper alive until Qt finishes it."""
    _retain_orphaned_close_thread(thread)


def _runend_waterfall_history_fields(displayframe) -> dict:
    history = getattr(displayframe, "_waterfall_history", None)
    ids = tuple(getattr(history, "ids", ()) or ())
    return {
        "waterfall_count": int(getattr(history, "count", 0) or 0),
        "waterfall_tail": list(ids[-3:]),
    }


def _processed_count_for_output(thread, output_file, run_total):
    """Return the processed count scoped to ``output_file`` when available."""
    counts = None
    for attr in ("files_processed_by_output",
                 "_last_files_processed_by_output"):
        value = getattr(thread, attr, None)
        if value is not None:
            counts = value
            break
    if counts is None:
        return run_total
    try:
        target = os.path.normcase(os.path.abspath(os.fspath(output_file)))
    except (TypeError, ValueError):
        return None
    try:
        items = counts.items()
    except AttributeError:
        logger.debug("invalid files_processed_by_output value: %r", counts)
        return None
    for path, value in items:
        try:
            key = os.path.normcase(os.path.abspath(os.fspath(path)))
        except (TypeError, ValueError):
            continue
        if key != target:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.debug("invalid per-output processed count: %r", value)
            return None
    logger.debug("no processed count recorded for output %s", output_file)
    return None


def _last_processed_output_file(thread, fallback=None):
    """Return the last output that actually accepted a frame this run.

    A live Directory worker can discover and initialize scan N+1 before Stop
    lands even though scan N supplied the last processed frame.  ``thread.fname``
    then names an empty ahead-of-data file.  The per-output accounting map is
    insertion ordered and records only successful dispatches, so its final
    positive entry is the authoritative run-end display target.
    """
    counts = None
    for attr in ("files_processed_by_output",
                 "_last_files_processed_by_output"):
        value = getattr(thread, attr, None)
        if value is not None:
            counts = value
            break
    try:
        items = counts.items()
    except AttributeError:
        return fallback

    last = None
    for path, value in items:
        try:
            if int(value) <= 0:
                continue
            candidate = os.fspath(path)
        except (TypeError, ValueError):
            continue
        if os.path.exists(candidate):
            last = candidate
    return last or fallback


def _finished_output_file(thread, wrangler, *, all_skipped_append=False):
    """Resolve the existing processed output owned by a completed run.

    Skip-before-read can consume an already-complete container without ever
    calling ``initialize_scan``.  In that path ``thread.fname`` still contains
    the provisional raw-derived name (for Eiger, ``*_master.nxs``), while the
    append cursor records the canonical processed scan name it reached.
    """
    if all_skipped_append:
        snapshots = getattr(thread, "_append_skip_frames_by_scan", None)
        try:
            items = list(snapshots.items())
        except AttributeError:
            items = []
        output_path = getattr(thread, "_append_output_path", None)
        for scan_name, frame_ids in reversed(items):
            if not frame_ids:
                continue
            try:
                candidate = (output_path(scan_name) if callable(output_path)
                             else os.path.join(
                                 os.fspath(getattr(thread, "h5_dir", "")),
                                 f"{scan_name}.nxs"))
            except (TypeError, ValueError):
                continue
            if candidate and os.path.exists(candidate):
                return os.fspath(candidate)

    for owner in (thread, wrangler):
        candidate = getattr(owner, "fname", None)
        try:
            if candidate and os.path.exists(candidate):
                return os.fspath(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _runend_callsite() -> str | None:
    if not browse_debug_enabled():
        return None
    try:
        import inspect
        frame = inspect.currentframe()
        caller = frame.f_back.f_back if frame and frame.f_back else None
        if caller is None:
            return None
        return f"{caller.f_code.co_name}:{caller.f_lineno}"
    except Exception:
        return None

# Viewer-mode row stores.  Normal scan display reads exclusively from the
# record/publication stores; these dicts back Image/XYE/NeXus file browsers.
_VIEWER_ROWS_1D_CACHE_MAX = 4096
_VIEWER_ROWS_2D_CACHE_MAX = 40
_LIVE_FLUSH_MIN_MS = 110
_LIVE_TIMER_MIN_MS = 10

# Qt imports
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    QtWidgets: Any = None
    QtCore: Any = None
else:
    from pyqtgraph.Qt import QtWidgets, QtCore

# This module imports
from xdart.modules.live import LiveFrame, LiveScan
from xdart.modules.frame_publication import (
    PublicationStore,
    legacy_to_canonical_1d,
    legacy_to_canonical_2d,
    publication_error_details,
    publication_from_live_frame,
    publication_has_2d_errors,
)
from xrd_tools.core import browse_publication_max_items
from xrd_tools.core.energy import normalize_wavelength_m, wavelength_m_to_energy_eV
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    GIIntent,
    RunIntent,
    ThresholdIntent,
)
from xrd_tools.sources.readiness import (
    capabilities_for_processed,
    describe_source_readiness,
    observe_raw_reachability,
    observe_source_readiness,
    quiet_capability_observation,
)


class _TransientReadinessObservation(RuntimeError):
    """H18-R5: a raw-probe observation that is not definitive — handled like
    a transient metadata failure (debounced retry, never cached)."""


#: Sentinel distinguishing "argument not supplied" from an explicit ``None``.
_UNSET = object()

#: pyFAI integration methods the Controls panels offer (union of the Integrate 1D
#: + 2D ``method`` lists in ``integrator.py``).  §13.4: a value outside this
#: whitelist is a typed staging refusal, never silently accepted.
_CONTROLS_V2_SUPPORTED_METHODS = frozenset({
    "numpy", "cython", "BBox", "splitpixel", "lut", "csr",
    "nosplit_csr", "full_csr", "lut_ocl", "csr_ocl",
})

#: §15.12-A.7: STRICT unit resolution — canonical code -> the EXACT accepted
#: spellings (casefolded), a deliberately enumerated alias set.  The strict
#: staging path accepts ONLY these (plus the display names in ``Units_dict`` and
#: the codes in ``Units_dict_inv``); ``2garbage`` / ``machine`` are refusals.
#: Fuzzy substring/startswith normalization is confined to the NON-strict
#: display/import path.
_CONTROLS_V2_STRICT_UNIT_ALIASES = {
    "q_A^-1": frozenset({
        "q", "q_a^-1", "q (å⁻¹)", "q(å⁻¹)", "å⁻¹", "a^-1", "1/a", "1/å",
        "q_a-1", "qa^-1",
    }),
    "2th_deg": frozenset({
        "2th", "tth", "2theta", "2th_deg", "2θ", "2θ (°)", "2θ(°)",
        "2theta_deg", "2th (°)",
    }),
    "chi_deg": frozenset({
        "chi", "chi_deg", "χ", "χ (°)", "χ(°)", "chi (°)",
    }),
}


class DeferredRunEditsPendingError(RuntimeError):
    """R4B-12 (O-1a-i.2): Run preparation refuses to freeze because a Controls
    edit queued during the previous run could not be applied.  Raised by
    ``_prepare_controls_v2_run_configuration`` and surfaced to the operator by
    ``imageWrangler.start()``; no ``FrozenRunConfiguration`` is produced."""


class ControlsTransactionError(Exception):
    """A PURE Controls staging failure — an edit could not be resolved/coerced.

    Carries the offending path, reason, and value so the caller can refuse to
    freeze with a typed, user-visible message and a structured event.  This is
    RETURNED by :meth:`staticWidget.stage_controls_transaction` (never raised
    from staging), so the stage phase mutates no production carrier before the
    failure is known (§9.10 step 3-stage / §12.3)."""

    def __init__(self, path, reason, value=None):
        self.path = tuple(path) if path is not None else None
        self.reason = str(reason)
        self.value = value
        pretty = "/".join(str(seg) for seg in (self.path or ()))
        super().__init__(f"invalid control edit ({pretty}): {self.reason}")


class GIMotorObservation:
    """A source-qualified snapshot of the GI theta-motor choices (§12.5).

    The ``state`` is one of ``UNKNOWN`` / ``KNOWN_EMPTY`` / ``KNOWN_NONEMPTY``,
    tied to the ``source_token`` (the immutable identity of the source the
    motors were observed FROM).  ``choices_for_freeze()`` maps the three states
    to the ``resolve_gi_motor`` contract — ``None`` (unknown, preserve the
    explicit motor), ``()`` (known-empty, resolve to Manual), or the motor
    tuple.  Choices from a DIFFERENT source token must never seed a freeze — a
    deferred source A→B edit invalidates the observation, so B never inherits
    A's motor.
    """

    UNKNOWN = "UNKNOWN"
    KNOWN_EMPTY = "KNOWN_EMPTY"
    KNOWN_NONEMPTY = "KNOWN_NONEMPTY"

    __slots__ = ("state", "motors", "source_token")

    def __init__(self, state=UNKNOWN, motors=(), source_token=None):
        motors = tuple(str(m) for m in (motors or ()) if str(m) and str(m) != "Manual")
        if state == self.KNOWN_NONEMPTY and not motors:
            state = self.KNOWN_EMPTY
        if state == self.KNOWN_EMPTY and motors:
            state = self.KNOWN_NONEMPTY
        self.state = state
        self.motors = motors
        self.source_token = source_token

    def matches(self, source_token) -> bool:
        return self.source_token == source_token

    def choices_for_freeze(self):
        if self.state == self.KNOWN_NONEMPTY:
            return self.motors
        if self.state == self.KNOWN_EMPTY:
            return ()
        return None


class JournalEntry:
    """Immutable, deep-copied revisioned edit (§13.9).

    The value is deep-copied on construction so the journal can never be mutated
    through a caller's aliased list/dict, and every outward read hands back a
    FRESH deep copy so projection cannot alias the stored value either.  Supports
    the historical ``entry["value"]`` / ``["revision"]`` / ``["origin"]`` mapping
    access (and ``.get``) so existing readers are unchanged."""

    __slots__ = ("_value", "revision", "origin")

    def __init__(self, value, revision, origin):
        self._value = copy.deepcopy(value)
        self.revision = int(revision)
        self.origin = str(origin)

    @property
    def value(self):
        return copy.deepcopy(self._value)

    def __getitem__(self, key):
        if key == "value":
            return self.value
        if key == "revision":
            return self.revision
        if key == "origin":
            return self.origin
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class ControlsStageCandidate:
    """Mutable working state for a PURE stage (§12.2).

    Every reducer reads and writes ONLY this candidate — never ``self.scan``, a
    legacy parameter, a widget, the wrangler, or a thread — so multiple edits in
    one transaction compose in revision order and staging installs nothing on the
    production widget.  ``gi_enabled`` reads the candidate intent, so a GI-enable
    followed by a GI-axis edit composes against the staged GI state."""

    __slots__ = (
        "intent",
        "threshold_state",
        "gi_selection_explicit",
        "source_energy_preference",
        "gi_motor_observation",
        "legacy_projection",
        "source_touched",
        "source_selection_touched",
        "poni_touched",
        "poni_object",
        "poni_values",
        "candidate_source_spec",
        "source_fingerprint",
        "committed_legacy",
    )

    def __init__(
        self,
        intent,
        threshold_state,
        gi_selection_explicit,
        source_energy_preference,
        gi_motor_observation,
        committed_legacy=None,
    ):
        #: §13.11 candidate 3 / §14.11.D: the COMMITTED legacy source-selection
        #: param values captured ONCE at stage entry.  Reducers compare and derive
        #: the candidate against THIS snapshot, never a live Qt Parameter — so the
        #: stage depends on a captured value, not a mutable carrier.
        self.committed_legacy = dict(committed_legacy or {})
        self.intent = intent
        self.threshold_state = threshold_state
        self.gi_selection_explicit = bool(gi_selection_explicit)
        self.source_energy_preference = str(source_energy_preference or "poni")
        self.gi_motor_observation = gi_motor_observation
        self.legacy_projection = {}
        self.source_touched = False
        #: True only when a SOURCE-SELECTION path (not mask/PONI/series-average)
        #: was reduced — the sole trigger for source reconciliation (§12.7).
        self.source_selection_touched = False
        #: Compound PONI candidate (§12.6/§13.8): a poni_file edit is parsed
        #: DURING staging into poni_object/poni_values so path, object, values,
        #: wrangler/thread carriers, and provenance publish or roll back as one.
        self.poni_touched = False
        self.poni_object = None
        self.poni_values = None
        #: Candidate SourceSpec + its fingerprint/epoch derived at reduce time
        #: from a source-SELECTION edit (§13.11 candidate 4-5), so subsequent GI
        #: reducers see the NEW source (motor knowledge UNKNOWN), never the old.
        self.candidate_source_spec = None
        self.source_fingerprint = None

    @property
    def gi_enabled(self) -> bool:
        return bool(self.intent.gi.enabled)

    @property
    def bai(self):
        if not isinstance(self.intent.bai_1d_args, dict):
            self.intent.bai_1d_args = {}
        if not isinstance(self.intent.bai_2d_args, dict):
            self.intent.bai_2d_args = {}
        return self.intent.bai_1d_args, self.intent.bai_2d_args


class StagedControlsTransaction:
    """A fully validated, PURE candidate for the next run (§12.2/§12.3).

    Produced by a pure reducer: native/int/GI/threshold edits are reduced onto
    ``staged_intent`` (a clone of the Controls-owned intent) and legacy-backed
    edits are coerced into ``legacy_projection`` WITHOUT touching a Parameter or
    any ``self`` state.  The complete candidate is validated before this object
    exists.  No production carrier (Qt param, display scan, source index/cache,
    session, wrangler, profile, threshold/energy self-state) is written while
    staging."""

    __slots__ = (
        "staged_intent",
        "legacy_projection",
        "threshold_state",
        "gi_selection_explicit",
        "source_touched",
        "source_selection_touched",
        "source_energy_preference",
        "gi_motor_observation",
        "poni_touched",
        "poni_object",
        "poni_values",
        "candidate_source_spec",
        "source_fingerprint",
    )

    def __init__(
        self,
        staged_intent,
        legacy_projection,
        threshold_state,
        gi_selection_explicit,
        source_touched,
        source_energy_preference,
        gi_motor_observation,
        source_selection_touched=False,
        poni_touched=False,
        poni_object=None,
        poni_values=None,
        candidate_source_spec=None,
        source_fingerprint=None,
    ):
        self.staged_intent = staged_intent
        self.legacy_projection = dict(legacy_projection)
        self.threshold_state = threshold_state
        self.gi_selection_explicit = bool(gi_selection_explicit)
        self.source_touched = bool(source_touched)
        self.source_selection_touched = bool(source_selection_touched)
        self.source_energy_preference = source_energy_preference
        self.gi_motor_observation = gi_motor_observation
        #: Compound PONI candidate (§12.6/§13.8) — installed/rolled back as one.
        self.poni_touched = bool(poni_touched)
        self.poni_object = poni_object
        self.poni_values = poni_values
        #: Candidate SourceSpec + fingerprint/epoch (§13.11 candidate 4-5).
        self.candidate_source_spec = candidate_source_spec
        self.source_fingerprint = source_fingerprint


class ControlsCommitResult:
    """Outcome of one transaction ATTEMPT (§14.11.A/C).

    Carries the ``phase`` that failed (``preflight`` / ``legacy_apply`` /
    ``install`` / ``source_reconcile`` / ``harvest`` / ``stage`` / ``recovery``),
    the offending ``failed_path``, a ``reason``, and — when rollback could not
    RESTORE one or more carriers — the FULL list ``recovery_failed_paths`` (a
    distinct recovery error: Run stays refused, the journal is retained, and both
    the original and every un-restorable carrier are surfaced).  This is the SOLE
    per-attempt diagnostic authority; there is no ambient recovery state
    (§14.11.C)."""

    __slots__ = ("ok", "phase", "failed_path", "reason", "recovery_failed_paths")

    def __init__(self, ok, failed_path=None, reason="",
                 recovery_failed_paths=(), phase=""):
        self.ok = bool(ok)
        self.phase = str(phase)
        self.failed_path = (
            tuple(failed_path) if failed_path is not None else None)
        self.reason = str(reason)
        self.recovery_failed_paths = tuple(
            tuple(p) for p in (recovery_failed_paths or ()) if p is not None)

    @property
    def recovery_failed_path(self):
        """The FIRST un-restorable carrier, for concise UI wording (§14.11.A.5)."""
        return (self.recovery_failed_paths[0]
                if self.recovery_failed_paths else None)
from .ui.staticUI import Ui_Form
from .h5viewer import H5Viewer, _qt_enum_value
from .display_frame_widget import displayFrameWidget
from .display_logic import AccumulatorLifecycle, LifecycleCause
from .display_overlay_utils import (
    frame_index_from_row_id,
    overlay_grid_key_for_widget,
    overlay_grid_keys_match,
    overlay_grid_spec_for_history,
    overlay_grid_spec_for_view,
    overlay_grid_spec_summary,
    overlay_grid_specs_match,
    row_id_belongs_to_widget_scan,
    scan_identity_key,
)
from .integrator import (
    DEFAULT_POLARIZATION_FACTOR,
    GI_LABELS_1D,
    GI_LABELS_2D,
    GI_MODES_1D,
    GI_MODES_2D,
    Units,
    Units_dict,
    Units_dict_inv,
    integratorTree,
)
from .scan_threads import stitchThread
from .metadata import metadataWidget
from .wranglers import imageWrangler, nexusWrangler, wranglerWidget
from .wranglers.wrangler_widget import GIMotorHydration
from .controls_logic import (
    AnalysisTool,
    BOUND_CONTROL_PATHS,
    ControlAction,
    INTEGRATOR_BACKED_CONTROL_PATHS,
    INTEGRATOR_BACKED_CONTROL_SPECS,
    INTEGRATION_CONTROL_PATHS,
    NATIVE_CONTROL_PATHS,
    ControlState,
    GeomState,
    MeasMode,
    ResultCaps,
    RunTarget,
    SourceCaps,
    Tool,
    build_control_panel_state,
    build_native_int_reduction_plan_from_scan,
    coerce_control_edit_value,
    processing_config_from_scan,
    run_target_readiness_note,
    tool_from_mode_text,
)
from xdart.utils.throttle import Coalescer
from xdart.utils._utils import FixSizeOrderedDict, get_fname_dir, get_img_data
from xdart.modules.reduction import ThresholdSaturationConfig

QWidget = QtWidgets.QWidget
QSizePolicy = QtWidgets.QSizePolicy
QFileDialog = QtWidgets.QFileDialog
QMessageBox = QtWidgets.QMessageBox
QDialog = QtWidgets.QDialog
QInputDialog = QtWidgets.QInputDialog
QCombo = QtWidgets.QComboBox

wranglers = {
    'Image Files': imageWrangler,
    'NeXus': nexusWrangler,
}


class _XyeOverlayInputFilter(QtCore.QObject):
    """Modifier-free, plotMethod-aware multi-select for the XYE file list (E1/E2).

    Active only in XYE viewer (the list is ``ExtendedSelection``).  Mirrors
    ``h5viewer._AccumulatingClickFilter`` semantics on the file list:

    * **Accumulating plot methods** (Overlay / Waterfall / Sum / Average) build a
      comparison set — a plain left-click toggles a file in/out, and Up/Down
      arrows EXTEND the selection (add the newly-current file without clearing),
      so arrow-browsing accumulates just like clicking.
    * **Single** mode browses one file — a plain click replaces (Qt default) and
      arrows move one row (Qt default).

    Directories / ``..`` keep default click-to-navigate; the filter is inert
    outside XYE mode.  (Vivek's model: the *selection* accumulates and the plot =
    the selected set per plotMethod, so Sum/Average accumulate on arrow too —
    which diverges from Int 1D's plan_overlay where Sum/Average are REPLACE.)
    """

    _ACCUMULATING = ('Overlay', 'Waterfall', 'Sum', 'Average')

    def __init__(self, list_widget, is_active, get_method):
        super().__init__(list_widget)
        self._list = list_widget
        self._is_active = is_active
        self._get_method = get_method

    def _accumulating(self):
        get_method = getattr(self, '_get_method', None)
        if not callable(get_method):
            return False
        try:
            return get_method() in self._ACCUMULATING
        except Exception:
            return False

    @staticmethod
    def _is_data_item(item):
        if item is None:
            return False
        text = item.text()
        return text != '..' and not text.endswith('/')

    def eventFilter(self, obj, event):
        is_active = getattr(self, '_is_active', None)
        if not callable(is_active):
            return False
        try:
            active = is_active()
        except Exception:
            return False
        if not active:
            return False
        etype = event.type()
        if etype == QtCore.QEvent.MouseButtonPress:
            return self._on_click(event)
        if etype == QtCore.QEvent.KeyPress:
            return self._on_key(event)
        return False

    @staticmethod
    def _meaningful_modifiers(event):
        """Return the shift/ctrl/meta bits held, coerced to int.

        Mirrors ``_AccumulatingClickFilter``: raw ``modifiers() != NoModifier``
        comparisons are unreliable under PySide6 (a plain click can carry a
        stray flag, notably on macOS), so coerce to int and mask to the only
        modifiers we care about.  Returns ``(has_shift, has_toggle_mod)``."""
        try:
            mods = _qt_enum_value(event.modifiers())
        except Exception:
            return False, False
        shift_bit = _qt_enum_value(QtCore.Qt.ShiftModifier)
        ctrl_bit = _qt_enum_value(QtCore.Qt.ControlModifier)
        meta_bit = _qt_enum_value(QtCore.Qt.MetaModifier)
        return bool(mods & shift_bit), bool(mods & (ctrl_bit | meta_bit))

    def _on_click(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return False
        list_widget = getattr(self, '_list', None)
        if list_widget is None:
            return False
        has_shift, has_toggle_mod = self._meaningful_modifiers(event)
        if has_shift:
            return False                  # let Qt handle shift range-select
        try:
            pos = event.position().toPoint()
        except AttributeError:            # Qt5 fallback
            pos = event.pos()
        item = list_widget.itemAt(pos)
        if not self._is_data_item(item):
            return False
        if not (has_toggle_mod or self._accumulating()):
            return False                  # Single plain click: Qt replace
        # Accumulating (or explicit ctrl/cmd-toggle): toggle this file in/out of
        # the overlay via the selection model (robust in ExtendedSelection).
        sm = list_widget.selectionModel()
        idx = list_widget.indexFromItem(item)
        sm.select(idx, QtCore.QItemSelectionModel.Toggle)
        sm.setCurrentIndex(idx, QtCore.QItemSelectionModel.NoUpdate)
        return True

    def _on_key(self, event):
        list_widget = getattr(self, '_list', None)
        if list_widget is None:
            return False
        has_shift, has_toggle_mod = self._meaningful_modifiers(event)
        if (not self._accumulating()
                or has_shift or has_toggle_mod
                or event.key() not in (QtCore.Qt.Key_Up, QtCore.Qt.Key_Down)):
            return False                  # Single / modified: Qt default browse
        step = -1 if event.key() == QtCore.Qt.Key_Up else 1
        row = list_widget.currentRow() + step
        while 0 <= row < list_widget.count():
            item = list_widget.item(row)
            if self._is_data_item(item):
                # Extend: add the newly-current file without clearing the rest,
                # so arrow-browsing builds the comparison set.
                item.setSelected(True)
                list_widget.setCurrentItem(
                    item, QtCore.QItemSelectionModel.NoUpdate)
                return True
            row += step
        return True                       # at an end / only dirs: consume


def scanlocked(func):
    """Decorator that acquires scan_lock before calling the wrapped method.

    If self.scan is not a LiveScan (e.g. during initialisation),
    the function is called without the lock rather than silently returning None.
    """
    def wrapper(self, *args, **kwargs):
        if isinstance(self.scan, LiveScan):
            with self.scan.scan_lock:
                return func(self, *args, **kwargs)
        return func(self, *args, **kwargs)

    return wrapper


def _scan_key_from_source(src):
    """Derive the wrangler's scan name from a frame's ``source_file``.

    Delegates to the ONE canonical ``scan_name_from_source`` (Codex F2) so the GUI
    attributes each frame to EXACTLY the scan the worker wrote — including keeping
    the FULL stem for a container ``.nxs``/``.h5``/``.hdf5`` (the numeric suffix is
    part of the identity; dropping it garbled the plot title / legend and merged
    distinct numeric ``.nxs`` scans).  Returns ``None`` for an empty/missing source
    (treated as "no scan change").  Pure string parse, no I/O.
    """
    if not src:
        return None
    from .wranglers.image_wrangler_thread import scan_name_from_source
    return scan_name_from_source(src) or None


def _drop_output_axis_ranges(args):
    """S-5: drop the output-axis RANGE keys from a native-int arg dict.

    The GI 1D modes q_oop/exit_angle/chi_gi share ``azimuth_range`` in DIFFERENT
    units (q_ip uses ``radial_range``), so a frozen/hydrated range from a prior
    mode would silently clip the new mode (e.g. χGI to a ~4° wedge) and be
    written.  Called only on an ACTUAL mode change, so a hydration-to-same-mode
    keeps its restored range."""
    args.pop("azimuth_range", None)
    args.pop("radial_range", None)


class staticWidget(QWidget):
    # DIR-2 lazy convergence: queued landing for container frame counts
    _sigV2CountLanded = QtCore.Signal(str, int)
    """Tab for integrating data collected by a scanning area detector.
    As of current version, only handles a single angle (2-theta).
    Displays raw images, stitched Q Chi arrays, and integrated I Q
    arrays. Also displays metadata and exposes parameters for
    controlling integration.

    children:
        displayframe: widget which handles displaying images and
            plotting data.
        h5viewer: Has a file explorer panel for loading scans, and
            a panel which shows images that are associated with the
            loaded scan. Has other file saving and loading functions
            as well as configuration saving and loading functions.
        integrator_thread: Not visible to user, but a sub-thread which
            handles integration to free resources for the gui
        integratorTree: Widget for setting the basic integration
            parameters. Also has buttons for starting integration.
        metawidget: Table wiget which displays metadata either for
            entire scan or individual image.

    attributes:
        frame: LiveFrame, currently loaded frame object
        frame_ids: List of LiveFrame indices currently loaded
        frames: Dictionary of currently loaded LiveFrames
        viewer_rows_1d: Dictionary object holding all 1D data in memory
        viewer_rows_2d: Dictionary object holding all 2D data in memory
        command_queue: Queue, used to send commands to wrangler
        dirname: str, absolute path of current directory for scan
        file_lock: mp.Condition, process safe lock
        fname: str, current data file name
        scan: LiveScan, current scan data
        timer: QTimer, currently unused but can be used for periodic
            functions.
        ui: Ui_Form, layout from qtdesigner

    methods:
        bai_1d: Sends signal to thread to start integrating 1d
        bai_2d:  Sends signal to thread to start integrating 2d
        clock: Unimplemented, used for periodic updates
        close: Handles cleanup prior to closing
        enable_integration: Sets enabled status of widgets related to
            integration
        first_frame, latest_frame, next_frame: Handle moving between
            different frames in the overall scan
        load_and_set: Combination of load and set methods. Also governs
            file explorer behavior in h5viewer.
        load_scan:
    """

    #: Worker → GUI handoff for the async directory frame count (bl17-2
    #: freeze, 2026-07-12): emitted from the sweep thread; the cross-thread
    #: connection queues delivery onto the GUI thread.

    def __init__(self, parent=None):
        super().__init__(parent)
        self._v2_source_count_is_files = False
        # DIR-2 lazy convergence: {path: ((size, mtime_ns), nframes)} —
        # filled by the run as containers open (sigContainerCount) and by
        # the click-to-count sweep; a stale stamp invalidates the entry.
        self._v2_container_count_memo = {}
        # Optimization-only subset whose count and exact stamp were captured
        # together when a finalized, self-contained container was retired.
        # Open-time, click-sweep, fabio, and external-link counts deliberately
        # never enter this map.
        self._v2_container_final_count_memo = {}
        self._v2_count_sweep_active = False
        self._controls_v2_source_widget = None
        self._controls_v2_directory_observation = None
        self._sigV2CountLanded.connect(self._on_container_count_landed)
        self._init_data_objects()
        self._init_ui()
        self._init_child_widgets()
        self._connect_signals()
        self._init_wranglers()
        self._strip_combo_checkmarks()
        self._init_defaults_and_timer()
        self.show()
        self.ui.wranglerFrame.activateWindow()

    def _strip_combo_checkmarks(self):
        """Polish every dropdown in the tab in one sweep:

        * Replace the item delegate with a plain QStyledItemDelegate so the
          popup shows the selection by highlight only — the default
          QComboBox delegate draws a current-item checkmark that QSS
          ``::indicator`` can't remove, and it clipped the longer names.
        * Widen the popup to its longest entry (the combo box itself stays
          compact in the toolbar) so options like "Image Viewer" / "Int 1D
          (XYE)" aren't truncated.

        Covers the mode combo (in wranglerStack), the 1D plot Single/Q-θ
        combos, the top-bar Scale/colormap, the 2D-unit combo, and the
        integrator unit combos — all are descendants here.
        """
        for combo in self.findChildren(QtWidgets.QComboBox):
            combo.setItemDelegate(QtWidgets.QStyledItemDelegate(combo))
            view = combo.view()
            try:
                view.setTextElideMode(QtCore.Qt.ElideNone)
                fm = combo.fontMetrics()
                widest = max(
                    (fm.horizontalAdvance(combo.itemText(i))
                     for i in range(combo.count())),
                    default=0,
                )
                if widest:
                    # + room for item padding and the popup scrollbar.
                    view.setMinimumWidth(widest + 44)
            except Exception:
                logger.debug("combo popup sizing skipped", exc_info=True)

    # ── Initialization helpers ─────────────────────────────────────

    def _init_data_objects(self):
        """Initialize data containers, file lock, and directory paths."""
        self.file_lock = threading.Condition()
        # Reentrant lock guarding concurrent access to viewer_rows_1d / viewer_rows_2d from
        # the GUI thread, integratorThread, and fileHandlerThread. Shared with
        # all child widgets and worker threads. Always the OUTER lock when
        # paired with scan.scan_lock (data_lock → scan_lock).
        self.data_lock = threading.RLock()
        # Scratch directory for working .nxs files (under the user's home).
        self.local_path = get_fname_dir()
        self.dirname = self.local_path

        self.fname = os.path.join(self.dirname, 'default.nxs')
        # H18 (H5 finding 3): remember the PRISTINE scratch placeholder so the
        # readiness gate can tell "nothing loaded yet" from a genuinely
        # loaded/processed scan.  LiveScan needs a default data_file, but that
        # default must not make an empty widget report a phantom loaded scan
        # (loaded_scan_available / ResultCaps.has_raw / run_target=LOADED_SCAN
        # were all True on a fresh widget).
        self._controls_v2_scratch_data_file = self.fname
        # J2: share ``file_lock`` with the scan so direct
        # LiveFrameSeries lazy loads use the same lock as the
        # wrangler's save paths.
        self.scan = LiveScan('null_main',
                               data_file=self.fname,
                               static=True,
                               file_lock=self.file_lock)
        self.frame = LiveFrame(static=True, gi=self.scan.gi)
        self.frame_ids = []
        self.frames = OrderedDict()
        # Browse 1D residency: hold up to a ~1 GiB byte budget (item 1) so a large
        # Show-All keeps every cheap 1D-light publication resident (no disk re-read
        # on later per-frame browsing).  Heavy raw/2D stays capped by
        # max_heavy_items/heavy_window (default), so MEM-1 is preserved.
        self.publication_store = PublicationStore(
            max_items=browse_publication_max_items())
        self._frame_record_store = None
        # MEM1-15: tier-0 eviction honors persist-before-evict — an unsaved
        # publication pins memory instead of being dropped (the store is the
        # cake's only render source; dropping an unsaved one blanks the frame
        # on scroll-back).  Resolves the ACTIVE record store at eviction time
        # (it is per-run and lazily created).
        self.publication_store.set_evictable_probe(
            self._publication_label_evictable)
        self._overlay_flush_last_t = 0.0
        # XYE/NeXus need a 1D row table; Image Viewer needs a bounded 2D raw
        # table.  Neither participates in scan display readiness.
        self.viewer_rows_1d = FixSizeOrderedDict(max=_VIEWER_ROWS_1D_CACHE_MAX)
        self.viewer_rows_2d = FixSizeOrderedDict(max=_VIEWER_ROWS_2D_CACHE_MAX)

    @staticmethod
    def _timer_ms_from_env(name, default, *, minimum=_LIVE_TIMER_MIN_MS):
        raw = os.environ.get(name)
        try:
            value = int(raw) if raw not in (None, "") else int(default)
        except (TypeError, ValueError):
            return int(default)
        minimum = int(minimum)
        if value < minimum:
            logger.warning(
                "%s=%dms is below the supported minimum %dms; clamping to %dms",
                name, value, minimum, minimum,
            )
            return minimum
        return value

    def _active_frame_record_modes(
            self, mode_1d: str | None = None,
            mode_2d: str | None = None) -> tuple[str | None, str | None]:
        """Return the active display mode keys for the current scan."""
        scan = getattr(self, "scan", None)
        if not bool(getattr(scan, "gi", False)):
            return mode_1d, mode_2d
        args_1d = getattr(scan, "bai_1d_args", {}) or {}
        args_2d = getattr(scan, "bai_2d_args", {}) or {}
        active_1d = mode_1d or args_1d.get("gi_mode_1d", "q_total")
        active_2d = mode_2d or args_2d.get("gi_mode_2d", "qip_qoop")
        if active_1d is not None:
            active_1d = legacy_to_canonical_1d(str(active_1d))
        if active_2d is not None:
            active_2d = legacy_to_canonical_2d(str(active_2d))
        return active_1d, active_2d

    def _active_frame_record_store(self):
        """The current per-scan session store, if the live path has one."""
        thread = getattr(getattr(self, "wrangler", None), "thread", None)
        store = getattr(thread, "_streaming_record_store", None)
        if store is not None:
            self._frame_record_store = store
            return store
        return getattr(self, "_frame_record_store", None)

    def _publication_label_evictable(self, label):
        """MEM1-15 persist gate for the publication store's tier-0 eviction.

        True = safe to drop the label's publication.  Only a label the active
        record store TRACKS as not-yet-persisted is pinned ("owed" — dropping
        it would blank the frame on scroll-back until the save lands).  A
        label with no record at all is NOT an owed live frame, so it stays
        evictable — otherwise it could pin forever.  With no record store in
        play (loaded-scan/viewer sessions) everything is evictable — the
        legacy behavior.  Called under the publication store's lock: keep it
        cheap and never touch the publication store from here."""
        store = self._active_frame_record_store()
        if store is None:
            return True
        try:
            if store.get(label) is None:
                return True
            return bool(store.is_persisted(label))
        except Exception:
            return True

    def _clear_frame_record_store(self, *_args):
        self._frame_record_store = None

    def _request_frame_record_hydration(self, label):
        display = getattr(self, "displayframe", None)
        request = getattr(display, "_request_frame_hydration", None)
        if request is None:
            return
        try:
            request(label)
        except Exception:
            logger.debug("record-store hydration request failed for %s", label,
                         exc_info=True)

    @staticmethod
    def _coerce_frame_label(idx):
        try:
            return int(idx)
        except (TypeError, ValueError):
            return idx

    def _publication_frame_view(
            self, idx, mode_1d: str | None, mode_2d: str | None,
            *, allow_blocking_read: bool = False):
        store = getattr(self, "publication_store", None)
        if store is None:
            return None
        getter = getattr(store, "get_or_hydrate", None) if allow_blocking_read else None
        try:
            publication = (
                getter(idx) if getter is not None else store.get(idx)
            )
        except Exception:
            logger.debug("publication lookup failed for %s", idx, exc_info=True)
            return None
        if publication is None:
            return None
        record = getattr(publication, "record", None)
        if record is not None:
            try:
                return record.project(mode_1d=mode_1d, mode_2d=mode_2d)
            except ValueError:
                pass
        return getattr(publication, "view", None)

    def store_first_frame_view(
            self, idx, *, mode_1d: str | None = None,
            mode_2d: str | None = None,
            allow_blocking_read: bool = False):
        """Return the selected scan frame view from the authoritative stores."""
        key = self._coerce_frame_label(idx)
        mode_1d, mode_2d = self._active_frame_record_modes(mode_1d, mode_2d)

        store = self._active_frame_record_store()
        if store is not None:
            try:
                record = (
                    store.get_or_hydrate(key)
                    if allow_blocking_read and hasattr(store, "get_or_hydrate")
                    else store.get(key)
                )
            except Exception:
                logger.debug("record_store lookup failed for %s", key, exc_info=True)
                record = None
            if record is not None:
                try:
                    heavy = store.has_heavy_payload(key)
                except Exception:
                    heavy = True
                if not heavy:
                    if not allow_blocking_read:
                        self._request_frame_record_hydration(key)
                else:
                    try:
                        return record.project(mode_1d=mode_1d, mode_2d=mode_2d)
                    except ValueError:
                        logger.debug("record_store projection missed for %s", key,
                                     exc_info=True)

        view = self._publication_frame_view(
            key, mode_1d, mode_2d, allow_blocking_read=allow_blocking_read)
        if view is not None:
            return view
        return None

    def _init_ui(self):
        """Set up the main UI form and detector dialog."""
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.detector_dialog = QDialog()
        self.detector_widget = QCombo()
        self.detector = None

    def _init_child_widgets(self):
        """Create H5Viewer, DisplayFrame, IntegratorTree, and Metadata widgets."""
        # H5Viewer
        self.h5viewer = H5Viewer(self.file_lock, self.local_path, self.dirname,
                                 self.scan, self.frame, self.frame_ids, self.frames,
                                 self.viewer_rows_1d, self.viewer_rows_2d,
                                 self.ui.hdf5Frame, data_lock=self.data_lock,
                                 publication_store=self.publication_store)
        self.ui.hdf5Frame.setLayout(self.h5viewer.layout)
        self.h5viewer.update_scans()

        # DisplayFrame
        self.displayframe = displayFrameWidget(self.scan, self.frame,
                                               self.frame_ids, self.frames,
                                               self.viewer_rows_1d, self.viewer_rows_2d,
                                               parent=self.ui.middleFrame,
                                               data_lock=self.data_lock,
                                               publication_store=self.publication_store)
        self.displayframe.store_first_frame_view = self.store_first_frame_view
        self.displayframe.frame_record_store = self._active_frame_record_store
        self.displayframe._resolve_overlay_grid_mismatch = (
            self._resolve_overlay_grid_mismatch)
        self.displayframe._cancel_overlay_grid_selection = (
            self._cancel_overlay_grid_selection)
        # The MEM-1[14] memo keys on the active GI projection modes so a
        # sub-mode switch invalidates memoized publications; the resolver
        # lives here, the memo on the displayframe.
        self.displayframe._active_frame_record_modes = \
            self._active_frame_record_modes
        # Back-ref so h5viewer.data_reset can re-arm display-side caches.
        self.h5viewer.displayframe = self.displayframe
        self.ui.middleFrame.setLayout(self.displayframe.ui.layout)

        # IntegratorTree
        self.integratorTree = integratorTree(
            self.scan, self.frame, self.file_lock,
            self.frames, self.frame_ids,
            data_lock=self.data_lock,
            publication_store=self.publication_store)
        # Stitch worker (Stitch 1D / Stitch 2D modes): a one-shot off-thread
        # reduction of the loaded scan, routed through the SAME run-state owner
        # (_enter/_exit_run_state) as a wrangler run or a reintegrate.
        self.stitch_thread = stitchThread(self.scan, parent=self)
        self.stitch_thread.started.connect(self._enter_run_state)
        self.stitch_thread.finished.connect(self.stitch_thread_finished)
        self.stitch_thread.errorSig.connect(self._on_stitch_error)
        # Default panel proportions: middle (image/plot) panels ~10% wider
        # than Qt's hint-based split (Vivek).  Applied via singleShot AFTER
        # the window has real geometry -- setSizes at __init__ ran before the
        # main window's resize() and got redistributed away.
        def _default_split():
            try:
                total = sum(self.ui.mainSplitter.sizes()) or 1000
                # Controls (right) and data-browser (left) columns start at the
                # SAME width; the middle display panels take the rest (Vivek).
                # The side columns are 10% narrower than before (0.252 -> 0.227)
                # so the central display isn't squished; the freed space goes to
                # the middle.  Min/max widths are untouched (set elsewhere) --
                # this only moves the default/initial split.  User-resizable via
                # the splitter (re-asserted only during the first-3s launch storm).
                self.ui.mainSplitter.setSizes(
                    [int(total * f) for f in (0.227, 0.546, 0.227)])
                self.ui.mainSplitter.setStretchFactor(1, 1)
                # Left column: the Tools card is now a compact 3-button panel, so
                # give it only a small share (~18%) and let the data browser take
                # the rest.  Stretch so a window resize grows the browser, not
                # Tools.
                ltotal = sum(self.ui.leftSplitter.sizes()) or 600
                self.ui.leftSplitter.setSizes(
                    [int(ltotal * 0.82), int(ltotal * 0.18)])
                self.ui.leftSplitter.setStretchFactor(0, 1)
                self.ui.leftSplitter.setStretchFactor(1, 0)
            except Exception:
                logger.debug("mainSplitter default sizing failed",
                             exc_info=True)
        # Re-assert through every resize for the first 3s after launch, then
        # never touch it again.  Timers lost to late window-manager resizes,
        # and gating on splitterMoved was unreliable (Qt can emit it from its
        # own redistribution during a native resize, which read as a user
        # drag and disabled the hook).  Time-gating is dumb but bulletproof:
        # no user drags the splitter within 3s of launch.
        import time as _time
        self._split_until = _time.monotonic() + 3.0
        self._apply_default_split = _default_split
        # Receiver-context overload: the Qt-internal single-shot timers die
        # with this widget.  The parentless form outlives close()+deleteLater()
        # (the dispatcher owns the timer), so every unfired timer pinned an
        # ENTIRE staticWidget graph alive into interpreter teardown — at
        # offscreen-suite scale (74 constructions) that graph made the linux
        # exit-SIGSEGV deterministic.
        QtCore.QTimer.singleShot(0, self, _default_split)
        QtCore.QTimer.singleShot(1000, self, _default_split)
        QtCore.QTimer.singleShot(2500, self, _default_split)
        self.ui.integratorFrame.setLayout(self.integratorTree.ui.verticalLayout)
        if len(self.scan.frames.index) > 0:
            self.integratorTree.update()
        self.integratorTree.ui.raw_to_tif.hide()
        # TOOLS section: lift the integrator's bottom row (frame_3 = Calibrate /
        # Make Mask; raw_to_tif hidden) into the top tools bar.  Reparent the
        # WHOLE frame_3 as one self-contained widget (NOT its individual buttons
        # — plucking buttons out of frame_3's layout leaves a dangling layout
        # item that double-frees on teardown / segfaults).  The buttons keep
        # their clicked wiring + the _apply_integration_control_state enable refs.
        try:
            self.ui.toolsLayout.addWidget(self.integratorTree.ui.frame_3)
        except Exception:
            logger.debug("could not move Calibrate/Make Mask to the tools bar",
                         exc_info=True)
        # CONTROLS section: the single shared run-controls widget, installed into
        # the bottom controlsFrame.  set_wrangler attaches it to the active
        # wrangler (routing its signals) — it is never reparented on swap.  Its
        # mode-change drives the staticWidget-level reaction (viewer reset +
        # display clear + integration-control state) here, ONCE.
        from .ui.static_controls import StaticControls
        self.controls = StaticControls()
        self.ui.controlsLayout.setContentsMargins(0, 0, 0, 0)
        self.ui.controlsLayout.addWidget(self.controls)
        # Hug the controls' own (snug, uniform-padded) content height and fix it,
        # so the bottom controls bar can't be resized by the splitter and has no
        # excess black space above/below the rows.  RECOMPUTED after profile /
        # run-row show-hide (set_wrangler, mode change) so it can't go stale
        # (slack in viewer modes where the run row is hidden, or clip if content
        # grows) -- see _fit_controls_height.
        self._fit_controls_height()
        self.controls.modeCombo.currentTextChanged.connect(
            self._on_processing_mode_changed)
        # The readiness bar's "Live" chip follows the toggle immediately.
        self.controls.liveToggled.connect(
            lambda *_: self._refresh_controls_v2_profile(immediate=True))
        # Click-to-count on the 'N files' chip (DIR-2 convergence).
        self.controls.readinessSummaryClicked.connect(
            self._on_readiness_summary_clicked)
        # Single owner of the shared Stop button: dispatch to whichever run is
        # active — a reintegrate (integrator thread) takes priority, else the
        # wrangler.  The wranglers no longer connect Stop directly, so a Stop
        # press during a reintegrate can't also trip the idle wrangler's stop()
        # side-effects (unchecking Live, command='stop', button morph).
        self.controls.stopButton.clicked.connect(self._on_stop_clicked)
        self.controls_v2 = None
        self._controls_v2_last_signature = None
        self._controls_v2_batch_refresh_deferred = False
        self._controls_v2_refresh_timer = Coalescer(
            250, mode="throttle", parent=self)
        self._controls_v2_refresh_timer.triggered.connect(
            self._refresh_controls_v2_profile_now)
        self._init_controls_v2_preview()
        self._configure_controls_v2_native_run_plan()
        # Reintegrate reuses the shared Batch toggle: Batch off -> live (per-frame,
        # the default); Batch on -> fast multicore.  The integrator reads it
        # through this provider at click time (no direct controls ref needed).
        self.integratorTree._reintegrate_batch_provider = (
            lambda: self.controls.batchButton.isChecked())
        # Restore the integration panel (units/pts/ranges/Auto flags/GI modes
        # + Advanced params) from the previous session; saved in close().
        try:
            from xdart.utils.session import load_session
            _integ = (load_session() or {}).get('integrator')
            if _integ and not self._controls_v2_enabled():
                self.integratorTree.restore_session_state(_integ)
        except Exception:
            logger.debug("integrator session restore failed", exc_info=True)
        self._restore_controls_v2_int_session_state()

        # Metadata
        self.metawidget = metadataWidget(self.scan, self.frame,
                                         self.frame_ids, self.frames,
                                         viewer_rows_1d=self.viewer_rows_1d,
                                         publication_store=self.publication_store,
                                         data_lock=self.data_lock,
                                         # X1 Slice 2: the display widget supplies
                                         # the shared scan-qualified projection for
                                         # the current frame's metadata.
                                         projection_source=self.displayframe)
        # Stage 4 (Direction A): the metadata table is no longer inline in the
        # bottom-left.  It opens on demand via the "Metadata" button, which
        # reparents this same metawidget into a popup dialog (see
        # _open_metadata_dialog).  The widget keeps every ctor reference
        # (scan/frame/frame_ids/frames/publication_store/data_lock) — they are
        # shared mutable objects, so it still refreshes from frame selection and
        # the publication store exactly as before.  The vacated metaFrame now
        # hosts a Tools placeholder for planned modules.
        self._metadata_dialog = None
        self._peak_fit_dialog = None
        self._phase_fit_dialog = None
        self._scan_plot_dialog = None
        # The fit dialog whose batch run is currently in flight (Peak or Phase).
        self._batch_dialog = None
        # Live analysis preview (analyzer framework Step 3): a latest-wins
        # background worker re-fits the newest frame while the dialog's "Live"
        # toggle is on.  Lazily created on first live request; generation gates
        # stale results.
        self._live_analysis_worker = None
        self._live_fit_gen = 0
        # Batch analysis: one worker fits every frame and streams params into the
        # dialog's embedded vs-frame trend (row 3).
        self._batch_analysis_worker = None
        # Set in close() before tearing the widget down — the analysis slots bail
        # on it so a worker signal queued just before teardown can't touch the
        # (about-to-be-destroyed) peak-fit dialog.
        self._tearing_down = False
        self._build_tools_placeholder()

    @staticmethod
    def _controls_v2_enabled() -> bool:
        value = os.environ.get("XDART_CONTROLS_PANEL_V2", "1")
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def _controls_v2_ensure_run_intent(self) -> RunIntent:
        """Return the Controls-owned mutable intent for the next run.

        ``self.scan`` is also the browser/display scan and may be replaced or
        hydrated while the operator is paused.  It is therefore only a seed at
        Controls construction time, never the long-lived owner of editable run
        settings.
        """

        intent = getattr(self, "_controls_v2_run_intent", None)
        if isinstance(intent, RunIntent):
            return intent

        scan = getattr(self, "scan", None)
        a1 = copy.deepcopy(getattr(scan, "bai_1d_args", {}) or {})
        a2 = copy.deepcopy(getattr(scan, "bai_2d_args", {}) or {})
        gic = copy.deepcopy(getattr(scan, "gi_config", {}) or {})
        motor = gic.get("incidence_motor")
        th_val = self._controls_v2_float(gic.get("th_val", 0.1), 0.1)
        if motor in (None, ""):
            motor = getattr(scan, "incidence_motor", None)
        if motor not in (None, ""):
            try:
                th_val = float(motor)
                motor = "Manual"
            except (TypeError, ValueError):
                motor = str(motor)
        else:
            motor = "Manual"

        controls = getattr(self, "controls", None)
        mode_getter = getattr(controls, "current_mode", None)
        try:
            processing_mode = str(mode_getter())
        except Exception:
            processing_mode = "Int 2D"
        if not processing_mode:
            processing_mode = "Int 2D"

        intent = RunIntent(
            processing_mode=processing_mode,
            bai_1d_args=a1,
            bai_2d_args=a2,
            gi=GIIntent(
                enabled=bool(getattr(scan, "gi", False) or gic),
                incidence_motor=str(motor),
                th_val=th_val,
                sample_orientation=self._controls_v2_int(
                    gic.get(
                        "sample_orientation",
                        getattr(scan, "sample_orientation", 4),
                    ),
                    4,
                ),
                tilt_angle=self._controls_v2_float(
                    gic.get("tilt_angle", getattr(scan, "tilt_angle", 0.0)),
                    0.0,
                ),
                mode_1d=str(a1.get("gi_mode_1d", "q_total")),
                mode_2d=str(a2.get("gi_mode_2d", "qip_qoop")),
            ),
            threshold=ThresholdIntent(
                apply_threshold=bool(
                    getattr(scan, "apply_threshold", False)),
                threshold_min=self._controls_v2_float(
                    getattr(scan, "threshold_min", 0.0), 0.0),
                threshold_max=self._controls_v2_float(
                    getattr(scan, "threshold_max", 0.0), 0.0),
                mask_saturation=bool(
                    getattr(scan, "mask_sentinel", True)),
            ),
        )
        self._controls_v2_run_intent = intent
        # A restored/session motor is deliberate.  A fresh LiveScan default is
        # not: when metadata later supplies ``halpha``/``th`` the existing
        # default-selection policy may adopt it as GI is enabled.
        self._controls_v2_gi_selection_explicit = bool(gic)
        self._controls_v2_threshold_state = intent.threshold.freeze().as_dict()
        return intent

    def _init_controls_v2_preview(self) -> None:
        """Mount the Controls Panel V2 editor.

        The panel is visible by default on the V2 branch.  It renders real
        editable rows backed by native Controls V2 state, while the legacy
        widgets stay alive only for delegated actions and the Advanced inspector.
        Set ``XDART_CONTROLS_PANEL_V2=0`` to compare against the legacy panel.
        """
        if not self._controls_v2_enabled():
            return
        try:
            from .ui.controls_panel_v2 import ControlsPanelV2
            panel = ControlsPanelV2(self.ui.wranglerFrame)
            preview = QtWidgets.QScrollArea(self.ui.wranglerFrame)
            preview.setObjectName("controlsPanelV2Preview")
            preview.setWidgetResizable(True)
            preview.setFrameShape(QtWidgets.QFrame.NoFrame)
            preview.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            preview.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            preview.setMinimumHeight(0)
            preview.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            panel.analysisLaunchRequested.connect(
                self._on_controls_v2_analysis_launch)
            panel.controlActionRequested.connect(
                self._on_controls_v2_action)
            panel.fieldValueChanged.connect(
                self._on_controls_v2_field_changed)
            panel.fieldDraftChanged.connect(
                self._on_controls_v2_field_draft_changed)
            panel.fieldBrowseRequested.connect(
                self._on_controls_v2_field_browse)
            preview.setWidget(panel)
            self.ui.verticalLayout.insertWidget(0, preview, 1)
            self.ui.verticalLayout.setStretchFactor(preview, 1)
            self.ui.wranglerStack.hide()
            # The legacy integrator Calibrate/Make Mask row (frame_3, lifted into
            # toolsLayout) is redundant under V2 — the V2 producer buttons render
            # in the Experiment section and delegate clicks to these.  Hide it so
            # it doesn't duplicate them (the buttons stay alive for delegation).
            try:
                self.integratorTree.ui.frame_3.hide()
            except Exception:
                pass
            self.controls_v2_preview = preview
            self.controls_v2 = panel
            from .scan_source_widget import ScanSourceWidget
            source_widget = ScanSourceWidget(
                mode="controls_source", parent=panel, async_probe=True)
            source_widget.sigDirectoryChanged.connect(
                self._on_controls_v2_directory_observation)
            self._controls_v2_source_widget = source_widget
            panel.set_source_widget(source_widget, visible=False)
            self._install_controls_v2_native_int_hooks()
            panel.set_processing_widget(self.ui.integratorFrame, visible=False)
            self._sync_controls_v2_source_index()
            self._refresh_controls_v2_profile(immediate=True)
        except Exception:
            self.controls_v2_preview = None
            self.controls_v2 = None
            self._controls_v2_source_widget = None
            logger.debug("Controls Panel V2 preview mount failed",
                         exc_info=True)

    def _install_controls_v2_native_int_hooks(self) -> None:
        """Make V2 native Int state the provider for legacy-owned actions."""

        integrator = getattr(self, "integratorTree", None)
        if integrator is None:
            return
        self._controls_v2_ensure_run_intent()
        integrator._controls_v2_native_args = True
        integrator.get_gi_config = self._controls_v2_gi_config
        integrator.get_threshold_config = self._controls_v2_threshold_config
        self._controls_v2_ensure_native_int_defaults()
        self._controls_v2_hydrate_advanced_from_scan()

    def _on_controls_v2_analysis_launch(self, tool) -> None:
        """Open the existing analysis popup for a V2 launcher intent."""
        if tool == AnalysisTool.PEAK_FIT:
            self._open_peak_fit_dialog()
        elif tool == AnalysisTool.PHASE_FIT:
            self._open_phase_fit_dialog()
        elif tool in (AnalysisTool.SCAN_PLOT, AnalysisTool.ROI_STATS):
            self._open_scan_plot_dialog()
        else:
            QMessageBox.information(
                self, "Tool not ready",
                "This analysis tool is scaffolded but not production-ready yet.")

    def _on_controls_v2_action(self, action) -> None:
        """Route Controls V2 preview actions through existing production hooks."""
        if self._controls_v2_run_active():
            logger.debug("Ignoring Controls V2 action during active run: %s", action)
            return
        if action == ControlAction.CHOOSE_SOURCE:
            self._controls_v2_choose_source()
        elif action == ControlAction.CHOOSE_PROJECT:
            self._controls_v2_choose_project()
        elif action == ControlAction.CHOOSE_OUTPUT:
            self._controls_v2_choose_output()
        elif action == ControlAction.CALIBRATE:
            import time as _time
            calib_started = _time.time()
            # §15.12-A.5: the PONI autofill is post-action work — it runs ONLY if
            # the calibrate click was actually PERFORMED (a pending invalid edit
            # refuses the click, so there is no new .poni to adopt).
            if self._controls_v2_click_integrator_button("pyfai_calib"):
                # pyFAI-calib2 runs as a BLOCKING external subprocess, so by here
                # it has closed.  It can't report the saved path back, so offer to
                # adopt any .poni it just wrote (confirmation popup).
                self._autofill_poni_after_calibrate(calib_started)
        elif action == ControlAction.MAKE_MASK:
            self._controls_v2_click_integrator_button("get_mask")
        elif action == ControlAction.REINTEGRATE_1D:
            self._controls_v2_click_integrator_button("reintegrate1D")
        elif action == ControlAction.REINTEGRATE_2D:
            self._controls_v2_click_integrator_button("reintegrate2D")
        elif action == ControlAction.ADVANCED_PROCESSING:
            refusal = self._commit_controls_v2_pending_edits()
            if refusal is not None:
                # §13.11 test 1: refuse visibly, do NOT open Advanced.
                self._controls_v2_report_pending_refusal(refusal, action)
            else:
                self._show_integration_advanced()
        elif action == ControlAction.REFINE_GEOMETRY:
            QMessageBox.information(
                self, "Refine geometry",
                "Geometry refinement is scaffolded and will be enabled after "
                "the real-data GUI gate lands.")
        else:
            QMessageBox.information(
                self, "Action not ready",
                "This control is scaffolded but not production-ready yet.")
        self._refresh_controls_v2_profile()

    def _controls_v2_click_integrator_button(self, button_name: str) -> bool:
        """Drive an integrator action button through the checked commit.

        §15.12-A.5: returns True when the action was PERFORMED (the delegated
        button click ran) and False when it was REFUSED (a pending invalid edit)
        or the button is unavailable — so an action owner runs its follow-up work
        (e.g. CALIBRATE's PONI autofill) only after a performed action."""
        if button_name in {"reintegrate1D", "reintegrate2D"}:
            refusal = self._apply_controls_v2_native_int_state(
                commit_pending=True,
                push_integrator=True,
            )
            if refusal is not None:
                # §13.11 test 1-2: an invalid focused edit refuses — zero
                # reintegration click, live state untouched, journal retained.
                self._controls_v2_report_pending_refusal(refusal, button_name)
                return False
            self._configure_controls_v2_native_run_plan(commit_pending=False)
        else:
            refusal = self._commit_controls_v2_pending_edits()
            if refusal is not None:
                # calibrate/mask routing: suppress the delegated click on refusal.
                self._controls_v2_report_pending_refusal(refusal, button_name)
                return False
        button = getattr(getattr(self.integratorTree, "ui", None), button_name, None)
        if button is None:
            return False
        click = getattr(button, "click", None)
        if callable(click):
            click()
            return True
        return False

    def _apply_controls_v2_field_value(self, path, value) -> bool:
        if self._set_controls_v2_native_source_field(path, value):
            return True
        if self._set_controls_v2_native_int_field(path, value):
            return True
        param = self._controls_v2_param(tuple(path))
        if param is None:
            return False
        try:
            current = param.value()
            new_value = coerce_control_edit_value(current, value)
            if current != new_value:
                param.setValue(new_value)
        except Exception:
            logger.debug("Controls Panel V2 field update failed for %s", path,
                         exc_info=True)
        return True

    def _commit_controls_v2_pending_edits(self):
        """Apply the panel's form edits for a NON-run consumer (reintegrate,
        advanced processing, calibration/mask routing, native-state apply)
        through the SAME validated stage→checked-commit owner (§12.9 item 7 /
        finding 1) — NEVER the permissive live setters.

        The revisioned journal (a differing focused value is recorded by
        ``_controls_v2_collect_pending_edits`` at harvest) is staged and
        checked-committed exactly once, so an older deferred entry cannot
        overwrite a newer focused correction and an invalid value is a typed
        refusal rather than a silent clamp/swallow.

        §13.11 owner 1-3 / §14.11.C.4: this is a NON-run helper, so it does NOT
        raise.  It returns ``None`` on success (edits applied, or nothing to
        apply) and, on a harvest/stage/commit/recovery refusal, returns the typed
        :class:`ControlsCommitResult` WITHOUT applying a partial or clamped value
        and WITHOUT leaking an exception through a Qt slot.  The journal is
        retained.  The CALLING production action is the ONE owner that renders a
        visible refusal + structured phase/path event and suppresses its
        delegated action/click (see ``_controls_v2_report_pending_refusal``)."""
        if getattr(self, "_run_active", False):
            return None  # run-active edits are journaled as deferred, not committed
        if not self._controls_v2_enabled():
            return None
        if not callable(getattr(
                getattr(self, "controls_v2", None), "current_form_edits", None)):
            return None
        # collect (inside the fold) journals a differing focused value, then the
        # journal is staged + checked-committed once.  A non-None fold IS the
        # typed refusal — hand it back to the action owner (never raise here).
        _fold = self._controls_v2_fold_deferred_edits_into_intent()
        if _fold is not None:
            return _fold
        # §15.12-A.2: a successful commit resolves any prior focus-loss refusal —
        # reset the dedupe signature so the NEXT distinct refusal messages again.
        self._controls_v2_last_refusal_signature = None
        self._refresh_controls_v2_profile(immediate=True)
        return None

    def _controls_v2_report_pending_refusal(self, result, action) -> None:
        """§14.11.C.4: render a NON-run checked-commit refusal (returned by
        ``_commit_controls_v2_pending_edits`` / ``_apply_controls_v2_native_int_state``)
        as ONE user-visible message + one structured phase/path event, at the
        production-action owner.  The delegated action/click is suppressed by the
        caller and the journal is retained so the user can correct and retry.  No
        exception leaks through the Qt slot."""
        if result is None:
            return
        # §15.12-A.2 / §17.7: ONE stable message per refused REVISION.  The
        # focus-loss idle commit surfaces the refusal first; the immediately-
        # following action click re-stages the SAME journal revision and gets the
        # SAME typed refusal — dedupe so it suppresses its action WITHOUT a
        # duplicate message.  The latch is keyed by the failed path's CURRENT
        # journal revision so a NEW invalid edit on the same path (e.g. "4.5"
        # then "5.5", a higher revision) is a DISTINCT visible refusal, not a
        # suppressed replay.  A successful commit resets the latch.
        revision = None
        if result.failed_path is not None:
            entry = self._controls_v2_edit_journal_dict().get(
                tuple(result.failed_path))
            if entry is not None:
                try:
                    revision = entry["revision"]
                except Exception:
                    revision = None
        signature = (
            revision, result.phase, result.failed_path, result.reason,
            result.recovery_failed_paths)
        if signature == getattr(
                self, "_controls_v2_last_refusal_signature", None):
            return
        self._controls_v2_last_refusal_signature = signature
        recovery = result.recovery_failed_path
        where = (
            "/".join(str(seg) for seg in result.failed_path)
            if result.failed_path else (result.phase or "control"))
        run_config_debug_log(
            logger,
            "controls_v2_action_refused_pending",
            widget=self,
            origin="controls_v2_action",
            action=str(action),
            phase=result.phase,
            failed_path=list(result.failed_path or ()),
            recovery_failed_path=list(recovery) if recovery else None,
            level="warning",
        )
        # §15.4-B.7: surface the engine's exact `reason` in the visible message
        # so the SAME specific cause survives into both the Run and non-run
        # refusals (the stack oracle asserts the reason string reaches the user).
        reason = (": " + result.reason) if result.reason else ""
        if recovery:
            message = (
                "A control edit could not be applied and a carrier ("
                + "/".join(str(seg) for seg in recovery)
                + ") could not be restored"
                + reason
                + " — the action was not performed.")
        else:
            message = (
                "A control edit is invalid (" + str(where) + ")"
                + reason
                + " — the action was not performed; re-check the control.")
        self._controls_v2_status_message(message)

    def _controls_v2_status_message(self, msg) -> None:
        """Non-modal user-visible status (mirrors ``_stitch_status``): route
        through the active wrangler's status text if present, else the window
        status bar.  Never raises."""
        wrangler = getattr(self, "wrangler", None)
        if wrangler is not None and hasattr(wrangler, "_set_status_text"):
            try:
                wrangler._set_status_text(msg)
                return
            except Exception:
                logger.debug("Controls V2 status route failed", exc_info=True)
        try:
            self.window().statusBar().showMessage(msg)
        except Exception:
            logger.debug("Controls V2 status message failed", exc_info=True)

    def _controls_v2_param(self, path):
        wrangler = getattr(self, "wrangler", None)
        params = getattr(wrangler, "parameters", None)
        if params is None:
            return None
        try:
            return params.child(*path)
        except Exception:
            return None

    def _controls_v2_field_paths(self):
        return BOUND_CONTROL_PATHS

    def _controls_v2_field_values(self, *, overlay_pending: bool = True):
        """Display field values for the panel.

        §13.11 owner 6: by DEFAULT the pending journal winners (the user's current
        uncommitted drafts/corrections) are overlaid on top of the committed
        values, so a forced ``set_state`` rebuild or a mode/schema change shows the
        pending draft (valid OR invalid) rather than reverting to the stale
        committed value.  Callers that need the COMMITTED baseline (e.g. the
        pending-edit collector, which compares harvested form values against
        what is committed) pass ``overlay_pending=False``."""
        values = {}
        for path in self._controls_v2_field_paths():
            if path in INTEGRATOR_BACKED_CONTROL_PATHS:
                continue
            param = self._controls_v2_param(path)
            if param is None:
                continue
            try:
                values[path] = param.value()
            except Exception:
                pass
        values.update(self._controls_v2_native_int_values())
        values.update(self._controls_v2_native_source_values())
        if overlay_pending:
            self._controls_v2_overlay_pending_values(values)
        return values

    def _controls_v2_overlay_pending_values(self, values) -> None:
        """Overlay the pending journal winners onto the committed display
        ``values`` in place (§13.11 owner 6).  Only display-bound paths are
        overlaid; a still-invalid draft is shown as its raw text so the user can
        see and correct it, and a rebuild never silently reverts it."""
        for path, value in self._controls_v2_journal_winners():
            if path in values:
                values[path] = value

    def _controls_v2_field_choices(self):
        choices = {}
        for path in self._controls_v2_field_paths():
            if path in INTEGRATOR_BACKED_CONTROL_PATHS:
                continue
            param = self._controls_v2_param(path)
            if param is None:
                continue
            opts = getattr(param, "opts", {}) or {}
            limits = opts.get("limits", None)
            if limits is None:
                limits = opts.get("values", None)
            if isinstance(limits, dict):
                vals = tuple(str(v) for v in limits.values())
            elif isinstance(limits, (list, tuple, set)):
                vals = tuple(str(v) for v in limits)
            else:
                vals = ()
            if vals:
                choices[path] = vals
        choices.update(self._controls_v2_native_int_choices())
        choices[("Source", "energy_preference")] = ("poni", "metadata")
        return choices

    def _controls_v2_native_source_values(self) -> dict[tuple[str, ...], object]:
        return {
            ("Source", "energy_preference"): self._controls_v2_energy_preference(),
        }

    @staticmethod
    def _controls_v2_number_text(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "" if value is None else str(value)
        if number.is_integer():
            return str(int(number))
        return str(number)

    @staticmethod
    def _controls_v2_float(value, default=0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _controls_v2_int(value, default=0, *, minimum=None) -> int:
        try:
            out = int(float(value))
        except (TypeError, ValueError):
            out = int(default)
        if minimum is not None:
            out = max(int(minimum), out)
        return out

    @staticmethod
    def _controls_v2_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "checked"}
        return bool(value)

    def _controls_v2_threshold_config(self):
        state = getattr(self, "_controls_v2_threshold_state", None)
        if not isinstance(state, dict):
            state = self._controls_v2_ensure_run_intent().threshold.freeze(
            ).as_dict()
            self._controls_v2_threshold_state = state
        intent = self._controls_v2_ensure_run_intent()
        intent.threshold = ThresholdIntent.from_mapping(state)
        return ThresholdSaturationConfig(
            apply_threshold=bool(state.get("apply_threshold", False)),
            threshold_min=self._controls_v2_float(
                state.get("threshold_min", 0.0), 0.0),
            threshold_max=self._controls_v2_float(
                state.get("threshold_max", 0.0), 0.0),
            mask_saturation=bool(state.get("mask_saturation", True)),
        )

    def _controls_v2_set_threshold_field(self, path, value) -> None:
        cfg = self._controls_v2_threshold_config()
        state = {
            "apply_threshold": bool(cfg.apply_threshold),
            "threshold_min": cfg.threshold_min,
            "threshold_max": cfg.threshold_max,
            "mask_saturation": bool(cfg.mask_saturation),
        }
        if path == ("Mask", "Threshold"):
            state["apply_threshold"] = self._controls_v2_bool(value)
        elif path == ("Mask", "min"):
            state["threshold_min"] = self._controls_v2_float(value, 0.0)
        elif path == ("Mask", "max"):
            state["threshold_max"] = self._controls_v2_float(value, 0.0)
        elif path == ("MaskSat", "mask_sentinel"):
            state["mask_saturation"] = self._controls_v2_bool(value)
        if path in {("Mask", "min"), ("Mask", "max")} and (
            state["threshold_min"] != 0.0 or state["threshold_max"] != 0.0
        ):
            state["apply_threshold"] = True
        self._controls_v2_threshold_state = state
        self._controls_v2_ensure_run_intent().threshold = (
            ThresholdIntent.from_mapping(state)
        )
        scan = getattr(self, "scan", None)
        if scan is not None:
            for attr, key in (
                ("apply_threshold", "apply_threshold"),
                ("threshold_min", "threshold_min"),
                ("threshold_max", "threshold_max"),
                ("mask_sentinel", "mask_saturation"),
            ):
                try:
                    setattr(scan, attr, state[key])
                except Exception:
                    pass

    def _controls_v2_default_gi_motor(self) -> str:
        # SINGLE source of truth for the GI incidence-motor default — the shared
        # policy (named preference th/eta/halpha/gonth/theta if present, else a
        # rotation-sounding motor, else Manual).  Do NOT re-implement a local
        # preference list here (it used to hard-prefer 'th' and fall back to the
        # first motor, re-injecting a phantom 'th' into the Controls-V2 θ-motor
        # row even after the wrangler/integrator combos were scoped correctly).
        from .gi_motor_defaults import pick_default_gi_motor
        choices = self._controls_v2_native_int_choices().get(("GI", "th_motor"), ())
        return pick_default_gi_motor([c for c in choices if str(c) != "Manual"])

    def _controls_v2_gi_config(self) -> dict:
        intent = self._controls_v2_ensure_run_intent()
        a1, a2 = self._controls_v2_scan_int_args()
        gi_intent = intent.gi
        gi = bool(gi_intent.enabled)
        motor = gi_intent.incidence_motor
        th_val = gi_intent.th_val
        if motor is None or str(motor) == "":
            motor = self._controls_v2_default_gi_motor()
        else:
            try:
                th_val = float(motor)
                motor = "Manual"
            except (TypeError, ValueError):
                motor = str(motor)
                # A stale/legacy incidence motor carried on the scan (notably the
                # 'th' default from LiveScan) that is NOT one of the loaded
                # source's real motors must not be shown — fall to the shared
                # default policy over the actual choices.  A genuine saved motor
                # (a real gi_config or an explicit user pick) is kept as-is even
                # when the source's motor list is not yet populated; but an
                # unverifiable, non-explicit leftover ('th' with no motors listed)
                # must still resolve to the default (Manual with no choices) so it
                # can never surface as a phantom selection (R4B-15).
                if motor != "Manual":
                    choices = self._controls_v2_native_int_choices().get(
                        ("GI", "th_motor"), ())
                    real_choices = tuple(
                        item for item in choices if str(item) != "Manual")
                    explicit = bool(
                        getattr(self, "_controls_v2_gi_selection_explicit", False))
                    if motor not in choices and (real_choices or not explicit):
                        motor = self._controls_v2_default_gi_motor()
        sample_orientation = gi_intent.sample_orientation
        tilt_angle = gi_intent.tilt_angle
        return {
            "gi": gi,
            "sample_orientation": self._controls_v2_int(sample_orientation, 4),
            "tilt_angle": self._controls_v2_float(tilt_angle, 0.0),
            "incidence_motor": str(motor or "Manual"),
            "th_val": self._controls_v2_float(th_val, 0.1),
            "gi_mode_1d": str(a1.get("gi_mode_1d", "q_total")),
            "gi_mode_2d": str(a2.get("gi_mode_2d", "qip_qoop")),
        }

    def _controls_v2_apply_gi_config_to_scan(self, cfg=None, scan=None) -> None:
        # §17.8: the target defaults to the LIVE scan (the real projection call
        # sites), but a PURE builder passes an aside copy so it can read the
        # projected GI fields without mutating the live scan.
        if scan is None:
            scan = getattr(self, "scan", None)
        if scan is None:
            return
        if cfg is None:
            cfg = self._controls_v2_gi_config()
        scan.gi = bool(cfg["gi"])
        if not cfg["gi"]:
            scan.gi_config = {}
            return
        scan.gi_config = {
            "gi_mode_1d": str(cfg["gi_mode_1d"]),
            "gi_mode_2d": str(cfg["gi_mode_2d"]),
            "incidence_motor": str(cfg["incidence_motor"] or ""),
            "th_val": float(cfg["th_val"] or 0.0),
            "tilt_angle": float(cfg["tilt_angle"] or 0.0),
            "sample_orientation": int(cfg["sample_orientation"] or 4),
        }
        incidence = (
            str(cfg["th_val"])
            if cfg["incidence_motor"] == "Manual"
            else str(cfg["incidence_motor"] or "")
        )
        scan.incidence_motor = incidence
        scan.th_mtr = incidence
        scan.sample_orientation = int(cfg["sample_orientation"] or 4)
        scan.tilt_angle = float(cfg["tilt_angle"] or 0.0)

    def _controls_v2_sync_integrator_gi_motor(self, motor) -> None:
        """Point the integrator's GI θ-motor combo at *motor* (the two θ-motor
        surfaces must never disagree — CLAUDE.md GI rule).  No-op when the combo
        does not offer *motor* or is already there."""
        it = getattr(self, "integratorTree", None)
        combo = getattr(getattr(it, "ui", None), "gi_motor", None)
        if combo is None:
            return
        try:
            idx = combo.findText(str(motor))
            if idx >= 0 and combo.currentIndex() != idx:
                combo.setCurrentIndex(idx)
        except Exception:
            logger.debug("integrator GI motor combo sync failed", exc_info=True)

    def _controls_v2_set_gi_field(self, leaf: str, value) -> None:
        scan = getattr(self, "scan", None)
        intent = self._controls_v2_ensure_run_intent()
        cfg = self._controls_v2_gi_config()
        if leaf == "Grazing":
            cfg["gi"] = self._controls_v2_bool(value)
            # Live-found 2026-07-12 (LaB6, halpha in the list but Manual
            # selected): enabling GI with NO saved gi_config means this
            # 'Manual' is a LEFTOVER — a session-restored default or the
            # scan-carried numeric theta — not a deliberate choice.  Apply the
            # shared default policy so a present preference motor
            # (th/halpha/…) wins.  A real saved gi_config (deliberate Manual
            # + its theta) is honored untouched.
            if (cfg["gi"] and cfg["incidence_motor"] == "Manual"
                    and not getattr(
                        self, "_controls_v2_gi_selection_explicit", False)):
                picked = self._controls_v2_default_gi_motor()
                if picked != "Manual":
                    cfg["incidence_motor"] = picked
            # Keep the integrator combo equal to whatever motor the config
            # resolved (the two θ-motor surfaces must never disagree — CLAUDE.md
            # GI rule).  R4B-15: ``_controls_v2_gi_config`` may already have
            # repicked a stale 'th'→real motor before this handler ran, so the
            # combo must be synced to the resolved value even when the Manual
            # re-pick branch above did not fire.
            if cfg["gi"]:
                self._controls_v2_sync_integrator_gi_motor(cfg["incidence_motor"])
        elif leaf == "th_motor":
            cfg["incidence_motor"] = str(value)
            self._controls_v2_gi_selection_explicit = True
        elif leaf == "th_val":
            cfg["th_val"] = self._controls_v2_float(value, cfg.get("th_val", 0.1))
        elif leaf == "sample_orientation":
            cfg["sample_orientation"] = self._controls_v2_int(value, 4)
        elif leaf == "tilt_angle":
            cfg["tilt_angle"] = self._controls_v2_float(value, 0.0)
        intent.gi = GIIntent(
            enabled=bool(cfg["gi"]),
            incidence_motor=str(cfg["incidence_motor"] or "Manual"),
            th_val=float(cfg["th_val"] or 0.0),
            sample_orientation=int(cfg["sample_orientation"] or 4),
            tilt_angle=float(cfg["tilt_angle"] or 0.0),
            mode_1d=str(cfg["gi_mode_1d"]),
            mode_2d=str(cfg["gi_mode_2d"]),
        )
        if intent.gi.enabled:
            a1, a2 = self._controls_v2_scan_int_args()
            a1.setdefault("gi_mode_1d", "q_total")
            a2.setdefault("gi_mode_2d", "qip_qoop")
            a1["unit"] = "q_A^-1"
            a2["unit"] = "q_A^-1"
        if scan is not None:
            self._controls_v2_apply_gi_config_to_scan(cfg)

    @staticmethod
    def _controls_v2_replace_dict_in_place(scan, attr, value) -> None:
        """Project a run-config mapping onto the display scan by CLEAR+UPDATE.

        R4B-15: replacing ``scan.bai_1d_args`` with a fresh object on every
        Controls edit invalidated held references in the reintegration/display
        paths (the S10 polarization round-trip lost its dict).  Mutating the
        existing dict in place keeps a stable identity while a deep copy of the
        snapshot value guarantees NO aliasing with the Controls-owned intent
        (the intent stays the single writer of run configuration).
        """
        fresh = copy.deepcopy(value) if isinstance(value, dict) else {}
        existing = getattr(scan, attr, None)
        if isinstance(existing, dict):
            existing.clear()
            existing.update(fresh)
        else:
            setattr(scan, attr, fresh)

    @staticmethod
    def _controls_v2_apply_native_int_snapshot_to_scan(
        snapshot: dict,
        scan,
    ) -> None:
        if scan is None or not isinstance(snapshot, dict):
            return
        lock = getattr(scan, "scan_lock", None)

        def _apply():
            staticWidget._controls_v2_replace_dict_in_place(
                scan, "bai_1d_args", snapshot.get("bai_1d_args", {}) or {})
            staticWidget._controls_v2_replace_dict_in_place(
                scan, "bai_2d_args", snapshot.get("bai_2d_args", {}) or {})
            scan.gi = bool(snapshot.get("gi", False))
            staticWidget._controls_v2_replace_dict_in_place(
                scan, "gi_config", snapshot.get("gi_config", {}) or {})
            for attr in (
                "incidence_motor",
                "th_mtr",
                "sample_orientation",
                "tilt_angle",
            ):
                if attr in snapshot:
                    setattr(scan, attr, copy.deepcopy(snapshot[attr]))

        if lock is None:
            _apply()
        else:
            with lock:
                _apply()

    def _controls_v2_apply_snapshot_to_scan(self, snapshot: dict, scan=None) -> None:
        scan = scan if scan is not None else getattr(self, "scan", None)
        self._controls_v2_apply_native_int_snapshot_to_scan(snapshot, scan)

    @staticmethod
    def _scan_data_reduction_config_snapshot(scan) -> dict:
        if scan is None:
            return {}
        try:
            from xrd_tools.reduction.provenance_config import (
                build_reduction_config,
            )
            # Display provenance needs only the integration ``config``; the
            # ``inputs`` (raw_files/meta_file) are discarded below.  Computing
            # them walks the ENTIRE frame series with a per-frame disk read under
            # ``file_lock`` (frame_series.__getitem__), which on the GUI thread
            # freezes the UI for the whole run when a large scan is loaded --
            # every non-resident frame contends with the live reduction pipeline
            # for the same lock.  The authoritative raw_files provenance is still
            # written by the nexus writer / headless core on background threads.
            config, _inputs = build_reduction_config(scan, include_inputs=False)
        except Exception:
            config = {
                "bai_1d_args": copy.deepcopy(
                    getattr(scan, "bai_1d_args", {}) or {}
                ),
                "bai_2d_args": copy.deepcopy(
                    getattr(scan, "bai_2d_args", {}) or {}
                ),
            }
            gic = copy.deepcopy(getattr(scan, "gi_config", {}) or {})
            if gic:
                config["gi_config"] = gic
        config = copy.deepcopy(config)
        config["gi"] = bool(getattr(scan, "gi", False))
        return config

    def _stamp_scan_data_reduction_config(self, scan=None) -> None:
        """Record the config that produced the data currently on display."""

        scan = scan if scan is not None else getattr(self, "scan", None)
        if scan is None:
            return
        config = staticWidget._scan_data_reduction_config_snapshot(scan)
        try:
            scan.reduction_config = copy.deepcopy(config)
            scan._display_reduction_config = copy.deepcopy(config)
        except Exception:
            logger.debug("failed to stamp scan reduction config", exc_info=True)

    def _controls_v2_push_threshold_to_integrator(self) -> None:
        thread = getattr(getattr(self, "integratorTree", None),
                         "integrator_thread", None)
        if thread is not None:
            thread.threshold_config = self._controls_v2_threshold_config()

    def _apply_controls_v2_native_int_state(
        self,
        *,
        commit_pending: bool = True,
        push_wrangler: bool = False,
        push_integrator: bool = False,
    ):
        """Apply native V2 Int/GI/threshold state to the scan and consumers.

        §13.11 owner 3 / §14.11.C.4: when ``commit_pending`` and the checked
        commit REFUSES an invalid focused edit, this returns the typed
        :class:`ControlsCommitResult` and applies NOTHING to the scan — the
        reintegrate action owner then suppresses its button click and renders the
        refusal.  Returns ``None`` on success."""

        run_config_debug_log(
            logger,
            "native_int_apply_enter",
            widget=self,
            origin="controls_v2_native_int_state",
            commit_pending=commit_pending,
            push_wrangler=push_wrangler,
            push_integrator=push_integrator,
        )
        if commit_pending:
            refusal = self._commit_controls_v2_pending_edits()
            if refusal is not None:
                # Fail closed: do NOT project a partial/clamped snapshot onto the
                # scan; hand the refusal back so the action owner suppresses.
                return refusal
        self._controls_v2_ensure_native_int_defaults()
        self._controls_v2_apply_snapshot_to_scan(
            self._controls_v2_native_int_snapshot()
        )
        threshold = self._controls_v2_threshold_config()
        scan = getattr(self, "scan", None)
        if scan is not None:
            for attr, value in (
                ("apply_threshold", threshold.apply_threshold),
                ("threshold_min", threshold.threshold_min),
                ("threshold_max", threshold.threshold_max),
                ("mask_sentinel", threshold.mask_saturation),
            ):
                setattr(scan, attr, value)
        run_config_debug_log(
            logger,
            "native_int_scan_applied",
            widget=self,
            origin="controls_v2_native_int_state",
        )
        if push_wrangler:
            run_config_debug_log(
                logger,
                "threshold_push_enter",
                widget=self,
                origin="controls_v2_native_int_state",
            )
            self._push_threshold_to_wrangler()
            run_config_debug_log(
                logger,
                "threshold_push_exit",
                widget=self,
                origin="controls_v2_native_int_state",
            )
            self._push_gi_to_wrangler()
            run_config_debug_log(
                logger,
                "gi_push_exit",
                widget=self,
                origin="controls_v2_native_int_state",
            )
        if push_integrator:
            self._controls_v2_push_threshold_to_integrator()
        run_config_debug_log(
            logger,
            "native_int_apply_exit",
            widget=self,
            origin="controls_v2_native_int_state",
        )

    def _controls_v2_native_int_snapshot(self) -> dict:
        intent = self._controls_v2_ensure_run_intent()
        cfg = self._controls_v2_gi_config()
        incidence = (
            str(cfg["th_val"])
            if cfg["incidence_motor"] == "Manual"
            else str(cfg["incidence_motor"] or "")
        )
        return {
            "bai_1d_args": copy.deepcopy(intent.bai_1d_args),
            "bai_2d_args": copy.deepcopy(intent.bai_2d_args),
            "gi": bool(cfg["gi"]),
            "gi_config": ({
                "gi_mode_1d": str(cfg["gi_mode_1d"]),
                "gi_mode_2d": str(cfg["gi_mode_2d"]),
                "incidence_motor": str(cfg["incidence_motor"]),
                "th_val": float(cfg["th_val"]),
                "sample_orientation": int(cfg["sample_orientation"]),
                "tilt_angle": float(cfg["tilt_angle"]),
            } if cfg["gi"] else {}),
            "incidence_motor": incidence,
            "th_mtr": incidence,
            "sample_orientation": int(cfg["sample_orientation"]),
            "tilt_angle": float(cfg["tilt_angle"]),
        }

    @staticmethod
    def _controls_v2_native_int_snapshot_key(value):
        if isinstance(value, dict):
            return tuple(
                (str(key), staticWidget._controls_v2_native_int_snapshot_key(val))
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(
                staticWidget._controls_v2_native_int_snapshot_key(val)
                for val in value
            )
        if isinstance(value, set):
            return tuple(
                sorted(
                    staticWidget._controls_v2_native_int_snapshot_key(val)
                    for val in value
                )
            )
        try:
            hash(value)
        except TypeError:
            tolist = getattr(value, "tolist", None)
            if callable(tolist):
                return staticWidget._controls_v2_native_int_snapshot_key(
                    tolist()
                )
            return repr(value)
        return value

    def _controls_v2_scan_int_args(self):
        intent = self._controls_v2_ensure_run_intent()
        if not isinstance(intent.bai_1d_args, dict):
            intent.bai_1d_args = {}
        if not isinstance(intent.bai_2d_args, dict):
            intent.bai_2d_args = {}
        return intent.bai_1d_args, intent.bai_2d_args

    def _controls_v2_ensure_native_int_defaults(self) -> None:
        a1, a2 = self._controls_v2_scan_int_args()
        defaults_1d = {
            "unit": "q_A^-1",
            "numpoints": 3000,
            "radial_range": None,
            "azimuth_range": None,
            "correctSolidAngle": True,
            "dummy": -1.0,
            "delta_dummy": 0.0,
            "chi_offset": 90.0,
            # ON by default (maintainer, 2026-07-13); None = deliberate off.
            "polarization_factor": DEFAULT_POLARIZATION_FACTOR,
            "method": "csr",
            "safe": True,
        }
        defaults_2d = {
            "unit": "q_A^-1",
            "npt_rad": 500,
            "npt_azim": 500,
            "radial_range": None,
            "azimuth_range": None,
            "correctSolidAngle": True,
            "dummy": -1.0,
            "delta_dummy": 0.0,
            "chi_offset": 90.0,
            # ON by default (maintainer, 2026-07-13); None = deliberate off.
            "polarization_factor": DEFAULT_POLARIZATION_FACTOR,
            "method": "csr",
            "safe": True,
        }
        for key, value in defaults_1d.items():
            a1.setdefault(key, copy.deepcopy(value))
        for key, value in defaults_2d.items():
            a2.setdefault(key, copy.deepcopy(value))
        if bool(getattr(getattr(self, "scan", None), "gi", False)):
            a1.setdefault("gi_mode_1d", "q_total")
            a2.setdefault("gi_mode_2d", "qip_qoop")
            # GI integration is Q-space only in this panel.
            a1["unit"] = "q_A^-1"
            a2["unit"] = "q_A^-1"

    def _controls_v2_unit_display(self, unit: object, *, dim: str = "1d") -> str:
        code = str(unit or "q_A^-1")
        idx = Units_dict_inv.get(code, 0)
        if dim == "2d" and idx >= 2:
            idx = 0
        try:
            return Units[idx]
        except Exception:
            return Units[0]

    def _controls_v2_unit_code(
            self, text: object, *, dim: str = "1d", strict: bool = False) -> str:
        value = str(text or "").strip()
        code = self._controls_v2_unit_code_exact(value)
        if code is None:
            if strict:
                # §15.12-A.7 / §13.4: on the STRICT staging path a value that is
                # not an EXACT approved unit/alias is a typed refusal — never a
                # fuzzy collapse (the caller wraps this ValueError into a
                # path-qualified ControlsTransactionError).  ``2garbage`` and
                # ``machine`` refuse here instead of matching ``2th``/``chi``.
                raise ValueError(f"unknown axis/unit {value!r}")
            code = self._controls_v2_unit_code_fuzzy(value)
        if dim == "2d" and code == "chi_deg":
            code = "q_A^-1"
        return code

    @staticmethod
    def _controls_v2_unit_code_exact(value: str):
        """§15.12-A.7: EXACT unit resolution (display name / code / enumerated
        alias).  Returns the canonical code, or ``None`` when the value is not an
        approved spelling."""
        if value in Units_dict:
            return Units_dict[value]
        if value in Units_dict_inv:
            return value
        key = value.casefold()
        for code, aliases in _CONTROLS_V2_STRICT_UNIT_ALIASES.items():
            if key in aliases:
                return code
        return None

    @staticmethod
    def _controls_v2_unit_code_fuzzy(value: str) -> str:
        """Lenient display/import normalization — NON-strict path ONLY.  Legacy
        spellings collapse to the nearest unit; a genuinely unknown value falls
        back to Q.  This substring/startswith heuristic must never run on the
        strict staging path (§15.7)."""
        low = value.lower()
        if value.startswith("2") or "2θ" in value or "2th" in low:
            return "2th_deg"
        if "chi" in low or "χ" in value:
            return "chi_deg"
        return "q_A^-1"

    def _controls_v2_axis_display(self, root: str) -> str:
        self._controls_v2_ensure_native_int_defaults()
        scan = getattr(self, "scan", None)
        a1, a2 = self._controls_v2_scan_int_args()
        if bool(getattr(scan, "gi", False)):
            if root == "Int1D":
                mode = a1.get("gi_mode_1d", "q_total")
                return GI_LABELS_1D[GI_MODES_1D.index(mode)] if mode in GI_MODES_1D else GI_LABELS_1D[0]
            mode = a2.get("gi_mode_2d", "qip_qoop")
            return GI_LABELS_2D[GI_MODES_2D.index(mode)] if mode in GI_MODES_2D else GI_LABELS_2D[0]
        if root == "Int1D":
            return self._controls_v2_unit_display(a1.get("unit"), dim="1d")
        return "2θ-χ" if a2.get("unit") == "2th_deg" else "Q-χ"

    def _controls_v2_axis_to_native(self, root: str, value: object) -> None:
        self._controls_v2_ensure_native_int_defaults()
        scan = getattr(self, "scan", None)
        a1, a2 = self._controls_v2_scan_int_args()
        text = str(value or "")
        if bool(getattr(scan, "gi", False)):
            if root == "Int1D":
                old_mode = a1.get("gi_mode_1d")
                try:
                    a1["gi_mode_1d"] = GI_MODES_1D[GI_LABELS_1D.index(text)]
                except ValueError:
                    a1["gi_mode_1d"] = "q_total"
                if a1["gi_mode_1d"] != old_mode:
                    _drop_output_axis_ranges(a1)
                if self._controls_v2_npts_oop_visible():
                    a1.setdefault("npt_oop", int(a1.get("numpoints", 3000)))
            else:
                old_mode = a2.get("gi_mode_2d")
                try:
                    a2["gi_mode_2d"] = GI_MODES_2D[GI_LABELS_2D.index(text)]
                except ValueError:
                    a2["gi_mode_2d"] = "qip_qoop"
                if a2["gi_mode_2d"] != old_mode:
                    _drop_output_axis_ranges(a2)
            a1["unit"] = "q_A^-1"
            a2["unit"] = "q_A^-1"
            # §10/SW-8: gi_mode lives authoritatively in bai_*_args; the
            # scan-carried gi_config COPY (persisted, pushed to the wrangler,
            # and fed verbatim into written provenance by
            # build_reduction_config) must not lag an Axis edit — the only
            # gi_mode edit path that does not re-stamp gi_config.
            gic = getattr(scan, "gi_config", None)
            if gic:
                gic["gi_mode_1d"] = str(a1.get("gi_mode_1d", "q_total"))
                gic["gi_mode_2d"] = str(a2.get("gi_mode_2d", "qip_qoop"))
            return
        if root == "Int1D":
            old_unit = a1.get("unit")
            a1["unit"] = self._controls_v2_unit_code(text, dim="1d")
            if a1["unit"] != old_unit:
                a1.pop("radial_range", None)   # S-5: q<->2theta invalidates the range
        else:
            old_unit = a2.get("unit")
            a2["unit"] = "2th_deg" if text.startswith("2") else "q_A^-1"
            if a2["unit"] != old_unit:
                a2.pop("radial_range", None)

    def _controls_v2_default_range(self, root: str, axis: str):
        self._controls_v2_ensure_native_int_defaults()
        scan = getattr(self, "scan", None)
        a1, a2 = self._controls_v2_scan_int_args()
        gi = bool(getattr(scan, "gi", False))
        if root == "Int1D":
            if gi:
                mode = a1.get("gi_mode_1d", "q_total")
                if axis == "radial":
                    return (-5.0, 5.0) if mode == "exit_angle" else (
                        (-10.0, 10.0) if mode in {"q_ip", "q_oop"} else (0.0, 5.0)
                    )
                if mode in {"q_ip", "q_oop"}:
                    return (0.0, 5.0)
                if mode == "exit_angle":
                    return (0.0, 90.0)
                return (-180.0, 180.0)
            if axis == "radial":
                return (0.0, 90.0) if a1.get("unit") == "2th_deg" else (0.0, 5.0)
            return (-180.0, 180.0)
        if gi:
            mode = a2.get("gi_mode_2d", "qip_qoop")
            if axis == "radial":
                return (-5.0, 5.0) if mode == "exit_angles" else (
                    (-10.0, 10.0) if mode == "qip_qoop" else (0.0, 5.0)
                )
            if mode == "qip_qoop":
                return (0.0, 5.0)
            if mode == "exit_angles":
                return (0.0, 90.0)
            return (-180.0, 180.0)
        if axis == "radial":
            return (0.0, 90.0) if a2.get("unit") == "2th_deg" else (0.0, 5.0)
        return (-180.0, 180.0)

    def _controls_v2_range_value(self, root: str, axis: str):
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        value = args.get(key)
        return value if value is not None else self._controls_v2_default_range(root, axis)

    def _controls_v2_set_range_auto(self, root: str, axis: str, auto: bool) -> None:
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        value = None if auto else self._controls_v2_range_value(root, axis)
        args[key] = value
        if root == "Int2D" and bool(getattr(getattr(self, "scan", None), "gi", False)):
            alt = "x_range" if axis == "radial" else "y_range"
            if value is None:
                args.pop(alt, None)
            else:
                args[alt] = value
        if root == "Int1D" and axis == "azimuth" and self._controls_v2_npts_oop_visible():
            args.setdefault("npt_oop", int(args.get("numpoints", 3000)))

    def _controls_v2_set_range_bound(
        self,
        root: str,
        axis: str,
        bound: str,
        value: object,
    ) -> None:
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        low, high = self._controls_v2_range_value(root, axis)
        number = self._controls_v2_float(value, low if bound == "low" else high)
        if bound == "low":
            low = number
        else:
            high = number
        args[key] = (float(low), float(high))
        if root == "Int2D" and bool(getattr(getattr(self, "scan", None), "gi", False)):
            args["x_range" if axis == "radial" else "y_range"] = args[key]
        if root == "Int1D" and axis == "azimuth" and self._controls_v2_npts_oop_visible():
            args.setdefault("npt_oop", int(args.get("numpoints", 3000)))

    def _controls_v2_npts_oop_visible(self) -> bool:
        scan = getattr(self, "scan", None)
        if not bool(getattr(scan, "gi", False)):
            return False
        a1, _ = self._controls_v2_scan_int_args()
        return (
            a1.get("gi_mode_1d", "q_total") != "q_total"
            or a1.get("azimuth_range") is not None
        )

    def _controls_v2_integrator_parameter(self, spec):
        integrator = getattr(self, "integratorTree", None)
        if integrator is None or not getattr(spec, "parameter_name", ""):
            return None
        tree_name = {
            "1d": "bai_1d_pars",
            "2d": "bai_2d_pars",
        }.get(spec.parameter_group)
        tree = getattr(integrator, tree_name, None)
        if tree is None:
            return None
        try:
            return tree.child(spec.parameter_name)
        except Exception:
            return None

    def _controls_v2_advanced_value(self, root: str, leaf: str):
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        if leaf == "apply_polarization":
            return args.get("polarization_factor") is not None
        if leaf == "polarization_factor":
            value = args.get("polarization_factor")
            return 0.0 if value is None else value
        defaults = {
            "correctSolidAngle": True,
            "method": "csr",
            "dummy": -1.0,
            "delta_dummy": 0.0,
            "chi_offset": 90.0,
            "safe": True,
        }
        return args.get(leaf, defaults.get(leaf, ""))

    def _controls_v2_native_int_values(self):
        self._controls_v2_ensure_native_int_defaults()
        a1, a2 = self._controls_v2_scan_int_args()
        gi_cfg = self._controls_v2_gi_config()
        threshold = self._controls_v2_threshold_config()
        r1 = self._controls_v2_range_value("Int1D", "radial")
        z1 = self._controls_v2_range_value("Int1D", "azimuth")
        r2 = self._controls_v2_range_value("Int2D", "radial")
        z2 = self._controls_v2_range_value("Int2D", "azimuth")

        values = {
            ("GI", "Grazing"): bool(gi_cfg["gi"]),
            ("GI", "th_motor"): str(gi_cfg["incidence_motor"]),
            ("GI", "th_val"): self._controls_v2_number_text(gi_cfg["th_val"]),
            ("GI", "sample_orientation"): int(gi_cfg["sample_orientation"]),
            ("GI", "tilt_angle"): self._controls_v2_number_text(gi_cfg["tilt_angle"]),
            ("Mask", "Threshold"): bool(threshold.apply_threshold),
            ("Mask", "min"): self._controls_v2_number_text(threshold.threshold_min),
            ("Mask", "max"): self._controls_v2_number_text(threshold.threshold_max),
            ("MaskSat", "mask_sentinel"): bool(threshold.mask_saturation),
            ("Int1D", "unit"): self._controls_v2_unit_display(a1.get("unit"), dim="1d"),
            ("Int1D", "axis"): self._controls_v2_axis_display("Int1D"),
            ("Int1D", "points"): str(int(a1.get("numpoints", 3000))),
            ("Int1D", "radial_auto"): a1.get("radial_range") is None,
            ("Int1D", "radial_low"): self._controls_v2_number_text(r1[0]),
            ("Int1D", "radial_high"): self._controls_v2_number_text(r1[1]),
            ("Int1D", "azim_auto"): a1.get("azimuth_range") is None,
            ("Int1D", "azim_low"): self._controls_v2_number_text(z1[0]),
            ("Int1D", "azim_high"): self._controls_v2_number_text(z1[1]),
            ("Int2D", "unit"): self._controls_v2_unit_display(a2.get("unit"), dim="2d"),
            ("Int2D", "axis"): self._controls_v2_axis_display("Int2D"),
            ("Int2D", "radial_points"): str(int(a2.get("npt_rad", 500))),
            ("Int2D", "azim_points"): str(int(a2.get("npt_azim", 500))),
            ("Int2D", "radial_auto"): a2.get("radial_range") is None,
            ("Int2D", "radial_low"): self._controls_v2_number_text(r2[0]),
            ("Int2D", "radial_high"): self._controls_v2_number_text(r2[1]),
            ("Int2D", "azim_auto"): a2.get("azimuth_range") is None,
            ("Int2D", "azim_low"): self._controls_v2_number_text(z2[0]),
            ("Int2D", "azim_high"): self._controls_v2_number_text(z2[1]),
        }
        if bool(gi_cfg["gi"]):
            values[("Int1D", "gi_mode")] = a1.get("gi_mode_1d", "q_total")
            values[("Int2D", "gi_mode")] = a2.get("gi_mode_2d", "qip_qoop")
        if self._controls_v2_npts_oop_visible():
            values[("Int1D", "points_oop")] = str(
                int(a1.get("npt_oop", a1.get("numpoints", 3000)))
            )
        for spec in INTEGRATOR_BACKED_CONTROL_SPECS:
            if not spec.parameter_name:
                continue
            values[spec.path] = self._controls_v2_advanced_value(
                spec.path[0], spec.path[1]
            )
        return values

    def _controls_v2_native_int_choices(self):
        integrator = getattr(self, "integratorTree", None)
        ui = getattr(integrator, "ui", None)
        gi = bool(self._controls_v2_ensure_run_intent().gi.enabled)

        def _combo_choices(name):
            combo = getattr(ui, name, None)
            if combo is None:
                return ()
            return tuple(combo.itemText(i) for i in range(combo.count()))

        def _gi_motor_choices():
            """θ-motor choices: the integrator combo's items, but fall back to the
            active wrangler's freshly-discovered motors when that combo hasn't
            been repopulated yet (e.g. a session-restored source, where the
            sigGIMotorOptions handshake hasn't fired).  Keeps 'Manual' always
            offered and never invents a motor the source doesn't have."""
            items = _combo_choices("gi_motor")
            if any(str(c) != "Manual" for c in items):
                return items
            wr = getattr(self, "wrangler", None)
            wr_motors = [
                str(m) for m in (getattr(wr, "motors", None) or [])
                if not any(x in str(m).lower() for x in ("roi", "pd"))
            ]
            return tuple(["Manual"] + wr_motors) if wr_motors else (items or ("Manual",))

        choices = {
            ("Int1D", "unit"): tuple(Units),
            ("Int2D", "unit"): tuple(Units[:2]),
            ("Int1D", "axis"): tuple(GI_LABELS_1D if gi else Units),
            ("Int2D", "axis"): tuple(GI_LABELS_2D if gi else ("Q-χ", "2θ-χ")),
            ("GI", "th_motor"): _gi_motor_choices(),
        }
        for spec in INTEGRATOR_BACKED_CONTROL_SPECS:
            if spec.kind.value == "combo" and spec.parameter_name:
                param = self._controls_v2_integrator_parameter(spec)
                opts = getattr(param, "opts", {}) or {}
                vals = opts.get("limits", None)
                if vals is None:
                    vals = opts.get("values", None)
                if isinstance(vals, dict):
                    choices[spec.path] = tuple(str(v) for v in vals.values())
                elif isinstance(vals, (list, tuple, set)):
                    choices[spec.path] = tuple(str(v) for v in vals)
        return {path: vals for path, vals in choices.items() if vals}

    def _controls_v2_sync_advanced_parameter(self, path) -> None:
        spec = next(
            (spec for spec in INTEGRATOR_BACKED_CONTROL_SPECS
             if spec.path == tuple(path) and spec.parameter_name),
            None,
        )
        if spec is None:
            return
        param = self._controls_v2_integrator_parameter(spec)
        if param is None:
            return
        value = self._controls_v2_advanced_value(spec.path[0], spec.path[1])
        if spec.path[1] == "polarization_factor" and value is None:
            value = 0.0
        try:
            if param.value() != value:
                param.setValue(value)
        except Exception:
            logger.debug("Controls Panel V2 advanced mirror failed for %s",
                         path, exc_info=True)

    def _controls_v2_hydrate_advanced_from_scan(self) -> None:
        integrator = getattr(self, "integratorTree", None)
        if integrator is None:
            return
        self._controls_v2_ensure_native_int_defaults()
        a1, a2 = self._controls_v2_scan_int_args()
        keys = {
            "correctSolidAngle",
            "dummy",
            "delta_dummy",
            "chi_offset",
            "polarization_factor",
            "method",
            "safe",
        }
        try:
            integrator._args_to_params(
                {k: v for k, v in a1.items() if k in keys},
                integrator.bai_1d_pars,
                dim="1D",
            )
            integrator._args_to_params(
                {k: v for k, v in a2.items() if k in keys},
                integrator.bai_2d_pars,
                dim="2D",
            )
        except Exception:
            logger.debug("Controls Panel V2 advanced hydrate failed",
                         exc_info=True)

    def _set_controls_v2_native_source_field(self, path, value) -> bool:
        path = tuple(path)
        if path not in NATIVE_CONTROL_PATHS:
            return False
        if path == ("Source", "energy_preference"):
            text = str(value or "").strip().lower()
            aliases = {
                "poni": "poni",
                "poni file": "poni",
                "metadata": "metadata",
                "meta": "metadata",
            }
            self._controls_v2_source_energy_preference = aliases.get(text, "poni")
            self._controls_v2_source_energy_cache = None
            return True
        return False

    def _controls_v2_energy_preference(self) -> str:
        pref = str(
            getattr(self, "_controls_v2_source_energy_preference", "poni")
            or "poni"
        ).strip().lower()
        return pref if pref in {"poni", "metadata"} else "poni"

    def _set_controls_v2_native_int_field(self, path, value) -> bool:
        path = tuple(path)
        if path not in INTEGRATOR_BACKED_CONTROL_PATHS:
            return False
        root = path[0]
        leaf = path[1] if len(path) > 1 else ""
        if root == "GI":
            self._controls_v2_set_gi_field(leaf, value)
            return True
        if root in {"Mask", "MaskSat"}:
            self._controls_v2_set_threshold_field(path, value)
            return True
        if root not in {"Int1D", "Int2D"}:
            return True
        self._controls_v2_ensure_native_int_defaults()
        a1, a2 = self._controls_v2_scan_int_args()
        args = a1 if root == "Int1D" else a2
        if leaf == "unit":
            args["unit"] = self._controls_v2_unit_code(
                value, dim="1d" if root == "Int1D" else "2d")
        elif leaf == "axis":
            self._controls_v2_axis_to_native(root, value)
        elif leaf == "points":
            args["numpoints"] = self._controls_v2_int(value, 3000, minimum=1)
            if self._controls_v2_npts_oop_visible():
                args.setdefault("npt_oop", args["numpoints"])
        elif leaf == "points_oop":
            args["npt_oop"] = self._controls_v2_int(
                value, args.get("numpoints", 3000), minimum=1)
        elif leaf == "radial_points":
            args["npt_rad"] = self._controls_v2_int(value, 500, minimum=1)
        elif leaf == "azim_points":
            args["npt_azim"] = self._controls_v2_int(value, 500, minimum=1)
        elif leaf == "radial_auto":
            self._controls_v2_set_range_auto(root, "radial", self._controls_v2_bool(value))
        elif leaf == "azim_auto":
            self._controls_v2_set_range_auto(root, "azimuth", self._controls_v2_bool(value))
        elif leaf == "radial_low":
            self._controls_v2_set_range_bound(root, "radial", "low", value)
        elif leaf == "radial_high":
            self._controls_v2_set_range_bound(root, "radial", "high", value)
        elif leaf == "azim_low":
            self._controls_v2_set_range_bound(root, "azimuth", "low", value)
        elif leaf == "azim_high":
            self._controls_v2_set_range_bound(root, "azimuth", "high", value)
        elif leaf == "apply_polarization":
            if self._controls_v2_bool(value):
                args["polarization_factor"] = self._controls_v2_float(
                    args.get("polarization_factor"),
                    DEFAULT_POLARIZATION_FACTOR)
            else:
                args["polarization_factor"] = None
            self._controls_v2_sync_advanced_parameter(path)
        elif leaf == "polarization_factor":
            args["polarization_factor"] = self._controls_v2_float(
                value, DEFAULT_POLARIZATION_FACTOR)
            self._controls_v2_sync_advanced_parameter(path)
        elif leaf in {"correctSolidAngle", "safe"}:
            args[leaf] = self._controls_v2_bool(value)
            self._controls_v2_sync_advanced_parameter(path)
        elif leaf in {"dummy", "delta_dummy", "chi_offset"}:
            args[leaf] = self._controls_v2_float(value, args.get(leaf, 0.0))
            self._controls_v2_sync_advanced_parameter(path)
        elif leaf == "method":
            args["method"] = str(value)
            self._controls_v2_sync_advanced_parameter(path)
        # Compatibility/display projection only.  The Controls-owned intent
        # remains authoritative, but existing reintegration and preview paths
        # expect edits to appear on the shared scan immediately.
        self._controls_v2_apply_snapshot_to_scan(
            self._controls_v2_native_int_snapshot()
        )
        return True

    @staticmethod
    def _controls_v2_frozen_native_snapshot(
        run_configuration: FrozenRunConfiguration,
    ) -> dict:
        gi = run_configuration.gi
        incidence = gi.scan_incidence_motor
        return {
            "bai_1d_args": run_configuration.bai_1d_args,
            "bai_2d_args": run_configuration.bai_2d_args,
            "gi": bool(gi.enabled),
            "gi_config": gi.scan_config(),
            "incidence_motor": incidence,
            "th_mtr": incidence,
            "sample_orientation": int(gi.sample_orientation),
            "tilt_angle": float(gi.tilt_angle),
        }

    def _controls_v2_apply_run_configuration_to_scan(
        self,
        run_configuration: FrozenRunConfiguration,
        scan=None,
    ) -> None:
        """Project one immutable run value into a mutable ``LiveScan``."""

        scan = scan if scan is not None else getattr(self, "scan", None)
        if scan is None:
            return
        self._controls_v2_apply_native_int_snapshot_to_scan(
            self._controls_v2_frozen_native_snapshot(run_configuration),
            scan,
        )
        scan.skip_2d = bool(run_configuration.skip_2d)
        scan.max_cores = int(run_configuration.max_cores)
        scan.apply_threshold = bool(
            run_configuration.threshold.apply_threshold)
        scan.threshold_min = run_configuration.threshold.threshold_min
        scan.threshold_max = run_configuration.threshold.threshold_max
        scan.mask_sentinel = bool(
            run_configuration.threshold.mask_saturation)
        scan.run_configuration_generation = int(
            run_configuration.generation)
        scan.run_configuration_fingerprint = run_configuration.fingerprint
        scan.run_configuration_provenance = (
            run_configuration.as_provenance()
        )

    def _controls_v2_poni_values_for(self, poni_file):
        """Parse a PONI file into its values dict for run-intent parity (item 11),
        or ``None`` for an empty / missing / unparseable path (the run then keeps
        its existing values and falls back to the wrangler's own calibration —
        staging is the fail-loud path)."""
        text = str(poni_file or "").strip()
        if not text or not os.path.exists(text):
            return None
        try:
            from xrd_tools.core.containers import PONI
            return PONI.from_poni_file(text).to_dict()
        except Exception:
            logger.debug("Controls V2 PONI re-parse failed for %s", text,
                         exc_info=True)
            return None

    def _prepare_controls_v2_run_configuration(
        self,
    ) -> FrozenRunConfiguration | None:
        """Freeze the next run before validation touches mutable GUI state."""

        if not self._controls_v2_enabled():
            return None
        # T-1 (§9.10 steps 1 + 3-stage): harvest the revisioned edit journal
        # (deferred + idle + correction) PLUS any in-progress panel form edit
        # (journaled edits win over the stale panel snapshot), pure-stage them
        # against a clone of the intent, then commit ONCE.  An invalid edit
        # (unsupported coercion / silent legacy setter no-op) aborts preparation
        # BEFORE any freeze/publish: typed, user-visible (surfaced by
        # imageWrangler.start()), structured event, full journal retained.  No
        # FrozenRunConfiguration is produced.  This replaces the i.3 sequential
        # fold: staging validates the COMPLETE delta before any carrier write,
        # so a later-field failure cannot half-apply an earlier field.
        _fold = self._controls_v2_fold_deferred_edits_into_intent()
        if _fold is not None:
            # §14.11.C: the fold's returned failure result is the SOLE authority —
            # no ambient recovery state, so a staging/harvest failure never
            # inherits an earlier attempt's recovery label.
            _recovery = _fold.recovery_failed_path
            run_config_debug_log(
                logger,
                "run_prepare_aborted_invalid_fold",
                widget=self,
                origin="controls_v2_prepare",
                phase=_fold.phase,
                failed_path=list(_fold.failed_path or ()),
                recovery_failed_path=list(_recovery) if _recovery else None,
                level="warning",
            )
            # §15.4-B.7: thread the engine's exact `reason` through to the Run
            # refusal so the operator (and the stack oracle) sees the specific
            # cause, not just the phase/path.
            _reason = (": " + _fold.reason) if _fold.reason else ""
            if _recovery:
                # §9.10 step 3-commit: a rollback failure is a distinct recovery
                # error — name the carrier that could not be restored.
                raise DeferredRunEditsPendingError(
                    "run configuration could not be applied and a carrier ("
                    + "/".join(str(seg) for seg in _recovery)
                    + ") could not be restored"
                    + _reason
                    + " — Run not started; check the control state before"
                    " retrying"
                )
            _where = (
                "/".join(str(seg) for seg in _fold.failed_path)
                if _fold.failed_path else (_fold.phase or "run configuration"))
            raise DeferredRunEditsPendingError(
                "deferred edit invalid ("
                + _where
                + ")"
                + _reason
                + " — Run not started; re-check the control and try again"
            )
        self._controls_v2_ensure_native_int_defaults()
        intent = self._controls_v2_ensure_run_intent()
        controls = getattr(self, "controls", None)
        wrangler = getattr(self, "wrangler", None)

        mode_getter = getattr(controls, "current_mode", None)
        try:
            intent.processing_mode = str(mode_getter())
        except Exception:
            intent.processing_mode = str(
                getattr(wrangler, "viewer_mode", "") or "Int 2D")
        write_mode = getattr(controls, "write_mode", None)
        try:
            intent.output_mode = str(write_mode())
        except Exception:
            active_mode = getattr(wrangler, "_active_write_mode", None)
            intent.output_mode = (
                str(active_mode()) if callable(active_mode) else "Append")
        intent.live_mode = bool(
            getattr(getattr(controls, "liveButton", None),
                    "isChecked", lambda: False)())
        intent.batch_mode = bool(
            getattr(getattr(controls, "batchButton", None),
                    "isChecked", lambda: False)())
        intent.max_cores = max(
            1,
            int(getattr(
                getattr(controls, "coresSpin", None),
                "value",
                lambda: 1,
            )()),
        )
        intent.source_spec = self._controls_v2_freeze_source_spec()
        intent.poni_file = str(
            self._controls_v2_param_value(("Signal", "poni_file"))
            or self._controls_v2_param_value(("Calibration", "poni_file"))
            or getattr(wrangler, "poni_file", "")
            or ""
        )
        # item 11 (idle-PONI parity): an idle PONI edit updates the file carrier
        # (via get_poni_dict()) but NOT the parsed intent values, so a stale
        # poni_values could reach the freeze while poni_file is new.  Re-parse the
        # current file so the frozen run carries values MATCHING poni_file.  A
        # successfully-parsed file overrides; an empty/unreadable path leaves the
        # existing values (the run is graceful — staging is the fail-loud path).
        _reparsed_poni_values = self._controls_v2_poni_values_for(intent.poni_file)
        if _reparsed_poni_values is not None:
            intent.poni_values = _reparsed_poni_values
        intent.mask_file = str(
            self._controls_v2_param_value(("Signal", "mask_file")) or "")
        intent.project_root = str(
            self._controls_v2_param_value(
                ("Project", "project_folder")) or "")
        intent.save_path = str(
            self._controls_v2_param_value(("Project", "h5_dir")) or "")
        intent.gi.mode_1d = str(
            intent.bai_1d_args.get("gi_mode_1d", "q_total"))
        intent.gi.mode_2d = str(
            intent.bai_2d_args.get("gi_mode_2d", "qip_qoop"))
        intent.run_options = {
            "xye_only": "XYE" in intent.processing_mode,
        }

        # Item 5 (R4B-8): resolve the effective GI motor ONCE at freeze from the
        # source's real motor list — passing None (not ()) when the dropdown is
        # not populated so an explicit motor is honored, never degraded to Manual.
        frozen = intent.freeze(
            gi_motor_choices=self._controls_v2_gi_motor_choices_for_freeze())
        self._pending_controls_v2_run_configuration = frozen
        if wrangler is not None:
            wrangler.run_configuration = frozen
            wrangler.source_spec = frozen.thaw_source_spec()
            thread = getattr(wrangler, "thread", None)
            if thread is not None:
                thread.run_configuration = frozen
        return frozen

    def _apply_controls_v2_run_state(
        self,
        run_configuration: FrozenRunConfiguration | None = None,
    ) -> dict:
        """Apply exactly one frozen Controls configuration to every run owner."""

        if run_configuration is None:
            run_configuration = getattr(
                self, "_pending_controls_v2_run_configuration", None)
        if not isinstance(run_configuration, FrozenRunConfiguration):
            run_configuration = self._prepare_controls_v2_run_configuration()
        if not isinstance(run_configuration, FrozenRunConfiguration):
            return {}

        self._controls_v2_apply_run_configuration_to_scan(run_configuration)
        self._push_threshold_to_wrangler(run_configuration)
        self._push_gi_to_wrangler(run_configuration)

        args = run_configuration.scan_args()
        wrangler = getattr(self, "wrangler", None)
        if wrangler is not None:
            wrangler.run_configuration = run_configuration
            wrangler.scan_args = args
            wrangler.gi = bool(run_configuration.gi.enabled)
            wrangler.incidence_motor = (
                run_configuration.gi.scan_incidence_motor)
            wrangler.sample_orientation = int(
                run_configuration.gi.sample_orientation)
            wrangler.tilt_angle = float(run_configuration.gi.tilt_angle)
            wrangler.apply_threshold = bool(
                run_configuration.threshold.apply_threshold)
            wrangler.threshold_min = (
                run_configuration.threshold.threshold_min)
            wrangler.threshold_max = (
                run_configuration.threshold.threshold_max)
            wrangler.mask_sentinel = bool(
                run_configuration.threshold.mask_saturation)
            thread = getattr(wrangler, "thread", None)
            if thread is not None:
                thread.run_configuration = run_configuration
                thread.scan_args = run_configuration.scan_args()
                thread.gi = bool(run_configuration.gi.enabled)
                thread.incidence_motor = (
                    run_configuration.gi.scan_incidence_motor)
                thread.sample_orientation = int(
                    run_configuration.gi.sample_orientation)
                thread.tilt_angle = float(
                    run_configuration.gi.tilt_angle)
                thread.apply_threshold = bool(
                    run_configuration.threshold.apply_threshold)
                thread.threshold_min = (
                    run_configuration.threshold.threshold_min)
                thread.threshold_max = (
                    run_configuration.threshold.threshold_max)
                thread.mask_sentinel = bool(
                    run_configuration.threshold.mask_saturation)
        return args

    def _controls_v2_config_save_veto(self) -> bool:
        """§15.12-A.4: pre-save veto for the explicit Config Save action.  Runs
        the checked commit; on a typed refusal it surfaces the message and
        returns True so ``defaultWidget.save_defaults`` writes NO file.  Returns
        False when Controls V2 is inactive, nothing is pending, or the pending
        edits commit cleanly (the save then serializes the committed state)."""
        if getattr(self, "_tearing_down", False):
            return False
        if not self._controls_v2_enabled():
            return False
        refusal = self._commit_controls_v2_pending_edits()
        if refusal is not None:
            self._controls_v2_report_pending_refusal(refusal, "config-save")
            return True
        return False

    def _controls_v2_config_save_veto_error(self) -> None:
        """§17.5: the Config-Save pre-save veto hook RAISED, so the save FAILED
        CLOSED (no file written).  Failure of the validation owner removes
        permission to save; report the refusal as a structured
        ``config_save_refused`` event (phase ``precondition``) and one
        user-visible status through the static-widget owner."""
        run_config_debug_log(
            logger,
            "config_save_refused",
            widget=self,
            origin="controls_v2_config_save",
            phase="precondition",
            reason="pre-save veto hook raised",
            level="warning",
        )
        try:
            self._controls_v2_status_message(
                "Config Save refused: the pending Controls edit could not be "
                "checked — no file was written.")
        except Exception:
            logger.debug("config-save veto error status failed", exc_info=True)

    def _controls_v2_int_session_state(self) -> dict:
        """Native Controls V2 Int state used for run/reintegrate plans.

        §15.12-A.3 / §15.8: this is a PURE serializer — it snapshots the already
        committed state and does NOT commit pending edits or show UI.  The action
        owners (Config Save veto, Run/reintegrate) run the checked commit FIRST;
        a passive serializer must never write a stale/invalid baseline nor decide
        policy."""

        self._controls_v2_ensure_native_int_defaults()
        snapshot = self._controls_v2_native_int_snapshot()

        cfg = self._controls_v2_threshold_config()
        threshold = {
            "apply_threshold": bool(cfg.apply_threshold),
            "threshold_min": cfg.threshold_min,
            "threshold_max": cfg.threshold_max,
            "mask_saturation": bool(cfg.mask_saturation),
        }

        return {
            "bai_1d_args": copy.deepcopy(snapshot["bai_1d_args"]),
            "bai_2d_args": copy.deepcopy(snapshot["bai_2d_args"]),
            "gi_config": copy.deepcopy(snapshot["gi_config"]),
            "gi": bool(snapshot["gi"]),
            "threshold_config": threshold,
        }

    def _apply_controls_v2_int_state(self, data: dict) -> bool:
        """Apply a canonical Controls V2 snapshot from session or config."""

        if not isinstance(data, dict):
            return False
        intent = self._controls_v2_ensure_run_intent()
        try:
            a1 = data.get("bai_1d_args")
            a2 = data.get("bai_2d_args")
            if isinstance(a1, dict):
                intent.bai_1d_args = copy.deepcopy(a1)
            if isinstance(a2, dict):
                intent.bai_2d_args = copy.deepcopy(a2)
            gic = data.get("gi_config")
            gic = copy.deepcopy(gic) if isinstance(gic, dict) else {}
            intent.gi = GIIntent(
                enabled=bool(data.get("gi", bool(gic))),
                incidence_motor=str(
                    gic.get("incidence_motor", "Manual") or "Manual"),
                th_val=self._controls_v2_float(gic.get("th_val", 0.1), 0.1),
                sample_orientation=self._controls_v2_int(
                    gic.get("sample_orientation", 4), 4),
                tilt_angle=self._controls_v2_float(
                    gic.get("tilt_angle", 0.0), 0.0),
                mode_1d=str(
                    gic.get(
                        "gi_mode_1d",
                        intent.bai_1d_args.get("gi_mode_1d", "q_total"),
                    )
                ),
                mode_2d=str(
                    gic.get(
                        "gi_mode_2d",
                        intent.bai_2d_args.get("gi_mode_2d", "qip_qoop"),
                    )
                ),
            )
            self._controls_v2_gi_selection_explicit = bool(gic)
        except Exception:
            logger.debug("Controls V2 native Int scan restore failed",
                         exc_info=True)
            return False

        self._controls_v2_hydrate_advanced_from_scan()

        threshold = data.get("threshold_config")
        if isinstance(threshold, dict):
            self._controls_v2_threshold_state = {
                "apply_threshold": bool(threshold.get("apply_threshold", False)),
                "threshold_min": self._controls_v2_float(
                    threshold.get("threshold_min", 0.0), 0.0),
                "threshold_max": self._controls_v2_float(
                    threshold.get("threshold_max", 0.0), 0.0),
                "mask_saturation": bool(threshold.get("mask_saturation", True)),
            }
            intent.threshold = ThresholdIntent.from_mapping(
                self._controls_v2_threshold_state
            )
            cfg = self._controls_v2_threshold_config()
        self._controls_v2_apply_snapshot_to_scan(
            self._controls_v2_native_int_snapshot()
        )
        scan = getattr(self, "scan", None)
        if scan is not None:
            cfg = self._controls_v2_threshold_config()
            scan.apply_threshold = cfg.apply_threshold
            scan.threshold_min = cfg.threshold_min
            scan.threshold_max = cfg.threshold_max
            scan.mask_sentinel = cfg.mask_saturation

        self._refresh_controls_v2_profile(immediate=True)
        return True

    def _restore_controls_v2_int_session_state(self) -> None:
        """Restore the native Controls V2 Int blob after the legacy fallback."""

        try:
            from xdart.utils.session import load_session
            data = (load_session() or {}).get("controls_v2_int")
        except Exception:
            logger.debug("Controls V2 native Int session load failed",
                         exc_info=True)
            return
        self._apply_controls_v2_int_state(data)

    def _controls_v2_native_reduction_plan(
        self,
        *,
        include_threshold: bool = True,
        integrate_1d: bool = True,
        integrate_2d: bool = True,
        commit_pending: bool = True,
    ):
        """Build the native Controls V2 reduction plan used by run/reintegrate.

        §15.12-A.3 / §15.8 / §17.8: this is a PURE builder — it snapshots the
        committed state, never commits pending edits, and NEVER projects into the
        live scan.  ``commit_pending`` is retained only for call-site
        compatibility; the action owners (Run / reintegrate / Config Save) run the
        checked commit BEFORE building a plan."""

        self._controls_v2_ensure_native_int_defaults()

        # §17.8: build from an ASIDE copy of the scan with the GI config projected
        # onto it — reading the projected GI fields WITHOUT mutating the live
        # scan.gi / scan.gi_config / scan.incidence_motor (the exact-object test).
        live_scan = getattr(self, "scan", None)
        scan = live_scan
        if live_scan is not None:
            scan = types.SimpleNamespace(
                skip_2d=getattr(live_scan, "skip_2d", False),
                detector_shape=getattr(live_scan, "detector_shape", None),
                frames=getattr(live_scan, "frames", None),
                bai_1d_args=getattr(live_scan, "bai_1d_args", {}),
                bai_2d_args=getattr(live_scan, "bai_2d_args", {}),
                _cached_fiber_integrator_angle=getattr(
                    live_scan, "_cached_fiber_integrator_angle", None),
                global_mask=getattr(live_scan, "global_mask", None),
                gi=getattr(live_scan, "gi", False),
                gi_config=getattr(live_scan, "gi_config", {}),
                incidence_motor=getattr(live_scan, "incidence_motor", None),
                th_mtr=getattr(live_scan, "th_mtr", None),
                sample_orientation=getattr(live_scan, "sample_orientation", 4),
                tilt_angle=getattr(live_scan, "tilt_angle", 0.0),
            )
            self._controls_v2_apply_gi_config_to_scan(scan=scan)

        threshold_min = None
        threshold_max = None
        mask_saturation = False
        if include_threshold:
            cfg = self._controls_v2_threshold_config()
            if cfg.apply_threshold:
                threshold_min = cfg.threshold_min
                threshold_max = cfg.threshold_max
            mask_saturation = bool(cfg.mask_saturation)

        return build_native_int_reduction_plan_from_scan(
            scan,
            integrate_1d=integrate_1d,
            integrate_2d=integrate_2d,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            mask_saturation=mask_saturation,
        )

    @staticmethod
    def _controls_v2_native_run_plan_enabled() -> bool:
        value = os.environ.get("XDART_CONTROLS_V2_NATIVE_RUN_PLAN", "1")
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def _controls_v2_native_run_plan_builder(
        self,
        snapshot: dict,
    ):
        snapshot = copy.deepcopy(snapshot or {})
        snapshot_key = self._controls_v2_native_int_snapshot_key(snapshot)
        apply_snapshot = type(self)._controls_v2_apply_native_int_snapshot_to_scan

        def _prepare_scan(scan):
            apply_snapshot(snapshot, scan)

        def _builder(
            scan,
            *,
            integrate_1d: bool = True,
            integrate_2d: bool = True,
        ):
            _prepare_scan(scan)
            return build_native_int_reduction_plan_from_scan(
                scan,
                integrate_1d=integrate_1d,
                integrate_2d=integrate_2d,
            )

        _builder.prepare_scan = _prepare_scan
        _builder.plan_cache_key = ("controls_v2_native_int", snapshot_key)
        return _builder

    def _configure_controls_v2_native_run_plan(
        self,
        *,
        commit_pending: bool = False,
    ) -> None:
        builder = None
        if (
            self._controls_v2_enabled()
            and self._controls_v2_native_run_plan_enabled()
        ):
            self._apply_controls_v2_native_int_state(
                commit_pending=commit_pending,
                push_integrator=True,
            )
            builder = self._controls_v2_native_run_plan_builder(
                self._controls_v2_native_int_snapshot()
            )
        owners = (
            getattr(getattr(self, "wrangler", None), "thread", None),
            getattr(getattr(self, "integratorTree", None), "integrator_thread", None),
        )
        for owner in owners:
            cache = getattr(owner, "_plan_cache", None)
            if cache is not None and hasattr(cache, "plan_builder"):
                cache.plan_builder = builder

    # ------------------------------------------------------------------
    # Edit journal (§9.10 step 1) — ONE revisioned owner for every edit.
    # Every accepted user action (deferred during a run, entered while idle,
    # harvested as an uncommitted panel form edit, or entered as a correction
    # after a failed Start) records ONE monotonic revision keyed by field path.
    # There is exactly one winner per path: the highest revision.  A newer idle
    # edit therefore supersedes an older deferred value by construction, rather
    # than leaving a stale value in a separate queue — chronology IS the
    # revision, not the order separate containers happen to be harvested at Start.
    # ------------------------------------------------------------------

    def _controls_v2_edit_journal_dict(self) -> dict:
        journal = getattr(self, "_controls_v2_edit_journal", None)
        if not isinstance(journal, dict):
            journal = {}
            self._controls_v2_edit_journal = journal
        return journal

    def _controls_v2_next_edit_revision(self) -> int:
        rev = int(getattr(self, "_controls_v2_edit_revision_counter", 0)) + 1
        self._controls_v2_edit_revision_counter = rev
        return rev

    def _controls_v2_record_edit(self, path, value, origin: str) -> int:
        """Record one revisioned edit into the journal (LWW per path).

        §13.9: stored as an immutable, deep-copied :class:`JournalEntry` so a
        caller cannot mutate a list/dict value in place without a new revision."""
        path = tuple(path)
        rev = self._controls_v2_next_edit_revision()
        self._controls_v2_edit_journal_dict()[path] = JournalEntry(
            value, rev, origin)
        return rev

    def _controls_v2_edit_journal_clear(self) -> None:
        self._controls_v2_edit_journal = {}

    def _controls_v2_clear_consumed_revisions(self, consumed) -> None:
        """Remove ONLY the exact ``{path: revision}`` entries this transaction
        consumed.  A path whose journal revision advanced after the harvest (a
        newer concurrent edit) is retained, never erased (§12.9)."""
        journal = self._controls_v2_edit_journal_dict()
        for path, revision in consumed.items():
            entry = journal.get(path)
            if entry is not None and entry.get("revision") == revision:
                del journal[path]

    def _controls_v2_journal_winners(self):
        """Winning ``(path, value)`` per path, ordered by ascending revision."""
        journal = self._controls_v2_edit_journal_dict()
        ordered = sorted(journal.items(), key=lambda kv: kv[1]["revision"])
        return [(path, entry["value"]) for path, entry in ordered]

    @property
    def _controls_v2_deferred_field_edits(self):
        """Read-only view of the deferred-origin journal entries as a list.

        The journal (:meth:`_controls_v2_edit_journal_dict`) is the single owner;
        this preserves the O-1a-i.3 accessor.  A newer idle/form correction for
        the same path OVERWRITES the deferred entry (higher revision), so it no
        longer appears here — that IS the global last-write-wins rule.
        """
        journal = self._controls_v2_edit_journal_dict()
        ordered = sorted(
            (item for item in journal.items()
             if item[1].get("origin") == "deferred"),
            key=lambda kv: kv[1]["revision"],
        )
        return [(path, entry["value"]) for path, entry in ordered]

    def _controls_v2_form_value_differs(self, path, form_value, committed_value) -> bool:
        """Whether a harvested form value is a REAL change vs the committed state.

        A stale post-run panel value equals the committed state (not a change),
        so it is not harvested and cannot clobber a deferred edit.  An
        un-coercible draft is treated as a differing edit so it is journaled and
        then refused by staging (never silently dropped, §12.4)."""
        if committed_value is None:
            return True
        try:
            coerced = coerce_control_edit_value(committed_value, form_value)
        except Exception:
            return True
        return coerced != committed_value

    def _controls_v2_collect_pending_edits(self):
        """Return the transaction's winning edits, ordered SOLELY by revision.

        Every user action — deferred, idle, focused draft, correction, or a
        reintegration form commit — carries an immutable revision recorded at
        ACTION time; there is no 'journal beats form' rule (§12.4).  A still
        uncommitted focused draft that differs from the committed state and is not
        already journaled at its true value is recorded here with a fresh revision
        so it participates by revision (never rank-below-journal).  A form-harvest
        failure is a typed refusal, not fail-open (§12.4 test 9).
        """
        journal = self._controls_v2_edit_journal_dict()
        panel = getattr(self, "controls_v2", None)
        committed = self._controls_v2_field_values(overlay_pending=False)
        # §13.11 owner 5: the transaction consumes the DIRTY REVISIONED ENTRIES
        # (the journal — every draft/idle/deferred/correction is recorded at
        # action time via draftChanged).  From the visible form we take AT MOST
        # ONE focused-editor flush — the editor the user is mid-editing, whose
        # final text may post-date its last draft signal.  The full-form snapshot
        # is NO LONGER imported: a non-focused row is only ever set to a committed
        # value (signal-blocked rebuild guard), so it is never dirty-unjournaled.
        focused = getattr(panel, "focused_form_edit", None)
        if callable(focused):
            try:
                edit = focused()
            except Exception:
                edit = None
            if edit is not None:
                self._controls_v2_flush_form_edit(edit, journal, committed)
                return self._controls_v2_journal_winners()
        # No editor is focused (a programmatic ``setText`` or a harvest-time
        # snapshot): flush the dirty, not-yet-journaled visible edits.  Still
        # dirty-only — a value equal to committed or already journaled is never
        # re-recorded.  A harvest failure is a typed refusal (§12.4), not
        # fail-open.
        get_edits = getattr(panel, "current_form_edits", None)
        if callable(get_edits):
            try:
                form_edits = get_edits()
            except Exception as exc:
                raise ControlsTransactionError(
                    None, "form edit harvest failed", None) from exc
            for edit in form_edits:
                self._controls_v2_flush_form_edit(edit, journal, committed)
        return self._controls_v2_journal_winners()

    def _controls_v2_flush_form_edit(self, edit, journal, committed) -> None:
        """Journal ONE visible-form edit iff it is a real change (differs from the
        committed value) and is not already journaled at that exact value
        (§12.4).  This is the dirty-only flush primitive — it never records a
        stale committed value nor a duplicate of an action-time draft."""
        path = tuple(edit.path)
        if not self._controls_v2_form_value_differs(
                path, edit.value, committed.get(path)):
            return
        existing = journal.get(path)
        if existing is not None and existing.get("value") == edit.value:
            return  # already journaled at action time (draft/idle)
        self._controls_v2_record_edit(path, edit.value, origin="form")

    @staticmethod
    def _source_token_from_spec(spec):
        """A hashable identity of a SourceSpec (§12.5).

        Excludes the directory request-generation so re-selecting the same
        directory keeps its motor observation; different sources (roots/kinds)
        produce different tokens.  Shared by the LIVE source token and the
        CANDIDATE source fingerprint so both are directly comparable."""
        if spec is None:
            return None
        from xrd_tools.sources.selection import DirectorySourceSpec
        if isinstance(spec, DirectorySourceSpec):
            return (
                "directory",
                str(spec.root),
                bool(spec.recursive),
                spec.name_filter,
                tuple(str(s) for s in spec.suffixes),
            )
        kind = getattr(spec, "kind", "")
        return (
            "source",
            str(getattr(spec, "uri", "")),
            str(getattr(kind, "value", kind)),
        )

    def _controls_v2_source_token(self):
        """A hashable identity of the CURRENTLY-configured source selection."""
        try:
            spec = self._controls_v2_freeze_source_spec()
        except Exception:
            return None
        return self._source_token_from_spec(spec)

    def _controls_v2_candidate_source_spec(self, cand):
        """Derive the CANDIDATE SourceSpec from a source-SELECTION reduce.

        §13.11 candidate 4-5: read the candidate's reduced source values,
        falling back to the live params, so the freeze and the motor observation
        see the NEW source, never the stale one.  Mirrors
        :meth:`_controls_v2_freeze_source_spec` (Image Directory / Series /
        Single Image); other source types have no typed candidate spec."""
        def cval(path):
            p = tuple(path)
            if p in cand.legacy_projection:
                return cand.legacy_projection[p]
            # §13.11 candidate 3: fall back to the COMMITTED snapshot captured at
            # stage entry, NEVER a live Qt Parameter read.
            return cand.committed_legacy.get(p, "")

        source_type = str(cval(("Signal", "inp_type")) or "")
        if source_type == "Image Directory":
            root_text = str(cval(("Signal", "img_dir")) or "").strip()
            ext = str(cval(("Signal", "img_ext")) or "").lstrip(".").lower()
            # Mirror _controls_v2_freeze_source_spec: only container directories
            # (h5/hdf5/nxs) freeze a typed DirectorySourceSpec, so the candidate
            # and live fingerprints agree for a same-value edit (§14.11.D.2).
            if not root_text or ext not in {"h5", "hdf5", "nxs"}:
                return None
            recursive = bool(cval(("Signal", "include_subdir")))
            name_filter = str(cval(("Signal", "Filter")) or "") or None
            if ext == "h5":
                suffixes = ("_master.h5",)
            elif ext == "hdf5":
                suffixes = ("_master.hdf5", "_master.h5")
            else:
                suffixes = (".nxs",)
            from xrd_tools.sources import DirectorySourceSpec
            return DirectorySourceSpec(
                root=Path(root_text).expanduser(),
                recursive=recursive,
                suffixes=suffixes,
                name_filter=name_filter,
                generation=0,
            )
        selected = str(cval(("Signal", "File")) or "").strip()
        selected_ext = Path(selected).suffix.lstrip(".").lower()
        if (source_type == "Image Series" and selected
                and selected_ext not in {"h5", "hdf5", "nxs"}):
            from xrd_tools.sources import image_series_spec
            return image_series_spec(selected)
        if source_type == "Single Image" and selected:
            from xrd_tools.core.scan import SourceKind, SourceSpec
            return SourceSpec(selected, SourceKind.IMAGE_FILE)
        return None

    def _controls_v2_record_gi_motor_observation(
            self, motors, for_token=_UNSET, state=None) -> None:
        """Store the GI motor observation for the CURRENT source (§12.5/§13.7).

        Called from the targeted-metadata hydration handler, so a motor list is
        trusted only when it is tied to the source it was observed from.  When the
        caller knows which source the motors came from it passes ``for_token``; a
        delayed result whose ``for_token`` no longer matches the current source is
        IGNORED (a stale A signal cannot seed B's freeze).  A later source edit
        also changes the token, so the CAPTURE side never returns a previous
        source's motors regardless.

        ``state`` is the AUTHORITATIVE knowledge from a structured hydration
        (§13.7): ``UNKNOWN`` records "not inspected" (never downgrading a proven
        observation for the same source); a legacy ``None`` state infers
        KNOWN_NONEMPTY / KNOWN_EMPTY from the list as before."""
        token = self._controls_v2_source_token()
        if for_token is not _UNSET and for_token != token:
            return
        real = [
            str(m) for m in (motors or [])
            if str(m) and str(m) != "Manual"
            and not any(x in str(m).lower() for x in ("roi", "pd"))
        ]
        if state == GIMotorObservation.UNKNOWN:
            # §13.7: an UNKNOWN result means "not inspected" — never overwrite a
            # PROVEN observation for the same source with UNKNOWN.
            stored = getattr(self, "_controls_v2_gi_motor_observation", None)
            if (isinstance(stored, GIMotorObservation) and stored.matches(token)
                    and stored.state != GIMotorObservation.UNKNOWN):
                return
            self._controls_v2_gi_motor_observation = GIMotorObservation(
                GIMotorObservation.UNKNOWN, (), token)
            return
        if state is None:
            # Legacy list-only inference (unchanged): a bare recorder call with an
            # empty list is KNOWN_EMPTY; a non-empty list is KNOWN_NONEMPTY.
            state = (
                GIMotorObservation.KNOWN_NONEMPTY if real
                else GIMotorObservation.KNOWN_EMPTY)
        self._controls_v2_gi_motor_observation = GIMotorObservation(
            state, real, token)

    def _controls_v2_capture_gi_motor_observation(self):
        """Return the motor observation for the CURRENT source, or UNKNOWN.

        A stored observation is trusted only when its source token still matches
        the current selection; otherwise the choices are UNKNOWN (freeze then
        preserves the operator's explicit motor rather than degrading it —
        §12.5), until targeted hydration records an observation for this source."""
        token = self._controls_v2_source_token()
        stored = getattr(self, "_controls_v2_gi_motor_observation", None)
        if isinstance(stored, GIMotorObservation) and stored.matches(token):
            return stored
        return GIMotorObservation(GIMotorObservation.UNKNOWN, (), token)

    def _controls_v2_gi_motor_choices_for_freeze(self):
        """GI motor choices for the freeze, source-qualified (§12.5).

        Maps the current source's motor observation to the ``resolve_gi_motor``
        contract: ``UNKNOWN -> None`` (preserve the explicit motor),
        ``KNOWN_EMPTY -> ()`` (resolve to Manual), ``KNOWN_NONEMPTY ->
        tuple(motors)``.  Choices are never derived from the persistent combo, so
        a source A→B edit cannot freeze B with A's motor."""
        return self._controls_v2_capture_gi_motor_observation().choices_for_freeze()

    # ------------------------------------------------------------------
    # Pure stage (§9.10 step 3-stage) — validate every edit against a CLONE.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pure candidate reducer (§12.2/§12.3) — NO self-install, NO production
    # setters.  Every reducer reads and writes ONLY the ControlsStageCandidate.
    # ------------------------------------------------------------------

    @staticmethod
    def _controls_v2_is_finite(value) -> bool:
        return value == value and value not in (float("inf"), float("-inf"))

    def _controls_v2_reduce_int(self, value, *, minimum=None) -> int:
        """Strict int coercion for the pure reducer — RAISES on invalid input
        (unlike the permissive live _controls_v2_int which clamps to a default).

        §13.4/§13.11-candidate 1: only a GENUINELY integral value is accepted —
        ``"4.5"`` is refused, not silently truncated to ``4``."""
        try:
            fval = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"expected an integer, got {value!r}")
        if fval != int(fval):
            raise ValueError(f"expected a whole number, got {value!r}")
        out = int(fval)
        if minimum is not None and out < int(minimum):
            raise ValueError(f"value {out} below minimum {minimum}")
        return out

    def _controls_v2_reduce_float(self, value) -> float:
        """Strict finite-float coercion for the pure reducer — RAISES on
        invalid or non-finite input."""
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"expected a number, got {value!r}")
        if not self._controls_v2_is_finite(out):
            raise ValueError("value must be finite")
        return out

    def _controls_v2_candidate_default_range(self, cand, root, axis):
        a1, a2 = cand.bai
        gi = cand.gi_enabled
        if root == "Int1D":
            if gi:
                mode = a1.get("gi_mode_1d", "q_total")
                if axis == "radial":
                    return (-5.0, 5.0) if mode == "exit_angle" else (
                        (-10.0, 10.0) if mode in {"q_ip", "q_oop"} else (0.0, 5.0))
                if mode in {"q_ip", "q_oop"}:
                    return (0.0, 5.0)
                if mode == "exit_angle":
                    return (0.0, 90.0)
                return (-180.0, 180.0)
            if axis == "radial":
                return (0.0, 90.0) if a1.get("unit") == "2th_deg" else (0.0, 5.0)
            return (-180.0, 180.0)
        if gi:
            mode = a2.get("gi_mode_2d", "qip_qoop")
            if axis == "radial":
                return (-5.0, 5.0) if mode == "exit_angles" else (
                    (-10.0, 10.0) if mode == "qip_qoop" else (0.0, 5.0))
            if mode == "qip_qoop":
                return (0.0, 5.0)
            if mode == "exit_angles":
                return (0.0, 90.0)
            return (-180.0, 180.0)
        if axis == "radial":
            return (0.0, 90.0) if a2.get("unit") == "2th_deg" else (0.0, 5.0)
        return (-180.0, 180.0)

    def _controls_v2_candidate_range_value(self, cand, root, axis):
        a1, a2 = cand.bai
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        value = args.get(key)
        return value if value is not None else (
            self._controls_v2_candidate_default_range(cand, root, axis))

    def _controls_v2_candidate_npts_oop_visible(self, cand) -> bool:
        if not cand.gi_enabled:
            return False
        a1, _ = cand.bai
        return (
            a1.get("gi_mode_1d", "q_total") != "q_total"
            or a1.get("azimuth_range") is not None)

    def _controls_v2_reduce_axis(self, cand, root, value) -> None:
        a1, a2 = cand.bai
        text = str(value or "")
        if cand.gi_enabled:
            # §13.4: an unknown GI axis is a path-qualified typed refusal, never a
            # silent default to q_total / qip_qoop.
            if root == "Int1D":
                try:
                    mode = GI_MODES_1D[GI_LABELS_1D.index(text)]
                except ValueError:
                    raise ControlsTransactionError(
                        (root, "axis"), "unknown GI axis", value)
                old_mode = a1.get("gi_mode_1d")
                a1["gi_mode_1d"] = mode
                if mode != old_mode:
                    _drop_output_axis_ranges(a1)
                if self._controls_v2_candidate_npts_oop_visible(cand):
                    a1.setdefault("npt_oop", int(a1.get("numpoints", 3000)))
            else:
                try:
                    mode = GI_MODES_2D[GI_LABELS_2D.index(text)]
                except ValueError:
                    raise ControlsTransactionError(
                        (root, "axis"), "unknown GI axis", value)
                old_mode = a2.get("gi_mode_2d")
                a2["gi_mode_2d"] = mode
                if mode != old_mode:
                    _drop_output_axis_ranges(a2)
            a1["unit"] = "q_A^-1"
            a2["unit"] = "q_A^-1"
            # The candidate intent carries gi_mode_* on both the bai args (the
            # authoritative reduction key) and the GI sub-intent; keep them in
            # sync WITHOUT touching any display scan (§12.2 test 6).
            cand.intent.gi.mode_1d = str(a1.get("gi_mode_1d", "q_total"))
            cand.intent.gi.mode_2d = str(a2.get("gi_mode_2d", "qip_qoop"))
            return
        if root == "Int1D":
            old_unit = a1.get("unit")
            a1["unit"] = self._controls_v2_unit_code(text, dim="1d", strict=True)
            if a1["unit"] != old_unit:
                a1.pop("radial_range", None)
        else:
            old_unit = a2.get("unit")
            a2["unit"] = self._controls_v2_unit_code(text, dim="2d", strict=True)
            if a2["unit"] != old_unit:
                a2.pop("radial_range", None)

    def _controls_v2_reduce_range_auto(self, cand, root, axis, auto) -> None:
        a1, a2 = cand.bai
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        value = None if auto else self._controls_v2_candidate_range_value(
            cand, root, axis)
        args[key] = value
        if root == "Int2D" and cand.gi_enabled:
            alt = "x_range" if axis == "radial" else "y_range"
            if value is None:
                args.pop(alt, None)
            else:
                args[alt] = value
        if root == "Int1D" and axis == "azimuth" and (
                self._controls_v2_candidate_npts_oop_visible(cand)):
            args.setdefault("npt_oop", int(args.get("numpoints", 3000)))

    def _controls_v2_reduce_range_bound(self, cand, root, axis, bound, value) -> None:
        a1, a2 = cand.bai
        args = a1 if root == "Int1D" else a2
        key = "radial_range" if axis == "radial" else "azimuth_range"
        low, high = self._controls_v2_candidate_range_value(cand, root, axis)
        number = self._controls_v2_reduce_float(value)
        if bound == "low":
            low = number
        else:
            high = number
        args[key] = (float(low), float(high))
        if root == "Int2D" and cand.gi_enabled:
            args["x_range" if axis == "radial" else "y_range"] = args[key]
        if root == "Int1D" and axis == "azimuth" and (
                self._controls_v2_candidate_npts_oop_visible(cand)):
            args.setdefault("npt_oop", int(args.get("numpoints", 3000)))

    def _controls_v2_reduce_gi_field(self, cand, leaf, value) -> None:
        from .gi_motor_defaults import pick_default_gi_motor
        a1, a2 = cand.bai
        gi_intent = cand.intent.gi
        obs = cand.gi_motor_observation
        real_choices = list(obs.motors) if obs is not None else []
        gi = bool(gi_intent.enabled)
        motor = str(gi_intent.incidence_motor or "Manual")
        th_val = float(gi_intent.th_val)
        sample_orientation = int(gi_intent.sample_orientation)
        tilt_angle = float(gi_intent.tilt_angle)
        mode_1d = str(gi_intent.mode_1d or a1.get("gi_mode_1d", "q_total"))
        mode_2d = str(gi_intent.mode_2d or a2.get("gi_mode_2d", "qip_qoop"))
        if leaf == "Grazing":
            gi = self._controls_v2_bool(value)
            # Enabling GI with a leftover Manual and no explicit pick adopts the
            # source's default motor (from the source-qualified observation, NOT
            # the persistent combo — §12.5).
            if gi and motor == "Manual" and not cand.gi_selection_explicit:
                picked = pick_default_gi_motor(real_choices)
                if picked != "Manual":
                    motor = picked
        elif leaf == "th_motor":
            motor = str(value)
            cand.gi_selection_explicit = True
        elif leaf == "th_val":
            th_val = self._controls_v2_reduce_float(value)
        elif leaf == "sample_orientation":
            sample_orientation = self._controls_v2_reduce_int(value)
        elif leaf == "tilt_angle":
            tilt_angle = self._controls_v2_reduce_float(value)
        cand.intent.gi = GIIntent(
            enabled=bool(gi),
            incidence_motor=str(motor or "Manual"),
            th_val=float(th_val),
            sample_orientation=int(sample_orientation),
            tilt_angle=float(tilt_angle),
            mode_1d=mode_1d,
            mode_2d=mode_2d,
        )
        if cand.intent.gi.enabled:
            a1.setdefault("gi_mode_1d", "q_total")
            a2.setdefault("gi_mode_2d", "qip_qoop")
            a1["unit"] = "q_A^-1"
            a2["unit"] = "q_A^-1"

    def _controls_v2_reduce_threshold_field(self, cand, path, value) -> None:
        state = dict(cand.threshold_state or {})
        state.setdefault("apply_threshold", False)
        state.setdefault("threshold_min", 0.0)
        state.setdefault("threshold_max", 0.0)
        state.setdefault("mask_saturation", True)
        if path == ("Mask", "Threshold"):
            state["apply_threshold"] = self._controls_v2_bool(value)
        elif path == ("Mask", "min"):
            state["threshold_min"] = self._controls_v2_reduce_float(value)
        elif path == ("Mask", "max"):
            state["threshold_max"] = self._controls_v2_reduce_float(value)
        elif path == ("MaskSat", "mask_sentinel"):
            state["mask_saturation"] = self._controls_v2_bool(value)
        if path in {("Mask", "min"), ("Mask", "max")} and (
                state["threshold_min"] != 0.0 or state["threshold_max"] != 0.0):
            state["apply_threshold"] = True
        cand.threshold_state = state
        cand.intent.threshold = ThresholdIntent.from_mapping(state)

    def _controls_v2_reduce_int_field(self, cand, path, value) -> None:
        root = path[0]
        leaf = path[1] if len(path) > 1 else ""
        if root == "GI":
            self._controls_v2_reduce_gi_field(cand, leaf, value)
            return
        if root in {"Mask", "MaskSat"}:
            self._controls_v2_reduce_threshold_field(cand, path, value)
            return
        if root not in {"Int1D", "Int2D"}:
            return
        a1, a2 = cand.bai
        args = a1 if root == "Int1D" else a2
        if leaf == "unit":
            args["unit"] = self._controls_v2_unit_code(
                value, dim="1d" if root == "Int1D" else "2d", strict=True)
        elif leaf == "axis":
            self._controls_v2_reduce_axis(cand, root, value)
        elif leaf == "points":
            args["numpoints"] = self._controls_v2_reduce_int(value, minimum=1)
            if self._controls_v2_candidate_npts_oop_visible(cand):
                args.setdefault("npt_oop", args["numpoints"])
        elif leaf == "points_oop":
            args["npt_oop"] = self._controls_v2_reduce_int(value, minimum=1)
        elif leaf == "radial_points":
            args["npt_rad"] = self._controls_v2_reduce_int(value, minimum=1)
        elif leaf == "azim_points":
            args["npt_azim"] = self._controls_v2_reduce_int(value, minimum=1)
        elif leaf == "radial_auto":
            self._controls_v2_reduce_range_auto(
                cand, root, "radial", self._controls_v2_bool(value))
        elif leaf == "azim_auto":
            self._controls_v2_reduce_range_auto(
                cand, root, "azimuth", self._controls_v2_bool(value))
        elif leaf == "radial_low":
            self._controls_v2_reduce_range_bound(cand, root, "radial", "low", value)
        elif leaf == "radial_high":
            self._controls_v2_reduce_range_bound(cand, root, "radial", "high", value)
        elif leaf == "azim_low":
            self._controls_v2_reduce_range_bound(cand, root, "azimuth", "low", value)
        elif leaf == "azim_high":
            self._controls_v2_reduce_range_bound(cand, root, "azimuth", "high", value)
        elif leaf == "apply_polarization":
            if self._controls_v2_bool(value):
                current = args.get("polarization_factor")
                args["polarization_factor"] = (
                    DEFAULT_POLARIZATION_FACTOR if current is None
                    else self._controls_v2_reduce_float(current))
            else:
                args["polarization_factor"] = None
        elif leaf == "polarization_factor":
            args["polarization_factor"] = self._controls_v2_reduce_float(value)
        elif leaf in {"correctSolidAngle", "safe"}:
            args[leaf] = self._controls_v2_bool(value)
        elif leaf in {"dummy", "delta_dummy", "chi_offset"}:
            args[leaf] = self._controls_v2_reduce_float(value)
        elif leaf == "method":
            method = str(value)
            if method not in _CONTROLS_V2_SUPPORTED_METHODS:
                raise ControlsTransactionError(
                    path, "unsupported integration method", value)
            args["method"] = method

    def _controls_v2_reduce_edit(self, cand, path, value) -> None:
        """Reduce ONE edit onto the candidate.  Raises ``ControlsTransactionError``
        for an unsupported path or invalid coercion (§12.3)."""
        path = tuple(path)
        if path and path[0] in {"Signal", "Source"}:
            cand.source_touched = True
        if path in self._CONTROLS_V2_SOURCE_SELECTION_PATHS:
            cand.source_selection_touched = True
            # §12.5 / finding 4: a source-SELECTION change invalidates the
            # candidate's GI motor observation — the new source's motors are
            # UNKNOWN until targeted hydration reports them, so a [source A→B +
            # GI-enable] transaction cannot default-pick A's motor into B.
            cand.gi_motor_observation = GIMotorObservation(
                GIMotorObservation.UNKNOWN, (), None)
        if path in NATIVE_CONTROL_PATHS:
            if path == ("Source", "energy_preference"):
                text = str(value or "").strip().lower()
                aliases = {
                    "poni": "poni", "poni file": "poni",
                    "metadata": "metadata", "meta": "metadata",
                }
                if text not in aliases:
                    # §13.4: an unknown source-energy preference is a typed
                    # refusal, never a silent default to 'poni'.
                    raise ControlsTransactionError(
                        path, "unknown source-energy preference", value)
                cand.source_energy_preference = aliases[text]
            return
        if path in INTEGRATOR_BACKED_CONTROL_PATHS:
            try:
                self._controls_v2_reduce_int_field(cand, path, value)
            except ControlsTransactionError:
                raise
            except Exception as exc:
                err = ControlsTransactionError(
                    path, "invalid value for this field", value)
                err.__cause__ = exc
                raise err
            return
        # Legacy-backed field: coerce WITHOUT setValue and WITHOUT resolving a
        # live Qt parameter (§15.12-A.6).  The committed baseline comes from the
        # stage-entry snapshot; an unsupported path (no captured backing value) is
        # a typed refusal, not a silent skip (§12.3).
        if path not in cand.committed_legacy:
            raise ControlsTransactionError(path, "unsupported control path", value)
        try:
            coerced = coerce_control_edit_value(
                cand.committed_legacy[path], value)
        except Exception as exc:
            err = ControlsTransactionError(
                path, "value cannot be coerced to the field type", value)
            err.__cause__ = exc
            raise err
        cand.legacy_projection[path] = coerced
        if path in self._CONTROLS_V2_PONI_PATHS:
            # §12.6/§13.8: PONI is a COMPOUND carrier.  The signal-blocked legacy
            # projection updates only the path, so parse it HERE into the
            # candidate object/values — otherwise wrangler setup would copy the
            # stale in-memory calibration to the thread while the frozen identity
            # names the new file.
            self._controls_v2_reduce_poni(cand, path, coerced)

    def _controls_v2_reduce_poni(self, cand, path, poni_path) -> None:
        """Parse a poni_file edit into the compound PONI candidate (§12.6/§13.8).

        Empty/absent OR a path that does not exist -> cleared calibration
        (``None`` object/values) — matching ``get_poni_dict``'s BUG-1 clear so a
        Start guard trips rather than running the previous scan's PONI.  A path
        that EXISTS but cannot be parsed is a path-qualified
        :class:`ControlsTransactionError` (staging is fail-loud: never freeze a
        run whose named calibration is unreadable)."""
        from xrd_tools.core.containers import PONI

        cand.poni_touched = True
        text = str(poni_path or "").strip()
        cand.intent.poni_file = text
        if not text or not os.path.exists(text):
            cand.poni_object = None
            cand.poni_values = None
            cand.intent.poni_values = None
            return
        try:
            poni_object = PONI.from_poni_file(text)
        except Exception as exc:
            err = ControlsTransactionError(
                tuple(path),
                "PONI calibration file could not be parsed", text)
            err.__cause__ = exc
            raise err
        cand.poni_object = poni_object
        cand.poni_values = poni_object.to_dict()
        cand.intent.poni_values = dict(cand.poni_values)

    def _controls_v2_validate_candidate(self, cand) -> None:
        """Validate the COMPLETE candidate before returning it (§12.3): finite
        numerics, sample-orientation range, threshold min<=max, and every
        cross-field invariant the frozen dataclasses enforce — raising a typed
        :class:`ControlsTransactionError` on the CLONE, before any install and
        before freeze.  Freeze is NOT a substitute for transaction validation."""
        # Range shape: a radial/azimuth range whose low exceeds its high is a
        # cross-field contradiction (§12.9 P3b) — a typed refusal, not a freeze
        # error or a silently-swapped range.
        a1, a2 = cand.bai
        for args, root in ((a1, "Int1D"), (a2, "Int2D")):
            for key, axis in (
                    ("radial_range", "radial"), ("azimuth_range", "azimuth")):
                rng = args.get(key)
                if (isinstance(rng, (tuple, list)) and len(rng) == 2
                        and rng[0] is not None and rng[1] is not None
                        and float(rng[0]) > float(rng[1])):
                    raise ControlsTransactionError(
                        (root, axis + "_range"),
                        "range low cannot exceed high", rng)
        try:
            # Freezing the CLONE runs the GI/threshold/output __post_init__
            # validators (sample_orientation 1-8, finite th_val/tilt/threshold,
            # threshold_min<=max, output_mode) against candidate data only.
            cand.intent.freeze(
                gi_motor_choices=(
                    cand.gi_motor_observation.choices_for_freeze()
                    if cand.gi_motor_observation is not None else None))
        except ControlsTransactionError:
            raise
        except Exception as exc:
            raise ControlsTransactionError(
                None, f"invalid run configuration: {exc}", None) from exc

    def _controls_v2_source_selection_changed(self, cand) -> bool:
        """Whether the candidate's EFFECTIVE source selection differs from live.

        §14.11.D.2: reconciliation need is a before/after comparison of the
        source-selection values, NOT mere path membership in the journal — an
        edit that coerces back to its live value (same-value / net-zero) changes
        nothing and must reconcile nothing."""
        for path in self._CONTROLS_V2_SOURCE_SELECTION_PATHS:
            if path not in cand.legacy_projection:
                continue
            # §14.11.D.2: compare against the COMMITTED snapshot captured at stage
            # entry, NEVER a live Qt Parameter read.
            if cand.legacy_projection[path] != cand.committed_legacy.get(path):
                return True
        return False

    def _controls_v2_committed_legacy_values(self) -> dict:
        """§15.12-A.6 / §13.11 candidate 3 / §14.11.D: snapshot the COMMITTED
        value of EVERY bound legacy param ONCE at stage entry.  All pure reducers
        (candidate source-spec derivation, effective-selection comparison, AND the
        generic legacy coercion) read this stable snapshot instead of a live Qt
        Parameter, so the stage never resolves a Qt parameter mid-transaction and
        cannot depend on a mutable carrier (§15.7)."""
        paths = set(self._CONTROLS_V2_SOURCE_SELECTION_PATHS)
        paths.update(self._CONTROLS_V2_PONI_PATHS)
        paths.update(self._controls_v2_field_paths())
        values = {}
        for path in paths:
            param = self._controls_v2_param(path)
            if param is None:
                continue
            try:
                values[path] = param.value()
            except Exception:
                continue
        return values

    def _controls_v2_new_stage_candidate(self):
        """§15.12-A.6: build a fresh PURE stage candidate — a frozen snapshot of
        the committed inputs (intent clone, threshold/selection state, GI
        observation, and the committed value of every bound legacy path).  The
        reducers read ONLY this candidate + the edits; they resolve no Qt
        parameter, scan, wrangler, or thread.  Shared by the full transaction
        stage and the per-field idle-edit validation so both see identical
        inputs."""
        live_intent = self._controls_v2_ensure_run_intent()
        saved_threshold = getattr(self, "_controls_v2_threshold_state", None)
        return ControlsStageCandidate(
            intent=copy.deepcopy(live_intent),
            threshold_state=(
                copy.deepcopy(saved_threshold)
                if isinstance(saved_threshold, dict) else None),
            gi_selection_explicit=bool(
                getattr(self, "_controls_v2_gi_selection_explicit", False)),
            source_energy_preference=getattr(
                self, "_controls_v2_source_energy_preference", "poni"),
            gi_motor_observation=self._controls_v2_capture_gi_motor_observation(),
            committed_legacy=self._controls_v2_committed_legacy_values(),
        )

    def stage_controls_transaction(self, edits):
        """Reduce *edits* onto a PURE candidate — no ``self`` state, no scan, no
        Qt param, no wrangler/thread is touched (§12.2).  The complete candidate
        is validated (§12.3) before a :class:`StagedControlsTransaction` is
        returned; any unsupported path, invalid coercion, non-finite number, or
        cross-field contradiction (e.g. threshold min>max) is a typed
        :class:`ControlsTransactionError` returned with the offending path and
        the original exception as its cause.  On refusal the live production
        state and the winning journal are unchanged.
        """
        cand = self._controls_v2_new_stage_candidate()
        try:
            for path, value in edits:
                self._controls_v2_reduce_edit(cand, path, value)
            if cand.source_selection_touched:
                # §13.11 candidate 4-5: derive the NEW candidate source and its
                # fingerprint/epoch NOW; keep the stored motor observation only
                # if it matches this exact source, else UNKNOWN — so a source
                # A→B edit followed by GI-enable never adopts A's motor.
                cand.candidate_source_spec = (
                    self._controls_v2_candidate_source_spec(cand))
                cand.source_fingerprint = self._source_token_from_spec(
                    cand.candidate_source_spec)
                # §14.11.D.2: reconcile ONLY when the EFFECTIVE selection changes.
                # A same-value / net-zero source edit (every edited selection path
                # coerces back to its live value) performs zero reconciliation.
                if not self._controls_v2_source_selection_changed(cand):
                    cand.source_selection_touched = False
                stored = getattr(
                    self, "_controls_v2_gi_motor_observation", None)
                if (isinstance(stored, GIMotorObservation)
                        and stored.matches(cand.source_fingerprint)):
                    cand.gi_motor_observation = stored
                else:
                    cand.gi_motor_observation = GIMotorObservation(
                        GIMotorObservation.UNKNOWN, (), cand.source_fingerprint)
            self._controls_v2_validate_candidate(cand)
        except ControlsTransactionError as err:
            return err
        return StagedControlsTransaction(
            staged_intent=cand.intent,
            legacy_projection=cand.legacy_projection,
            threshold_state=(
                copy.deepcopy(cand.threshold_state)
                if isinstance(cand.threshold_state, dict) else None),
            gi_selection_explicit=cand.gi_selection_explicit,
            source_touched=cand.source_touched,
            source_selection_touched=cand.source_selection_touched,
            source_energy_preference=cand.source_energy_preference,
            gi_motor_observation=cand.gi_motor_observation,
            poni_touched=cand.poni_touched,
            poni_object=cand.poni_object,
            poni_values=cand.poni_values,
            candidate_source_spec=cand.candidate_source_spec,
            source_fingerprint=cand.source_fingerprint,
        )

    # ------------------------------------------------------------------
    # Commit (T-1: install with per-write readback detection; T-2 adds the
    # snapshot + reverse-order verified rollback + transactional source owner).
    # ------------------------------------------------------------------

    #: Intent fields copied by the in-place install/snapshot (identity kept).
    _CONTROLS_V2_INTENT_FIELDS = (
        "source_spec", "processing_mode", "output_mode", "live_mode",
        "batch_mode", "max_cores", "bai_1d_args", "bai_2d_args", "gi",
        "threshold", "poni_file", "poni_values", "mask_file", "project_root",
        "save_path", "run_options",
    )

    #: Legacy paths that change the SOURCE SELECTION (the SourceSpec).  Only these
    #: trigger source reconciliation; mask/PONI/series-average/BG edits do NOT
    #: (§12.7 / §12 test 13 — an unrelated edit or an unchanged Start reconciles
    #: nothing and requests no directory poll).
    _CONTROLS_V2_SOURCE_SELECTION_PATHS = frozenset({
        ("Signal", "inp_type"),
        ("Signal", "File"),
        ("Signal", "img_dir"),
        ("Signal", "include_subdir"),
        ("Signal", "img_ext"),
        ("Signal", "Filter"),
        ("Signal", "meta_dir"),
        ("Signal", "meta_ext"),
        ("NeXus File", "nexus_file"),
        ("NeXus File", "entry"),
    })

    #: Legacy paths that carry a PONI calibration file (§12.6/§13.8).  A reduce
    #: on either one parses the file into the compound PONI candidate; the image
    #: wrangler uses ``("Signal", "poni_file")``, the NeXus wrangler
    #: ``("Calibration", "poni_file")``.
    _CONTROLS_V2_PONI_PATHS = frozenset({
        ("Signal", "poni_file"),
        ("Calibration", "poni_file"),
    })

    @classmethod
    def _controls_v2_install_intent_values(cls, live, staged) -> None:
        """Copy the staged intent's VALUES onto *live* in place (identity kept).

        The monotonic ``generation`` is deliberately NOT copied: the live intent
        owns the counter and the staged clone started from the same value, so a
        later ``freeze`` advances the live intent exactly once."""
        for field_name in cls._CONTROLS_V2_INTENT_FIELDS:
            setattr(live, field_name, getattr(staged, field_name))

    @classmethod
    def _controls_v2_snapshot_intent_values(cls, live) -> dict:
        """Shallow field-reference snapshot of *live* for atomic rollback.

        Install REPLACES the field references (never mutates the prior objects),
        so restoring the snapshot references puts the intent back exactly."""
        return {name: getattr(live, name) for name in cls._CONTROLS_V2_INTENT_FIELDS}

    @classmethod
    def _controls_v2_restore_intent_values(cls, live, snapshot) -> None:
        for name, value in snapshot.items():
            setattr(live, name, value)

    def _controls_v2_write_legacy_carrier(self, params, path, value) -> None:
        """Write one legacy carrier signal-blocked (no readback).  The mirror
        swallows a broken wrangler setter, so the caller MUST verify via
        :meth:`_controls_v2_legacy_readback_ok` — the readback, not the setter's
        return, is the authority (§14.11.A.2)."""
        self._mirror_wrangler_parameter_values(params, ((tuple(path), value),))

    def _controls_v2_force_restore_legacy_carrier(self, path, value) -> None:
        """Force a parameter back to *value* DIRECTLY (bypassing any wrangler
        setter override), signal-blocked.

        The Parameter value is the carrier of record; a carrier whose forward
        setter proved unreliable (wrote the wrong value / raised) must still be
        restorable, so rollback resets its Parameter directly rather than through
        the same broken channel (§14.11.A.3)."""
        param = self._controls_v2_param(tuple(path))
        if param is None:
            return
        prev = param.blockSignals(True)
        try:
            type(param).setValue(param, value)
        except Exception:
            logger.debug("force-restore of %s failed", path, exc_info=True)
        finally:
            try:
                param.blockSignals(prev)
            except Exception:
                pass

    def _controls_v2_legacy_readback_ok(self, path, expected) -> bool:
        """Whether *path*'s parameter reads back as *expected*.

        A carrier that DISAPPEARED between preflight and readback (param is
        ``None``) is a failure, never a silent success (§14.11.A.6).  A getter
        that raises is likewise a failure, contained inside the boundary."""
        param = self._controls_v2_param(tuple(path))
        if param is None:
            return False
        try:
            return param.value() == expected
        except Exception:
            return False

    def _controls_v2_rollback_legacy_all(self, params, applied, forced_paths=()):
        """Restore ALL attempted ``(path, prior)`` carriers in REVERSE order.

        §14.11.A.4-5: NEVER stop at the first restore failure — continue
        restoring older carriers and collect EVERY path that could not be
        restored (verified by readback).  A ``forced_paths`` carrier (its forward
        setter proved unreliable) is force-restored directly rather than through
        the same broken wrangler channel."""
        failed = []
        if params is None:
            return failed
        forced = {tuple(p) for p in forced_paths}
        for path, prior in reversed(applied):
            path = tuple(path)
            try:
                if path in forced:
                    self._controls_v2_force_restore_legacy_carrier(path, prior)
                else:
                    self._controls_v2_write_legacy_carrier(params, path, prior)
                ok = self._controls_v2_legacy_readback_ok(path, prior)
            except Exception:
                logger.debug("rollback of %s raised", path, exc_info=True)
                ok = False
            if not ok:
                failed.append(path)
        return failed

    #: Display-scan fields the native-int projection writes (kept in sync with
    #: ``_controls_v2_apply_native_int_snapshot_to_scan``).  §14.11.B.1: the
    #: display projection is a rollback carrier — a mid-projection throw must
    #: restore these under the same scan lock, not leave the scan half-projected.
    _CONTROLS_V2_DISPLAY_SCAN_FIELDS = (
        "bai_1d_args", "bai_2d_args", "gi", "gi_config",
        "incidence_motor", "th_mtr", "sample_orientation", "tilt_angle",
    )

    def _controls_v2_snapshot_display_scan(self, scan):
        """Deep snapshot of the projected display-scan fields for rollback.

        §15.4-B.3: taken UNDER ``scan_lock`` so no background writer tears a field
        mid-copy, and with NO alias fallback — an alias of the live object is not a
        recoverable snapshot.  A field that cannot be deep-copied is an UNCOPYABLE
        rollback state: this RAISES so the caller fails the transaction at
        PREFLIGHT (zero writes), never silently aliasing the live object."""
        if scan is None:
            return None

        def _snap():
            snap = {}
            for name in self._CONTROLS_V2_DISPLAY_SCAN_FIELDS:
                if not hasattr(scan, name):
                    continue
                snap[name] = copy.deepcopy(getattr(scan, name))
            return snap

        lock = getattr(scan, "scan_lock", None)
        if lock is None:
            return _snap()
        with lock:
            return _snap()

    def _controls_v2_restore_display_scan(self, scan, snapshot) -> None:
        """Restore the projected display-scan fields under the scan lock.

        Dict fields are restored by CLEAR+UPDATE so their object IDENTITY is
        preserved (matching the forward projection, whose held references the
        reintegration/display paths depend on); scalars are reassigned."""
        if scan is None or not snapshot:
            return
        lock = getattr(scan, "scan_lock", None)

        def _restore():
            for name, value in snapshot.items():
                if isinstance(value, dict):
                    staticWidget._controls_v2_replace_dict_in_place(
                        scan, name, value)
                else:
                    try:
                        setattr(scan, name, copy.deepcopy(value))
                    except Exception:
                        setattr(scan, name, value)

        if lock is None:
            _restore()
        else:
            with lock:
                _restore()

    def commit_controls_transaction(self, staged) -> ControlsCommitResult:
        """Install a validated :class:`StagedControlsTransaction` atomically.

        §9.10 step 3-commit / step 4 + §14.11.A/B + §15.4 Correction B.  ONE
        checked rollback boundary:

        * PREFLIGHT (ZERO writes) — a non-empty legacy projection with NO
          parameter root REFUSES (B.1, never skip-and-succeed); resolve a STABLE
          carrier record (path, handle identity, prior, coerced expected) for
          every legacy carrier; a missing carrier, getter failure, OR coercion
          failure is a typed preflight failure (B.1/B.2); and EVERY rollback
          snapshot — display scan under ``scan_lock`` with no alias fallback,
          intent, threshold/GI flags, compound PONI carriers, energy/probe caches
          + observation BEFORE any invalidation — is captured before the first
          write (B.3/B.6).  An uncopyable snapshot is a preflight failure.
        * FORWARD — legacy carriers apply using ONLY the preflighted handle;
          replacement is detected by object identity, never re-resolved by path
          (B.4).  Each carrier is pushed onto the rollback stack BEFORE its setter;
          setter AND readback are wrapped so nothing escapes after the first write.
        * RECOVERY — every failure runs :meth:`_controls_v2_recover_all`, a
          collector that attempts EVERY carrier class independently and collects
          all un-restorable paths in reverse-application order (B.5), never
          aborting on the first restore exception.
        * SOURCE — reconciled through ONE owner only when the source SELECTION
          changed; the recovery collector performs the verified second `_sync` and
          restores the energy/probe caches + observation (B.2 — the mutate-then-
          restore/two-`_sync` shape is retained per §14; Correction C reshapes it).

        Returns a typed :class:`ControlsCommitResult` (phase / failed_path /
        reason / all recovery failures) — never an ambient side effect.
        """
        wrangler = getattr(self, "wrangler", None)
        params = getattr(wrangler, "parameters", None)
        thread = getattr(wrangler, "thread", None)
        scan = getattr(self, "scan", None)
        prior_energy_pref = getattr(
            self, "_controls_v2_source_energy_preference", "poni")

        # ===================== PREFLIGHT — ZERO writes ======================
        # B.1: a non-empty legacy projection with NO parameter root cannot be
        # applied — refuse (the old code skipped-and-returned success).
        projection = list(staged.legacy_projection.items())
        if projection and params is None:
            first_path = tuple(projection[0][0])
            return ControlsCommitResult(
                False, failed_path=first_path,
                reason="parameter root missing at preflight",
                phase="preflight")

        # B.1/B.2: resolve a STABLE carrier record for EVERY legacy carrier.  A
        # missing carrier, a getter failure, OR a coercion failure (no longer
        # swallowed into expected=value) is a typed preflight failure, zero writes.
        carriers = []  # (path, param, value, prior, expected)
        for path, value in projection:
            path = tuple(path)
            param = self._controls_v2_param(path)
            if param is None:
                return ControlsCommitResult(
                    False, failed_path=path,
                    reason="legacy carrier missing at preflight",
                    phase="preflight")
            try:
                prior = param.value()
            except Exception:
                logger.debug("preflight getter failed for %s", path,
                             exc_info=True)
                return ControlsCommitResult(
                    False, failed_path=path,
                    reason="legacy carrier getter failed at preflight",
                    phase="preflight")
            try:
                expected = coerce_control_edit_value(param.value(), value)
            except Exception:
                logger.debug("preflight coercion failed for %s", path,
                             exc_info=True)
                return ControlsCommitResult(
                    False, failed_path=path,
                    reason="legacy carrier coercion failed at preflight",
                    phase="preflight")
            carriers.append((path, param, value, prior, expected))

        # B.3: display snapshot BEFORE the first write, under scan_lock; an
        # uncopyable field RAISES -> typed preflight failure (never an alias).
        try:
            display_scan_snapshot = self._controls_v2_snapshot_display_scan(scan)
        except Exception:
            logger.debug("controls commit display snapshot failed", exc_info=True)
            return ControlsCommitResult(
                False, failed_path=None,
                reason="display scan snapshot failed (uncopyable state)",
                phase="preflight")
        # B.3: intent snapshot also before the first write.
        try:
            live = self._controls_v2_ensure_run_intent()
            intent_snapshot = self._controls_v2_snapshot_intent_values(live)
        except Exception:
            logger.debug("controls commit intent snapshot failed", exc_info=True)
            return ControlsCommitResult(
                False, failed_path=None, reason="intent snapshot failed",
                phase="preflight")

        # B.3/B.6: every other rollback carrier, captured BEFORE the first write —
        # threshold/GI flags, compound PONI carriers, and the energy/probe caches +
        # directory observation captured BEFORE any preference invalidation (the
        # old code nulled the energy cache first and then "restored" None).
        prior_gi_explicit = getattr(
            self, "_controls_v2_gi_selection_explicit", False)
        prior_threshold_state = getattr(
            self, "_controls_v2_threshold_state", None)
        prior_wrangler_poni = getattr(wrangler, "poni", None)
        prior_wrangler_poni_file = getattr(wrangler, "poni_file", None)
        prior_thread_poni = getattr(thread, "poni", None)
        prior_observation = getattr(
            self, "_controls_v2_directory_observation", None)
        prior_energy_cache = getattr(
            self, "_controls_v2_source_energy_cache", None)
        prior_probe_cache = getattr(
            self, "_controls_v2_metadata_probe_cache", None)

        ctx = {
            "live": live, "intent_snapshot": intent_snapshot,
            "prior_gi_explicit": prior_gi_explicit,
            "prior_threshold_state": prior_threshold_state,
            "prior_energy_pref": prior_energy_pref,
            "scan": scan, "display_scan_snapshot": display_scan_snapshot,
            "wrangler": wrangler, "thread": thread, "staged": staged,
            "prior_wrangler_poni": prior_wrangler_poni,
            "prior_wrangler_poni_file": prior_wrangler_poni_file,
            "prior_thread_poni": prior_thread_poni,
            "prior_observation": prior_observation,
            "prior_energy_cache": prior_energy_cache,
            "prior_probe_cache": prior_probe_cache,
            "params": params,
        }

        # ===================== FORWARD — legacy carriers ====================
        applied = []  # (path, prior) in application order
        for path, param, value, prior, expected in carriers:
            applied.append((path, prior))  # push BEFORE the setter (A.3)
            # B.4: the carrier handle must be the SAME object preflight resolved;
            # a replacement (path now maps to a different Parameter) is a contained
            # failure, never a silent re-resolve that inspects a different object.
            if self._controls_v2_param(path) is not param:
                recovery = self._controls_v2_recover_all(
                    ctx, applied, forced_paths=(path,))
                return ControlsCommitResult(
                    False, failed_path=path,
                    reason="legacy carrier handle replaced before apply",
                    recovery_failed_paths=recovery, phase="legacy_apply")
            try:
                self._controls_v2_write_legacy_carrier(params, path, value)
                ok = (self._controls_v2_param(path) is param
                      and self._controls_v2_legacy_readback_ok(path, expected))
            except Exception:
                logger.debug("legacy carrier apply/readback raised for %s", path,
                             exc_info=True)
                ok = False
            if not ok:
                recovery = self._controls_v2_recover_all(
                    ctx, applied, forced_paths=(path,))
                return ControlsCommitResult(
                    False, failed_path=path,
                    reason="legacy carrier readback failed",
                    recovery_failed_paths=recovery, phase="legacy_apply")

        # ===================== INSTALL — intent + display + PONI ============
        try:
            self._controls_v2_install_intent_values(live, staged.staged_intent)
            self._controls_v2_gi_selection_explicit = staged.gi_selection_explicit
            if staged.threshold_state is not None:
                self._controls_v2_threshold_state = staged.threshold_state
            self._controls_v2_source_energy_preference = (
                staged.source_energy_preference)
            self._controls_v2_apply_snapshot_to_scan(
                self._controls_v2_native_int_snapshot())
            # §12.6/§13.8: the compound PONI carrier installs as ONE value — path
            # (legacy projection, above) + wrangler object/path + thread object +
            # intent values (already copied by _controls_v2_install_intent_values).
            if staged.poni_touched:
                if wrangler is not None:
                    wrangler.poni = staged.poni_object
                    wrangler.poni_file = str(staged.staged_intent.poni_file or "")
                if thread is not None:
                    thread.poni = staged.poni_object
        except Exception:
            logger.debug("controls commit intent install failed", exc_info=True)
            recovery = self._controls_v2_recover_all(ctx, applied)
            return ControlsCommitResult(
                False, failed_path=None, reason="intent install failed",
                recovery_failed_paths=recovery, phase="install")

        # An energy-preference change invalidates only the energy cache (§12.7).
        if staged.source_energy_preference != prior_energy_pref:
            self._controls_v2_source_energy_cache = None

        # ===================== SOURCE reconciliation ========================
        # §14.11.B.3 DECISION (recorded, not silently kept): the full "build a
        # candidate source/index ASIDE, then ONE swap" is DEFERRED; the verified
        # mutate-then-restore shape is retained deliberately because the §14
        # acceptance oracle asserts the source owner is reconciled EXACTLY TWICE
        # (forward + the recovery collector's restore = ``len(calls) == 2``), and a
        # true aside-build needs a new aside/swap API on the DirectoryIndexSession
        # (a deeper refactor Codex reshapes in Correction C).  B.4/B.5: on failure
        # the recovery collector performs the SECOND `_sync` and restores the
        # caches/observation; an unrestorable source owner is a DISTINCT recovery
        # failure naming the ("Source",) carrier, never log-and-discard.
        if staged.source_selection_touched:
            try:
                self._controls_v2_source_energy_cache = None
                self._controls_v2_metadata_probe_cache = None
                self._sync_controls_v2_source_index()  # forward = call 1
            except Exception:
                logger.debug("controls commit source reconcile failed",
                             exc_info=True)
                recovery = self._controls_v2_recover_all(
                    ctx, applied, source_reconciled=True)
                return ControlsCommitResult(
                    False, failed_path=("Source",),
                    reason="source reconciliation failed",
                    recovery_failed_paths=recovery, phase="source_reconcile")
        return ControlsCommitResult(True)

    def _controls_v2_recover_all(self, ctx, applied, forced_paths=(),
                                 source_reconciled=False):
        """§15.4-B.5 RECOVERY COLLECTOR — restore EVERY carrier class through a
        PER-CLASS ``try``/``except`` in reverse-APPLICATION order, verifying each
        readback/identity, collecting ALL un-restorable paths, and NEVER aborting
        on the first restore exception.  Returns the deterministic reverse-order
        failure list for :class:`ControlsCommitResult.recovery_failed_paths`.

        Order (reverse of application): source owner -> caches/observation ->
        compound PONI -> display scan -> intent/self-state -> legacy carriers."""
        failures = []
        # 1. source owner (applied LAST => restored FIRST).  §14.11.B.4: the
        #    verified restore is the SECOND `_sync` (keeps `len(calls) == 2`).
        if source_reconciled:
            try:
                self._sync_controls_v2_source_index()
            except Exception:
                logger.debug("controls source restore reconcile failed",
                             exc_info=True)
                failures.append(("Source",))
        # Restore observation + caches from the preflight snapshot AFTER any
        # restore-sync, so they are authoritative even if it mutated them again.
        try:
            self._controls_v2_directory_observation = ctx["prior_observation"]
            self._controls_v2_source_energy_cache = ctx["prior_energy_cache"]
            self._controls_v2_metadata_probe_cache = ctx["prior_probe_cache"]
        except Exception:
            logger.debug("controls source cache restore failed", exc_info=True)
            failures.append(("Source", "cache"))
        # 2. compound PONI carriers (wrangler object/path + thread object).
        staged = ctx["staged"]
        if staged.poni_touched:
            try:
                w = ctx["wrangler"]
                th = ctx["thread"]
                if w is not None:
                    w.poni = ctx["prior_wrangler_poni"]
                    w.poni_file = ctx["prior_wrangler_poni_file"]
                if th is not None:
                    th.poni = ctx["prior_thread_poni"]
            except Exception:
                logger.debug("controls PONI carrier restore failed",
                             exc_info=True)
                failures.append(("Signal", "poni_file"))
        # 3. display-scan projection (dict identity preserved under scan_lock).
        try:
            self._controls_v2_restore_display_scan(
                ctx["scan"], ctx["display_scan_snapshot"])
        except Exception:
            logger.debug("controls display scan restore failed", exc_info=True)
            failures.append(("Display",))
        # 4. intent field references + self-state (energy pref/GI-explicit/thresh).
        try:
            self._controls_v2_restore_intent_values(
                ctx["live"], ctx["intent_snapshot"])
            self._controls_v2_gi_selection_explicit = ctx["prior_gi_explicit"]
            self._controls_v2_threshold_state = ctx["prior_threshold_state"]
            self._controls_v2_source_energy_preference = ctx["prior_energy_pref"]
        except Exception:
            logger.debug("controls intent restore failed", exc_info=True)
            failures.append(("Intent",))
        # 5. legacy carriers (reverse projection order; its own verified collector).
        failures.extend(self._controls_v2_rollback_legacy_all(
            ctx["params"], applied, forced_paths=forced_paths))
        return failures

    def _controls_v2_defer_field_edit(self, path, value) -> None:
        """Record a run-active Controls edit into the journal for the NEXT run.

        The edit is PURE data (never applied to a carrier while the run is
        active); it is staged + committed exactly once at the next Start.  The
        operator is told the change is QUEUED and applies to the next run.
        """
        path = tuple(path)
        revision = self._controls_v2_record_edit(path, value, origin="deferred")
        try:
            self.wrangler.showLabel.emit(
                "Change queued — it applies to the next run.")
        except Exception:
            logger.debug("could not surface deferred-edit notice",
                         exc_info=True)
        # §12.4 / C3: include the path, revision, AND origin so ordering failures
        # are diagnosable.
        run_config_debug_log(
            logger,
            "controls_field_deferred_active_run",
            widget=self,
            origin="controls_v2_ui",
            field_path=list(path),
            field_value=value,
            revision=revision,
            edit_origin="deferred",
            deferred_pending=len(self._controls_v2_deferred_field_edits),
        )

    def _controls_v2_fold_deferred_edits_into_intent(self):
        """Stage + commit the revisioned edit journal at the next Start.

        Harvests the journal winners plus any in-progress panel form edit
        (journaled edits win over the stale panel snapshot), pure-stages them
        against a clone of the intent, and — only if staging AND commit succeed —
        installs the result, reconciling any source edit ONCE.  Returns ``None``
        on success (journal cleared), else a typed :class:`ControlsCommitResult`
        (§14.11.C) carrying phase / failed_path / reason / recovery failures.
        The RETURNED value is the sole diagnostic authority — there is no ambient
        recovery state — so a later independent harvest/stage failure cannot
        inherit an earlier recovery label.  On any failure the full journal is
        retained and the caller aborts BEFORE any freeze/publish.
        """
        try:
            edits = self._controls_v2_collect_pending_edits()
        except ControlsTransactionError as harvest_err:
            # §12.4 test 9: a form/draft collection failure REFUSES preparation,
            # it does not continue fail-open.
            run_config_debug_log(
                logger,
                "controls_deferred_fold_invalid",
                widget=self,
                origin="controls_v2_prepare_fold",
                failed_path=list(harvest_err.path or ()),
                reason=harvest_err.reason,
                deferred_pending=len(self._controls_v2_journal_winners()),
                level="warning",
            )
            return ControlsCommitResult(
                False, failed_path=(harvest_err.path or ("Controls",)),
                reason=harvest_err.reason, phase="harvest")
        return self._controls_v2_stage_commit_edits(edits)

    def _controls_v2_commit_journal_winners(self):
        """§17.4 idle-commit owner: stage + checked-commit the CURRENT journal
        winners WITHOUT re-harvesting the panel form.

        The idle ``fieldValueChanged`` signal already recorded its edit as a
        revision, so the transaction input is the journal itself; re-importing a
        full-form snapshot (as the Run/action fold does for an unflushed focused
        editor) would let a STALE non-focused row clobber the just-recorded idle
        value.  Returns ``None`` on success or the typed
        :class:`ControlsCommitResult` failure (journal retained).

        §17.4: the commit batch is over the per-field-VALID winners.  A draft that
        the strict per-field reducer REFUSES (e.g. a stale GI-mode axis label left
        after GI was disabled) was already refused when entered and stays
        JOURNAL-OWNED for the panel overlay — but it must NOT block the commit of
        other valid edits.  A per-field-VALID edit that only forms a CROSS-FIELD
        contradiction (threshold min>max) stays in the batch and pends as a group,
        so the resolving edit still commits both at once."""
        winners = [
            (path, value)
            for path, value in self._controls_v2_journal_winners()
            if self._controls_v2_validate_idle_edit(path, value) is None
        ]
        return self._controls_v2_stage_commit_edits(winners)

    def _controls_v2_stage_commit_edits(self, edits):
        """Stage + checked-commit a RESOLVED winner set through the atomic engine
        (§14.11.C).  Shared by the Run/action fold (form-harvested winners) and the
        idle-commit owner (journal winners).  Returns ``None`` on success (only the
        exact consumed revisions cleared) or the typed
        :class:`ControlsCommitResult` failure (full journal retained; the RETURNED
        value is the sole diagnostic authority)."""
        if not edits:
            return None
        # Snapshot the exact revisions THIS transaction consumes (only the staged
        # edits), so success clears ONLY them — a newer concurrent edit (higher
        # revision) AND an excluded per-field-invalid draft both survive (§12.9).
        journal = self._controls_v2_edit_journal_dict()
        consumed = {}
        for path, _ in edits:
            entry = journal.get(tuple(path))
            if entry is not None:
                consumed[tuple(path)] = entry["revision"]
        staged = self.stage_controls_transaction(edits)
        if isinstance(staged, ControlsTransactionError):
            run_config_debug_log(
                logger,
                "controls_deferred_fold_invalid",
                widget=self,
                origin="controls_v2_prepare_fold",
                failed_path=list(staged.path or ()),
                reason=staged.reason,
                deferred_pending=len(self._controls_v2_journal_winners()),
                level="warning",
            )
            return ControlsCommitResult(
                False, failed_path=(staged.path or ("Controls",)),
                reason=staged.reason, phase="stage")
        result = self.commit_controls_transaction(staged)
        if not result.ok:
            # A rollback failure is a DISTINCT recovery error: retain the journal,
            # refuse Run, and name every carrier that could not be restored
            # (§14.11.A); the caller surfaces the returned result visibly.
            run_config_debug_log(
                logger,
                "controls_deferred_fold_invalid",
                widget=self,
                origin="controls_v2_prepare_fold",
                phase=result.phase,
                failed_path=list(result.failed_path or ()),
                recovery_failed_path=(
                    [list(p) for p in result.recovery_failed_paths]
                    if result.recovery_failed_paths else None),
                reason=result.reason,
                deferred_pending=len(self._controls_v2_journal_winners()),
                level="warning",
            )
            return result
        # Clear ONLY the exact revisions this transaction consumed, so a newer
        # concurrent edit recorded after the harvest is never erased (§12.9).
        folded = [list(p) for p, _ in edits]
        self._controls_v2_clear_consumed_revisions(consumed)
        bump_run_config_debug_generation(self, "config")
        run_config_debug_log(
            logger,
            "controls_deferred_folded",
            widget=self,
            origin="controls_v2_prepare_fold",
            folded=folded,
        )
        return None

    def _on_controls_v2_field_changed(self, path, value) -> None:
        path = tuple(path)
        if self._controls_v2_run_active():
            # R4B-12: never silently drop a run-active edit — record it as pure
            # next-run data (deferred origin) and tell the operator.
            self._controls_v2_defer_field_edit(path, value)
            return
        # §15.12-A.1 / §15.3 / §17.4: an idle edit records a revision and is
        # classified by the PURE per-field reducer BEFORE any carrier is touched.
        # A per-field-INVALID value (e.g. a non-integral "4.5" the old permissive
        # setter would have clamped to 4) is a typed refusal — journal retained,
        # ONE stable message, ZERO carrier mutation (§15.12-A.2).  A per-field-VALID
        # value is installed ONLY through the checked atomic engine (never the
        # retired second setter `_apply_controls_v2_field_value`): the COMPLETE
        # journal winner set is staged and committed as one, so a value that is
        # individually valid but part of a still-incomplete CROSS-FIELD
        # contradiction (e.g. threshold min entered before max) stays JOURNAL-
        # OWNED and panel-overlaid — never installed, never serialized — until a
        # later related edit completes a valid winner set (§17.4).
        _idle_rev = self._controls_v2_record_edit(path, value, origin="idle")
        refusal = self._controls_v2_validate_idle_edit(path, value)
        if refusal is not None:
            run_config_debug_log(
                logger,
                "controls_field_refused",
                widget=self,
                origin="controls_v2_ui",
                field_path=path,
                field_value=value,
                revision=_idle_rev,
                phase=refusal.phase,
                failed_path=list(refusal.failed_path or ()),
                level="warning",
            )
            self._controls_v2_report_pending_refusal(refusal, "focus-loss")
            self._refresh_controls_v2_profile(immediate=True)
            return
        # The current edit is per-field-valid.  Stage the COMPLETE journal winner
        # set and commit through the SAME checked atomic engine.  The commit owns
        # the display projection, source reconciliation, energy/probe cache
        # invalidation, generation bump, and exact-revision clearing — there is no
        # second setter authority.  The `applying_field` guard blocks the
        # parameter-tree echo while the engine writes signal-blocked carriers.
        self._controls_v2_applying_field = True
        try:
            result = self._controls_v2_commit_journal_winners()
        finally:
            self._controls_v2_applying_field = False
        if result is None:
            # Complete-valid: installed atomically, consumed revisions cleared.
            self._controls_v2_last_refusal_signature = None
            # §14.11.D.4: a PONI edit invalidates the energy cache LOCALLY
            # (energy-preference and source-selection cache invalidation already
            # happen inside the commit engine).
            if path in self._CONTROLS_V2_PONI_PATHS:
                self._controls_v2_source_energy_cache = None
            # The pure staging reducer resolves the GI θ-motor from the candidate
            # OBSERVATION, which is UNKNOWN for a lazily-hydrated directory source
            # (Correction C); the LIVE _controls_v2_gi_config applies the R4B-15
            # stale-motor repick over the REAL offered motors.  Reconcile the
            # installed intent + display scan + integrator combo with that live
            # resolution so all θ-motor surfaces AND the freeze agree without
            # injecting a motor not offered by the source (CLAUDE.md GI rule).
            if path and path[0] == "GI":
                cfg = self._controls_v2_gi_config()
                live_intent = self._controls_v2_ensure_run_intent()
                live_intent.gi = GIIntent(
                    enabled=bool(cfg["gi"]),
                    incidence_motor=str(cfg["incidence_motor"] or "Manual"),
                    th_val=float(cfg["th_val"] or 0.0),
                    sample_orientation=int(cfg["sample_orientation"] or 4),
                    tilt_angle=float(cfg["tilt_angle"] or 0.0),
                    mode_1d=str(cfg["gi_mode_1d"]),
                    mode_2d=str(cfg["gi_mode_2d"]),
                )
                if live_intent.gi.enabled and getattr(self, "scan", None) is not None:
                    self._controls_v2_apply_gi_config_to_scan(cfg)
                if cfg.get("gi") and cfg.get("incidence_motor"):
                    self._controls_v2_sync_integrator_gi_motor(
                        cfg["incidence_motor"])
            run_config_debug_log(
                logger,
                "controls_field_applied",
                widget=self,
                origin="controls_v2_ui",
                field_path=path,
                field_value=value,
                revision=_idle_rev,
                edit_origin="idle",
            )
            self._refresh_controls_v2_profile(immediate=True)
            return
        if result.phase in ("stage", "harvest"):
            # §17.4: the current edit is individually valid but the complete
            # candidate is a transient cross-field contradiction (or a still-
            # pending prior draft) — keep it JOURNAL-OWNED and panel-overlaid; NO
            # install, NO committed mutation, NO refusal message.  A later
            # resolving edit stages the complete winner set and commits both.
            run_config_debug_log(
                logger,
                "controls_field_pending",
                widget=self,
                origin="controls_v2_ui",
                field_path=path,
                field_value=value,
                revision=_idle_rev,
                phase=result.phase,
                failed_path=list(result.failed_path or ()),
            )
            self._refresh_controls_v2_profile(immediate=True)
            return
        # An install/source_reconcile/legacy_apply failure on a VALID candidate:
        # the engine contained the exception and rolled back (nothing half-
        # applied) — surface ONE typed refusal (the fold already logged the
        # structured phase/path event).
        self._controls_v2_report_pending_refusal(result, "focus-loss")
        self._refresh_controls_v2_profile(immediate=True)

    def _controls_v2_validate_idle_edit(self, path, value):
        """§15.12-A.1 / §15.3: validate ONE committed idle edit through the pure
        strict PER-FIELD reducer, WITHOUT applying anything.  Returns ``None``
        when the edit is acceptable, or a typed :class:`ControlsCommitResult`
        (``phase="stage"``) when the strict reducer refuses it — so the caller
        declines to mutate any carrier and surfaces a stable refusal.

        Only PER-FIELD validity is enforced here (non-integral / unknown enum /
        unsupported / uncoercible — the values the old permissive setter would
        have silently clamped, §15.3).  CROSS-FIELD contradictions (threshold
        min>max, reversed range) are NOT refused: an idle edit may be transiently
        inconsistent (min entered before max) and the complete candidate is
        validated at the action/Run boundary.  The reducer mutates only a private
        candidate clone, so this is side-effect free."""
        cand = self._controls_v2_new_stage_candidate()
        try:
            self._controls_v2_reduce_edit(cand, tuple(path), value)
        except ControlsTransactionError as err:
            return ControlsCommitResult(
                False, failed_path=err.path, reason=err.reason, phase="stage")
        return None

    def _on_controls_v2_field_draft_changed(self, path, value) -> None:
        """A focused, still-uncommitted form DRAFT changed (§12.4).

        The draft receives its own revision at the time the user types it (so a
        newer draft supersedes an older deferred/idle value by revision, and the
        draft survives a panel rebuild/refresh because it lives in the journal),
        but it is NOT applied to a carrier until it commits through
        :meth:`_on_controls_v2_field_changed`.  Programmatic ``setText``/
        projection must NOT reach this slot."""
        path = tuple(path)
        revision = self._controls_v2_record_edit(path, value, origin="draft")
        run_config_debug_log(
            logger,
            "controls_field_draft",
            widget=self,
            origin="controls_v2_ui",
            field_path=list(path),
            field_value=value,
            revision=revision,
            edit_origin="draft",
        )

    def _on_controls_v2_source_tree_changed(self, _param, changes) -> None:
        if self._controls_v2_run_active():
            return
        # §14.11.D.3: an ECHO of this widget's OWN programmatic idle apply — the
        # field handler already owns the (membership-conditional) reconcile, so
        # skip to avoid the double reconcile.  A GENUINE direct-in-tree user edit
        # (no guard set) still reconciles here.
        if getattr(self, "_controls_v2_applying_field", False):
            return
        try:
            if not any(change[1] == "value" for change in changes):
                return
        except (IndexError, TypeError):
            pass
        self._sync_controls_v2_source_index()

    def _connect_controls_v2_source_tree(self) -> None:
        """Follow the active wrangler's source parameters.

        The Controls V2 panel is built before the wrangler stack, so this must
        be connected from ``set_wrangler`` rather than panel construction.
        """
        previous = getattr(self, "_controls_v2_source_param_signal", None)
        slot = getattr(self, "_controls_v2_source_tree_slot", None)
        if previous is not None and slot is not None:
            try:
                previous.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        parameters = getattr(getattr(self, "wrangler", None), "parameters", None)
        signal = getattr(parameters, "sigTreeStateChanged", None)
        if signal is None:
            self._controls_v2_source_param_signal = None
            return
        if slot is None:
            slot = self._on_controls_v2_source_tree_changed
            self._controls_v2_source_tree_slot = slot
        signal.connect(slot)
        self._controls_v2_source_param_signal = signal

    def _on_controls_v2_field_browse(self, path) -> None:
        if self._controls_v2_run_active():
            logger.debug("Ignoring Controls V2 browse during active run: %s", path)
            return
        path = tuple(path)
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        handlers = {
            ("Project", "project_folder"): self._controls_v2_choose_project,
            ("Project", "h5_dir"): self._controls_v2_choose_output,
            ("Output", "h5_dir"): self._controls_v2_choose_output,
            ("Signal", "poni_file"): getattr(wrangler, "set_poni_file", None),
            ("Calibration", "poni_file"): getattr(wrangler, "browse_poni", None),
            ("Signal", "File"): getattr(wrangler, "set_img_file", None),
            ("Signal", "img_dir"): getattr(wrangler, "set_img_dir", None),
            ("Signal", "meta_dir"): getattr(wrangler, "set_meta_dir", None),
            ("Signal", "mask_file"): (
                getattr(wrangler, "set_mask_file", None)
                or getattr(wrangler, "browse_mask", None)
            ),
            ("NeXus File", "nexus_file"): getattr(wrangler, "browse_nexus", None),
            ("BG", "File"): getattr(wrangler, "set_bg_file", None),
        }
        handler = handlers.get(path)
        if callable(handler):
            handler()
        if path and path[0] in {"Signal", "Source"}:
            self._controls_v2_source_energy_cache = None
            self._controls_v2_metadata_probe_cache = None
        self._refresh_controls_v2_profile(immediate=True)

    def _on_mask_created(self, mask_file) -> None:
        """Auto-populate the Mask File field after Make Mask saves a mask.

        Single source of truth: write the same wrangler param the Mask File
        browse handler sets (``("Signal", "mask_file")``) + mirror the cached
        ``wrangler.mask_file`` attr, then refresh the V2 panel so the new path
        shows.  No-op if the path is empty or the param doesn't exist (e.g. a
        wrangler without a mask_file param).
        """
        if not mask_file:
            return
        param = self._controls_v2_param(("Signal", "mask_file"))
        if param is not None:
            try:
                param.setValue(str(mask_file))
            except Exception:
                logger.debug("Auto-populate mask_file failed", exc_info=True)
        wrangler = getattr(self, "wrangler", None)
        if wrangler is not None:
            try:
                wrangler.mask_file = str(mask_file)
            except Exception:
                pass
        self._force_controls_v2_rebuild()

    def _autofill_poni_after_calibrate(self, since_ts) -> None:
        """Offer to adopt a PONI written by the just-closed pyFAI-calib2.

        pyFAI-calib2 is an external program that doesn't tell us where the user
        saved the ``.poni``, so we rglob the project folder (falling back to the
        image directory) for a ``*.poni`` modified at/after the calibration
        launch and, if one is found, populate the Poni field after a
        confirmation popup so the user can double-check.  Picks the newest if
        several match; does nothing if none are found.
        """
        from pathlib import Path
        folder = ""
        for path in (("Project", "project_folder"), ("Signal", "img_dir")):
            param = self._controls_v2_param(path)
            try:
                value = param.value() if param is not None else ""
            except Exception:
                value = ""
            if value and os.path.isdir(value):
                folder = value
                break
        if not folder:
            return
        try:
            recent = [
                p for p in Path(folder).rglob("*.poni")
                if p.stat().st_mtime >= since_ts - 2.0
            ]
        except Exception:
            logger.debug("PONI auto-detect scan failed", exc_info=True)
            return
        if not recent:
            return
        newest = max(recent, key=lambda p: p.stat().st_mtime)
        poni_path = str(newest)
        extra = (f"\n\n(newest of {len(recent)} created during calibration)"
                 if len(recent) > 1 else "")
        answer = QMessageBox.question(
            self, "Calibration complete",
            "A new PONI file was found:\n\n"
            f"{poni_path}{extra}\n\nUse it as the calibration for this scan?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            self._set_poni_field(poni_path)

    def _set_poni_field(self, poni_path) -> None:
        """Adopt ``poni_path`` as the calibration, mirroring a manual browse.

        Sets the active Poni param — Image: ``("Signal", "poni_file")`` (its
        ``sigValueChanged`` runs ``get_poni_dict``); NeXus: ``("Calibration",
        "poni_file")`` — mirrors the wrangler's cached ``poni_file``, reloads
        the NeXus wrangler's cached ``poni`` (which only reloads on browse /
        session-restore, not a bare ``setValue``), then refreshes the V2 panel.
        """
        poni_path = str(poni_path)
        wrangler = getattr(self, "wrangler", None)
        if wrangler is not None:
            # The Image wrangler's value-change callback reads this attribute,
            # so publish the new path before emitting the parameter signal.
            try:
                wrangler.poni_file = poni_path
            except Exception:
                pass
        for path in (("Signal", "poni_file"), ("Calibration", "poni_file")):
            param = self._controls_v2_param(path)
            if param is not None:
                try:
                    param.setValue(poni_path)
                except Exception:
                    logger.debug("Set poni_file param failed for %s", path,
                                 exc_info=True)
                break
        if wrangler is not None:
            loader = getattr(wrangler, "get_poni_dict", None)
            if callable(loader):
                try:
                    loader()
                except Exception:
                    logger.debug("PONI reload after config adoption failed",
                                 exc_info=True)
            elif isinstance(wrangler, nexusWrangler):
                try:
                    from xrd_tools.core.containers import PONI
                    if os.path.exists(poni_path):
                        wrangler.poni = PONI.from_poni_file(poni_path)
                except Exception:
                    logger.debug("PONI reload after autofill failed",
                                 exc_info=True)
            loaded_poni = getattr(wrangler, "poni", None)
            thread = getattr(wrangler, "thread", None)
            if thread is not None and loaded_poni is not None:
                try:
                    thread.poni = loaded_poni
                except Exception:
                    pass
        self._force_controls_v2_rebuild()

    def _controls_v2_choose_source(self) -> None:
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        try:
            mode_text = self.controls.current_mode()
        except Exception:
            mode_text = ""
        if hasattr(wrangler, "browse_nexus"):
            wrangler.browse_nexus()
            return
        if "directory" in str(getattr(wrangler, "inp_type", "")).lower():
            browse = getattr(wrangler, "set_img_dir", None)
        else:
            browse = getattr(wrangler, "set_img_file", None)
        if mode_text in ("Image Viewer", "XYE Viewer"):
            browse = getattr(wrangler, "set_img_file", None) or browse
        if callable(browse):
            browse()

    def _controls_v2_choose_project(self) -> None:
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        browse = getattr(wrangler, "set_project_folder", None)
        if browse is None:
            browse = getattr(wrangler, "browse_project_folder", None)
        if callable(browse):
            browse()

    def _controls_v2_choose_output(self) -> None:
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        browse = getattr(wrangler, "set_h5_dir", None)
        if browse is None:
            browse = getattr(wrangler, "browse_h5_dir", None)
        if callable(browse):
            browse()

    def _refresh_controls_v2_profile(
        self,
        *,
        immediate: bool = False,
        preserve_focused_editor: bool = True,
    ) -> None:
        if getattr(self, "_tearing_down", False):
            return
        timer = getattr(self, "_controls_v2_refresh_timer", None)
        if not preserve_focused_editor:
            self._controls_v2_force_next_refresh = True
        if immediate or timer is None:
            self._refresh_controls_v2_profile_now(
                preserve_focused_editor=preserve_focused_editor)
        else:
            timer.trigger()

    def _refresh_controls_v2_profile_now(
        self,
        *,
        preserve_focused_editor: bool = True,
    ) -> None:
        panel = getattr(self, "controls_v2", None)
        if panel is None:
            return
        if self._controls_v2_batch_run_active():
            self._controls_v2_batch_refresh_deferred = True
            self._note_controls_v2_refresh("deferred_run")
            return
        import time as _time
        _t0 = _time.perf_counter()
        try:
            self._controls_v2_batch_refresh_deferred = False
            state = self._controls_v2_state()
            values = self._controls_v2_field_values()
            choices = self._controls_v2_field_choices()
            render_state = build_control_panel_state(state, values, choices)
            self._controls_v2_update_run_summary(state, render_state.profile)
            signature = render_state
            schema_signature = self._controls_v2_render_schema_signature(render_state)
            force_refresh = bool(
                getattr(self, "_controls_v2_force_next_refresh", False)
            )
            self._controls_v2_force_next_refresh = False
            if (
                not force_refresh
                and signature == getattr(self, "_controls_v2_last_signature", None)
            ):
                self._controls_v2_refresh_failure_warned = False
                self._note_controls_v2_refresh(
                    "noop", (_time.perf_counter() - _t0) * 1000)
                return
            schema_changed = (
                schema_signature
                != getattr(self, "_controls_v2_last_schema_signature", None)
            )
            if not schema_changed:
                updater = getattr(panel, "apply_state_update", None)
                if callable(updater) and updater(render_state):
                    self._cancel_deferred_controls_v2_refresh()
                    self._controls_v2_last_signature = signature
                    self._controls_v2_last_schema_signature = schema_signature
                    self._controls_v2_refresh_failure_warned = False
                    self._note_controls_v2_refresh(
                        "in_place", (_time.perf_counter() - _t0) * 1000)
                    return
            # A background rebuild (set_state -> clear_rows) would destroy a line
            # editor the user is mid-way through and silently drop the uncommitted
            # text.  If one is focused, defer until it COMMITS (editingFinished =
            # Enter / focus loss) rather than re-arming the throttle every interval
            # (which would keep a timer waking through a long acquisition).  The
            # rebuild is then SCHEDULED through the throttle, never run inside the
            # editingFinished slot (which would delete the editor mid-emission).
            # Signature is left unstamped so the deferred pass still detects the
            # change and rebuilds.
            editor = panel.focusWidget()
            if (
                preserve_focused_editor
                and not force_refresh
                and isinstance(editor, QtWidgets.QLineEdit)
            ):
                self._defer_controls_v2_refresh_until_commit(editor)
                self._controls_v2_refresh_failure_warned = False
                self._note_controls_v2_refresh(
                    "deferred_focus", (_time.perf_counter() - _t0) * 1000)
                return
            self._cancel_deferred_controls_v2_refresh()
            panel.set_state(render_state)
            self._controls_v2_last_signature = signature
            self._controls_v2_last_schema_signature = schema_signature
            self._controls_v2_refresh_failure_warned = False
            self._note_controls_v2_refresh(
                "full_rebuild", (_time.perf_counter() - _t0) * 1000)
        except Exception:
            if not getattr(self, "_controls_v2_refresh_failure_warned", False):
                self._controls_v2_refresh_failure_warned = True
                logger.warning(
                    "Controls Panel V2 profile refresh failed; readiness may be stale",
                    exc_info=True,
                )
            else:
                logger.debug("Controls Panel V2 profile refresh failed",
                             exc_info=True)

    def _note_controls_v2_refresh(self, kind: str, elapsed_ms: float = 0.0) -> None:
        stats = getattr(self, "_controls_v2_refresh_stats", None)
        if stats is None:
            stats = {}
            self._controls_v2_refresh_stats = stats
        stats[kind] = int(stats.get(kind, 0)) + 1
        if os.environ.get("XDART_PERF"):
            logger.info(
                "[PERF] controls-v2 refresh: kind=%s elapsed=%.0fms counts=%s",
                kind, elapsed_ms, dict(sorted(stats.items())),
            )

    def _invalidate_controls_v2_render_cache(self) -> None:
        self._cancel_deferred_controls_v2_refresh()
        self._controls_v2_last_signature = None
        self._controls_v2_last_schema_signature = None

    def _force_controls_v2_rebuild(self) -> None:
        self._cancel_deferred_controls_v2_refresh()
        self._controls_v2_force_next_refresh = True
        self._refresh_controls_v2_profile(
            immediate=True,
            preserve_focused_editor=False,
        )

    @staticmethod
    def _controls_v2_render_schema_signature(render_state) -> tuple:
        profile = render_state.profile
        bound = render_state.bound_controls

        def _value(value):
            return getattr(value, "value", value)

        if bound is not None:
            field_sig = tuple(
                (
                    _value(field.section),
                    field.label,
                    tuple(field.path),
                    _value(field.kind),
                    tuple(str(choice) for choice in field.choices),
                    bool(field.browse),
                    str(field.parameter_group or ""),
                )
                for field in bound.fields
            )
        else:
            field_sig = tuple(
                (
                    _value(field_id),
                    status.label,
                    _value(status.section),
                )
                for field_id, status in sorted(
                    profile.fields.items(), key=lambda item: _value(item[0]))
            )

        action_sig = tuple(
            (
                _value(section),
                tuple(
                    (
                        _value(spec.action),
                        spec.label,
                        bool(spec.enabled),
                        spec.reason,
                    )
                    for spec in specs
                ),
            )
            for section, specs in sorted(
                profile.section_actions.items(), key=lambda item: _value(item[0]))
        )
        analysis_sig = tuple(
            (
                _value(spec.tool),
                spec.label,
                bool(spec.enabled),
                spec.reason,
            )
            for spec in profile.analysis_launchers
        )
        return (
            field_sig,
            action_sig,
            analysis_sig,
            bool(profile.show_experiment_card),
            bool(profile.show_processing_card),
        )

    def _controls_v2_update_run_summary(self, state: ControlState, profile) -> None:
        controls = getattr(self, "controls", None)
        setter = getattr(controls, "set_readiness_summary", None)
        text, ready, tooltip = self._controls_v2_run_summary(state, profile)
        live_btn = getattr(controls, "liveButton", None)
        live = bool(live_btn is not None and live_btn.isChecked()
                    and live_btn.isVisibleTo(controls))
        if (not ready and not live
                and self._controls_v2_live_watch_armable()):
            mode = str(getattr(state, "processing_mode", "") or "").strip()
            text = "Needs setup · Enable Live to watch this directory"
            if mode:
                text += f" · {mode}"
            hint = "Enable Live to arm this configured directory before files arrive"
            tooltip = f"{hint}; {tooltip}" if tooltip else hint
        # A live watch that is idling between files keeps its waiting text
        # against profile-refresh repaints (maintainer, 2026-07-13).
        if (text and getattr(self, "_live_waiting_status", False)
                and self._controls_v2_run_active()):
            text = "Waiting for new images…"
            ready, tooltip, live = True, "Live watch: no new files yet", True
        files_mode = bool(getattr(state, "frame_count_is_files", False)
                          and text)
        if files_mode:
            hint = "Click to count frames in all files"
            tooltip = f"{tooltip}; {hint}" if tooltip else hint
        if callable(setter):
            changed = setter(text, ready=ready, tooltip=tooltip, live=live)
            if changed:
                self._fit_controls_height()
        try:
            label = getattr(controls, "readinessLabel", None)
            if label is not None:
                label.setCursor(
                    QtCore.Qt.PointingHandCursor if files_mode
                    else QtCore.Qt.ArrowCursor)
        except Exception:
            logger.debug("readiness cursor update failed", exc_info=True)
        self._controls_v2_sync_run_row(profile)

    def _on_wrangler_status_text(self, text) -> None:
        """Mirror the live-watch state into the readiness summary (maintainer,
        2026-07-13): while a live run idles waiting for new files the bar says
        so; any other wrangler status restores the normal summary."""
        waiting = str(text or "").startswith("Watching for new files")
        if waiting == getattr(self, "_live_waiting_status", False):
            return
        self._live_waiting_status = waiting
        setter = getattr(getattr(self, "controls", None),
                         "set_readiness_summary", None)
        if not callable(setter):
            return
        if waiting and self._controls_v2_run_active():
            setter("Waiting for new images…", ready=True,
                   tooltip="Live watch: no new files yet", live=True)
        else:
            # Restore the normal summary DIRECTLY: mid-run the throttled
            # profile rebuild defers (_controls_v2_batch_run_active), so it
            # would never repaint over the waiting text.
            try:
                state = self._controls_v2_state()
                render_state = build_control_panel_state(
                    state,
                    self._controls_v2_field_values(),
                    self._controls_v2_field_choices(),
                )
                self._controls_v2_update_run_summary(
                    state, render_state.profile)
            except Exception:
                logger.debug("live-watch summary restore failed",
                             exc_info=True)

    def _controls_v2_sync_run_row(self, profile) -> None:
        controls = getattr(self, "controls", None)
        if controls is None or self._controls_v2_run_active():
            return
        try:
            viewer = str(getattr(profile.processing_page, "value", "")) == "viewer"
            if viewer or controls.actionRow.isHidden():
                return
            can_run = bool(getattr(profile, "can_run", False))
            live_armable = self._controls_v2_live_watch_armable()
            row_enabled = can_run or live_armable
            controls.set_run_row_enabled(row_enabled)
            phase = getattr(controls, "action_phase", lambda: "idle")()
            if phase == "idle":
                # Child widgets retain explicit disabled state independently
                # of their parent on native Qt.  This method owns idle
                # readiness, so restore both affordances explicitly.
                controls.liveButton.setEnabled(row_enabled)
                controls.startButton.setEnabled(can_run)

            signature = (
                bool(can_run), bool(live_armable), bool(row_enabled), str(phase),
                bool(controls.liveButton.isEnabled()),
                bool(controls.liveButton.isChecked()),
                bool(controls.startButton.isEnabled()),
            )
            if signature != getattr(
                    self, "_controls_v2_run_affordance_debug_signature", None):
                self._controls_v2_run_affordance_debug_signature = signature
                browse_debug_log(
                    logger,
                    "controls_run_affordances",
                    can_run=can_run,
                    live_armable=live_armable,
                    row_enabled=row_enabled,
                    phase=phase,
                    live_enabled=controls.liveButton.isEnabled(),
                    live_checked=controls.liveButton.isChecked(),
                    run_enabled=controls.startButton.isEnabled(),
                )
        except Exception:
            logger.debug("Controls V2 run-row readiness sync failed",
                         exc_info=True)

    @staticmethod
    def _controls_v2_run_summary(state: ControlState, profile) -> tuple[str, bool, str]:
        if str(getattr(getattr(profile, "processing_page", None), "value", "")) == "viewer":
            return "", False, ""
        ready = bool(getattr(profile, "can_run", False))
        append_confirm_reason = str(
            getattr(profile, "append_confirm_reason", "") or "")
        status = (
            "Confirm overwrite"
            if append_confirm_reason and ready else
            "Ready" if ready else "Needs setup"
        )
        mode = str(getattr(state, "processing_mode", "") or "").strip()
        if not mode:
            page = getattr(getattr(profile, "processing_page", None),
                           "value", "")
            mode = str(page).replace("_", " ").title()
        blockers = tuple(getattr(profile, "run_blockers", ()) or ())
        parts = [status]
        note = run_target_readiness_note(state, ready=ready).rstrip(".")
        if append_confirm_reason and ready:
            parts.append(staticWidget._controls_v2_visible_run_blocker(
                append_confirm_reason))
        elif not ready:
            if blockers:
                parts.append(
                    staticWidget._controls_v2_visible_run_blocker(blockers[0]))
            elif note:
                parts.append(staticWidget._controls_v2_visible_run_blocker(note))
        if mode:
            parts.append(mode)
        frame_count = int(getattr(state, "frame_count", 0) or 0)
        if ready and frame_count:
            unit = ("file" if getattr(state, "frame_count_is_files", False)
                    else "frame")
            plural = "" if frame_count == 1 else "s"
            parts.append(f"{frame_count} {unit}{plural}")
        tooltip_parts = [str(append_confirm_reason)] if append_confirm_reason else []
        tooltip_parts.extend(str(b) for b in blockers[:3])
        if note and note not in tooltip_parts:
            tooltip_parts.append(note)
        tooltip = "; ".join(tooltip_parts) if tooltip_parts else ""
        return " · ".join(parts), ready, tooltip

    @staticmethod
    def _controls_v2_visible_run_blocker(message: object) -> str:
        text = str(message).rstrip(".")
        if text.startswith("Run needs a frame source"):
            return "Run needs a frame source"
        return text

    def _defer_controls_v2_refresh_until_commit(self, editor) -> None:
        """Arm a one-shot: when ``editor`` finishes editing (Enter / focus loss),
        schedule the deferred Controls V2 rebuild through the throttle.  Replaces
        re-arming the throttle each interval, so a focused field during a long
        acquisition no longer keeps a timer alive."""
        if getattr(self, "_controls_v2_pending_editor", None) is editor:
            return
        self._cancel_deferred_controls_v2_refresh()
        self._controls_v2_pending_editor = editor
        editor.editingFinished.connect(self._on_controls_v2_pending_editor_done)

    def _cancel_deferred_controls_v2_refresh(self) -> None:
        """Drop a pending editingFinished one-shot (editor committed, was
        destroyed by a rebuild, or teardown)."""
        editor = getattr(self, "_controls_v2_pending_editor", None)
        if editor is not None:
            try:
                editor.editingFinished.disconnect(
                    self._on_controls_v2_pending_editor_done)
            except (TypeError, RuntimeError):
                pass
        self._controls_v2_pending_editor = None

    def _on_controls_v2_pending_editor_done(self, *args) -> None:
        """The deferred-on editor committed: schedule (do NOT run) the rebuild
        via the throttle, so it lands after this slot returns and we never delete
        the editor while it is still emitting editingFinished."""
        self._cancel_deferred_controls_v2_refresh()
        self._refresh_controls_v2_profile(immediate=False)

    def _on_gi_motor_options_changed(self, payload=None) -> None:
        """THE single static-widget owner of GI motor hydration (§13.6).

        Receives a structured :class:`GIMotorHydration` from the wrangler and,
        BEFORE it touches any stored knowledge or the visible theta dropdown,
        verifies the token/epoch is still current — so a delayed result from a
        superseded source (fingerprint mismatch) or an older same-root request
        (lower generation) is IGNORED.  Only a current result updates BOTH the
        stored source-qualified observation AND the integrator's GI-motor combo
        (the old direct signal→integrator connection is removed).  A bare legacy
        list payload has no identity and is treated as current."""
        wrangler = getattr(self, "wrangler", None)
        if isinstance(payload, GIMotorHydration):
            checker = getattr(wrangler, "gi_hydration_is_current", None)
            if callable(checker) and not checker(payload):
                run_config_debug_log(
                    logger,
                    "gi_hydration_ignored_stale",
                    widget=self,
                    origin="controls_v2_gi_hydration",
                    state=payload.state,
                    generation=payload.generation,
                )
                return
            motors = list(payload.motors)
            state = payload.state
        else:
            # Legacy/duck-typed list payload (kept working for direct callers).
            motors = payload if isinstance(payload, (list, tuple)) else None
            if motors is None:
                motors = getattr(wrangler, "motors", None) or []
            motors = list(motors)
            state = None
        # Update the stored motor knowledge (source-qualified) ...
        self._controls_v2_record_gi_motor_observation(motors, state=state)
        # ... and the visible theta options.  Skip a wipe when the source was
        # NOT inspected (UNKNOWN with no motors), so an explicit / session-
        # restored motor stays visible (§13.7).
        if state != GIMotorObservation.UNKNOWN or motors:
            integrator = getattr(self, "integratorTree", None)
            setter = getattr(integrator, "set_gi_motor_options", None)
            if callable(setter):
                try:
                    setter(motors)
                except Exception:
                    logger.debug("integrator GI-motor option update failed",
                                 exc_info=True)
        self._refresh_controls_v2_profile(immediate=True)

    def _controls_v2_state(self) -> ControlState:
        """Build a lightweight, best-effort Controls V2 state snapshot."""
        mode_text = ""
        try:
            mode_text = self.controls.current_mode()
        except Exception:
            pass
        tool = tool_from_mode_text(mode_text)
        gi_cfg = {}
        try:
            gi_cfg = (
                self._controls_v2_gi_config()
                if self._controls_v2_enabled()
                else self.integratorTree.get_gi_config()
            )
        except Exception:
            gi_cfg = {}
        display_scan = getattr(self, "scan", None)
        gi_on = bool(
            gi_cfg.get("gi", getattr(display_scan, "gi", False)))
        meas_mode = MeasMode.GI if gi_on else MeasMode.STANDARD

        loaded_frame_count = 0
        try:
            loaded_frame_count = len(getattr(self.scan.frames, "index", ()) or ())
        except Exception:
            loaded_frame_count = 0
        source_label = self._controls_v2_source_label()
        try:
            source_frame_count = self._controls_v2_source_frame_count()
        except Exception:
            logger.debug("Controls V2 source frame count failed", exc_info=True)
            source_frame_count = 0
        source_live_unknown = source_frame_count is None
        frame_count = 0 if source_live_unknown else int(source_frame_count or 0)
        frame_count_is_files = bool(
            getattr(self, "_v2_source_count_is_files", False)
            and not source_live_unknown)
        project_root = str(getattr(getattr(self, "wrangler", None), "project_folder", "") or "")
        project_root_valid = bool(
            project_root and os.path.isdir(os.path.expanduser(project_root))
        )
        has_scan_data = False
        try:
            has_scan_data = not getattr(self.scan, "scan_data", None).empty
        except Exception:
            has_scan_data = False
        wrangler = getattr(self, "wrangler", None)
        has_motors = bool(getattr(wrangler, "motors", None)) or has_scan_data
        calibration_energy_eV, source_energy_eV = (
            self._controls_v2_energy_values())
        energy_known = (
            calibration_energy_eV is not None
            or source_energy_eV is not None
        )
        source_caps, source_ready = self._controls_v2_source_caps(
            source_label=source_label,
            frame_count=frame_count,
            live_unknown=source_live_unknown,
            has_metadata=has_scan_data or bool(getattr(wrangler, "scan_args", None)),
            has_motors=has_motors,
            has_energy=energy_known,
            has_geometry=self._controls_v2_calibrated(),
            has_psi_metadata=has_scan_data,
        )

        has_1d = bool(self.viewer_rows_1d)
        has_2d = bool(self.viewer_rows_2d)
        labels = ()
        pubs = ()
        try:
            labels = self.publication_store.labels()
            recent = labels[-16:] if len(labels) > 16 else labels
            pubs = tuple(self.publication_store.get_many(recent).values())
            has_1d = has_1d or any(getattr(pub.view, "int_1d", None) is not None
                                   for pub in pubs)
            has_2d = has_2d or any(getattr(pub.view, "int_2d", None) is not None
                                   for pub in pubs)
        except Exception:
            labels = ()
            pubs = ()
        loaded_scan_file = self._controls_v2_loaded_scan_file(
            loaded_frame_count=loaded_frame_count, labels=labels)
        loaded_scan_available = bool(
            loaded_frame_count or labels or loaded_scan_file)
        # H18: ResultCaps delegate to record truth (capabilities_for_processed
        # over the loaded .nxs, with raw_reachable consulting the frame-0
        # probe — H5 finding 2) OR-composed with in-memory truth (viewer rows,
        # publications, hydrated scan_data).
        # H18-R1: the raw pair is IDENTITY-AWARE.  When a real processed scan
        # is selected (browsed/loaded), ITS record truth plus identity-
        # qualified resident browse evidence (a publication of the displayed
        # scan carrying an actual raw payload) owns the result raw facts —
        # the frozen configured acquisition source must never contaminate
        # them (during Pause the source is the acquisition identity while
        # the selected record is the explicit browse target; an orphaned
        # record must not advertise ROI reachability because the source is
        # reachable).  An UNCACHED record during Run/Pause is pending:
        # "not probed" is never reported as reachable — only resident
        # evidence may prove it.  Source caps supply source-only ROI
        # readiness when no processed result is selected.
        # H18-R4: resident evidence is qualified against the SELECTED (or,
        # with nothing selected, the configured acquisition) identity through
        # the one canonical scan-name helper; unknown or mismatched identity
        # fails closed, only actual resident pixels count, and the outgoing
        # store is never borrowed while a manual browser rescope is pending.
        resident_raw = self._controls_v2_resident_selected_raw(
            pubs, loaded_scan_file, source_label)
        record_caps = self._controls_v2_loaded_result_caps(loaded_scan_file)
        if loaded_scan_file:
            if record_caps is not None:
                loaded_has_raw = bool(record_caps.has_raw or resident_raw)
                loaded_raw_reachable = bool(
                    record_caps.raw_reachable or resident_raw)
            else:
                loaded_has_raw = loaded_raw_reachable = bool(resident_raw)
            result_has_raw = loaded_has_raw
            result_raw_reachable = loaded_raw_reachable
        else:
            result_has_raw = bool(source_caps.has_raw or resident_raw)
            result_raw_reachable = bool(
                source_caps.raw_reachable or resident_raw)
        result_caps = ResultCaps(
            has_1d=bool(
                has_1d or (record_caps is not None and record_caps.has_1d)),
            has_2d=bool(
                has_2d or (record_caps is not None and record_caps.has_2d)),
            has_raw=result_has_raw,
            raw_reachable=result_raw_reachable,
            has_scan_metadata=bool(
                has_scan_data
                or (record_caps is not None and record_caps.has_scan_metadata)),
            has_rsm=bool(
                getattr(self.scan, "rsm_result", None)
                or (record_caps is not None and record_caps.has_rsm)),
            has_phase_result=bool(
                record_caps is not None and record_caps.has_phase_result),
            has_psi_metadata=bool(
                has_scan_data
                or (record_caps is not None and record_caps.has_psi_metadata)),
        )
        geom = GeomState(
            calibrated=self._controls_v2_calibrated(),
            energy_known=energy_known,
            calibration_energy_eV=calibration_energy_eV,
            source_energy_eV=source_energy_eV,
            gi_enabled=gi_on,
            sample_orientation_known=(
                not gi_on or bool(gi_cfg.get("sample_orientation"))),
            ub_known=bool(getattr(self.scan, "ub_matrix", None)),
            material_known=False,
        )
        run_active = self._controls_v2_run_active()
        display_frame_count = frame_count
        if run_active:
            # During a run the processed-frame count ticks every frame; letting
            # it into the render signature would rebuild the whole controls panel
            # ~5 Hz (clear_rows + recreate every row).  Live progress is shown in
            # the status bar, so freeze the panel's frame count at the run-start
            # value (snapshot once) — it resyncs when the run ends and the
            # snapshot clears.  The panel is locked during a run, so a frozen
            # count loses nothing.
            if getattr(self, "_controls_v2_run_frame_count", None) is None:
                self._controls_v2_run_frame_count = display_frame_count
            display_frame_count = self._controls_v2_run_frame_count
        else:
            self._controls_v2_run_frame_count = None
        if source_ready:
            run_target = RunTarget.SOURCE
        elif loaded_scan_available:
            run_target = RunTarget.LOADED_SCAN
        else:
            run_target = RunTarget.NONE
        write_mode = "Append"
        try:
            write_mode = self.controls.write_mode()
        except Exception:
            pass
        scan = getattr(self, "scan", None)
        processed_config = None
        if self._controls_v2_append_target_matches_displayed_scan():
            processed_config = processing_config_from_scan(
                scan,
                prefer_stored=True,
            )
        return ControlState(
            tool=tool,
            mode=meas_mode,
            source_caps=source_caps,
            result_caps=result_caps,
            geom=geom,
            backend=self._controls_v2_backend(tool),
            project_root=project_root,
            project_root_required=True,
            project_root_valid=project_root_valid,
            source_label=source_label,
            run_target=run_target,
            loaded_scan_available=loaded_scan_available,
            save_path=str(getattr(getattr(self, "wrangler", None), "h5_dir", "") or ""),
            write_mode=write_mode,
            processed_config=processed_config,
            current_config=processing_config_from_scan(scan),
            detector_summary=self._controls_v2_detector_summary(),
            frame_count=display_frame_count,
            frame_count_is_files=frame_count_is_files,
            processing_mode=mode_text,
            real_data_gates=frozenset(),
            controls_locked=run_active,
        )

    def _controls_v2_source_caps(
        self,
        *,
        source_label: str,
        frame_count: int,
        live_unknown: bool,
        has_metadata: bool,
        has_motors: bool,
        has_energy: bool,
        has_geometry: bool,
        has_psi_metadata: bool,
    ) -> tuple[SourceCaps, bool]:
        """H18: ONE gating truth source.  The tri-fields
        (``has_frames``/``has_raw``/``raw_reachable``) come from the headless
        ``describe_source_readiness`` instead of the retired inline
        ``has_frames = has_raw = raw_reachable = source_ready`` collapse.
        Per the H5 parity findings the merge policy is:

        * tri-fields — headless truth.  The panel's own wrangler frame count
          is OR-ed into has_frames/has_raw (the wrangler counts filtered image
          series the generic probe cannot parse) but NEVER into raw_reachable,
          which stays frame-0 probe truth (live escape hatch included).
        * metadata family — for NON-live sources, panel truth OR what the
          source itself serves (a SPEC scan table / a processed record carries
          metadata, motors, psi columns, geometry BEFORE any hydration — the
          headless side genuinely knows more).  For LIVE sources the panel is
          authoritative: ``LiveFrameSource`` optimistically advertises
          metadata/geometry, and a live run without real panel metadata or
          calibration must stay gated (its claims are advisory).
        * has_energy — panel truth both ways: BEAM_ENERGY is a run-required
          field and the run consumes the panel's energy resolution, never a
          source-side claim.
        * ``source_ready`` (what drives ``run_target=SOURCE``) stays the
          panel's wrangler-runnability answer: a configured label plus a live
          source or a positive wrangler frame count.  Capability truth for
          readiness rows and launchers is delegated; what a fresh Run can
          consume is still the wrangler's call (e.g. a processed ``.nxs``
          reports record-truth caps but runs through Reintegrate).
        """
        directory_config = self._controls_v2_container_index_config()
        directory_intent_ready = bool(
            directory_config is not None
            and directory_config[0].is_dir()
            and (
                int(frame_count or 0) > 0
                or bool(directory_config[1])
                or bool(live_unknown)
            )
        )
        source_ready = bool(source_label) and (
            bool(live_unknown)
            or int(frame_count or 0) > 0
            or directory_intent_ready
        )
        headless = self._controls_v2_headless_source_caps(
            source_label, live=bool(live_unknown))
        counted = bool(
            not live_unknown
            and (int(frame_count or 0) > 0 or directory_intent_ready)
        )
        if live_unknown:
            merged_metadata = bool(has_metadata)
            merged_motors = bool(has_motors)
            merged_geometry = bool(has_geometry)
            merged_psi = bool(has_psi_metadata)
        else:
            merged_metadata = bool(has_metadata or headless.has_metadata)
            merged_motors = bool(has_motors or headless.has_motors)
            merged_geometry = bool(has_geometry or headless.has_geometry)
            merged_psi = bool(has_psi_metadata or headless.has_psi_metadata)
        return (
            SourceCaps(
                has_frames=bool(headless.has_frames or counted),
                has_raw=bool(headless.has_raw or counted),
                raw_reachable=bool(
                    headless.raw_reachable or directory_intent_ready),
                has_metadata=merged_metadata,
                has_motors=merged_motors,
                has_energy=bool(has_energy),
                has_geometry=merged_geometry,
                has_psi_metadata=merged_psi,
            ),
            source_ready,
        )

    def _controls_v2_headless_source_caps(
            self, source_label: str, *, live: bool) -> SourceCaps:
        """The delegated ``describe_source_readiness`` call.

        A live source is described as ``SourceSpec(label, LIVE)`` — no file
        IO: the true-live escape hatch answers without opening anything, so
        live refresh stays cheap.  A non-live description opens and frame-0
        probes the source, so it is cached on (label, mtime) exactly like the
        wrangler frame count."""
        label = str(source_label or "")
        if not label:
            return SourceCaps()
        try:
            if live:
                return describe_source_readiness(
                    SourceSpec(label, SourceKind.LIVE))
            expanded = os.path.expanduser(label)
            if os.path.isdir(expanded):
                # DIR-2: a directory source is GUI-owned filtered-count
                # territory.  Describing it headlessly would walk / open the
                # contained HDF5 masters on every refresh — the exact
                # per-refresh container opens the I/O contract forbids.  The
                # wrangler's filtered count adds has_frames/has_raw evidence
                # (composition rule 1); a count never manufactures
                # raw_reachable (rule 2).
                return SourceCaps()
            cached = getattr(self, "_v2_source_caps_cache", None)
            if self._controls_v2_run_active():
                # Rule 5: an active run (including Pause) is never probed
                # synchronously from the GUI thread — reuse the same-label
                # snapshot, else answer conservatively without flapping.
                if cached is not None and cached[0][0] == label:
                    return cached[1]
                return SourceCaps()
            key = (label, self._controls_v2_source_identity_stamp(expanded))
            if cached is not None and cached[0] == key:
                return cached[1]
            retry = getattr(self, "_v2_source_caps_retry", None)
            if (retry is not None and retry[0] == key
                    and time.monotonic() < retry[1]):
                return SourceCaps()
            observation = observe_source_readiness(label)
            if not observation.definitive:
                self._v2_source_caps_retry = (
                    key,
                    time.monotonic() + float(getattr(
                        self, "_v2_source_caps_retry_delay", 2.0)),
                )
                return SourceCaps()
            caps = observation.caps
            self._v2_source_caps_retry = None
            self._v2_source_caps_cache = (key, caps)
            return caps
        except Exception:
            logger.debug(
                "Controls V2 headless source readiness failed", exc_info=True)
            return SourceCaps()

    @staticmethod
    def _controls_v2_source_identity_stamp(path):
        """R1-strength cheap cache identity: one ``stat`` plus the name-only
        owning adapter — ``(size, mtime_ns, adapter_id)``; ``None`` when the
        path cannot be statted.  Composition rules 2/3: cache identity is
        SOURCE identity (path + version stamp + adapter owner), invalidated
        on any of them changing, computed with zero file opens."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        try:
            # Importing the registry installs the built-in adapters.  Looking
            # up candidate ownership before that bootstrap made an unchanged
            # file's cache key flap from owner=None to owner=nexus_hdf5 on its
            # first readiness call.
            import xrd_tools.sources.registry  # noqa: F401, PLC0415
            from xrd_tools.sources.adapters import candidate_owner
            owner = candidate_owner(Path(path))
            adapter_id = getattr(owner, "id", None)
        except Exception:
            adapter_id = None
        return (int(st.st_size), int(st.st_mtime_ns), adapter_id)

    _v2_source_caps_retry_delay = 2.0

    def _controls_v2_loaded_scan_file(
            self, *, loaded_frame_count: int, labels) -> str:
        """Path of the ACTUALLY loaded processed scan, or ``""``.

        H18 fix for H5 finding 3 (pre-existing bug): a fresh widget's LiveScan
        is constructed around the scratch ``default.nxs`` placeholder
        (``_init_data_objects``); until something is genuinely loaded or
        processed (frames hydrated / publications present / ``data_file``
        re-pointed) that placeholder must not count as a loaded scan."""
        data_file = str(
            getattr(getattr(self, "scan", None), "data_file", "") or "")
        if not data_file:
            return ""
        if loaded_frame_count or labels:
            return data_file
        # H18-R1: with nothing genuinely loaded, a nonexistent (stale,
        # session-restored) path is not a loaded scan either — it must not
        # produce a LOADED_SCAN run target or optimistic result caps.
        try:
            if not os.path.exists(os.path.expanduser(data_file)):
                return ""
        except OSError:
            return ""
        scratch = str(
            getattr(self, "_controls_v2_scratch_data_file", "") or "")
        try:
            if scratch and (
                os.path.abspath(os.path.expanduser(data_file))
                == os.path.abspath(os.path.expanduser(scratch))
            ):
                return ""
        except Exception:
            logger.debug(
                "Controls V2 scratch data-file compare failed", exc_info=True)
        return data_file

    @staticmethod
    def _controls_v2_normalized_path(value) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            return os.path.normcase(os.path.realpath(
                os.path.abspath(os.path.expanduser(text))))
        except OSError:
            return ""

    def _controls_v2_selected_record_identities(self, loaded_scan_file) -> set:
        """H18-R9: STRONG identities that prove a publication belongs to the
        selected processed record — the record's own normalized path, plus
        its stored raw-source provenance (the resolved acquisition master,
        cached under the SAME full identity key as the record caps).  A raw
        source and its processed output may legitimately live in different
        directories; the provenance link is what proves they are the same
        acquisition record.  Unknown provenance contributes nothing (fail
        closed)."""
        path = self._controls_v2_normalized_path(loaded_scan_file)
        if not path:
            return set()
        identities = {path}
        ident_cache = getattr(self, "_v2_result_source_ident", None)
        if ident_cache is not None and ident_cache[1]:
            root = self._controls_v2_processed_source_root()
            key = (os.path.expanduser(str(loaded_scan_file)),
                   self._controls_v2_source_identity_stamp(
                       os.path.expanduser(str(loaded_scan_file))), root)
            if ident_cache[0] == key:
                identities.add(ident_cache[1])
        # During Run/Pause, the writer changes its own output stamp.  The
        # record path remains the same run identity, and manual browsing is
        # distinguishable because only outputs announced by THIS run are in
        # the frozen set.  Let the configured acquisition source prove only
        # those exact active outputs; an unrelated paused browse target stays
        # fail-closed.
        active_outputs = getattr(self, "_v2_active_result_paths", set())
        if self._controls_v2_run_active() and path in active_outputs:
            active_source = getattr(self, "_v2_active_source_identity", "")
            if active_source:
                identities.add(active_source)
        return identities

    def _controls_v2_resident_selected_raw(
            self, pubs, loaded_scan_file, source_label) -> bool:
        """H18-R4/R9: identity-qualified resident raw evidence.

        A recent publication may prove raw capability ONLY when (a) no manual
        browser rescope is pending (the outgoing scan's store must not be
        borrowed), (b) its normalized ``source_identity`` matches a STRONG
        identity of the selected record — the record path itself, or the
        record's stored raw-source provenance resolved under the current
        identity key (H18-R9: basename-derived stems are NOT record
        identity; same-stem records from different roots never match; with
        no record selected, the configured acquisition path/directory is the
        target), and (c) actual nonempty full-resolution pixels are resident
        (``view.raw``, or a live ``raw_ref`` whose ``map_raw``/``image`` is
        a real array).  A bare/lazy/dead reference proves a reference
        exists, never that it is reachable without a probe."""
        if not pubs:
            return False
        if getattr(getattr(self, "h5viewer", None),
                   "_browser_scan_reset_pending", False):
            return False
        dir_prefix = ""
        if loaded_scan_file:
            targets = self._controls_v2_selected_record_identities(
                loaded_scan_file)
        else:
            label = self._controls_v2_normalized_path(source_label)
            if not label:
                return False
            targets = {label}
            if os.path.isdir(label):
                dir_prefix = label + os.sep
        if not targets:
            return False
        for pub in pubs:
            ident = self._controls_v2_normalized_path(
                getattr(pub, "source_identity", ""))
            if not ident:
                continue
            target_dirs = tuple(
                target + os.sep for target in targets if os.path.isdir(target))
            if ident not in targets and not (
                    (dir_prefix and ident.startswith(dir_prefix))
                    or any(ident.startswith(prefix) for prefix in target_dirs)):
                continue
            raw = getattr(getattr(pub, "view", None), "raw", None)
            if raw is not None and getattr(raw, "size", 0):
                return True
            ref = getattr(pub, "raw_ref", None)
            if ref is not None:
                for attr in ("map_raw", "image"):
                    pixels = getattr(ref, attr, None)
                    if pixels is not None and getattr(pixels, "size", 0):
                        return True
        return False

    def _controls_v2_loaded_result_caps(self, loaded_scan_file: str):
        """Record-truth ``ResultCaps`` for the loaded processed scan, or None.

        H18: delegate to ``capabilities_for_processed`` over the record's own
        metadata instead of mirroring GUI hydration state (the record knows it
        has 1D results and a scan table before the load worker hydrates any
        viewer rows).  ``raw_reachable`` consults the headless frame-0 probe
        (H5 finding 2: the ``frames_record`` capability alone overstates
        reachability once the raw master is gone).  Cached on the R1-strength
        identity stamp (path + size + mtime_ns + adapter owner) PLUS the
        explicit source root (H18-R2); never touches the file during an
        active run — the record is being appended and the panel is locked
        anyway.  A transient read failure is never cached as stable truth
        (H18-R3): a previous valid same-identity snapshot keeps answering
        where safe, and the read retries after a bounded debounce."""
        if not loaded_scan_file:
            return None
        path = os.path.expanduser(str(loaded_scan_file))
        if not os.path.isfile(path):
            return None
        root = self._controls_v2_processed_source_root()
        key = (path, self._controls_v2_source_identity_stamp(path), root)
        cached = getattr(self, "_v2_result_caps_cache", None)
        if cached is not None and cached[0] == key and cached[1] is not None:
            return cached[1]
        # H18-R6: a previous snapshot may answer ONLY on a FULL identity-key
        # match (path + version stamp + adapter owner + normalized root) —
        # that case is the cache hit above.  A changed identity (replaced
        # record at the same pathname, root change) is conservatively
        # pending/unavailable; resident browse evidence may still prove the
        # capability at the composition layer.
        if self._controls_v2_run_active():
            return None
        retry = getattr(self, "_v2_result_caps_retry", None)
        if (retry is not None and retry[0] == key
                and time.monotonic() < retry[1]):
            # H18-R3 debounce window: don't hammer a hiccuping share.
            return None
        try:
            from xrd_tools.io.read import get_metadata

            # H18-R2: raw reachability resolves through the operator-owned
            # explicit source root via the existing SourceSpec seam (N1
            # precedence: source_root > @source_base > scan directory).
            spec = SourceSpec(
                path, SourceKind.PROCESSED_NEXUS,
                options=({"source_root": root} if root else {}))
            # H18-R7: the record read is a capability observation — expected
            # energy/wavelength absence stays quiet and contextual.
            with quiet_capability_observation(
                    path, "selected processed result"):
                metadata = get_metadata(path)
            # H18-R5: the typed probe seam — a transient probe error is NOT a
            # definitive unreachable observation; treat it exactly like a
            # transient metadata failure (debounce + retry, never cached).
            observation = observe_raw_reachability(spec)
            if not observation.definitive:
                raise _TransientReadinessObservation(observation.detail)
            caps = capabilities_for_processed(
                metadata, raw_reachable=observation.reachable)
            # H18-R9: cache the record's stored raw-source provenance under
            # the SAME identity key — the strong identity that lets a live
            # acquisition master's resident publication prove ITS OWN
            # processed output across directories.  Resolution failure
            # contributes nothing (fail closed).
            try:
                from xrd_tools.io.read import observe_resolved_raw_source

                with quiet_capability_observation(
                        path, "selected processed result"):
                    provenance = observe_resolved_raw_source(
                        path, source_root=root or None)
                if not provenance.definitive:
                    raise _TransientReadinessObservation(provenance.detail)
                raw_src = provenance.source
            except _TransientReadinessObservation:
                raise
            except Exception:
                raw_src = None
            self._v2_result_source_ident = (
                key,
                self._controls_v2_normalized_path(raw_src) if raw_src else "")
        except Exception:
            logger.debug(
                "Controls V2 loaded-scan result caps failed", exc_info=True)
            # H18-R3/R6: never store the failure and never answer with a
            # different identity's snapshot; schedule a bounded retry.
            self._v2_result_caps_retry = (
                key,
                time.monotonic() + float(getattr(
                    self, "_v2_result_caps_retry_delay", 2.0)))
            return None
        self._v2_result_caps_retry = None
        self._v2_result_caps_cache = (key, caps)
        return caps

    #: bounded debounce (seconds) before a transiently failed processed-record
    #: read is retried (H18-R3); tests may lower it.
    _v2_result_caps_retry_delay = 2.0

    def _controls_v2_processed_source_root(self) -> str:
        """Explicit operator-owned source root for processed-record raw
        resolution (H18-R2).  GUI ownership rule: the configured Project
        Folder is the explicit ``source_root`` for records browsed in this
        widget — the frozen acquisition root is never substituted for an
        unrelated browsed scan.  Empty or invalid configures no override
        (the reader falls back to ``@source_base`` then the scan directory,
        the accepted N1 precedence)."""
        root = str(getattr(
            getattr(self, "wrangler", None), "project_folder", "") or "")
        root = os.path.expanduser(root.strip())
        if root and os.path.isdir(root):
            return os.path.normpath(os.path.abspath(root))
        return ""

    def _controls_v2_source_label(self) -> str:
        return self._controls_v2_configured_source_label()

    def _controls_v2_configured_source_label(self) -> str:
        source_type = ""
        param = self._controls_v2_param(("Signal", "inp_type"))
        if param is not None:
            try:
                source_type = str(param.value() or "")
            except Exception:
                source_type = ""

        source_paths = [("NeXus File", "nexus_file")]
        if source_type == "Image Directory":
            source_paths.append(("Signal", "img_dir"))
        else:
            source_paths.append(("Signal", "File"))
        source_paths.extend((("Signal", "File"), ("Signal", "img_dir")))

        seen = set()
        candidates = []
        for path in source_paths:
            if path in seen:
                continue
            seen.add(path)
            param = self._controls_v2_param(path)
            if param is None:
                continue
            try:
                candidates.append(param.value())
            except Exception:
                pass

        wrangler = getattr(self, "wrangler", None)
        candidates.extend((
            getattr(wrangler, "img_file", None),
            getattr(wrangler, "img_dir", None),
            getattr(wrangler, "nexus_file", None),
        ))
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return ""

    def _controls_v2_container_index_config(self):
        """Return the one authoritative container-directory configuration."""
        if (not self._controls_v2_enabled()
                or getattr(self, "_controls_v2_source_widget", None) is None):
            return None
        source_type = str(
            self._controls_v2_param_value(("Signal", "inp_type")) or "")
        ext = str(
            self._controls_v2_param_value(("Signal", "img_ext")) or ""
        ).lstrip(".").lower()
        if source_type != "Image Directory" or ext not in {
                "h5", "hdf5", "nxs"}:
            return None
        root_text = str(
            self._controls_v2_param_value(("Signal", "img_dir")) or ""
        ).strip()
        if not root_text:
            return None
        root = Path(root_text).expanduser()
        recursive = bool(self._controls_v2_param_value(
            ("Signal", "include_subdir"), False))
        name_filter = str(self._controls_v2_param_value(
            ("Signal", "Filter")) or "") or None
        if ext == "h5":
            suffixes = ("_master.h5",)
        elif ext == "hdf5":
            suffixes = ("_master.hdf5", "_master.h5")
        else:
            suffixes = (".nxs",)
        return root, recursive, name_filter, suffixes

    def _controls_v2_live_watch_armable(self) -> bool:
        """Whether Live may be selected before a container is READY.

        The Source card owns this decision: a supported container-directory
        configuration with an existing root can be armed while empty or while
        its first file is still provisional.  Run readiness remains separate
        and turns true only after Live is selected and the remaining setup is
        valid.
        """
        config = self._controls_v2_container_index_config()
        return bool(config is not None and config[0].is_dir())

    def _sync_controls_v2_source_index(self) -> None:
        widget = getattr(self, "_controls_v2_source_widget", None)
        panel = getattr(self, "controls_v2", None)
        if widget is None or panel is None:
            return
        config = self._controls_v2_container_index_config()
        panel.set_source_widget(widget, visible=config is not None)
        if config is None or not config[0].is_dir():
            widget.clear_directory()
            self._controls_v2_directory_observation = None
            return
        root, recursive, name_filter, suffixes = config
        widget.configure_directory(
            root, recursive=False, name_filter=name_filter,
            suffixes=suffixes, subdirs_lazy=recursive)

    def _on_controls_v2_directory_observation(self, observation) -> None:
        self._controls_v2_directory_observation = observation
        self._v2_frame_count_cache = None
        # Metadata/energy caches are already keyed by source identity. A
        # converged directory poll must not discard them every cadence tick.
        self._refresh_controls_v2_profile(immediate=False)

    def _controls_v2_current_directory_observation(self):
        observation = getattr(
            self, "_controls_v2_directory_observation", None)
        widget = getattr(self, "_controls_v2_source_widget", None)
        session = getattr(widget, "directory_session", None)
        config = self._controls_v2_container_index_config()
        if observation is None or session is None or config is None:
            return None
        desired = session.configured
        root, recursive, name_filter, suffixes = config
        if desired is None or (
            desired.root != root
            or desired.recursive
            or desired.name_filter != name_filter
            or desired.suffixes != suffixes
            or bool(getattr(widget, "directory_subdirs_lazy", False))
            != recursive
            or observation.request_generation != session.request_generation
        ):
            return None
        return observation

    def _controls_v2_freeze_source_run_authority(self):
        """Compatibility seam: directory membership is now worker-owned.

        Source selection freezes only :class:`DirectorySourceSpec`; it never
        freezes a content-classified READY plan.
        """
        return None, 0

    def _controls_v2_freeze_source_run_plan(self):
        """Compatibility accessor for callers that need only the value plan."""
        return self._controls_v2_freeze_source_run_authority()[0]

    def _controls_v2_freeze_source_spec(self):
        """Freeze typed mode-specific source membership for the next Run."""
        source_type = str(
            self._controls_v2_param_value(("Signal", "inp_type")) or "")
        if source_type == "Image Directory":
            config = self._controls_v2_container_index_config()
            if config is not None:
                from xrd_tools.sources import DirectorySourceSpec

                root, recursive, name_filter, suffixes = config
                widget = getattr(self, "_controls_v2_source_widget", None)
                session = getattr(widget, "directory_session", None)
                return DirectorySourceSpec(
                    root=root,
                    recursive=recursive,
                    suffixes=suffixes,
                    name_filter=name_filter,
                    generation=int(getattr(
                        session, "request_generation", 0) or 0),
                )
        selected = str(
            self._controls_v2_param_value(("Signal", "File")) or "").strip()
        selected_ext = Path(selected).suffix.lstrip(".").lower()
        if (
            source_type == "Image Series"
            and selected
            and selected_ext not in {"h5", "hdf5", "nxs"}
        ):
            from xrd_tools.sources import image_series_spec
            return image_series_spec(selected)
        if source_type == "Single Image" and selected:
            from xrd_tools.core.scan import SourceKind, SourceSpec
            return SourceSpec(selected, SourceKind.IMAGE_FILE)
        return None

    def _controls_v2_freeze_container_frame_counts(self, source_plan=None):
        """Copy the lazy, stamp-qualified container counts for one Run.

        This is value-only optimization evidence, not a second source index.
        The processing worker compares each entry with the exact Candidate it
        is about to consume and falls back to the normal raw open on any miss
        or stamp mismatch.
        """
        frozen = (
            source_plan.frame_count_snapshot(
                finalized_only=True, require_self_contained=True)
            if source_plan is not None else {}
        )
        memo = getattr(self, "_v2_container_final_count_memo", None) or {}
        for path, entry in memo.items():
            try:
                stamp, count = entry
                stamp = (int(stamp[0]), int(stamp[1]))
                count = int(count)
            except (TypeError, ValueError, IndexError):
                continue
            if count >= 0:
                frozen.setdefault(str(path), (stamp, count))
        return frozen

    def _controls_v2_source_frame_count(self) -> int | None:
        """Images the configured raw source will yield; ``None`` for live."""

        source_type = str(
            self._controls_v2_param_value(("Signal", "inp_type")) or ""
        )
        img_file = str(self._controls_v2_param_value(("Signal", "File")) or "")
        img_dir = str(self._controls_v2_param_value(("Signal", "img_dir")) or "")
        nexus_file = str(
            self._controls_v2_param_value(("NeXus File", "nexus_file")) or ""
        )
        img_ext = str(self._controls_v2_param_value(("Signal", "img_ext")) or "")
        effective_ext = (
            Path(img_file).suffix.lstrip(".").lower()
            if source_type == "Image Series" and img_file
            else img_ext.lstrip(".").lower()
        )
        include_subdir = bool(
            self._controls_v2_param_value(("Signal", "include_subdir"), False)
        )
        file_filter = str(self._controls_v2_param_value(("Signal", "Filter")) or "")
        source_path = (
            img_dir
            if source_type == "Image Directory"
            else img_file or nexus_file
        )
        if not source_path:
            return 0
        if self._controls_v2_live_source_active():
            return None

        source_stamp = self._controls_v2_source_cache_stamp(source_path)
        if (
            source_type == "Image Series"
            and effective_ext not in {"h5", "hdf5", "nxs"}
            and img_file
        ):
            # Series membership is a property of the selected file *and* its
            # sibling directory.  This makes a landed/removed member invalidate
            # the cache while avoiding a directory walk on every profile paint.
            source_stamp = (
                source_stamp,
                self._controls_v2_source_cache_stamp(Path(img_file).parent),
            )
        key = (
            source_type,
            img_file,
            img_dir,
            nexus_file,
            effective_ext,
            include_subdir,
            file_filter,
            source_stamp,
        )
        cached = getattr(self, "_v2_frame_count_cache", None)
        if cached is not None and cached[0] == key:
            self._v2_source_count_is_files = bool(
                cached[2] if len(cached) > 2 else False)
            return cached[1]
        _run_active = getattr(self, "_controls_v2_run_active", None)
        if cached is not None and callable(_run_active) and _run_active():
            # H18 rule 5: an active run (including Pause) is never probed
            # synchronously from the GUI thread.  A mid-run version-stamp
            # change (the writer/detector appending to the very source being
            # consumed) must not trigger a fresh count/open — the run-start
            # snapshot answers until the run ends; the config half of the key
            # cannot change while the plan is frozen.
            if cached[0][:7] == key[:7]:
                self._v2_source_count_is_files = bool(
                    cached[2] if len(cached) > 2 else False)
                return cached[1]
        self._v2_source_count_is_files = False

        if (
            source_type == "Image Series"
            and effective_ext not in {"h5", "hdf5", "nxs"}
        ):
            source_spec = self._controls_v2_freeze_source_spec()
            if source_spec is not None:
                count = len(tuple(source_spec.options.get("files", ())))
                self._v2_frame_count_cache = (key, count, False)
                return count

        # Directory selection reports a direct-child FILE count only.  It does
        # not recurse when Subdirs is selected and never opens container
        # metadata.  Recursive membership and frame counts belong to Run-time.
        if source_type == "Image Directory":
            ext = effective_ext
            if ext in {"h5", "hdf5", "nxs"}:
                observation = self._controls_v2_current_directory_observation()
                if observation is None:
                    return 0
                count = len(observation.discovered_snapshot.candidates)
                self._v2_source_count_is_files = True
                self._v2_frame_count_cache = (key, count, True)
                return count

        # H18-R7: the frame count opens the container — a capability
        # observation; expected energy/wavelength absence stays quiet.
        with quiet_capability_observation(
                source_path, "configured acquisition source"):
            count = self._controls_v2_count_source_frames(
                source_type=source_type,
                img_file=img_file or nexus_file,
                img_dir=img_dir,
                img_ext=effective_ext,
                include_subdir=include_subdir,
                file_filter=file_filter,
            )
        self._v2_frame_count_cache = (key, count, False)
        return count

    @staticmethod
    def _v2_file_stamp(path):
        st = os.stat(path)
        return (int(st.st_size), int(st.st_mtime_ns))

    def _v2_memoized_frame_total(self, files):
        """Sum of memoized frame counts when EVERY file has a fresh entry
        (stamp-matching), else None.  Empty file lists stay None so a bare
        directory keeps the 0-files chip."""
        if not files:
            return None
        memo = getattr(self, "_v2_container_count_memo", None) or {}
        total = 0
        for path in files:
            hit = memo.get(str(path))
            if hit is None:
                return None
            try:
                if hit[0] != staticWidget._v2_file_stamp(path):
                    return None
            except OSError:
                return None
            total += int(hit[1])
        return total

    def _on_container_count_landed(
            self, path, nframes, stamp=None, authoritative=False) -> None:
        """GUI-thread landing for a container's frame count (from the run's
        sigContainerCount or the click-to-count sweep)."""
        if getattr(self, "_tearing_down", False):
            return
        if stamp is None:
            try:
                stamp = staticWidget._v2_file_stamp(path)
            except OSError:
                return
        try:
            stamp = (int(stamp[0]), int(stamp[1]))
            nframes = int(nframes)
        except (TypeError, ValueError, IndexError):
            return
        self._v2_container_count_memo[str(path)] = (stamp, int(nframes))
        if len(self._v2_container_count_memo) > 8192:
            self._v2_container_count_memo.clear()
        if authoritative:
            final_memo = getattr(
                self, "_v2_container_final_count_memo", None)
            if final_memo is None:
                final_memo = self._v2_container_final_count_memo = {}
            final_memo[str(path)] = (stamp, nframes)
            if len(final_memo) > 8192:
                final_memo.clear()
        self._v2_frame_count_cache = None
        if not self._controls_v2_run_active():
            self._refresh_controls_v2_profile(immediate=False)

    def _on_readiness_summary_clicked(self) -> None:
        """Directory summaries remain file counts.

        Container extents are inspected lazily by the processing worker.  A
        summary click must not turn an intentionally cheap Source projection
        into an eager recursive metadata sweep.
        """
        return

    def _controls_v2_kick_container_count_sweep(self) -> None:
        if getattr(self, "_v2_count_sweep_active", False):
            return
        source_type = str(
            self._controls_v2_param_value(("Signal", "inp_type")) or "")
        img_dir = str(self._controls_v2_param_value(("Signal", "img_dir")) or "")
        img_ext = str(self._controls_v2_param_value(("Signal", "img_ext")) or "")
        include_subdir = bool(
            self._controls_v2_param_value(("Signal", "include_subdir"), False))
        file_filter = str(
            self._controls_v2_param_value(("Signal", "Filter")) or "")
        ext = img_ext.lstrip(".").lower()
        if source_type != "Image Directory" or ext not in {"h5", "hdf5", "nxs"}:
            return
        observation = self._controls_v2_current_directory_observation()
        if observation is None:
            return
        files = tuple(
            candidate.path for candidate in observation.ready_snapshot.candidates)
        memo = self._v2_container_count_memo
        todo = []
        for path in files:
            hit = memo.get(str(path))
            try:
                if hit is not None and hit[0] == staticWidget._v2_file_stamp(path):
                    continue
            except OSError:
                continue
            todo.append(path)
        if not todo:
            self._refresh_controls_v2_profile(immediate=False)
            return
        self._v2_count_sweep_active = True

        def _sweep():
            from xrd_tools.io import image as image_io
            try:
                for path in todo:
                    try:
                        n = int(image_io.count_frames(path) or 0)
                    except Exception:
                        continue
                    try:
                        self._sigV2CountLanded.emit(str(path), n)
                    except RuntimeError:
                        return          # widget torn down mid-sweep
            finally:
                self._v2_count_sweep_active = False

        threading.Thread(target=_sweep, daemon=True,
                         name="v2-count-on-demand").start()

    @staticmethod
    def _controls_v2_source_cache_stamp(path) -> int | None:
        if not path:
            return None
        try:
            return Path(str(path)).expanduser().stat().st_mtime_ns
        except OSError:
            return None

    def _controls_v2_param_value(self, path, default=""):
        param = self._controls_v2_param(tuple(path))
        if param is None:
            return default
        try:
            return param.value()
        except Exception:
            return default

    def _controls_v2_live_source_active(self) -> bool:
        controls = getattr(self, "controls", None)
        try:
            if controls is not None and controls.is_live():
                return True
        except Exception:
            pass
        wrangler = getattr(self, "wrangler", None)
        if bool(getattr(wrangler, "live_mode", False)):
            return True
        return bool(getattr(getattr(wrangler, "thread", None), "live_mode", False))

    def _controls_v2_append_target_matches_displayed_scan(self) -> bool:
        wrangler = getattr(self, "wrangler", None)
        matcher = getattr(wrangler, "_append_target_matches_scan_file", None)
        if not callable(matcher):
            return False
        try:
            return bool(matcher(getattr(self, "scan", None), refresh_source=False))
        except Exception:
            logger.debug("Controls V2 append target match failed", exc_info=True)
            return False

    @classmethod
    def _controls_v2_count_source_frames(
            cls,
            *,
            source_type: str,
            img_file: str,
            img_dir: str,
            img_ext: str,
            include_subdir: bool,
            file_filter: str,
    ) -> int:
        from xrd_tools.io import image as image_io
        from .wranglers.image_wrangler_thread import _get_scan_info, _name_filter

        container_exts = {"h5", "hdf5", "nxs"}
        ext = str(img_ext or "").lstrip(".").lower()
        try:
            if source_type == "Image Directory":
                base = Path(str(img_dir or "")).expanduser()
                if not base.is_dir():
                    return 0
                match = _name_filter(file_filter)
                if ext in container_exts:
                    # DIR-2 (maintainer, 2026-07-13): container directories
                    # report the FILE count — never one HDF5 open per file.
                    files = cls._controls_v2_container_directory_files(
                        base, ext, include_subdir, match)
                    return len(files)
                pattern = f"*.{ext}" if ext else "*"
                candidates = (
                    base.rglob(pattern) if include_subdir else base.glob(pattern)
                )
                return sum(
                    1 for path in candidates
                    if path.is_file() and match(path.stem)
                )

            path = Path(str(img_file or "")).expanduser()
            if not path.is_file():
                return 0
            file_ext = (
                path.suffix.lstrip(".").lower()
                if source_type == "Image Series"
                else (ext or path.suffix.lstrip(".")).lower()
            )
            if source_type == "Image Series" and file_ext not in container_exts:
                scan_name, _img_number = _get_scan_info(path)
                import re
                series_re = re.compile(
                    rf"^{re.escape(scan_name)}_\d+\.{re.escape(file_ext)}$"
                )
                files = [
                    sibling for sibling in path.parent.glob(
                        f"{scan_name}_*.{file_ext}"
                    )
                    if series_re.match(sibling.name)
                ]
                return len(files) if files else int(
                    image_io.count_frames(path) or 0
                )
            return int(image_io.count_frames(path) or 0)
        except Exception:
            logger.debug("Controls V2 source frame count failed", exc_info=True)
            return 0

    @staticmethod
    def _controls_v2_container_directory_files(
            base: Path, ext: str, include_subdir: bool, match
    ) -> tuple[Path, ...]:
        if ext in {"h5", "hdf5"}:
            patterns = (("*_master.h5", "_master.h5"),)
            if ext == "hdf5":
                patterns = (("*_master.hdf5", "_master.hdf5"), *patterns)
        else:
            suffix = f".{ext}"
            patterns = ((f"*{suffix}", suffix),)

        files = []
        seen = set()
        for pattern, suffix in patterns:
            candidates = base.rglob(pattern) if include_subdir else base.glob(pattern)
            for path in candidates:
                key = str(path)
                if key in seen or not path.is_file():
                    continue
                seen.add(key)
                name = path.name[:-len(suffix)] if suffix else path.stem
                if match(name):
                    files.append(path)
        return tuple(sorted(files))

    def _controls_v2_calibrated(self) -> bool:
        return self._controls_v2_current_poni() is not None

    def _controls_v2_detector_summary(self) -> str:
        """Compact detector/PONI summary for the Experiment subsection header."""
        poni = self._controls_v2_current_poni()
        scan = getattr(self, "scan", None)
        integrator = getattr(scan, "_cached_integrator", None)

        detector = self._controls_v2_detector_name(
            getattr(poni, "detector", None))
        if not detector:
            detector = self._controls_v2_detector_name(
                getattr(getattr(integrator, "detector", None), "name", None))
        if not detector:
            detector = self._controls_v2_detector_name(
                getattr(integrator, "detector", None))

        dist_m = self._controls_v2_positive_float(getattr(poni, "dist", None))
        if dist_m is None:
            dist_m = self._controls_v2_positive_float(
                getattr(integrator, "dist", None))

        parts = []
        if detector:
            parts.append(detector)
        if dist_m is not None:
            parts.append(self._controls_v2_detector_distance_text(dist_m))
        if parts:
            parts.append("fitted")
        return " · ".join(parts)

    def _controls_v2_current_poni(self):
        scan = getattr(self, "scan", None)
        wrangler = getattr(self, "wrangler", None)
        poni_path = self._controls_v2_poni_path()
        if poni_path:
            for candidate in (
                getattr(wrangler, "poni", None),
                getattr(getattr(wrangler, "thread", None), "poni", None),
            ):
                if candidate is not None:
                    return candidate
            if os.path.exists(poni_path):
                try:
                    from xrd_tools.core.containers import PONI
                    return PONI.from_poni_file(poni_path)
                except Exception:
                    logger.debug("Controls V2 PONI summary load failed for %s",
                                 poni_path, exc_info=True)
            # A configured but invalid source calibration must not silently
            # display the PONI cached on an unrelated processed scan.
            return None

        for candidate in (
            getattr(scan, "_cached_poni", None),
            getattr(getattr(self, "integratorTree", None), "_cached_poni", None),
        ):
            if candidate is not None:
                return candidate
        return None

    def _controls_v2_poni_path(self) -> str:
        candidates = []
        for path in (("Signal", "poni_file"), ("Calibration", "poni_file")):
            param = self._controls_v2_param(path)
            if param is None:
                continue
            try:
                candidates.append(param.value())
            except Exception:
                pass
        wrangler = getattr(self, "wrangler", None)
        candidates.append(getattr(wrangler, "poni_file", ""))
        for candidate in candidates:
            if candidate:
                return str(candidate)
        return ""

    @staticmethod
    def _controls_v2_detector_name(value) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = (
                getattr(value, "name", None)
                or getattr(value, "alias", None)
                or getattr(value, "__class__", type(value)).__name__
            )
        name = str(value).strip()
        if name.lower() in {"", "none", "detector"}:
            return ""
        return name

    @staticmethod
    def _controls_v2_positive_float(value) -> float | None:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) and out > 0 else None

    @staticmethod
    def _controls_v2_detector_distance_text(dist_m: float) -> str:
        return f"{dist_m * 1000.0:.1f}mm"

    def _controls_v2_energy_known(self) -> bool:
        calibration_energy_eV, source_energy_eV = self._controls_v2_energy_values()
        return calibration_energy_eV is not None or source_energy_eV is not None

    def _controls_v2_energy_values(self) -> tuple[float | None, float | None]:
        """Return ``(calibration_energy_eV, source_energy_eV)`` for run gating.

        PONI wavelength is the default authority.  The native Source energy
        preference can intentionally privilege metadata for beamlines where the
        per-frame/source metadata is the more reliable energy record.
        """
        scan = getattr(self, "scan", None)
        poni_energy_eV = self._controls_v2_calibration_energy_eV(
            scan,
            poni=self._controls_v2_current_poni(),
        )
        metadata_energy_eV = self._controls_v2_metadata_energy_eV(scan)
        if self._controls_v2_energy_preference() == "metadata":
            if metadata_energy_eV is not None:
                return metadata_energy_eV, None
            return poni_energy_eV, None
        return poni_energy_eV, metadata_energy_eV

    @classmethod
    def _controls_v2_energy_from_wavelength(
            cls, value, *, allow_default_sentinel: bool = False
    ) -> float | None:
        wavelength_m = normalize_wavelength_m(
            value,
            allow_default_sentinel=allow_default_sentinel,
        )
        if wavelength_m is None:
            return None
        try:
            energy = float(wavelength_m_to_energy_eV(wavelength_m))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return None
        return energy if energy > 0 else None

    @classmethod
    def _controls_v2_calibration_energy_eV(cls, scan, *, poni=None) -> float | None:
        from_poni = cls._controls_v2_energy_from_wavelength(
            getattr(poni, "wavelength", None),
            allow_default_sentinel=True,
        )
        if from_poni is not None:
            return from_poni
        if scan is not None:
            persisted = cls._controls_v2_energy_from_wavelength(
                getattr(scan, "_persisted_wavelength_m", None),
                allow_default_sentinel=True,
            )
            if persisted is not None:
                return persisted
        if scan is None:
            return None
        integrator = getattr(scan, "_cached_integrator", None)
        from_integrator = cls._controls_v2_energy_from_wavelength(
            getattr(integrator, "wavelength", None))
        if from_integrator is not None:
            return from_integrator
        mg_args = getattr(scan, "mg_args", {}) or {}
        if isinstance(mg_args, dict):
            return cls._controls_v2_energy_from_wavelength(
                mg_args.get("wavelength"))
        return None

    @classmethod
    def _controls_v2_extract_energy_eV(cls, mapping) -> float | None:
        if not hasattr(mapping, "items"):
            return None
        try:
            lowered = {str(key).lower(): value for key, value in mapping.items()}
        except Exception:
            return None
        for key in (
            "energy_ev",
            "energyev",
            "beam_energy_ev",
            "source_energy_ev",
            "calibration_energy_ev",
        ):
            energy = cls._controls_v2_positive_float(lowered.get(key))
            if energy is not None:
                return energy
        for key in (
            "energy_kev",
            "energykev",
            "beam_energy_kev",
            "source_energy_kev",
        ):
            energy = cls._controls_v2_positive_float(lowered.get(key))
            if energy is not None:
                return energy * 1000.0
        return None

    @classmethod
    def _controls_v2_source_energy_eV(cls, scan) -> float | None:
        if scan is None:
            return None
        for attr in (
            "source_energy_eV",
            "beam_energy_eV",
            "energy_eV",
        ):
            energy = cls._controls_v2_positive_float(getattr(scan, attr, None))
            if energy is not None:
                return energy
        for attr in ("source_energy_keV", "beam_energy_keV", "energy_keV"):
            energy = cls._controls_v2_positive_float(getattr(scan, attr, None))
            if energy is not None:
                return energy * 1000.0
        for attr in ("metadata", "meta", "scan_info"):
            energy = cls._controls_v2_extract_energy_eV(getattr(scan, attr, None))
            if energy is not None:
                return energy
        return None

    def _controls_v2_metadata_energy_eV(self, scan) -> float | None:
        energy = self._controls_v2_source_energy_eV(scan)
        if energy is not None:
            return energy
        return self._controls_v2_configured_source_energy_eV()

    def _controls_v2_configured_source_energy_eV(self) -> float | None:
        meta_ext = str(
            self._controls_v2_param_value(("Signal", "meta_ext")) or ""
        ).strip()
        if not meta_ext or meta_ext == "None":
            return None
        img_file = self._controls_v2_first_metadata_file()
        if not img_file:
            return None
        meta_dir = str(
            self._controls_v2_param_value(("Signal", "meta_dir")) or ""
        ).strip()
        key = (str(img_file), meta_ext, meta_dir)
        cached = getattr(self, "_controls_v2_source_energy_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            from xrd_tools.io.metadata import read_image_metadata
            metadata = read_image_metadata(
                img_file,
                meta_format=meta_ext,
                meta_dir=meta_dir or None,
            )
            energy = self._controls_v2_extract_energy_eV(metadata)
        except Exception:
            logger.debug("Controls V2 metadata energy read failed",
                         exc_info=True)
            energy = None
        self._controls_v2_source_energy_cache = (key, energy)
        return energy

    def _controls_v2_first_metadata_file(self) -> str:
        source_type = str(
            self._controls_v2_param_value(("Signal", "inp_type")) or ""
        )
        if source_type == "Image Directory":
            current_observation = getattr(
                self, "_controls_v2_current_directory_observation", None)
            observation = (
                current_observation() if callable(current_observation) else None)
            if observation is not None:
                if observation.ready_snapshot.candidates:
                    return str(observation.ready_snapshot.candidates[0].path)
                return ""
            # A configured container directory is owned by the Source card.
            # Before its first observation lands, do not independently glob a
            # provisional file or a data sidecar as metadata authority.
            container_config = getattr(
                self, "_controls_v2_container_index_config", None)
            if (callable(container_config)
                    and container_config() is not None):
                return ""
            wrangler = getattr(self, "wrangler", None)
            img_file = str(getattr(wrangler, "img_file", "") or "")
            if img_file:
                return img_file
            img_dir = str(
                self._controls_v2_param_value(("Signal", "img_dir")) or ""
            )
            img_ext = str(
                self._controls_v2_param_value(("Signal", "img_ext")) or ""
            ).lstrip(".")
            if not img_dir or not img_ext:
                return ""
            try:
                from .wranglers.image_wrangler_thread import _name_filter
                filter_text = str(
                    self._controls_v2_param_value(("Signal", "Filter")) or "")
                match = _name_filter(filter_text)
                base = Path(img_dir).expanduser()
                include_subdir = bool(self._controls_v2_param_value(
                    ("Signal", "include_subdir"), False))
                pattern = f"*.{img_ext}"
                candidates = (
                    base.rglob(pattern) if include_subdir else base.glob(pattern)
                )
                cache_key = (
                    str(base),
                    img_ext,
                    filter_text,
                    include_subdir,
                    self._controls_v2_source_cache_stamp(base),
                )
                cached = getattr(self, "_controls_v2_metadata_probe_cache", None)
                if cached is not None and cached[0] == cache_key:
                    return cached[1]
                for path in candidates:
                    if path.is_file() and match(path.stem):
                        result = str(path)
                        self._controls_v2_metadata_probe_cache = (
                            cache_key, result)
                        return result
            except Exception:
                logger.debug("Controls V2 metadata source probe failed",
                             exc_info=True)
                return ""
            self._controls_v2_metadata_probe_cache = (cache_key, "")
            return ""
        img_file = str(
            self._controls_v2_param_value(("Signal", "File")) or ""
        )
        return img_file if img_file else str(
            getattr(getattr(self, "wrangler", None), "img_file", "") or ""
        )

    def _controls_v2_batch_run_active(self) -> bool:
        run_active = self._controls_v2_run_active()
        if not run_active:
            return False
        controls = getattr(self, "controls", None)
        try:
            if controls is not None and controls.current_mode() in (
                    "Image Viewer", "XYE Viewer", "NeXus Viewer"):
                return False
        except Exception:
            pass
        return True

    @staticmethod
    def _controls_v2_backend(tool) -> str | None:
        if tool == Tool.STITCH:
            return "multigeometry"
        if tool == Tool.RSM:
            return "rsm"
        return None

    def _build_tools_placeholder(self):
        """Fill the vacated bottom-left ``metaFrame`` with the compact 'Tools' card.

        Each tool is a full-width button labelled with the tool name; the hover
        tooltip carries the description (the old dot+label+Open rows and the
        wrapped note below took too much vertical space).  Clicking opens the
        tool's dialog.  Reclaims the corner freed by moving the metadata table
        into a popup."""
        lay = QtWidgets.QVBoxLayout(self.ui.metaFrame)
        lay.setContentsMargins(13, 9, 13, 10)
        lay.setSpacing(6)

        header = QtWidgets.QLabel('TOOLS')
        header.setObjectName('toolsHeader')
        lay.addWidget(header)

        card = QtWidgets.QFrame()
        card.setObjectName('toolsPlaceholder')
        card_lay = QtWidgets.QVBoxLayout(card)
        card_lay.setContentsMargins(9, 9, 9, 9)
        card_lay.setSpacing(6)
        # (symbol, label, handler-or-None, tooltip).  Handler => active tool; the
        # button opens it.  A None handler is a not-yet-built tool (button
        # disabled).  Symbols are decorative glyphs (standard-font safe).
        tools = [
            ('∧', 'Peak Fitting', self._open_peak_fit_dialog,
             'Structure-agnostic peak fitting — selected frame and across the scan.'),
            ('≈', 'Phase Fitting', self._open_phase_fit_dialog,
             'CIF-based phase fitting — selected frame and across the scan.'),
            ('▤', 'Plot Metadata', self._open_scan_plot_dialog,
             'Plot scan metadata + image-ROI statistics vs frame.'),
        ]
        for symbol, name, handler, tip in tools:
            btn = QtWidgets.QPushButton(f'{symbol}   {name}')
            btn.setObjectName('toolButton')
            btn.setToolTip(tip)
            if handler is not None:
                btn.clicked.connect(handler)
            else:
                btn.setEnabled(False)
            card_lay.addWidget(btn)
        lay.addWidget(card)
        lay.addStretch(1)

    def _pattern_for_frame(self, idx):
        """Return ``(x, y, x_label)`` for ONE frame's 1-D pattern, or ``None``.
        Reads the same data + axis unit the main 1-D plot shows.  Shared by the
        single-frame fit (selected frame) and batch (every frame)."""
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None
        try:
            # Block-and-read on a store MISS so a non-resident frame's 1D
            # actually loads (else the read returns None -> the popup lies "No
            # frame selected").  Idle-gated exactly like Set-Bkg
            # (display_frame_widget Set-Bkg precedent): during a run the live
            # push path feeds the dialog, so don't contend with the writer.
            ydata, xdata = self.displayframe.get_frames_int_1d(
                [idx], rv='all',
                allow_blocking_read=not getattr(
                    self.displayframe, '_processing_active', False))
        except Exception:
            logger.exception("peak-fit: get_frames_int_1d failed")
            return None
        if xdata is None or ydata is None:
            return None
        import numpy as np
        x = np.asarray(xdata)
        y = np.asarray(ydata)
        if y.ndim > 1:
            y = y[0]
        if x.size == 0 or y.size == 0:
            return None
        try:
            label = self.displayframe.ui.plotUnit.currentText()
        except Exception:
            label = 'q'
        return x, y, label

    def _current_pattern_for_fit(self):
        """Return ``(x, y, x_label)`` for the SELECTED frame's 1-D pattern, or
        ``None`` — so a fit always matches what the user is looking at.

        The authoritative "what the 1-D plot draws" is ``displayframe.idxs_1d``
        -- the very list ``get_frames_int_1d`` defaults to and ``set_data``
        populates on every frame show.  The staticWidget's own ``frame_ids`` is
        never populated in the real GUI, so reading it (or h5viewer.frame_ids,
        which isn't set on a manual frame click either) left the fit popup stuck
        on "No frame selected".  Prefer ``self.frame_ids`` (tests set it), then
        the drawn 1-D frames, then the h5viewer selection.
        """
        idxs = (getattr(self, 'frame_ids', None)
                or list(getattr(self.displayframe, 'idxs_1d', None) or [])
                or getattr(self.h5viewer, 'frame_ids', None) or [])
        if not idxs:
            return None
        return self._pattern_for_frame(idxs[0])

    def _analysis_context(self):
        """Stable analysis-data contract for popup tools.

        Popups consume this context rather than reading staticWidget,
        displayframe, wrangler, or integrator internals.  The providers still
        point at the current live/reloaded publication path, so live fitting
        continues to update on each processed frame.
        """
        from .analysis_context import AnalysisContext
        return AnalysisContext(
            current_pattern_provider=self._current_pattern_for_fit,
            frame_pattern_provider=self._pattern_for_frame,
            scan_uri_provider=self._current_scan_uri,
            mask_provider=self._scan_plot_mask_provider,
            read_lock_provider=self._scan_read_lock_provider,
            frame_labels_provider=lambda: tuple(
                getattr(self, 'frame_ids', ())
                or (getattr(self.displayframe, 'idxs_1d', None) or ())
                or getattr(self.h5viewer, 'frame_ids', ()) or ()),
            metadata_provider=lambda: {})

    def _open_peak_fit_dialog(self):
        """Open (or re-show) the Peak Fitting popup — lazy, single-instance,
        non-modal (so the live scan + frame browsing stay responsive; Reload
        re-grabs the current frame)."""
        if self._peak_fit_dialog is None:
            from .peak_fit_dialog import PeakFitDialog
            self._peak_fit_dialog = PeakFitDialog(
                analysis_context=self._analysis_context(), parent=self)
            # Toggling Live on re-fits the current frame at once (then every new
            # frame, via set_data); off just stops pushing.
            self._peak_fit_dialog.live_check.toggled.connect(
                self._on_live_fit_toggled)
            self._peak_fit_dialog.batch_btn.clicked.connect(
                lambda: self._on_batch_clicked(self._peak_fit_dialog))
        dlg = self._peak_fit_dialog
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.refresh_pattern()

    def _on_live_fit_toggled(self, on):
        """Live checkbox flipped — fit the current frame immediately on enable so
        there's no wait for the next frame; disabling just stops the pushes."""
        if on:
            dlg = self._peak_fit_dialog
            if dlg is not None:
                dlg.reset_param_trend()      # fresh vs-frame trend for this run
            self._maybe_live_fit()

    def _ensure_live_analysis_worker(self):
        """Lazily create + start the latest-wins live analysis worker."""
        if self._live_analysis_worker is None:
            from .analysis_worker import LiveAnalysisWorker
            self._live_analysis_worker = LiveAnalysisWorker(self)
            self._live_analysis_worker.sigAnalyzed.connect(self._on_live_analyzed)
            self._live_analysis_worker.start()
        return self._live_analysis_worker

    def _maybe_live_fit(self):
        """If the Peak Fitting dialog is open with Live on, push the current
        frame's pattern to it and request a background re-fit (latest-wins).

        Called from ``set_data`` (per frame) and on Live-toggle.  The analyzer +
        request are built through the dialog's own ``build_fit_request`` so live
        and the manual Fit button behave identically."""
        dlg = self._peak_fit_dialog
        if dlg is None or not dlg.isVisible() or not dlg.live_check.isChecked():
            return
        data = self._analysis_context().current_pattern_tuple()
        if not data:
            return
        x, y, label = data
        dlg.set_live_pattern(x, y, label)   # show the data now; fit overlays async
        req = dlg.build_fit_request()
        if req is None:                      # nothing fittable (status set by dialog)
            return
        inp, analyzer = req
        # Label the analysis by the FRAME index (not the axis unit) so the
        # dialog keys the vs-frame trend by frame; build_fit_request defaults the
        # label to "current".
        idxs = getattr(self, 'frame_ids', None) or []
        inp.label = str(idxs[0]) if idxs else ""
        self._live_fit_gen += 1
        self._ensure_live_analysis_worker().request(
            inp.label, self._live_fit_gen, analyzer, inp)

    def _on_live_analyzed(self, label, generation, outcome):
        """Draw a live fit result — but only if it's still the newest request and
        the dialog is still open + Live (a stale or superseded result is dropped,
        so the overlay never lags behind the displayed frame)."""
        if self._tearing_down:
            return                              # widget closing — dialog may be gone
        if generation != self._live_fit_gen:
            return
        dlg = self._peak_fit_dialog
        if dlg is None or not dlg.isVisible() or not dlg.live_check.isChecked():
            return
        if outcome is not None and outcome.ok:
            dlg._draw_outcome(outcome, auto=dlg.auto_check.isChecked())

    # ---- Batch fit (Peak or Phase — dialog-parameterized) --------------
    def _on_batch_clicked(self, dialog):
        """Batch button: start a batch fit for ``dialog``, or cancel one in
        flight.  Shared by the Peak and Phase fitters."""
        worker = self._batch_analysis_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            return
        self._run_batch_fit(dialog)

    def _run_batch_fit(self, dialog):
        """Fit every frame in the scan with ``dialog``'s current settings, then
        plot the parameters vs frame number.

        The analyzer is fixed ONCE from the current frame (via the dialog's
        ``build_fit_request``) and applied to every frame, so each parameter
        series tracks the same thing across frames."""
        import numpy as np
        from xrd_tools.analysis.runner import AnalysisInput
        dlg = dialog
        if dlg is None:
            return
        if dlg._x is None or dlg._y is None:
            dlg.refresh_pattern()
        req = dlg.build_fit_request()
        if req is None:
            return                              # status set by the dialog
        _, analyzer = req
        lo, hi = dlg.batch_x_range()        # Peak: fit range; Phase: full extent
        try:
            frame_idxs = list(self.scan.frames.index)
        except Exception:
            frame_idxs = []
        if not frame_idxs:
            dlg.status.setText("No frames to batch-fit.")
            return
        x_unit = dlg._x_label
        inputs = []
        ctx = self._analysis_context()
        for idx in frame_idxs:
            data = ctx.pattern_tuple_for_frame(idx)
            if not data:
                continue
            fx, fy, _lbl = data
            fx = np.asarray(fx, dtype=float)
            fy = np.asarray(fy, dtype=float)
            mask = (np.isfinite(fx) & np.isfinite(fy)
                    & (fx >= lo) & (fx <= hi))
            if not np.any(mask):
                continue
            inputs.append(AnalysisInput(label=str(idx), x=fx[mask], y=fy[mask],
                                        x_unit=x_unit))
        if not inputs:
            dlg.status.setText("No fittable frames in the selected range.")
            return
        from .analysis_worker import BatchAnalysisWorker
        if self._batch_analysis_worker is None:
            self._batch_analysis_worker = BatchAnalysisWorker(self)
            self._batch_analysis_worker.sigProgress.connect(self._on_batch_progress)
            self._batch_analysis_worker.sigFrameFit.connect(self._on_batch_frame_fit)
            self._batch_analysis_worker.sigBatchDone.connect(self._on_batch_done)
        dlg.reset_param_trend()             # fresh vs-frame trend for this batch
        self._batch_dialog = dlg            # the slots route results back to it
        self._batch_analysis_worker.configure(analyzer, inputs)
        dlg.set_batch_running(True)
        dlg.set_batch_progress(0, len(inputs))
        self._batch_analysis_worker.start()

    def _on_batch_progress(self, done, total):
        if self._tearing_down:
            return
        dlg = self._batch_dialog
        if dlg is not None:
            dlg.set_batch_progress(done, total)

    def _on_batch_frame_fit(self, label, params):
        """A batch frame finished: grow the dialog's vs-frame trend (row 3)."""
        if self._tearing_down:
            return
        dlg = self._batch_dialog
        if dlg is None or not dlg.isVisible():
            return
        try:
            frame_idx = int(label)
        except (TypeError, ValueError):
            return
        dlg._accumulate_frame_params(frame_idx, params)

    def _on_batch_done(self, labels, columns):
        """Batch finished: re-enable the dialog (or report a cancel).  The
        vs-frame trend already filled row 3 incrementally via sigFrameFit."""
        if self._tearing_down:
            return
        dlg = self._batch_dialog
        if dlg is not None:
            dlg.set_batch_running(False)
        if labels is None:                      # cancelled before completion
            if dlg is not None:
                dlg.status.setText("Batch fit cancelled.")
            return
        if dlg is not None:
            dlg.status.setText(
                f"Batch fit done — {len(labels)} frames. Pick a parameter to "
                "track below; Save CSV to export.")

    def _open_phase_fit_dialog(self):
        """Open (or re-show) the Phase Fitting popup — lazy, single-instance,
        non-modal.  Shares the batch worker + vs-frame trend with Peak Fitting."""
        if self._phase_fit_dialog is None:
            from .phase_fit_dialog import PhaseFitDialog
            self._phase_fit_dialog = PhaseFitDialog(
                analysis_context=self._analysis_context(), parent=self)
            self._phase_fit_dialog.batch_btn.clicked.connect(
                lambda: self._on_batch_clicked(self._phase_fit_dialog))
        dlg = self._phase_fit_dialog
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        dlg.refresh_pattern()

    def _current_scan_uri(self):
        """Best-effort path to the currently-loaded scan (for Scan Plot's default
        source); None when nothing real is loaded (the dialog starts blank).

        Falls back to the active image wrangler's ``img_file`` so a RAW image
        series being browsed (e.g. a Bluesky ``.nxs`` whose per-frame metadata
        lives in the file itself) auto-loads into Scan Plot rather than opening
        blank when no processed scan has been saved yet."""
        import os
        candidates = [getattr(self.scan, 'data_file', None),
                      getattr(self, 'fname', None),
                      getattr(getattr(self, 'wrangler', None), 'img_file', None)]
        for cand in candidates:
            try:
                if cand and os.path.exists(str(cand)):
                    return str(cand)
            except (TypeError, ValueError):
                continue
        return None

    def _scan_plot_mask_provider(self, uri):
        """The loaded scan's static detector mask (``scan.global_mask``) — but
        ONLY when the Scan Plot's picked source IS that loaded scan.  An
        arbitrary other source has its own detector/geometry, so the loaded
        scan's mask must not be applied to it."""
        import os
        loaded = self._current_scan_uri()
        try:
            same = bool(loaded and uri and os.path.realpath(str(uri))
                        == os.path.realpath(str(loaded)))
        except (TypeError, ValueError):
            same = False
        return getattr(self.scan, 'global_mask', None) if same else None

    def _scan_read_lock_provider(self, uri):
        """The loaded scan's writer-coordinating ``file_lock`` — but ONLY when
        ``uri`` IS that scan's data file (the one file a live run's writer
        holds the lock around; ``_locked_scan_read`` is the display-side
        counterpart).  An arbitrary other file has no in-process writer to
        coordinate with, so dialogs read it unlocked (None)."""
        import os
        data_file = getattr(self.scan, 'data_file', None)
        try:
            same = bool(data_file and uri and os.path.realpath(str(uri))
                        == os.path.realpath(str(data_file)))
        except (TypeError, ValueError):
            same = False
        if not same:
            return None
        from .display_data import DisplayDataMixin
        return DisplayDataMixin._scan_file_lock(self)

    def _open_scan_plot_dialog(self):
        """Open (or re-show) the Scan Plot popup — lazy, single-instance,
        non-modal.  Starts on the currently-loaded scan (or blank)."""
        if self._scan_plot_dialog is None:
            from .scan_plot_dialog import ScanPlotDialog
            ctx = self._analysis_context()
            self._scan_plot_dialog = ScanPlotDialog(
                default_uri=ctx.current_scan_uri(),
                mask_provider=ctx.mask_for_scan_uri,
                lock_provider=ctx.read_lock_for_uri, parent=self)
        dlg = self._scan_plot_dialog
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _open_metadata_dialog(self):
        """Open (or re-show) the frame-metadata popup.

        Lazy, single-instance, NON-modal: built once on first click by reparenting
        the live ``self.metawidget`` into a ``QDialog`` so its table becomes
        visible.  Non-modal keeps the live scan + h5viewer frame selection
        responsive, and because the widget holds the shared frame_ids /
        publication store, it refreshes as you browse frames (its ``update()``
        is gated on ``tableview.isVisible()``, which is now exactly 'dialog
        open')."""
        if self._metadata_dialog is None:
            dlg = QDialog(self)
            dlg.setObjectName('metadataDialog')
            dlg.setWindowTitle('Frame metadata')
            dlg.resize(460, 460)
            dlg_lay = QtWidgets.QVBoxLayout(dlg)
            dlg_lay.setContentsMargins(0, 0, 0, 0)
            dlg_lay.addWidget(self.metawidget)
            self._metadata_dialog = dlg
        self._metadata_dialog.show()
        self._metadata_dialog.raise_()
        self._metadata_dialog.activateWindow()
        # The table only renders while visible; refresh now it is shown.
        self.metawidget.update()

    def _connect_signals(self):
        """Wire signal/slot connections for H5Viewer, DisplayFrame, and Integrator."""
        # H5Viewer signals
        self.h5viewer.sigUpdate.connect(self.set_data)
        self.h5viewer.file_thread.sigTaskStarted.connect(self.thread_state_changed)
        self.h5viewer.sigThreadFinished.connect(self.thread_state_changed)
        self.h5viewer.ui.listData.itemClicked.connect(self.disable_auto_last)
        self.h5viewer.ui.auto_last.clicked.connect(self.enable_auto_last)
        self.h5viewer.ui.auto_last.clicked.connect(self.latest_frame)
        # Stage 4: open the frame-metadata popup (local open-dialog connection).
        self.h5viewer.ui.metadata_btn.clicked.connect(self._open_metadata_dialog)

        # DisplayFrame signals.  (The "Update 2D" toggle was removed — 2D
        # now always renders.  The File ▸ Export menu actions still drive
        # the save_image / save_1D methods even though the in-panel Save
        # buttons are gone.)
        self.h5viewer.actionSaveImage.triggered.connect(self.displayframe.save_image)
        self.h5viewer.actionSaveArray.triggered.connect(self.displayframe.save_1D)
        # Plot method changes drive the H5 data list selection mode so
        # accumulating modes (Overlay/Waterfall/Sum/Average) auto-add
        # clicked points without requiring shift/ctrl.
        self.displayframe.sigPlotMethodChanged.connect(
            self.h5viewer.set_data_selection_mode)
        # Initialize once with the current plot method.
        self.h5viewer.set_data_selection_mode(
            self.displayframe.ui.plotMethod.currentText())
        # Viewer-mode Clear: also drop the file-list selection so the cleared
        # plot, the selection and the title agree (the displayframe reset the
        # title; the selection lives on the H5Viewer).
        self.displayframe.sigCleared.connect(self._on_display_cleared)

        # Integrator signals
        # GI on/off now lives on the integrator panel — route its toggle through
        # the same handler the wrangler's GI checkbox used (sets scan.gi +
        # refreshes the panel axis units/labels).
        self.integratorTree.sigUpdateGI.connect(self.update_scattering_geometry)
        # Make Mask just wrote a mask file — auto-populate the Mask File field.
        self.integratorTree.sigMaskCreated.connect(self._on_mask_created)
        self.integratorTree.integrator_thread.started.connect(self.thread_state_changed)
        # Re-integration is a "run" too: route its START through the single
        # run-state owner (task #68) — keeps the 2D panels persistent AND
        # disables the processing controls (task #71) for its duration; cleared
        # in integrator_thread_finished via _exit_run_state.
        self.integratorTree.integrator_thread.started.connect(self._enter_run_state)
        self.integratorTree.integrator_thread.update.connect(self.integrator_thread_update)
        self.integratorTree.integrator_thread.writeError.connect(
            self._show_reintegration_write_error)
        self.integratorTree.integrator_thread.finished.connect(self.integrator_thread_finished)
        # Advanced (re-homed from the wrangler's button onto the integrator's own
        # Reintegrate row): the single combined 1D+2D advanced-settings dialog.
        # Wired ONCE here — the integratorTree persists across wrangler swaps, so
        # there's no per-wrangler connect/disconnect dance to manage.
        if hasattr(self.integratorTree.ui, 'advanced_int'):
            self.integratorTree.ui.advanced_int.clicked.connect(
                self._show_integration_advanced)
        # Pixel rejection (Intensity Threshold + Mask Saturated) now lives in the
        # integrator panel and is read straight from its own
        # integratorTree.get_threshold_config() — for Reintegrate (the integrator
        # reads itself) and for live runs (injected into the wrangler at
        # run-setup; see _push_threshold_to_wrangler).

    def _show_reintegration_write_error(self, message: str) -> None:
        """Surface reintegration save failures in the same status area as runs."""
        try:
            self.wrangler.showLabel.emit(message)
        except Exception:
            logger.debug("could not surface reintegration write failure",
                         exc_info=True)

    @staticmethod
    def _mirror_wrangler_parameter_values(parameters, values) -> None:
        """Mirror value-only run config into the hidden compatibility tree.

        The tree's root signal owns ``imageWrangler.setup()``.  Blocking that
        root for the whole transaction makes these hidden parameters output
        adapters: a mirror cannot synchronously re-enter setup and reread a
        half-updated run configuration.

        R4B-9: blocking the ROOT alone is not enough.  Individual child
        Parameters carry their own direct ``sigValueChanged`` connections
        (``GI.th_motor``/``GI.th_val`` → ``set_gi_th_motor``, which writes
        ``wrangler.incidence_motor``).  Those fire synchronously on ``setValue``
        regardless of the root block, so each written child's own signals are
        blocked for the duration of its ``setValue`` too — the hidden tree stays
        a pure output adapter with no write-back into the run.
        """
        if parameters is None:
            return
        previous = parameters.blockSignals(True)
        child_blocked = []
        try:
            for path, value in values:
                try:
                    parameter = parameters.child(*path)
                    if parameter.value() != value:
                        prev_child = parameter.blockSignals(True)
                        child_blocked.append((parameter, prev_child))
                        parameter.setValue(value)
                except Exception:
                    # Wrangler schemas are intentionally heterogeneous.
                    continue
        finally:
            for parameter, prev_child in child_blocked:
                try:
                    parameter.blockSignals(prev_child)
                except Exception:
                    continue
            parameters.blockSignals(previous)

    def _push_threshold_to_wrangler(self, run_configuration=None):
        """Inject the native V2 pixel-rejection policy into the active
        wrangler setup params so live and reintegrate share one policy.

        Called from ``start_wrangler`` BEFORE ``wrangler.setup()`` (which reads
        those params and pushes them to the thread).  Per-field guarded: a
        wrangler without an 'Intensity Threshold' group (e.g. NeXus) just skips
        it, and still receives 'Mask Saturated'.
        """
        try:
            cfg = (
                run_configuration.threshold
                if run_configuration is not None
                else (
                    self._controls_v2_threshold_config()
                    if self._controls_v2_enabled()
                    else self.integratorTree.get_threshold_config()
                )
            )
        except Exception:
            # Fail LOUD: silently falling back means the LIVE run applies the
            # wrangler's default pixel-rejection instead of the integrator's
            # setting (a quiet live≠reintegrate divergence).
            logger.warning(
                "Could not read integrator threshold config; the live run will "
                "use the wrangler default pixel-rejection, which may differ from "
                "the integrator setting.", exc_info=True)
            return
        if cfg is None:
            return
        params = getattr(self.wrangler, 'parameters', None)
        if params is None:
            return

        self._mirror_wrangler_parameter_values(
            params,
            (
                (("Mask", "Threshold"), bool(cfg.apply_threshold)),
                (("Mask", "min"), cfg.threshold_min),
                (("Mask", "max"), cfg.threshold_max),
                (("MaskSat", "mask_sentinel"), bool(cfg.mask_saturation)),
            ),
        )

    def _push_gi_to_wrangler(self, run_configuration=None):
        """Inject the native V2 GI geometry into the active wrangler setup params."""
        try:
            if run_configuration is not None:
                frozen_gi = run_configuration.gi
                cfg = {
                    "gi": bool(frozen_gi.enabled),
                    "sample_orientation": int(
                        frozen_gi.sample_orientation),
                    "tilt_angle": float(frozen_gi.tilt_angle),
                    "incidence_motor": str(
                        frozen_gi.incidence_motor),
                    "th_val": float(frozen_gi.th_val),
                }
            else:
                cfg = (
                    self._controls_v2_gi_config()
                    if self._controls_v2_enabled()
                    else self.integratorTree.get_gi_config()
                )
        except Exception:
            # Fail LOUD: a silent fallback means the LIVE run uses the wrangler's
            # default GI geometry instead of the integrator's (quiet divergence).
            logger.warning(
                "Could not read integrator GI config; the live run will use the "
                "wrangler default GI geometry, which may differ from the "
                "integrator setting.", exc_info=True)
            return
        params = getattr(self.wrangler, 'parameters', None)
        if params is None or cfg is None:
            return

        self._mirror_wrangler_parameter_values(
            params,
            (
                (("GI", "Grazing"), bool(cfg["gi"])),
                (("GI", "sample_orientation"),
                 int(cfg["sample_orientation"])),
                (("GI", "tilt_angle"), float(cfg["tilt_angle"])),
                (("GI", "th_motor"), str(cfg["incidence_motor"])),
                (("GI", "th_val"), str(cfg["th_val"])),
            ),
        )

    def _init_wranglers(self):
        """Initialize the wrangler stack and select the default wrangler."""
        self.wrangler = wranglerWidget("uninitialized", threading.Condition())
        for name, w in wranglers.items():
            self.ui.wranglerStack.addWidget(
                w(self.fname, self.file_lock, self.scan)
            )
        self.ui.wranglerStack.currentChanged.connect(self.set_wrangler)
        self.command_queue = Queue()
        self.set_wrangler(self.ui.wranglerStack.currentIndex())

    _CONFIG_STATE_KEY = "_xdart_static_controls"

    def _augment_config_snapshot(self, document) -> None:
        """Add the active native-control state to a legacy config document."""

        if not isinstance(document, dict):
            return
        wrangler = getattr(self, "wrangler", None)
        params = getattr(wrangler, "parameters", None)
        if params is None:
            return
        active_name = str(params.name())
        native = self._controls_v2_int_session_state()
        gi_fields = self._controls_v2_gi_config()
        poni_file = self._controls_v2_poni_path()
        mode = self.controls.modeCombo.currentText()
        document[self._CONFIG_STATE_KEY] = {
            "schema_version": 1,
            "active_wrangler": active_name,
            "poni_file": poni_file,
            "processing_mode": mode,
            "controls_v2_int": native,
        }

        # Keep the active legacy tree truthful too, so the file degrades well
        # when opened by a pre-Controls-V2 xdart.  Do not mutate live QObject
        # parameters merely to save a file.
        outer = document.get(active_name)
        tree = outer.get(active_name) if isinstance(outer, dict) else None
        if not isinstance(tree, dict):
            return
        gi = tree.get("GI")
        if isinstance(gi, dict):
            gi["Grazing"] = bool(native.get("gi", False))
            gi["th_motor"] = str(gi_fields.get("incidence_motor", "Manual"))
            gi["th_val"] = gi_fields.get("th_val", 0.1)
            gi["sample_orientation"] = int(
                gi_fields.get("sample_orientation", 4))
            gi["tilt_angle"] = float(gi_fields.get("tilt_angle", 0.0))
        for group in ("Signal", "Calibration"):
            values = tree.get(group)
            if isinstance(values, dict) and "poni_file" in values:
                values["poni_file"] = poni_file
        threshold = native.get("threshold_config")
        mask = tree.get("Mask")
        if isinstance(threshold, dict) and isinstance(mask, dict):
            mask["Threshold"] = bool(threshold.get("apply_threshold", False))
            mask["min"] = threshold.get("threshold_min", 0.0)
            mask["max"] = threshold.get("threshold_max", 0.0)
        mask_sat = tree.get("MaskSat")
        if isinstance(threshold, dict) and isinstance(mask_sat, dict):
            mask_sat["mask_sentinel"] = bool(
                threshold.get("mask_saturation", True))

    def _activate_config_wrangler(self, name: str) -> None:
        stack = getattr(getattr(self, "ui", None), "wranglerStack", None)
        if stack is None:
            return
        for index in range(stack.count()):
            candidate = stack.widget(index)
            params = getattr(candidate, "parameters", None)
            if params is not None and str(params.name()) == str(name):
                if stack.currentIndex() != index:
                    stack.setCurrentIndex(index)
                return

    @staticmethod
    def _legacy_config_poni_file(document, active_name: str) -> str:
        """Return a PONI path from either historical wrangler schema."""
        if not isinstance(document, dict) or not active_name:
            return ""
        outer = document.get(active_name)
        if not isinstance(outer, dict):
            return ""
        tree = outer.get(active_name, outer)
        if not isinstance(tree, dict):
            return ""
        for group in ("Signal", "Calibration"):
            values = tree.get(group)
            value = values.get("poni_file") if isinstance(values, dict) else None
            if isinstance(value, str) and value:
                return value
        return ""

    def _apply_loaded_config_snapshot(self, document) -> None:
        """Commit a loaded config after every legacy tree has settled."""

        payload = (
            document.get(self._CONFIG_STATE_KEY)
            if isinstance(document, dict) else None
        )
        if isinstance(payload, dict):
            self._activate_config_wrangler(payload.get("active_wrangler", ""))
            poni_file = payload.get("poni_file")
            if isinstance(poni_file, str):
                self._set_poni_field(poni_file)
            self._apply_controls_v2_int_state(payload.get("controls_v2_int"))
            self._push_threshold_to_wrangler()
            self._push_gi_to_wrangler()
            mode = payload.get("processing_mode")
            if isinstance(mode, str):
                combo = self.controls.modeCombo
                index = combo.findText(mode)
                if index >= 0:
                    combo.setCurrentIndex(index)
        else:
            # Legacy files have only parameter trees.  Their active wrangler's
            # current Signal schema is applied by the parameter tree, while
            # older TIFF sessions stored the same field under Calibration.
            # Adopt either spelling explicitly so a preceding session cannot
            # leave its calibration cached.
            params = getattr(getattr(self, "wrangler", None), "parameters", None)
            try:
                active_name = str(params.name()) if params is not None else ""
            except Exception:
                active_name = ""
            legacy_poni = self._legacy_config_poni_file(document, active_name)
            if legacy_poni:
                self._set_poni_field(legacy_poni)
            try:
                gi = self.wrangler.parameters.child("GI").child("Grazing").value()
                self.update_scattering_geometry(bool(gi))
            except Exception:
                logger.debug("legacy config GI finalization skipped",
                             exc_info=True)
        self._configure_controls_v2_native_run_plan()
        self._refresh_controls_v2_profile(immediate=True)
        bump_run_config_debug_generation(self, "config")
        run_config_debug_log(
            logger,
            "config_snapshot_applied",
            widget=self,
            origin="config_load",
            native_snapshot=isinstance(payload, dict),
        )

    def _init_defaults_and_timer(self):
        """Set up default parameters and the coalescing update timer."""
        # Register all parameter trees with the defaultWidget
        parameters = [self.integratorTree.parameters]
        for i in range(self.ui.wranglerStack.count()):
            w = self.ui.wranglerStack.widget(i)
            parameters.append(w.parameters)
        self.h5viewer.defaultWidget.set_parameters(parameters)
        # §15.12-A.4: Config Save is an action owner — a pending invalid Controls
        # edit must veto the save (write no file), so give the defaultWidget a
        # synchronous pre-save hook that runs the checked commit first.
        self.h5viewer.defaultWidget._pre_save_veto = (
            self._controls_v2_config_save_veto)
        # §17.5: a veto hook that RAISES fails CLOSED at the save boundary; the
        # defaultWidget calls this owner hook to emit the structured refusal
        # event and surface a status message.
        self.h5viewer.defaultWidget._pre_save_veto_error = (
            self._controls_v2_config_save_veto_error)
        self.h5viewer.defaultWidget.sigConfigSaving.connect(
            self._augment_config_snapshot)
        self.h5viewer.defaultWidget.sigConfigLoaded.connect(
            self._apply_loaded_config_snapshot)

        # Single source of truth for "a wrangler/integrator run is in
        # progress" (task #68).  Flipped only by _enter_run_state /
        # _exit_run_state, which drive the display persist flag AND the
        # processing-control disable (task #71) so the two can never desync.
        self._run_active = False

        # Coalescing timer for wrangler updates: when the wrangler thread
        # processes images faster than the GUI can render, only the most
        # recent update is rendered at the configured flush interval.
        self._pending_update_idx = None
        # Per-frame display refresh: THROTTLE (not debounce) — a steady
        # live stream must still paint at the configured flush interval; the latest index
        # wins via _pending_update_idx (the shared coalescing idiom,
        # xdart.utils.throttle).
        # Live timers, both TERMINAL-TUNABLE for live sweeps (no rebuild):
        #   XDART_FLUSH_MS (default 150; floor 110) — the heavy image-update quantum.
        #     Keep it >= the 100 ms user-selection debounce and the median flush
        #     total (drain+list+render, ~70-90 ms) or the event loop re-saturates.
        #   XDART_LIST_MS  (default 60)  — the fast list/cursor refresh so the Frames
        #     list + auto-last cursor + status scroll continuously between renders
        #     (_flush_frame_list runs the LIGHT legs only: O(new) list refresh + a
        #     signal-blocked cursor advance — no drain, no render).
        # See docs/design/design_gui_liveness_jul2026.md.
        _flush_ms = staticWidget._timer_ms_from_env(
            "XDART_FLUSH_MS", 150, minimum=_LIVE_FLUSH_MIN_MS)
        _list_ms = staticWidget._timer_ms_from_env("XDART_LIST_MS", 60)
        if os.environ.get("XDART_PERF"):
            logger.info("[PERF] live timers: flush=%dms list=%dms", _flush_ms, _list_ms)
        # Main-thread liveness heartbeat (XDART_PERF only): a parented QTimer that
        # records the MAX event-loop gap seen during a run, logged at run end.  A
        # frozen GUI can't service the timer, so the tick-to-tick gap balloons to
        # the freeze duration -- turning "every handler measures fast but the GUI
        # is frozen" into a one-run diagnosis.  BB-1 survived multiple
        # instrumentation passes precisely because this metric did not exist.
        # Created ONLY under the env flag -> zero production behavior change.
        # See docs/design/design_gui_liveness_jul2026.md.
        self._perf_hb_timer = None
        self._perf_hb_active = False
        self._perf_hb_last = 0.0
        self._perf_hb_max_gap_ms = 0.0
        if os.environ.get("XDART_PERF"):
            self._perf_hb_timer = QtCore.QTimer(self)
            self._perf_hb_timer.setInterval(250)
            self._perf_hb_timer.timeout.connect(self._perf_heartbeat_tick)
            self._perf_hb_timer.start()
        self._update_timer = Coalescer(_flush_ms, mode="throttle", parent=self)
        self._update_timer.triggered.connect(self._flush_pending_update)
        self._list_timer = Coalescer(_list_ms, mode="throttle", parent=self)
        self._list_timer.triggered.connect(self._flush_frame_list)
        # Reintegrate gets its OWN throttle: bai_*_all (live, batch=1) fires a
        # per-frame `update` signal, and rendering each one synchronously floods
        # the GUI (esp. the 2D cake at ~hundreds-of-ms each) -> the whole GUI
        # freezes + paints nothing until the run ends.  Coalesce to ~5 Hz so the
        # display tracks progress smoothly, like the wrangler's update_data path.
        self._pending_reint_idx = None
        self._reint_update_timer = Coalescer(200, mode="throttle", parent=self)
        self._reint_update_timer.triggered.connect(self._flush_reintegrate_update)
        # A newly-created SMB directory can remain absent from one or more
        # cached listings. Keep retries on one parented, single-shot timer so
        # they cannot outlive the widget or multiply per frame.
        self._xye_refresh_timer = QtCore.QTimer(self)
        self._xye_refresh_timer.setSingleShot(True)
        self._xye_refresh_timer.timeout.connect(
            self._retry_pending_xye_output_refreshes)
        self._pending_xye_output_dirs = {}
        self._xye_refresh_retry_index = 0
        # Per-frame work is COALESCED off the GUI event loop: update_data only
        # POPs the freshly-integrated frame (cheap) into _pending_frames; the
        # heavy build/upsert/scan_data runs once per ~200 ms flush over ALL frames
        # stashed since the last flush.  Running it per frame on the GUI thread
        # flooded the event loop (esp. once lz4 removed gzip's accidental
        # write-throttle) and froze the GUI for the whole scan.  _scan_info_rows
        # accumulates metadata rows so scan_data is rebuilt as one DataFrame per
        # flush instead of an O(N^2) per-frame `sd.loc[idx] = ser` enlargement.
        self._pending_frames = {}
        self._scan_info_rows = {}
        self._restore_controls_v2_int_session_state()

    def _fit_controls_height(self):
        """Pin the bottom controls bar to its current content height.

        Recomputed (not set once) because StaticControls shows/hides rows after
        init -- the run row hides in viewer modes, mode-specific widgets toggle
        per profile -- so a height frozen from the initial sizeHint would leave
        slack (run row hidden) or clip (content grown).  Called at init and after
        every profile / mode change."""
        try:
            self.ui.controlsFrame.setFixedHeight(
                self.controls.sizeHint().height()
                + 2 * self.ui.controlsFrame.frameWidth())
        except Exception:
            logger.debug("fit controls height failed", exc_info=True)

    def _commit_shortcut_focus(self):
        """Commit focused editors before a command-key action reads controls."""
        app = QtWidgets.QApplication.instance()
        widget = app.focusWidget() if app is not None else None
        if widget is None:
            return
        try:
            if isinstance(widget, QtWidgets.QAbstractSpinBox):
                widget.interpretText()
                widget.clearFocus()
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.clearFocus()
        except Exception:
            logger.debug("shortcut focus commit failed", exc_info=True)
            return
        try:
            QtWidgets.QApplication.processEvents()
        except Exception:
            pass

    def shortcut_run_pause(self):
        """Cmd+R handler: press the shared Run/Pause/Resume button if enabled."""
        self._commit_shortcut_focus()
        button = getattr(getattr(self, "controls", None), "startButton", None)
        if button is not None and button.isEnabled():
            button.click()

    def shortcut_stop(self):
        """Cmd+Shift+C handler: press the shared Stop button if enabled."""
        button = getattr(getattr(self, "controls", None), "stopButton", None)
        if button is not None and button.isEnabled():
            button.click()

    def shortcut_toggle_write_mode(self):
        """Cmd+Shift+A handler: toggle Append/Replace if the mode row is open."""
        button = getattr(getattr(self, "controls", None), "writeModeButton", None)
        if button is not None and button.isEnabled():
            button.click()

    def shortcut_pin_slice_cut(self):
        """Cmd+P handler: pin the current slice cut when the display allows it."""
        self._commit_shortcut_focus()
        pin = getattr(getattr(self, "displayframe", None),
                      "pin_current_slice_cut", None)
        if callable(pin):
            pin()

    def shortcut_load_settings(self):
        """Cmd+O handler: route through the existing Config -> Load action."""
        action = getattr(getattr(self, "h5viewer", None), "actionLoadParams", None)
        if action is not None and action.isEnabled():
            action.trigger()

    def shortcut_save_settings(self):
        """Cmd+S handler: route through the existing Config -> Save action."""
        action = getattr(getattr(self, "h5viewer", None), "actionSaveParams", None)
        if action is not None and action.isEnabled():
            action.trigger()

    def set_wrangler(self, qint):
        """Sets the wrangler based on the selected item in the dropdown.
        Syncs the wrangler's attributes and wires signals as needed.

        args:
            qint: Qt int, index of the new wrangler
        """
        if 'wrangler' in self.__dict__:
            self.disconnect_wrangler()

        self.wrangler = self.ui.wranglerStack.widget(qint)
        # RR-1: the wrangler's _inputs_valid consults the host panel to decide
        # whether an empty image source is an armed H19 authoritative-directory
        # Live run (see imageWrangler._h19_empty_directory_live_run_ok).
        self.wrangler._h19_host = self
        self.wrangler.input_q = self.command_queue
        self.wrangler.fname = self.fname
        self.wrangler.file_lock = self.file_lock
        self.wrangler.publication_store = self.publication_store
        if hasattr(self.wrangler, "thread"):
            self.wrangler.thread.publication_store = self.publication_store
        self.wrangler.sigStart.connect(self.start_wrangler)
        if hasattr(self.wrangler, 'sigStitchRequested'):
            self.wrangler.sigStitchRequested.connect(self.start_stitch)
        self.wrangler.sigUpdateData.connect(self.update_data)
        self.wrangler.sigUpdateFile.connect(self.new_scan)
        self.wrangler.sigXyeOutputReady.connect(self._on_xye_output_ready)
        # self.wrangler.sigUpdateFrame.connect(self.new_frame)
        self.wrangler.sigUpdateGI.connect(self.update_scattering_geometry)
        # GI move (Stage B) / §13.6: the wrangler hands its available SPEC motor
        # columns to ONE static-widget owner as a source-qualified
        # GIMotorHydration.  The owner verifies the token/epoch and only THEN
        # updates the stored knowledge AND the integrator's GI-motor combo.  The
        # former DIRECT sigGIMotorOptions → integratorTree.set_gi_motor_options
        # connection is REMOVED (it could not reject a stale result before it
        # replaced the visible choices).
        if hasattr(self.wrangler, 'sigGIMotorOptions'):
            self.wrangler.sigGIMotorOptions.connect(
                self._on_gi_motor_options_changed)
            # §10/SW-7: the integrator combo mirrors the ACTIVE wrangler's motor
            # discovery — a swap must not leave the previous wrangler's list
            # showing.  Ask the new wrangler to re-announce what it already
            # knows through the same signal that populates the combo.  Gated on
            # a non-empty discovery: an empty announce would wipe session-
            # restored choices before this wrangler's own discovery fires.
            if (getattr(self.wrangler, 'motors', None)
                    and callable(getattr(self.wrangler,
                                         'set_gi_motor_options', None))):
                self.wrangler.set_gi_motor_options()
        # Live-watch status → readiness bar ("Waiting for new images…").  The
        # watching message is emitted by the THREAD's showLabel; connect both
        # levels once per signal OBJECT (set_wrangler re-runs on every swap;
        # tracking avoids duplicate connections and PySide's noisy
        # failed-disconnect warning).
        prev_status_sigs = getattr(self, '_v2_status_sigs_connected', ())
        status_sigs = tuple(
            sig for sig in (
                getattr(self.wrangler, 'showLabel', None),
                getattr(getattr(self.wrangler, 'thread', None),
                        'showLabel', None))
            if sig is not None)
        for sig in prev_status_sigs:
            if sig not in status_sigs:
                try:
                    sig.disconnect(self._on_wrangler_status_text)
                except (TypeError, RuntimeError):
                    pass
        for sig in status_sigs:
            if sig not in prev_status_sigs:
                sig.connect(self._on_wrangler_status_text)
        self._v2_status_sigs_connected = status_sigs
        # DIR-2 lazy convergence: the run's per-container frame counts
        # land in the memo as each file is opened/retired.  Track the
        # connected signal OBJECT so a swap back to the same wrangler
        # neither duplicates the connection nor trips PySide's
        # failed-disconnect warning.
        count_sig = getattr(getattr(self.wrangler, 'thread', None),
                            'sigContainerCount', None)
        prev_sig = getattr(self, '_v2_count_sig_connected', None)
        if prev_sig is not None and prev_sig is not count_sig:
            try:
                prev_sig.disconnect(self._on_container_count_landed)
            except (TypeError, RuntimeError):
                pass
            self._v2_count_sig_connected = None
        if count_sig is not None and prev_sig is not count_sig:
            count_sig.connect(self._on_container_count_landed)
            self._v2_count_sig_connected = count_sig
        self.wrangler.started.connect(self.thread_state_changed)
        self.wrangler.finished.connect(self.wrangler_finished)
        # Pause/Resume (Phase B): lift the freeze guard once paused (frozen at a
        # frame boundary), re-engage it just before resuming.  No-op for
        # wranglers that never emit these (nexus).
        self.wrangler.sigPaused.connect(self._on_run_paused)
        self.wrangler.sigResuming.connect(self._on_run_resuming)
        # CONTROLS: attach the shared run-controls to this wrangler, apply its
        # capability profile (mode items + Live/Batch/cores) and restore its
        # persisted mode with the combo's signals BLOCKED — so item population
        # can't fire a mode-change against a half-attached wrangler — then sync
        # the wrangler's mode flags once.  The staticWidget-level mode reaction
        # (_on_processing_mode_changed) is wired ONCE to the shared combo in
        # _init_child_widgets, so it isn't reconnected per wrangler here.
        prof = self.wrangler.controls_profile()
        self.wrangler.attach_controls(self.controls)
        _combo = self.controls.modeCombo
        _combo.blockSignals(True)
        self.controls.apply_profile(
            modes=prof.get('modes'), live=prof.get('live', False),
            batch=prof.get('batch', False), cores=prof.get('cores', True))
        _cur = prof.get('current')
        if _cur:
            _i = _combo.findText(_cur)
            if _i >= 0:
                _combo.setCurrentIndex(_i)
        _combo.blockSignals(False)
        self.wrangler._on_mode_changed(_combo.currentText())
        if hasattr(self.wrangler, 'sigViewerModeChanged'):
            self.wrangler.sigViewerModeChanged.connect(self._on_viewer_mode_changed)
            # Sync current viewer mode (may have been restored from session).
            # Defer to after show() so the QSplitter layout is established
            # before we collapse panels.
            vm = getattr(self.wrangler, 'viewer_mode', None)
            if vm is not None:
                QtCore.QTimer.singleShot(0, lambda v=vm: self._on_viewer_mode_changed(v))
        if hasattr(self.wrangler, 'sigStitchModeChanged'):
            self.wrangler.sigStitchModeChanged.connect(self._on_stitch_mode_changed)
        if hasattr(self.wrangler, 'sigSavePathChanged'):
            self.wrangler.sigSavePathChanged.connect(self._sync_h5viewer_save_dir)
        # Advanced is now re-homed onto the integrator's Reintegrate row
        # (advanced_int, wired once above) so there's exactly ONE Advanced
        # button.  Hide the wrangler's old advancedButton (kept in the .ui so
        # existing layouts/refs don't break) rather than wiring it.
        if hasattr(self.wrangler, 'ui') and hasattr(self.wrangler.ui, 'advancedButton'):
            self.wrangler.ui.advancedButton.hide()
        native_gi_cfg = (
            self._controls_v2_gi_config()
            if self._controls_v2_enabled()
            else None
        )
        self.wrangler.setup()
        self._connect_controls_v2_source_tree()
        self._sync_controls_v2_source_index()
        if native_gi_cfg is not None:
            self._controls_v2_apply_gi_config_to_scan(native_gi_cfg)
            self._push_gi_to_wrangler()
        self._configure_controls_v2_native_run_plan()
        self._sync_h5viewer_save_dir(getattr(self.wrangler, 'h5_dir', None))
        # currentTextChanged (above) only fires on a CHANGE, so seed the control
        # and display state once now.  This is especially important on a fresh
        # startup: the mode combo is restored while signals are blocked, so the
        # display must not wait for a later scan/run to learn whether the current
        # mode is 1D-only or 2D.
        self._on_processing_mode_changed(_combo.currentText())
        # E1/E2: modifier-free, plotMethod-aware overlay build for the XYE file
        # list (active only in xye mode).  Mouse presses go to the viewport, key
        # presses to the list widget — install on both.
        try:
            scans = self.h5viewer.ui.listScans
            # listScans persists across wrangler swaps, so REMOVE the prior
            # filter before installing a new one — otherwise repeated mode/source
            # swaps stack filters on the same widget (duplicate handling + a
            # small memory creep).
            prev = getattr(self, '_xye_input_filter', None)
            if prev is not None:
                try:
                    scans.viewport().removeEventFilter(prev)
                    scans.removeEventFilter(prev)
                except Exception:
                    logger.debug("removing prior XYE input filter failed",
                                 exc_info=True)
            self._xye_input_filter = _XyeOverlayInputFilter(
                scans,
                lambda: getattr(self.h5viewer, 'viewer_mode', None) == 'xye',
                lambda: self.displayframe.ui.plotMethod.currentText(),
            )
            scans.viewport().installEventFilter(self._xye_input_filter)
            scans.installEventFilter(self._xye_input_filter)
        except Exception:
            logger.debug("XYE overlay input filter install failed", exc_info=True)
        self.h5viewer.sigNewFile.connect(self.wrangler.set_fname)
        # The display-clearing cascade (axes rebuild + bkg clear + cache wipe +
        # A-Step FrameRecordStore reset) is DEFERRED on a manual browser .nxs select
        # in Int 1D/2D so the plots stay as-is until a frame is clicked; runs
        # immediately for the run path + viewers.
        self.h5viewer.sigNewFile.connect(self._on_new_file_display_reset)
        # Stage C (2-way sync): on a .nxs load, populate the integration panel
        # from the saved scan so it shows the saved settings + reintegrate
        # reproduces them.  (Re-added per wrangler swap, like the lines above.)
        self.h5viewer.sigNewFile.connect(self._hydrate_integrator_on_load)
        # self.h5viewer.sigNewFile.connect(self.disable_displayframe_update)
        # The freshly-attached profile may have shown/hidden run rows -> refit the
        # controls bar to the new content height.
        self._fit_controls_height()
        self._refresh_controls_v2_profile(immediate=True)

    def _on_new_file_display_reset(self, *args):
        """sigNewFile handler for the display-clearing cascade (axes rebuild + bkg
        clear + cache wipe).  DEFERRED on a manual browser .nxs select in Int 1D/2D
        (``h5viewer._browser_scan_reset_pending``) so the plots stay as-is until the
        user clicks a frame — ``set_data`` runs it then.  Runs immediately for the
        run path (new_scan's set_file(internal=True)) and viewer modes, which never
        set the flag."""
        # A new file supersedes any pending run-end overlay catch-up (the spec's
        # set_file/data_reset clear): sigNewFile rides every set_datafile, and
        # the deferred browser-select case must cancel too — so clear BEFORE the
        # early-return below.
        self._runend_catchup_token = None
        self._runend_generation = getattr(self, "_runend_generation", 0) + 1
        if getattr(self.h5viewer, "_browser_scan_reset_pending", False):
            return
        self.displayframe.set_axes()
        self._clear_frame_record_store()   # A-Step (Phase 5): reset the per-file store
        self.displayframe._clear_bkg()
        self.h5viewer.data_reset()

    def disconnect_wrangler(self):
        """Disconnects all signals attached the the current wrangler
        """
        import warnings
        # These signals belong to the wrangler being torn down — a bare
        # disconnect() is fine (the whole wrangler is going away).
        signals = [self.wrangler.sigStart,
                   self.wrangler.sigUpdateData,
                   self.wrangler.sigUpdateFile,
                   self.wrangler.sigXyeOutputReady,
                   self.wrangler.finished,
                   self.wrangler.sigPaused,
                   self.wrangler.sigResuming]
        if hasattr(self.wrangler, 'sigViewerModeChanged'):
            signals.append(self.wrangler.sigViewerModeChanged)
        if hasattr(self.wrangler, 'sigSavePathChanged'):
            signals.append(self.wrangler.sigSavePathChanged)
        if hasattr(self.wrangler, 'sigUpdateGI'):
            signals.append(self.wrangler.sigUpdateGI)
        if hasattr(self.wrangler, 'sigGIMotorOptions'):
            signals.append(self.wrangler.sigGIMotorOptions)
        # (Advanced is no longer wired to the wrangler button — it lives on the
        # integrator's persistent advanced_int now, so there's nothing per-wrangler
        # to disconnect here.)
        for signal in signals:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    signal.disconnect()
            except (TypeError, RuntimeError, SystemError) as e:
                logger.debug("Failed to disconnect signal: %s", e)
        # h5viewer.sigNewFile is on a PERSISTENT object (the viewer survives
        # wrangler swaps), so a bare .disconnect() would also drop any future /
        # other subscriber.  Disconnect ONLY the slots set_wrangler attached.
        for slot in (getattr(self.wrangler, 'set_fname', None),
                     self._on_new_file_display_reset,
                     self._hydrate_integrator_on_load):
            if slot is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self.h5viewer.sigNewFile.disconnect(slot)
            except (TypeError, RuntimeError, SystemError):
                pass
        # Release the shared run-controls so the next wrangler can re-alias them
        # cleanly (drops this wrangler's tracked signal connections).
        try:
            self.wrangler.detach_controls()
        except Exception:
            logger.debug("detach_controls failed", exc_info=True)

    def _on_processing_mode_changed(self, mode_text):
        """staticWidget-level reaction to a processing-mode change.

        Wired ONCE to the shared controls' mode combo (see _init_child_widgets),
        so it survives wrangler swaps.  Per-mode integration-control state +
        forcing the display out of any stuck viewer mode for a non-viewer mode.
        """
        # This slot is connected before the active wrangler's mode handler.  Do
        # not let display geometry depend on the wrangler updating scan.skip_2d
        # first, or a fresh startup can briefly use the previous/default mode and
        # show the opposite panel.  The selected mode text is the source of truth
        # for layout.
        self._sync_processing_mode_to_scan(mode_text)
        # Per-mode integration control enable/dim (C3/C4) — runs for every
        # processing-mode change, including the viewer modes.
        self._apply_integration_control_state()
        # The mode change may show/hide the run row (hidden in viewer modes) —
        # refit the controls bar height.  BEFORE the viewer-mode early return so
        # it runs for viewer modes too (that's exactly when the row hides).
        self._fit_controls_height()
        # Skip the rest when in viewer mode — set_viewer_display_mode controls
        # panels.
        if 'Viewer' in mode_text:
            self._refresh_controls_v2_profile()
            return
        # A non-viewer processing mode (Int 1D/2D, Int 1D (XYE)) must take the
        # display OUT of any viewer mode it is stuck in.  The wrangler's
        # sigViewerModeChanged is guarded by its own _prev_viewer_mode, which
        # misses the case where the display was auto-switched to XYE after an
        # Int 1D (XYE) batch (the wrangler's viewer_mode stayed None, so _prev
        # stays '' and no reset emits).  Force the display reset here so the
        # combo and display can't desync.
        if getattr(self.displayframe, 'viewer_mode', None) is not None:
            self._on_viewer_mode_changed('')
        self.displayframe._apply_1d_only_visibility()
        # Drop any visible/cached content from the previous mode, then reload the
        # current selection for the new processing mode.  Calling update() alone
        # can leave a stale image/cake or curve visible when the new mode needs
        # data that has not been loaded yet.
        self.displayframe.clear_display_state()
        self.displayframe.request_plot_autorange()
        self.h5viewer.data_changed()
        self._refresh_controls_v2_profile()

    @staticmethod
    def _mode_skips_2d(mode_text):
        """Return whether a processing-mode label represents a 1D-only run."""
        text = str(mode_text or '')
        if 'Viewer' in text:
            return False
        return ('1D' in text) and ('2D' not in text)

    def _sync_processing_mode_to_scan(self, mode_text):
        """Synchronize scan/display 1D-only state from the selected mode text."""
        skip_2d = self._mode_skips_2d(mode_text)
        targets = [
            getattr(self, 'scan', None),
            getattr(getattr(self, 'displayframe', None), 'scan', None),
            getattr(getattr(self, 'wrangler', None), 'scan', None),
        ]
        thread = getattr(getattr(self, 'wrangler', None), 'thread', None)
        if thread is not None:
            targets.append(getattr(thread, 'scan', None))
        for target in targets:
            if target is None:
                continue
            try:
                target.skip_2d = skip_2d
            except Exception:
                logger.debug("could not sync skip_2d for %r", target,
                             exc_info=True)

    def _sync_h5viewer_save_dir(self, path, *, refresh=True):
        """Point the Scans browser at the active processed-output directory."""
        if not path:
            return
        target_path = os.path.abspath(os.path.expanduser(str(path)))
        self._h5viewer_save_target = target_path
        # If the processed-data dir doesn't exist yet (fresh project, no run
        # has created it), browse the nearest existing ancestor -- typically
        # the project folder -- instead of an empty nonexistent path.  Once
        # the first run creates xdart_processed_data, the next save-path
        # signal re-points the browser at it.
        probe = target_path
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        if probe and os.path.isdir(probe):
            path = probe
        else:
            path = target_path
        self._h5viewer_save_fallback = (
            path if os.path.normcase(path) != os.path.normcase(target_path)
            else None
        )
        self.dirname = path
        self.h5viewer.dirname = path
        if refresh:
            self.h5viewer.update_scans()

    def _on_xye_output_ready(self, output_dir):
        """Refresh a visible XYE location after its first file is durable.

        ``sigUpdateFile`` announces a scan before ``save_1d`` creates
        ``<save-dir>/<scan-name>``.  Without this post-write edge, every scan
        except the last is discovered incidentally at the next scan boundary;
        the final folder remains absent until Pause/Refresh.  Do not redirect
        the browser when the user has navigated elsewhere during a live run.
        """
        if not output_dir:
            return
        try:
            output_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
            current_dir = os.path.abspath(os.path.expanduser(
                str(getattr(self.h5viewer, 'dirname', '') or '')
            ))
            output_key = os.path.normcase(output_dir)
            visible_dirs = {
                output_key,
                os.path.normcase(os.path.dirname(output_dir)),
            }
            current_key = os.path.normcase(current_dir)
            if current_key not in visible_dirs:
                output_parent = os.path.dirname(output_dir)
                target = getattr(self, '_h5viewer_save_target', None)
                fallback = getattr(self, '_h5viewer_save_fallback', None)
                follows_fresh_target = (
                    target and fallback
                    and os.path.normcase(output_parent) == os.path.normcase(target)
                    and current_key == os.path.normcase(fallback)
                )
                if not follows_fresh_target:
                    return
                # The first durable XYE created the intended root. Repoint only
                # when the browser is still at the exact fallback recorded by
                # _sync_h5viewer_save_dir; explicit user navigation always wins.
                self.dirname = output_parent
                self.h5viewer.dirname = output_parent
                self._h5viewer_save_fallback = None
            self._pending_xye_output_dirs[output_key] = output_dir
            was_retrying = self._xye_refresh_retry_index > 0
            self._xye_refresh_retry_index = 0
            # Throttle one-frame scan bursts into at most one rebuild per 50 ms
            # without starving a continuous stream. A new output also pulls a
            # long stale-listing retry back to this short initial edge.
            if was_retrying or not self._xye_refresh_timer.isActive():
                self._xye_refresh_timer.start(_XYE_REFRESH_COALESCE_MS)
        except Exception:
            logger.debug(
                'Could not refresh browser for XYE output %s', output_dir,
                exc_info=True,
            )

    def _retry_pending_xye_output_refreshes(self):
        """Owned-timer callback for a listing that was stale on SMB/NFS."""
        self._refresh_pending_xye_output_dirs()

    def _refresh_pending_xye_output_dirs(self):
        pending = getattr(self, '_pending_xye_output_dirs', None)
        if not pending:
            return
        current_dir = os.path.abspath(os.path.expanduser(
            str(getattr(self.h5viewer, 'dirname', '') or '')
        ))
        current_key = os.path.normcase(current_dir)

        # Navigation wins over background refresh. A later scan notification or
        # explicit Refresh will rebuild the location if the user returns.
        for key, output_dir in list(pending.items()):
            if current_key not in {
                    key, os.path.normcase(os.path.dirname(output_dir))}:
                pending.pop(key, None)
        if not pending:
            self._xye_refresh_timer.stop()
            self._xye_refresh_retry_index = 0
            return

        try:
            self.h5viewer.update_scans(preserve_selection=True)
        except Exception:
            logger.debug("XYE output directory refresh failed", exc_info=True)

        list_widget = self.h5viewer.ui.listScans
        listed = {
            list_widget.item(row).text()
            for row in range(list_widget.count())
        }
        viewer_mode = getattr(self.h5viewer, 'viewer_mode', None)
        for key, output_dir in list(pending.items()):
            if current_key == os.path.normcase(os.path.dirname(output_dir)):
                visible = os.path.basename(output_dir) + '/' in listed
            elif viewer_mode == 'xye':
                visible = any(name.lower().endswith('.xye') for name in listed)
            else:
                # Normal mode intentionally hides XYE files inside the folder;
                # the refresh itself is the only observable action there.
                visible = True
            if visible:
                pending.pop(key, None)

        if not pending:
            self._xye_refresh_timer.stop()
            self._xye_refresh_retry_index = 0
            return

        retry_index = self._xye_refresh_retry_index
        if retry_index >= len(_XYE_REFRESH_RETRY_MS):
            logger.warning(
                "XYE output folder not visible after bounded refresh retries: %s",
                ', '.join(sorted(pending.values())),
            )
            pending.clear()
            self._xye_refresh_timer.stop()
            self._xye_refresh_retry_index = 0
            return
        self._xye_refresh_retry_index += 1
        self._xye_refresh_timer.start(_XYE_REFRESH_RETRY_MS[retry_index])

    def thread_state_changed(self):
        """Called whenever a thread is started or finished.
        """
        return

    def update_data(self, idx):
        """Called by signal from wrangler when a new frame is processed.

        Instead of rendering immediately (which blocks the main thread
        and causes frame-skipping when the wrangler is faster than the
        GUI), we update the in-memory data structures and schedule a
        throttled display refresh via a short single-shot timer.  The
        timer is started only when it isn't already pending, so during
        a fast scan burst the display refreshes at roughly the timer
        interval (~200 ms) instead of waiting for the burst to settle.
        Each flush renders the most recently received index.

        Special case: ``idx == -1`` is the batch-complete signal —
        trigger a full display refresh without touching the h5 viewer.
        """
        if idx == -1:
            # Batch mode finished — just refresh the display
            self._pending_update_idx = -1
            self._update_timer.trigger()
            return

        # A real frame was processed this run (Append-mode feedback gate).
        self._run_saw_frame = True

        # Frame-driven scan boundary.  The wrangler's new_scan signal races the
        # frame stream (emitted from a different thread + Eiger prefetch read-ahead),
        # so detect a genuinely-new scan from the FRAME itself — its source_file
        # carries the scan identity — and rescope the panel BEFORE appending, so
        # THIS frame becomes #1 of the new scan and its own frames can never be
        # dropped (unlike reacting to the mis-timed new_scan signal).  PEEK the
        # frame (never pop — the pop below stays the sole consumer).  Skipped in
        # batch (which suppresses per-frame update_data) and for the very first
        # scan (name "null_main", whose new_scan reliably precedes its frames).
        published = getattr(self.wrangler, "thread", None)
        if published is not None and not getattr(published, "batch_mode", False):
            _peek = getattr(published, "_published_frames", {}).get(idx)
            if _peek is not None:
                _key = _scan_key_from_source(getattr(_peek, "source_file", ""))
                _cur = getattr(self.scan, "name", None)
                if _key and _key != _cur and _cur != "null_main":
                    # DIR-1 (bl17-2): paint the OUTGOING scan's complete frame
                    # list before the rescope clears it.  Per-frame updates
                    # only ARM the throttled list timer (no leading-edge
                    # fire), so a 5-frame container often bursts entirely
                    # inside one ~60 ms window and the boundary used to clear
                    # the index before the tail ever painted ("shows 1-2
                    # frames then jumps to the next scan").  LIGHT legs only
                    # (list + cursor, O(new)) — the heavy drain/render below
                    # stays Overlay/Waterfall-gated (re-saturating the GUI
                    # thread per ~1 s boundary is the failure mode the
                    # liveness arc engineered away).
                    try:
                        self.h5viewer.update_data(emit_update=False)
                        if self.h5viewer.auto_last:
                            self.latest_frame(emit_update=False)
                    except Exception:
                        logger.debug("pre-rescope list flush failed",
                                     exc_info=True)
                    # Directory scans can be shorter than the heavy-render cadence
                    # (often one frame per scan).  Publish + accumulate the outgoing
                    # scan before rescope clears its pending/store state; otherwise
                    # a fast boundary silently drops that scan from Overlay/Waterfall.
                    staticWidget._flush_overlay_before_frame_rescope(self)
                    self._rescope_frame_panel_to(_key, first_frame=_peek)
                    # Tell new_scan a frame ALREADY rescoped THIS run, so its late
                    # signal skips the (now-destructive) clear.  A consumed flag —
                    # NOT a name match — so a same-name re-run (where no frame has
                    # rescoped yet) still clears.
                    self._frame_driven_rescoped_pending = True

        # Per-frame mid-scan refresh.  Append idx to scan.frames.index
        # and bypass the file_thread.load_frame disk read by pulling the
        # freshly-integrated frame directly out of the wrangler's in-memory
        # publication slot (see wrangler_thread._published_frames).
        # The frame contains map_raw, mask, int_1d, int_2d, gi_2d, etc.
        # All we need for the displayframe — no disk hit, no file-lock
        # contention with the wrangler's per-frame write.
        try:
            # Guard the in-memory index mutation with the scan's own
            # lock — the file-thread (load_frames / set_datafile) also
            # touches this list, and h5viewer.update_data reads it on the
            # GUI thread.  This is the GUI scan's lock, distinct from
            # the wrangler scan's, so it never contends with the
            # wrangler's disk writes (no GUI stall).
            with self.scan.scan_lock:
                index = self.scan.frames.index
                if idx not in index:
                    # Common case — frames arrive in order: append without
                    # the O(N log N) re-sort.  Only out-of-order inserts
                    # (rare: reload/replace) pay for a sort, so a long scan
                    # stays O(1) per frame instead of O(N log N).
                    if not index or idx > index[-1]:
                        index.append(idx)
                    else:
                        index.append(idx)
                        index.sort()
        except AttributeError:
            # frames may briefly be None or replaced during set_datafile.
            pass

        # Per-frame: POP the freshly-integrated frame from the wrangler's slot
        # (cheap -- just moves the reference) and stash it for the coalesced flush.
        # The heavy build/upsert/scan_data used to run HERE on the GUI thread for
        # EVERY frame; once lz4 removed gzip's accidental write-throttle that
        # flooded the event loop and froze the GUI for the whole scan.  Now
        # _drain_pending_frames does it at ~5/sec over all stashed frames.  Pop
        # drains the wrangler slot so frames can't leak there.
        published = getattr(self.wrangler, "thread", None)
        if published is not None:
            frame = getattr(published, "_published_frames", {}).pop(idx, None)
            if frame is not None:
                self._pending_frames[idx] = frame

        # P4: per-frame the *only* thing we do is remember the latest
        # idx + restart the coalescing timer.  The heavy list-widget
        # rebuild (``h5viewer.update_data()``) and the cursor advance
        # (``latest_frame()``) both fire from ``_flush_pending_update``
        # on the configured flush interval — running them per-frame made
        # the GUI O(N) per frame (full list clear + insertItems for
        # every new frame in a long scan), which compounded to O(N²)
        # over the run and showed up as visible stutter on slow
        # machines / very long scans.  The latest_idx assignment is
        # cheap and must stay per-frame so the flush handler knows
        # which frame to advance the cursor to.
        self.h5viewer.latest_idx = idx

        # Record the latest index and start the coalescing timer if it
        # isn't already running.  Throttle, not debounce: during a fast
        # scan burst (frame inter-arrival < timer interval) the display
        # must still refresh on the configured flush interval, not only after the burst
        # settles (debounce here froze plots until end-of-scan).  The
        # Coalescer is constructed mode="throttle", so trigger() keeps
        # the pending fire instead of restarting the countdown.
        self._pending_update_idx = idx
        self._update_timer.trigger()
        # ...and the fast list/cursor timer so the Frames list scrolls between
        # the (paced) heavy renders.
        self._list_timer.trigger()

    def _flush_frame_list(self):
        """LIGHT flush (fast timer): refresh the Frames list + advance the
        auto-last cursor, WITHOUT the heavy drain/render.

        The frame-list widget is built purely from ``scan.frames.index`` (which
        ``update_data`` appends per-frame), so it does not need the coalesced
        publication drain; ``latest_frame(emit_update=False)`` advances the cursor
        with signals blocked (no ``data_changed``/``setImage``).  This lets the
        list + selection + status scroll continuously while the expensive
        image/plot render stays paced on ``_update_timer``.  Idempotent
        with ``_flush_pending_update``, which redoes these O(new) legs before it
        renders."""
        if self._pending_update_idx is None:
            return
        try:
            self.h5viewer.update_data(emit_update=False)
            if self.h5viewer.auto_last:
                self.latest_frame(emit_update=False)
        except Exception:
            logger.debug("light frame-list flush failed", exc_info=True)

    def _drain_pending_frames(self):
        """Build + store publications and refresh scan_data for every frame
        stashed since the last flush.

        This is the heavy per-frame work (mask fold + publication build +
        validation + store upsert + scan_data row) moved OFF the per-event GUI
        path and batched here at ~5/sec, so GUI smoothness no longer tracks the
        frame/write rate (the whole-scan freeze, worsened when lz4 removed gzip's
        accidental write-throttle).  Display-only: the writer persists
        independently, so deferring this never loses data or touches
        persist-before-evict; the builds are stamped with the store's current
        generation (a mode switch bumps it and forces a rebuild anyway)."""
        pending = self._pending_frames
        if not pending:
            return
        self._pending_frames = {}
        import os as _os
        import time as _time
        _perf = bool(_os.environ.get("XDART_PERF"))
        t0 = _time.perf_counter()
        _t_mask = _t_build = _t_upsert = _t_scan = 0.0   # per-leg accumulators

        published = getattr(self.wrangler, "thread", None)
        record_store = getattr(published, "_streaming_record_store", None) \
            if published is not None else None
        if record_store is not None:
            self._frame_record_store = record_store
        global_mask = getattr(published, "mask", None) if published is not None else None
        if global_mask is not None:
            # Publish the detector gap mask ONCE per drain for the display.  We do
            # NOT fold it into each frame's own mask any more: the raw panel renders
            # via the raw_image payload, whose full-res path (_apply_detector_mask)
            # AND thumbnail gap-bake (combine_flat_masks) both apply scan.global_mask
            # DIRECTLY -- so the per-frame fold (an O(M log M) setdiff1d over the
            # large gap mask, EVERY frame) was pure redundant work and the dominant
            # drain-runaway cost.  frame.mask stays the per-frame map_raw<0; the
            # display unions scan.global_mask for the gaps.
            self.scan.global_mask = global_mask

        _is_gi = bool(getattr(self.scan, "gi", False))
        skip_2d = getattr(self.scan, "skip_2d", False)
        active_1d = (self.scan.bai_1d_args.get("gi_mode_1d", "q_total")
                     if _is_gi else None)
        active_2d = (self.scan.bai_2d_args.get("gi_mode_2d", "qip_qoop")
                     if _is_gi else None)
        from xdart.modules.ewald.scan import _coerce_scan_info

        new_rows = False
        for idx in sorted(pending):
            frame = pending[idx]
            try:
                _ts = _time.perf_counter() if _perf else 0.0
                # Step 6: key the live record under the real GI mode so a later
                # reintegrate at the same mode folds onto it.  .view is unaffected.
                publication = publication_from_live_frame(
                    frame,
                    generation=self.publication_store.generation,
                    active_mode_1d=active_1d,
                    active_mode_2d=active_2d,
                    # map_raw is already resident here. Publishing the same
                    # ndarray reference avoids a needless HDF5 hydration pass;
                    # bounded-store eviction releases both references together.
                    include_raw=True,
                    # X1 3c (S3-OR1): stamp the immutable scan owner from the
                    # authoritative run scan — never from a source filename.
                    scan_key=scan_identity_key(self.scan),
                )
                if not skip_2d and publication_has_2d_errors(publication):
                    logger.warning(
                        "Skipping frame %s 2D publication: %s", idx,
                        publication_error_details(publication, "2d"))
                if _perf:
                    _t1 = _time.perf_counter(); _t_build += _t1 - _ts; _ts = _t1
                self.publication_store.upsert(publication)
                if _perf:
                    _t_upsert += _time.perf_counter() - _ts
            except Exception:
                # Non-fatal — displayframe lazy-loads from disk as fallback.
                logger.debug("In-memory frame hand-off failed for idx=%s", idx,
                             exc_info=True)
            # Accumulate the scan_data row (numeric coerced, non-numeric kept).
            info = getattr(frame, "scan_info", None)
            if info:
                coerced = _coerce_scan_info(info)
                if coerced:
                    self._scan_info_rows[int(idx)] = coerced
                    new_rows = True

        # Rebuild scan_data as ONE DataFrame from the accumulated rows -- O(N) per
        # flush, not the O(N^2) per-frame `sd.loc[idx] = ser` enlargement.  Mirrors
        # LiveScan.add_frame (heterogeneous dtypes; pandas infers per column).
        if new_rows:
            _ts = _time.perf_counter() if _perf else 0.0
            import pandas as pd
            try:
                df = pd.DataFrame.from_dict(self._scan_info_rows, orient="index")
                df.sort_index(inplace=True)
                with self.scan.scan_lock:
                    self.scan.scan_data = df
            except (ValueError, TypeError):
                logger.debug("scan_data rebuild skipped", exc_info=True)
            if _perf:
                _t_scan = _time.perf_counter() - _ts

        if _perf:
            logger.info(
                "[PERF] drain %d frame(s): mask=%.0fms build=%.0fms upsert=%.0fms "
                "scan_data=%.0fms total=%.0fms",
                len(pending), _t_mask * 1000, _t_build * 1000, _t_upsert * 1000,
                _t_scan * 1000, (_time.perf_counter() - t0) * 1000)
        logger.debug("[PERF] drained %d frame(s) in %.1f ms",
                     len(pending), (_time.perf_counter() - t0) * 1000)

    def _h5viewer_data_changed_now(self, *, show_all=False):
        """Publish a programmatic live flush without the user-selection debounce."""
        immediate = getattr(self.h5viewer, "_data_changed_now", None)
        if callable(immediate):
            immediate(show_all=show_all)
            return
        try:
            self.h5viewer.data_changed(show_all=show_all)
        except TypeError:
            if show_all:
                self.h5viewer.data_changed(True)
            else:
                self.h5viewer.data_changed()

    def _request_render(self, reason=None, *, generation=None) -> bool:
        """Request a current-selection paint through the display authority."""
        df = getattr(self, "displayframe", None)
        if df is None:
            return False
        request = getattr(df, "request_current_selection_repaint", None)
        if callable(request):
            accepted = bool(request(generation=generation, reason=reason))
            browse_debug_log(
                logger,
                "render_request",
                requestor="staticWidget._request_render",
                reason=reason,
                generation=generation,
                display_generation=getattr(df, "display_generation", None),
                selected=sequence_summary(
                    getattr(getattr(self, "h5viewer", None), "frame_ids", ())),
                granted=accepted,
                suppressed_by=None if accepted else "display_authority",
            )
            flush = getattr(df, "_flush_current_selection_repaint", None)
            if accepted and callable(flush):
                try:
                    flush()
                except Exception:
                    logger.debug("render-authority flush failed for %s", reason,
                                 exc_info=True)
            return accepted
        update = getattr(df, "update", None)
        if not callable(update):
            return False
        browse_debug_log(
            logger,
            "render_request",
            requestor="staticWidget._request_render.legacy_update",
            reason=reason,
            generation=generation,
            display_generation=getattr(df, "display_generation", None),
            selected=sequence_summary(
                getattr(getattr(self, "h5viewer", None), "frame_ids", ())),
            granted=True,
        )
        try:
            update(expected_generation=generation)
        except TypeError:
            update()
        return True

    def _render_authority_stale_for_flush(self):
        """Return True when selection generation changed as this flush fired."""
        df = getattr(self, "displayframe", None)
        if df is None:
            return False
        before_generation = getattr(df, "display_generation", None)
        signature = getattr(df, "_selection_generation_signature", None)
        before_signature = None
        if callable(signature):
            try:
                before_signature = signature()
            except Exception:
                logger.debug("render-authority signature snapshot failed",
                             exc_info=True)
        sync = getattr(df, "_sync_selection_generation", None)
        if not callable(sync):
            return False
        try:
            current_generation = sync()
        except Exception:
            logger.debug("render-authority sync failed", exc_info=True)
            return False
        current_signature = before_signature
        if callable(signature):
            try:
                current_signature = signature()
            except Exception:
                logger.debug("render-authority signature refresh failed",
                             exc_info=True)
        stale = current_generation != before_generation
        if stale and os.environ.get("XDART_PERF"):
            logger.info(
                "[PERF] flush stale selection: gen=%r->%r sig=%r->%r",
                before_generation, current_generation,
                before_signature, current_signature,
            )
        return stale

    def _rearm_stale_flush(self) -> bool:
        """Re-arm a stale heavy flush, bounded so churn eventually converges."""
        count = int(getattr(self, "_render_authority_rearms", 0) or 0) + 1
        self._render_authority_rearms = count
        max_rearms = int(getattr(self, "_render_authority_max_rearms", 3) or 3)
        if count > max_rearms:
            logger.debug("render-authority stale flush forced after %d rearm(s)",
                         count - 1)
            self._render_authority_rearms = 0
            return False
        timer = getattr(self, "_update_timer", None)
        trigger = getattr(timer, "trigger", None)
        if callable(trigger):
            trigger()
        else:
            start = getattr(timer, "start", None)
            if callable(start):
                start()
        return True

    def _flush_overlay_before_frame_rescope(self):
        """Force an outgoing Overlay/Waterfall tick before destructive rescope."""
        if staticWidget._overlay_plot_method(self) not in ("Overlay", "Waterfall"):
            return False
        self._update_timer.stop()
        self._list_timer.stop()
        previous_reconcile = getattr(
            self.displayframe, "_overlay_reconcile_full_batch", False)
        self.displayframe._overlay_reconcile_full_batch = True
        try:
            if getattr(self, "_pending_update_idx", None) is not None:
                self._flush_pending_update(force=True)
            else:
                # A latest-only live tick can consume the pending marker while the
                # append-only accumulator still trails the complete per-scan index.
                # Reconcile that index at the one safe synchronous boundary before
                # rescope destroys the outgoing PublicationStore.
                with self.scan.scan_lock:
                    has_outgoing_frames = bool(self.scan.frames.index)
                if not has_outgoing_frames:
                    return False
                staticWidget._render_overlay_full_scan(self)
            # _flush_pending_update updates the selection through H5Viewer, whose
            # terminal sigUpdate is normally coalesced by another 100 ms timer.  A
            # scan boundary cannot wait for that timer because rescope clears the
            # outgoing PublicationStore immediately; render through the authority.
            staticWidget._request_render(self, "scan-boundary-overlay-flush")
        finally:
            self.displayframe._overlay_reconcile_full_batch = previous_reconcile
        return True

    def _flush_pending_update(self, *, force=False):
        """Render the most recently received wrangler update.

        Called by _update_timer at the configured flush interval.  Coalesces
        all per-frame GUI work that doesn't have to happen immediately:

        * ``h5viewer.update_data()`` — refresh the listData widget
          (incremental append when possible — see
          :meth:`h5viewer.update_data`).
        * ``latest_frame()`` — advance the auto-last cursor to whatever
          ``latest_idx`` is now (P4: was per-frame, now per-flush so
          we don't rebuild the list widget more than once per timer
          tick).
        * immediate ``h5viewer`` selection publish — publish the selected frame
          through the normal ``sigUpdate`` path exactly once.
        """
        if self._pending_update_idx is None:
            return
        if not force and staticWidget._render_authority_stale_for_flush(self):
            if staticWidget._rearm_stale_flush(self):
                return
        self._render_authority_rearms = 0
        self._pending_update_idx = None
        # Optional per-flush profiling: set XDART_PERF=1 in the shell to log the
        # drain / list-widget / render split at INFO so the dominant GUI-thread leg
        # is measured, not guessed.
        import os as _os
        import time as _t
        _perf = bool(_os.environ.get("XDART_PERF"))
        _t0 = _t.perf_counter() if _perf else 0.0
        # Build + store publications + scan_data for every frame stashed since the
        # last flush (the coalesced heavy work, off the per-frame GUI event loop).
        self._drain_pending_frames()
        try:
            self.displayframe._aggregate_live_scan = getattr(
                getattr(self.wrangler, "thread", None), "_active_scan", None)
        except Exception:
            logger.debug("aggregate live-scan handoff failed", exc_info=True)
        _t1 = _t.perf_counter() if _perf else 0.0
        # Heavy list-widget refresh first — auto-last cursor needs the
        # list to contain the new index before it can select it.
        self.h5viewer.update_data(emit_update=False)
        if self.h5viewer.auto_last:
            self.latest_frame(emit_update=False)
        _t2 = _t.perf_counter() if _perf else 0.0

        method = staticWidget._overlay_plot_method(self)
        if self.h5viewer.auto_last and method in ("Overlay", "Waterfall"):
            # Overlay/Waterfall must show EVERY processed frame, not just the
            # frames that landed in this timer window.  Fast (non-GI) scans
            # produce several frames between ticks; selecting only the tick's
            # pending set dropped earlier curves (visible only in slow GI runs).
            # Re-select the FULL processed set only at ~2 Hz during a live run:
            # rebuilding the full WaterfallHistory payload + painted stack every
            # 150 ms grows O(flushes x N) and beachballs the GUI on long catch-up.
            # Throttled ticks still render the latest frame incrementally; run-end
            # clears _processing_active before the final flush/reconcile, so the
            # final full stack is unthrottled and complete.
            if getattr(self.displayframe, "_processing_active", False):
                now = _t.perf_counter()
                last = getattr(self, "_overlay_flush_last_t", 0.0)
                if force:
                    # ``force`` is reserved for explicit scan/run boundaries.
                    # Complete the outgoing scan before its store is rescoped;
                    # never use this path from a timer or polling loop.
                    self._overlay_flush_last_t = now
                    staticWidget._render_overlay_full_scan(self, method=method)
                elif now - last < 0.5:
                    staticWidget._h5viewer_data_changed_now(self)
                else:
                    self._overlay_flush_last_t = now
                    staticWidget._render_overlay_full_scan(self, method=method)
            else:
                self._overlay_flush_last_t = 0.0
                staticWidget._render_overlay_full_scan(self, method=method)
        else:
            staticWidget._h5viewer_data_changed_now(self)  # → sigUpdate → set_data → metawidget.update()

        if _perf:
            _t3 = _t.perf_counter()
            logger.info(
                "[PERF] flush: drain=%.0fms list=%.0fms render=%.0fms total=%.0fms",
                (_t1 - _t0) * 1000, (_t2 - _t1) * 1000,
                (_t3 - _t2) * 1000, (_t3 - _t0) * 1000)

    def _reconcile_h5viewer_frame_list_after_run(self, written_file=None) -> int:
        """Refresh the frame browser from the authoritative run-end frame index."""

        scan = getattr(self, "scan", None)
        viewer = getattr(self, "h5viewer", None)
        if scan is None or viewer is None:
            browse_debug_log(
                logger,
                "runend_reconcile_skip",
                reason="missing_scan_or_viewer",
                written_file=written_file,
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            return 0

        # The frame-index reload below can legitimately rebuild listData and
        # clear its Qt selection (notably when ``new_scan_loaded`` is still set
        # for the last one-frame scan in a Directory run).  Remember the frame
        # that actually owns the visible raw/cake panels so Stop can restore the
        # same coherent selection after the rebuild.  Prefer rendered 2D state
        # over the browser cursor: Overlay/Waterfall may carry many 1D rows, but
        # raw/cake always belong to one representative frame.
        runend_2d_anchor = None
        display = getattr(self, "displayframe", None)
        rendered_2d = list(getattr(display, "idxs_2d", ()) or ())
        if rendered_2d:
            runend_2d_anchor = rendered_2d[-1]
        if runend_2d_anchor is None:
            list_widget = getattr(getattr(viewer, "ui", None), "listData", None)
            try:
                current = list_widget.currentItem() if list_widget is not None else None
                if current is not None:
                    runend_2d_anchor = current.text()
            except Exception:
                logger.debug("run-end 2D anchor capture skipped", exc_info=True)
        if runend_2d_anchor is None:
            selected = list(getattr(viewer, "frame_ids", ()) or ())
            if selected:
                runend_2d_anchor = selected[-1]
        browse_debug_log(
            logger,
            "runend_reconcile_enter",
            written_file=written_file,
            scan=getattr(scan, "name", None),
            runend_2d_anchor=runend_2d_anchor,
            method=staticWidget._overlay_plot_method(self),
            auto_last=getattr(viewer, "auto_last", None),
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )

        import time as _time
        _perf = bool(os.environ.get("XDART_PERF"))
        _t0 = _time.perf_counter() if _perf else 0.0
        indexed = 0
        if written_file and os.path.exists(written_file):
            loader = getattr(scan, "load_frame_index_only", None)
            if callable(loader):
                try:
                    indexed = int(loader(written_file) or 0)
                    if indexed:
                        logger.info(
                            "post-live: indexed %d frame(s) from %s",
                            indexed,
                            os.path.basename(written_file),
                        )
                    browse_debug_log(
                        logger,
                        "runend_reconcile_index_loaded",
                        written_file=written_file,
                        indexed=indexed,
                        **_runend_waterfall_history_fields(
                            getattr(self, "displayframe", None)),
                    )
                except Exception:
                    logger.warning(
                        "post-live frame-index populate failed",
                        exc_info=True,
                    )

        try:
            frame_index = list(getattr(getattr(scan, "frames", None), "index", ()) or ())
            if frame_index:
                try:
                    viewer.latest_idx = int(frame_index[-1])
                except (TypeError, ValueError):
                    viewer.latest_idx = frame_index[-1]
            list_widget = getattr(getattr(viewer, "ui", None), "listData", None)
            visible = []
            if list_widget is not None:
                visible = [
                    list_widget.item(row).text()
                    for row in range(list_widget.count())
                ]
            expected = [str(idx) for idx in frame_index]
            force_rebuild = (
                bool(getattr(viewer, "new_scan_loaded", False))
                or visible != expected
            )
            browse_debug_log(
                logger,
                "runend_reconcile_before_update_data",
                indexed=indexed,
                latest_idx=getattr(viewer, "latest_idx", None),
                visible=sequence_summary(visible),
                expected=sequence_summary(expected),
                force_rebuild=force_rebuild,
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            viewer.update_data(emit_update=False, force_rebuild=force_rebuild)
            list_widget = getattr(getattr(viewer, "ui", None), "listData", None)

            # If the rebuild dropped every selected row, restore the exact
            # rendered 2D anchor (or the final frame when no rendered anchor is
            # available).  Do this without enabling Auto Last and publish once
            # through the immediate selection path so the final update_all()
            # cannot replace valid raw/cake with an empty payload.  Existing
            # selections are untouched, including Show-All/Waterfall selections.
            restored_anchor = None
            if list_widget is not None:
                try:
                    has_selection = bool(list_widget.selectedItems())
                except Exception:
                    has_selection = bool(getattr(viewer, "frame_ids", ()))
                if not has_selection and expected:
                    candidate = str(runend_2d_anchor)
                    if runend_2d_anchor is None or candidate not in expected:
                        candidate = str(expected[-1])
                    item = None
                    for row in range(list_widget.count()):
                        candidate_item = list_widget.item(row)
                        if candidate_item is not None and candidate_item.text() == candidate:
                            item = candidate_item
                            break
                    if item is not None:
                        was_blocked = list_widget.blockSignals(True)
                        try:
                            setter = getattr(viewer, "set_current_frame", None)
                            if callable(setter):
                                setter(item)
                            else:
                                list_widget.setCurrentItem(item)
                            try:
                                viewer.frame_ids[:] = [candidate]
                            except (AttributeError, TypeError):
                                viewer.frame_ids = [candidate]
                        finally:
                            list_widget.blockSignals(was_blocked)
                        restored_anchor = candidate
                        staticWidget._h5viewer_data_changed_now(self)
            visible_after = []
            if list_widget is not None:
                visible_after = [
                    list_widget.item(row).text()
                    for row in range(list_widget.count())
                ]
            browse_debug_log(
                logger,
                "runend_reconcile_after_update_data",
                latest_idx=getattr(viewer, "latest_idx", None),
                visible=sequence_summary(visible_after),
                restored_anchor=restored_anchor,
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            rendered = staticWidget._render_overlay_full_scan(self)
            browse_debug_log(
                logger,
                "runend_reconcile_after_render",
                render_owned=rendered,
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            if _perf:
                logger.info(
                    "[PERF] post-live frame-list reconcile: indexed=%d "
                    "visible=%d expected=%d force=%s total=%.0fms",
                    indexed, len(visible), len(expected), force_rebuild,
                    (_time.perf_counter() - _t0) * 1000,
                )
        except Exception:
            logger.debug("post-live frame-list rebuild failed", exc_info=True)
            browse_debug_log(
                logger,
                "runend_reconcile_exception",
                level="warning",
                indexed=indexed,
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
        browse_debug_log(
            logger,
            "runend_reconcile_exit",
            indexed=indexed,
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        return indexed

    def _overlay_plot_method(self):
        try:
            return self.displayframe.ui.plotMethod.currentText()
        except Exception:
            return ""

    def _render_overlay_full_scan(self, *, method=None) -> bool:
        """Render Overlay/Waterfall from the full processed frame set.

        Live throttling callers decide when this expensive path is allowed. This
        helper only performs the full reselect and reports whether it owned the
        render.
        """
        method = staticWidget._overlay_plot_method(self) if method is None else method
        caller = _runend_callsite()
        browse_debug_log(
            logger,
            "runend_render_overlay_full_scan_enter",
            caller=caller,
            method=method,
            auto_last=getattr(getattr(self, "h5viewer", None), "auto_last", None),
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        if not (self.h5viewer.auto_last and method in ("Overlay", "Waterfall")):
            browse_debug_log(
                logger,
                "runend_render_overlay_full_scan_skip",
                caller=caller,
                reason="not_auto_last_overlay",
                method=method,
                auto_last=getattr(self.h5viewer, "auto_last", None),
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            return False
        with self.scan.scan_lock:
            selected = [str(int(i)) for i in self.scan.frames.index]
        browse_debug_log(
            logger,
            "runend_render_overlay_full_scan_selection",
            caller=caller,
            selected=sequence_summary(selected),
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        if selected:
            self.h5viewer.frame_ids[:] = selected
            staticWidget._h5viewer_data_changed_now(self, show_all=True)
        else:
            staticWidget._h5viewer_data_changed_now(self)
        browse_debug_log(
            logger,
            "runend_render_overlay_full_scan_exit",
            caller=caller,
            selected=sequence_summary(selected),
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        return True

    def _owns_finished_run(self) -> bool:
        """H18-R8: run ownership by CANONICAL run identity, never display
        filename spelling.  Eiger ``*_master.h5`` processing canonicalizes
        the worker/output name by stripping ``_master`` while the widget
        owner can retain the pre-canonical spelling; the literal name
        comparison then skipped shared finalization AND the run-end overlay
        catch-up for the run's own display (the 651-frame waterfall that
        ended at 635 identities).  Compare through the one canonical
        ``scan_name_from_source`` (the same identity run-end reconciliation
        and scan-qualified rows use) so legitimate aliases still own their
        finish exactly once."""
        scan_name = str(getattr(getattr(self, "scan", None), "name", "") or "")
        wrangler_name = str(getattr(self.wrangler, "scan_name", "") or "")
        return (staticWidget._canonical_run_token(scan_name)
                == staticWidget._canonical_run_token(wrangler_name))

    @staticmethod
    def _canonical_run_token(name):
        """Exact run token with only the documented Eiger alias removed.

        Inputs here are usually already-canonical bare scan names.  Applying
        the image-series suffix parser a second time collapsed distinct
        container runs ending in ``_00005`` and ``_00006``.  Preserve every
        other character and strip only a file extension and ``_master``.
        """
        if not name:
            return None
        text = str(name)
        path = Path(text)
        token = path.stem if path.suffix.lower() in {
            ".h5", ".hdf5", ".nxs", ".cxi"
        } else path.name
        if token.lower().endswith("_master"):
            token = token[:-7]
        return token or None

    def _hook_waterfall_zoom_recording(self):
        """H18-R8(b): record GENUINE user zooms on the active bottom plot
        (``sigRangeChangedManually`` fires only for mouse/keyboard range
        changes, never for programmatic ``setRange``), so the run-end
        auto-fit can distinguish an operator's zoom from a mid-run
        programmatic crop.  Idempotent."""
        df = getattr(self, "displayframe", None)
        if df is None:
            return
        hooked = getattr(df, "_wf_zoom_hooked_ids", None)
        if hooked is None:
            hooked = df._wf_zoom_hooked_ids = set()
        # H18-R14: connect BOTH concrete switchable bottom plots regardless
        # of which is currently active — an Overlay run starts below the
        # Waterfall threshold with the line plot active, and a zoom made
        # after acquisition crosses the threshold must still be observed.
        plots = []
        line_plot = getattr(df, "plot", None)
        if line_plot is not None:
            plots.append(line_plot)
        wf_plot = getattr(getattr(df, "wf_widget", None), "image_plot", None)
        if wf_plot is not None and wf_plot not in plots:
            plots.append(wf_plot)
        for plot in plots:
            try:
                viewbox = plot.getViewBox()
                if id(viewbox) in hooked:
                    continue
                viewbox.sigRangeChangedManually.connect(
                    lambda *_, _df=df: setattr(_df, "_wf_user_zoomed", True))
                hooked.add(id(viewbox))
            except Exception:
                logger.debug("waterfall zoom hook failed", exc_info=True)

    def _runend_waterfall_autofit(self):
        """H18-R8(b): full-history auto-fit after the automatic run-end
        reseed/catch-up UNLESS a genuine user zoom is recorded — a mid-run
        programmatic crop (mode projection, cadence render) must not leave
        the finished waterfall viewport stuck at a partial window."""
        df = getattr(self, "displayframe", None)
        if df is None or getattr(df, "_wf_user_zoomed", False):
            return
        try:
            plot = df._active_bottom_plot()
            if plot is None:
                return
            plot.autoRange()
            plot.enableAutoRange()
        except Exception:
            logger.debug("run-end waterfall auto-fit failed", exc_info=True)

    _runend_catchup_timeout_s = 30.0
    _runend_autofit_timeout_s = 30.0

    def _runend_autofit_when_quiet(self, generation=None, deadline=None):
        """Fit after disk load settles, scoped to the run that requested it."""
        generation = (getattr(self, "_runend_autofit_generation", None)
                      if generation is None else generation)
        if generation is None:
            generation = getattr(self, "_runend_generation", 0)
        if (generation != getattr(self, "_runend_generation", 0)
                or bool(getattr(self, "_run_active", False))):
            return
        if deadline is None:
            deadline = time.monotonic() + float(
                getattr(self, "_runend_autofit_timeout_s", 30.0))
        if staticWidget._runend_catchup_busy(self):
            if time.monotonic() < deadline:
                QtCore.QTimer.singleShot(
                    250,
                    lambda: staticWidget._runend_autofit_when_quiet(
                        self, generation, deadline),
                )
            else:
                logger.warning(
                    "run-end waterfall auto-fit timed out while the GUI was busy")
            return
        staticWidget._runend_waterfall_autofit(self)

    def _arm_runend_overlay_catchup(self):
        """PERF-3 Option A — one-shot run-end overlay catch-up.

        After a live run the Overlay/Waterfall can end short: the lagged tail
        frames were dropped at the cap-128 display hand-off and NEVER published,
        so no in-memory reselect can paint them (Item-2's "cheap in-memory
        reselect" premise was false).  Recovery needs a REAL async disk load ==
        a manual Show All.  Arm a ONE-SHOT catch-up that waits out the run-end
        debounce cascade (the ``set_run_writing(False)`` + auto-last selection-
        collapse echoes) and then calls ``show_all()`` ONCE.  Replaces the failed
        Item-2 ``_render_overlay_full_scan`` call.  No-op unless Overlay/Waterfall
        + auto_last.

        Cancellation deliberately does NOT gate on ``display_generation``: the
        automatic run-end selection-collapse echoes bump it via
        ``_sync_selection_generation`` (SELECTION), so a generation-equality gate
        would false-abort and the fix would silently never fire (the Item-2 trap).
        The real user gestures are caught by the robust signals in the callback:
        auto_last off (frame click -> ``disable_auto_last``), scan.name changed
        (new scan/file), token cleared (new run via ``_enter_run_state``; new
        file via ``_on_new_file_display_reset`` on the set_file/data_reset
        cascade).
        """
        method = staticWidget._overlay_plot_method(self)
        viewer = getattr(self, "h5viewer", None)
        # Log arm/skip at INFO (not just debug): the catch-up is a safety net, so
        # a run-end must show whether it armed -- silence otherwise reads
        # ambiguously as "not needed" vs "never ran".
        if not (getattr(viewer, "auto_last", False)
                and method in ("Overlay", "Waterfall")):
            logger.info(
                "[PERF] run-end overlay catch-up: not armed (auto_last=%s mode=%s)",
                getattr(viewer, "auto_last", None), method)
            self._runend_catchup_token = None
            return
        # H18-R8: the token is the CANONICAL run identity so a post-arm
        # rename between alias spellings of the SAME run (e.g. *_master vs
        # canonical output) cannot false-cancel the catch-up.
        self._runend_catchup_token = staticWidget._canonical_run_token(
            getattr(self.scan, "name", None))
        self._runend_catchup_generation = getattr(self, "_runend_generation", 0)
        self._runend_catchup_deadline = time.monotonic() + float(
            getattr(self, "_runend_catchup_timeout_s", 30.0))
        self._runend_catchup_tries = 0
        staticWidget._hook_waterfall_zoom_recording(self)
        logger.info("[PERF] run-end overlay catch-up: armed (scan=%r)",
                    self._runend_catchup_token)
        QtCore.QTimer.singleShot(
            250,
            lambda: staticWidget._runend_overlay_catchup(
                self, self._runend_catchup_generation),
        )

    def _runend_catchup_busy(self) -> bool:
        """True while the run-end debounce cascade / a disk load is still in
        flight — fire the catch-up only once the GUI has quiesced."""
        viewer = getattr(self, "h5viewer", None)
        if viewer is None:
            return False
        if getattr(viewer, "_load_worker", None) is not None:
            return True
        for name in ("_selection_coalesce_timer", "_load_coalesce_timer",
                     "_update_coalesce_timer"):
            timer = getattr(viewer, name, None)
            if timer is not None and timer.is_pending():
                return True
        return bool(getattr(viewer, "_browse_one_shot_pending_render", False))

    def _runend_overlay_missing_ids(self) -> set:
        """Frame indices in the current scan NOT yet in the waterfall accumulator.

        Length-tolerant id decode: slice-mode rows are 3-tuples
        ``(scan_key, frame_idx, projection_id)``, so never destructure the row id
        directly (a literal ``(skey, fidx)`` unpack ValueErrors).  A cleared
        accumulator (``None`` after ``clear_overlay``) reads as all-missing, which
        correctly triggers a full rebuild.
        """
        df = getattr(self, "displayframe", None)
        history = getattr(df, "_waterfall_history", None)
        with self.scan.scan_lock:
            full = {int(i) for i in self.scan.frames.index}
        if not full:
            return set()
        have = set()
        for row_id in (getattr(history, "ids", ()) or ()):
            if row_id_belongs_to_widget_scan(df, row_id):
                have.add(frame_index_from_row_id(row_id))
        return full - have

    def _runend_overlay_catchup(self, generation=None):
        """One-shot post-quiescence auto-Show-All (see _arm_runend_overlay_catchup)."""
        generation = (getattr(self, "_runend_catchup_generation", None)
                      if generation is None else generation)
        if (generation is not None
                and generation != getattr(self, "_runend_generation", 0)):
            return
        if getattr(self, "_runend_catchup_token", None) is None:
            return                                   # cleared by a new run / reset
        df = getattr(self, "displayframe", None)
        viewer = getattr(self, "h5viewer", None)
        method = staticWidget._overlay_plot_method(self)
        # Guard 1 — cancellation (robust; NOT generation-gated, see arm docstring).
        # Abort reasons are logged at INFO (not debug): a run-end no-fire should be
        # visible in the log you're already reading, not hidden behind a flag.
        current_token = staticWidget._canonical_run_token(
            getattr(self.scan, "name", None))
        if (viewer is None
                or not getattr(viewer, "auto_last", False)
                or self._runend_catchup_token != current_token
                or method not in ("Overlay", "Waterfall")):
            reason = (
                "no-viewer" if viewer is None
                else "auto_last-off (user clicked a frame)"
                if not getattr(viewer, "auto_last", False)
                else "scan-changed"
                if self._runend_catchup_token != current_token
                else "mode=%s" % method)
            logger.info("[PERF] run-end overlay catch-up: aborted (%s)", reason)
            self._runend_catchup_token = None
            return
        # Guard 2 — quiescence: re-arm (bounded ~2 s) until the debounce cascade
        # and any load worker settle, then give up (degrades to manual Show All).
        if staticWidget._runend_catchup_busy(self):
            self._runend_catchup_tries = getattr(self, "_runend_catchup_tries", 0) + 1
            if time.monotonic() < getattr(
                    self, "_runend_catchup_deadline", 0.0):
                QtCore.QTimer.singleShot(
                    250,
                    lambda: staticWidget._runend_overlay_catchup(
                        self, generation),
                )
            else:
                self._runend_catchup_token = None
                logger.warning(
                    "run-end overlay catch-up timed out while the GUI was busy; "
                    "the displayed waterfall may be incomplete (use Show All)")
            return
        # Guard 3 — missing set.  Empty -> already complete -> idempotent no-op.
        missing = staticWidget._runend_overlay_missing_ids(self)
        if not missing:
            logger.info("[PERF] run-end overlay catch-up: already complete, no-op")
            self._runend_catchup_token = None
            staticWidget._runend_waterfall_autofit(self)     # H18-R8(b)
            return
        # Guard 4 — fire once.  The click path (show_all) does the rest: select
        # all, split resident/missing, ONE _LoadFramesWorker for the missing tail,
        # generation-gated absorb, append into the carried accumulator.
        self._runend_catchup_token = None
        logger.info(
            "[PERF] run-end overlay catch-up: %d missing frame(s) -> show_all()",
            len(missing))
        browse_debug_log(logger, "runend_overlay_catchup_fire", missing=len(missing))
        try:
            self.h5viewer.show_all()
        except Exception:
            logger.debug("run-end overlay catch-up show_all failed", exc_info=True)
        # H18-R8(b): once the disk load settles, auto-fit the full history
        # unless a genuine user zoom was recorded.
        self._runend_autofit_generation = generation
        autofit_deadline = time.monotonic() + float(
            getattr(self, "_runend_autofit_timeout_s", 30.0))
        QtCore.QTimer.singleShot(
            250,
            lambda: staticWidget._runend_autofit_when_quiet(
                self, generation, autofit_deadline),
        )

    def disable_auto_last(self, q):
        """
        Parameters
        ----------
        q : Qt.QtWidgets.QListWidgetItem
        """
        self.h5viewer.auto_last = False

    def enable_auto_last(self, q):
        """
        Parameters
        ----------
        q : Qt.QtWidgets.QListWidgetItem
        """
        self.h5viewer.auto_last = True

    def _on_display_cleared(self):
        """Viewer-mode Clear: drop the H5Viewer file-list selection so the
        cleared plot, the (now empty) selection and the title all agree.

        ``data_changed`` clears ``frame_ids`` and then early-returns on the empty
        selection (no re-render), so this won't repaint or restore a stale title.
        """
        try:
            self.h5viewer.ui.listData.clearSelection()
        except Exception:
            logger.debug("clear listData selection on display Clear failed",
                         exc_info=True)

    def set_data(self):
        """Connected to h5viewer, sets the data in displayframe based
        on the selected image or overall data.
        """
        if getattr(self.h5viewer, "_browser_restore_in_progress", False):
            return
        # Deferred browser-select reset (Int 1D/2D): a manual .nxs select left the
        # display + caches untouched (see _on_new_file_display_reset).  Now that a
        # frame is actually clicked, run that reset — clear the previous scan's
        # caches + rebuild axes — BEFORE rendering, so the clicked frame loads fresh
        # (this is the only frame-selection path that reaches set_data; the
        # scan-select deselect is signal-blocked, so it never fires here).
        if getattr(self.h5viewer, "_browser_scan_reset_pending", False):
            selected_ids = [
                str(frame_id)
                for frame_id in getattr(self.h5viewer, "frame_ids", ())
                if str(frame_id) and str(frame_id) != "No data"
            ]
            if not selected_ids:
                try:
                    selected_ids = [
                        str(item.text())
                        for item in self.h5viewer.ui.listData.selectedItems()
                        if str(item.text()) and str(item.text()) != "No data"
                    ]
                except Exception:
                    selected_ids = []
            if not selected_ids:
                return
            self.h5viewer._browser_scan_reset_pending = False
            try:
                self.displayframe.set_axes()
                self._clear_frame_record_store()   # A-Step (Phase 5): reset the store
                self.displayframe._clear_bkg()
                self.h5viewer.data_reset()
                try:
                    self.h5viewer.frame_ids[:] = selected_ids
                except TypeError:
                    self.h5viewer.frame_ids = list(selected_ids)
                staticWidget._h5viewer_data_changed_now(self)
                return
            except Exception:
                logger.debug("deferred browser-select reset failed", exc_info=True)
        # In viewer mode, always update display (no scan dependency)
        is_viewer = getattr(self.h5viewer, 'viewer_mode', None) is not None
        if is_viewer or self.scan.name != 'null_main':
            # Propagate the Image-Viewer file classification from the H5Viewer
            # (which classifies on file select) to the display widget (which
            # renders).  Without this, displayframe._viewer_is_xdart is always
            # False, so the Image Viewer's raw-preview payload takes the
            # *standalone* branch even for processed xdart .nxs frames and fills
            # their baked NaN mask (the inverse of the intended behaviour).
            self.displayframe._viewer_is_xdart = getattr(
                self.h5viewer, '_viewer_is_xdart', False)
            self.displayframe._browse_one_shot_target_labels = tuple(
                getattr(self.h5viewer, "_browse_one_shot_target_labels", ())
                or ()
            )
            self.displayframe._browse_one_shot_publications = dict(
                getattr(self.h5viewer, "_browse_one_shot_publications", {})
                or {}
            )
            self.displayframe._browse_one_shot_anchor_label = getattr(
                self.h5viewer, "_browse_one_shot_anchor_label", None)
            overlay_pending = list(
                getattr(self.h5viewer, "_overlay_hydrated_pending_append_labels", ())
                or ()
            )
            if overlay_pending:
                append_queue = getattr(
                    self.displayframe,
                    "_overlay_hydrated_pending_append_labels",
                    None,
                )
                if append_queue is not None:
                    queued = set(append_queue)
                    for label in overlay_pending:
                        if label not in queued:
                            append_queue.append(label)
                            queued.add(label)
                else:
                    self.displayframe._overlay_hydrated_pending_append_labels = (
                        overlay_pending
                    )
                self.h5viewer._overlay_hydrated_pending_append_labels = []
            staticWidget._request_render(self, "h5viewer-selection")
            # # if (len(self.frames.keys()) > 0) and (len(self.scan.frames.index) > 0):
            # if ((len(self.viewer_rows_1d.keys()) > 0) and
            #         (len(self.frame_ids) > 0) and
            #         (self.frame_ids[0] != 'No data') and
            #         (len(self.scan.frames.index) > 0)):

            # Frame availability just changed (a scan finished / was loaded /
            # cleared), so refresh the integration controls — this is what
            # toggles the Reintegrate row on once a processed scan exists.
            # _apply_integration_control_state is the single source of truth
            # (mode + run-state + frames + reachable-raw + skip_2d).
            self._apply_integration_control_state()

            self.metawidget.update()
            # self.integratorTree.update()

            # Live peak-fit preview (no-op unless the dialog is open + Live on).
            self._maybe_live_fit()
            self._refresh_controls_v2_profile()

    def _hydrate_integrator_on_load(self, *args):
        """Stage C: when a ``.nxs`` is loaded, populate the integration panel from
        the saved scan (units/npts/ranges/GI), so the panel shows the saved
        reduction and Reintegrate reproduces it.  Skipped during an active run —
        the wrangler owns the config then, and the scan is mid-write."""
        if getattr(self, '_run_active', False):
            return
        if self._controls_v2_enabled():
            self._controls_v2_ensure_native_int_defaults()
            self._controls_v2_hydrate_advanced_from_scan()
            self._refresh_controls_v2_profile(immediate=True)
            return
        try:
            self.integratorTree.hydrate_from_scan()
        except Exception:
            logger.debug("integrator hydrate_from_scan failed", exc_info=True)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the fresh-session default split through the launch-time window-
        # manager resize storm (first 3s); afterwards the splitter is the
        # user's.
        import time as _time
        if _time.monotonic() < getattr(self, '_split_until', 0):
            apply = getattr(self, '_apply_default_split', None)
            if callable(apply):
                apply()

    def enable_async_hydration(self):
        """Turn on off-GUI-thread rehydration of evicted frames (D2, greenfield
        Phase 3).  Called by the live app entry (``_gui_main``) — NOT during
        construction — so headless widget tests keep the synchronous blocking
        reads their assertions expect.  Idempotent + defensive."""
        try:
            df = getattr(self, 'displayframe', None)
            if df is not None and hasattr(df, 'enable_async_hydration'):
                df.enable_async_hydration()
        except Exception:
            logger.debug("enable_async_hydration failed", exc_info=True)

    def close(self):
        """Tries a graceful close.
        """
        # Block the analysis slots first: a worker signal queued just before we
        # stop + destroy must not touch the about-to-be-destroyed dialog.
        self._tearing_down = True
        # Persist the integration panel settings (the wrangler tree saves
        # continuously; the integrator panel saves here at exit).
        try:
            from xdart.utils.session import save_session
            state = {'controls_v2_int': self._controls_v2_int_session_state()}
            if not self._controls_v2_enabled():
                state['integrator'] = self.integratorTree.session_state()
            save_session(state)
        except Exception:
            logger.debug("integrator session save failed", exc_info=True)
        # Pause/Resume (Phase B): a PAUSED run blocks the wrangler thread in its
        # `while command == 'pause'` wait.  Closing the window must break that
        # wait so run() returns and the QThread isn't "destroyed while running".
        # Setting command='stop' (the universal run-end signal) exits the pause
        # wait from any state; bound-wait the thread so teardown is clean.
        self._stop_wrangler_thread_on_close()
        # The wrangler consumes this same serialized directory authority. Close
        # it only after processing has stopped so teardown cannot cancel an
        # observation future that the worker is currently reconciling.
        source_widget = getattr(self, "_controls_v2_source_widget", None)
        if source_widget is not None:
            source_widget.shutdown_probe_worker()
        # Reintegration thread: request a between-batches stop and wait --
        # close() never touched it, so a multi-minute reintegrate-all
        # running at close was destroyed mid-loop (Qt6 qFatal) with its
        # cached reduction session never finished.
        self._stop_integrator_thread_on_close()
        # Stitch is also a one-shot QThread parented to this widget.  It can be
        # inside a long MultiGeometry call at app close, so request stop for the
        # pre-start window and bound-wait like the other run threads.
        self._stop_stitch_thread_on_close()
        # Stop the viewer's long-running background threads BEFORE teardown so
        # the persistent fileHandlerThread / async load worker aren't destroyed
        # while running ("QThread: Destroyed while thread is still running") on
        # tab/app close.  Mirrors the GUI-test fixture teardown.
        try:
            h5v = getattr(self, 'h5viewer', None)
            if h5v is not None and hasattr(h5v, 'shutdown_threads'):
                h5v.shutdown_threads()
        except Exception:
            logger.debug("background-thread shutdown on close failed",
                         exc_info=True)
        # D2 (greenfield Phase 3): stop the off-thread frame-hydration worker
        # before teardown (same "destroyed while running" guard as above).
        try:
            df = getattr(self, 'displayframe', None)
            if df is not None and hasattr(df, 'stop_hydration_worker'):
                df.stop_hydration_worker()
            if df is not None and hasattr(df, 'stop_aggregation_worker'):
                df.stop_aggregation_worker()
        except Exception:
            logger.debug("hydration-worker shutdown on close failed",
                         exc_info=True)
        # Stop the analysis workers (live preview + batch) before teardown.
        try:
            law = getattr(self, '_live_analysis_worker', None)
            if law is not None:
                law.stop()
            baw = getattr(self, '_batch_analysis_worker', None)
            if baw is not None:
                baw.stop()
        except Exception:
            logger.debug("analysis-worker shutdown on close failed",
                         exc_info=True)
        # Close the scan-plot dialog EXPLICITLY: Qt does not deliver closeEvent
        # to children on parent close, and the dialog's closeEvent is what
        # stops its ROI worker, redraw timer, and the source-probe executor
        # (a non-daemon thread that otherwise outlives the app).
        try:
            dlg = getattr(self, '_scan_plot_dialog', None)
            if dlg is not None:
                self._scan_plot_dialog = None
                dlg.close()
        except Exception:
            logger.debug("scan-plot dialog close failed", exc_info=True)
        del self.scan
        del self.displayframe.scan
        del self.frame
        del self.displayframe.frame
        super().close()

        gc.collect()

    def _detach_and_retain_slow_close_thread(self, thread, label):
        try:
            thread.setParent(None)
        except Exception:
            logger.debug("detaching slow %s thread failed", label,
                         exc_info=True)
        _retain_orphaned_close_thread(thread)

    def _stop_wrangler_thread_on_close(self):
        try:
            w = getattr(self, 'wrangler', None)
            wt = getattr(w, 'thread', None) if w is not None else None
            if wt is not None and wt.isRunning():
                w.command = 'stop'
                wt.command = 'stop'
                # The run() finally performs the end-of-run session finish
                # (writer join up to 60s) + final .nxs flush -- 5s routinely
                # lost that race and Qt aborted on the still-running thread,
                # killing the very flush that protects the data.  30s covers
                # everything but a wedged NFS write.
                if not wt.wait(30000):
                    logger.warning("wrangler thread still finishing at "
                                   "close after 30s; final flush may be "
                                   "incomplete")
                    staticWidget._detach_and_retain_slow_close_thread(
                        self, wt, "wrangler")
        except Exception:
            logger.debug("stopping wrangler thread on close failed",
                         exc_info=True)

    def _stop_integrator_thread_on_close(self):
        try:
            it = getattr(getattr(self, 'integratorTree', None),
                         'integrator_thread', None)
            if it is not None and it.isRunning():
                it.stop_requested = True
                if not it.wait(15000):
                    logger.warning("integrator thread still running at "
                                   "close after 15s")
                    staticWidget._detach_and_retain_slow_close_thread(
                        self, it, "integrator")
        except Exception:
            logger.debug("stopping integrator thread on close failed",
                         exc_info=True)

    def _stop_stitch_thread_on_close(self):
        try:
            st = getattr(self, 'stitch_thread', None)
            if st is not None and st.isRunning():
                st.stop_requested = True
                if not st.wait(15000):
                    logger.warning("stitch thread still running at close after 15s")
                    try:
                        st.setParent(None)
                    except Exception:
                        logger.debug("detaching slow stitch thread failed",
                                     exc_info=True)
                    _retain_orphaned_stitch_thread(st)
        except Exception:
            logger.debug("stopping stitch thread on close failed", exc_info=True)

    def _show_integration_advanced(self):
        """Show a combined dialog with the integratorTree's existing
        1D and 2D advanced parameter widgets."""
        if self._controls_v2_run_active():
            logger.debug("Ignoring advanced integration dialog during active run")
            return
        self._controls_v2_hydrate_advanced_from_scan()
        if not hasattr(self, '_integ_adv_combined_dlg'):
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Integration \u2014 Advanced Settings')
            dlg.resize(420, 450)
            layout = QtWidgets.QVBoxLayout(dlg)

            lbl1d = QtWidgets.QLabel('<b>Integrate 1D</b>')
            layout.addWidget(lbl1d)
            # Re-parent the existing advancedWidget trees into our dialog
            layout.addWidget(self.integratorTree.advancedWidget1D.tree)

            line = QtWidgets.QFrame()
            line.setFrameShape(QtWidgets.QFrame.HLine)
            line.setFrameShadow(QtWidgets.QFrame.Sunken)
            layout.addWidget(line)

            lbl2d = QtWidgets.QLabel('<b>Integrate 2D</b>')
            layout.addWidget(lbl2d)
            layout.addWidget(self.integratorTree.advancedWidget2D.tree)

            self._integ_adv_combined_dlg = dlg

        self._integ_adv_combined_dlg.show()
        self._integ_adv_combined_dlg.raise_()

    def enable_integration(self, enable=True):
        """Calls the integratorTree setEnabled function.
        """
        self.integratorTree.setEnabled(enable)

    def _apply_integration_control_state(self):
        """Enable/disable the integration controls for the current mode (C3/C4).

        - Int 1D / Int 1D (XYE): the 2-D integration panel is disabled — there
          is no cake in a 1D-only run.
        - Image / XYE / NeXus Viewer: the 1-D and 2-D integration panels are
          disabled (file-browser modes; the wrangler processing params are
          disabled separately via the wrangler ``tree``), but **Calibrate** and
          **Make Mask** stay enabled — they're still useful in a viewer.
        - Int 2D: everything enabled.

        While a run is active (``self._run_active``, task #71) the
        processing-affecting controls are force-disabled *on top of* the
        per-mode state: the 1-D and 2-D panels (whose children are the range
        fields, point counts, Auto toggles, unit + GI-mode combos, and the
        Re-Integrate buttons), plus **Calibrate** and **Make Mask** (they mutate
        the PONI / mask the run depends on).  The running frames use a
        deep-copied arg snapshot, but a mid-run edit would otherwise leak into
        the next scan of a multi-scan run and into a later reintegrate.  The
        frame1D/frame2D contents are plain Qt widgets (checkable ``QPushButton``
        Auto toggles, ``QComboBox`` units/modes), so ``setEnabled(False)`` here
        keeps their checked look — the pyqtgraph readonly-checkbox repaint bug
        only affects the wrangler's ParameterTree, not these.  (The Advanced
        1D/2D dialogs ARE pyqtgraph ParameterTrees; they also feed bai_*_args and
        are locked per-widget in _enter_run_state — not blanket-disabled here.)
        ``Stop`` lives on the wrangler and is left enabled; display/h5viewer
        browsing is untouched.

        Disabled widgets dim via the theme's ``:disabled`` style (D2).  Keyed
        off the processing-mode combo + run-state so it's one source of truth.
        """
        itree = getattr(self, 'integratorTree', None)
        if itree is None or not hasattr(itree, 'ui'):
            return
        try:
            mode_text = self.controls.current_mode()
        except Exception:
            mode_text = ''
        is_viewer = mode_text in ('Image Viewer', 'XYE Viewer', 'NeXus Viewer')
        is_1d_only = mode_text in ('Int 1D', 'Int 1D (XYE)')
        # 4d: the streaming session is the authoritative run-state when present,
        # but `_run_active` remains the cache that covers the windows the session
        # can't: the start→first-frame gap (the adapter opens on the first frame)
        # and the reintegrate-via-integratorThread path (no adapter at all).  OR
        # them so controls can only ever be *more* disabled mid-run, never wrongly
        # re-enabled before `_exit_run_state` re-asserts the mode-correct state.
        # (The disk-read-guard timing stays on sigPaused/sigResuming — R7 — never
        # on these reads.)
        run_active = self._controls_v2_run_active()
        ui = itree.ui
        # 2-D integration panel: only in Int 2D, and never during a run.
        frame2d = getattr(ui, 'frame2D', None)
        if frame2d is not None:
            frame2d.setEnabled(not is_viewer and not is_1d_only and not run_active)
        # 1-D integration panel: any Int mode, not viewers, never during a run.
        frame1d = getattr(ui, 'frame1D', None)
        if frame1d is not None:
            frame1d.setEnabled(not is_viewer and not run_active)
        # Calibrate / Make Mask stay enabled everywhere (incl. viewers) EXCEPT
        # during a run — they mutate the PONI / mask the run depends on.
        for name in ('pyfai_calib', 'get_mask'):
            btn = getattr(ui, name, None)
            if btn is not None:
                btn.setEnabled(not run_active)

        # Reintegrate / Advanced row tracks the SAME enable as the integration
        # panels above: available in any non-viewer Int mode, never during a run.
        # Reintegrate 2D follows frame2D exactly — also disabled in Int-1D-only
        # modes (no cake to reintegrate).  We do NOT gate on scan.frames or
        # raw-reachability: bai_1d/bai_2d no-op on an empty scan and pop a clear
        # message when raw is unreachable (R3), and probing here opened the .nxs
        # read-only — the mid-run writer crash.  So enable mirrors the panels;
        # "is there anything to reintegrate / is the raw reachable" is enforced
        # (with feedback) only when the user actually clicks.
        reint1d = getattr(ui, 'reintegrate1D', None)
        if reint1d is not None:
            reint1d.setEnabled(not is_viewer and not run_active)
        reint2d = getattr(ui, 'reintegrate2D', None)
        if reint2d is not None:
            reint2d.setEnabled(not is_viewer and not is_1d_only and not run_active)
        adv = getattr(ui, 'advanced_int', None)
        if adv is not None:
            adv.setEnabled(not is_viewer and not run_active)

        # GI (Fiber) + Threshold rows (added this cycle) follow the SAME rule as
        # the integration panels: disabled in viewer modes and during a run.
        # They were omitted before, so they stayed bright/active while the rest of
        # the integrator was greyed -- now the whole integrator dims together.
        for name in ('gi_frame', 'frame_pixreject'):
            frame = getattr(ui, name, None)
            if frame is not None:
                frame.setEnabled(not is_viewer and not run_active)

        if getattr(self, "controls_v2", None) is not None and run_active:
            self._set_controls_v2_current_fields_enabled(False)
        elif getattr(self, "controls_v2", None) is not None:
            self._refresh_controls_v2_profile(
                immediate=True,
                preserve_focused_editor=not getattr(
                    self, "_controls_v2_unlocking_run", False),
            )

    def _set_controls_v2_current_fields_enabled(self, enabled: bool) -> None:
        """Lock existing V2 editors without rebuilding the panel."""

        panel = getattr(self, "controls_v2", None)
        if panel is None:
            return
        try:
            from .ui.controls_panel_v2 import (  # local import avoids init-time Qt churn
                FormRow,
                PillRow,
                RangeRow,
                SegmentedControl,
            )
        except Exception:
            logger.debug("Controls Panel V2 lock import failed", exc_info=True)
            return

        enabled = bool(enabled)
        for row in panel.findChildren(FormRow):
            editor = getattr(row, "editor", None)
            if editor is not None:
                editor.setEnabled(enabled)
            browse = getattr(row, "browse_button", None)
            if browse is not None:
                browse.setEnabled(enabled)
        for row in panel.findChildren(RangeRow):
            for editor_name in ("_low", "_high"):
                editor = getattr(row, editor_name, None)
                if editor is not None:
                    editor.setEnabled(enabled)
            toggle = getattr(row, "_toggle", None)
            if toggle is not None:
                toggle[1].setEnabled(enabled)
        for row in panel.findChildren(PillRow):
            for _path, button in getattr(row, "_pills", ()):
                button.setEnabled(enabled)
        for row in panel.findChildren(SegmentedControl):
            for _value, button in getattr(row, "_segments", ()):
                button.setEnabled(enabled)
        if not enabled:
            for button in panel.findChildren(QtWidgets.QAbstractButton):
                if button.objectName() in {
                    "controlsV2ActionButton",
                    "controlsV2MoreButton",
                }:
                    button.setEnabled(False)

    def _session_run_active(self):
        """4d: True iff a streaming session is open AND reports it is running.

        Reads the wrangler's ``scan_session`` seam (the ``ScanSessionAdapter``),
        never the private slot.  Returns False when no session is open (so the
        OR with ``_run_active`` falls through to the cache) — robustly guarded so
        a duck/partial wrangler in a test never raises here."""
        wrangler = getattr(self, 'wrangler', None)
        session = getattr(wrangler, 'scan_session', None) if wrangler else None
        if session is None:
            thread = getattr(wrangler, 'thread', None) if wrangler else None
            session = getattr(thread, 'scan_session', None) if thread else None
        if session is None:
            return False
        try:
            return bool(session.is_running)
        except Exception:
            return False

    def _controls_v2_run_active(self) -> bool:
        return (
            bool(getattr(self, '_run_active', False))
            or self._session_run_active()
        )

    def _wrangler_run_active(self) -> bool:
        """True when the wrangler is in an actual acquisition/reduction run.

        ``QThread.isRunning()`` alone can be a stale/over-broad signal during
        finish-slot ordering.  The shared run-state owner only needs to keep
        controls locked for a wrangler that is also in a live run phase.
        """
        wrangler = getattr(self, 'wrangler', None)
        thread = getattr(wrangler, 'thread', None) if wrangler else None
        if thread is None:
            return False
        try:
            if not bool(thread.isRunning()):
                return False
        except Exception:
            return False
        phase = str(getattr(wrangler, '_run_phase', '') or '').lower()
        if phase:
            return phase in {'running', 'pausing', 'paused'}
        command = str(
            getattr(wrangler, 'command', '')
            or getattr(thread, 'command', '')
            or ''
        ).lower()
        return command in {'start', 'pause'}

    def _set_scan_integrated_reads_transient(self, active):
        frames = getattr(getattr(self, "scan", None), "frames", None)
        setter = getattr(frames, "set_integrated_reads_transient", None)
        if callable(setter):
            setter(active)

    def _enter_run_state(self):
        """Single owner of run START (task #68): mark a wrangler/integrator run
        in progress.  Idempotent — re-entry while already active is a no-op so
        re-fired ``started`` signals don't double-toggle.

        Drives BOTH the display persist flag (so the 2-D panels keep their last
        content during the run, matching the 1-D plot) AND the processing-control
        disable (task #71), so the two can't desync.  Wired through the paths
        that always fire on a run start: ``start_wrangler`` (wrangler live/batch)
        and the ``integrator_thread.started`` signal (reintegrate).
        """
        if self._run_active:
            return
        self._runend_generation = getattr(self, "_runend_generation", 0) + 1
        self._runend_catchup_generation = None
        self._runend_autofit_generation = None
        self._v2_active_result_paths = set()
        try:
            source_label = staticWidget._controls_v2_configured_source_label(self)
            self._v2_active_source_identity = (
                staticWidget._controls_v2_normalized_path(source_label))
        except (AttributeError, TypeError):
            # Minimal lifecycle hosts and source panels without a configured
            # path still participate in run ownership; they simply cannot
            # contribute source provenance until a result path is registered.
            self._v2_active_source_identity = ""
        # H18-R8(b)/R12: a new run opens a fresh genuine-zoom window — zooms
        # recorded during THIS run survive the run-end auto-fit; older ones
        # don't pin the next run's finished viewport.  The manual-range
        # listeners are connected HERE (both switchable bottom plots,
        # idempotent) so gestures DURING processing are observed, not only
        # after the run has already finished.
        df = getattr(self, "displayframe", None)
        if df is not None:
            df._wf_user_zoomed = False
        staticWidget._hook_waterfall_zoom_recording(self)
        self._run_active = True
        # Frame-driven scan-boundary flag: clear it at run START so the first
        # new_scan of THIS run always clears the panel (fixes a same-name re-run
        # not clearing) — an unconsumed flag from a prior run can't leak in.
        self._frame_driven_rescoped_pending = False
        # Re-snapshot the frame-count freeze from scratch each run: clear any
        # leftover snapshot so a new run can't freeze at the PREVIOUS run's frame
        # count if no run_active=False refresh happened to clear it in between
        # (F6 — the clear in _controls_v2_state is timing-dependent; this is the
        # authoritative reset at run START).
        self._controls_v2_run_frame_count = None
        # Append-mode feedback: track whether THIS run displayed any frame, so a
        # run that processes 0 new frames (Append over an already-complete scan)
        # can still show the scan's last frame at the end (visual confirmation
        # that something happened).  Reset per run start.
        self._run_saw_frame = False
        # A new run supersedes any pending run-end overlay catch-up from the last.
        self._runend_catchup_token = None
        self._set_scan_integrated_reads_transient(True)
        self.displayframe.set_processing_active(True)
        # X1 Slice 3a (R3-P5/P6): the run lifecycle — not set_processing_active
        # — owns the run wavelength capture.  Capture the GUI-owned run scan
        # object + its canonical key and bind the cache to THIS run.
        self._x1_run_scan_capture = self.scan
        _begin = getattr(self.displayframe, "begin_processing", None)
        if callable(_begin):
            _begin(self.scan, scan_identity_key(self.scan))
        # Start the main-thread liveness window for this run (XDART_PERF only).
        self._perf_hb_start_window()
        # Same run-state, pushed to the h5viewer so the frame-selection disk-load
        # guard (data_changed) and the reader-side hydration guard
        # (_processing_active, just set above) share one source of truth and can't
        # drift across live/batch/reintegrate (the GUI must not read the .nxs the
        # writer is churning — that's the frame-click freeze).
        self.h5viewer.set_run_writing(True)
        self._apply_integration_control_state()   # run_active=True → disable
        # Lock the MODE row (mode combo + Batch + Cores) for the run.  A wrangler
        # run also does this via wrangler.enabled(), but a reintegrate does not —
        # so own it here (the single run-start owner).  The ACTION row stays
        # enabled (Pause/Resume/Stop).
        try:
            self.controls.set_mode_row_enabled(False)
            self.controls.set_run_row_enabled(True)
        except Exception:
            logger.debug("lock mode row on run enter failed", exc_info=True)
        # Enable the shared Stop button for the run.  For a wrangler run this is
        # redundant (the wrangler enables it via the alias); for a reintegrate it
        # is the ONLY thing that makes Stop usable -> abort + retune.
        try:
            self.controls.set_stop_enabled(True)
        except Exception:
            logger.debug("enable Stop on run enter failed", exc_info=True)
        # Reintegrate also LOCKS Start: launching a scan mid-reintegrate starts a
        # wrangler run that rebuilds scan.frames out from under the reintegrate
        # loop (the 'Frame not found' KeyError crash).  A wrangler run instead
        # morphs Start->Pause (still clickable), so only lock it for reintegrate.
        _it = getattr(getattr(self, 'integratorTree', None),
                      'integrator_thread', None)
        if _it is not None and _it.isRunning():
            try:
                self.controls.startButton.setEnabled(False)
            except Exception:
                logger.debug("disable Start on reintegrate failed",
                             exc_info=True)
        # The Advanced 1D/2D parameter dialogs also feed bai_*_args (their
        # sigUpdateArgs → get_args mutates scan.bai_1d/2d_args), so a dialog left
        # open from before the run could leak a mid-run edit into the next scan.
        # Disable them too (per-widget, as reintegrate already does via
        # integratorTree.setEnabled(False)); re-enabled by enable_integration in
        # _exit_run_state.
        itree = getattr(self, 'integratorTree', None)
        for name in ('advancedWidget1D', 'advancedWidget2D'):
            adv = getattr(itree, name, None)
            if adv is not None:
                adv.setEnabled(False)
        dlg = getattr(self, '_integ_adv_combined_dlg', None)
        if dlg is not None:
            try:
                dlg.setEnabled(False)
            except Exception:
                logger.debug("disable combined advanced dialog failed",
                             exc_info=True)

    def _exit_run_state(self):
        """Single owner of run END (task #68): mark the run finished.
        Idempotent — exiting an already-idle state is a no-op.

        Reached on every end path including Stop and exceptions, because it is
        driven from the ``finished`` handlers (``QThread.finished`` fires
        whenever ``run()`` returns).  Ends the display persist window, then
        re-enables the integration controls and re-asserts the *mode-correct*
        per-mode state (Int 1D vs Int 2D vs viewer) rather than a blanket enable,
        so the controls are right for the current mode after the run.
        """
        if not self._run_active:
            return
        self._run_active = False
        # X1 Slice 3a (R3-P5): FINAL exit — stamp the CAPTURED run scan's
        # persisted wavelength from the run cache before the capture is
        # cleared (Pause never reaches here; it goes through _on_run_paused).
        _captured = getattr(self, "_x1_run_scan_capture", None)
        _finish = getattr(self.displayframe, "finish_processing", None)
        if callable(_finish):
            _finish(_captured, scan_identity_key(_captured))
        self._x1_run_scan_capture = None
        self._set_scan_integrated_reads_transient(False)
        self.displayframe.set_processing_active(False)
        # File is idle again: clear the disk-load guard.  set_run_writing(False)
        # also re-fires the standing frame selection so any frame skipped during
        # the run (evicted + disk-load suppressed) loads now from the idle file.
        self.h5viewer.set_run_writing(False)
        # Re-enable the tree (restores the auto-range field gating) then overlay
        # the mode-correct state (run_active is now False).
        self.enable_integration(True)
        dlg = getattr(self, '_integ_adv_combined_dlg', None)
        if dlg is not None:
            try:
                dlg.setEnabled(True)
            except Exception:
                logger.debug("re-enable combined advanced dialog failed",
                             exc_info=True)
        self._invalidate_controls_v2_render_cache()
        self._controls_v2_unlocking_run = True
        try:
            self._apply_integration_control_state()
        finally:
            self._controls_v2_unlocking_run = False
        # Unlock the mode row (matched with _enter_run_state).  A wrangler end
        # also does this via wrangler.enabled(True); both agree on the idle state.
        try:
            self.controls.set_mode_row_enabled(True)
        except Exception:
            logger.debug("unlock mode row on run exit failed", exc_info=True)
        # Run fully ended: drop the Stop button (reintegrate enabled it in
        # _enter_run_state; a wrangler end agrees on the idle state).
        try:
            self.controls.set_stop_enabled(False)
        except Exception:
            logger.debug("disable Stop on run exit failed", exc_info=True)
        # Re-enable Start (reintegrate disabled it in _enter_run_state; a wrangler
        # end resets it via its own idle morph, so True here is consistent).
        try:
            self.controls.startButton.setEnabled(True)
        except Exception:
            logger.debug("re-enable Start on run exit failed", exc_info=True)
        self._controls_v2_run_frame_count = None
        if getattr(self, "_controls_v2_batch_refresh_deferred", False):
            self._controls_v2_batch_refresh_deferred = False
        if getattr(self, "_controls_v2_last_signature", None) is None:
            self._refresh_controls_v2_profile(
                immediate=True,
                preserve_focused_editor=False,
            )
        # R4B-12 (O-1a-i.3): nothing to do here for deferred edits.  They are a
        # pure data delta folded into the intent + hidden carriers exactly once,
        # synchronously, inside the next Start's preparation
        # (_controls_v2_fold_deferred_edits_into_intent) — there is no timer, no
        # post-finalization owner, and no carrier write on the run-exit path.

    def _on_stop_clicked(self):
        """Single owner of the shared Stop button — route to the active run.

        A running **reintegrate** takes priority. Stopped reintegrations roll
        back their shadow write and leave the persisted scan unchanged, so Stop
        asks before discarding the in-progress pass, then sets the integrator
        thread's cooperative ``stop_requested`` (checked between batches; one
        frame in Live mode). The thread unwinds within a frame, fires
        ``finished`` → ``integrator_thread_finished`` → ``_exit_run_state``
        (re-enabling the panel). Otherwise delegate to the active
        **wrangler**'s ``stop()`` (its run-end UI reset)."""
        st = getattr(self, 'stitch_thread', None)
        if st is not None and st.isRunning():
            # Stitch is one MultiGeometry call — not abortable mid-reduction;
            # flag it (honoured only before the heavy work starts) and give
            # immediate Stop-button feedback.  The in-flight reduction completes,
            # then finished → stitch_thread_finished → _exit_run_state.
            st.stop_requested = True
            try:
                self.controls.set_stop_enabled(False)
            except Exception:
                logger.debug("disable Stop after stitch-stop failed",
                             exc_info=True)
            return
        it = getattr(getattr(self, 'integratorTree', None),
                     'integrator_thread', None)
        if it is not None and it.isRunning():
            if not self._confirm_discard_reintegrate():
                return                                    # let it finish
            it.stop_requested = True
            try:
                self.controls.set_stop_enabled(False)   # immediate feedback
            except Exception:
                logger.debug("disable Stop after reintegrate-stop failed",
                             exc_info=True)
            return
        w = getattr(self, 'wrangler', None)
        if w is not None and hasattr(w, 'stop'):
            w.stop()

    def _confirm_discard_reintegrate(self) -> bool:
        """Modal warning before stopping a reintegrate.

        Streaming reintegrate writes into shadow groups and atomically swaps
        them only when every requested frame finishes. Stopping rolls back the
        shadow groups, so the persisted scan stays unchanged. Returns True to
        stop and discard the in-progress pass, False to keep running. Isolated
        so tests can stub it without a live dialog.
        """
        from pyqtgraph import Qt
        mb = Qt.QtWidgets.QMessageBox(self)
        mb.setIcon(Qt.QtWidgets.QMessageBox.Icon.Warning)
        mb.setWindowTitle("Stop reintegration?")
        mb.setText("Stop this reintegration?")
        mb.setInformativeText(
            "Frames processed so far are only staged in a temporary write. If "
            "you stop now, that staged work will be discarded and the saved "
            "scan will remain unchanged.\n\n"
            "Stop and discard the in-progress reintegration, or let it finish "
            "so everything is saved?")
        stop_btn = mb.addButton("Stop && Discard",
                                Qt.QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        keep_btn = mb.addButton("Let it finish",
                                Qt.QtWidgets.QMessageBox.ButtonRole.RejectRole)
        mb.setDefaultButton(keep_btn)
        mb.exec()
        return mb.clickedButton() is stop_btn

    def _on_run_paused(self):
        """Pause (Phase B): the run is FROZEN at a frame boundary (the worker has
        drained the in-flight window + flushed the .nxs and emitted sigPaused).
        LIFT the disk-read freeze guard so the user can browse ANY frame from
        disk while paused -- but the run is still active, so keep ``_run_active``
        True and leave the parameter/integration controls hard-disabled (#72).

        Safe ordering: this runs only AFTER the worker is provably idle (sigPaused
        is emitted post-drain/flush), so a disk read here can't race a write.
        ``set_run_writing(False)`` also re-fires the standing frame selection, so
        a frame skipped during the run now loads from the quiesced file."""
        if not self._run_active:
            return                       # not in a run; nothing to lift
        self._set_scan_integrated_reads_transient(False)
        self.displayframe.set_processing_active(False)
        # X1 Slice 3a (R3-P6): Pause PRESERVES the run's scan-qualified
        # wavelength cache and captured target; only the writing window shut.
        _pause = getattr(self.displayframe, "pause_processing", None)
        if callable(_pause):
            _pause()
        invalidate_levels = getattr(
            self.displayframe, "invalidate_image_level_caches", None)
        if callable(invalidate_levels):
            invalidate_levels()
        # The live-run file-thread mode deliberately turns set_datafile into a
        # path-only repoint so a writer flush cannot replace the in-memory live
        # frame index.  The writer is quiescent now, so manual browser opens must
        # use the normal full load or the selected scan inherits the paused
        # run's frame list.  _run_active stays true, which also keeps integrator
        # hydration and all processing controls locked to the frozen run plan.
        file_thread = getattr(self.h5viewer, "file_thread", None)
        if file_thread is not None:
            file_thread.live_run = False
        self.h5viewer.set_run_writing(False)
        request_repaint = getattr(
            self.displayframe, "request_current_selection_repaint", None)
        if callable(request_repaint):
            request_repaint(
                generation=getattr(
                    self.displayframe, "display_generation", None),
                reason="pause",
            )

    def _on_run_resuming(self):
        """Resume (Phase B): RE-ENGAGE the freeze guard BEFORE the worker flips
        the command back to the run state, so a browse read can't overlap the
        restarted writer.  ``set_run_writing(True)`` also cancels any in-flight
        browse load on its rising edge.  Runs synchronously (same GUI thread)
        from the wrangler's sigResuming, ahead of the command flip."""
        if not self._run_active:
            return
        self._set_scan_integrated_reads_transient(True)
        # Restore path-only live repoints before re-engaging the writer guard.
        # The next frame-driven scan rescope can then return the browser to the
        # active output without reloading a file that has resumed writing.
        file_thread = getattr(self.h5viewer, "file_thread", None)
        if file_thread is not None:
            file_thread.live_run = bool(
                getattr(self.h5viewer, "live_run_active", False))
        self.h5viewer.set_run_writing(True)
        self.displayframe.set_processing_active(True)
        # X1 Slice 3a (R3-P6): Resume may consult only the captured run's
        # scan-qualified cache (enforced by the key guard) — no reseed.
        _resume = getattr(self.displayframe, "resume_processing", None)
        if callable(_resume):
            _resume(scan_identity_key(
                getattr(self, "_x1_run_scan_capture", None)))

    def update_all(self, idx=None):
        """Updates all data in displays.

        This is the main-thread refresh path for the static scan tab. The
        forced ``gc.collect()`` that used to live here has been removed:
        the GIL-interacting stop-the-world pause was contributing to the
        UI stutter noted in the old TODO.  Cycle collection is left to
        the default GC schedule, which is run by CPython between
        allocation bursts.  If profiling ever shows a leak driven by
        reference cycles here, re-add a scoped ``gc.collect()`` with a
        comment explaining the specific object graph being collected.
        """
        if idx is not None:
            self.h5viewer.latest_idx = idx

        self.h5viewer.update_data()
        if self.h5viewer.auto_last:
            self.latest_frame()

        staticWidget._request_render(self, "update-all")
        self.metawidget.update()

    def integrator_thread_update(self, idx):
        """Per-frame reintegrate signal — THROTTLED.

        The reintegrate worker emits this once per frame (live = every frame).
        Rendering each synchronously floods the GUI event loop (the 2D cake is
        ~hundreds of ms a frame) so nothing paints until the run ends — the
        "freezes + no live updates" report.  Coalesce to the ~5 Hz timer (like
        the wrangler's update_data) and do the actual refresh in
        ``_flush_reintegrate_update``.  set_open_enabled is cheap + wants to be
        prompt, so it stays here."""
        self.h5viewer.set_open_enabled(True)
        self._pending_reint_idx = idx
        self._reint_update_timer.trigger()

    def _flush_reintegrate_update(self):
        """Coalesced reintegrate display refresh (≤ ~5 Hz).  Advances to the most
        recent reintegrated frame and renders it from the in-memory publication
        store (the run-write disk guard is fine — reintegrate has no concurrent
        writer until the end save)."""
        idx = self._pending_reint_idx
        self._pending_reint_idx = None
        if idx is not None:
            self.h5viewer.latest_idx = idx
        self.h5viewer.update_data()
        # Live reintegrate auto-FOLLOWS each frame as it's reduced — that's the
        # whole point (watch progress + decide whether to retune) — so advance
        # the displayed frame even when Auto-Last is off.
        it = getattr(self.integratorTree, 'integrator_thread', None)
        live_reint = bool(getattr(it, 'reintegrate_live', False))
        if self.h5viewer.auto_last or live_reint:
            self.latest_frame()
        staticWidget._request_render(self, "reintegrate-flush")
        self.metawidget.update()

    def _finalize_processing_run(self, *, reset_overlay, origin):
        """Finish shared run UI state, optionally resetting plot history.

        Overlay history belongs to the acquisition run and must survive normal
        wrangler completion.  A true reintegration invalidates that history, so
        only its finished slot requests the reset.
        """
        browse_debug_log(
            logger,
            f"runend_{origin}_finalize_enter",
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        self.thread_state_changed()
        # End the run through the single run-state owner (task #68) BEFORE the
        # final refresh so the 2D panels resume normal blank-on-missing for the
        # final frame.  _exit_run_state re-enables the integration controls and
        # re-asserts the mode-correct per-mode state (it folds in the former
        # enable_integration(True) call).  Only exit if no wrangler run is still
        # in flight: a wrangler can be started while a reintegrate runs, and its
        # frames still need the controls locked — its own finished handler will
        # exit the shared run-state then (mirrors the wrangler-enable guard
        # below).
        wrangler_running = self._wrangler_run_active()
        browse_debug_log(
            logger,
            f"runend_{origin}_finalize_state",
            wrangler_running=wrangler_running,
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        if not wrangler_running:
            self._exit_run_state()
        self.h5viewer.set_open_enabled(True)
        if reset_overlay:
            try:
                browse_debug_log(
                    logger,
                    f"runend_{origin}_before_clear_overlay",
                    **_runend_waterfall_history_fields(
                        getattr(self, "displayframe", None)),
                )
                self.displayframe.clear_overlay(
                    LifecycleCause.REINTEGRATE,
                    site=f"{origin}_thread_finished[reintegrate finish]")
                browse_debug_log(
                    logger,
                    f"runend_{origin}_after_clear_overlay",
                    **_runend_waterfall_history_fields(
                        getattr(self, "displayframe", None)),
                )
            except Exception:
                logger.debug("display overlay reset after reintegrate failed",
                             exc_info=True)
        self.update_all()
        browse_debug_log(
            logger,
            f"runend_{origin}_finalize_after_update_all",
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        if not wrangler_running:
            self.wrangler.enabled(True)
        browse_debug_log(
            logger,
            f"runend_{origin}_finalize_exit",
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )

    def integrator_thread_finished(self):
        """Finish a true reintegration and invalidate prior overlay history."""
        staticWidget._finalize_processing_run(
            self,
            reset_overlay=True,
            origin="integrator",
        )

    # ── Stitch (Stitch 1D / Stitch 2D modes) ───────────────────────────
    def _stitch_status(self, msg):
        """Surface a stitch status message in the bottom status bar (via the
        active wrangler's router), falling back to the window status bar."""
        w = getattr(self, 'wrangler', None)
        if w is not None and hasattr(w, '_set_status_text'):
            w._set_status_text(msg)
            return
        try:
            self.window().statusBar().showMessage(msg)
        except Exception:
            logger.debug("stitch status failed", exc_info=True)

    def start_stitch(self, mode):
        """Launch the one-shot stitch worker for the loaded scan (Stitch 1D/2D).

        Diverted here from imageWrangler.start() when a Stitch mode is active.
        Gates on frames + geometry up front (run_stitch raises on the worker
        thread otherwise), reads the stitch params from the integrator's
        existing 1D/2D fields, and starts the worker — whose started/finished
        route through the shared run-state owner (_enter/_exit_run_state)."""
        if self.stitch_thread.isRunning() or getattr(self, '_run_active', False):
            return
        scan = self.scan
        if not getattr(scan, 'frames', None):
            self._stitch_status('Load a scan before stitching.')
            return
        if getattr(scan, 'geometry', None) is None:
            self._stitch_status(
                'Stitch needs a calibration/geometry on the scan.')
            return
        # GI guard: the GUI stitch uses the multigeometry backend, which applies
        # NO GI correction (footprint/Fresnel/refraction).  Running it with GI
        # (Fiber) mode ON would silently produce a *non-GI* merge.  The GI-corrected
        # stitch (pyfai_hist + GISettings) is gated on the real-data GIXSGUI
        # convention check, so block rather than mislead.  Toggle GI off for a
        # standard q-stitch.
        if getattr(scan, 'gi', False):
            self._stitch_status(
                'GI-corrected stitch is pending real-data validation — toggle GI '
                '(Fiber) off to run a standard (non-GI) stitch.')
            return
        try:
            params = self._build_stitch_params(mode)
        except Exception:
            logger.error("build stitch params failed", exc_info=True)
            self._stitch_status('Could not read stitch settings.')
            return
        self.stitch_thread.mode = mode
        self.stitch_thread.params = params
        self.stitch_thread.stop_requested = False
        # Fail-loud UX: a detector mask that can't be applied to the stitch
        # geometry is dropped to None by _flat_mask_as_bool with only a log
        # warning, so the stitch would silently run UNMASKED.  Surface it in
        # the run status rather than letting it pass unseen.
        if getattr(scan, 'global_mask', None) is not None and params.get('mask') is None:
            self._stitch_status(
                f'Detector mask could not be applied — stitching '
                f'({mode.upper()}) UNMASKED…')
        else:
            self._stitch_status(f'Stitching ({mode.upper()})…')
        self.stitch_thread.start()         # started -> _enter_run_state

    def _build_stitch_params(self, mode):
        """run_stitch kwargs from the integrator's existing 1D/2D fields (reused
        — no separate stitch options in Phase 1).

        The wrangler's detector/global mask is stored on the scan as flat indices
        (``scan.global_mask``) with the full-res ``scan.detector_shape``; convert
        it to the 2D boolean (True = exclude) run_stitch → pyFAI expect, reusing
        the canonical fail-soft converter (a mask that doesn't fit the detector is
        dropped with a warning, never crashes the stitch).  No `backend` kwarg:
        run_stitch is MultiGeometry-only today (the GI→histogram backend needs a
        headless change first)."""
        from xdart.modules.reduction import _flat_mask_as_bool
        args = self.scan.bai_1d_args if mode == '1d' else self.scan.bai_2d_args
        mask = _flat_mask_as_bool(
            getattr(self.scan, 'global_mask', None),
            getattr(self.scan, 'detector_shape', None),
        )
        p = dict(
            unit=args.get('unit', 'q_A^-1'),
            method=args.get('method', 'BBox'),
            radial_range=args.get('radial_range'),
            azimuth_range=args.get('azimuth_range'),
            mask=mask,
        )
        if mode == '1d':
            p['npt_1d'] = int(self.scan.bai_1d_args.get('numpoints') or 2000)
        else:
            p['npt_rad_2d'] = int(self.scan.bai_2d_args.get('npt_rad') or 1500)
            p['npt_azim_2d'] = int(self.scan.bai_2d_args.get('npt_azim') or 720)
        return p

    def _on_stitch_mode_changed(self, stitch_mode_str):
        """Route the display to/from the persistent stitch view when the wrangler
        Mode dropdown enters/leaves a Stitch mode (``'1d'``/``'2d'``/``''``).

        Only flips the flag + refreshes; ``displayFrameWidget._active_stitch_mode``
        gates the actual render on a matching ``scan.stitched_*`` result, so
        selecting Stitch before a run leaves the per-frame view untouched and
        leaving Stitch restores it."""
        self.displayframe.stitch_display_mode = stitch_mode_str or None
        self.displayframe._bump_display_generation()
        self.update_all()
        self._refresh_controls_v2_profile()

    def stitch_thread_finished(self):
        """Stitch worker done: end the shared run-state (unless a wrangler run is
        also in flight) and refresh.  On success the result becomes the persistent
        display source (StitchDisplayController) — set the flag + bump generation
        BEFORE the refresh so update_all() routes through it (it now survives
        subsequent update() calls instead of the old one-shot paint)."""
        self.thread_state_changed()
        if not self.wrangler.thread.isRunning():
            self._exit_run_state()
        self.h5viewer.set_open_enabled(True)
        if getattr(self.stitch_thread, 'ok', False):
            self.displayframe.stitch_display_mode = self.stitch_thread.mode
            self.displayframe._bump_display_generation()
            # Surface a partial skip so the merge isn't silently a subset.
            skipped = getattr(self.scan, 'stitch_skipped', None) or []
            suffix = (f' — WARNING: {len(skipped)} frame(s) skipped (no raw data)'
                      if skipped else '')
            self._stitch_status(
                f'Stitch {self.stitch_thread.mode.upper()} complete.{suffix}')
        # else: _on_stitch_error already surfaced the failure — don't overwrite.
        self.update_all()
        if not self.wrangler.thread.isRunning():
            self.wrangler.enabled(True)

    def _on_stitch_error(self, msg):
        """Stitch worker raised (caught in the worker, so the thread survived and
        still fires finished → run-state exit).  Surface the message."""
        self._stitch_status(f'Stitch failed: {msg}')
        logger.error("Stitch error: %s", msg)

    def _overlay_clear_needed_for_scan_boundary(self, first_frame=None):
        """Return whether a scan boundary should drop Overlay/Waterfall history."""
        df = getattr(self, "displayframe", None)
        if df is None:
            return True
        try:
            method = df.ui.plotMethod.currentText()
        except Exception:
            method = None
        history = getattr(df, "_waterfall_history", None)
        if method not in ("Overlay", "Waterfall") or not getattr(history, "count", 0):
            return True
        new_key = overlay_grid_key_for_widget(df, first_frame=first_frame)
        keep = overlay_grid_keys_match(getattr(history, "reset_key", None), new_key)
        old_spec = overlay_grid_spec_for_history(history)
        new_spec = None
        if first_frame is not None:
            try:
                publication = publication_from_live_frame(
                    first_frame, include_raw=False, include_2d=True,
                    include_thumbnail=False, retain_raw_ref=False,
                    validate=False,
                )
                new_spec = overlay_grid_spec_for_view(df, publication.view)
                if keep and new_spec is not None:
                    keep = overlay_grid_specs_match(old_spec, new_spec)
            except Exception:
                logger.debug("scan-boundary concrete-grid probe failed",
                             exc_info=True)
        if not keep and new_spec is not None:
            logger.info(
                "Overlay axis changed during live processing; starting a new "
                "overlay. Current: %s; Selected: %s",
                overlay_grid_spec_summary(old_spec),
                overlay_grid_spec_summary(new_spec),
            )
        if os.environ.get("XDART_PERF"):
            logger.info(
                "[PERF] scan boundary overlay rows=%d keep=%s old_key=%r new_key=%r",
                getattr(history, "count", 0),
                keep,
                getattr(history, "reset_key", None),
                new_key,
            )
        return not keep

    def _ask_overlay_grid_mismatch(self, current_spec, selected_spec):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Axes do not match")
        box.setText("Axes do not match the current overlay")
        box.setInformativeText(
            "Current: %s\nSelected: %s" % (
                overlay_grid_spec_summary(current_spec),
                overlay_grid_spec_summary(selected_spec),
            ))
        start = box.addButton(
            "Start New Overlay", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return "reset" if box.clickedButton() is start else "cancel"

    def _resolve_overlay_grid_mismatch(self, current_spec, selected_spec):
        """Prompt manual selections; keep unattended acquisition non-blocking.

        ``_run_active`` deliberately does not participate: it remains true while
        paused, when browser selections are explicitly user-driven and must get
        the same transactional warning as idle browsing.  The two writer/display
        flags describe actual automated publication, not merely run ownership.
        """
        automatic = bool(
            getattr(getattr(self, "h5viewer", None), "_run_writing", False)
            or getattr(getattr(self, "displayframe", None),
                       "_processing_active", False)
        )
        if automatic:
            logger.info(
                "Overlay axis changed during live processing; starting a new "
                "overlay. Current: %s; Selected: %s",
                overlay_grid_spec_summary(current_spec),
                overlay_grid_spec_summary(selected_spec),
            )
            return "reset"
        decision = self._ask_overlay_grid_mismatch(current_spec, selected_spec)
        if decision == "reset":
            self.h5viewer._browser_previous_context = None
        return decision

    def _cancel_overlay_grid_selection(self):
        restore = getattr(self.h5viewer, "restore_browser_context", None)
        if callable(restore):
            restore()

    def _rescope_frame_panel_to(self, name, first_frame=None):
        """Reset the Frames-panel / display state to a NEW scan identity.

        The scan boundary is FRAME-DRIVEN: update_data() detects a source_file scan
        change and calls this before appending the new scan's first frame, so it
        runs on the GUI thread INDEPENDENTLY of the mis-timed new_scan signal.  It
        does the light state clears only (the coalesced flush / new_scan's tail
        redraws), so it stays cheap on the frame hot path.  It carries the FULL
        destructive set: clearing _pending_frames / publication_store /
        _scan_info_rows alongside the index is required — otherwise a late new_scan
        (or this call) would strand the new scan's undrained frames.
        """
        self.scan.name = name
        # Wire the viewer to THIS scan's output HERE (driven by the frame stream),
        # NOT from the new_scan signal — set_file queues an async set_datafile that
        # RENAMES scan.name (scan.py:set_datafile), so calling it from an out-of-sync
        # new_scan would flip `cur` mid-scan and make the current scan's continuing
        # frames rescope + drop.  fname comes from the per-scan stash new_scan fills.
        _fname = getattr(self, "_scan_fname_cache", {}).get(name)
        if _fname:
            try:
                self._sync_h5viewer_save_dir(os.path.dirname(_fname), refresh=False)
                self.h5viewer.set_file(_fname, internal=True)
            except Exception:
                logger.debug("scan-rescope viewer rewire failed", exc_info=True)
        # Stamp which scan the panel is now scoped to (retained for diagnostics).
        self._frame_driven_scan_key = name
        self.frames.clear()
        self.frame_ids.clear()
        # A-Step (Phase 5): reset the per-scan FrameRecordStore here so it clears
        # on the frame-driven boundary too. The next streaming session installs
        # its own store; serial/live without one falls back to publication_store.
        self._frame_record_store = None
        self.publication_store.clear()
        # Undrained stash + scan_data row cache from the previous scan.
        self._pending_frames = {}
        self._scan_info_rows = {}
        # S-14: clear the overlay accumulator + pins when re-scoping to a name that
        # ALREADY has rows in the accumulator -- consecutive A->A OR A->B->A -- so
        # the new run's (name, frame_idx) row-ids never collide with the old run's.
        # The append-only accumulator's first-occurrence dedup would otherwise DROP
        # the new frames and leave the OLD run's curves under the new labels.
        # Deriving the seen names from the accumulator ITSELF (not the immediate-
        # prev key) is what catches the A->B->A case the prev==name guard missed;
        # a boundary to a name NOT yet in the accumulator appends (OV-6).
        _hist = getattr(self.displayframe, "_waterfall_history", None)
        _seen = {i[0] for i in (getattr(_hist, "ids", ()) or ())
                 if isinstance(i, tuple) and i}
        if name in _seen:
            try:
                # V2 owner: SAME_NAME_RERUN resets pins + history + the
                # pending-append queue together (the queue could otherwise
                # replay old-run labels against the new run's store).  The
                # display mirrors deliberately stay: the outgoing curves
                # linger until the new run's first frame draws.
                AccumulatorLifecycle(self.displayframe).reset(
                    LifecycleCause.SAME_NAME_RERUN,
                    site="_rescope_frame_panel_to[S-14 same-name re-run]")
            except Exception:
                logger.debug("S-14 re-run accumulator clear failed", exc_info=True)
        # Reset the Overlay/Waterfall accumulator only for incompatible grids.
        # Compatible scan boundaries append by design: cross-scan comparison is the
        # point of Overlay, while Clear remains the explicit relief valve.
        # (X1 Slice 3a: the per-selection HDF5 wavelength cache is GONE with
        # its deleted tier — nothing to reset here anymore.)
        try:
            if self._overlay_clear_needed_for_scan_boundary(first_frame=first_frame):
                self.displayframe.clear_overlay(
                    LifecycleCause.INCOMPATIBLE_GRID,
                    site="_rescope_frame_panel_to[OV-6 scan boundary]")
        except Exception:
            logger.debug("display overlay reset on scan rescope failed", exc_info=True)
        try:
            import pandas as pd
            with self.scan.scan_lock:
                self.scan.frames.index.clear()
                self.scan.frames._in_memory.clear()
                self.scan.scan_data = pd.DataFrame()
        except AttributeError:
            pass
        # Keep only the currently-rendered frame(s) so the outgoing scan's image
        # lingers until the new scan's first frame draws.
        keep = set()
        try:
            df = self.displayframe
            for lst in (df.idxs, df.idxs_1d, df.idxs_2d):
                keep.update(int(i) for i in (lst or ()))
        except Exception:
            pass
        try:
            with self.data_lock:
                for cache in (self.viewer_rows_1d, self.viewer_rows_2d):
                    for k in [k for k in list(cache.keys()) if int(k) not in keep]:
                        cache.pop(k, None)
            # Frame indices restart per scan: re-arm the raw self-heal neg cache.
            self.displayframe._raw_resolve_failed = set()
            self.displayframe._raw_full_shape = None
        except Exception:
            logger.debug("scan-rescope cache purge skipped", exc_info=True)
        # Point the viewer at the new scan (the frame-driven path has no new_scan
        # tail of its own).
        self.h5viewer.scan_name = name
        self.h5viewer.auto_last = True
        self.h5viewer.latest_idx = 1
        self._seed_append_processed_frame_browser(name)
        # Follow the Scans panel to the new scan AS SOON AS its frames appear (not
        # only at run-finish): the writer opened <name>.nxs at run start, so it is
        # already in the directory listing by this first-frame boundary.
        # update_scans re-runs its select-by-scan_name path (signals blocked, so no
        # re-load).  wrangler_finished re-runs it too as a fallback.
        self.h5viewer.update_scans()

    def _seed_append_processed_frame_browser(self, name):
        """Show append-skipped processed frames as soon as a run starts.

        Append mode already primes a read-only snapshot of the target file so
        the worker can skip processed rows before reading source images. Reuse
        that in-memory label set to seed the GUI frame list; no 1D/2D payloads
        are loaded here.
        """
        thread = getattr(getattr(self, "wrangler", None), "thread", None)
        if thread is None or getattr(thread, "write_mode", None) != "Append":
            return 0
        if getattr(thread, "xye_only", False):
            return 0
        snapshot = getattr(thread, "_append_skip_snapshot", None)
        try:
            existing = snapshot(name) if callable(snapshot) else (
                getattr(thread, "_append_skip_frames_by_scan", {}) or {}
            ).get(str(name), ())
            labels = sorted({int(label) for label in (existing or ())})
        except Exception:
            logger.debug("append frame-list seed failed for %s", name,
                         exc_info=True)
            return 0
        if not labels:
            return 0
        # Kickoff perf (XDART_PERF): split the MAIN-THREAD seed cost into the
        # index seed-loop (O(1) membership via _IndexedList + one O(N log N) sort)
        # vs update_data(force_rebuild=True) (a full Qt listData clear + re-insert
        # of N items).  This pins which dominates the run-start beachball on a
        # many-thousand-frame Append.  (The append-skip snapshot priming runs off
        # the main thread in the wrangler, so it is not measured here.)
        import time as _time
        _perf = bool(os.environ.get("XDART_PERF"))
        _t0 = _time.perf_counter() if _perf else 0.0
        try:
            with self.scan.scan_lock:
                index = self.scan.frames.index
                added = 0
                for label in labels:
                    if label not in index:
                        index.append(label)
                        added += 1
                if added:
                    index.sort()
            _t_seed = _time.perf_counter() if _perf else 0.0
            self.h5viewer.latest_idx = labels[-1]
            self.h5viewer.update_data(emit_update=False, force_rebuild=True)
            if _perf:
                _t_end = _time.perf_counter()
                logger.info(
                    "[PERF] append seed: n=%d added=%d seed_loop=%.0fms "
                    "update_data_rebuild=%.0fms total=%.0fms",
                    len(labels), added,
                    (_t_seed - _t0) * 1000.0,
                    (_t_end - _t_seed) * 1000.0,
                    (_t_end - _t0) * 1000.0,
                )
            return len(labels)
        except Exception:
            logger.debug("append frame-list seed skipped for %s", name,
                         exc_info=True)
            return 0

    def new_scan(self, name, fname, gi, incidence_motor, single_img,
                 series_average):
        """Connected to sigUpdateFile from wrangler. Called when a new
        scan is started.

        args:
            name: str, scan name
            fname: str, path to data file for scan
            incidence_motor: str, GI incidence-motor name (J1 rename;
                previously this slot was ``th_mtr``).  Qt signals are
                positional so the rename is purely cosmetic at this
                boundary — the value still flows through unchanged.
        """
        # The frame-panel boundary is FRAME-DRIVEN (update_data attributes each
        # frame to its scan via source_file), because Image Directory mode fires
        # new_scan signals in a BURST wildly out of sync with the frame stream — the
        # diagnostic log showed new_scan(LaB6) arriving while scan 03271005 was only
        # at frame 8, so a signal-keyed clear wiped 03271005's frames and it
        # restarted at 9.  So new_scan only touches scan.name / the panel / the
        # cursor when it MATCHES the scan the frame stream is currently rendering (a
        # same-name re-run or the first scan); an out-of-sync signal for a DIFFERENT
        # scan defers entirely to the frame stream (which clears + rescopes when
        # that scan's frames actually arrive).  Capture the pre-eager-set name.
        _prev_name = getattr(self.scan, "name", None)
        _in_sync = (name == _prev_name) or _prev_name in (None, "", "null_main")
        # Eager name set unblocks the synchronous render path (past the
        # ``scan.name == "null_main"`` guard) — but ONLY in sync, or an out-of-sync
        # signal would hijack scan.name and make the currently-arriving scan's
        # continuing frames spuriously rescope + drop.
        if _in_sync:
            self.scan.name = name
        # Stash THIS scan's output path so the FRAME-DRIVEN boundary
        # (_rescope_frame_panel_to) can wire the viewer when the scan's frames
        # actually arrive.  new_scan must NOT set_file here: its async set_datafile
        # renames scan.name, and out-of-sync signals would flip the current scan
        # mid-stream (the "flickers to LaB6 then reverts to Combi" symptom).
        self._scan_fname_cache = getattr(self, "_scan_fname_cache", {})
        self._scan_fname_cache[name] = fname
        run_active = bool(getattr(self, "_run_active", False))
        run_state = getattr(self, "_controls_v2_run_active", None)
        if callable(run_state):
            run_active = bool(run_state())
        if run_active:
            active_path = staticWidget._controls_v2_normalized_path(fname)
            if active_path:
                self._v2_active_result_paths = getattr(
                    self, "_v2_active_result_paths", set())
                self._v2_active_result_paths.add(active_path)
        # G1/T0-1: a new run is a new data identity — drop the wavelength
        # restored from whatever file was open before, synchronously (the
        # async file-thread set_datafile also clears, but frames can render
        # in the window before it lands).  getattr: tests drive this slot
        # with duck-typed scan stubs.
        _clear_wl = getattr(self.scan, '_clear_persisted_wavelength', None)
        if callable(_clear_wl):
            _clear_wl()
        # (X1 Slice 3a: the per-selection HDF5 wavelength cache is GONE with
        # its deleted tier — no display-side cache to reset here anymore.)
        # (Viewer wiring — save-dir + set_file — is done frame-driven in
        # _rescope_frame_panel_to, NOT here; see the stash above.)
        self.scan.gi = gi
        self.scan.incidence_motor = incidence_motor
        self.scan.single_img = single_img
        self.scan.series_average = series_average
        # New scan identity: drop any prior whole-scan stitch result and leave the
        # persistent stitch display.  The scan object is REUSED across scans, so
        # stale stitched_* would otherwise keep an old merge on screen; the
        # result-existence guard then returns the display to the per-frame view.
        self.scan.stitched_1d = None
        self.scan.stitched_2d = None
        self.displayframe.stitch_display_mode = None
        self._refresh_controls_v2_profile()
        # Propagate the wrangler-loaded mask (detector + user Mask File,
        # combined into flat indices) into the main scan so the
        # displayframe can overlay it on the raw image.  Without this,
        # self.scan.global_mask stays None after a scan and no mask
        # overlay is drawn (regression introduced by the v2 refactor).
        # Sync to the wrangler thread's CURRENT mask — including ``None``.
        # ``setup()`` rebuilds ``thread.mask`` every run from (detector mask |
        # Mask File); with no detector mask and the Mask File cleared it is
        # ``None``.  The old ``if ... is not None`` guard only ever SET the
        # mask, so removing the Mask File left the previous run's mask stale on
        # ``scan.global_mask`` and it kept rendering on the raw image (and in
        # the cake payload path).  Assign unconditionally so removal clears it;
        # only skip when there is no wrangler thread at all (test stubs / pre-
        # run), where the mask state is genuinely unknown.
        _wthread = getattr(self.wrangler, 'thread', None)
        if _wthread is not None:
            self.scan.global_mask = getattr(_wthread, 'mask', None)
            # Carry the full-res detector shape too, so the display can map the
            # gap mask into thumbnail coords without a resident full-res frame.
            self.scan.detector_shape = getattr(_wthread, 'detector_shape', None)
        # Also carry the run's intensity-threshold settings so the raw-image
        # preview can show the image AS INTEGRATED (mask + threshold).
        # mask_sentinel gates the always-on uint16-65535 saturation mask on the
        # display the same way it gates it in the integration.
        for _attr in ('apply_threshold', 'threshold_min', 'threshold_max',
                      'mask_sentinel'):
            try:
                setattr(self.scan, _attr, getattr(self.wrangler, _attr))
            except Exception:
                pass

        if self._controls_v2_enabled():
            self._controls_v2_ensure_native_int_defaults()
            self._controls_v2_apply_gi_config_to_scan()
        else:
            self.integratorTree.get_args('bai_1d')
            self.integratorTree.get_args('bai_2d')
            self.integratorTree.set_image_units()
        staticWidget._stamp_scan_data_reduction_config(self)

        # ── Panel + cursor: FRAME-STREAM-AUTHORITATIVE ─────────────────────────
        # Only act when this new_scan is IN SYNC with the scan the frame stream is
        # rendering.  An out-of-sync signal for a DIFFERENT scan (directory mode
        # bursts them out of order) must NOT clear or move the cursor — update_data's
        # frame-driven boundary owns the transition and clears + rescopes when this
        # scan's frames actually land (the config above is already applied, ready
        # for then).  This is what stops new_scan(LaB6) wiping scan 03271005's
        # in-flight frames.
        _pending_was = getattr(self, "_frame_driven_rescoped_pending", False)
        if os.environ.get("XDART_PERF"):
            logger.info("[PERF] new_scan boundary: live=%s prev=%r name=%r in_sync=%s "
                        "pending=%s index_len=%d",
                        getattr(self.h5viewer, "live_run_active", None), _prev_name, name,
                        _in_sync, _pending_was,
                        len(getattr(getattr(self.scan, "frames", None), "index", []) or []))
        if not _in_sync:
            return
        # Kickoff perf (XDART_PERF): time the MAIN-THREAD new_scan boundary reset
        # (rescope + list/scan rebuild + display/metadata refresh) over the full
        # index.  If this is small on the 3621 scan, the run-start beachball is NOT
        # the main thread -- it is the first-frame reduction warm-up (the
        # [DISPATCH] dispatch=~1s line, in the reduction thread), which is inherent
        # pyFAI integrator init, not a fixable O(N) here.
        import time as _time
        _perf = bool(os.environ.get("XDART_PERF"))
        _perf_t0 = _time.perf_counter() if _perf else 0.0
        # In sync: flush the previous scan's throttled update, then reset the panel.
        self._update_timer.stop()
        self._list_timer.stop()
        timer = getattr(self, "_reint_update_timer", None)
        if timer is not None:
            timer.stop()
        self._flush_pending_update()
        # Consume the frame-driven flag.  Clear ONLY for a same-name RE-RUN (the
        # frame stream can't detect it — same key); a LATE new_scan whose frames
        # already rescoped this run (pending) must NOT clear.  First scan (index
        # empty) clears harmlessly.  (A-Step's per-scan _frame_record_store reset
        # moved into _rescope_frame_panel_to so it fires on the frame-driven
        # boundary too.)
        self._frame_driven_rescoped_pending = False
        if not _pending_was:
            self._rescope_frame_panel_to(name)

        self.displayframe.set_axes()
        self.h5viewer.scan_name = name
        self.h5viewer.auto_last = True
        self.h5viewer.latest_idx = 1
        self.h5viewer.update_scans()
        self.h5viewer.update()
        # Refresh the metadata panel when a new scan starts.
        self.metawidget.update()
        if _perf:
            logger.info(
                "[PERF] new_scan boundary reset: elapsed=%.0fms index_len=%d",
                (_time.perf_counter() - _perf_t0) * 1000.0,
                len(getattr(getattr(self.scan, "frames", None), "index", [])
                    or []),
            )

    def update_scattering_geometry(self, gi):
        """Connected to sigUpdateGI from wrangler. Called when scattering
        geometry changes between transmission and GI

        args:
            gi: bool, flag for determining if in Grazing incidence
        """
        run_config_debug_log(
            logger,
            "scattering_geometry_update_enter",
            widget=self,
            origin="sigUpdateGI_or_direct",
            requested_gi=bool(gi),
        )
        scan = getattr(self, "scan", None)
        if scan is not None:
            scan.gi = gi
        # Update the integration-panel options now (the next run integrates in
        # the new geometry).  Do NOT rebuild the *display* axis combos here (C1):
        # the displayed plot is still the old-mode data, so switching the
        # plotUnit/imageUnit combos to GI/non-GI axes immediately is misleading.
        # The display combos rebuild via new_scan -> set_axes once a run actually
        # produces plots in the new mode.
        if self._controls_v2_enabled():
            self._controls_v2_ensure_native_int_defaults()
            self._refresh_controls_v2_profile(immediate=True)
        else:
            self.integratorTree.set_image_units()
        run_config_debug_log(
            logger,
            "scattering_geometry_update_exit",
            widget=self,
            origin="sigUpdateGI_or_direct",
            requested_gi=bool(gi),
        )

    def new_frame(self, frame_data):
        """Connected to sigUpdateFile from wrangler. Called when a new
        scan is started.

        args:
            name: str, scan name
            fname: str, path to data file for scan
        """
        frame = LiveFrame(idx=frame_data['idx'], map_raw=frame_data['map_raw'],
                         mask=frame_data['mask'], scan_info=frame_data['scan_info'],
                         poni_file=frame_data['poni_file'], static=self.scan.static, gi=self.scan.gi)
        frame.int_1d = frame_data['int_1d']
        frame.int_2d = frame_data['int_2d']
        frame.map_norm = frame_data['map_norm']
        # self.viewer_rows_2d[str(frame.idx)] = frame

    def start_wrangler(self):
        """Sets up wrangler, ensures properly synced args, and starts
        the wrangler.thread main method.
        """
        # i_qChi = np.zeros((1000, 1000), dtype=float)

        # Kickoff perf (XDART_PERF): the run-start beachball is the MAIN-THREAD
        # work here (Run click -> thread.start), and it scales with the PREVIOUS
        # (loaded) scan -- teardown of a long scan.  Split it so the offending
        # phase (controls run-state / wrangler.setup / enter_run_state) is named.
        import time as _time
        _perf = bool(os.environ.get("XDART_PERF"))
        _t0 = _time.perf_counter() if _perf else 0.0
        bump_run_config_debug_generation(self, "run")
        run_config_debug_log(
            logger,
            "run_start_enter",
            widget=self,
            origin="start_wrangler",
        )
        self._apply_controls_v2_run_state()
        run_config_debug_log(
            logger,
            "run_state_applied",
            widget=self,
            origin="start_wrangler",
        )
        self._sync_controls_v2_source_index()
        source_plan = None
        self.wrangler.source_run_plan = source_plan
        self.wrangler.source_spec = self._controls_v2_freeze_source_spec()
        self.wrangler.source_index_session = None
        self.wrangler.source_frame_count_snapshot = (
            self._controls_v2_freeze_container_frame_counts(source_plan))
        self.wrangler.source_pending_count = 0
        _t1 = _time.perf_counter() if _perf else 0.0
        self.wrangler.enabled(False)
        self.wrangler.setup()
        run_config_debug_log(
            logger,
            "wrangler_setup_exit",
            widget=self,
            origin="start_wrangler",
        )
        _t2 = _time.perf_counter() if _perf else 0.0
        self._configure_controls_v2_native_run_plan()
        self.h5viewer.auto_last = True

        # Live (non-batch) runs drive the display from the in-memory
        # per-frame hand-off.  Flag the run so the async set_datafile
        # repoints the file without a disk reload and data_reset doesn't
        # wipe the live caches (the multi-scan Eiger blank-plot fix).
        # Batch / XYE-only runs keep the original reload-on-new-file
        # behaviour — their final refresh reads frames from disk.
        live = not getattr(self.wrangler.thread, 'batch_mode', False)
        self.h5viewer.live_run_active = live
        self.h5viewer.file_thread.live_run = live
        # Int 1D (XYE) writes only .xye files (no .nxs); tell the file thread
        # not to try loading a .nxs that will never exist.  Cleared in
        # wrangler_finished so a later normal open still loads from disk.
        self.h5viewer.file_thread.no_nxs = getattr(
            self.wrangler.thread, 'xye_only', False)

        # Mark the run active through the single run-state owner (task #68):
        # the 2D panels keep their last-rendered content (instead of blanking)
        # while the run's frames arrive — matching the 1D plot's persistence —
        # AND the integration controls are disabled for the run (task #71).
        # Called synchronously here (GUI thread) so the controls lock before the
        # thread starts.  Cleared in wrangler_finished.
        _t3 = _time.perf_counter() if _perf else 0.0
        self._enter_run_state()
        if _perf:
            _t4 = _time.perf_counter()
            logger.info(
                "[PERF] start_wrangler: apply_run_state=%.0fms setup=%.0fms "
                "configure=%.0fms enter_run_state=%.0fms total=%.0fms "
                "prev_index_len=%d",
                (_t1 - _t0) * 1000.0, (_t2 - _t1) * 1000.0,
                (_t3 - _t2) * 1000.0, (_t4 - _t3) * 1000.0,
                (_t4 - _t0) * 1000.0,
                len(getattr(getattr(self.scan, "frames", None), "index", [])
                    or []),
            )

        run_config_debug_log(
            logger,
            "worker_start",
            widget=self,
            origin="start_wrangler",
        )
        self.wrangler.thread.start()

    def _perf_heartbeat_tick(self):
        # Update the max event-loop gap.  The gap between successive ~250ms ticks
        # is how long the GUI thread could not service the timer == how long the
        # event loop was blocked.  (XDART_PERF only; timer created in __init__.)
        import time
        now = time.perf_counter()
        last = self._perf_hb_last
        self._perf_hb_last = now
        if self._perf_hb_active and last and now > last:
            gap_ms = (now - last) * 1000.0
            if gap_ms > self._perf_hb_max_gap_ms:
                self._perf_hb_max_gap_ms = gap_ms
            # Log large stalls AS THEY HAPPEN, with elapsed-since-run-start, so the
            # window can be pinned (kickoff vs mid-run vs teardown) -- "max over
            # the run" alone can't locate a one-off freeze.
            if gap_ms >= 500.0:
                logger.info(
                    "[PERF] main-thread stall: gap=%.0f ms at t+%.1fs into the run",
                    gap_ms, now - getattr(self, "_perf_hb_window_t0", now))

    def _perf_hb_start_window(self):
        # Begin accumulating the max event-loop gap for a run (XDART_PERF only;
        # no-op when the heartbeat timer was not created).
        if getattr(self, "_perf_hb_timer", None) is None:
            return
        import time
        self._perf_hb_max_gap_ms = 0.0
        self._perf_hb_last = time.perf_counter()
        self._perf_hb_window_t0 = self._perf_hb_last
        self._perf_hb_active = True

    def _perf_hb_end_window(self):
        # Log the worst event-loop stall observed during the run (XDART_PERF only).
        # A large value here with all handler timings fast is the fingerprint of a
        # blocking call on the GUI thread (as in BB-1).
        if not getattr(self, "_perf_hb_active", False):
            return
        self._perf_hb_active = False
        logger.info(
            "[PERF] main-thread heartbeat: max event-loop gap during run = "
            "%.0f ms (probe interval 250 ms)",
            getattr(self, "_perf_hb_max_gap_ms", 0.0))

    def _select_finished_scan_row(self, nxs_file):
        """Point the Scans panel at the finished scan and re-run update_scans'
        select-by-scan_name path.  The .nxs FILE just written/displayed is the
        authoritative scan identity -- self.scan can lag the display in directory
        mode, and update_scans matches scan_name against the "<stem>.nxs" list
        entries (exact or "<stem>_N" fuzzy).  Best-effort; update_scans blocks
        signals so this re-select never triggers a reload."""
        try:
            if not nxs_file:
                return
            self.h5viewer.scan_name = os.path.splitext(
                os.path.basename(nxs_file))[0]
            self.h5viewer.update_scans()
        except Exception:
            logger.debug("run-end scan-row select skipped", exc_info=True)

    def wrangler_finished(self):
        """Called by the wrangler finished signal.

        RR-2: the run-end tail runs many unguarded GUI finalization steps
        (set_file, update_scans, selection, display catch-up).  Whatever any of
        them does, the frozen Source-card handoff must be dropped afterward — a
        raise part-way through must not leak the plan/session into the next
        between-run ``setup()`` seeding.  Always clear it in a ``finally``.

        Both calls use the explicit ``staticWidget.<method>(self)`` form (the
        same idiom the original tail already used for the clear) so the run-end
        unit tests, which drive this as ``MethodType(staticWidget.wrangler_finished,
        host)`` on a lightweight ``SimpleNamespace`` double, keep working — the
        double carries the attributes the body touches, not these helpers.
        """
        try:
            staticWidget._wrangler_finished_body(self)
        finally:
            staticWidget._clear_controls_v2_run_source_authority(self)

    def _wrangler_finished_body(self):
        """The run-end finalization body (see :meth:`wrangler_finished`). If the
        current scan matches the wrangler scan, allows for integration.
        """
        # End the run through the single run-state owner (task #68) BEFORE the
        # final flush so the 2D panels resume normal blank-on-missing for the
        # final frame.  Idempotent: a later integrator_thread_finished() (when
        # the scan matches, below) calls _exit_run_state again as a no-op.
        # Overlap guard (symmetric with integrator_thread_finished): a wrangler
        # can be started while a reintegrate is still running, and if it
        # finishes first the reintegrate's frames still need the controls
        # locked — only exit the shared run-state when the integrator run is
        # also done; its finished handler exits it otherwise.
        # Capture once: a concurrent reintegrate-all that is still WRITING means
        # we must neither exit the shared run-state (controls stay locked) nor
        # force a reload of a possibly half-written file (review finding — the
        # internal=True auto-loads below bypass the _run_writing guard).
        _reintegrate_running = self.integratorTree.integrator_thread.isRunning()
        browse_debug_log(
            logger,
            "runend_wrangler_finished_enter",
            reintegrate_running=_reintegrate_running,
            run_saw_frame=getattr(self, "_run_saw_frame", None),
            pending_update_idx=getattr(self, "_pending_update_idx", None),
            method=staticWidget._overlay_plot_method(self),
            auto_last=getattr(self.h5viewer, "auto_last", None),
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        if not _reintegrate_running:
            self._exit_run_state()
            browse_debug_log(
                logger,
                "runend_wrangler_after_exit_run_state",
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )

        # Flush any pending coalesced update so the final frame is shown.
        self._update_timer.stop()
        self._list_timer.stop()
        timer = getattr(self, "_reint_update_timer", None)
        if timer is not None:
            timer.stop()
        browse_debug_log(
            logger,
            "runend_wrangler_before_pending_flush",
            pending_update_idx=getattr(self, "_pending_update_idx", None),
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        self._flush_pending_update()
        browse_debug_log(
            logger,
            "runend_wrangler_after_pending_flush",
            pending_update_idx=getattr(self, "_pending_update_idx", None),
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )

        # End the live-run window before the end-of-batch reload below:
        # the auto-load set_file(generated_file) must run the full
        # set_datafile (disk reload) so frames/scan_data come back from
        # the finished file, and data_reset must be free to clear stale
        # caches again.
        self.h5viewer.live_run_active = False
        self.h5viewer.file_thread.live_run = False
        # Clear the XYE-only no-load flag so the end-of-batch auto-load (and any
        # later normal file open) reads the .nxs from disk again.
        self.h5viewer.file_thread.no_nxs = False
        browse_debug_log(
            logger,
            "runend_wrangler_after_live_flags_cleared",
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )

        self.thread_state_changed()
        self.wrangler.stop()

        # Auto-load the final file generated from the batch if applicable
        thread = self.wrangler.thread
        is_batch = getattr(thread, 'batch_mode', False)
        is_xye_only = getattr(thread, 'xye_only', False)

        processed_count = None
        for attr in ("files_processed", "_files_processed",
                     "_last_files_processed"):
            value = getattr(thread, attr, None)
            if value is None:
                continue
            try:
                processed_count = int(value)
            except (TypeError, ValueError):
                logger.debug("invalid files_processed value: %r", value)
            break
        try:
            append_skipped = int(
                getattr(thread, "_append_skip_without_reading", 0) or 0)
        except (TypeError, ValueError):
            append_skipped = 0
        all_skipped_append = (
            getattr(thread, "write_mode", None) == 'Append'
            and processed_count == 0
            and append_skipped > 0
        )
        append_config_failed = bool(
            getattr(thread, "_append_config_mismatch", False))
        finished_file = _finished_output_file(
            thread,
            self.wrangler,
            all_skipped_append=all_skipped_append,
        )
        browse_debug_log(
            logger,
            "runend_wrangler_counts",
            is_batch=is_batch,
            is_xye_only=is_xye_only,
            processed_count=processed_count,
            append_skipped=append_skipped,
            all_skipped_append=all_skipped_append,
            append_config_failed=append_config_failed,
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )

        if (is_batch and not is_xye_only and not _reintegrate_running
                and not (append_config_failed
                         and not getattr(self, '_run_saw_frame', True))):
            # Prefer the thread's fname — it's the source of truth for
            # where data was actually written. The widget-level
            # wrangler.fname is set in setup() before the thread runs
            # and may diverge (e.g. spec strips the ``_master`` suffix
            # from eiger master filenames inside the thread, so the
            # widget's fname ends with ``_master.nxs`` but the actual
            # scan output is ``<stem>.nxs``).
            generated_file = finished_file
            if generated_file and os.path.exists(generated_file):
                # Update directory display to point at the generated folder natively
                generated_dir = os.path.dirname(generated_file)
                if self.h5viewer.dirname != generated_dir:
                    self.h5viewer.dirname = generated_dir
                    self.h5viewer.update_scans()
                # Inform H5Viewer to load the file and set the flag to auto-select its last point.
                # internal=True: this is the app's own post-run wiring, not a user click —
                # the live run already pointed file_thread.fname at this same output file
                # (new_scan, set_file internal=True), so a non-internal call would hit the
                # same-file dedupe and silently skip the end-of-batch reload + select-last
                # (the "last frame doesn't show after batch" regression).  The run has ended
                # (_exit_run_state + live_run_active=False above), so bypassing the run guard
                # is safe.
                if all_skipped_append:
                    self._reconcile_h5viewer_frame_list_after_run(generated_file)
                self.h5viewer._auto_select_last_on_finish = True
                self.h5viewer.set_file(generated_file, internal=True)
                # Select the scan ROW too: batch shows the last FRAME, but the
                # Scans panel otherwise followed only when the directory changed
                # (and to a possibly-stale scan_name in directory batch).
                self._select_finished_scan_row(generated_file)

        # Append-mode feedback: a NON-batch run that processed 0 new frames
        # (Append over an already-complete scan/directory) displayed nothing
        # live.  Load the existing scan file and auto-select its LAST frame so
        # the user gets visual confirmation the run actually ran.  Batch already
        # auto-loads + selects-last above; XYE-only has no .nxs to load.
        if (not is_batch and not is_xye_only and not _reintegrate_running
                and not getattr(self, '_run_saw_frame', True)
                and not append_config_failed):
            existing_file = finished_file
            if existing_file and os.path.exists(existing_file):
                existing_dir = os.path.dirname(existing_file)
                if self.h5viewer.dirname != existing_dir:
                    self.h5viewer.dirname = existing_dir
                    self.h5viewer.update_scans()
                if all_skipped_append:
                    self._reconcile_h5viewer_frame_list_after_run(existing_file)
                self.h5viewer._auto_select_last_on_finish = True
                # internal=True for the same reason as the batch branch: force the
                # reload past the same-file dedupe (the run wired file_thread.fname
                # to this file) so the last-frame select-last actually fires.
                self.h5viewer.set_file(existing_file, internal=True)
                # THE 0-new-frames append fix: this branch showed the last FRAME but
                # never selected the scan ROW -- the scans_select for a live run is
                # gated on _run_saw_frame (False here), so nothing followed the Scans
                # panel to the finished scan.  Select it by the displayed file.
                self._select_finished_scan_row(existing_file)

        # Live run (saw frames): the streaming path can leave the lazy frame index
        # empty OR partial if frame production outruns the GUI coalescer.  Now that
        # the writer has closed the file, rebuild ONLY the lazy frame index from it
        # and force the H5Viewer Frames list to rebuild from that complete index.
        # Batch already gets this via its end-of-batch reload above.  Skip when a
        # reintegrate is still running (don't repoint frames mid-reintegrate) or
        # when the run saw 0 frames (the append-feedback branch already reloaded).
        if (not is_batch and not is_xye_only and not _reintegrate_running
                and getattr(self, '_run_saw_frame', True)):
            worker_output = (getattr(self.wrangler.thread, 'fname', None)
                             or getattr(self.wrangler, 'fname', None))
            written = _last_processed_output_file(thread, worker_output)
            try:
                restore_processed_output = (
                    written is not None
                    and worker_output is not None
                    and os.path.normcase(os.path.abspath(os.fspath(written)))
                    != os.path.normcase(os.path.abspath(os.fspath(worker_output)))
                )
            except (TypeError, ValueError):
                restore_processed_output = False
            output_processed_count = _processed_count_for_output(
                thread,
                written,
                processed_count,
            )
            browse_debug_log(
                logger,
                "runend_wrangler_before_live_reconcile",
                written_file=written,
                worker_output=worker_output,
                restore_processed_output=restore_processed_output,
                processed_count=processed_count,
                output_processed_count=output_processed_count,
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            indexed = self._reconcile_h5viewer_frame_list_after_run(written)
            browse_debug_log(
                logger,
                "runend_wrangler_after_live_reconcile",
                written_file=written,
                indexed=indexed,
                processed_count=processed_count,
                output_processed_count=output_processed_count,
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            if (output_processed_count is not None
                    and indexed < output_processed_count):
                logger.warning(
                    "post-live indexed fewer frames than processed: "
                    "indexed=%d processed=%d file=%s",
                    indexed, output_processed_count, written,
                )
            if indexed:
                self._apply_integration_control_state()
            if restore_processed_output and written and os.path.exists(written):
                # Stop can catch Directory mode after it initialized the next
                # scan but before that scan processed a frame.  Reconciliation
                # above restores the final positive output's frame labels
                # immediately; force one idle-file reload as well so raw/cake
                # hydrate from that same file instead of the empty ahead scope.
                logger.info(
                    "post-live: restoring last processed output %s "
                    "(worker ended at unprocessed %s)",
                    os.path.basename(written),
                    os.path.basename(worker_output),
                )
                self.h5viewer._auto_select_last_on_finish = True
                self.h5viewer.set_file(written, internal=True)
            # scans_select_after_run: update_scans' select-by-scan_name path ran at
            # scan START (before <name>.nxs existed), so the Scans panel kept the
            # PRIOR scan highlighted.  Re-select the finished scan by the file just
            # written -- its .nxs now exists.  update_scans blocks signals, so this
            # re-select does not re-trigger a load.
            self._select_finished_scan_row(written)

        # Finalize shared run UI state when this scan owns the display.  A normal
        # wrangler finish must preserve Overlay/Waterfall history; only a true
        # reintegration invalidates it.  Skip finalization while reintegration is
        # still running so controls remain locked until its own finished slot.
        if (staticWidget._owns_finished_run(self)
                and not self.integratorTree.integrator_thread.isRunning()):
            browse_debug_log(
                logger,
                "runend_wrangler_before_finalize",
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            self._finalize_processing_run(
                reset_overlay=False,
                origin="wrangler",
            )
            browse_debug_log(
                logger,
                "runend_wrangler_after_finalize",
                **_runend_waterfall_history_fields(
                    getattr(self, "displayframe", None)),
            )
            # PERF-3 Option A (was Item 2): the live incremental paint can lag, so
            # the Overlay/Waterfall ends short (e.g. 3476/3621).  Item-2's direct
            # _render_overlay_full_scan reselect FAILED — the lagged tail frames
            # were dropped at the cap-128 hand-off and never published, so a
            # resident reselect can't paint them, and the run-end selection-collapse
            # echoes killed its deferred render.  Instead arm a ONE-SHOT post-
            # quiescence catch-up that waits out those echoes and does ONE real
            # show_all() (async disk load of the missing tail).  See
            # docs/design/runend_overlay_catchup_spec_jul2026.md.
            # LIVE saw-frames runs only (the spec's hook — same guard cluster as
            # the reconcile above): batch/XYE and the 0-new-frames append end
            # with their own reload + select-last recovery, and arming there
            # would flip the end-of-run selection to all frames for runs that
            # never had a lagged live paint to catch up.
            if (not is_batch and not is_xye_only
                    and getattr(self, '_run_saw_frame', True)):
                staticWidget._arm_runend_overlay_catchup(self)
        else:
            self.wrangler.enabled(True)

        # Kickoff/teardown perf (XDART_PERF): this forced full gc.collect() at
        # run-finish is the strong suspect for the Replace-only run-end beachball.
        # In Replace the prior scan's display caches are reset, dereferencing its
        # whole frame graph (amplified by the 1 GiB browse cap holding N frames vs
        # 512), so this synchronous cycle collection walks+frees N frames on the
        # GUI thread (the update_all note removed exactly this anti-pattern for the
        # same UI-stutter reason).  Append PRESERVES the caches, so the graph stays
        # referenced and there is nothing to collect -> fast.  Timed to confirm.
        if os.environ.get("XDART_PERF"):
            import time as _time
            _gc_t0 = _time.perf_counter()
            _gc_n = gc.collect()
            logger.info(
                "[PERF] wrangler_finished gc.collect: elapsed=%.0fms collected=%d",
                (_time.perf_counter() - _gc_t0) * 1000.0, _gc_n)
        else:
            gc.collect()

        # Run-end liveness readout (XDART_PERF): report the worst event-loop stall
        # seen during this run (placed after teardown so it captures it too).
        self._perf_hb_end_window()

        # XYE-only batch (Int 1D (XYE)): there is no .nxs to auto-load, so the
        # block above skipped the end-of-batch reload.  Show the folder of
        # generated iq_/itth_ files (written to <scan_dir>/<scan_name> by
        # save_1d) in XYE Viewer mode so the outputs are actually listed.
        # Done last so integrator_thread_finished()'s refresh doesn't undo it.
        if is_batch and is_xye_only:
            try:
                xye_dir = os.path.join(
                    os.path.dirname(self.scan.data_file), self.scan.name)
                if os.path.isdir(xye_dir):
                    self.h5viewer.dirname = xye_dir
                    # Same path the XYE Viewer combo takes: set viewer_mode,
                    # panels, selection mode, and refresh listScans.
                    self._on_viewer_mode_changed('xye')
                    # Auto-select the most recently *written* file (by mtime),
                    # not the name-last one, so the final pattern from this run
                    # is shown without a manual click.
                    self.h5viewer.select_most_recent_scan_entry()
                else:
                    logger.debug(
                        'XYE-only batch finished but output dir not found: %s',
                        xye_dir)
            except Exception:
                logger.debug(
                        'Could not show XYE output folder after batch',
                        exc_info=True)
        browse_debug_log(
            logger,
            "runend_wrangler_finished_exit",
            **_runend_waterfall_history_fields(getattr(self, "displayframe", None)),
        )
        # RR-2: the authoritative-source clear is NOT here — it runs in the
        # wrangler_finished() ``finally`` so it survives a raise in the tail above.

    def _clear_controls_v2_run_source_authority(self) -> None:
        """Drop the frozen Source-card handoff after run-end consumers finish."""
        wrangler = getattr(self, "wrangler", None)
        if wrangler is None:
            return
        wrangler.source_run_plan = None
        wrangler.source_spec = None
        wrangler.source_index_session = None
        wrangler.source_frame_count_snapshot = {}
        wrangler.source_pending_count = 0
        thread = getattr(wrangler, "thread", None)
        if thread is not None:
            thread.source_run_plan = None
            thread.source_spec = None
            thread.source_index_session = None
            thread.source_frame_count_snapshot = {}
            thread.source_pending_count = 0

    def _on_viewer_mode_changed(self, viewer_mode_str):
        """Enable or disable the integrator panel and update h5viewer for viewer mode.

        Args:
            viewer_mode_str: 'image', 'xye', or '' (normal mode)
        """
        viewer_mode = viewer_mode_str or None  # '' → None
        is_viewer = viewer_mode is not None
        is_file_viewer = viewer_mode in ('image', 'xye')
        from PySide6.QtWidgets import QAbstractItemView

        scans = self.h5viewer.ui.listScans
        prev_suspend = getattr(
            self.h5viewer, '_suspend_scan_selection_loads', False,
        )
        was_blocked = scans.blockSignals(True)
        self.h5viewer._suspend_scan_selection_loads = True
        try:
            self.h5viewer.viewer_mode = viewer_mode
            tree = getattr(self.wrangler, 'tree', None)
            if tree is not None:
                # Only the actual file-Viewer *processing* modes disable the
                # wrangler inputs.  Int 1D (XYE) is a processing mode whose
                # display auto-switches to XYE to list the generated files, but
                # its inputs (Image File / mask / …) must stay enabled so the
                # user can keep processing.  Key off the processing-mode combo,
                # not the display viewer_mode; fall back to the display when the
                # combo is unavailable.
                mode_text = ''
                try:
                    mode_text = self.controls.current_mode()
                except Exception:
                    mode_text = ''
                # Tree stays enabled in viewers: processing groups are
                # disabled per-group by the wrangler, while Project Folder /
                # Save Path remain usable (they drive the file browser).
                tree.setEnabled(True)
            # Per-mode integration control enable/dim (C3/C4): disable the 1-D/2-D
            # integration panels in viewers, keep Calibrate / Make Mask enabled.
            self._apply_integration_control_state()
            # Relax the Frames panel width so NeXus dataset labels aren't
            # clipped; restored on exit / other modes.
            self.h5viewer._apply_frames_panel_width(viewer_mode)
            if hasattr(self, "metawidget"):
                self.metawidget.viewer_mode = viewer_mode
            # Give displayframe a reference to the wrangler for mask/threshold
            self.displayframe._wrangler = self.wrangler if is_viewer else None
            # In viewer mode, disable New/Save (keep Open Folder and Export)
            self.h5viewer.actionNewFile.setEnabled(not is_viewer)
            self.h5viewer.actionSaveDataAs.setEnabled(not is_viewer)
            # XYE viewer: ExtendedSelection so arrow keys browse one file at a
            # time with the plot following (shift+arrow / shift+click = range);
            # _XyeOverlayInputFilter layers on modifier-free, plotMethod-aware
            # accumulation (toggle/extend on click+arrow in Overlay/Waterfall/
            # Sum/Average).  Others: single select.  Start clean — show the
            # current row only, never a default overlay.
            if viewer_mode == 'xye':
                scans.setSelectionMode(QAbstractItemView.ExtendedSelection)
                scans.clearSelection()
            else:
                scans.setSelectionMode(QAbstractItemView.SingleSelection)
            # Configure display panels for the viewer mode
            self.displayframe._viewer_is_xdart = False
            self.displayframe.set_viewer_display_mode(viewer_mode)
            if is_viewer:
                save_path = getattr(self.wrangler, 'h5_dir', None)
                current_raw = str(getattr(self.h5viewer, 'dirname', '') or '')
                current_dir = (
                    os.path.abspath(os.path.expanduser(current_raw))
                    if current_raw else ''
                )
                default_dir = os.path.abspath(os.path.expanduser(
                    str(getattr(self, 'local_path', get_fname_dir())),
                ))
                if save_path and (not current_dir or current_dir == default_dir):
                    self._sync_h5viewer_save_dir(save_path, refresh=False)
                self.h5viewer.enter_viewer_mode_cleanup()
            else:
                self.h5viewer.cancel_pending_loads()
                if hasattr(self, 'scan') and hasattr(self.scan, 'global_mask'):
                    self.scan.global_mask = None
                self.displayframe.clear_display_state()
            # Refresh scan list to show/hide appropriate file types
            self.h5viewer.update_scans()
        finally:
            self.h5viewer._suspend_scan_selection_loads = prev_suspend
            scans.blockSignals(was_blocked)
        self._fit_controls_height()
        self._refresh_controls_v2_profile()

    def latest_frame(self, checked=None, *, emit_update=True):
        """Advances to last frame in data list, updates displayframe, and
        set auto_last to True.

        Wraps the cursor advance in ``blockSignals`` and invokes
        ``data_changed()`` explicitly — same pattern as
        ``H5Viewer.update_data``.  Without the block, the
        ``ClearAndSelect`` cursor move would fire
        ``itemSelectionChanged`` → ``data_changed`` → ``sigUpdate`` →
        ``set_data`` → a redundant render-authority request on top of whatever
        the caller does next (every call site that drives ``latest_frame`` follows
        it with its own display refresh).
        """
        self.h5viewer.auto_last = True
        if self.h5viewer.ui.listData.count() <= 1:
            return

        lw = self.h5viewer.ui.listData
        idx = self.h5viewer.latest_idx
        lw.blockSignals(True)
        try:
            if isinstance(idx, int):
                items = lw.findItems(str(idx), QtCore.Qt.MatchExactly)
                for item in items:
                    self.h5viewer.set_current_frame(item)
            else:
                last_row = lw.count() - 1
                if last_row >= 0:
                    item = lw.item(last_row)
                    if item is not None:
                        try:
                            self.h5viewer.latest_idx = int(item.text())
                        except ValueError:
                            self.h5viewer.latest_idx = item.text()
                    self.h5viewer.set_current_frame(last_row)
        finally:
            lw.blockSignals(False)
        if emit_update:
            self.h5viewer.data_changed()

    def raw_to_tiff(self):
        self.popup_detector_options()

    def popup_detector_options(self):
        """
        Popup Qt Window to select options for Waterfall Plot
        Options include Y-axis unit and number of points to skip
        """
        if self.detector_dialog.layout() is None:
            self.setup_detector_options_widget()

        self.detector_dialog.show()

    def setup_detector_options_widget(self):
        """
        Setup y-axis option for Waterfall plot
        Setup first image and step size for wf and overlay plots
        """
        layout = QtWidgets.QGridLayout()
        self.detector_dialog.setLayout(layout)

        self.detector_widget = QCombo()
        accept_button = QtWidgets.QPushButton('Okay')
        cancel_button = QtWidgets.QPushButton('Cancel')

        layout.addWidget(QtWidgets.QLabel('Choose Detector'), 0, 0)
        layout.addWidget(self.detector_widget, 1, 0)
        layout.addWidget(accept_button, 2, 1)
        layout.addWidget(cancel_button, 2, 2)

        detectors = ['Pilatus 1M', 'Pilatus 100k', 'Pilatus 300kw']
        self.detector_widget.addItems(detectors)

        accept_button.clicked.connect(self.set_detector)
        cancel_button.clicked.connect(self.close_detector_popup)

    def close_detector_popup(self):
        self.detector_dialog.close()

    def set_detector(self):
        detector_name = self.detector_widget.currentText()
        self.detector = pyFAI.detector_factory(name=detector_name)
        self.detector_dialog.close()

        from xdart.utils.browse import browse_start_dir, remember_browse_path
        rawFile, _ = QFileDialog().getOpenFileName(
            filter='RAW (*.raw)',
            caption='Choose Raw File',
            dir=browse_start_dir(),
            options=QFileDialog.DontUseNativeDialog
        )

        if os.path.isfile(rawFile):
            remember_browse_path(rawFile)
            img = get_img_data(rawFile, self.detector, return_float=False)
            if img is not None:
                tifFile = os.path.splitext(rawFile)[0] + '.tif'
                imageio.imwrite(tifFile, img)
                message = f'{os.path.basename(tifFile)} saved'
            else:
                message = 'File does not match detector..'
        else:
            message = 'Invalid Raw File'

        out_dialog = QMessageBox()
        out_dialog.setText(message)
        out_dialog.exec()
