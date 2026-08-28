# -*- coding: utf-8 -*-
"""OV acceptance-contract invariant harness (QW-3, robustness design §4.3).

ONE offscreen driver for scripted display-event sequences against the REAL
production seams — the same wiring the ledgered OV tests drive, unified:

* real ``PublicationStore`` (+ its real heavy-window eviction),
* real ``publication_from_live_frame`` publications,
* real ``ScanDisplayController().compute_state`` → ``compute_display_state``,
* real ``PublicationDisplayAdapter.plot_payload`` → ``_overlay_waterfall_payload``
  → ``append_row`` → ``accumulate_waterfall`` (the seam under test — no fakes),
* real ``displayFrameWidget`` lifecycle methods bound onto the driver widget
  (``pin_current_slice_cut`` / ``_clear_pinned_slice_cuts`` / ``clear_overlay``),
* the renderer store-back mirrored exactly (``_draw_payload``: plot_history →
  ``_waterfall_history``, display/overlaid ids → ``overlaid_idxs``).

The widget itself is the same ambient-context duck the production-wired OV
tests use (test_aggregation_wiring / test_frame_publication): Qt combos and
spinboxes are tiny mutable stand-ins because the contract under test lives in
the adapter + accumulator, not in QComboBox.  Every seam the contract names is
the real object.

After EVERY event the harness renders and asserts the OV acceptance contract
(live_findings_ledger, "Acceptance test that covers the OV family"):

INV-1  Accumulator row count is MONOTONIC non-decreasing, except at an
       explicitly-allowed reset cause: CLEAR, INCOMPATIBLE_GRID (coarse key or
       concrete sampled-axis change), REINTEGRATE, SAME_NAME_RERUN,
       METHOD_SWITCH — the
       :class:`~xrd_tools.session.display_logic.LifecycleCause` enum of
       V2's single ``AccumulatorLifecycle`` owner, adopted 1:1.  A
       display-unit flip RELABELS, never resets; a REAL norm-channel change
       RE-SCALES at draw, never resets (V1 Stage 4: S-16 dissolved —
       NORM_CHANGE stays retired and must not return).  The transient live
       slice "current" cut (the OV-7b/7c sentinel row) is excluded from the
       count: it is a preview that pin-absorption legitimately drops.
INV-2  ``history.x`` is one strictly-monotonic grid, row width == len(x),
       one unit at a time.
INV-3  No constant-clamped rows: ``np.ptp(row) > 0`` for every accumulated
       row (the BL-6 disjoint-domain interp failure signature).  Harness
       frames are built with a peak on a ramp so a genuine row is NEVER
       constant — a flat row can only come from a clamp.
INV-4  Pinned slice cuts ⊆ history rows; pins and history reset TOGETHER.

A violation raises :class:`InvariantViolation` carrying the full numbered
event trace, so a failure names the exact step sequence — the substrate the
future V6 fuzzer shrinks on.

RESET ACCOUNTING (V2, exact — the two QW-3 soundness gaps are CLOSED):
every owner reset appends a ``LifecycleReset(cause, site)`` to the widget's
``_accumulator_lifecycle_log``; after EVERY event the harness drains that
log and

* (gap b — attribution by code-site, not arming order) each drained entry
  must match the ONE armed ``expect_reset`` window — the armed cause is
  checked against the cause the owner actually LOGGED at the code site, so
  a reset can never be mis-attributed to whichever expectation happened to
  be armed first;
* (gap a — armed ≠ consumed windows are no longer silent) a count decrease
  with no logged cause is an INV-1 violation (an un-owned wipe), a logged
  reset with no armed window is a violation, and an armed window is either
  consumed by its logged reset or must be explicitly retired via
  :meth:`OVHarness.cancel_expected_reset` (sequences end with
  :meth:`OVHarness.assert_lifecycle_settled`).  Because the owner logs even
  when a follow-up render rebuilds the accumulator to its prior count (an
  identical-grid reintegrate), "allowed ≠ required" ambiguity is gone: the
  log proves whether the reset fired.

NORM_CHANGE (cause-vocabulary history): RETIRED at V1 Stage 4 — since the
canonical-grid flip the accumulator stores acquisition-native rows and the
norm divides at draw, so a REAL channel change re-renders with the new
scaling and never resets (S-16 dissolved); it must NOT return as an allowed
reset cause and has no lifecycle enum member.

NOT a test module — import it: ``from tests.xdart.ov_harness import OVHarness``.
"""

from __future__ import annotations

from collections import deque
from threading import RLock
from types import MethodType, SimpleNamespace

import numpy as np
from xrd_tools.session import HydrationPurpose

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

from xdart.modules.frame_publication import (
    PublicationStore,
    publication_from_live_frame,
)
from xdart.gui.tabs.static_scan.display_constants import plotUnits, imageUnits
from xdart.gui.tabs.static_scan.display_controllers import ScanDisplayController
from xdart.gui.tabs.static_scan.display_logic import (
    AccumulatorLifecycle,
    LifecycleCause,
    Mode,
    accumulator_clearable,
)
from xdart.gui.tabs.static_scan.display_overlay_utils import (
    LIVE_SLICE_PROJECTION_ID,
)
from xdart.gui.tabs.static_scan.display_publication import (
    PublicationDisplayAdapter,
)

# V2 lifecycle causes — the owner's enum adopted 1:1 (design §5 V2): the only
# causes allowed to shrink the accumulator.  NORM_CHANGE stays retired (V1
# Stage 4) and must not return.
CLEAR = LifecycleCause.CLEAR
INCOMPATIBLE_GRID = LifecycleCause.INCOMPATIBLE_GRID
REINTEGRATE = LifecycleCause.REINTEGRATE
SAME_NAME_RERUN = LifecycleCause.SAME_NAME_RERUN
METHOD_SWITCH = LifecycleCause.METHOD_SWITCH


class InvariantViolation(AssertionError):
    """An OV-contract invariant failed; the message carries the event trace."""


class _Ctl:
    """Mutable stand-in for the one Qt combo/spinbox/checkbox surface the
    adapter reads (currentText/currentIndex/value/isChecked/isEnabled/text,
    plus the combo-item surface the V3 Share-Axis chain drives:
    count/itemText/setCurrentIndex/blockSignals/setEnabled/setChecked).
    Ambient context only — never the seam under test."""

    def __init__(self, *, text="", index=0, value=0.0, checked=False,
                 enabled=True, items=None):
        self._text = text
        self._index = index
        self._value = value
        self._checked = checked
        self._enabled = enabled
        self._items = list(items) if items is not None else None

    def currentText(self):
        return self._text

    def currentIndex(self):
        return self._index

    def text(self):
        return self._text

    def value(self):
        return self._value

    def isChecked(self):
        return self._checked

    def isEnabled(self):
        return self._enabled

    # ── combo-item surface (V3 Share-Axis chain) ──
    def count(self):
        return len(self._items) if self._items is not None else 0

    def itemText(self, index):
        if self._items is not None and 0 <= int(index) < len(self._items):
            return self._items[int(index)]
        return ""

    def setCurrentIndex(self, index):
        self._index = int(index)
        if self._items is not None and 0 <= self._index < len(self._items):
            self._text = self._items[self._index]

    def blockSignals(self, block):
        return False

    def setEnabled(self, enabled):
        self._enabled = bool(enabled)

    def setChecked(self, checked):
        self._checked = bool(checked)


class _RecordingPlot:
    """Recording stand-in for the bottom 1D plot the Share-Axis align
    touches.  The align/skip DECISION under test is the real
    ``displayFrameWidget._align_plot_under_cake`` (bound onto the harness
    widget); this records what it does to the panel — ``enableAutoRange``
    re-arms and ``setXRange`` impositions — so sequences can assert the 1D
    never takes a foreign range.  The geometry legs beyond the identity
    guard early-return on the duck (no cake window), which is exactly the
    scope of the harness: identity, not pixels."""

    def __init__(self):
        self.autorange_calls = []
        self.xrange_calls = []

    def enableAutoRange(self, *args, **kwargs):
        self.autorange_calls.append((args, dict(kwargs)))

    def setXRange(self, *args, **kwargs):
        self.xrange_calls.append((args, dict(kwargs)))

    def getViewBox(self):
        return self


def _is_live_sentinel(row_id):
    """True for the transient live slice "current" row (OV-7b/7c sentinel)."""
    return (
        isinstance(row_id, tuple)
        and len(row_id) >= 3
        and isinstance(row_id[2], tuple)
        and len(row_id[2]) >= 1
        and row_id[2][0] == LIVE_SLICE_PROJECTION_ID
    )


class OVHarness:
    """Scripted event driver + step-hook invariant checker (spec §4.3).

    One harness instance = one scripted sequence.  Every event method mutates
    the widget/store exactly the way the production event does, then renders
    through the real controller→adapter→accumulator path and re-checks the
    acceptance contract.  ``expect_reset(cause)`` arms the ONE window under
    which an owner-logged reset is legal; the logged cause must MATCH the
    armed cause (exact code-site attribution — see the module docstring),
    and anything else raises.
    """

    #: default synthetic grid (npt chosen small for speed; > any decimation
    #: threshold is unnecessary — the invariant reads the history, not paint).
    NPT = 48
    X_RANGE = (1.0, 5.0)

    def __init__(self, *, method="Overlay", slice_mode=False, cake_mode=False,
                 max_heavy_items=None, max_items=None, wavelength_m=1e-10,
                 scan_name="scanA"):
        # cake_mode: INT_2D layout WITHOUT the live slice — the plain
        # cake-over-1D arrangement the V3 Share-Axis identity sequences
        # drive (the cake payload builds; the 1D stays the radial overlay).
        self.mode = (Mode.INT_2D if (slice_mode or cake_mode)
                     else Mode.INT_1D)
        self.slice_mode = bool(slice_mode)
        self.cake_mode = bool(cake_mode)
        self._wavelength_m = wavelength_m
        self._heavy_cap = max_heavy_items
        self._norm = {"channel": None}
        self.store = PublicationStore(
            max_items=max_items, max_heavy_items=max_heavy_items)
        self.trace = []
        self.hydration_requests = []
        self.repaint_requests = []
        self.last_cake_payload = None       # V3: last-built cake ImagePayload
        self._pending_reset = None          # armed allowed-reset cause or None
        self.resets_observed = []           # [(cause, event, site)] owner-logged
        self.resets_cancelled = []          # explicitly retired armed windows
        self._persistent_floor = 0
        self._grid = {}                     # per-scan publish defaults
        self.widget = self._build_widget(scan_name)
        self._configure_scan(scan_name, npt=self.NPT, x_range=self.X_RANGE)

    # ── widget assembly ────────────────────────────────────────────────

    def _build_widget(self, scan_name):
        from xdart.gui.tabs.static_scan.display_frame_widget import (
            displayFrameWidget,
        )

        if self.slice_mode:
            axis_entry = {"source": "2d", "axis": "radial",
                          "slice_axis": "χ (°)"}
        elif self.cake_mode:
            # INT_2D radial rows are '1d_2d' (1D result + cake radial merge)
            # when the slice is off — the production set_axes shape.
            axis_entry = {"source": "1d_2d", "axis": "radial",
                          "slice_axis": "χ (°)"}
        else:
            axis_entry = {"source": "1d", "axis": "radial", "slice_axis": None}
        ui = SimpleNamespace(
            plotMethod=_Ctl(text="Overlay"),
            plotUnit=_Ctl(text=plotUnits[0], index=0,
                          items=[plotUnits[0], plotUnits[1]]),
            imageUnit=_Ctl(text=imageUnits[0]),
            shareAxis=_Ctl(checked=False),
            slice=_Ctl(text="χ (c/w)", checked=self.slice_mode, enabled=True),
            slice_center=_Ctl(value=0.0),
            slice_width=_Ctl(value=1.0),
        )
        widget = SimpleNamespace(
            publication_store=self.store,
            viewer_mode=None,
            data_lock=RLock(),
            viewer_rows_1d={},
            viewer_rows_2d={},
            frame_ids=[],
            overlaid_idxs=[],
            frame_names=[],
            plot_data=[np.zeros(0), np.zeros(0)],
            plot_data_range=[[0, 0], [0, 0]],
            _waterfall_history=None,
            _accumulator_lifecycle_log=[],   # V2 owner reset log (drained per event)
            display_generation=1,
            _processing_active=False,
            normChannel=None,
            scan=self._make_scan(scan_name),
            ui=ui,
            # A Q↔2θ plot-unit flip needs both combo entries resolvable.
            # The explicit unit_key per row feeds the real _plot_axis_key
            # (production populates it lazily; the read exists at
            # DFW._plot_axis_key) so _share_axis_plot_index can match the
            # cake's rendered key without Qt combo internals.
            _plot_axis_info=(
                dict(axis_entry, unit_key="q_A^-1"),
                dict(axis_entry, unit_key="2th_deg"),
            ),
            # V3 Share-Axis identity surface: rendered-payload stashes (the
            # renderer store-backs mirrored in _render_once) + the recording
            # bottom plot the real align acts on.
            _payload_x_axis_label=None,
            _cake_rendered_axis_key=None,
            _cake_rendered_axis_labels=None,
            _share_link_on=False,
            _share_axis_syncing=False,
            _last_plot_unit=0,
            plot=_RecordingPlot(),
            _overlay_hydrated_pending_append_labels=deque(),
            _pinned_slice_cuts={},
            _slice_2d_data_ready=lambda: True,
            get_normChannel=lambda: self._norm["channel"],
            # X1 Slice 3a: accept the selected-frame opt-in keyword the three
            # current-frame call sites now pass (the harness stub stays
            # scan-constant either way).
            _get_wavelength=lambda ref, **_kw: self._wavelength_m,
            _request_frame_hydration=(
                lambda label, *, purpose=HydrationPurpose.FULL:
                self.hydration_requests.append((int(label), purpose))),
            request_current_selection_repaint=(
                lambda **kw: self.repaint_requests.append(kw)),
        )
        widget.normalize = self._normalize
        # The lifecycle seams are the REAL widget methods, bound unbound-style
        # exactly like the ledgered OV-7b/7c tests.  The V3 Share-Axis chain
        # (identity key → plot-index match → silent re-point → align guard)
        # is likewise the REAL methods: the seam under test is the
        # payload-identity decision, driven end-to-end.
        for name in ("pin_current_slice_cut", "_slice_pin_selection",
                     "_slice_pin_trace_name", "_pinned_slice_cut_recipes",
                     "_clear_pinned_slice_cuts", "clear_overlay",
                     "_current_image_axis_key", "_plot_axis_key",
                     "_share_axis_plot_index", "_set_plot_unit_index_silently",
                     "_apply_share_axis_state",
                     "_share_axis_rendered_units_agree",
                     "_align_plot_under_cake", "_active_bottom_plot",
                     "_active_bottom_window"):
            setattr(widget, name,
                    MethodType(getattr(displayFrameWidget, name), widget))
        return widget

    @staticmethod
    def _make_scan(name):
        return SimpleNamespace(
            name=name, data_file=f"{name}.nxs", gi=False,
            bai_1d_args={}, bai_2d_args={},
            scan_lock=RLock(),
            frames=SimpleNamespace(index=[]),
        )

    def _configure_scan(self, name, *, npt, x_range, unit="q_A^-1"):
        # ``unit`` is the scan's acquisition-NATIVE integration unit;
        # ``x_range`` (and ``peak``) are expressed in it.  V1 D1: cross-scan
        # native units may differ while the reset key stays unit-blind.
        self._grid[name] = {"npt": int(npt), "x_range": tuple(x_range),
                            "unit": str(unit)}

    def _normalize(self, data, metadata):
        """Monitor normalization keyed on the CURRENT channel — the same
        channel ``get_normChannel`` reports, so a real S-16 channel change
        rescales rows exactly like the production widget."""
        channel = self._norm["channel"]
        data = np.asarray(data, dtype=float)
        if not channel:
            return data
        value = (metadata or {}).get(channel, 1.0) or 1.0
        return data / float(value)

    # ── synthetic frames (never constant → INV-3 stays meaningful) ─────

    def _make_frame(self, label, *, peak=None, npt=None, x_range=None,
                    amplitude=100.0, empty=False):
        grid = self._grid[self.widget.scan.name]
        npt = int(npt if npt is not None else grid["npt"])
        x_range = tuple(x_range if x_range is not None else grid["x_range"])
        unit = grid.get("unit", "q_A^-1")
        if empty:
            radial = np.zeros(0, dtype=np.float32)
            int_1d = IntegrationResult1D(
                radial=radial, intensity=radial.copy(), sigma=None,
                unit=unit)
            int_2d = None
        else:
            radial = np.linspace(x_range[0], x_range[1], npt)
            if peak is None:
                peak = x_range[0] + 0.25 * (x_range[1] - x_range[0])
            # A Gaussian peak on a gentle ramp: every genuine row has
            # np.ptp > 0, so INV-3 can only trip on a real clamp.
            profile = (
                amplitude * np.exp(-0.5 * ((radial - peak) / 0.15) ** 2)
                + np.linspace(1.0, 2.0, npt)
                + float(label)
            )
            int_1d = IntegrationResult1D(
                radial=radial.astype(np.float32),
                intensity=profile.astype(np.float32),
                sigma=np.ones(npt, dtype=np.float32),
                unit=unit)
            chi = np.asarray([-20.0, -10.0, 0.0, 10.0, 20.0],
                             dtype=np.float32)
            # (radial, azimuthal) orientation; per-χ scaling keeps every slice
            # window's projected row non-constant AND distinguishable.
            cake = (profile[:, None]
                    * (1.0 + 0.1 * np.arange(chi.size))[None, :])
            int_2d = IntegrationResult2D(
                radial=radial.astype(np.float32), azimuthal=chi,
                intensity=cake.astype(np.float32),
                unit=unit, azimuthal_unit="chi_deg")
        return SimpleNamespace(
            idx=int(label), int_1d=int_1d, int_2d=int_2d,
            map_raw=None, mask=None, gi=False, gi_2d={}, thumbnail=None,
            bg_raw=0,
            scan_info={"i0": 2.0, "i1": 4.0, "monitor": 1.0},
            source_file=f"{self.widget.scan.name}_{label}.tif",
            source_frame_idx=int(label),
        )

    # ── render + step hook ─────────────────────────────────────────────

    def render(self, reason="repaint"):
        """One production render tick: real controller state → real adapter
        payload → renderer store-back → invariant check.  Also the bare
        "repaint" event (norm refresh echo, imageUnit echo, run-end repaint).
        """
        return self._step(f"render({reason})")

    def _render_once(self):
        # Pre-build Share-Axis sync — the production order (_update_impl
        # calls _apply_share_axis_state BEFORE build_payload for INT modes)
        # so a checked Share-Axis silently re-points plotUnit off the CAKE
        # payload identity before the 1D payload is built.
        self.widget._apply_share_axis_state()
        state = ScanDisplayController().compute_state(self.widget, self.mode)
        pending = tuple(
            self.widget._overlay_hydrated_pending_append_labels or ())
        labels = tuple(dict.fromkeys(
            (*pending, *state.selected_ids, *state.render_ids)))
        adapter = PublicationDisplayAdapter(
            self.store, widget=self.widget, labels=labels)
        payload = adapter.plot_payload(state)
        # Renderer store-back, mirroring _draw_payload exactly.
        if payload is not None:
            display_ids = getattr(payload, "display_ids", None)
            overlaid = getattr(payload, "overlaid_ids", None)
            self.widget.overlaid_idxs = list(
                display_ids if display_ids is not None
                else (overlaid if overlaid else state.render_ids))
            history = getattr(payload, "plot_history", None)
            if history is not None:
                self.widget._waterfall_history = history
            if getattr(payload, "traces", ()):
                axis = payload.axis_x
                # V3: the rendered 1D identity store-back (_draw_payload).
                self.widget._payload_x_axis_label = (axis.label, axis.unit)
        # V3: the 2D panel — build the cake payload through the same real
        # adapter and mirror _draw_image_payload's rendered-identity
        # store-back (INT_2D states carry a CAKE_2D panel; INT_1D states
        # yield None and, matching render's clear delegate, drop the stash).
        cake = adapter.cake_image(state)
        self.last_cake_payload = cake
        if cake is not None:
            self.widget._cake_rendered_axis_key = getattr(
                cake, "rendered_axis_key", None)
            ax, ay = cake.axis_x, cake.axis_y
            self.widget._cake_rendered_axis_labels = (
                (ax.label, ax.unit), (ay.label, ay.unit))
            # _draw_image_payload tail: _on_plotUnit_changed re-applies the
            # share state after the cake identity store-back.
            self.widget._apply_share_axis_state()
        else:
            # clear_binned_view resets the stash when the cake blanks.
            self.widget._cake_rendered_axis_key = None
            self.widget._cake_rendered_axis_labels = None
        return state, payload

    def _step(self, event):
        self.trace.append(event)
        state, payload = self._render_once()
        self._check_invariants(event)
        return state, payload

    def _fail(self, event, message):
        steps = "\n".join(
            f"  {k + 1:3d}. {step}" for k, step in enumerate(self.trace))
        raise InvariantViolation(
            f"{message}\n  at event: {event}\n  event trace:\n{steps}")

    # ── the acceptance contract, checked after EVERY event ─────────────

    @property
    def history(self):
        return self.widget._waterfall_history

    @property
    def persistent_count(self):
        """Accumulated rows excluding the transient live "current" sentinel."""
        history = self.history
        if history is None:
            return 0
        return sum(1 for i in history.ids if not _is_live_sentinel(i))

    @property
    def pending_reset(self):
        """The armed-but-unconsumed allowed-reset cause (or ``None``).
        Windows are consumed by OWNER-LOGGED resets whose cause matches
        (exact attribution), not by count arithmetic; a window whose site
        never fires stays armed and must be retired explicitly via
        :meth:`cancel_expected_reset` — sequences end with
        :meth:`assert_lifecycle_settled`."""
        return self._pending_reset

    def _accumulator_clearable(self):
        """Observable accumulator state exists — THE owner's own predicate
        (imported), so armed windows can never drift from logged resets."""
        return accumulator_clearable(self.widget)

    def expect_reset(self, cause):
        """Arm ONE allowed-reset window (a :class:`LifecycleCause` member).
        The next owner-logged reset must carry exactly this cause; arming a
        second window over an unconsumed one raises (consume or
        :meth:`cancel_expected_reset` first)."""
        cause = LifecycleCause(cause)
        assert self._pending_reset is None, (
            f"expect_reset({cause}): window {self._pending_reset} is still "
            f"armed — consume it or cancel_expected_reset() first")
        self._pending_reset = cause

    def cancel_expected_reset(self):
        """Explicitly retire an armed-but-unneeded window (soundness gap a:
        'allowed ≠ required' is never silent — a sequence either observes
        its owner-logged reset or cancels the expectation)."""
        assert self._pending_reset is not None, (
            "cancel_expected_reset(): no armed reset window")
        self.resets_cancelled.append(self._pending_reset)
        self._pending_reset = None

    def assert_reset_observed(self, cause):
        """The armed ``cause`` actually reset the accumulator — consumed by
        the owner-logged entry from the code site (an allowed cause is
        permitted to reset; sequences assert it DID)."""
        cause = LifecycleCause(cause)
        assert self._pending_reset is None, (
            f"expected a {cause} reset but none was observed "
            f"(window still armed)")
        assert self.resets_observed and self.resets_observed[-1][0] == cause, (
            f"last observed reset {self.resets_observed[-1:]} != {cause}")

    def assert_lifecycle_settled(self):
        """No armed-but-unconsumed reset window remains (sequence end)."""
        assert self._pending_reset is None, (
            f"armed reset window {self._pending_reset} was never consumed "
            f"nor cancelled (soundness gap a)")

    def check_invariants(self, event="explicit check"):
        """Public step-hook (also callable mid-test)."""
        self._check_invariants(event)

    def _check_invariants(self, event):
        history = self.history
        if history is not None and history.count:
            x = np.asarray(history.x, dtype=float)
            rows = np.atleast_2d(np.asarray(history.rows, dtype=float))
            # INV-2: one strictly-monotonic grid, rows on it, one unit.
            if x.ndim != 1 or x.size == 0:
                self._fail(event, f"INV-2: degenerate grid shape {x.shape}")
            if rows.shape[1] != x.size:
                self._fail(
                    event,
                    f"INV-2: row width {rows.shape[1]} != grid {x.size}")
            if x.size > 1:
                dx = np.diff(x)
                if not (np.all(dx > 0) or np.all(dx < 0)):
                    self._fail(event, "INV-2: history.x is not strictly "
                                      "monotonic (mixed-unit grid?)")
            if not isinstance(history.unit, str):
                self._fail(event, f"INV-2: non-string unit {history.unit!r}")
            # INV-3: no constant-clamped rows (BL-6 signature).
            for k in range(rows.shape[0]):
                finite = rows[k][np.isfinite(rows[k])]
                if finite.size == 0 or np.ptp(finite) <= 0:
                    self._fail(
                        event,
                        f"INV-3: constant/empty row for id {history.ids[k]} "
                        f"(disjoint-domain clamp?)")
        # INV-1 (V2 exact accounting): drain the owner's reset log; every
        # logged reset must consume the ONE armed window with a MATCHING
        # cause (attribution by code site), and any count decrease must be
        # backed by a logged reset — an un-owned wipe is a violation.
        log = getattr(self.widget, "_accumulator_lifecycle_log", None)
        entries = list(log or ())
        if entries:
            del log[:]
        count = self.persistent_count
        for entry in entries:
            if self._pending_reset is None:
                self._fail(
                    event,
                    f"owner reset {entry.cause} at site {entry.site!r} "
                    f"with NO armed expectation")
            if entry.cause != self._pending_reset:
                self._fail(
                    event,
                    f"armed {self._pending_reset} but the owner logged "
                    f"{entry.cause} at site {entry.site!r} (attribution "
                    f"mismatch)")
            self.resets_observed.append((entry.cause, event, entry.site))
            self._pending_reset = None
        if count < self._persistent_floor and not entries:
            self._fail(
                event,
                f"INV-1: accumulator shrank {self._persistent_floor} → "
                f"{count} with no owner-logged cause (un-owned wipe)")
        self._persistent_floor = count
        # INV-4: pins ⊆ history (and they reset together).
        pin_ids = set(self.widget._pinned_slice_cuts or {})
        history_ids = set(history.ids) if history is not None else set()
        missing = pin_ids - history_ids
        if missing:
            self._fail(
                event,
                f"INV-4: pinned cuts missing from history: {sorted(missing, key=repr)}")

    # ── event vocabulary (each event = mutate → render → check) ────────

    def publish(self, label, *, peak=None, npt=None, x_range=None,
                empty=False, select="append"):
        """A processed frame arrives (live tick).  ``select='append'`` mirrors
        live auto-last selection growth; ``'only'`` a browse click landing on
        the fresh frame; ``None`` publishes without touching the selection."""
        frame = self._make_frame(
            label, peak=peak, npt=npt, x_range=x_range, empty=empty)
        self.store.upsert(publication_from_live_frame(
            frame, scan_key=self.widget.scan.name))
        index = self.widget.scan.frames.index
        if int(label) not in index:
            index.append(int(label))
        if select == "append":
            if str(label) not in self.widget.frame_ids:
                self.widget.frame_ids.append(str(label))
        elif select == "only":
            self.widget.frame_ids[:] = [str(label)]
        return self._step(
            f"publish(label={label}, scan={self.widget.scan.name}, "
            f"npt={npt or self._grid[self.widget.scan.name]['npt']}, "
            f"empty={empty}, select={select})")

    def evict(self, label):
        """Slide the REAL heavy window until ``label``'s heavy payload is
        thinned — via the public ``set_max_heavy_items`` resize, the exact
        production enforcement path.  Oldest-first is the true store
        semantic, so frames older than ``label`` thin with it."""
        key = int(label)

        def _resident(lbl):
            pub = self.store.get(lbl)
            view = getattr(pub, "view", None)
            return bool(view is not None
                        and (getattr(view, "has_1d", False)
                             or getattr(view, "has_2d", False)))

        for _ in range(len(self.store.labels()) + 1):
            if not _resident(key):
                break
            heavy = [l for l in self.store.labels() if _resident(l)]
            if not heavy:
                break
            self.store.set_max_heavy_items(len(heavy) - 1)
        self.store.set_max_heavy_items(self._heavy_cap)
        assert not _resident(key), f"evict({label}): label still resident"
        return self._step(f"evict(label={label})")

    def click(self, label):
        """Browse click: the selection becomes exactly this frame."""
        self.widget.frame_ids[:] = [str(label)]
        return self._step(f"click(label={label})")

    def select(self, labels):
        """Multi-select (ctrl/shift click set)."""
        self.widget.frame_ids[:] = [str(l) for l in labels]
        return self._step(f"select(labels={list(labels)})")

    def deselect_all(self):
        """Empty selection (whitespace click) → the OV-5 empty repaint."""
        self.widget.frame_ids[:] = []
        return self._step("deselect_all()")

    def unit_toggle(self):
        """Flip the 1D display unit Q↔2θ.  A RELABEL by contract: the
        accumulator keeps every row; only the grid labels convert."""
        ui = self.widget.ui
        to_tth = ui.plotUnit._index == 0
        ui.plotUnit._index = 1 if to_tth else 0
        ui.plotUnit._text = plotUnits[ui.plotUnit._index]
        return self._step(
            f"unit_toggle(→ {'2θ' if to_tth else 'Q'})")

    def image_unit_toggle(self):
        """Flip the 2D display unit combo (an OV-5 repaint source for the
        1D overlay — must never wipe it)."""
        ui = self.widget.ui
        ui.imageUnit._text = (
            imageUnits[1] if ui.imageUnit._text == imageUnits[0]
            else imageUnits[0])
        ui.imageUnit._index = imageUnits.index(ui.imageUnit._text)
        return self._step(f"image_unit_toggle(→ {ui.imageUnit._text})")

    def share_axis(self, on=True):
        """Check/uncheck Share Axis.  The chain under test is the REAL V3
        one: ``_apply_share_axis_state`` (run by every render, production
        order) keys the silent plotUnit re-point off the CAKE payload's
        rendered identity via ``_current_image_axis_key``, and the real
        ``_align_plot_under_cake`` (drivable directly on ``widget``) only
        engages when the rendered payload identities agree.  The geometric
        link itself (``_set_share_link``) early-returns on the duck (no Qt
        viewbox); sequences model the linked state with
        ``widget._share_link_on = True``."""
        self.widget.ui.shareAxis._checked = bool(on)
        return self._step(f"share_axis(on={bool(on)})")

    def norm_change(self, *, real, channel=None):
        """Normalization event.  ``real=True`` switches the channel — since
        V1 Stage 4 a pure re-render: the accumulator is PRESERVED and every
        row re-scales at draw under the new channel (S-16 dissolved;
        NORM_CHANGE is retired as a reset cause, so INV-1 now enforces
        no-shrink across it).  ``real=False`` is the repaint echo
        (refresh_norm_channels re-applying the same channel)."""
        if real:
            previous = self._norm["channel"]
            if channel is None:
                channel = "i1" if previous != "i1" else "i0"
            self._norm["channel"] = channel
            return self._step(
                f"norm_change(real=True, {previous!r} → {channel!r})")
        return self._step("norm_change(real=False)")

    def rescope(self, new_scan, *, compatible=True, npt=None, x_range=None,
                native_unit=None, clear_store=True):
        """Scan boundary.  The store resets (production scan boundary);
        the accumulator must NOT — unless the NEW grid is incompatible, in
        which case the reset happens at the first new-grid row and is
        allowed (OV-6), or the new name ALREADY has rows in the accumulator
        (consecutive A→A or A→B→A), in which case the S-14 rule resets
        through the owner NOW with SAME_NAME_RERUN so the new run's
        (name, frame_idx) row-ids never collide with the old run's —
        mirroring ``_rescope_frame_panel_to``'s seen-set derivation from
        the accumulator itself.  ``native_unit`` sets the new scan's
        acquisition-native integration unit (default: inherit the current
        scan's) — with a matching npt the reset key stays compatible and
        the V1 D1 canonicalization path is exercised; ``x_range`` must then
        be given in that unit."""
        grid = self._grid[self.widget.scan.name]
        if npt is None:
            npt = grid["npt"] if compatible else grid["npt"] + 7
        if x_range is None:
            x_range = grid["x_range"]
        if native_unit is None:
            native_unit = grid.get("unit", "q_A^-1")
        if clear_store:
            self.store.clear()
        # S-14 (production order: before the OV-6 grid rule): seen names come
        # from the accumulator ITSELF, catching A→B→A.
        _hist = self.widget._waterfall_history
        _seen = {i[0] for i in (getattr(_hist, "ids", ()) or ())
                 if isinstance(i, tuple) and i}
        same_name_rerun = new_scan in _seen and self._accumulator_clearable()
        if same_name_rerun:
            self.expect_reset(SAME_NAME_RERUN)
            AccumulatorLifecycle(self.widget).reset(
                LifecycleCause.SAME_NAME_RERUN,
                site="harness.rescope[S-14 same-name re-run]")
        self.widget.scan = self._make_scan(new_scan)
        self._configure_scan(new_scan, npt=npt, x_range=x_range,
                             unit=native_unit)
        self.widget.frame_ids[:] = []
        self.widget.display_generation += 1
        concrete_compatible = bool(
            compatible
            and int(npt) == int(grid["npt"])
            and str(native_unit) == str(grid.get("unit", "q_A^-1"))
            and np.allclose(
                np.asarray(x_range, dtype=float),
                np.asarray(grid["x_range"], dtype=float),
                rtol=1e-5, atol=1e-8, equal_nan=True)
        )
        if not concrete_compatible and not same_name_rerun:
            # The S-14 reset (if any) already emptied the accumulator, so an
            # incompatible concrete grid then builds fresh without a second reset.
            self.expect_reset(INCOMPATIBLE_GRID)
        return self._step(
            f"rescope(scan={new_scan}, compatible={compatible}, npt={npt}, "
            f"x_range={tuple(x_range)}, native_unit={native_unit}, "
            f"same_name_rerun={same_name_rerun})")

    def reintegrate_finish(self, *, npt=None):
        """Same-scan reintegrate pass completing: the store resets and every
        indexed frame republishes recomputed, then the accumulator resets
        through the REAL owner site — ``clear_overlay(REINTEGRATE)``, the
        exact call ``integrator_thread_finished`` makes — and the follow-up
        render rebuilds from the CURRENT selection.  V2: the reset is
        owner-logged, so an identical regrid that rebuilds straight back to
        the same count still consumes its window exactly (gap a closed: the
        log proves the reset fired even without a net count decrease)."""
        if npt is not None:
            self._grid[self.widget.scan.name]["npt"] = int(npt)
        self.store.begin_reintegrate()
        try:
            for label in list(self.widget.scan.frames.index):
                frame = self._make_frame(label)
                self.store.upsert(publication_from_live_frame(
            frame, scan_key=self.widget.scan.name))
        finally:
            self.store.end_reintegrate()
        self.widget.display_generation += 1
        if self._accumulator_clearable():
            self.expect_reset(REINTEGRATE)
        self.widget.clear_overlay(LifecycleCause.REINTEGRATE)
        return self._step(f"reintegrate_finish(npt={npt})")

    def hydration_complete(self, label, *, stale=False):
        """An async hydration lands: the store re-gains the full publication.
        ``stale=True`` mirrors a completion whose generation lapsed — it joins
        the pending-append queue (OV-3/BR-2 path) instead of the selection."""
        frame = self._make_frame(label)
        self.store.upsert(publication_from_live_frame(
            frame, scan_key=self.widget.scan.name))
        if stale:
            queue = self.widget._overlay_hydrated_pending_append_labels
            if int(label) not in queue:
                queue.append(int(label))
        return self._step(
            f"hydration_complete(label={label}, stale={stale})")

    def pin_current_cut(self):
        """Freeze the live slice c/w as a pinned overlay row — the REAL
        ``displayFrameWidget.pin_current_slice_cut``."""
        assert self.slice_mode, "pin_current_cut() needs slice_mode=True"
        pinned = self.widget.pin_current_slice_cut()
        return self._step(f"pin_current_cut(pinned={pinned})")

    def move_live_cut(self, center, width=None):
        """Spin the live slice center/width — the mutable current cut."""
        assert self.slice_mode, "move_live_cut() needs slice_mode=True"
        self.widget.ui.slice_center._value = float(center)
        if width is not None:
            self.widget.ui.slice_width._value = float(width)
        return self._step(f"move_live_cut(center={center}, width={width})")

    def method_switch(self, method):
        """Flip the plotMethod combo.  Leaving Overlay/Waterfall for a
        non-accumulating method (Single/Sum/Average) drops the accumulator
        trio — history + pins + pending queue TOGETHER — through the owner
        on the next render (METHOD_SWITCH: the ``plot_payload``
        non-accumulating branch; the legacy ``_on_plotMethod_changed``
        wipes carry the same cause).  Switching back re-enters accumulation
        FRESH from the current selection — never resurrecting the
        pre-switch stack."""
        ui = self.widget.ui
        prev = ui.plotMethod._text
        ui.plotMethod._text = str(method)
        if (str(method) not in ("Overlay", "Waterfall")
                and prev in ("Overlay", "Waterfall")
                and self._accumulator_clearable()):
            self.expect_reset(METHOD_SWITCH)
        return self._step(f"method_switch({prev} → {method})")

    def clear(self):
        """The Clear button — the REAL ``clear_overlay`` through the V2
        owner (history + pins + pending queue together), the canonical
        allowed reset."""
        if self._accumulator_clearable():
            self.expect_reset(CLEAR)
        self.widget.clear_overlay()
        return self._step("clear()")
