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
       explicitly-allowed reset cause: CLEAR, INCOMPATIBLE_GRID (reset_key
       change), REINTEGRATE, NORM_CHANGE (a REAL channel change).  A
       display-unit flip RELABELS, never resets.  The transient live slice
       "current" cut (the OV-7b/7c sentinel row) is excluded from the count:
       it is a preview that pin-absorption legitimately drops.
INV-2  ``history.x`` is one strictly-monotonic grid, row width == len(x),
       one unit at a time.
INV-3  No constant-clamped rows: ``np.ptp(row) > 0`` for every accumulated
       row (the BL-6 disjoint-domain interp failure signature).  Harness
       frames are built with a peak on a ramp so a genuine row is NEVER
       constant — a flat row can only come from a clamp.
INV-4  Pinned slice cuts ⊆ history rows; pins and history reset TOGETHER.

A violation raises :class:`InvariantViolation` carrying the full numbered
event trace, so a failure names the exact step sequence — the substrate the
future V6 fuzzer shrinks on.  The allowed-reset causes are named with the V2
lifecycle-cause vocabulary (``CLEAR`` / ``INCOMPATIBLE_GRID`` / ``REINTEGRATE``
/ ``NORM_CHANGE``) so V2's single AccumulatorLifecycle owner can adopt this
harness's cause accounting unchanged.

NOT a test module — import it: ``from tests.xdart.ov_harness import OVHarness``.
"""

from __future__ import annotations

from collections import deque
from threading import RLock
from types import MethodType, SimpleNamespace

import numpy as np

from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D

from xdart.modules.frame_publication import (
    PublicationStore,
    publication_from_live_frame,
)
from xdart.gui.tabs.static_scan.display_constants import plotUnits, imageUnits
from xdart.gui.tabs.static_scan.display_controllers import ScanDisplayController
from xdart.gui.tabs.static_scan.display_logic import Mode
from xdart.gui.tabs.static_scan.display_overlay_utils import (
    LIVE_SLICE_PROJECTION_ID,
)
from xdart.gui.tabs.static_scan.display_publication import (
    PublicationDisplayAdapter,
)

# V2 lifecycle-cause names (design §5 V2): the only causes allowed to shrink
# the accumulator.  SAME_NAME_RERUN arrives with V2; the harness models the
# four causes the current code exercises.
CLEAR = "CLEAR"
INCOMPATIBLE_GRID = "INCOMPATIBLE_GRID"
REINTEGRATE = "REINTEGRATE"
NORM_CHANGE = "NORM_CHANGE"


class InvariantViolation(AssertionError):
    """An OV-contract invariant failed; the message carries the event trace."""


class _Ctl:
    """Mutable stand-in for the one Qt combo/spinbox/checkbox surface the
    adapter reads (currentText/currentIndex/value/isChecked/isEnabled/text).
    Ambient context only — never the seam under test."""

    def __init__(self, *, text="", index=0, value=0.0, checked=False,
                 enabled=True):
        self._text = text
        self._index = index
        self._value = value
        self._checked = checked
        self._enabled = enabled

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
    acceptance contract.  ``expect_reset(cause)`` arms the ONE allowance under
    which the next observed count decrease is legal; anything else raises.
    """

    #: default synthetic grid (npt chosen small for speed; > any decimation
    #: threshold is unnecessary — the invariant reads the history, not paint).
    NPT = 48
    X_RANGE = (1.0, 5.0)

    def __init__(self, *, method="Overlay", slice_mode=False,
                 max_heavy_items=None, max_items=None, wavelength_m=1e-10,
                 scan_name="scanA"):
        self.mode = Mode.INT_2D if slice_mode else Mode.INT_1D
        self.slice_mode = bool(slice_mode)
        self._wavelength_m = wavelength_m
        self._heavy_cap = max_heavy_items
        self._norm = {"channel": None}
        self.store = PublicationStore(
            max_items=max_items, max_heavy_items=max_heavy_items)
        self.trace = []
        self.hydration_requests = []
        self.repaint_requests = []
        self._pending_reset = None          # armed allowed-reset cause or None
        self.resets_observed = []           # [(cause, event)] consumed windows
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
        else:
            axis_entry = {"source": "1d", "axis": "radial", "slice_axis": None}
        ui = SimpleNamespace(
            plotMethod=_Ctl(text="Overlay"),
            plotUnit=_Ctl(text=plotUnits[0], index=0),
            imageUnit=_Ctl(text=imageUnits[0]),
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
            display_generation=1,
            _processing_active=False,
            normChannel=None,
            scan=self._make_scan(scan_name),
            ui=ui,
            # A Q↔2θ plot-unit flip needs both combo entries resolvable.
            _plot_axis_info=(dict(axis_entry), dict(axis_entry)),
            _overlay_hydrated_pending_append_labels=deque(),
            _pinned_slice_cuts={},
            _slice_2d_data_ready=lambda: True,
            get_normChannel=lambda: self._norm["channel"],
            _get_wavelength=lambda ref: self._wavelength_m,
            _request_frame_hydration=(
                lambda label, *, purpose="full":
                self.hydration_requests.append((int(label), purpose))),
            request_current_selection_repaint=(
                lambda **kw: self.repaint_requests.append(kw)),
        )
        widget.normalize = self._normalize
        # The lifecycle seams are the REAL widget methods, bound unbound-style
        # exactly like the ledgered OV-7b/7c tests.
        for name in ("pin_current_slice_cut", "_slice_pin_selection",
                     "_slice_pin_trace_name", "_pinned_slice_cut_recipes",
                     "_clear_pinned_slice_cuts", "clear_overlay"):
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

    def _configure_scan(self, name, *, npt, x_range):
        self._grid[name] = {"npt": int(npt), "x_range": tuple(x_range)}

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
        if empty:
            radial = np.zeros(0, dtype=np.float32)
            int_1d = IntegrationResult1D(
                radial=radial, intensity=radial.copy(), sigma=None,
                unit="q_A^-1")
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
                unit="q_A^-1")
            chi = np.asarray([-20.0, -10.0, 0.0, 10.0, 20.0],
                             dtype=np.float32)
            # (radial, azimuthal) orientation; per-χ scaling keeps every slice
            # window's projected row non-constant AND distinguishable.
            cake = (profile[:, None]
                    * (1.0 + 0.1 * np.arange(chi.size))[None, :])
            int_2d = IntegrationResult2D(
                radial=radial.astype(np.float32), azimuthal=chi,
                intensity=cake.astype(np.float32),
                unit="q_A^-1", azimuthal_unit="chi_deg")
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
        """The armed-but-unconsumed allowed-reset cause (or ``None``).  An
        allowed cause is permitted to reset, not required to — e.g. a
        same-grid reintegrate dedupes instead of shrinking, leaving its
        window armed.  Sequences end (or assert on this) before relying on
        INV-1 again."""
        return self._pending_reset

    def expect_reset(self, cause):
        """Arm ONE allowed-reset window (ledger causes only).  The next count
        decrease consumes it; an unconsumed window is reported by
        :meth:`assert_reset_observed`."""
        assert cause in (CLEAR, INCOMPATIBLE_GRID, REINTEGRATE, NORM_CHANGE), (
            f"not a ledger-allowed reset cause: {cause!r}")
        self._pending_reset = cause

    def assert_reset_observed(self, cause):
        """The armed ``cause`` actually reset the accumulator (an allowed
        cause is permitted to reset — sequences assert it DID)."""
        assert self._pending_reset is None, (
            f"expected a {cause} reset but none was observed "
            f"(window still armed)")
        assert self.resets_observed and self.resets_observed[-1][0] == cause, (
            f"last observed reset {self.resets_observed[-1:]} != {cause}")

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
        # INV-1: monotonic persistent count except one armed allowed cause.
        count = self.persistent_count
        if count < self._persistent_floor:
            if self._pending_reset is not None:
                self.resets_observed.append((self._pending_reset, event))
                self._pending_reset = None
            else:
                self._fail(
                    event,
                    f"INV-1: accumulator shrank {self._persistent_floor} → "
                    f"{count} with no allowed reset cause armed")
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
        self.store.upsert(publication_from_live_frame(frame))
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
        return self._step(f"image_unit_toggle(→ {ui.imageUnit._text})")

    def norm_change(self, *, real, channel=None):
        """Normalization event.  ``real=True`` switches the channel — the ONE
        norm event allowed to reset (S-16).  ``real=False`` is the repaint
        echo (refresh_norm_channels re-applying the same channel) — never a
        reset."""
        if real:
            previous = self._norm["channel"]
            if channel is None:
                channel = "i1" if previous != "i1" else "i0"
            self._norm["channel"] = channel
            self.expect_reset(NORM_CHANGE)
            return self._step(
                f"norm_change(real=True, {previous!r} → {channel!r})")
        return self._step("norm_change(real=False)")

    def rescope(self, new_scan, *, compatible=True, npt=None, x_range=None,
                clear_store=True):
        """Scan boundary.  The store resets (production scan boundary);
        the accumulator must NOT — unless the NEW grid is incompatible, in
        which case the reset happens at the first new-grid row and is
        allowed (OV-6)."""
        grid = self._grid[self.widget.scan.name]
        if npt is None:
            npt = grid["npt"] if compatible else grid["npt"] + 7
        if x_range is None:
            x_range = grid["x_range"]
        if clear_store:
            self.store.clear()
        self.widget.scan = self._make_scan(new_scan)
        self._configure_scan(new_scan, npt=npt, x_range=x_range)
        self.widget.frame_ids[:] = []
        self.widget.display_generation += 1
        if not compatible:
            self.expect_reset(INCOMPATIBLE_GRID)
        return self._step(
            f"rescope(scan={new_scan}, compatible={compatible}, npt={npt}, "
            f"x_range={tuple(x_range)})")

    def reintegrate_finish(self, *, npt=None):
        """Same-scan reintegrate pass completing: the store resets and every
        indexed frame republishes recomputed (production Step-6 shape).  An
        ALLOWED reset cause — a regrid (npt change) resets; an identical
        regrid may keep the rows (dedupe)."""
        if npt is not None:
            self._grid[self.widget.scan.name]["npt"] = int(npt)
        self.store.begin_reintegrate()
        try:
            for label in list(self.widget.scan.frames.index):
                frame = self._make_frame(label)
                self.store.upsert(publication_from_live_frame(frame))
        finally:
            self.store.end_reintegrate()
        self.widget.display_generation += 1
        self.expect_reset(REINTEGRATE)
        return self._step(f"reintegrate_finish(npt={npt})")

    def hydration_complete(self, label, *, stale=False):
        """An async hydration lands: the store re-gains the full publication.
        ``stale=True`` mirrors a completion whose generation lapsed — it joins
        the pending-append queue (OV-3/BR-2 path) instead of the selection."""
        frame = self._make_frame(label)
        self.store.upsert(publication_from_live_frame(frame))
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

    def clear(self):
        """The Clear button — the REAL ``clear_overlay`` (history + pins +
        pending queue), the canonical allowed reset."""
        self.widget.clear_overlay()
        self.expect_reset(CLEAR)
        return self._step("clear()")
