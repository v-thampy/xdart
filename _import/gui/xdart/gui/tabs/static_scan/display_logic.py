# -*- coding: utf-8 -*-
"""Pure, Qt-free display-decision layer for the static-scan display.

This module is the single source of truth for *what should be on screen*.
It is deliberately free of Qt, pyqtgraph, h5py and pyFAI so its decision
logic can be unit-tested headlessly (``pytest -m display_logic``) — see
``tests/test_display_logic.py`` and the design doc
(``display_refactor_plan.md``).

Populated across the staged refactor:

* Stage 0 (this commit) — scaffold only.  The contract *surface* is
  declared here (the :class:`DisplayState`/:class:`DisplayPayload` data
  shapes from the plan, plus stubs for the pure functions).  No
  production code imports this module yet, so adding it changes no
  behaviour.  The pure functions raise :class:`NotImplementedError`; the
  tests that exercise them start **red** by design and go green as the
  later stages land.
* Stage 1 — fill in the pure selectors (``resolve_selection``,
  ``resolve_render_ids``, ``choose_raw_source``, ``sentinel_mask``,
  the axis-label tables) and call them from the widget.
* Stage 2+ — ``compute_display_state``, generation, overlay/GI logic.

Guardrail: this module must import **no** Qt, pyqtgraph, h5py or pyFAI.
``from __future__ import annotations`` keeps the numpy type hints as plain
strings (numpy itself is the only heavy import the purity guard allows).

§10 seam 6: this core stays module-agnostic (selection, overlay, axes,
sentinel, generation, the panel/trace shapes, the controller registry).
Future modules add their OWN pure-logic modules — ``stitch_logic.py``,
``fit_logic.py`` — that contribute ``DisplayState``/``PlotPayload``
fragments and carry their own headless tests; this core never imports
them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np  # allowed: the purity guard forbids only Qt/pyqtgraph/h5py/pyFAI/fabio

__all__ = [
    "Mode",
    "RawSource",
    "OverlayAction",
    "LoadStatus",
    "PanelRole",
    "PanelKey",
    "PanelPlan",
    "Axis",
    "Trace",
    "PlotPayload",
    "ResultsView",
    "DisplayState",
    "DisplayPayload",
    "register_controller",
    "controller_for",
    "resolve_selection",
    "resolve_render_ids",
    "choose_raw_source",
    "apply_mask_for",
    "x_axis_for_unit",
    "xye_unit_from_filename",
    "default_plot_unit",
    "plan_overlay",
    "sentinel_mask",
    "gi_axes_uniform",
    "compute_display_state",
]


# ── Enums ─────────────────────────────────────────────────────────────

class Mode(Enum):
    INT_1D = "int_1d"
    INT_2D = "int_2d"
    IMAGE_VIEWER = "image_viewer"
    XYE_VIEWER = "xye_viewer"


class PanelRole(Enum):
    """Identifies the *kind* of a render panel.  ``render`` lays panels out
    by ``DisplayState.layout`` and dispatches each panel to a widget by its
    role, so a module can add a new role/arrangement without editing core
    render/compute logic (§10 seam 1).

    Used by the integration view today; the rest are reserved so the
    stitching/fitting/RSM modules plug in later without reshaping the core."""
    RAW_2D = "raw_2d"            # full/thumbnail detector image
    CAKE_2D = "cake_2d"          # 2D integrated (cake) image
    PLOT_1D = "plot_1d"          # 1D pattern(s)
    RESIDUAL_1D = "residual_1d"  # reserved: fitting residual trace panel
    STITCH_2D = "stitch_2d"      # reserved: stitched 2D image
    SLICE_2D = "slice_2d"        # reserved: RSM reciprocal-space 2D slice (repeats)
    PROJ_1D = "proj_1d"          # reserved: RSM 1D projection (repeats)
    RESULTS = "results"          # reserved: tables/scalars (non-array)


class RawSource(Enum):
    RAW = "raw"              # full-res detector array; detector mask applies
    THUMBNAIL = "thumbnail"  # mask already baked in; do NOT re-apply flat mask
    NONE = "none"            # nothing available → clear the panel


class OverlayAction(Enum):
    REPLACE = "replace"  # Single/Sum/Average, or fresh start
    APPEND = "append"    # add new frames to existing overlay (same unit)
    REBUILD = "rebuild"  # unit changed: re-express the whole overlay in new unit


class LoadStatus(Enum):
    EMPTY = "empty"      # nothing selected/loaded — panels blank, intentionally
    LOADING = "loading"  # a load is in flight — show "loading"/blank, not stale
    READY = "ready"      # payload is valid for this generation — render it
    ERROR = "error"      # load failed — blank + error_message, never half-populated


# ── Data shapes (§4 + §10 of the plan) ────────────────────────────────

@dataclass(frozen=True)
class PanelKey:
    """Identity of one panel instance.  ``instance`` disambiguates a role
    that repeats within a layout — e.g. RSM's three SLICE_2D panels
    (instance ``"HK"`` / ``"HL"`` / ``"KL"``) and three PROJ_1D panels
    (``"H"`` / ``"K"`` / ``"L"``).  For a role that never repeats the
    instance is ``""``, so ``PanelKey(PanelRole.RAW_2D)`` is the whole
    identity.  Frozen ⇒ hashable, so it works as a dict/lookup key."""
    role: PanelRole
    instance: str = ""


@dataclass(frozen=True)
class PanelPlan:
    visible: bool
    has_data: bool                       # False ⇒ render() clears this panel
    source: RawSource = RawSource.NONE   # 2D-raw panel only
    apply_mask: bool = False             # 2D-raw panel only


@dataclass(frozen=True)
class Axis:
    """One plot/image axis.  Replaces the loose ``(label, unit)`` string
    pair everywhere (§10 seam 2)."""
    label: str
    unit: str = ""
    log: bool = False


@dataclass(frozen=True)
class Trace:
    """One named curve on a 1D plot.  Integration/overlay emits
    ``kind="data"`` today; fitting later layers ``fit`` / ``component`` /
    ``background`` / ``residual`` traces onto the same payload with zero
    change to ``render`` (§10 seam 2)."""
    label: str
    x: "np.ndarray"
    y: "np.ndarray"
    kind: str = "data"   # data | fit | component | background | residual


@dataclass(frozen=True)
class PlotPayload:
    """Resolved content of a 1D plot panel: an x-axis plus layered traces."""
    axis_x: Axis
    traces: tuple = ()   # tuple[Trace, ...]


@dataclass(frozen=True)
class ResultsView:
    """Stub for non-array results (fit parameters, CIs, tables) routed to a
    results widget via the ``DisplayState.results`` channel (§10 seam 5).

    Reserved only — nothing populates it in this refactor; every current
    mode leaves ``DisplayState.results`` as ``None``."""
    rows: tuple = ()     # tuple[tuple, ...] — table rows, when implemented


@dataclass(frozen=True)
class DisplayState:
    mode: Mode
    load_status: LoadStatus          # EMPTY/LOADING/READY/ERROR — blanks are intentional
    error_message: "str | None"      # populated only when load_status is ERROR
    generation: int                  # DataStore generation this state was computed against
    selected_ids: tuple              # frame labels the user selected (viewer ids in viewer modes)
    render_ids: tuple                # labels actually used (∩ loaded data)
    overall: bool                    # aggregate across the whole scan (scan modes only)
    gi: bool
    x_unit: str                      # 'q_A^-1' | '2th_deg' | 'chi_deg' | gi units | 'unknown'
    x_label: str
    method: str                      # Single/Overlay/Waterfall/Sum/Average
    overlay: OverlayAction
    overlaid_ids: tuple
    title: str
    # §10 seam 1: panels are a keyed collection + a layout descriptor, not
    # three named fields.  ``panels`` maps a PanelKey to its plan; ``layout``
    # is a tuple of rows, each a tuple of PanelKeys, describing the ARRANGEMENT
    # (Int-2D: raw|cake / plot; Stitch-2D: cake / plot; RSM: a 2×3 grid of
    # repeated SLICE_2D/PROJ_1D roles).  render lays out by ``layout`` and
    # dispatches each panel to a widget by role — it never branches on mode.
    panels: tuple = ()               # tuple[tuple[PanelKey, PanelPlan], ...]
    layout: tuple = ()               # tuple[tuple[PanelKey, ...], ...] — rows of keys
    # §10 seam 5: non-array results channel; None for every current mode.
    results: "ResultsView | None" = None

    def panel(self, key):
        """Return the :class:`PanelPlan` for ``key``, or ``None``.

        ``key`` may be a :class:`PanelKey` (exact match) or a bare
        :class:`PanelRole` (returns the first panel with that role — the
        ergonomic path for the non-repeating integration roles)."""
        for k, plan in self.panels:
            if k == key:
                return plan
            if isinstance(key, PanelRole) and getattr(k, 'role', k) is key:
                return plan
        return None


@dataclass(frozen=True)
class DisplayPayload:
    """Resolved arrays/traces for one :class:`DisplayState` (assembled from
    the DataStore).  Kept separate so :func:`compute_display_state` stays
    pure/cheap and array assembly is tested on its own.  ``None`` ⇒ that
    panel renders blank.

    §10 seam 4: payloads are **source-agnostic** — they carry no provenance
    field.  ``render``/``build_payload`` must not branch on whether the data
    came from integration, stitch or a reload; only the controller that
    produced it knew, and it is gone by the time we render."""
    generation: int                 # must match the DisplayState it pairs with
    raw_image: "np.ndarray | None"
    cake_image: "np.ndarray | None"
    plot: "PlotPayload | None"      # 1D traces (§10 seam 2)


# ── Controller registry (§10 seam 3) ──────────────────────────────────
#
# Open Mode -> controller map that modules register into, instead of a
# closed switch in the core.  The core registers the scan/image/xye
# controllers (Stage 5); stitch/fit modules register their own later, so
# adding Mode.STITCH_2D / Mode.FIT never touches the dispatch core.  Only
# the hook exists now — no controllers are implemented in this refactor.

_CONTROLLER_REGISTRY = {}   # dict[Mode, controller]


def register_controller(mode, ctrl):
    """Register the controller that owns ``mode``'s selection rules and
    loading lifecycle.  Idempotent overwrite by mode."""
    _CONTROLLER_REGISTRY[mode] = ctrl
    return ctrl


def controller_for(mode):
    """Return the controller registered for ``mode``, or ``None``."""
    return _CONTROLLER_REGISTRY.get(mode)


# ── Pure functions (§5 of the plan) ───────────────────────────────────
#
# Stage 1 selectors are implemented below and called from the widget.
# Functions owned by later stages still raise NotImplementedError so the
# behaviour-contract tests stay red until their stage lands.

_STAGE2 = "implemented in Stage 2 of the display refactor"
_STAGE4 = "implemented in Stage 4 of the display refactor"


def resolve_selection(frame_ids, all_frame_index):
    """Return ``(sorted_ids, overall)``.  ``overall`` is True iff every
    scan frame is selected and there is more than one.  Replaces the body
    of ``displayFrameWidget.get_idxs``.

    Ids are cast to ``int`` and sorted *numerically* (the old code sorted
    the raw labels before casting, which mis-ordered string ids like
    ``'10'`` < ``'2'``).  Raises ``ValueError`` if a label is not an int,
    matching the old ``np.asarray(..., dtype=int)`` contract so callers
    can bail out cleanly.
    """
    n_all = len(all_frame_index)
    overall = (len(frame_ids) == n_all) and (n_all > 1)
    base = all_frame_index if overall else frame_ids
    ids = tuple(sorted(int(i) for i in base))
    return ids, overall


def resolve_render_ids(selected_ids, overall, all_frame_index, loaded_keys):
    """Frames we can actually draw = ``(overall ? all : selected) ∩ loaded_keys``,
    sorted numerically."""
    base = all_frame_index if overall else selected_ids
    loaded = set(loaded_keys)
    return tuple(sorted(i for i in (int(x) for x in base) if i in loaded))


def choose_raw_source(has_raw, has_thumbnail, *, prefer_thumbnail, want_raw):
    """The map_raw vs thumbnail vs none decision currently inside
    ``get_frames_map_raw``.

    Priority: an explicit thumbnail preference wins when a thumbnail
    exists; otherwise full raw (when wanted and present); otherwise a
    thumbnail; otherwise nothing.
    """
    if prefer_thumbnail and has_thumbnail:
        return RawSource.THUMBNAIL
    if want_raw and has_raw:
        return RawSource.RAW
    if has_thumbnail:
        return RawSource.THUMBNAIL
    return RawSource.NONE


def apply_mask_for(source):
    """True only for :attr:`RawSource.RAW` — detector/flat masks are
    applied to full-resolution raw arrays only, never to thumbnails
    (their mask is already baked in) and never to an absent panel."""
    return source is RawSource.RAW


# Canonical-unit → (axis label, unit symbol) table.  The Unicode glyphs
# mirror ``display_constants`` (AA_inv / Th / Chi / Deg); they are inlined
# here rather than imported because ``display_constants`` pulls in pyFAI
# (via ``integrator``), which would break this module's purity guarantee.
_AA_INV = u'\u212B\u207B\u00B9'  # Angstrom^-1 (matches display_constants.AA_inv)
_TH = u'\u03B8'                     # theta (display_constants.Th)
_CHI = u'\u03C7'                    # chi (display_constants.Chi)
_DEG = u'\u00B0'                    # degree (display_constants.Deg)

_X_AXIS_TABLE = {
    'q_A^-1': ('Q', _AA_INV),
    '2th_deg': (f"2{_TH}", _DEG),
    'chi_deg': (_CHI, _DEG),
}


def x_axis_for_unit(unit):
    """``(label, unit_symbol)`` for a plot/integration unit.  One table,
    used by both normal mode and the XYE viewer.  ``'unknown'`` (and any
    unrecognised unit) → ``('x', '')`` — never an assumed 2θ."""
    return _X_AXIS_TABLE.get(unit, ('x', ''))


def xye_unit_from_filename(name):
    """``'iq'``/``'iq_'`` → ``'q_A^-1'``; ``'itth'``/``'itth_'`` →
    ``'2th_deg'``; otherwise ``'unknown'`` (no assumption)."""
    raise NotImplementedError(_STAGE4)


def default_plot_unit(bai_1d_unit, available_units):
    """Index of the plot-unit combo entry matching the integration unit
    (fixes 'integrate in 2θ but plot defaults to Q')."""
    raise NotImplementedError(_STAGE4)


def plan_overlay(method, unit_changed, has_existing, new_ids, prev_overlaid_ids):
    """Overlay/Waterfall accumulation incl. the unit-switch rebuild;
    Single/Sum/Average → REPLACE."""
    raise NotImplementedError(_STAGE4)


def sentinel_mask(arr):
    """Return a float copy of ``arr`` with detector sentinels masked to NaN.

    Masks non-finite values and the uint32 dead/hot-pixel ceiling
    (4294967295, e.g. from Eiger masters).  Some 16-bit readers preserve
    invalid pixels at the uint16 ceiling (65535); when enough pixels sit
    exactly there, that ceiling is treated as a display sentinel too, so
    autoscale uses the real image range instead of rendering nearly black.
    """
    a = np.asarray(arr, dtype=float)
    bad = ~np.isfinite(a) | (a >= 4294967295.0)
    if a.size and np.isfinite(a).any():
        finite = np.isfinite(a)
        sentinel16 = finite & (a == 65535.0)
        if sentinel16.any() and sentinel16.sum() / a.size > 1e-4:
            bad |= sentinel16
    if bad.any():
        a = a.copy()
        a[bad] = np.nan
    return a


def gi_axes_uniform(axes_per_frame, *, rtol=1e-5, atol=1e-8):
    """True iff every frame shares one axis set (the writer's stacking
    precondition).  Decides whether a GI scan needs a frozen common grid."""
    raise NotImplementedError(_STAGE4)


_SCAN_MODES = (Mode.INT_1D, Mode.INT_2D)
_PLOT_PRIMARY_MODES = (Mode.INT_1D, Mode.XYE_VIEWER)  # primary data is 1D


def _availability(raw_availability, fid):
    """Per-frame ``{'has_raw', 'has_thumbnail'}`` lookup, tolerant of int
    vs label keys and a missing entry."""
    if not isinstance(raw_availability, dict):
        return {}
    entry = raw_availability.get(fid)
    if entry is None:
        try:
            entry = raw_availability.get(int(fid))
        except (TypeError, ValueError):
            entry = None
    return entry or {}


def compute_display_state(*, mode, selected_ids, all_frame_index, loaded_1d_keys,
                          loaded_2d_keys, gi, plot_unit, method, unit_changed,
                          prev_overlaid_ids, raw_availability, titles,
                          generation=0, loading=False):
    """Compose the pure selectors into one immutable :class:`DisplayState`
    describing exactly what each panel should show.  THE function the GUI
    calls.  Pure: no Qt, no I/O, no mutation of inputs.

    ``raw_availability`` maps a frame id to ``{'has_raw': bool,
    'has_thumbnail': bool}``; a special ``'__error__'`` key (mapped to a
    message) marks a failed load.  ``titles`` maps ``mode.value`` to the
    title/filename for the current selection.  ``loading`` is True while a
    load is in flight (lets EMPTY and LOADING be distinguished).
    """
    loaded_1d = set(loaded_1d_keys)
    loaded_2d = set(loaded_2d_keys)

    # Effective selection: scan modes may aggregate the whole scan; viewer
    # modes never do (their ids are *viewer* ids, not scan frame ids, and
    # must not consult scan.frames — §8 invariant).
    if mode in _SCAN_MODES:
        try:
            ids, overall = resolve_selection(selected_ids, all_frame_index)
        except (TypeError, ValueError):
            ids, overall = (), False
    else:
        try:
            ids = tuple(sorted(int(i) for i in selected_ids))
        except (TypeError, ValueError):
            ids = ()
        overall = False

    render_1d = resolve_render_ids(ids, overall, all_frame_index, loaded_1d)
    render_2d = resolve_render_ids(ids, overall, all_frame_index, loaded_2d)
    primary = render_1d if mode in _PLOT_PRIMARY_MODES else render_2d

    x_label, _sym = x_axis_for_unit(plot_unit)

    # Failed load -> ERROR with a message; never a half-populated display
    # (§8 invariant).  Blank panels + blank title.
    err = raw_availability.get('__error__') if isinstance(raw_availability, dict) else None
    if err:
        load_status = LoadStatus.ERROR
        error_message = str(err)
        render_ids = ()
        title = ''
    else:
        error_message = None
        render_ids = primary
        if render_ids:
            load_status = LoadStatus.READY
        elif loading:
            load_status = LoadStatus.LOADING
        else:
            load_status = LoadStatus.EMPTY
        # Title is computed *in* the state from the same inputs, so it can
        # never drift from the payload it describes (§8 invariant).  Only a
        # READY state carries a title; EMPTY/LOADING/ERROR blank it.
        title = titles.get(mode.value, '') if load_status is LoadStatus.READY else ''

    ready = load_status is LoadStatus.READY

    # 2D-raw panel: the raw-vs-thumbnail-vs-none decision (mask only on full
    # raw — §8 invariant).  Overall aggregation prefers thumbnails, matching
    # update_image's prefer_thumbnail path.
    if ready and render_2d:
        avail = _availability(raw_availability, render_2d[0])
        prefer_thumb = overall and len(render_2d) > 1
        raw_src = choose_raw_source(
            bool(avail.get('has_raw')), bool(avail.get('has_thumbnail')),
            prefer_thumbnail=prefer_thumb, want_raw=True)
    else:
        raw_src = RawSource.NONE

    raw_panel = PanelPlan(
        visible=True, has_data=(raw_src is not RawSource.NONE),
        source=raw_src, apply_mask=apply_mask_for(raw_src))
    cake_panel = PanelPlan(visible=True, has_data=ready and bool(render_2d))
    plot_panel = PanelPlan(visible=True, has_data=ready and bool(render_1d))

    raw_key = PanelKey(PanelRole.RAW_2D)
    cake_key = PanelKey(PanelRole.CAKE_2D)
    plot_key = PanelKey(PanelRole.PLOT_1D)

    if mode is Mode.IMAGE_VIEWER:
        panels = ((raw_key, raw_panel),)
        layout = ((raw_key,),)
    elif mode is Mode.XYE_VIEWER:
        panels = ((plot_key, plot_panel),)
        layout = ((plot_key,),)
    else:  # INT_1D / INT_2D: raw | cake on top, 1D plot below
        panels = (
            (raw_key, raw_panel),
            (cake_key, cake_panel),
            (plot_key, plot_panel),
        )
        layout = ((raw_key, cake_key), (plot_key,))

    return DisplayState(
        mode=mode,
        load_status=load_status,
        error_message=error_message,
        generation=generation,
        selected_ids=tuple(ids),
        render_ids=tuple(render_ids),
        overall=overall,
        gi=bool(gi),
        x_unit=plot_unit,
        x_label=x_label,
        method=method,
        overlay=OverlayAction.REPLACE,   # plan_overlay wiring lands in Stage 4
        overlaid_ids=tuple(prev_overlaid_ids or ()),
        title=title,
        panels=panels,
        layout=layout,
        results=None,
    )
