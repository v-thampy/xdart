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
    "ImagePayload",
    "ResultsView",
    "DisplayState",
    "DisplayPayload",
    "RenderPlan",
    "build_payload",
    "empty_display_state",
    "render_plan",
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
    NEXUS_VIEWER = "nexus_viewer"


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
    values: "np.ndarray | None" = None


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
    axis_y: "Axis | None" = None


@dataclass(frozen=True)
class ImagePayload:
    """Resolved content of a 2D image panel."""
    image: "np.ndarray"
    axis_x: Axis = Axis("x", "")
    axis_y: Axis = Axis("y", "")


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
    raw_image: "np.ndarray | ImagePayload | None"
    cake_image: "np.ndarray | ImagePayload | None"
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
# The full testable core is implemented below and called from the widget
# (selection, raw-source, sentinel, axes, overlay, GI uniformity).


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


def two_d_kind_from_units(unit, azimuthal_unit):
    """Classify a 2D integration result's axis identity from its unit strings.

    The qip/qoop (and exit-angle) GI axis identity is persisted in the
    NeXus file only via the ``q``/``chi`` dataset ``units`` attrs (e.g.
    ``qip_A^-1`` / ``qoop_A^-1``).  When a saved scan is reloaded and the
    GUI's ``scan.gi`` flag wasn't restored, the display would otherwise
    treat a qip/qoop map as a standard q/χ cake — and, worse, run the qip
    axis through the q→2θ conversion (arcsin of out-of-range values →
    collapsed/blank cake).  Reconstructing the kind from the units lets
    qip/qoop round-trip through the display.  This is the minimal version
    of the ``TwoDKind`` seam from the data-source unification plan.

    Returns one of ``'qip_qoop'``, ``'exit_angles'``, or ``'standard'``
    (q/χ, the back-compatible default — GI polar q/χ is indistinguishable
    from a standard cake by units alone and is treated as standard here).
    """
    u = str(unit or "").lower()
    au = str(azimuthal_unit or "").lower()
    if "qip" in u or "qip" in au or "qoop" in u or "qoop" in au:
        return "qip_qoop"
    if "exit" in u or "exit" in au:
        return "exit_angles"
    return "standard"


def is_gi_2d_units(unit, azimuthal_unit):
    """True if the 2D result's units mark it as a GI reciprocal-space map
    whose axes must be displayed verbatim (no Q↔2θ conversion).  See
    :func:`two_d_kind_from_units`."""
    return two_d_kind_from_units(unit, azimuthal_unit) != "standard"


def xye_unit_from_filename(name):
    """``'iq'``/``'iq_'`` → ``'q_A^-1'``; ``'itth'``/``'itth_'`` →
    ``'2th_deg'``; otherwise ``'unknown'`` (no assumption — an unknown
    prefix is NOT silently treated as 2θ; it labels the axis plain ``x``
    via :func:`x_axis_for_unit`)."""
    base = str(name).replace('\\', '/').rsplit('/', 1)[-1].lower()
    if base.startswith('itth'):
        return '2th_deg'
    if base.startswith('iq'):
        return 'q_A^-1'
    return 'unknown'


def default_plot_unit(bai_1d_unit, available_units):
    """Index of the plot-unit entry matching the integration unit so the
    1D plot opens on the integrated axis (fixes 'integrate in 2θ but plot
    defaults to Q').  ``available_units`` is the canonical-unit list in
    combo order; unknown unit falls back to index 0."""
    try:
        return list(available_units).index(bai_1d_unit)
    except (ValueError, TypeError):
        return 0


def plan_overlay(method, unit_changed, has_existing, new_ids, prev_overlaid_ids):
    """Decide how Overlay/Waterfall accumulates, including the unit-switch
    rebuild.  Returns ``(OverlayAction, ids)``:

    * Single/Sum/Average → REPLACE with the current selection.
    * Overlay/Waterfall + unit changed + existing curves → REBUILD: re-express
      the SAME accumulated frames in the new unit (never drop to the last one).
    * Overlay/Waterfall + existing curves (same unit) → APPEND the new frames.
    * Overlay/Waterfall with nothing yet → REPLACE (fresh start).
    """
    new_ids = tuple(new_ids)
    prev = tuple(prev_overlaid_ids)
    if method not in ('Overlay', 'Waterfall'):
        return OverlayAction.REPLACE, new_ids
    if unit_changed and has_existing:
        return OverlayAction.REBUILD, prev
    if has_existing:
        merged = prev + tuple(i for i in new_ids if i not in prev)
        return OverlayAction.APPEND, merged
    return OverlayAction.REPLACE, new_ids


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
    precondition).  Decides whether a GI scan needs a frozen common grid.

    ``axes_per_frame`` is a sequence of per-frame axis tuples (e.g.
    ``[(q, chi), (q, chi), ...]``); a frame mismatching the first in length,
    shape or values (within ``rtol``/``atol``) makes the stack non-uniform.
    This is the contract the GI common-grid freeze must satisfy — it never
    relaxes the writer's uniform-axis validators, it asserts the result."""
    if len(axes_per_frame) <= 1:
        return True
    first = axes_per_frame[0]
    for axes in axes_per_frame[1:]:
        if len(axes) != len(first):
            return False
        for a, b in zip(axes, first):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
                return False
    return True


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
    if mode is Mode.NEXUS_VIEWER:
        primary = render_2d if render_2d else render_1d
    else:
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

    if mode in (Mode.NEXUS_VIEWER,):
        raw_panel = PanelPlan(
            visible=True,
            has_data=ready and bool(render_2d),
            source=RawSource.RAW if render_2d else RawSource.NONE,
            apply_mask=False,
        )
        plot_panel = PanelPlan(visible=True, has_data=ready and bool(render_1d))
        panels = ((raw_key, raw_panel), (plot_key, plot_panel))
        layout = ((raw_key,), (plot_key,))
    elif mode in (Mode.IMAGE_VIEWER,):
        panels = ((raw_key, raw_panel),)
        layout = ((raw_key,),)
    elif mode in (Mode.XYE_VIEWER, Mode.INT_1D):
        # INT_1D is 1D-only (skip_2d): collapse to a plot-only layout,
        # matching the widget's _apply_1d_only_visibility.  The XYE viewer
        # is likewise plot-only.
        panels = ((plot_key, plot_panel),)
        layout = ((plot_key,),)
    else:  # INT_2D: raw | cake on top, 1D plot below
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


# ── Stage 3: payload assembly + render plan (the testable render core) ─

# The panel roles the integration renderer manages.  render iterates these,
# drawing the ones the state wants and clearing the rest — so a role left
# over from a previous mode/selection is always blanked (kills mode-switch
# staleness), without render branching on mode.
_RENDER_ROLES = (PanelRole.PLOT_1D, PanelRole.RAW_2D, PanelRole.CAKE_2D)


def empty_display_state(mode, generation, *, title=""):
    """A panel-less :class:`DisplayState` with ``EMPTY`` status.

    :func:`render_plan` puts every managed panel in ``clear`` for this state,
    so :meth:`render_display` blanks the plot, raw and cake panels.  Used to
    render an *explicit* blank on an empty selection / failed load / cache
    miss instead of early-returning and leaving stale content on screen
    (the blank is intentional, §8)."""
    return DisplayState(
        mode=mode,
        load_status=LoadStatus.EMPTY,
        error_message=None,
        generation=generation,
        selected_ids=(),
        render_ids=(),
        overall=False,
        gi=False,
        x_unit="unknown",
        x_label="x",
        method="Single",
        overlay=OverlayAction.REPLACE,
        overlaid_ids=(),
        title=title,
        panels=(),
        layout=(),
        results=None,
    )


def build_payload(state, store=None):
    """Resolve the arrays/traces for ``state`` into a :class:`DisplayPayload`.

    Pure and Qt-free.  Stamped with ``state.generation`` so render can drop
    a payload that no longer matches the state (the §8 generation
    invariant).  Arrays are resolved from ``store`` ONLY for panels that are
    present, ``has_data`` and ``READY``; everything else is ``None`` (blank).

    ``store`` is the source adapter (``raw_image(state)`` /
    ``cake_image(state)`` / ``plot_payload(state)``).  When ``store`` is
    ``None`` the payload resolves nothing — the renderer then delegates the
    pixel push to its legacy draw methods.  This is the Stage 3 default; the
    real store (and direct payload rendering) arrives with the controllers
    in Stage 4–5.  Tests pass a fake store to exercise the gating here.
    """
    raw = cake = plot = None
    if store is not None and state.load_status is LoadStatus.READY:
        rp = state.panel(PanelRole.RAW_2D)
        if rp is not None and rp.has_data:
            raw = store.raw_image(state)
        cp = state.panel(PanelRole.CAKE_2D)
        if cp is not None and cp.has_data:
            cake = store.cake_image(state)
        pp = state.panel(PanelRole.PLOT_1D)
        if pp is not None and pp.has_data:
            plot = store.plot_payload(state)
    return DisplayPayload(generation=state.generation, raw_image=raw,
                          cake_image=cake, plot=plot)


@dataclass(frozen=True)
class RenderPlan:
    """The pure decision render executes: drop a stale payload, blank
    intentionally, and which panels to draw vs clear.  Same (state, payload)
    ⇒ same plan — this is what makes render testable without Qt."""
    drop: bool                       # generation mismatch ⇒ render nothing
    error_message: "str | None"      # surfaced when load_status is ERROR
    title: str
    draw: tuple                      # roles to draw (present, has_data, READY)
    clear: tuple                     # roles to blank (absent / no data / EMPTY / ERROR)


def render_plan(state, payload):
    """Decide what render should do for ``(state, payload)``.

    A payload whose generation no longer matches the state is dropped
    (``drop=True``).  In EMPTY/ERROR every managed panel is cleared (blank is
    intentional, §8).  Otherwise a panel is drawn iff it is present in the
    state with ``has_data``; the rest are cleared.
    """
    if payload is not None and payload.generation != state.generation:
        return RenderPlan(drop=True, error_message=None, title=state.title,
                          draw=(), clear=())
    ready = state.load_status is LoadStatus.READY
    draw, clear = [], []
    for role in _RENDER_ROLES:
        plan = state.panel(role)
        if ready and plan is not None and plan.has_data:
            draw.append(role)
        else:
            clear.append(role)
    return RenderPlan(drop=False, error_message=state.error_message,
                      title=state.title, draw=tuple(draw), clear=tuple(clear))
