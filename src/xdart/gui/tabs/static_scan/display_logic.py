# -*- coding: utf-8 -*-
"""Pure, Qt-free display-decision layer for the static-scan display.

This module is the single source of truth for *what should be on screen*.
It is deliberately free of Qt, pyqtgraph, h5py and pyFAI so its decision
logic can be unit-tested headlessly (``pytest -m display_logic``) — see
``tests/test_display_logic.py``.

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

import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np  # allowed: the purity guard forbids only Qt/pyqtgraph/h5py/pyFAI/fabio

# The Qt-free core contracts are likewise allowed (import-light by
# design): the purity guard asserts this pulls no Qt/pyqtgraph/h5py/
# pyFAI/fabio.
from xrd_tools.core.frame_view import (
    two_d_kind_from_units as _core_kind,
)
from xrd_tools.core.invalid import (
    UINT32_CEILING as _UINT32_CEILING,
    integer_saturation_ceiling as _core_saturation_ceiling,
    saturation_pixels as _saturation_pixels,
)

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
    "WaterfallHistory",
    "accumulate_waterfall",
    "waterfall_display_rows",
    "ImagePayload",
    "ResultsView",
    "DisplayState",
    "DisplayPayload",
    "RenderPlan",
    "build_payload",
    "empty_display_state",
    "render_plan",
    "render_roles_for_state",
    "register_controller",
    "controller_for",
    "resolve_selection",
    "resolve_render_ids",
    "choose_raw_source",
    "apply_mask_for",
    "x_axis_for_unit",
    "pretty_unit",
    "xye_unit_from_filename",
    "xye_prefix_for_unit",
    "default_plot_unit",
    "plan_overlay",
    "sentinel_mask",
    "standalone_viewer_image",
    "convert_2d_radial",
    "resample_image_axis_to_uniform",
    "resample_cake_to_unit",
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
    STITCH_1D = "stitch_1d"      # whole-scan merged 1D pattern (scan.stitched_1d)
    STITCH_2D = "stitch_2d"      # whole-scan merged 2D cake (scan.stitched_2d)


# ── Panel layout table (Stage 4/5 step 1) ────────────────────────────
#
# Panel geometry used to be set by *deltas* scattered across
# ``set_viewer_display_mode`` (per-mode height/width/visibility pokes) and
# ``_apply_1d_only_visibility`` (collapse/restore the 2D pane).  Each path set
# only the fields it cared about and assumed a baseline the other may have
# changed, so state leaked across mode transitions (e.g. a 1D-only mode left
# ``twoDWindow`` at ``maximumHeight(0)``; a viewer that draws the 2D pane then
# rendered its image into a zero-height widget → invisible).
#
# ``PANEL_LAYOUT`` makes geometry a pure, idempotent function of ``Mode``: the
# *full* end state of every managed widget, for every mode, with no reliance on
# prior state.  ``displayFrameWidget._apply_layout`` applies it unconditionally.
# This is the Qt-free data half; the Qt application lives in the widget.

_FULL = 16777215  # Qt's QWIDGETSIZE_MAX — "no maximum" sentinel for min/max.


@dataclass(frozen=True)
class PanelLayout:
    """Complete panel geometry for one :class:`Mode`.

    Every field is set unconditionally by ``_apply_layout`` — there are no
    "leave it alone" fields, which is the whole point (a field left untouched
    is exactly how geometry leaked across modes).  ``*_h`` / ``*_w`` are
    ``(minimum, maximum)`` pairs in pixels; ``_FULL`` means "no maximum".

    Widget roles (hierarchy: ``imageWindow`` is the top primary panel holding
    the title bar + 2D container + middle toolbar; ``plotWindow`` is the bottom
    primary panel; ``binnedFrame`` is the cake panel inside ``twoDWindow``):

    * ``frame_top``    — title bar (filename + process controls); always shown.
    * ``twoDWindow``   — the 2D image container (raw + cake).
    * ``imageWindow``  — top primary panel height (title+2D+toolbar).
    * ``plotWindow``   — bottom primary panel height (the 1D plot).
    * ``binnedFrame``  — cake panel *width* (collapsed to show raw only).
    * ``imageToolbar`` — middle control bar (40px tall by UI default).
    * ``frame_4``/``frame_6`` — process-mode controls (norm/bkg, scale/cmap);
      hidden in viewer modes.  ``_showImageBtn`` lives inside ``frame_6``.
    * ``plotToolBar``  — legacy bottom plot toolbar, emptied by
      ``_reflow_controls`` and reused as the Image Viewer top intensity row.
    * ``show_image_btn`` — the raw-preview button (only meaningful in 1D-only
      Int mode; its host ``frame_6`` is hidden in viewer modes regardless).
    """
    frame_top_vis: bool
    twoDWindow_vis: bool
    imageToolbar_vis: bool
    frame_4_vis: bool
    frame_6_vis: bool
    plotToolBar_vis: bool
    show_image_btn_vis: bool
    twoDWindow_h: tuple
    imageWindow_h: tuple
    plotWindow_h: tuple
    imageToolbar_h: tuple
    plotToolBar_h: tuple
    binnedFrame_w: tuple


# Values extracted faithfully from the pre-table end states (see
# ``set_viewer_display_mode`` / ``_apply_1d_only_visibility`` history).  Two
# deliberate fixes vs the old scattered code, both behaviour-improving and
# called out in the plan:
#   * INT_1D sets ``plotWindow_h`` explicitly (the old 1D-only path left it
#     implicit — a latent gap), and
#   * viewer modes now set ``show_image_btn_vis`` False explicitly (the button's
#     host ``frame_6`` is hidden there anyway, so this is invisible but removes a
#     latent leak from a prior 1D-only mode).
PANEL_LAYOUT = {
    # Int 1D / Int 1D (XYE): 2D pane collapsed (height 0) but still "visible";
    # imageWindow shrinks to the title + middle bar; raw-preview button shown.
    Mode.INT_1D: PanelLayout(
        frame_top_vis=True, twoDWindow_vis=True, imageToolbar_vis=True,
        frame_4_vis=True, frame_6_vis=True, plotToolBar_vis=False,
        show_image_btn_vis=True,
        twoDWindow_h=(0, 0), imageWindow_h=(80, 85), plotWindow_h=(200, _FULL),
        imageToolbar_h=(40, 40), plotToolBar_h=(0, 0), binnedFrame_w=(0, _FULL),
    ),
    # Int 2D: full 2D pane (raw + cake) over the 1D plot; all controls shown.
    Mode.INT_2D: PanelLayout(
        frame_top_vis=True, twoDWindow_vis=True, imageToolbar_vis=True,
        frame_4_vis=True, frame_6_vis=True, plotToolBar_vis=False,
        show_image_btn_vis=False,
        twoDWindow_h=(0, _FULL), imageWindow_h=(200, _FULL),
        plotWindow_h=(200, _FULL), imageToolbar_h=(40, 40),
        plotToolBar_h=(0, 0), binnedFrame_w=(0, _FULL),
    ),
    # Image Viewer: raw image only; 1D plot collapsed, cake collapsed,
    # process controls hidden.  frame_6 (scale + cmap) kept so the Linear/Log
    # scale and colormap apply to the raw image.
    Mode.IMAGE_VIEWER: PanelLayout(
        # frame_4 shown to host the Set BG button (Norm Channel is hidden inside
        # it at runtime, so only Set BG shows, left-justified where Norm is in Int).
        frame_top_vis=True, twoDWindow_vis=True, imageToolbar_vis=False,
        frame_4_vis=True, frame_6_vis=True, plotToolBar_vis=True,
        show_image_btn_vis=False,
        twoDWindow_h=(0, _FULL), imageWindow_h=(200, _FULL),
        plotWindow_h=(0, 0), imageToolbar_h=(40, 40),
        plotToolBar_h=(40, 40), binnedFrame_w=(0, 0),
    ),
    # XYE Viewer: 1D overlay only; 2D container hidden, middle bar kept
    # (Single/Options/Legend/Clear), process controls hidden.  frame_6 kept so
    # the Log toggle applies to the 1D plot and the colormap stays available
    # (the XYE waterfall image uses it).
    Mode.XYE_VIEWER: PanelLayout(
        # frame_4 shown to host the Set BG button (Norm Channel hidden at runtime).
        frame_top_vis=True, twoDWindow_vis=False, imageToolbar_vis=True,
        frame_4_vis=True, frame_6_vis=True, plotToolBar_vis=False,
        show_image_btn_vis=False,
        twoDWindow_h=(0, _FULL), imageWindow_h=(80, 85),
        plotWindow_h=(200, _FULL), imageToolbar_h=(40, 40),
        plotToolBar_h=(0, 0), binnedFrame_w=(0, _FULL),
    ),
    # NeXus Viewer: 2D dataset preview over a 1D dataset preview; cake
    # collapsed, process controls + middle bar hidden.
    Mode.NEXUS_VIEWER: PanelLayout(
        frame_top_vis=True, twoDWindow_vis=True, imageToolbar_vis=False,
        frame_4_vis=False, frame_6_vis=False, plotToolBar_vis=False,
        show_image_btn_vis=False,
        twoDWindow_h=(0, _FULL), imageWindow_h=(200, _FULL),
        plotWindow_h=(200, _FULL), imageToolbar_h=(40, 40),
        plotToolBar_h=(0, 0), binnedFrame_w=(0, 0),
    ),
    # Stitch 1D: the whole-scan merged 1D pattern — plot-only, identical
    # geometry to INT_1D (2D pane collapsed, raw-preview button hidden since
    # there is no per-frame raw for a merge).
    Mode.STITCH_1D: PanelLayout(
        frame_top_vis=True, twoDWindow_vis=True, imageToolbar_vis=True,
        frame_4_vis=True, frame_6_vis=True, plotToolBar_vis=False,
        show_image_btn_vis=False,
        twoDWindow_h=(0, 0), imageWindow_h=(80, 85), plotWindow_h=(200, _FULL),
        imageToolbar_h=(40, 40), plotToolBar_h=(0, 0), binnedFrame_w=(0, _FULL),
    ),
    # Stitch 2D: the whole-scan merged cake — cake-focused; the 1D plot is
    # collapsed and the cake fills the 2D pane (the raw panel carries no
    # per-frame image for a merge, so render_plan blanks it).
    Mode.STITCH_2D: PanelLayout(
        frame_top_vis=True, twoDWindow_vis=True, imageToolbar_vis=True,
        frame_4_vis=True, frame_6_vis=True, plotToolBar_vis=False,
        show_image_btn_vis=False,
        twoDWindow_h=(0, _FULL), imageWindow_h=(200, _FULL),
        plotWindow_h=(0, 0), imageToolbar_h=(40, 40),
        plotToolBar_h=(0, 0), binnedFrame_w=(0, _FULL),
    ),
}


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
    """Resolved content of a 1D plot panel: an x-axis plus layered traces.

    ``overlaid_ids`` / ``plot_history`` are the (optional) accumulation state for
    the Overlay/Waterfall 1D modes, carried so the *payload* owns the accumulator
    instead of the widget once those modes are routed through the payload path
    (the renderer flip is live-gated — see the publish-direct plan).  They stay
    ``None`` for the stateless Single/Sum/Average payloads, preserving the
    frozen-dataclass invariant (keep any value immutable: a tuple of ids, a
    ``MappingProxyType`` history).
    """
    axis_x: Axis
    traces: tuple = ()   # tuple[Trace, ...]
    axis_y: "Axis | None" = None
    overlaid_ids: "tuple | None" = None      # frame ids accumulated (Overlay/Waterfall)
    plot_history: "object | None" = None     # immutable accumulator view


@dataclass(frozen=True)
class WaterfallHistory:
    """Immutable Overlay/Waterfall accumulator carried IN the payload
    (``PlotPayload.plot_history``) -- the payload-owned successor to the legacy
    mutable widget triple (``plot_data`` / ``frame_names`` / ``overlaid_idxs``).

    Carrying it in the payload (rather than rebuilding from the publication store
    each render) is the keystone of the flip: the store evicts heavy frames past a
    cap, so a per-render rebuild from the store would re-introduce the cap-truncation
    regression.  The accumulator instead retains every row it has captured.

    ``reset_key`` is the accumulation IDENTITY (scan + 1D/2D source) -- the ONLY
    reset trigger.  It is deliberately NOT the display generation: the display
    generation bumps on every effective-selection change, and live auto-last GROWS
    the selection every tick, so keying the reset on it would reset the accumulator
    each tick and rebuild from only the resident (un-evicted) frames -- the exact
    cap-truncation this design exists to prevent.  Scan/source changes change the
    key (reset); selection growth and a Q<->2theta unit toggle do not (append /
    relabel in place).

    All rows live on ONE shared sample grid (``x``); the adapter interpolates each
    incoming frame onto it before accumulating.  ``rows[k]`` is frame ``ids[k]`` /
    ``names[k]`` -- maintained row-for-row.  ``unit`` is the radial unit the grid is
    currently labelled in (a Q<->2theta toggle relabels ``x`` in place, since the
    intensities are unit-invariant)."""
    reset_key: "object"      # accumulation identity (scan, source); reset on change
    unit: str
    x: "np.ndarray"          # shared radial sample grid (1-D, in `unit`)
    rows: "np.ndarray"       # (n, len(x)) stacked intensities; row order == ids
    ids: tuple               # captured frame ids, in row order
    names: tuple             # captured frame names, in row order
    label: str = ""          # x-axis label for `unit` (carried, not re-derived:
    #                          the display unit doesn't always round-trip through
    #                          x_axis_for_unit, e.g. the 2θ conversion's symbol)

    @property
    def count(self) -> int:
        return len(self.ids)


def _dedup_first(ids, names, rows):
    """Keep the FIRST occurrence of each id (arrival order)."""
    seen = set()
    ki, kn, kr = [], [], []
    for i, n, r in zip(ids, names, rows):
        ii = int(i)
        if ii in seen:
            continue
        seen.add(ii)
        ki.append(ii)
        kn.append(n)
        kr.append(r)
    return ki, kn, kr


def accumulate_waterfall(history, *, reset_key, unit, x, rows, ids, names, label=""):
    """Pure, append-only Overlay/Waterfall accumulator keyed on ``reset_key`` (the
    payload-owned successor to ``update_plot_accumulator`` + the widget triple).

    Append-only WITHIN one ``reset_key``: a partial / out-of-order / re-delivered
    read can only ADD frames not yet captured -- it never shrinks the stack or
    re-stacks a frame (the collapse/restack class, structurally precluded).  A
    ``reset_key`` change (a new scan, or a 1D<->2D-slice source change -- NOT a mere
    selection change) is the ONLY reset.

    Crucially the key is NOT the display generation: that bumps on every
    effective-selection change, and live auto-last grows the selection each tick, so
    keying on it would reset every tick and rebuild from only the un-evicted frames
    -- the cap-truncation this accumulator exists to prevent.  Selection growth
    therefore APPENDS (same key), retaining rows for frames since evicted past the
    store cap.

    A plotUnit Q<->2theta toggle does NOT change ``reset_key``, so ``unit`` changes
    while ``reset_key`` does not: the accumulated rows are unit-invariant, so we
    RELABEL the grid to the incoming (new-unit) ``x`` in place -- no re-read, no loss
    of evicted frames -- keeping every captured row.

    ``x`` / ``rows`` / ``ids`` / ``names`` are the incoming frames, already on the
    one shared grid (the adapter interpolated them).  Returns the next
    :class:`WaterfallHistory`.
    """
    x = np.asarray(x, dtype=float).ravel()
    rows = np.atleast_2d(np.asarray(rows, dtype=float))
    ids = [int(i) for i in ids]
    names = list(names)

    # RESET: new accumulation identity (scan/source change), no prior history, or an
    # empty incoming grid (nothing to anchor on -- keep the incoming as-is).
    if history is None or history.reset_key != reset_key or x.size == 0:
        ki, kn, kr = _dedup_first(ids, names, rows)
        return WaterfallHistory(
            reset_key=reset_key, unit=unit, label=label, x=x,
            rows=(np.asarray(kr, dtype=float) if kr
                  else np.empty((0, x.size), dtype=float)),
            ids=tuple(ki), names=tuple(kn))

    # Same identity: keep the accumulated rows; relabel the grid on a unit toggle
    # (incoming x is the same sample grid in the new unit), then append new ids.
    base_x = (x if history.unit != unit and x.size == history.x.size
              else history.x)
    base_rows = history.rows
    out_ids = list(history.ids)
    out_names = list(history.names)
    have = set(history.ids)

    add = []
    for i, n, r in zip(ids, names, rows):
        if i in have:
            continue
        have.add(i)
        out_ids.append(i)
        out_names.append(n)
        # The incoming row is on the shared grid already; reinterp defensively only
        # if the grid sample-count differs (e.g. a mid-generation Npts hiccup).
        r = np.asarray(r, dtype=float)
        if r.size != base_x.size and x.size == r.size and x.size > 0:
            r = np.interp(base_x, x, r)
        add.append(r)

    if add:
        add = np.atleast_2d(np.asarray(add, dtype=float))
        new_rows = (np.vstack([base_rows, add]) if base_rows.size else add)
    else:
        new_rows = base_rows

    return WaterfallHistory(
        reset_key=reset_key, unit=unit, label=label, x=base_x, rows=new_rows,
        ids=tuple(out_ids), names=tuple(out_names))


def waterfall_display_rows(rows, ids, max_rows):
    """Display-only row decimation for a large Waterfall image.

    The accumulator keeps every row.  This helper bounds only the rows painted in
    one render and applies the same stride to the row ids so axis labels remain
    aligned with the displayed image.
    """
    rows = np.asarray(rows)
    ids = tuple(ids)
    if max_rows is None:
        return rows, ids, 1
    max_rows = int(max_rows)
    n_rows = int(rows.shape[0]) if rows.ndim else 0
    if len(ids) != n_rows:
        ids = tuple(range(n_rows))
    if max_rows <= 0 or n_rows <= max_rows:
        return rows, ids, 1
    stride = int(np.ceil(n_rows / max_rows))
    return rows[::stride], ids[::stride], stride


@dataclass(frozen=True)
class ImagePayload:
    """Resolved content of a 2D image panel.

    ``gap_mask_indices`` / ``raw_full_shape`` are informational metadata for the
    raw detector panel: the flat detector-gap indices (into the full-resolution
    ``raw_full_shape``) that the builder masked to NaN.  Detector module gaps are
    0-valued pixels — NOT sentinels — so ``sentinel_mask`` never masks them; they
    are masked via the detector mask, and the builder bakes them into ``image``
    for both the full-res and the thumbnail source (the latter via
    :func:`nan_gaps_in_thumbnail`).  The fields let a consumer know where the
    gaps are without re-deriving them; they stay ``None`` for cake/viewer images.
    """
    image: "np.ndarray"
    axis_x: Axis = Axis("x", "")
    axis_y: Axis = Axis("y", "")
    gap_mask_indices: "np.ndarray | None" = None
    raw_full_shape: "tuple | None" = None


def combine_flat_masks(*masks, size=None):
    """Union detector-mask specs into one sorted flat-index array.

    Each spec may be ``None``, a 2-D boolean image-mask (-> ``flatnonzero``), or
    a 1-D flat-index array.  Returns ``None`` when nothing masks.  When ``size``
    is given, indices are bounded to ``[0, size)``.  Pure/Qt-free — shared by the
    legacy raw-render path and the publication raw payload so both derive the
    detector gap mask identically.
    """
    parts = []
    for m in masks:
        if m is None:
            continue
        arr = np.asarray(m)
        if arr.size == 0:
            continue
        arr = (np.flatnonzero(arr) if arr.ndim >= 2
               else np.asarray(arr, dtype=np.int64).ravel())
        if arr.size:
            parts.append(arr)
    if not parts:
        return None
    flat = np.unique(np.concatenate(parts))
    if size is not None:
        flat = flat[(flat >= 0) & (flat < size)]
    return flat if flat.size else None


def nan_gaps_in_thumbnail(data, gap_indices, full_shape):
    """NaN the detector-gap pixels in a downsampled thumbnail, in place.

    ``gap_indices`` are flat indices into the full-resolution ``full_shape``
    detector; map each to its thumbnail pixel via the per-axis downsample ratio
    and set it to NaN.  No-op (returns ``data`` unchanged) when the shape or
    indices are unusable — it never applies full-res flat indices directly to a
    smaller thumbnail (that would corrupt unrelated pixels).  Pure/Qt-free, so
    both the legacy ``update_image`` thumbnail path and the publication
    ``raw_image`` builder mask gaps identically.
    """
    if data is None or getattr(data, "ndim", 0) != 2 or full_shape is None:
        return data
    if gap_indices is None or np.size(gap_indices) == 0:
        return data
    H, W = int(full_shape[0]), int(full_shape[1])
    if H <= 0 or W <= 0:
        return data
    flat = np.asarray(gap_indices, dtype=np.int64).ravel()
    flat = flat[(flat >= 0) & (flat < H * W)]
    if flat.size == 0:
        return data
    h, w = data.shape
    rows, cols = np.unravel_index(flat, (H, W))
    tr = np.clip((rows * h) // H, 0, h - 1)
    tc = np.clip((cols * w) // W, 0, w - 1)
    data[tr, tc] = np.nan
    return data


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
    # HTML <sub> renders as a real subscript in pyqtgraph's setLabel (the only
    # consumer of these labels — state.x_label is unused, combos use gi_plotUnits).
    'qip_A^-1': ('Q<sub>ip</sub>', _AA_INV),
    'qoop_A^-1': ('Q<sub>oop</sub>', _AA_INV),
    # GI polar (q_chi) radial axis: the total scattering-vector magnitude.  Without
    # these entries the cake x-axis fell back to the raw ssrl label "Q_total" (a
    # LITERAL underscore); the <sub> renders the subscript like the χ_GI y-axis.
    'qtot_A^-1': ('Q<sub>total</sub>', _AA_INV),
    'qtot_nm^-1': ('Q<sub>total</sub>', 'nm⁻¹'),
    'exit_angle_deg': ('Exit Angle', _DEG),
    'exit_angle': ('Exit Angle', _DEG),
    'chigi_deg': (f'{_CHI}<sub>GI</sub>', _DEG),
    'r_mm': ('r', 'mm'),
}


def x_axis_for_unit(unit):
    """``(label, unit_symbol)`` for a plot/integration unit.  One table,
    used by both normal mode and the XYE viewer.  ``'unknown'`` (and any
    unrecognised unit) → ``('x', '')`` — never an assumed 2θ."""
    return _X_AXIS_TABLE.get(unit, ('x', ''))


def nanmean_slice(arr, axis):
    """``np.nanmean`` over ``axis`` for a 2D->1D slice projection that (a) returns
    ``None`` when the reduce axis is empty (no bins selected) and (b) never emits
    the "Mean of empty slice" RuntimeWarning on an all-NaN column (GI empty/padded
    bins legitimately reduce to a NaN gap — kept, not plotted).  Used by both the
    publication slice path and the legacy get_int_1d 2D path so neither warns."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim <= axis or arr.shape[axis] == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(arr, axis=axis)


def pretty_unit(unit):
    """Display symbol for a unit string (``'q_A^-1'`` -> ``Å⁻¹``, ``'2th_deg'``
    -> ``°``, ...).  Unrecognised/empty units pass through unchanged.  This is
    display-only -- the stored/headless unit stays the canonical pyFAI token
    (N3-units, done at the display layer so the NeXus ``units`` attr and the
    FrameView equivalence contract are untouched)."""
    if not unit:
        return unit
    symbol = _X_AXIS_TABLE.get(unit, (None, None))[1]
    return symbol if symbol else unit


#: TwoDKind -> the display layer's legacy kind strings.  GI polar
#: (QTOT_CHIGI) renders like a standard cake (q-like radial vs chi-like
#: azimuthal axis), so it maps to 'standard' — the pre-enum behavior.
_TWO_D_KIND_TO_LEGACY = {
    "q_chi": "standard",
    "qtot_chigi": "standard",
    "qip_qoop": "qip_qoop",
    "exit_angles": "exit_angles",
}


def two_d_kind_from_units(unit, azimuthal_unit):
    """Classify a 2D integration result's axis identity from its unit strings.

    The qip/qoop (and exit-angle) GI axis identity is persisted in the
    NeXus file only via the ``q``/``chi`` dataset ``units`` attrs (e.g.
    ``qip_A^-1`` / ``qoop_A^-1``).  When a saved scan is reloaded and the
    GUI's ``scan.gi`` flag wasn't restored, the display would otherwise
    treat a qip/qoop map as a standard q/χ cake — and, worse, run the qip
    axis through the q→2θ conversion (arcsin of out-of-range values →
    collapsed/blank cake).  Reconstructing the kind from the units lets
    qip/qoop round-trip through the display.

    The classification itself is the core ``TwoDKind`` seam
    (:func:`xrd_tools.core.frame_view.two_d_kind_from_units` — Qt-free,
    allowed by the purity guard); this wrapper maps the enum to the
    display layer's legacy strings ``'qip_qoop'`` / ``'exit_angles'`` /
    ``'standard'`` (q/χ — the back-compatible default, which also covers
    GI polar q/χ).
    """
    return _TWO_D_KIND_TO_LEGACY[_core_kind(str(unit or ""),
                                            str(azimuthal_unit or "")).value]


def is_gi_2d_units(unit, azimuthal_unit):
    """True if the 2D result's units mark it as a GI reciprocal-space map
    whose axes must be displayed verbatim (no Q↔2θ conversion).  See
    :func:`two_d_kind_from_units`."""
    return two_d_kind_from_units(unit, azimuthal_unit) != "standard"


def xye_prefix_for_unit(unit):
    """Filename prefix encoding the 1D integration axis, so the XYE reader can
    recover the x-axis from the name (inverse of :func:`xye_unit_from_filename`):

    * Q → ``iq``; 2θ → ``itth``;
    * GI Q_ip → ``iqip``; Q_oop → ``iqoop``; exit-angle → ``iexit``;
    * anything else → ``iq`` (Q default).

    Matched on a normalised (underscore-stripped, lowercased) unit so it's robust
    to ``q_ip`` vs ``qip_A^-1`` etc.  This replaces the old ``'iq' if q else
    'itth'`` rule, which mislabeled every non-Q axis (Q_ip/Q_oop/exit) as 2θ."""
    u = str(unit).lower().replace('_', '')
    if 'qip' in u:
        return 'iqip'
    if 'qoop' in u:
        return 'iqoop'
    if 'exit' in u:
        return 'iexit'
    if '2th' in u or 'tth' in u:
        return 'itth'
    return 'iq'


def xye_unit_from_filename(name):
    """Recover the x-axis unit from an XYE filename prefix (inverse of
    :func:`xye_prefix_for_unit`): ``iqip``→``qip_A^-1``, ``iqoop``→``qoop_A^-1``,
    ``iexit``→``exit_angle_deg``, ``itth``→``2th_deg``, ``iq``→``q_A^-1``.

    Anything else falls back to ``q_A^-1`` (Q): XRD 1D patterns are Q by
    convention, so a non-prefixed file is assumed Q rather than left unlabelled.
    NB the GI prefixes are checked before the generic ``iq`` (``iqip`` etc. also
    start with ``iq``)."""
    base = str(name).replace('\\', '/').rsplit('/', 1)[-1].lower()
    if base.startswith('iqip'):
        return 'qip_A^-1'
    if base.startswith('iqoop'):
        return 'qoop_A^-1'
    if base.startswith('iexit'):
        return 'exit_angle_deg'
    if base.startswith('itth'):
        return '2th_deg'
    if base.startswith('iq'):
        return 'q_A^-1'
    return 'q_A^-1'


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


def overlay_read_failure_action(method, has_accumulator):
    """Decide what an Overlay/Waterfall render does when its INCREMENTAL read of
    the not-yet-accumulated frames comes back empty.

    Append-only invariant: a failed/partial read must NEVER shrink an existing
    accumulator.  When Overlay/Waterfall already has accumulated frames, the
    missing ones are simply in flight (being written, or evicted past the store
    cap and awaiting async hydration) and arrive on a later tick -- so PRESERVE
    the accumulator and redraw what we have.  CLEAR only when nothing is
    accumulated yet (a genuine empty selection) or for the non-accumulating
    methods (Single/Sum/Average), which rebuild from the current selection.

    This is the fix for the cap-store Overlay/Waterfall regression: a slow GUI
    (e.g. toggling Share Axis) let the reduction race ahead, the non-blocking read
    of the newest 'missing' frames returned nothing, and the panel was cleared --
    collapsing the whole stack to ~0 and then repopulating + re-stacking frames as
    it caught up.

    Returns ``'preserve'`` or ``'clear'``.
    """
    if method in ('Overlay', 'Waterfall') and has_accumulator:
        return 'preserve'
    return 'clear'


def integer_saturation_ceiling(arr):
    """GUI wrapper over :func:`xrd_tools.core.invalid.integer_saturation_ceiling`:
    the dtype-derived ceiling (``np.iinfo(dtype).max`` — 65535 for uint16, 255
    for uint8, learned from the detector bit depth, never assumed 16-bit),
    falling back to ``65535.0`` when ``arr`` is already float (the integer dtype
    was lost upstream).  The 65535 fallback is the legacy GUI policy and stays
    in xdart — core returns ``None`` there and never hardcodes 65535.  The
    fallback is SAFE: ``== 65535`` won't match a non-16-bit frame's values.
    Capture the ceiling from the RAW frame (before any float conversion) to get
    the exact value for 8/32-bit detectors.
    """
    ceiling = _core_saturation_ceiling(arr)
    return ceiling if ceiling is not None else 65535.0


def sentinel_mask(arr, mask_saturation=True, ceiling=None):
    """Return a float copy of ``arr`` with detector sentinels masked to NaN.

    Always masks the UNAMBIGUOUS invalids: non-finite values and the uint32
    dead/hot-pixel ceiling (4294967295, e.g. from Eiger masters) — neither can
    be a real photon count.

    The detector SATURATION ceiling (uint16 65535, or whatever ``iinfo.max``
    the raw dtype implies — see :func:`integer_saturation_ceiling`) is
    AMBIGUOUS: it is both the max real count and a common overflow/invalid
    sentinel — so masking it is OPT-IN via ``mask_saturation`` (the "Mask
    Saturated" wrangler toggle, default ON).  ``ceiling`` overrides the
    dtype-derived value (callers that already converted to float pass the
    ceiling captured from the raw dtype).  The fraction-guarded saturation
    policy itself lives in :func:`xrd_tools.core.invalid.saturation_pixels`
    (R3-C — shared with the integration path); here it only feeds the NaN fill
    so the raw-display autoscale uses the real image range.  When disabled a
    real saturated Bragg peak is left intact (the raw display relies on a robust
    percentile level-clamp to avoid blowing out — it never hides it).
    """
    orig = np.asarray(arr)
    a = orig.astype(float)
    bad = ~np.isfinite(a) | (a >= _UINT32_CEILING)
    if mask_saturation and a.size and np.isfinite(a).any():
        if ceiling is None:
            ceiling = integer_saturation_ceiling(orig)
        bad |= _saturation_pixels(a, ceiling=ceiling)
    if bad.any():
        a = a.copy()
        a[bad] = np.nan
    return a


def standalone_viewer_image(data):
    """Display-only cleanup for standalone Image Viewer files.

    Standalone detector-file viewing is inspection, not processing: keep normal
    high values and do not turn uint16 ceilings into NaN masks. Only true
    non-finite values and the Eiger uint32 sentinel are filled with the low
    finite value so autoscale remains usable without painting white mask holes.
    """
    arr = np.asarray(data, dtype=float)
    bad = ~np.isfinite(arr) | (arr >= 4294967295.0)
    if not bad.any():
        return arr
    valid = np.isfinite(arr) & ~bad
    out = arr.astype(float, copy=True)
    if not valid.any():
        out[...] = np.nan
        return out
    out[bad] = float(np.nanmin(arr[valid]))
    return out


def convert_2d_radial(radial, *, data_unit, want_tth, want_q, wavelength_m):
    """Convert a cake *radial* axis between Q (Å⁻¹) and 2θ (deg) on the fly,
    mirroring ``display_data.get_xydata`` so the payload cake and the legacy
    cake agree exactly under the 2D-unit (imageUnit) toggle.

    ``want_tth`` / ``want_q`` come from the selected imageUnit label (does it
    name 2θ / Q?); ``data_unit`` is the integration unit of the axis. The
    conversion fires only when the wanted unit differs from the data's, and is
    a no-op when the wavelength is unknown.  GI reciprocal-space axes must NOT
    be passed here (their imageUnit combo is disabled; axes are verbatim)."""
    radial = np.asarray(radial, dtype=float)
    have_tth = '2th' in str(data_unit or '')
    if not wavelength_m or wavelength_m <= 0:
        return radial
    lam_A = wavelength_m * 1e10
    if want_tth and not have_tth:
        arg = np.clip(radial * lam_A / (4 * np.pi), -1, 1)
        return 2 * np.degrees(np.arcsin(arg))
    if want_q and have_tth:
        return (4 * np.pi / lam_A) * np.sin(np.radians(radial / 2))
    return radial


_CAKE_RESAMPLE_MIN_COVERAGE = 1.0


def _interp_image_axis_at(image, source_axis, target_source_axis, *, axis=-1):
    """NaN-aware interpolation of an image-like array along one axis."""
    img = np.asarray(image, dtype=float)
    source_axis = np.asarray(source_axis, dtype=float)
    target_source_axis = np.asarray(target_source_axis, dtype=float)
    if img.ndim == 0 or source_axis.ndim != 1:
        return img

    axis = int(axis)
    if axis < 0:
        axis += img.ndim
    if (
        axis < 0
        or axis >= img.ndim
        or img.shape[axis] != source_axis.size
        or source_axis.size < 2
    ):
        return img

    order = np.argsort(source_axis)
    xp = source_axis[order]
    valid_xp = np.isfinite(xp)
    if valid_xp.sum() < 2:
        return np.full(
            img.shape[:axis] + (target_source_axis.size,) + img.shape[axis + 1:],
            np.nan,
            dtype=float,
        )
    xp = xp[valid_xp]
    ordered_valid = order[valid_xp]

    moved = np.moveaxis(img, axis, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    out = np.empty((flat.shape[0], target_source_axis.size), dtype=float)
    for i, row in enumerate(flat):
        row = row[ordered_valid]
        finite = np.isfinite(row)
        if not finite.any():
            out[i] = np.nan
            continue
        filled = np.where(finite, row, 0.0)
        num = np.interp(target_source_axis, xp, filled, left=np.nan, right=np.nan)
        cov = np.interp(
            target_source_axis,
            xp,
            finite.astype(float),
            left=0.0,
            right=0.0,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            values = num / np.where(cov > 0, cov, 1.0)
        out[i] = np.where(cov >= _CAKE_RESAMPLE_MIN_COVERAGE, values, np.nan)
    out = out.reshape(moved.shape[:-1] + (target_source_axis.size,))
    return np.moveaxis(out, -1, axis)


def resample_image_axis_to_uniform(image, source_axis, *, axis=-1):
    """Resample image-like data onto a grid uniform in ``source_axis`` units.

    This is display-only glue for pyqtgraph ``ImageItem`` renderers: the image
    is placed through one affine rectangle, so non-uniform x coordinates must be
    made uniform before the image is drawn.  NaN gaps are preserved through a
    strict coverage gate instead of being smeared by interpolation.
    """
    img = np.asarray(image, dtype=float)
    source_axis = np.asarray(source_axis, dtype=float)
    if img.ndim == 0 or source_axis.ndim != 1 or source_axis.size < 2:
        return img, source_axis
    finite = np.isfinite(source_axis)
    if finite.sum() < 2:
        return img, source_axis
    target_axis = np.linspace(
        float(np.nanmin(source_axis[finite])),
        float(np.nanmax(source_axis[finite])),
        source_axis.size,
    )
    if np.allclose(source_axis, target_axis, rtol=1e-12, atol=1e-12,
                   equal_nan=True):
        return img, source_axis
    return _interp_image_axis_at(
        img,
        source_axis,
        target_axis,
        axis=axis,
    ), target_axis


def resample_cake_to_unit(
        image, radial, *, data_unit, want_tth, want_q, wavelength_m,
        axis=-1):
    """Convert a cake radial axis and resample image columns/rows onto it.

    ``ImageItem`` renders images through one affine rectangle, so a non-linear
    display conversion such as uniform-Q -> 2θ cannot be represented by merely
    changing tick values.  Re-sample along the radial image axis to a uniform
    target grid in the requested display unit; otherwise peaks move relative to
    1D curves even though the axis labels look correct.

    Returns ``(image, radial)``.  If no unit conversion is requested or the
    wavelength is unknown, both are returned unchanged apart from float array
    coercion.
    """
    img = np.asarray(image, dtype=float)
    radial = np.asarray(radial, dtype=float)
    if img.ndim == 0 or radial.ndim != 1 or radial.size < 2:
        return img, radial
    data_unit = str(data_unit or "")
    have_tth = "2th" in data_unit
    converts = (want_tth and not have_tth) or (want_q and have_tth)
    if not converts or not wavelength_m or wavelength_m <= 0:
        return img, radial

    target_native = convert_2d_radial(
        radial,
        data_unit=data_unit,
        want_tth=want_tth,
        want_q=want_q,
        wavelength_m=wavelength_m,
    )
    target_native = np.asarray(target_native, dtype=float)
    if target_native.shape != radial.shape:
        return img, radial
    if np.allclose(target_native, radial, rtol=1e-12, atol=1e-12, equal_nan=True):
        return img, radial
    finite_axis = np.isfinite(radial) & np.isfinite(target_native)
    if finite_axis.sum() < 2:
        return img, target_native

    axis = int(axis)
    if axis < 0:
        axis += img.ndim
    if axis < 0 or axis >= img.ndim or img.shape[axis] != radial.size:
        return img, target_native

    target_u = np.linspace(
        float(np.nanmin(target_native[finite_axis])),
        float(np.nanmax(target_native[finite_axis])),
        radial.size,
    )
    target_data_unit = "2th_deg" if want_tth else "q_A^-1"
    source_at = convert_2d_radial(
        target_u,
        data_unit=target_data_unit,
        want_tth=have_tth,
        want_q=not have_tth,
        wavelength_m=wavelength_m,
    )
    source_at = np.asarray(source_at, dtype=float)
    if not np.isfinite(source_at).any():
        return img, target_u

    return _interp_image_axis_at(
        img,
        radial,
        source_at,
        axis=axis,
    ), target_u


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

    # Overlay/Waterfall preserve (Phase-5 cap-store regression fix): when the 1D
    # accumulator already holds frames but THIS selection's 1D read is empty (the
    # selected frame is evicted past the store cap, awaiting async hydration), keep
    # PLOT_1D drawable so the overlay is NOT wiped — _overlay_waterfall_payload
    # re-emits the existing accumulator and the evicted frame backfills on a later
    # hydration tick.  The 2D panels stay blank (render_2d empty).  Clearing still
    # happens on Clear / a mode change / a new scan (those empty prev_overlaid_ids
    # or change the reset_key).  overlay_read_failure_action encodes the policy.
    _overlay_preserve_1d = (
        mode in _SCAN_MODES
        and not render_1d
        and overlay_read_failure_action(method, bool(prev_overlaid_ids)) == 'preserve'
    )

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
        elif _overlay_preserve_1d:
            # Keep the state READY so PLOT_1D draws + re-emits the 1D overlay
            # accumulator; the 2D panels stay blank (render_2d empty -> has_data
            # False), and the evicted frame backfills on a later hydration tick.
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
    # raw — §8 invariant).  UNIVERSAL raw-display policy: the Int 2D raw panel is
    # display-only, so ALWAYS prefer the (cheap, ~70x smaller) thumbnail and fall
    # back to full-res RAW ONLY when no thumbnail exists (e.g. a no-.nxs run).  This
    # is the single consistent policy across Single/Overlay/Waterfall/Sum/Average,
    # and it keeps the live raw repaint cheap (thumbnail copy/levels/upload instead
    # of the full detector).  raw_image rect-scales the thumbnail to the true
    # detector extent, so the displayed dimensions stay correct.  Probe the
    # displayed frame (the latest for the single-frame views; the first for the
    # Sum/Average aggregation) so the full-res fallback keys off the right frame.
    if ready and render_2d:
        probe = render_2d[0] if method in ("Sum", "Average") else render_2d[-1]
        avail = _availability(raw_availability, probe)
        raw_src = choose_raw_source(
            bool(avail.get('has_raw')), bool(avail.get('has_thumbnail')),
            prefer_thumbnail=True, want_raw=True)
    else:
        raw_src = RawSource.NONE

    raw_panel = PanelPlan(
        visible=True, has_data=(raw_src is not RawSource.NONE),
        source=raw_src, apply_mask=apply_mask_for(raw_src))
    cake_panel = PanelPlan(visible=True, has_data=ready and bool(render_2d))
    # PLOT_1D draws on the normal READY path OR the Overlay/Waterfall preserve path
    # (accumulator kept alive though this frame's 1D read was empty).
    plot_panel = PanelPlan(
        visible=True,
        has_data=(ready and bool(render_1d)) or _overlay_preserve_1d)

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

# The legacy panel roles the current Qt widget knows how to clear/draw.  The
# render decision is descriptor-first now (see ``render_roles_for_state``), but
# these remain as the cleanup fallback so switching from a 2D mode into a
# plot-only/viewer mode still blanks stale raw/cake panels.
_LEGACY_RENDER_ROLES = (PanelRole.PLOT_1D, PanelRole.RAW_2D, PanelRole.CAKE_2D)
# Compatibility alias for old tests/imports.  Do not extend this tuple for new
# modules; put new panel roles in DisplayState.layout instead.
_RENDER_ROLES = _LEGACY_RENDER_ROLES


def _role_from_panel_key(key):
    return getattr(key, "role", key)


def render_roles_for_state(state):
    """Return the panel-role order managed for ``state``.

    The state layout is the primary contract: a future RSM/Stitch/Fit mode can
    add roles by adding panel keys to ``DisplayState.layout``.  The current Qt
    widget still has hard-coded delegates for the legacy integration roles, so
    those roles are appended as cleanup fallbacks when absent from the layout.

    RESTRUCTURE-TODO(WS-X2): promote RenderPlan from role-level draw/clear to
    PanelKey-level draw/clear once the widget has draw delegates for repeated
    roles such as RSM SLICE_2D/PROJ_1D panels.  Role-level planning is enough
    for the current non-repeating integration/viewer panels.
    """
    ordered = []
    rows = getattr(state, "layout", ()) or ()
    if rows:
        keys = [key for row in rows for key in row]
    else:
        keys = [key for key, _plan in (getattr(state, "panels", ()) or ())]

    for key in keys:
        role = _role_from_panel_key(key)
        if isinstance(role, PanelRole) and role not in ordered:
            ordered.append(role)

    for role in _LEGACY_RENDER_ROLES:
        if role not in ordered:
            ordered.append(role)

    return tuple(ordered)


def render_keys_for_state(state):
    """Return the PanelKey order managed for ``state`` — the #69 / WS-X2
    promotion of :func:`render_roles_for_state` from role-level to PanelKey-level.

    Unlike the role version this does NOT dedupe by role, so a repeated-role
    layout (a future RSM/Stitch viewer's multiple ``SLICE_2D``/``PROJ_1D``
    panels) keeps every instance.  Legacy roles absent from the layout are
    appended as singleton ``PanelKey(role)`` cleanup fallbacks, exactly mirroring
    :func:`render_roles_for_state`.  For the current one-panel-per-role
    integration/viewer view every key is a singleton, so this is 1:1 with the
    role list (behaviour-preserving); the renderer still consumes the role-level
    ``RenderPlan.draw``/``clear`` until the widget grows repeated-role delegates
    (layer-(b), deferred)."""
    ordered: list = []
    rows = getattr(state, "layout", ()) or ()
    if rows:
        keys = [key for row in rows for key in row]
    else:
        keys = [key for key, _plan in (getattr(state, "panels", ()) or ())]

    for key in keys:
        if isinstance(_role_from_panel_key(key), PanelRole) and key not in ordered:
            ordered.append(key)

    present_roles = {_role_from_panel_key(k) for k in ordered}
    for role in _LEGACY_RENDER_ROLES:
        if role not in present_roles:
            ordered.append(PanelKey(role))

    return tuple(ordered)


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


def stitch_display_state(mode, generation, *, has_1d, has_2d, title=""):
    """A :class:`DisplayState` for a whole-scan stitch result (STITCH_1D/2D).

    Pure / Qt-free.  Unlike :func:`compute_display_state` there is no per-frame
    selection: the stitch is a single synthetic panel (PLOT_1D for STITCH_1D,
    CAKE_2D for STITCH_2D) whose ``has_data`` is just "does the matching result
    exist".  ``render_roles_for_state`` appends the other legacy roles as cleanup,
    so the unused panels (raw + the other dimension) are always blanked.

    ``load_status`` is READY when the relevant result exists, else EMPTY — so a
    Stitch mode selected before a run renders an explicit blank rather than stale
    per-frame content.
    """
    if mode is Mode.STITCH_2D:
        role = PanelRole.CAKE_2D
        has = bool(has_2d)
    else:
        role = PanelRole.PLOT_1D
        has = bool(has_1d)
    key = PanelKey(role)
    panels = ((key, PanelPlan(visible=True, has_data=has)),)
    layout = ((key,),)
    return DisplayState(
        mode=mode,
        load_status=LoadStatus.READY if has else LoadStatus.EMPTY,
        error_message=None,
        generation=generation,
        selected_ids=(),
        render_ids=(),
        overall=True,                # a stitch aggregates the whole scan
        gi=False,
        x_unit="q_A^-1",
        x_label="q",
        method="Single",
        overlay=OverlayAction.REPLACE,
        overlaid_ids=(),
        title=title,
        panels=panels,
        layout=layout,
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


def stitch_plot_payload(result):
    """``IntegrationResult1D`` (a stitched 1-D pattern) → ``PlotPayload``.

    Pure / Qt-free.  Returns ``None`` for a missing/empty/all-NaN result so the
    caller skips drawing.  One ``data`` trace; the x-axis label/unit come from
    the result's pyFAI unit string (the same mapping the integration view uses).
    """
    if result is None:
        return None
    radial = np.asarray(getattr(result, "radial", None), dtype=float)
    inten = np.asarray(getattr(result, "intensity", None), dtype=float)
    if radial.size == 0 or inten.size == 0 or not np.isfinite(inten).any():
        return None
    unit = getattr(result, "unit", "q_A^-1") or "q_A^-1"
    label, _sym = x_axis_for_unit(unit)
    return PlotPayload(
        axis_x=Axis(label=label, unit=unit, values=radial),
        traces=(Trace(label="Stitch", x=radial, y=inten),),
    )


def stitch_image_payload(result):
    """``IntegrationResult2D`` (a stitched cake) → ``ImagePayload``.

    Pure / Qt-free.  ``intensity`` is ``(len(radial), len(azimuthal))``; the
    image-draw delegate transposes ``payload.image`` (rows=y, cols=x), so we
    store ``intensity.T`` with x=radial, y=azimuthal — matching the integration
    cake's orientation.  Returns ``None`` for a missing/empty/all-NaN result.
    """
    if result is None:
        return None
    radial = np.asarray(getattr(result, "radial", None), dtype=float)
    azim = np.asarray(getattr(result, "azimuthal", None), dtype=float)
    inten = np.asarray(getattr(result, "intensity", None), dtype=float)
    if inten.ndim != 2 or inten.size == 0 or not np.isfinite(inten).any():
        return None
    r_unit = getattr(result, "unit", "q_A^-1") or "q_A^-1"
    a_unit = getattr(result, "azimuthal_unit", "chi_deg") or "chi_deg"
    r_label, _ = x_axis_for_unit(r_unit)
    a_label, _ = x_axis_for_unit(a_unit)
    return ImagePayload(
        image=inten.T,                       # (azimuthal, radial) = rows=y, cols=x
        axis_x=Axis(label=r_label, unit=r_unit, values=radial),
        axis_y=Axis(label=a_label, unit=a_unit, values=azim),
    )


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
    # #69 / WS-X2: PanelKey-level draw/clear (no role dedupe), so repeated-role
    # layouts (RSM/Stitch) are expressible.  For the current one-panel-per-role
    # view these mirror draw/clear 1:1 by role; the renderer still consumes the
    # role-level draw/clear until the widget grows repeated-role delegates.
    draw_keys: tuple = ()
    clear_keys: tuple = ()


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
    for role in render_roles_for_state(state):
        plan = state.panel(role)
        if ready and plan is not None and plan.has_data:
            draw.append(role)
        else:
            clear.append(role)
    # #69 / WS-X2: the same decision at PanelKey granularity (no role dedupe).
    # For the current view these mirror draw/clear 1:1 by role; for a
    # repeated-role layout they carry every instance.  Additive — the renderer
    # still reads draw/clear today.
    draw_keys, clear_keys = [], []
    for key in render_keys_for_state(state):
        plan = state.panel(key)
        if ready and plan is not None and plan.has_data:
            draw_keys.append(key)
        else:
            clear_keys.append(key)
    return RenderPlan(drop=False, error_message=state.error_message,
                      title=state.title, draw=tuple(draw), clear=tuple(clear),
                      draw_keys=tuple(draw_keys), clear_keys=tuple(clear_keys))
