# -*- coding: utf-8 -*-
"""Pure, Qt-free display decisions for the Scattering Workspace.

This module is the single source of truth for *what should be on screen*.
It is deliberately free of Qt, pyqtgraph, h5py and pyFAI so its decision
logic can be unit-tested headlessly.  Current Scattering Workspace adapters
consume these contracts; this is production decision logic, not a staged shim.

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

import logging
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
# X1 Slice 3b: the typed capability contracts (import-light session module —
# the display_logic purity guard stays green).
from xrd_tools.session.frame_projection import (
    Capability,
    CapabilityDisposition,
    CapabilityState,
    DisplayCapabilities,
)
from xrd_tools.session.hydration import HydrationPurpose
from xrd_tools.core.invalid import (
    UINT32_CEILING as _UINT32_CEILING,
    integer_saturation_ceiling as _core_saturation_ceiling,
    saturation_pixels as _saturation_pixels,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Mode",
    "RawSource",
    "OverlayAction",
    "DataTier",
    "ReadStatus",
    "ReadPolicyAction",
    "HydrationSupersedeAction",
    "SupersedeReason",
    "ConsumerKind",
    "LoadStatus",
    "PanelRole",
    "PanelKey",
    "PanelPlan",
    "SelectedCapabilityContext",
    "ReadResult",
    "Axis",
    "Trace",
    "PlotPayload",
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
    "resolve_frame_data",
    "read_policy_action",
    "hydration_supersede_action",
    "choose_raw_source",
    "apply_mask_for",
    "x_axis_for_unit",
    "pretty_unit",
    "canonical_axis_key",
    "canonical_axis_unit",
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


class DataTier(Enum):
    ONE_D = "1d"
    TWO_D = "2d"
    RAW_OR_THUMBNAIL = "raw_or_thumbnail"
    PUBLICATION = "publication"


class ReadStatus(Enum):
    RESIDENT = "resident"
    EVICTED_HYDRATING = "evicted_hydrating"
    ABSENT = "absent"


class ReadPolicyAction(Enum):
    DRAW = "draw"
    PRESERVE_EXISTING = "preserve-existing"
    BLANK_AWAIT = "blank-await"
    ANNOTATE = "annotate"


class HydrationSupersedeAction(Enum):
    CANCEL = "cancel"
    COMPLETE_AND_APPEND = "complete-and-append"


class SupersedeReason(Enum):
    SELECTION = "selection"
    RESET = "reset"
    SCAN_SWITCH = "scan-switch"
    GENERATION = "generation"


class ConsumerKind(Enum):
    PLOT_1D = "plot_1d"
    OVERLAY_1D = "overlay_1d"
    RAW_2D = "raw_2d"
    CAKE_2D = "cake_2d"
    IMAGE_VIEWER = "image_viewer"
    XYE_VIEWER = "xye_viewer"
    NEXUS_VIEWER = "nexus_viewer"


# ── Data shapes (§4 + §10 of the plan) ────────────────────────────────

@dataclass(frozen=True)
class ReadResult:
    label: object
    tier_needed: DataTier
    status: ReadStatus
    data: object = None
    source: str = ""
    has_raw: bool = False
    has_thumbnail: bool = False

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
    #: X1 Slice 3b (R3-P3): the typed capability fact for THIS panel's
    #: selected frame, when the probe matches the pinned scan-qualified
    #: projection — the Capability ITSELF (state/disposition/reason), never
    #: copied into fields that can drift.  ``None`` on legacy/aggregate/
    #: viewer paths.  Policy consumers key on ``capability.disposition``;
    #: the prose reason stays diagnostic.
    capability: "Capability | None" = None


@dataclass(frozen=True)
class SelectedCapabilityContext:
    """Value-only snapshot of the pinned selected-frame projection (X1 3b).

    ``(scan_key, label)`` is the pin's scan-qualified identity — gates compare
    BOTH, never a bare label (contract 8).  ``integrated_1d`` drives the typed
    hydration eligibility at the :func:`resolve_frame_data` request boundary
    (R3-P8); ``capabilities`` carries the full per-panel facts for
    :func:`compute_display_state` (R3-P3)."""

    scan_key: "object" = None
    label: "object" = None
    integrated_1d: "Capability | None" = None
    capabilities: "DisplayCapabilities | None" = None


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

    ``overlaid_ids`` / ``plot_history`` / ``display_ids`` are reserved typed
    provenance slots for future operation-owned display payloads.  The current
    ScientificView owns its live Overlay/Waterfall rows and does not use these
    fields.  Optional values must preserve the frozen-dataclass invariant (for
    example, use tuples rather than mutable lists).
    """
    axis_x: Axis
    traces: tuple = ()   # tuple[Trace, ...]
    axis_y: "Axis | None" = None
    overlaid_ids: "tuple | None" = None      # frame ids accumulated (Overlay/Waterfall)
    plot_history: "object | None" = None     # immutable accumulator view
    display_ids: "tuple | None" = None       # ids for the emitted/painted traces


def waterfall_display_rows(rows, ids, max_rows):
    """Display-only row decimation for a large Waterfall image.

    The caller retains the full source sequence.  This helper bounds only the
    rows painted in one render and applies ONE index set to the rows and row ids
    so axis labels remain aligned with the displayed image.  The index set
    always preserves the FIRST and the TERMINAL row and spans the full history
    extent — WF-END-2: the old stride slice returned ``1, 4, ..., 649`` for a
    complete 651-row source at ``max_rows=256``, silently omitting the finished
    run's top row even though all 651 rows were available.

    Returns ``(rows_display, ids_display, indices)``; ``indices`` is ``None``
    when no decimation was applied, else the selected row indices.  Consumers
    must slice any parallel per-row columns with the SAME set.
    """
    rows = np.asarray(rows)
    ids = tuple(ids)
    if max_rows is None:
        return rows, ids, None
    max_rows = int(max_rows)
    n_rows = int(rows.shape[0]) if rows.ndim else 0
    if len(ids) != n_rows:
        ids = tuple(range(n_rows))
    if max_rows <= 0 or n_rows <= max_rows:
        return rows, ids, None
    if max_rows == 1:
        indices = np.array([n_rows - 1], dtype=int)   # the terminal row
    else:
        indices = np.unique(np.round(
            np.linspace(0, n_rows - 1, max_rows)).astype(int))
    return rows[indices], tuple(ids[j] for j in indices), indices


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

    ``rendered_axis_key`` / ``rendered_kind`` are the V3 rendered identity
    (design_gui_display_robustness §5, F-B): stamped by the CAKE builder at
    payload-build time, they describe what is actually ON the 2D panel —
    never a combo read at render time.  ``rendered_axis_key`` is the
    canonical radial-axis key AS RENDERED (:func:`canonical_axis_key` over
    the rendered unit token: the imageUnit Q↔2θ resample target when the
    conversion fired, else the publication's native token — so an
    unconvertible cake, e.g. no wavelength, is honestly keyed native even
    while the combo says otherwise).  ``rendered_kind`` is the
    ``two_d_kind`` classifier output (``'q_chi'`` / ``'qip_qoop'`` /
    ``'qtot_chigi'`` / ``'exit_angles'``), so GI panels are fully
    self-describing.  Both stay ``None`` for raw-pixel/viewer images.
    """
    image: "np.ndarray"
    axis_x: Axis = Axis("x", "")
    axis_y: Axis = Axis("y", "")
    gap_mask_indices: "np.ndarray | None" = None
    raw_full_shape: "tuple | None" = None
    rendered_axis_key: "str | None" = None
    rendered_kind: "str | None" = None


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


# GI subscript spellings (display_constants.Qip_s / Qoop_s), inlined so the
# canonical-key classifier stays headless (xrd_tools cannot import xdart).
_QIP_S = u'Qᵢₚ'          # Q_ip subscript (Qᵢₚ)
_QOOP_S = u'Qₒₒₚ'   # Q_oop subscript (Qₒₒₚ)


def canonical_axis_key(text):
    """Canonical axis-identity key for a plot/cake axis (V3 rendered identity).

    Accepts any axis spelling in circulation — pyFAI unit tokens
    (``'q_A^-1'``, ``'2th_deg'``, ``'qip_A^-1'``), combo labels
    (``'Q (Å⁻¹)'``, ``'2θ (°)'``), rendered-axis label+symbol pairs, the GI
    unicode subscripts and the ``_X_AXIS_TABLE`` HTML ``<sub>`` labels — and
    maps them all onto ONE comparable vocabulary: ``'qoop_A^-1'`` /
    ``'qip_A^-1'`` / ``'exit_angle_deg'`` / ``'2th_deg'`` / ``'chigi_deg'``
    / ``'chi_deg'`` / ``'q_A^-1'`` (q_total deliberately keys as
    ``'q_A^-1'`` — GI polar pairs with the plain Q plot axis).  Unknown
    axes return their lowercased text; empty input returns ``''``.

    This is THE key both sides of the Share-Axis identity comparison use:
    the cake payload's ``rendered_axis_key`` is stamped with it at build,
    and the widget derives the 1D key from the rendered plot payload axis
    with it (the former ``displayFrameWidget._axis_key_from_label``, moved
    headless so payload builders can stamp identity without Qt)."""
    text = str(text or '')
    # Normalize the HTML subscript spellings ('Q<sub>ip</sub>') so they
    # classify like their plain forms ('Qip').
    lower = text.replace('<sub>', '').replace('</sub>', '').lower()
    if _QOOP_S in text or 'qoop' in lower or 'q_oop' in lower:
        return 'qoop_A^-1'
    if _QIP_S in text or 'qip' in lower or 'q_ip' in lower:
        return 'qip_A^-1'
    if 'exit' in lower:
        return 'exit_angle_deg'
    if '2th' in lower or f'2{_TH}' in text:
        return '2th_deg'
    if (_CHI in text or 'chi' in lower) and 'gi' in lower:
        return 'chigi_deg'
    if _CHI in text or 'chi' in lower:
        return 'chi_deg'
    if 'q' in lower or _AA_INV in text:
        return 'q_A^-1'
    return lower.strip()


def canonical_axis_unit(label, unit):
    """Recover a canonical storage token from a rendered axis pair.

    Plot payloads intentionally carry presentation symbols such as ``Å⁻¹``
    and ``°``.  Accumulator storage must instead retain tokens such as
    ``q_A^-1`` and ``2th_deg`` so a later native row is not rejected as a
    cross-unit append.  Prefer an exact reverse lookup through the shared axis
    table, then classify known display symbols for alternate label spellings
    (notably GI unicode subscripts).  Unknown and scaled units such as
    ``q_nm^-1`` pass through unchanged.
    """
    label = str(label or "")
    unit = str(unit or "")
    if unit in _X_AXIS_TABLE:
        return unit
    for token, axis_pair in _X_AXIS_TABLE.items():
        if axis_pair == (label, unit):
            return token

    label_text = label.replace("<sub>", "").replace("</sub>", "")
    label_lower = label_text.lower().replace("_", "")
    if unit in {_AA_INV, "Å⁻¹", "A^-1", "angstrom^-1"}:
        if "total" in label_lower or "qtot" in label_lower:
            return "qtot_A^-1"
        key = canonical_axis_key(f"{label} ({unit})")
        if key in {"q_A^-1", "qip_A^-1", "qoop_A^-1"}:
            return key
    if unit == _DEG:
        key = canonical_axis_key(f"{label} ({unit})")
        if key in {"2th_deg", "chi_deg", "chigi_deg", "exit_angle_deg"}:
            return key
    return unit


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


def _label_key(label):
    try:
        return int(label)
    except (TypeError, ValueError):
        return label


def _tier(value):
    if isinstance(value, DataTier):
        return value
    try:
        return DataTier(str(value))
    except ValueError:
        return DataTier.PUBLICATION


def _mode(value):
    if isinstance(value, Mode):
        return value
    try:
        return Mode(str(value))
    except ValueError:
        return value


def _scan_mode(value) -> bool:
    return _mode(value) in (Mode.INT_1D, Mode.INT_2D)


def _view_has_tier(view, tier_needed):
    if view is None:
        return False
    tier_needed = _tier(tier_needed)
    if tier_needed is DataTier.ONE_D:
        return bool(
            getattr(view, "has_1d", False)
            or getattr(view, "intensity_1d", None) is not None
        )
    if tier_needed is DataTier.TWO_D:
        return bool(
            getattr(view, "has_2d", False)
            or getattr(view, "intensity_2d", None) is not None
        )
    if tier_needed is DataTier.RAW_OR_THUMBNAIL:
        return bool(
            getattr(view, "raw", None) is not None
            or getattr(view, "thumbnail", None) is not None
        )
    return True


def _publication_has_tier(publication, tier_needed):
    if publication is None:
        return False
    return _view_has_tier(getattr(publication, "view", None), tier_needed)


def _publication_raw_flags(publication):
    view = getattr(publication, "view", None)
    if view is None:
        return False, False
    return (
        getattr(view, "raw", None) is not None,
        getattr(view, "thumbnail", None) is not None,
    )


def _viewer_row_for_tier(label, tier_needed, viewer_rows_1d=None, viewer_rows_2d=None):
    key = _label_key(label)
    tier_needed = _tier(tier_needed)
    if tier_needed in (DataTier.ONE_D, DataTier.PUBLICATION):
        if hasattr(viewer_rows_1d, "get"):
            frame = viewer_rows_1d.get(key)
            if frame is not None:
                return frame, False, False
    if tier_needed in (DataTier.TWO_D, DataTier.RAW_OR_THUMBNAIL):
        if hasattr(viewer_rows_2d, "get"):
            frame_2d = viewer_rows_2d.get(key)
            if isinstance(frame_2d, dict):
                if tier_needed is DataTier.TWO_D:
                    # Image/NeXus viewer rows are file-browser payloads.  A row
                    # can be considered loaded even when it carries only raw or
                    # thumbnail data for the display panel.
                    return frame_2d, False, False
                else:
                    has_raw = frame_2d.get("map_raw") is not None
                    has_thumbnail = frame_2d.get("thumbnail") is not None
                    if has_raw or has_thumbnail:
                        return frame_2d, has_raw, has_thumbnail
    return None, False, False


def _request_tier_hydration(request_hydration, label, tier_needed):
    if request_hydration is None:
        return False
    purpose = (
        HydrationPurpose.ONE_D
        if _tier(tier_needed) is DataTier.ONE_D
        else HydrationPurpose.FULL
    )
    try:
        request_hydration(label, purpose=purpose)
        return True
    except Exception:
        return False


def _hydration_suppressed_by_selected_capability(
        selected_capability, request_scan_key, key, tier_needed):
    """X1 Slice 3b (R3-P8): the typed hydration-eligibility decision at the
    request boundary — BEFORE anything is queued.

    Suppress a 1D request iff the request's exact ``(scan_key, label)``
    matches the pinned selection AND the pinned ``integrated_1d`` disposition
    is DROPPED or PERSISTED_NO_HYDRATOR — a TYPED comparison, never a
    reason-string parse (R3-P2).  Any identity mismatch or missing pin keeps
    the ordinary behavior; ABSENT does not suppress (record-not-present is not
    proof recovery is pointless).  ``HydrationPurpose.FULL`` is never gated in this
    slice: it serves both the integrated-2D and raw/thumbnail tiers, and
    record capabilities cannot see every production recovery path."""
    if selected_capability is None:
        return False
    if _tier(tier_needed) is not DataTier.ONE_D:
        return False
    fact = selected_capability.integrated_1d
    if fact is None:
        return False
    if selected_capability.scan_key != request_scan_key:
        return False
    if _label_key(selected_capability.label) != key:
        return False
    return getattr(fact, "disposition", None) in (
        CapabilityDisposition.DROPPED,
        CapabilityDisposition.PERSISTED_NO_HYDRATOR,
    )


def resolve_frame_data(
    label,
    mode,
    tier_needed,
    *,
    store_first_lookup=None,
    publication_store=None,
    viewer_rows_1d=None,
    viewer_rows_2d=None,
    include_legacy=True,
    request_hydration=None,
    selected_capability=None,
    request_scan_key=None,
):
    """Resolve one display datum through the H8 store-first read chain.

    The function is intentionally duck-typed and Qt-free.  It never calls a
    blocking hydrate method; an unavailable-but-expected frame becomes
    ``EVICTED_HYDRATING`` and the optional request callback is invoked instead.
    Scan displays read store-first accessor -> PublicationStore -> hydration.
    Viewer modes may still read their file-browser row stores.  Scan modes
    never consult those rows.

    X1 Slice 3b: ``selected_capability`` (a :class:`SelectedCapabilityContext`)
    plus the request's ``request_scan_key`` gate a pointless ``"1d"`` request
    for a selected frame whose typed 1D disposition proves recovery is
    impossible — see :func:`_hydration_suppressed_by_selected_capability`.
    """
    key = _label_key(label)
    tier_needed = _tier(tier_needed)
    scan_mode = _scan_mode(mode)

    if callable(store_first_lookup):
        publication = store_first_lookup(key, allow_blocking_read=False)
        if _publication_has_tier(publication, tier_needed):
            has_raw, has_thumbnail = _publication_raw_flags(publication)
            return ReadResult(
                key, tier_needed, ReadStatus.RESIDENT, publication,
                source="store_first", has_raw=has_raw,
                has_thumbnail=has_thumbnail)
        if publication is not None and tier_needed is DataTier.PUBLICATION:
            has_raw, has_thumbnail = _publication_raw_flags(publication)
            return ReadResult(
                key, tier_needed, ReadStatus.RESIDENT, publication,
                source="store_first", has_raw=has_raw,
                has_thumbnail=has_thumbnail)

    if publication_store is not None:
        publication = None
        get = getattr(publication_store, "get", None)
        if callable(get):
            publication = get(key)
        if _publication_has_tier(publication, tier_needed):
            has_raw, has_thumbnail = _publication_raw_flags(publication)
            return ReadResult(
                key, tier_needed, ReadStatus.RESIDENT, publication,
                source="publication_store", has_raw=has_raw,
                has_thumbnail=has_thumbnail)
        if publication is not None and tier_needed is DataTier.PUBLICATION:
            has_raw, has_thumbnail = _publication_raw_flags(publication)
            return ReadResult(
                key, tier_needed, ReadStatus.RESIDENT, publication,
                source="publication_store", has_raw=has_raw,
                has_thumbnail=has_thumbnail)

    if include_legacy and not scan_mode:
        data, has_raw, has_thumbnail = _viewer_row_for_tier(
            key, tier_needed, viewer_rows_1d=viewer_rows_1d, viewer_rows_2d=viewer_rows_2d)
        if data is not None:
            return ReadResult(
                key, tier_needed, ReadStatus.RESIDENT, data,
                source="viewer_row", has_raw=has_raw,
                has_thumbnail=has_thumbnail)

    # Scan display paths with a store/store-first accessor can expect a later
    # async completion.  Viewer-only row misses are genuinely absent.
    can_hydrate = scan_mode and (
        callable(store_first_lookup)
        or publication_store is not None
        or request_hydration is not None
    )
    if can_hydrate:
        # X1 Slice 3b (R3-P8): typed eligibility BEFORE the queue — a selected
        # 1D row proven DROPPED / PERSISTED_NO_HYDRATOR gets no futile request
        # (and honestly reports ABSENT: nothing is coming).
        if _hydration_suppressed_by_selected_capability(
                selected_capability, request_scan_key, key, tier_needed):
            return ReadResult(
                key, tier_needed, ReadStatus.ABSENT,
                source="capability-suppressed")
        _request_tier_hydration(request_hydration, key, tier_needed)
        return ReadResult(
            key, tier_needed, ReadStatus.EVICTED_HYDRATING,
            source="hydration")
    return ReadResult(key, tier_needed, ReadStatus.ABSENT, source="absent")


def _consumer_kind(value):
    if isinstance(value, PanelRole):
        return {
            PanelRole.PLOT_1D: ConsumerKind.PLOT_1D,
            PanelRole.RAW_2D: ConsumerKind.RAW_2D,
            PanelRole.CAKE_2D: ConsumerKind.CAKE_2D,
        }.get(value, ConsumerKind.PLOT_1D)
    if isinstance(value, ConsumerKind):
        return value
    try:
        return ConsumerKind(str(value))
    except ValueError:
        return ConsumerKind.PLOT_1D


def _supersede_reason(value):
    if isinstance(value, SupersedeReason):
        return value
    try:
        return SupersedeReason(str(value))
    except ValueError:
        return SupersedeReason.GENERATION


def read_policy_action(consumer_kind, status, *, method=None, has_accumulator=False):
    """Single display policy table for typed read outcomes."""
    consumer_kind = _consumer_kind(consumer_kind)
    if not isinstance(status, ReadStatus):
        status = ReadStatus(str(status))

    if status is ReadStatus.RESIDENT:
        return ReadPolicyAction.DRAW
    if (
        consumer_kind is ConsumerKind.PLOT_1D
        and method in ("Overlay", "Waterfall")
        and has_accumulator
        and status is ReadStatus.EVICTED_HYDRATING
    ):
        return ReadPolicyAction.PRESERVE_EXISTING
    if consumer_kind in (ConsumerKind.IMAGE_VIEWER, ConsumerKind.NEXUS_VIEWER):
        return ReadPolicyAction.ANNOTATE if status is ReadStatus.ABSENT else ReadPolicyAction.BLANK_AWAIT
    return ReadPolicyAction.BLANK_AWAIT


def hydration_supersede_action(consumer_kind, reason):
    """Policy for queued async hydrations when a newer display generation exists.

    Ordinary consumers are latest-wins: stale queued work is cancelled.  Overlay
    1D is additive, so a request made by a previous selection is still wanted by
    the accumulator and must finish in request order.  Hard boundaries (reset,
    scan switch, explicit generation cancellation) still cancel it so old scans
    cannot append after the display identity changed.
    """
    consumer_kind = _consumer_kind(consumer_kind)
    reason = _supersede_reason(reason)
    if (
        consumer_kind is ConsumerKind.OVERLAY_1D
        and reason is SupersedeReason.SELECTION
    ):
        return HydrationSupersedeAction.COMPLETE_AND_APPEND
    return HydrationSupersedeAction.CANCEL


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
    action = read_policy_action(
        ConsumerKind.PLOT_1D,
        ReadStatus.EVICTED_HYDRATING,
        method=method,
        has_accumulator=has_accumulator,
    )
    if action is ReadPolicyAction.PRESERVE_EXISTING:
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
    """Return a float copy with selected detector values replaced by NaN.

    Non-finite values remain invalid. Finite sentinels and every native ceiling
    pixel are excluded only when Mask Saturated is enabled. A supplied ceiling
    preserves the native dtype's limit after a caller converts to float.
    """
    orig = np.asarray(arr)
    a = orig.astype(float)
    bad = ~np.isfinite(a)
    if mask_saturation:
        bad |= (a < 0) | (a >= _UINT32_CEILING)
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


def _none_drawable_raw_fact(caps):
    """The raw panel's typed fact when NO source is drawable by the active
    view (X1 3b): ERROR if either raw fact is ERROR; else the AVAILABLE fact
    (the caps-vs-active-view divergence — its typed truth is carried while
    the behavior stays pending-equivalent: nothing drawn, recovery allowed);
    else a PENDING fact (thumbnail preferred, mirroring the draw preference);
    else the raw fact (UNAVAILABLE/ABSENT)."""
    for fact in (caps.raw, caps.thumbnail):
        if fact.state is CapabilityState.ERROR:
            return fact
    for fact in (caps.raw, caps.thumbnail):
        if fact.state is CapabilityState.AVAILABLE:
            return fact
    for fact in (caps.thumbnail, caps.raw):
        if fact.state is CapabilityState.PENDING:
            return fact
    return caps.raw


def compute_display_state(*, mode, selected_ids, all_frame_index, loaded_1d_keys,
                          loaded_2d_keys, gi, plot_unit, method, unit_changed,
                          prev_overlaid_ids, raw_availability, titles,
                          generation=0, loading=False,
                          selected_capability=None, current_scan_key=None):
    """Compose the pure selectors into one immutable :class:`DisplayState`
    describing exactly what each panel should show.  THE function the GUI
    calls.  Pure: no Qt, no I/O, no mutation of inputs.

    ``raw_availability`` maps a frame id to ``{'has_raw': bool,
    'has_thumbnail': bool}``; a special ``'__error__'`` key (mapped to a
    message) marks a failed load.  ``titles`` maps ``mode.value`` to the
    title/filename for the current selection.  ``loading`` is True while a
    load is in flight (lets EMPTY and LOADING be distinguished).

    X1 Slice 3b (R3-P3): ``selected_capability`` is the pinned selected-frame
    :class:`SelectedCapabilityContext`; ``current_scan_key`` is the REQUESTING
    display's scan identity.  When a panel's probe matches the pin's
    scan-qualified ``(scan_key, label)``, that panel carries its own typed
    :class:`Capability` and blanks on its OWN fact's ERROR/UNAVAILABLE — never
    on "any error anywhere".  A genuine identity conflict (every fact ERROR)
    fails the whole selected-frame request explicitly.  Aggregate probes that
    differ from the pin and legacy/viewer paths are untouched.
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
    elif (
        mode is Mode.INT_2D
        and method in ("Single", "Overlay", "Waterfall")
        and render_1d
        and (not render_2d or len(render_1d) > len(render_2d))
    ):
        primary = render_1d
    else:
        primary = render_1d if mode in _PLOT_PRIMARY_MODES else render_2d

    # Overlay/Waterfall preserve (Phase-5 cap-store regression fix): when the 1D
    # accumulator already holds frames but THIS selection's 1D read is empty (the
    # selected frame is evicted past the store cap, awaiting async hydration), keep
    # PLOT_1D drawable so the overlay is NOT wiped — _overlay_waterfall_payload
    # re-emits the existing accumulator and the evicted frame backfills on a later
    # hydration tick.  The 2D panels stay blank (render_2d empty).  Clearing still
    # happens on Clear / a mode change / an incompatible grid-source change (those
    # empty prev_overlaid_ids or change the reset_key).  overlay_read_failure_action
    # encodes the policy.
    _overlay_preserve_1d = (
        mode in _SCAN_MODES
        and not render_1d
        and read_policy_action(
            ConsumerKind.PLOT_1D,
            ReadStatus.EVICTED_HYDRATING,
            method=method,
            has_accumulator=bool(prev_overlaid_ids),
        ) is ReadPolicyAction.PRESERVE_EXISTING
    )

    x_label, _sym = x_axis_for_unit(plot_unit)

    # X1 Slice 3b: the pinned selected-frame capability context.  A panel's
    # typed layer applies only when its probe matches the pin's scan-qualified
    # (scan_key, label) — never a bare label (contract 8), and never for an
    # aggregate probe that differs from the pin (S2-R3 analog).
    # X1 3c (S3-OR1): the typed layer applies to EVERY scan-matching pin —
    # including an ABSENT one, whose all-UNAVAILABLE facts blank the selected
    # frame fail-closed (publication ownership is now explicit, so absence is
    # a real scan-qualified verdict, not an unprovable-identity artifact).
    _sel_caps = (getattr(selected_capability, "capabilities", None)
                 if selected_capability is not None else None)
    _sel_scan_ok = (
        _sel_caps is not None
        and selected_capability.scan_key == current_scan_key)

    def _pin_matches(probe_pool):
        if not _sel_scan_ok:
            return False
        pool = tuple(probe_pool) if probe_pool else ids
        if not pool:
            return False
        probe = pool[0] if method in ("Sum", "Average") else pool[-1]
        return _label_key(probe) == _label_key(selected_capability.label)

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

    # X1 Slice 3b: a GENUINE identity conflict — every pinned fact ERROR — is
    # a whole-selected-frame failure: blank every panel explicitly with the
    # reason, never a stale panel (required test 4/3a).  LoadStatus keeps its
    # whole-request meaning (this IS the whole request failing).
    if (load_status is LoadStatus.READY and _sel_scan_ok
            and _pin_matches(())):
        _facts = (_sel_caps.metadata, _sel_caps.integrated_1d,
                  _sel_caps.integrated_2d, _sel_caps.raw, _sel_caps.thumbnail)
        if all(f.state is CapabilityState.ERROR for f in _facts):
            load_status = LoadStatus.ERROR
            error_message = (_sel_caps.metadata.reason
                             or "selected record identity conflict")
            render_ids = ()
            title = ''

    ready = load_status is LoadStatus.READY

    # 2D-raw panel: the raw-vs-thumbnail-vs-none decision (mask only on full
    # raw — §8 invariant).  UNIVERSAL raw-display policy: the Int 2D raw panel is
    # display-only, so ALWAYS prefer the (cheap, ~70x smaller) thumbnail and fall
    # back to full-res RAW ONLY when no thumbnail exists (e.g. a no-output run).  This
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

    # X1 Slice 3b (R3-P3): the per-panel typed layer.  DRAW-source truth stays
    # the active view (the booleans above); capabilities govern the typed
    # layer on top — the carried Capability, never-stale/error blanking, and
    # the pending-equivalent divergence.  Each panel blanks on its OWN fact's
    # ERROR/UNAVAILABLE only; PENDING keeps today's behavior.
    if _sel_caps is not None:
        if _pin_matches(render_2d):
            _fact_2d = _sel_caps.integrated_2d
            cake_panel = PanelPlan(
                visible=True,
                has_data=(cake_panel.has_data and _fact_2d.state not in
                          (CapabilityState.ERROR, CapabilityState.UNAVAILABLE)),
                capability=_fact_2d)
            if raw_src is RawSource.RAW:
                _fact_raw = _sel_caps.raw
            elif raw_src is RawSource.THUMBNAIL:
                _fact_raw = _sel_caps.thumbnail
            else:
                _fact_raw = _none_drawable_raw_fact(_sel_caps)
            if _fact_raw.state in (CapabilityState.ERROR,
                                   CapabilityState.UNAVAILABLE):
                raw_panel = PanelPlan(
                    visible=True, has_data=False, source=RawSource.NONE,
                    apply_mask=False, capability=_fact_raw)
            else:
                raw_panel = PanelPlan(
                    visible=True, has_data=raw_panel.has_data,
                    source=raw_panel.source, apply_mask=raw_panel.apply_mask,
                    capability=_fact_raw)
        if _pin_matches(render_1d):
            _fact_1d = _sel_caps.integrated_1d
            plot_panel = PanelPlan(
                visible=True,
                has_data=((plot_panel.has_data and _fact_1d.state not in
                           (CapabilityState.ERROR, CapabilityState.UNAVAILABLE))
                          or _overlay_preserve_1d),
                capability=_fact_1d)

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
