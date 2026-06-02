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
strings so we need not import numpy at module load either.
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
    "PanelPlan",
    "DisplayState",
    "DisplayPayload",
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


# ── Data shapes (§4 of the plan) ──────────────────────────────────────

@dataclass(frozen=True)
class PanelPlan:
    visible: bool
    has_data: bool                       # False ⇒ render() clears this panel
    source: RawSource = RawSource.NONE   # 2D-raw panel only
    apply_mask: bool = False             # 2D-raw panel only


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
    raw_panel: PanelPlan
    cake_panel: PanelPlan
    plot_panel: PanelPlan


@dataclass(frozen=True)
class DisplayPayload:
    """Resolved arrays for one :class:`DisplayState` (assembled from the
    DataStore).  Kept separate so :func:`compute_display_state` stays
    pure/cheap and array assembly is tested on its own.  ``None`` ⇒ that
    panel renders blank."""
    generation: int                 # must match the DisplayState it pairs with
    raw_image: "np.ndarray | None"
    cake_image: "np.ndarray | None"
    plot_x: "np.ndarray | None"
    plot_y: "np.ndarray | None"     # 2D (rows × x) for overlay/waterfall


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


def compute_display_state(*, mode, selected_ids, all_frame_index, loaded_1d_keys,
                          loaded_2d_keys, gi, plot_unit, method, unit_changed,
                          prev_overlaid_ids, raw_availability, titles):
    """Compose the above into one :class:`DisplayState`.  THE function the
    GUI calls."""
    raise NotImplementedError(_STAGE2)
