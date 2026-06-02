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
# Stage 0: stubs only.  Bodies move here from the widget in Stage 1+.
# They are declared now so the test file (the behaviour contract) can
# import and call them; they raise until the owning stage implements them.

_STAGE1 = "implemented in Stage 1 of the display refactor"
_STAGE2 = "implemented in Stage 2 of the display refactor"
_STAGE4 = "implemented in Stage 4 of the display refactor"


def resolve_selection(frame_ids, all_frame_index):
    """Return ``(sorted_ids, overall)``.  ``overall`` is True iff every
    scan frame is selected and there is more than one.  Replaces the body
    of ``displayFrameWidget.get_idxs``."""
    raise NotImplementedError(_STAGE1)


def resolve_render_ids(selected_ids, overall, all_frame_index, loaded_keys):
    """Frames we can actually draw = ``(overall ? all : selected) ∩ loaded_keys``,
    sorted."""
    raise NotImplementedError(_STAGE1)


def choose_raw_source(has_raw, has_thumbnail, *, prefer_thumbnail, want_raw):
    """The map_raw vs thumbnail vs none decision currently inside
    ``get_frames_map_raw``."""
    raise NotImplementedError(_STAGE1)


def apply_mask_for(source):
    """True only for :attr:`RawSource.RAW` — detector/flat masks are
    applied to full-resolution raw arrays only, never to thumbnails."""
    raise NotImplementedError(_STAGE1)


def x_axis_for_unit(unit):
    """``(label, unit_symbol)`` for a plot/integration unit.  One table,
    used by both normal mode and the XYE viewer.  ``'unknown'`` →
    ``('x', '')`` — never an assumed 2θ."""
    raise NotImplementedError(_STAGE1)


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
    """Return ``arr`` with NaN where values are non-finite or hit the
    uint32 dead/hot-pixel sentinel (e.g. 4294967295 from Eiger)."""
    raise NotImplementedError(_STAGE1)


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
