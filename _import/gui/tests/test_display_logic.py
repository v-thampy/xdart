# -*- coding: utf-8 -*-
"""Headless contract for the pure display-decision layer.

These tests define the behaviour contract for
``xdart/gui/tabs/static_scan/display_logic.py`` — the §8 *Display
invariants* of ``display_refactor_plan.md`` are the acceptance contract.

Per the plan (Stage 0): these assert the **correct intended** outcomes,
including the ones the old code got wrong.  They therefore start RED
where the implementation does not yet exist (Stage 0 ships only the
scaffold) and go GREEN as the owning stage lands.  They are the
regression guard, not a snapshot of today's bugs — do NOT relax an
assertion to make it pass; implement the function instead.

All pure Python: no Qt, pyqtgraph, h5py or pyFAI.  Run just these with::

    pytest -m display_logic

The module is imported as a whole (``import ... as dl``) so a not-yet-
implemented function fails its own test rather than breaking collection
for the green scaffold tests.
"""

import dataclasses
import math
import os
import subprocess
import sys
import textwrap

import pytest

import xdart.gui.tabs.static_scan.display_logic as dl

# Absolute path to the module under test, derived from this test's
# location (tests/ is a sibling of xdart/) so the purity guard need not
# import the xdart package at all.
_DISPLAY_LOGIC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "xdart", "gui", "tabs", "static_scan", "display_logic.py",
)

pytestmark = pytest.mark.display_logic


# ── §7: the first six tests — the core contract ───────────────────────

def test_resolve_selection_overall_vs_single():
    # all frames selected (>1) -> overall; subset -> not overall
    assert dl.resolve_selection(['0', '1', '2'], [0, 1, 2]) == ((0, 1, 2), True)
    assert dl.resolve_selection(['1'], [0, 1, 2]) == ((1,), False)


def test_resolve_render_ids_intersects_loaded():
    # render only frames whose data is actually loaded
    assert dl.resolve_render_ids((0, 1, 2), False, [0, 1, 2], {0, 2}) == (0, 2)


def test_choose_raw_source_priority_and_mask():
    assert dl.choose_raw_source(True,  True,  prefer_thumbnail=False, want_raw=True) is dl.RawSource.RAW
    assert dl.choose_raw_source(False, True,  prefer_thumbnail=False, want_raw=True) is dl.RawSource.THUMBNAIL
    assert dl.choose_raw_source(False, False, prefer_thumbnail=False, want_raw=True) is dl.RawSource.NONE
    assert dl.choose_raw_source(True,  True,  prefer_thumbnail=True,  want_raw=True) is dl.RawSource.THUMBNAIL
    # mask only on full raw (invariant: never mask a thumbnail):
    assert dl.apply_mask_for(dl.RawSource.RAW) is True
    assert dl.apply_mask_for(dl.RawSource.THUMBNAIL) is False
    assert dl.apply_mask_for(dl.RawSource.NONE) is False


def test_xye_unit_from_filename():
    assert dl.xye_unit_from_filename('iq_scan_0001.xye') == 'q_A^-1'
    assert dl.xye_unit_from_filename('itth_scan_0001.xye') == '2th_deg'
    # unknown prefix: NO assumption (must not become 2θ)
    assert dl.xye_unit_from_filename('random_scan.xye') == 'unknown'
    # unknown unit -> plain x, no unit symbol
    assert dl.x_axis_for_unit('unknown') == ('x', '')


def test_plan_overlay_rebuild_on_unit_change_keeps_all_ids():
    # Overlay + unit change -> REBUILD all previously overlaid ids (not drop to last)
    action, ids = dl.plan_overlay('Overlay', unit_changed=True, has_existing=True,
                                  new_ids=(3,), prev_overlaid_ids=(0, 1, 2))
    assert action is dl.OverlayAction.REBUILD and set(ids) == {0, 1, 2}
    # Single always replaces:
    action2, _ = dl.plan_overlay('Single', unit_changed=False, has_existing=True,
                                 new_ids=(5,), prev_overlaid_ids=(0, 1, 2))
    assert action2 is dl.OverlayAction.REPLACE


def test_gi_axes_uniform_detects_mismatch():
    q = [0.0, 1.0, 2.0]
    assert dl.gi_axes_uniform([(q, q), (q, q)]) is True
    assert dl.gi_axes_uniform([(q, q), ([0.0, 1.0, 2.5], q)]) is False


# ── §7 bonus tests (the rest of the contract) ─────────────────────────

def test_sentinel_mask_sets_nan():
    np = pytest.importorskip("numpy")
    UINT32_MAX = 4294967295
    arr = np.array([1.0, UINT32_MAX, np.inf, -np.inf, 3.0])
    out = dl.sentinel_mask(arr)
    assert math.isnan(out[1])              # uint32 dead/hot-pixel sentinel
    assert math.isnan(out[2]) and math.isnan(out[3])  # non-finite
    assert out[0] == 1.0 and out[4] == 3.0            # real values untouched


def test_default_plot_unit_follows_2theta():
    # integrate in 2θ -> the plot-unit combo should default to the 2θ entry,
    # not fall back to Q (the 'integrate in 2θ but plot defaults to Q' bug).
    units = ('q_A^-1', '2th_deg', 'chi_deg')
    assert dl.default_plot_unit('2th_deg', units) == 1
    assert dl.default_plot_unit('q_A^-1', units) == 0


def test_x_axis_for_unit_known_units_carry_a_symbol():
    # Known units must resolve to a non-empty unit symbol (only 'unknown'
    # is blank).  Exact symbol strings are pinned by display_constants in
    # the implementing stage; here we guard the invariant, not the glyph.
    for unit in ('q_A^-1', '2th_deg', 'chi_deg'):
        label, sym = dl.x_axis_for_unit(unit)
        assert label and sym, f"{unit!r} must carry a label and a unit symbol"


# ── §8 invariants exercised through compute_display_state ─────────────

def _base_state_kwargs(**overrides):
    """Minimal kwargs for compute_display_state; override per test."""
    kwargs = dict(
        mode=dl.Mode.INT_1D,
        selected_ids=(0,),
        all_frame_index=[0, 1, 2],
        loaded_1d_keys={0},
        loaded_2d_keys={0},
        gi=False,
        plot_unit='q_A^-1',
        method='Single',
        unit_changed=False,
        prev_overlaid_ids=(),
        raw_availability={},
        titles={},
    )
    kwargs.update(overrides)
    return kwargs


def test_compute_display_state_image_viewer_no_data_clears_panel():
    # Invariant: a panel never keeps old content when there is no data.
    # IMAGE_VIEWER with neither raw nor thumbnail -> RAW_2D panel has_data False.
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.IMAGE_VIEWER,
        selected_ids=(0,),
        loaded_2d_keys=set(),
        raw_availability={0: dict(has_raw=False, has_thumbnail=False)},
    ))
    assert state.panel(dl.PanelRole.RAW_2D).has_data is False
    assert state.load_status in (dl.LoadStatus.EMPTY, dl.LoadStatus.ERROR)


def test_error_load_yields_error_status_not_partial():
    # Invariant: a failed load -> LoadStatus.ERROR with an error_message,
    # never a half-populated display.  Every panel reports has_data=False.
    state = dl.compute_display_state(**_base_state_kwargs(
        raw_availability={'__error__': 'boom'},
    ))
    assert state.load_status is dl.LoadStatus.ERROR
    assert state.error_message
    assert all(not plan.has_data for _role, plan in state.panels)


def test_mode_switch_bumps_generation_and_clears_title():
    # Invariant: title/filename never updates independently of the payload;
    # switching modes produces a fresh state whose title matches the new
    # mode (no stale title from the previous mode), atomically.
    img = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.IMAGE_VIEWER,
        titles={'image_viewer': 'frame_0007.tif'},
    ))
    xye = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.XYE_VIEWER,
        titles={'xye_viewer': 'iq_scan_0001.xye'},
    ))
    # The two modes must not share a title.
    assert img.title != xye.title
    assert img.title == 'frame_0007.tif'
    assert xye.title == 'iq_scan_0001.xye'


def test_image_viewer_does_not_depend_on_scan_frames():
    # Invariant: viewer modes do not depend on scan.frames / the
    # integration-unit combo.  selected_ids are *viewer* ids and must be
    # honoured even when no scan frame index is present.
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.IMAGE_VIEWER,
        selected_ids=(0,),
        all_frame_index=[],                       # no scan loaded
        loaded_2d_keys={0},
        raw_availability={0: dict(has_raw=True, has_thumbnail=False)},
    ))
    assert 0 in state.render_ids
    assert state.panel(dl.PanelRole.RAW_2D).has_data is True


def test_load_status_transitions():
    # EMPTY: nothing selected.
    empty = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_1D, selected_ids=(), loaded_1d_keys=set()))
    assert empty.load_status is dl.LoadStatus.EMPTY
    # LOADING: selected, nothing loaded yet, a load is in flight.
    loading = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_1D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys=set(), loading=True))
    assert loading.load_status is dl.LoadStatus.LOADING
    # READY: selected frame is loaded.
    ready = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_1D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}))
    assert ready.load_status is dl.LoadStatus.READY
    # ERROR: load failed -> message, no partial display.
    error = dl.compute_display_state(**_base_state_kwargs(
        raw_availability={'__error__': 'kaboom'}))
    assert error.load_status is dl.LoadStatus.ERROR
    assert error.error_message == 'kaboom'


def test_compute_stamps_generation():
    # The state carries the generation it was computed against (Stage 2
    # plumbing for dropping stale worker results in Stage 5).
    state = dl.compute_display_state(**_base_state_kwargs(generation=7))
    assert state.generation == 7


def test_render_ids_intersect_loaded_in_state():
    # render_ids = (overall ? all : selected) ∩ loaded keys, in-state.
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_1D, selected_ids=(0, 1, 2), all_frame_index=[0, 1, 2],
        loaded_1d_keys={0, 2}))
    assert state.render_ids == (0, 2)
    assert state.overall is True   # all 3 of 3 frames selected


def test_compute_display_state_is_deterministic_and_frozen():
    kw = _base_state_kwargs(mode=dl.Mode.INT_1D, selected_ids=(0,),
                            loaded_1d_keys={0})
    a = dl.compute_display_state(**kw)
    b = dl.compute_display_state(**kw)
    assert a == b                              # same inputs ⇒ same state
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.title = 'mutated'                    # frozen: cannot mutate


def test_title_blank_unless_ready():
    # §8: a title is only ever set together with a valid (READY) payload.
    empty = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.IMAGE_VIEWER, selected_ids=(), loaded_2d_keys=set(),
        titles={'image_viewer': 'frame_0007.tif'}))
    assert empty.load_status is dl.LoadStatus.EMPTY
    assert empty.title == ''                   # no stale/early title
    err = dl.compute_display_state(**_base_state_kwargs(
        raw_availability={'__error__': 'x'},
        titles={'int_1d': 'should_not_show'}))
    assert err.title == ''


# ── §10 seams: the contract surface is open to future modules ─────────

def _make_display_state(**over):
    """Construct a DisplayState directly (no compute) for shape tests."""
    base = dict(
        mode=dl.Mode.INT_1D,
        load_status=dl.LoadStatus.READY,
        error_message=None,
        generation=0,
        selected_ids=(0,),
        render_ids=(0,),
        overall=False,
        gi=False,
        x_unit='q_A^-1',
        x_label='Q',
        method='Single',
        overlay=dl.OverlayAction.REPLACE,
        overlaid_ids=(),
        title='',
        panels=(),
        layout=(),
    )
    base.update(over)
    return dl.DisplayState(**base)


def test_panels_are_keyed_and_results_defaults_none():
    # §10 seam 1 + 5: panels is a keyed collection; results defaults to None.
    state = _make_display_state(panels=(
        (dl.PanelRole.RAW_2D, dl.PanelPlan(visible=True, has_data=True)),
        (dl.PanelRole.PLOT_1D, dl.PanelPlan(visible=True, has_data=False)),
    ))
    assert state.panel(dl.PanelRole.RAW_2D).has_data is True
    assert state.panel(dl.PanelRole.PLOT_1D).has_data is False
    assert state.panel(dl.PanelRole.CAKE_2D) is None   # not present -> None
    assert state.results is None


def test_payload_is_source_agnostic():
    # §10 seam 4: the payload carries no provenance field — render must not
    # be able to tell integration from stitch from reload.
    np = pytest.importorskip("numpy")
    field_names = {f.name for f in dataclasses.fields(dl.DisplayPayload)}
    assert field_names == {'generation', 'raw_image', 'cake_image', 'plot'}
    for forbidden in ('source', 'provenance', 'origin', 'producer'):
        assert forbidden not in field_names


def test_extension_panel_role_and_fit_trace_round_trip():
    """A panel role the core never emits (RESIDUAL_1D, reserved for
    fitting) plus a kind='fit' Trace round-trip through a generic,
    render-style dispatch without any change to display_logic core —
    proving §10 seams 1, 2 are actually open (not three hardcoded panels
    / a single data trace)."""
    np = pytest.importorskip("numpy")
    x = np.linspace(0.0, 1.0, 5)
    data = dl.Trace(label="pattern", x=x, y=x, kind="data")
    fit = dl.Trace(label="model", x=x, y=x * 2, kind="fit")
    plot = dl.PlotPayload(axis_x=dl.Axis("Q", unit="A^-1"), traces=(data, fit))

    state = _make_display_state(panels=(
        (dl.PanelRole.PLOT_1D, dl.PanelPlan(visible=True, has_data=True)),
        (dl.PanelRole.RESIDUAL_1D, dl.PanelPlan(visible=True, has_data=True)),
    ))

    # Generic "render contract": iterate panels, dispatch each role to a
    # handler from a registry.  An extension just registers a handler; the
    # core dispatch loop below is unchanged and role-agnostic.
    drawn = {}
    handlers = {
        dl.PanelRole.PLOT_1D: lambda plan: drawn.__setitem__('plot_1d', plan.has_data),
        dl.PanelRole.RESIDUAL_1D: lambda plan: drawn.__setitem__('residual_1d', plan.has_data),
    }
    for role, plan in state.panels:
        handlers[role](plan)

    assert drawn == {'plot_1d': True, 'residual_1d': True}
    # The fit trace survives, distinct from the data trace, on one payload.
    assert [t.kind for t in plot.traces] == ['data', 'fit']
    assert plot.axis_x.unit == "A^-1" and plot.axis_x.log is False


def test_compute_display_state_emits_layout():
    # §10.1: the computed state carries a layout descriptor — arrangement is
    # data, not mode-branching.  Int-2D = raw|cake on top, plot below.
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_2D, selected_ids=(0,), loaded_1d_keys={0},
        loaded_2d_keys={0}, raw_availability={0: dict(has_raw=True)}))
    roles = tuple(tuple(k.role for k in row) for row in state.layout)
    assert roles == (
        (dl.PanelRole.RAW_2D, dl.PanelRole.CAKE_2D),
        (dl.PanelRole.PLOT_1D,),
    )
    # Every key in the layout resolves to a panel plan.
    for row in state.layout:
        for key in row:
            assert state.panel(key) is not None
    # Viewer mode is a single-panel layout (no raw/cake).
    img = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.IMAGE_VIEWER, selected_ids=(0,), loaded_2d_keys={0},
        raw_availability={0: dict(has_raw=True)}))
    assert img.layout == ((dl.PanelKey(dl.PanelRole.RAW_2D),),)


def test_int_1d_is_plot_only_int_2d_has_raw_cake_plot():
    # INT_1D is 1D-only (skip_2d) -> plot-only panels/layout, matching
    # _apply_1d_only_visibility.  INT_2D keeps raw + cake + plot.
    kw = dict(selected_ids=(0,), all_frame_index=[0], loaded_1d_keys={0},
              loaded_2d_keys={0}, raw_availability={0: dict(has_raw=True)})
    one_d = dl.compute_display_state(**_base_state_kwargs(mode=dl.Mode.INT_1D, **kw))
    two_d = dl.compute_display_state(**_base_state_kwargs(mode=dl.Mode.INT_2D, **kw))

    roles_1d = {k.role for k, _ in one_d.panels}
    roles_2d = {k.role for k, _ in two_d.panels}
    assert roles_1d == {dl.PanelRole.PLOT_1D}
    assert roles_2d == {dl.PanelRole.RAW_2D, dl.PanelRole.CAKE_2D, dl.PanelRole.PLOT_1D}
    assert one_d.layout == ((dl.PanelKey(dl.PanelRole.PLOT_1D),),)
    assert one_d.panel(dl.PanelRole.RAW_2D) is None      # no 2D panels in 1D-only
    assert one_d.panel(dl.PanelRole.CAKE_2D) is None


def test_rsm_2x3_layout_with_repeated_roles_round_trips():
    """The RSM mockup — 2×3 grid of 3 reciprocal-space slices over their 3
    projections — round-trips through a generic, render-style dispatch.
    This is the case a fixed 3-field DisplayState could not express: 6
    panels, repeated SLICE_2D / PROJ_1D roles disambiguated by instance id,
    and arbitrary H/K/L axes.  No display_logic core change is needed."""
    np = pytest.importorskip("numpy")

    slices = [dl.PanelKey(dl.PanelRole.SLICE_2D, i) for i in ("HK", "HL", "KL")]
    projs = [dl.PanelKey(dl.PanelRole.PROJ_1D, i) for i in ("H", "K", "L")]
    panels = tuple(
        (k, dl.PanelPlan(visible=True, has_data=True)) for k in slices + projs)
    layout = (tuple(slices), tuple(projs))   # 2 rows × 3 columns
    state = _make_display_state(panels=panels, layout=layout)

    # Arbitrary reciprocal-space axes (H/K/L, r.l.u.) — Axis takes any label.
    ax = dl.Axis("H", unit="r.l.u.")
    assert ax.label == "H" and ax.unit == "r.l.u."

    # Generic "render contract": lay out by the descriptor, dispatch each
    # panel by role.  The dispatch loop is fixed and role-agnostic; an
    # extension only registers handlers.
    handlers = {
        dl.PanelRole.SLICE_2D: lambda key, plan: ("slice", key.instance, plan.has_data),
        dl.PanelRole.PROJ_1D: lambda key, plan: ("proj", key.instance, plan.has_data),
    }
    grid = []
    for row in state.layout:
        grid.append([handlers[key.role](key, state.panel(key)) for key in row])

    assert len(grid) == 2 and all(len(r) == 3 for r in grid)
    assert grid[0] == [("slice", "HK", True), ("slice", "HL", True), ("slice", "KL", True)]
    assert grid[1] == [("proj", "H", True), ("proj", "K", True), ("proj", "L", True)]

    # Repeated roles are disambiguated by instance via exact-key lookup,
    # while a bare-role lookup still returns the first matching panel.
    assert state.panel(slices[0]) is not state.panel(slices[1])
    assert state.panel(dl.PanelRole.SLICE_2D) is state.panel(slices[0])


def test_controller_registry_register_and_lookup():
    # §10 seam 3: open Mode -> controller registry; no controllers wired
    # in core yet, so an unregistered mode resolves to None.
    sentinel = object()
    assert dl.controller_for(dl.Mode.INT_2D) is None
    try:
        dl.register_controller(dl.Mode.INT_2D, sentinel)
        assert dl.controller_for(dl.Mode.INT_2D) is sentinel
    finally:
        dl._CONTROLLER_REGISTRY.pop(dl.Mode.INT_2D, None)  # don't leak across tests


# ── §6/§9 guardrail: display_logic stays pure (no Qt/pyqtgraph/h5py/pyFAI) ──

def test_display_logic_imports_no_heavy_deps():
    """Load display_logic.py *by file path* in a clean subprocess and
    assert it pulled in none of the forbidden heavy modules.

    Loading by path (not via the dotted ``xdart.gui...`` package) is
    deliberate: importing the package runs its ``__init__`` chain, which
    pulls in Qt/pyFAI/h5py regardless of this module.  The guardrail
    (§9) is about *display_logic.py's own* imports — keep it pure so CI
    runs it anywhere, no Qt/pyFAI install needed."""
    forbidden = ('PySide6', 'PySide2', 'PyQt5', 'PyQt6',
                 'pyqtgraph', 'h5py', 'pyFAI', 'fabio')
    code = textwrap.dedent(f"""
        import sys, importlib.util
        spec = importlib.util.spec_from_file_location(
            "display_logic_isolated", {_DISPLAY_LOGIC_PATH!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclasses needs the module registered
        spec.loader.exec_module(mod)
        bad = [m for m in {forbidden!r} if m in sys.modules]
        if bad:
            print(','.join(bad))
            sys.exit(1)
    """)
    proc = subprocess.run([sys.executable, '-c', code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"display_logic pulled in forbidden modules: {proc.stdout.strip()}\n"
        f"{proc.stderr.strip()}"
    )
