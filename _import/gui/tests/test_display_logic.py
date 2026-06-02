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
    # IMAGE_VIEWER with neither raw nor thumbnail -> raw_panel.has_data False.
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.IMAGE_VIEWER,
        selected_ids=(0,),
        loaded_2d_keys=set(),
        raw_availability={0: dict(has_raw=False, has_thumbnail=False)},
    ))
    assert state.raw_panel.has_data is False
    assert state.load_status in (dl.LoadStatus.EMPTY, dl.LoadStatus.ERROR)


def test_error_load_yields_error_status_not_partial():
    # Invariant: a failed load -> LoadStatus.ERROR with an error_message,
    # never a half-populated display.
    state = dl.compute_display_state(**_base_state_kwargs(
        raw_availability={'__error__': 'boom'},
    ))
    assert state.load_status is dl.LoadStatus.ERROR
    assert state.error_message
    assert state.raw_panel.has_data is False
    assert state.cake_panel.has_data is False
    assert state.plot_panel.has_data is False


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
    assert state.raw_panel.has_data is True


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
