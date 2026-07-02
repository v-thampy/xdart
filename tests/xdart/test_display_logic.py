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
from threading import RLock
from types import SimpleNamespace

import pytest

import xdart.gui.tabs.static_scan.display_logic as dl
from xdart.gui.tabs.static_scan.display_controllers import ScanDisplayController

# Absolute path to the module under test, derived from this test's
# location (tests/xdart is two levels below the repo root; the module
# lives under src/) so the purity guard need not
# import the xdart package at all.
_DISPLAY_LOGIC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src", "xdart", "gui", "tabs", "static_scan", "display_logic.py",
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
    # GI prefixes (checked before the generic 'iq')
    assert dl.xye_unit_from_filename('iqip_scan_0001.xye') == 'qip_A^-1'
    assert dl.xye_unit_from_filename('iqoop_scan_0001.xye') == 'qoop_A^-1'
    assert dl.xye_unit_from_filename('iexit_scan_0001.xye') == 'exit_angle_deg'
    # unknown prefix now falls back to Q (XRD 1D is Q by convention)
    assert dl.xye_unit_from_filename('random_scan.xye') == 'q_A^-1'
    # unknown unit -> plain x, no unit symbol (x_axis_for_unit unchanged)
    assert dl.x_axis_for_unit('unknown') == ('x', '')


def test_pretty_unit():
    """Display-layer prettify: raw pyFAI tokens -> symbols; unknown/empty pass
    through unchanged (the stored/headless unit stays canonical)."""
    assert dl.pretty_unit('q_A^-1') == dl._AA_INV       # Å⁻¹
    assert dl.pretty_unit('qip_A^-1') == dl._AA_INV
    assert dl.pretty_unit('qoop_A^-1') == dl._AA_INV
    assert dl.pretty_unit('2th_deg') == dl._DEG          # °
    assert dl.pretty_unit('chi_deg') == dl._DEG
    assert dl.pretty_unit('r_mm') == 'mm'
    # unknown / empty / None pass through unchanged
    assert dl.pretty_unit('counts') == 'counts'
    assert dl.pretty_unit('') == ''
    assert dl.pretty_unit(None) is None


def test_xye_prefix_unit_roundtrip():
    """Writer prefix <-> reader unit must be consistent, and every recovered
    unit must resolve to a real axis label (not plain 'x')."""
    cases = {
        'q_A^-1': 'iq', 'q_nm^-1': 'iq', '2th_deg': 'itth', '2th_rad': 'itth',
        'qip_A^-1': 'iqip', 'qoop_A^-1': 'iqoop',
        'exit_angle_horz_deg': 'iexit',
    }
    for unit, prefix in cases.items():
        assert dl.xye_prefix_for_unit(unit) == prefix
        recovered = dl.xye_unit_from_filename(f'{prefix}_scan_0001.xye')
        # the recovered unit resolves to a labelled axis (never ('x', ''))
        assert dl.x_axis_for_unit(recovered) != ('x', '')


def test_plan_overlay_rebuild_on_unit_change_keeps_all_ids():
    # Overlay + unit change -> REBUILD all previously overlaid ids (not drop to last)
    action, ids = dl.plan_overlay('Overlay', unit_changed=True, has_existing=True,
                                  new_ids=(3,), prev_overlaid_ids=(0, 1, 2))
    assert action is dl.OverlayAction.REBUILD and set(ids) == {0, 1, 2}
    # Single always replaces:
    action2, _ = dl.plan_overlay('Single', unit_changed=False, has_existing=True,
                                 new_ids=(5,), prev_overlaid_ids=(0, 1, 2))
    assert action2 is dl.OverlayAction.REPLACE


def test_overlay_read_failure_preserves_existing_accumulator():
    # Append-only invariant: a failed/partial incremental read of the newest
    # frames must NEVER wipe an existing Overlay/Waterfall accumulator (the
    # cap-store regression).  With an accumulator present -> PRESERVE.
    assert dl.overlay_read_failure_action('Overlay', has_accumulator=True) == 'preserve'
    assert dl.overlay_read_failure_action('Waterfall', has_accumulator=True) == 'preserve'


def test_overlay_read_failure_clears_when_nothing_accumulated():
    # Genuine empty selection (fresh reload, no frames yet) -> CLEAR is correct.
    assert dl.overlay_read_failure_action('Overlay', has_accumulator=False) == 'clear'
    assert dl.overlay_read_failure_action('Waterfall', has_accumulator=False) == 'clear'


def test_overlay_read_failure_clears_for_non_accumulating_methods():
    # Single/Sum/Average rebuild from the current selection; a failed read clears.
    for method in ('Single', 'Sum', 'Average'):
        assert dl.overlay_read_failure_action(method, has_accumulator=True) == 'clear'


def test_accumulate_waterfall_builds_and_appends_monotonically():
    np = pytest.importorskip("numpy")
    x = np.array([0.0, 1.0, 2.0])
    h = dl.accumulate_waterfall(None, reset_key="A", unit="q", x=x,
                                rows=np.array([[1.0, 1.0, 1.0]]), ids=[0], names=["s0"])
    assert h.count == 1 and h.ids == (0,) and h.unit == "q"
    h = dl.accumulate_waterfall(h, reset_key="A", unit="q", x=x,
                                rows=np.array([[2.0, 2.0, 2.0]]), ids=[1], names=["s1"])
    assert h.ids == (0, 1) and h.rows.shape == (2, 3)
    np.testing.assert_array_equal(h.rows[0], [1, 1, 1])
    np.testing.assert_array_equal(h.rows[1], [2, 2, 2])


def test_accumulate_waterfall_partial_read_never_shrinks_or_restacks():
    # Append-only within one reset_key: a failed/partial/re-delivered read can only
    # ADD frames it hasn't captured -- never shrink the stack or re-stack a frame
    # (the collapse/restack class, precluded by construction).
    np = pytest.importorskip("numpy")
    x = np.array([0.0, 1.0, 2.0])
    h = None
    for i in range(5):
        h = dl.accumulate_waterfall(h, reset_key="A", unit="q", x=x,
                                    rows=np.full((1, 3), float(i)), ids=[i], names=[f"s{i}"])
    assert h.count == 5
    h2 = dl.accumulate_waterfall(h, reset_key="A", unit="q", x=x,   # re-deliver an old id
                                 rows=np.full((1, 3), 1.0), ids=[1], names=["s1"])
    assert h2.ids == (0, 1, 2, 3, 4) and h2.rows.shape == (5, 3)


def test_accumulate_waterfall_reset_key_change_resets():
    np = pytest.importorskip("numpy")
    x = np.array([0.0, 1.0, 2.0])
    h = dl.accumulate_waterfall(None, reset_key="scanA", unit="q", x=x,
                                rows=np.ones((1, 3)), ids=[7], names=["s7"])
    h = dl.accumulate_waterfall(h, reset_key="scanB", unit="q", x=x,  # new scan/source
                                rows=np.full((1, 3), 9.0), ids=[0], names=["s0"])
    assert h.reset_key == "scanB" and h.ids == (0,) and h.count == 1
    np.testing.assert_array_equal(h.rows[0], [9, 9, 9])


def test_accumulate_waterfall_selection_growth_does_not_reset():
    # Regression for the per-tick reset bug: live auto-last GROWS the selection each
    # tick (which bumps the display generation), but the accumulator is keyed on the
    # STABLE scan/source reset_key -- so a growing selection APPENDS and frames since
    # evicted past the store cap are RETAINED, not truncated.
    np = pytest.importorskip("numpy")
    x = np.array([0.0, 1.0, 2.0])
    h = None
    # Ticks 0,1: frames 0,1 captured while resident.
    for i in range(2):
        h = dl.accumulate_waterfall(h, reset_key=("scanA", False), unit="q", x=x,
                                    rows=np.full((1, 3), float(i)), ids=[i], names=[f"s{i}"])
    assert h.ids == (0, 1)
    # Tick 2: frame 0 has been EVICTED (no longer resident -> not in the incoming
    # build); only frames 1,2 come in.  The reset_key is unchanged (same scan), so
    # frame 0's captured row is RETAINED.
    h = dl.accumulate_waterfall(h, reset_key=("scanA", False), unit="q", x=x,
                                rows=np.vstack([np.full(3, 1.0), np.full(3, 2.0)]),
                                ids=[1, 2], names=["s1", "s2"])
    assert h.ids == (0, 1, 2) and h.count == 3      # frame 0 NOT lost to eviction
    np.testing.assert_array_equal(h.rows[0], [0, 0, 0])


def test_accumulate_waterfall_unit_toggle_relabels_grid_keeps_rows():
    # A Q<->2theta toggle does NOT change the reset_key: the rows are unit-invariant,
    # so the grid is RELABELLED in place (incoming x in the new unit) and every
    # captured row is kept -- no re-read, no loss.
    np = pytest.importorskip("numpy")
    xq = np.array([1.0, 2.0, 3.0])
    h = None
    for i in range(3):
        h = dl.accumulate_waterfall(h, reset_key="A", unit="q_A^-1", x=xq,
                                    rows=np.full((1, 3), float(i)), ids=[i], names=[f"s{i}"])
    xtth = np.array([5.0, 10.0, 15.0])     # the same samples re-expressed in 2theta
    h2 = dl.accumulate_waterfall(
        h, reset_key="A", unit="2th_deg", x=xtth,
        rows=np.vstack([np.full(3, float(i)) for i in range(3)]),
        ids=[0, 1, 2], names=["s0", "s1", "s2"])
    assert h2.count == 3 and h2.unit == "2th_deg"
    np.testing.assert_array_equal(h2.x, xtth)              # grid relabelled in place
    np.testing.assert_array_equal(h2.rows[0], [0, 0, 0])   # rows unchanged
    np.testing.assert_array_equal(h2.rows[2], [2, 2, 2])


def test_accumulate_waterfall_unit_toggle_with_evicted_frames_keeps_full_stack():
    # Unit toggle where only the resident tail comes in (older frames evicted past
    # the store cap): the accumulator keeps the FULL stack and relabels the grid.
    np = pytest.importorskip("numpy")
    xq = np.array([1.0, 2.0, 3.0])
    h = None
    for i in range(4):
        h = dl.accumulate_waterfall(h, reset_key="A", unit="q_A^-1", x=xq,
                                    rows=np.full((1, 3), float(i)), ids=[i], names=[f"s{i}"])
    xtth = np.array([5.0, 10.0, 15.0])
    h2 = dl.accumulate_waterfall(
        h, reset_key="A", unit="2th_deg", x=xtth,
        rows=np.vstack([np.full(3, 2.0), np.full(3, 3.0)]),
        ids=[2, 3], names=["s2", "s3"])
    assert h2.count == 4                                   # nothing lost
    np.testing.assert_array_equal(h2.x, xtth)


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


@pytest.mark.display_logic
def test_sentinel_mask_uint16_is_opt_in():
    """The uint16 ceiling (65535) is masked only when ``mask_saturation`` is
    True (the 'Mask saturated' toggle, default ON).  With it OFF a saturated
    65535 pixel survives, while non-finite + the uint32 ceiling stay masked."""
    np = pytest.importorskip("numpy")
    arr = np.full(10000, 100.0)
    arr[:2000] = 65535.0                       # 20% at the uint16 ceiling
    arr[2000] = np.inf                         # always invalid
    arr[2001] = 4294967295.0                   # uint32 ceiling, always invalid

    on = dl.sentinel_mask(arr, mask_saturation=True)
    assert np.isnan(on[:2000]).all()           # 65535 masked when ON (default)

    off = dl.sentinel_mask(arr, mask_saturation=False)
    assert not np.isnan(off[:2000]).any()      # 65535 NOT masked when OFF
    assert math.isnan(off[2000])               # non-finite still masked
    assert math.isnan(off[2001])               # uint32 ceiling still masked
    assert off[5000] == 100.0                  # real values untouched

    # default keeps the long-standing behaviour (ON)
    assert np.isnan(dl.sentinel_mask(arr)[:2000]).all()


@pytest.mark.display_logic
def test_integer_saturation_ceiling_from_dtype():
    """The saturation ceiling is learned from the raw integer dtype's iinfo.max,
    with a safe 65535 fallback once the dtype is lost to float."""
    np = pytest.importorskip("numpy")
    assert dl.integer_saturation_ceiling(np.zeros(4, dtype=np.uint16)) == 65535.0
    assert dl.integer_saturation_ceiling(np.zeros(4, dtype=np.uint8)) == 255.0
    assert dl.integer_saturation_ceiling(np.zeros(4, dtype=np.uint32)) == 4294967295.0
    assert dl.integer_saturation_ceiling(np.zeros(4, dtype=np.int16)) == 32767.0
    assert dl.integer_saturation_ceiling(np.zeros(4, dtype=float)) == 65535.0


@pytest.mark.display_logic
def test_sentinel_mask_uses_dtype_ceiling_for_8bit():
    """An 8-bit frame saturates at 255, not 65535 — sentinel_mask derives that
    from the dtype so a uint8 dead block is masked while 65535 is irrelevant."""
    np = pytest.importorskip("numpy")
    arr = np.full(10000, 50, dtype=np.uint8)
    arr[:2000] = 255                            # 20% at the uint8 ceiling
    out = dl.sentinel_mask(arr)                 # ceiling derived from uint8 dtype
    assert np.isnan(out[:2000]).all()           # 255 masked
    assert out[5000] == 50.0
    # OFF leaves the 255 band intact
    assert not np.isnan(dl.sentinel_mask(arr, mask_saturation=False)[:2000]).any()
    # explicit ceiling override (caller already converted to float)
    farr = arr.astype(float)
    assert np.isnan(dl.sentinel_mask(farr, ceiling=255.0)[:2000]).all()
    # without the override a float uint8-valued frame is NOT masked (255 != 65535)
    assert not np.isnan(dl.sentinel_mask(farr)[:2000]).any()


@pytest.mark.display_logic
def test_convert_2d_radial_q_to_2theta_and_back():
    np = pytest.importorskip("numpy")
    lam_A = 1.0
    q = np.array([0.5, 1.0, 2.0])
    # Q -> 2θ (data is q, selection names 2θ)
    tth = dl.convert_2d_radial(q, data_unit="q_A^-1", want_tth=True,
                               want_q=False, wavelength_m=lam_A * 1e-10)
    expected = 2 * np.degrees(np.arcsin(np.clip(q * lam_A / (4 * np.pi), -1, 1)))
    np.testing.assert_allclose(tth, expected)
    # 2θ -> Q (data is 2θ, selection names Q) round-trips
    back = dl.convert_2d_radial(tth, data_unit="2th_deg", want_tth=False,
                                want_q=True, wavelength_m=lam_A * 1e-10)
    np.testing.assert_allclose(back, q, atol=1e-9)
    # No-op when the selection already matches the data, or wavelength unknown.
    same = dl.convert_2d_radial(q, data_unit="q_A^-1", want_tth=False,
                                want_q=True, wavelength_m=lam_A * 1e-10)
    np.testing.assert_allclose(same, q)
    none_wl = dl.convert_2d_radial(q, data_unit="q_A^-1", want_tth=True,
                                   want_q=False, wavelength_m=None)
    np.testing.assert_allclose(none_wl, q)


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


def test_overlay_preserves_1d_panel_when_selected_frame_evicted():
    # Phase-5 cap-store regression: selecting an evicted frame (not resident) in
    # Overlay/Waterfall must NOT clear the 1D overlay when an accumulator already
    # exists — PLOT_1D stays drawable (READY, has_data) so the payload re-emits the
    # accumulator; the 2D cake stays blank (its frame is evicted too).
    for mode in (dl.Mode.INT_1D, dl.Mode.INT_2D):
        for method in ('Overlay', 'Waterfall'):
            state = dl.compute_display_state(**_base_state_kwargs(
                mode=mode, method=method,
                selected_ids=(2,), loaded_1d_keys=set(), loaded_2d_keys=set(),
                prev_overlaid_ids=(0, 1),          # accumulator already exists
            ))
            assert state.load_status is dl.LoadStatus.READY, (mode, method)
            assert state.panel(dl.PanelRole.PLOT_1D).has_data is True, (mode, method)
            cake = state.panel(dl.PanelRole.CAKE_2D)   # INT_1D has no cake panel
            if cake is not None:
                assert cake.has_data is False, (mode, method)


def test_overlay_clears_1d_when_no_accumulator():
    # A genuine empty selection (no accumulator yet) must still clear — the
    # preserve must not spuriously keep stale curves.
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_1D, method='Overlay',
        selected_ids=(2,), loaded_1d_keys=set(), loaded_2d_keys=set(),
        prev_overlaid_ids=(),                  # no accumulator
    ))
    assert state.load_status is not dl.LoadStatus.READY
    assert state.panel(dl.PanelRole.PLOT_1D).has_data is False


def test_non_accumulating_methods_do_not_preserve_on_evicted():
    # Single/Sum/Average rebuild from the selection — an evicted read clears; the
    # overlay preserve must NOT engage even with a stale accumulator present.
    for method in ('Single', 'Sum', 'Average'):
        state = dl.compute_display_state(**_base_state_kwargs(
            mode=dl.Mode.INT_1D, method=method,
            selected_ids=(2,), loaded_1d_keys=set(), loaded_2d_keys=set(),
            prev_overlaid_ids=(0, 1),
        ))
        assert state.panel(dl.PanelRole.PLOT_1D).has_data is False, method


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


def test_raw_panel_prefers_thumbnail_universally():
    # Universal raw-display policy: the Int 2D raw panel is display-only, so EVERY
    # method prefers the (cheap) thumbnail when one exists -- Single/Overlay/
    # Waterfall AND the Sum/Average aggregation.  raw_image rect-scales it to the
    # true detector extent, so the displayed dimensions stay correct.  Thumbnails
    # are pre-baked, so the full-detector mask is NOT applied (apply_mask False).
    raw_avail = {i: dict(has_raw=True, has_thumbnail=True) for i in (0, 1, 2)}
    for method in ("Single", "Overlay", "Waterfall", "Sum", "Average"):
        state = dl.compute_display_state(**_base_state_kwargs(
            mode=dl.Mode.INT_2D, method=method,
            selected_ids=(0, 1, 2), all_frame_index=[0, 1, 2],
            loaded_1d_keys={0, 1, 2}, loaded_2d_keys={0, 1, 2},
            raw_availability=raw_avail,
        ))
        panel = state.panel(dl.PanelRole.RAW_2D)
        assert panel.source is dl.RawSource.THUMBNAIL, method
        assert panel.apply_mask is False, method


def test_raw_panel_falls_back_to_full_res_only_when_no_thumbnail():
    # Full-res RAW is the FALLBACK, used only when no thumbnail exists (e.g. a
    # no-.nxs run) -- then the full-detector mask DOES apply (apply_mask True).
    raw_avail = {i: dict(has_raw=True, has_thumbnail=False) for i in (0, 1, 2)}
    for method in ("Single", "Overlay", "Sum"):
        state = dl.compute_display_state(**_base_state_kwargs(
            mode=dl.Mode.INT_2D, method=method,
            selected_ids=(0, 1, 2), all_frame_index=[0, 1, 2],
            loaded_1d_keys={0, 1, 2}, loaded_2d_keys={0, 1, 2},
            raw_availability=raw_avail,
        ))
        panel = state.panel(dl.PanelRole.RAW_2D)
        assert panel.source is dl.RawSource.RAW, method
        assert panel.apply_mask is True, method


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


def test_nexus_viewer_routes_previewable_rows_to_plot_or_image():
    empty = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.NEXUS_VIEWER,
        selected_ids=(4,),
        all_frame_index=[],
        loaded_1d_keys=set(),
        loaded_2d_keys=set(),
    ))
    assert empty.load_status is dl.LoadStatus.EMPTY
    assert not empty.panel(dl.PanelRole.PLOT_1D).has_data
    assert not empty.panel(dl.PanelRole.RAW_2D).has_data

    one_d = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.NEXUS_VIEWER,
        selected_ids=(4,),
        all_frame_index=[],
        loaded_1d_keys={4},
        loaded_2d_keys=set(),
    ))
    assert one_d.load_status is dl.LoadStatus.READY
    assert one_d.render_ids == (4,)
    assert one_d.panel(dl.PanelRole.PLOT_1D).has_data
    assert not one_d.panel(dl.PanelRole.RAW_2D).has_data

    two_d = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.NEXUS_VIEWER,
        selected_ids=(5,),
        all_frame_index=[],
        loaded_1d_keys=set(),
        loaded_2d_keys={5},
        raw_availability={5: dict(has_raw=True, has_thumbnail=False)},
    ))
    assert two_d.load_status is dl.LoadStatus.READY
    assert two_d.render_ids == (5,)
    assert two_d.panel(dl.PanelRole.RAW_2D).has_data
    assert not two_d.panel(dl.PanelRole.PLOT_1D).has_data


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
    # §10 seam 3: open Mode -> controller registry.  An unregistered mode
    # resolves to None; register_controller overrides by mode.
    sentinel = object()
    fresh_mode = dl.Mode.INT_1D
    prev = dl.controller_for(fresh_mode)
    try:
        dl.register_controller(fresh_mode, sentinel)
        assert dl.controller_for(fresh_mode) is sentinel
    finally:
        if prev is not None:
            dl.register_controller(fresh_mode, prev)   # restore the real one
        else:
            dl._CONTROLLER_REGISTRY.pop(fresh_mode, None)


class _PlotMethod:
    def currentText(self):
        return "Single"


class _IndexThatMustNotIterate:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __iter__(self):
        raise AssertionError("single-frame render should not iterate full scan index")


class _CountingIndex:
    def __init__(self, labels):
        self.labels = tuple(labels)
        self.iterations = 0

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        self.iterations += 1
        return iter(self.labels)


def _controller_widget(index, *, frame_ids, data_1d=None, data_2d=None):
    return SimpleNamespace(
        scan=SimpleNamespace(
            scan_lock=RLock(),
            frames=SimpleNamespace(index=index),
            gi=False,
        ),
        frame_ids=list(frame_ids),
        publication_store=None,
        data_lock=RLock(),
        data_1d={} if data_1d is None else data_1d,
        data_2d={} if data_2d is None else data_2d,
        ui=SimpleNamespace(plotMethod=_PlotMethod()),
        overlaid_idxs=(),
        display_generation=0,
    )


def test_scan_controller_does_not_copy_full_index_for_single_frame_render():
    """I4: Auto-last/Single updates only need the scan length, not every label.

    A long scan used to copy/convert the whole frame index on every render
    tick.  This fake index fails if iterated; the single-frame path should
    still resolve readiness from the selected label.
    """
    widget = _controller_widget(
        _IndexThatMustNotIterate(10_000),
        frame_ids=[5],
        data_1d={5: object()},
    )

    state = ScanDisplayController().compute_state(widget, dl.Mode.INT_1D)

    assert state.overall is False
    assert state.render_ids == (5,)
    assert state.load_status is dl.LoadStatus.READY


def test_scan_controller_materializes_full_index_for_overall_render():
    index = _CountingIndex([0, 1, 2])
    widget = _controller_widget(
        index,
        frame_ids=[0, 1, 2],
        data_1d={0: object(), 1: object(), 2: object()},
    )

    state = ScanDisplayController().compute_state(widget, dl.Mode.INT_1D)

    assert index.iterations == 1
    assert state.overall is True
    assert state.render_ids == (0, 1, 2)


# ── Stage 3: build_payload + render_plan (the testable render core) ───

class _FakeStore:
    """Resolves every panel to a sentinel array/trace so build_payload's
    gating (has_data + READY) is observable in a test."""
    def raw_image(self, state):
        return "RAW"
    def cake_image(self, state):
        return "CAKE"
    def plot_payload(self, state):
        return dl.PlotPayload(axis_x=dl.Axis("Q"), traces=())


def test_build_payload_gates_on_has_data_and_stamps_generation():
    # READY INT_2D with raw+cake+plot all present -> all resolved; generation
    # stamped from the state.
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_2D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0},
        raw_availability={0: dict(has_raw=True)}, generation=5))
    p = dl.build_payload(state, _FakeStore())
    assert p.generation == 5
    assert p.raw_image == "RAW" and p.cake_image == "CAKE"
    assert isinstance(p.plot, dl.PlotPayload)

    # A panel with has_data=False is not resolved (renders blank).
    state_nodraw = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_2D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0},
        raw_availability={0: dict(has_raw=False, has_thumbnail=False)}))
    p2 = dl.build_payload(state_nodraw, _FakeStore())
    assert p2.raw_image is None              # RAW_2D has_data False -> blank
    assert p2.cake_image == "CAKE"           # cake still has data


def test_build_payload_blank_when_not_ready_or_no_store():
    empty = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_2D, selected_ids=(), loaded_1d_keys=set(),
        loaded_2d_keys=set()))
    p = dl.build_payload(empty, _FakeStore())
    assert (p.raw_image, p.cake_image, p.plot) == (None, None, None)
    # No store -> nothing resolved (renderer delegates), generation still set.
    ready = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_2D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0},
        raw_availability={0: dict(has_raw=True)}, generation=9))
    p2 = dl.build_payload(ready)             # store=None
    assert p2.generation == 9
    assert (p2.raw_image, p2.cake_image, p2.plot) == (None, None, None)


def test_render_plan_draws_present_clears_absent():
    # INT_1D (plot-only): draw PLOT_1D, clear the two 2D panels that aren't
    # in this state (kills stale panels from a previous mode).
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_1D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0}))
    plan = dl.render_plan(state, dl.build_payload(state))
    assert plan.drop is False
    assert plan.draw == (dl.PanelRole.PLOT_1D,)
    assert set(plan.clear) == {dl.PanelRole.RAW_2D, dl.PanelRole.CAKE_2D}


def test_render_roles_follow_layout_but_keep_legacy_cleanup():
    # The descriptor order is primary, but absent legacy panels are still
    # managed so stale content from a prior mode is blanked.
    state = _make_display_state(
        panels=(
            (dl.PanelKey(dl.PanelRole.RAW_2D), dl.PanelPlan(visible=True, has_data=True)),
        ),
        layout=((dl.PanelKey(dl.PanelRole.RAW_2D),),),
    )
    assert dl.render_roles_for_state(state) == (
        dl.PanelRole.RAW_2D,
        dl.PanelRole.PLOT_1D,
        dl.PanelRole.CAKE_2D,
    )


def test_render_plan_includes_extension_roles_from_layout():
    # RSM/Stitch/Fit roles are not tied to the legacy three-panel tuple: the
    # render decision sees roles from the layout descriptor.  The Qt widget
    # still needs concrete delegates before those roles become visible panels.
    slice_key = dl.PanelKey(dl.PanelRole.SLICE_2D, "HK")
    proj_key = dl.PanelKey(dl.PanelRole.PROJ_1D, "H")
    state = _make_display_state(
        panels=(
            (slice_key, dl.PanelPlan(visible=True, has_data=True)),
            (proj_key, dl.PanelPlan(visible=True, has_data=True)),
        ),
        layout=((slice_key,), (proj_key,)),
    )
    plan = dl.render_plan(state, dl.build_payload(state))
    assert plan.draw[:2] == (dl.PanelRole.SLICE_2D, dl.PanelRole.PROJ_1D)
    assert set(plan.clear) == {dl.PanelRole.PLOT_1D, dl.PanelRole.RAW_2D, dl.PanelRole.CAKE_2D}


def test_render_plan_empty_and_error_clear_everything():
    for kw in (dict(selected_ids=(), loaded_1d_keys=set()),
               dict(raw_availability={'__error__': 'boom'})):
        state = dl.compute_display_state(**_base_state_kwargs(mode=dl.Mode.INT_2D, **kw))
        plan = dl.render_plan(state, dl.build_payload(state))
        assert plan.draw == ()
        assert set(plan.clear) == {dl.PanelRole.RAW_2D, dl.PanelRole.CAKE_2D, dl.PanelRole.PLOT_1D}
    assert plan.error_message == 'boom'   # last loop (ERROR) surfaces the message


def test_render_plan_drops_stale_generation_payload():
    # §8: a payload whose generation no longer matches the state is dropped
    # (a worker result computed before a mode switch must never render).
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_2D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0}, generation=7))
    stale = dl.DisplayPayload(generation=6, raw_image=None, cake_image=None, plot=None)
    plan = dl.render_plan(state, stale)
    assert plan.drop is True
    fresh = dl.build_payload(state)          # generation 7
    assert dl.render_plan(state, fresh).drop is False


# ── §6/§9 guardrail: display_logic stays pure (no Qt/pyqtgraph/h5py/pyFAI) ──

def test_display_logic_imports_no_heavy_deps():
    """Load display_logic.py *by file path* in a clean subprocess and
    assert it pulled in none of the forbidden heavy modules.

    Loading by path (not via the dotted ``xdart.gui...`` package) is
    deliberate: importing the package runs its ``__init__`` chain, which
    pulls in Qt/pyFAI/h5py regardless of this module.  The guardrail
    (§9) is about *display_logic.py's own* imports — keep it pure so CI
    runs it anywhere, no Qt/pyFAI install needed.

    ``xrd_tools.core`` is ALLOWED (6d): it is the Qt-free contracts
    surface and is import-light by design (its h5py codec re-exports are
    lazy) — this test still asserts none of the forbidden modules get
    pulled through it."""
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


# ── #69 / WS-X2: PanelKey-level render planning (additive, behavior-preserving) ──

def test_render_keys_for_state_keeps_repeated_role_instances():
    slices = [dl.PanelKey(dl.PanelRole.SLICE_2D, i) for i in ("HK", "HL", "KL")]
    projs = [dl.PanelKey(dl.PanelRole.PROJ_1D, i) for i in ("H", "K", "L")]
    panels = tuple((k, dl.PanelPlan(visible=True, has_data=True))
                   for k in (*slices, *projs))
    state = _make_display_state(panels=panels, layout=(tuple(slices), tuple(projs)))
    keys = dl.render_keys_for_state(state)
    # every repeated instance survives (no role collapse) ...
    for k in (*slices, *projs):
        assert k in keys
    # ... while the role version still collapses to 2 roles
    assert dl.render_roles_for_state(state)[:2] == (
        dl.PanelRole.SLICE_2D, dl.PanelRole.PROJ_1D,
    )
    # absent legacy roles are appended as singleton cleanup keys
    assert dl.PanelKey(dl.PanelRole.PLOT_1D) in keys


def test_render_plan_draw_keys_mirror_draw_for_singleton_view():
    state = dl.compute_display_state(**_base_state_kwargs(
        mode=dl.Mode.INT_1D, selected_ids=(0,), all_frame_index=[0],
        loaded_1d_keys={0}, loaded_2d_keys={0}))
    plan = dl.render_plan(state, dl.build_payload(state))
    # behavior-preserving: keys are 1:1 by role with the role-level draw/clear
    assert plan.draw_keys == (dl.PanelKey(dl.PanelRole.PLOT_1D),)
    assert set(plan.clear_keys) == {
        dl.PanelKey(dl.PanelRole.RAW_2D), dl.PanelKey(dl.PanelRole.CAKE_2D),
    }
    assert tuple(k.role for k in plan.draw_keys) == plan.draw
    assert tuple(k.role for k in plan.clear_keys) == plan.clear


def test_render_plan_draw_keys_carry_all_rsm_instances():
    slices = [dl.PanelKey(dl.PanelRole.SLICE_2D, i) for i in ("HK", "HL", "KL")]
    projs = [dl.PanelKey(dl.PanelRole.PROJ_1D, i) for i in ("H", "K", "L")]
    panels = tuple((k, dl.PanelPlan(visible=True, has_data=True))
                   for k in (*slices, *projs))
    state = _make_display_state(panels=panels, layout=(tuple(slices), tuple(projs)))
    plan = dl.render_plan(state, dl.build_payload(state))
    for k in (*slices, *projs):
        assert k in plan.draw_keys           # every instance drawn (key-level)
    # role-level draw still collapses (what the renderer consumes today)
    assert plan.draw[:2] == (dl.PanelRole.SLICE_2D, dl.PanelRole.PROJ_1D)


def test_nanmean_slice_guards_empty_and_all_nan():
    # codex P2: the 2D->1D slice projection must not emit "Mean of empty slice"
    # on a 0-bin slice or an all-NaN column (GI empty bins).
    import warnings as _w
    import numpy as _np
    a = _np.array([[1.0, 2.0], [3.0, 4.0]])
    _np.testing.assert_allclose(dl.nanmean_slice(a, 0), [2.0, 3.0])   # normal mean
    assert dl.nanmean_slice(a[0:0, :], 0) is None                     # 0-bin -> None
    # all-NaN column -> NaN (gap), NOT a warning
    nan_col = _np.array([[_np.nan, 1.0], [_np.nan, 3.0]])
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)     # a "Mean of empty slice" would raise
        out = dl.nanmean_slice(nan_col, 0)
    assert _np.isnan(out[0]) and out[1] == 2.0


# ── detector gap-mask helpers (raw-panel payload unification) ──────────────────
# Detector module gaps are 0-valued pixels (NOT sentinels); they become NaN only
# via the detector mask.  These pure helpers let the legacy update_image path and
# the publication raw_image builder mask gaps identically — full-res applies the
# flat indices directly, thumbnails remap them through the downsample ratio.

def test_combine_flat_masks_unions_dedupes_and_bounds():
    import numpy as np
    a = np.array([5, 7, 7], dtype=int)
    b2d = np.zeros((4, 4), dtype=bool)
    b2d[0, 1] = True                                       # flat index 1
    out = dl.combine_flat_masks(a, b2d, None, size=16)
    assert out.tolist() == [1, 5, 7]                       # unioned, deduped, sorted
    assert dl.combine_flat_masks(np.array([3, 99]), size=16).tolist() == [3]  # bound
    assert dl.combine_flat_masks(None, None) is None
    assert dl.combine_flat_masks(np.array([], dtype=int)) is None


def test_nan_gaps_in_thumbnail_maps_full_res_indices_to_thumbnail():
    import numpy as np
    # full-res rows 48-51 of a 100x100 detector -> thumbnail rows 24-25 of 50x50
    rows = np.arange(100 * 100) // 100
    gap = np.flatnonzero((rows >= 48) & (rows <= 51))
    data = np.ones((50, 50), dtype=float)
    out = dl.nan_gaps_in_thumbnail(data, gap, (100, 100))
    assert out is data                                     # mutates in place
    assert np.isnan(data[24:26, :]).all()                  # gap band -> NaN
    assert not np.isnan(data[:24, :]).any()                # nothing else touched
    assert not np.isnan(data[26:, :]).any()


def test_nan_gaps_in_thumbnail_noops_without_shape_or_indices():
    import numpy as np
    # No full_shape -> cannot map -> never apply flat indices directly (that would
    # corrupt unrelated thumbnail pixels).
    d1 = np.ones((50, 50), dtype=float)
    assert not np.isnan(dl.nan_gaps_in_thumbnail(d1, np.array([4805]), None)).any()
    d2 = np.ones((50, 50), dtype=float)
    assert not np.isnan(dl.nan_gaps_in_thumbnail(d2, None, (100, 100))).any()


def test_image_and_plot_payload_carry_new_fields_and_freeze():
    import numpy as np
    import dataclasses
    import pytest as _pt
    ip = dl.ImagePayload(image=np.ones((2, 2)))
    assert ip.gap_mask_indices is None and ip.raw_full_shape is None
    ip2 = dl.ImagePayload(image=np.ones((2, 2)),
                          gap_mask_indices=np.array([1]), raw_full_shape=(4, 4))
    assert ip2.raw_full_shape == (4, 4)
    pp = dl.PlotPayload(axis_x=dl.Axis("x", ""))
    assert pp.overlaid_ids is None and pp.plot_history is None
    with _pt.raises(dataclasses.FrozenInstanceError):
        ip.raw_full_shape = (1, 1)                          # frozen invariant preserved


def test_stitch_plot_payload_builds_one_data_trace():
    """stitch_plot_payload turns an IntegrationResult1D into a PlotPayload with a
    single data trace + a unit-labelled x-axis; empty/None -> None."""
    import numpy as np
    from xrd_tools.core.containers import IntegrationResult1D
    r = IntegrationResult1D(radial=np.linspace(0.5, 5, 50),
                            intensity=np.arange(50.0), unit="q_A^-1")
    pp = dl.stitch_plot_payload(r)
    assert pp is not None and len(pp.traces) == 1
    assert pp.axis_x.unit == "q_A^-1" and pp.axis_x.values.size == 50
    assert pp.traces[0].x.size == 50 and pp.traces[0].kind == "data"
    assert dl.stitch_plot_payload(None) is None
    assert dl.stitch_plot_payload(
        IntegrationResult1D(radial=np.array([]), intensity=np.array([]))) is None


def test_stitch_image_payload_transposes_for_display():
    """stitch_image_payload stores intensity.T (rows=y=azimuthal, cols=x=radial)
    so the image-draw delegate's own transpose yields radial-on-x."""
    import numpy as np
    from xrd_tools.core.containers import IntegrationResult2D
    R, A = 7, 4
    inten = np.arange(R * A, dtype=float).reshape(R, A)     # (radial, azimuthal)
    r = IntegrationResult2D(radial=np.linspace(0, 5, R),
                            azimuthal=np.linspace(-90, 90, A),
                            intensity=inten, unit="q_A^-1", azimuthal_unit="chi_deg")
    ip = dl.stitch_image_payload(r)
    assert ip is not None
    assert ip.image.shape == (A, R)
    assert np.array_equal(ip.image, inten.T)
    assert ip.axis_x.values.size == R and ip.axis_y.values.size == A
    assert dl.stitch_image_payload(None) is None


def test_stitch_display_state_1d_draws_plot_clears_2d():
    """STITCH_1D with a result is READY, lays out PLOT_1D, and render_plan draws
    the plot while clearing the raw + cake panels (the merge has no per-frame raw
    or cake)."""
    s = dl.stitch_display_state(dl.Mode.STITCH_1D, 9, has_1d=True, has_2d=False)
    assert s.mode is dl.Mode.STITCH_1D
    assert s.load_status is dl.LoadStatus.READY
    assert s.overall is True and s.generation == 9
    assert s.layout == ((dl.PanelKey(dl.PanelRole.PLOT_1D),),)
    assert s.panel(dl.PanelRole.PLOT_1D).has_data is True
    plan = dl.render_plan(s, None)
    assert dl.PanelRole.PLOT_1D in plan.draw
    assert dl.PanelRole.RAW_2D in plan.clear
    assert dl.PanelRole.CAKE_2D in plan.clear


def test_stitch_display_state_2d_draws_cake_clears_others():
    """STITCH_2D lays out CAKE_2D and clears raw + the 1D plot."""
    s = dl.stitch_display_state(dl.Mode.STITCH_2D, 3, has_1d=False, has_2d=True)
    assert s.mode is dl.Mode.STITCH_2D and s.load_status is dl.LoadStatus.READY
    assert s.layout == ((dl.PanelKey(dl.PanelRole.CAKE_2D),),)
    plan = dl.render_plan(s, None)
    assert dl.PanelRole.CAKE_2D in plan.draw
    assert dl.PanelRole.RAW_2D in plan.clear
    assert dl.PanelRole.PLOT_1D in plan.clear


def test_stitch_display_state_missing_result_is_empty():
    """A Stitch mode whose matching result does NOT exist is EMPTY — render_plan
    clears every panel (an explicit blank, never stale per-frame content)."""
    s = dl.stitch_display_state(dl.Mode.STITCH_1D, 1, has_1d=False, has_2d=True)
    assert s.load_status is dl.LoadStatus.EMPTY
    assert s.panel(dl.PanelRole.PLOT_1D).has_data is False
    plan = dl.render_plan(s, None)
    assert plan.draw == ()
    assert dl.PanelRole.PLOT_1D in plan.clear


def test_stitch_modes_have_panel_layout_geometry():
    """_apply_layout indexes PANEL_LAYOUT[mode] directly — both stitch modes must
    have geometry entries or the widget KeyErrors entering a stitch view."""
    assert dl.Mode.STITCH_1D in dl.PANEL_LAYOUT
    assert dl.Mode.STITCH_2D in dl.PANEL_LAYOUT
    # STITCH_1D is plot-only (2D pane collapsed); STITCH_2D collapses the 1D plot.
    assert dl.PANEL_LAYOUT[dl.Mode.STITCH_1D].twoDWindow_h == (0, 0)
    assert dl.PANEL_LAYOUT[dl.Mode.STITCH_2D].plotWindow_h == (0, 0)
