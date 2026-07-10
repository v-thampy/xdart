# -*- coding: utf-8 -*-
"""F1/F4 (unit-flip regression, corrected forms) + the V3 structural rewire.

F1: a run-scoped display wavelength, stamped from the first frame-backed row's
integrator and consulted for hydrated rows (raw_ref=None) that lack a frame
mid-run -- so an Overlay append batch never mixes units.  Drives the real
DisplayDataMixin._get_wavelength / _stamp_run_wavelength and the real
displayFrameWidget.set_processing_active (no fakes on the seam).

F4: the Share-Axis silent plotUnit switch mirrors _last_plot_unit (the stale-combo
fix).  Under V3 the mirror is KEPT (the legacy update_plot path still consumes
it) and the follow-up render exists in a CHANGE-GATED form only: scheduled when
the silent re-point actually moved the combo, so it terminates (unlike the
unconditional singleShot the F4 ledger note rejected).

V3 (Stage 6): _current_image_axis_key is payload-first — the RENDERED cake
identity stashed by _draw_image_payload wins over any combo read; the
combo/GI-args derivation survives only as the documented GI/no-cake
bootstrap fallback.  The full Share-Axis identity sequences live in
test_ov_invariant_harness.py.
"""
from __future__ import annotations

from types import MethodType, SimpleNamespace

from xdart.gui.tabs.static_scan.display_data import DisplayDataMixin
from xdart.gui.tabs.static_scan.display_frame_widget import displayFrameWidget


# ── F1: run-scoped wavelength ─────────────────────────────────────────────────

def _wl_host(*, run_writing=False):
    host = SimpleNamespace(
        scan=SimpleNamespace(
            data_file=None, mg_args=None, _persisted_wavelength_m=None),
        _run_writing=run_writing,
        _run_wavelength_m=None,
        _wavelength_cache_key=None,
        _wavelength_cache_value=None,
    )
    for name in ("_get_wavelength", "_clear_wavelength_cache"):
        setattr(host, name, MethodType(getattr(DisplayDataMixin, name), host))
    return host


def _frame(wl):
    return SimpleNamespace(integrator=SimpleNamespace(wavelength=wl), poni=None)


def test_run_wavelength_stamped_from_frame_and_used_for_hydrated_rows():
    host = _wl_host(run_writing=True)
    # A frame-backed row resolves from its integrator AND stamps the run value.
    assert host._get_wavelength(_frame(1.5e-10)) == 1.5e-10
    assert host._run_wavelength_m == 1.5e-10
    # A hydrated row (no frame) in the same run now gets that wavelength instead
    # of None -- the mixed-unit fix.
    assert host._get_wavelength(None) == 1.5e-10


def test_sentinel_wavelength_is_not_stamped():
    host = _wl_host(run_writing=True)
    # 1e-10 m is the historical constructor sentinel: source #1 still returns it
    # (unchanged), but it must NOT be stamped for hydrated rows.
    host._get_wavelength(_frame(1e-10))
    assert host._run_wavelength_m is None
    # So a hydrated row gets no false conversion wavelength (stays None).
    assert host._get_wavelength(None) is None


def test_run_wavelength_only_consulted_while_writing():
    host = _wl_host(run_writing=True)
    host._get_wavelength(_frame(1.5e-10))
    host._run_writing = False                     # run ended
    # Outside a run the run-scoped value is not consulted (persisted/HDF5 own it).
    assert host._get_wavelength(None) is None


def test_run_wavelength_reset_at_run_boundary():
    host = _wl_host(run_writing=True)
    host._get_wavelength(_frame(1.5e-10))
    assert host._run_wavelength_m == 1.5e-10
    # The real set_processing_active resets it at the boundary.
    host._processing_active = True
    host._aggregate_live_scan = None
    host._wf_last_draw_t = 0.0
    MethodType(displayFrameWidget.set_processing_active, host)(False)
    assert host._run_wavelength_m is None
    assert host._run_writing is False


# ── F4: Share-Axis mirrors _last_plot_unit ────────────────────────────────────

def test_share_axis_silent_switch_mirrors_last_plot_unit(monkeypatch):
    # _set_share_link is a side-effect collaborator (touches plot link state);
    # stub it so the test isolates the _last_plot_unit mirror (the F4 seam).
    monkeypatch.setattr(displayFrameWidget, "_set_share_link", lambda self, on: None)
    host = SimpleNamespace(
        _last_plot_unit=-1,
        ui=SimpleNamespace(
            shareAxis=SimpleNamespace(
                isChecked=lambda: True, setEnabled=lambda *a: None,
                setChecked=lambda *a: None),
            plotUnit=SimpleNamespace(
                currentIndex=lambda: 3, setEnabled=lambda *a: None),
        ),
    )
    host._share_axis_plot_index = lambda: 3          # can_share, target_idx=3
    host._set_plot_unit_index_silently = lambda idx: None
    host._apply_share_axis_state = MethodType(
        displayFrameWidget._apply_share_axis_state, host)

    assert host._apply_share_axis_state() is True
    assert host._last_plot_unit == 3                 # F4: mirrored (was -1, stale)


# ── V3: change-gated follow-up render + payload-first cake identity ──────────

def _share_host(monkeypatch, *, current, target):
    """Duck host driving the REAL _apply_share_axis_state with a mutable
    plotUnit whose silent re-point actually moves currentIndex."""
    monkeypatch.setattr(
        displayFrameWidget, "_set_share_link", lambda self, on: None)
    state = {"idx": current}
    host = SimpleNamespace(
        _last_plot_unit=-1,
        update=lambda: None,                         # follow-up target
        ui=SimpleNamespace(
            shareAxis=SimpleNamespace(
                isChecked=lambda: True, setEnabled=lambda *a: None,
                setChecked=lambda *a: None),
            plotUnit=SimpleNamespace(
                currentIndex=lambda: state["idx"],
                setEnabled=lambda *a: None),
        ),
    )
    host._share_axis_plot_index = lambda: target
    host._set_plot_unit_index_silently = (
        lambda idx: state.__setitem__("idx", idx))
    host._apply_share_axis_state = MethodType(
        displayFrameWidget._apply_share_axis_state, host)
    return host


def test_share_axis_followup_render_is_change_gated(monkeypatch):
    # The silent re-point MOVED the combo → exactly one follow-up render is
    # scheduled; a second apply (combo already on target) schedules nothing —
    # the gate that makes the V3 follow-up terminate instead of cascading.
    shots = []
    from pyqtgraph import Qt as _Qt
    monkeypatch.setattr(
        _Qt.QtCore.QTimer, "singleShot",
        staticmethod(lambda ms, fn: shots.append((ms, fn))))

    host = _share_host(monkeypatch, current=0, target=1)
    assert host._apply_share_axis_state() is True
    assert len(shots) == 1                           # one follow-up scheduled
    assert host._last_plot_unit == 1                 # F4 mirror moved with it

    assert host._apply_share_axis_state() is True    # combo already on target
    assert len(shots) == 1                           # change-gated: no repeat


def test_share_axis_no_followup_when_combo_already_matches(monkeypatch):
    shots = []
    from pyqtgraph import Qt as _Qt
    monkeypatch.setattr(
        _Qt.QtCore.QTimer, "singleShot",
        staticmethod(lambda ms, fn: shots.append((ms, fn))))

    host = _share_host(monkeypatch, current=1, target=1)
    assert host._apply_share_axis_state() is True
    assert shots == []                               # nothing to reconverge


def test_current_image_axis_key_is_payload_first():
    # The stashed RENDERED cake identity wins over any combo state — the
    # F-B structural rule.  The combo says 2θ (index 1); the panel shows Q.
    host = SimpleNamespace(
        _cake_rendered_axis_key="q_A^-1",
        scan=SimpleNamespace(gi=False),
        ui=SimpleNamespace(imageUnit=SimpleNamespace(currentIndex=lambda: 1)),
    )
    host._current_image_axis_key = MethodType(
        displayFrameWidget._current_image_axis_key, host)
    assert host._current_image_axis_key() == "q_A^-1"

    # Bootstrap fallback (documented): no rendered cake yet → the combo
    # INTENT derivation keeps the share gate alive.
    host._cake_rendered_axis_key = None
    assert host._current_image_axis_key() == "2th_deg"
