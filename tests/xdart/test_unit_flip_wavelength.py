# -*- coding: utf-8 -*-
"""F1/F4 (unit-flip regression, corrected forms).

F1: a run-scoped display wavelength, stamped from the first frame-backed row's
integrator and consulted for hydrated rows (raw_ref=None) that lack a frame
mid-run -- so an Overlay append batch never mixes units.  Drives the real
DisplayDataMixin._get_wavelength / _stamp_run_wavelength and the real
displayFrameWidget.set_processing_active (no fakes on the seam).

F4: the Share-Axis silent plotUnit switch mirrors _last_plot_unit (the stale-combo
fix), with NO follow-up render.
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
