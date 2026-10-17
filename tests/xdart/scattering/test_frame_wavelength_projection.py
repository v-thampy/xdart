"""An ordinary Run's admitted frame wavelength owns its display conversions."""
from dataclasses import replace
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_predicted_ring_overlays import run_page, select
from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xdart.gui.tabs.scattering.scientific_axes import heavy_projection, trace_projection
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize('mode', ('Int 1D', 'Int 2D'))
def test_run_frame_wavelength_drives_traces_cake_and_joint_energy_selection(qapp, tmp_path, mode):
    with run_page(qapp, tmp_path, energies=(16, 17), mode=mode) as (page, view):
        controller = page._context_controller
        keys = controller.frame_keys
        payloads = tuple(select(qapp, page, key) for key in keys)
        expected = tuple(energy_eV_to_wavelength_m(value) for value in (16000., 17000.))
        # Behavioral first failure: the second admitted frame must not use the
        # artifact's first-frame wavelength. No source or conversion seam is fake.
        np.testing.assert_allclose(tuple(payload.wavelength_m for payload in payloads),
                                   expected, rtol=1e-14, atol=0.)
        for payload, wavelength in zip(payloads, expected):
            trace = trace_projection(payload, requested_axis='2theta', allow_cake=False,
                                     slice_enabled=False, slice_center=0., slice_width=1.)
            assert trace is not None and trace.wavelength_m == wavelength
            native_q = payload.view.axis_1d.values
            expected_theta = np.rad2deg(2.*np.arcsin(native_q*wavelength*1e10/(4.*np.pi)))
            np.testing.assert_allclose(trace.axis.values, expected_theta, rtol=1e-14, atol=1e-12)
            np.testing.assert_array_equal(trace.intensity, payload.view.intensity_1d)
            if mode == 'Int 2D':
                heavy = heavy_projection(payload, requested_axis='2theta')
                assert heavy is not None and heavy.cake_x is not None
                native_cake_q = payload.view.axis_2d_x.values
                expected_cake_theta = np.rad2deg(2.*np.arcsin(
                    native_cake_q*wavelength*1e10/(4.*np.pi)))
                np.testing.assert_allclose(heavy.cake_x.values, expected_cake_theta,
                                           rtol=1e-14, atol=1e-12)
            # File-derived views carry no transient PONI; their existing scalar
            # stays authoritative. An unusable transient scalar also falls back.
            fallback = replace(payload, view=replace(payload.view, geometry=None), wavelength_m=1e-10)
            assert fallback.wavelength_m == 1e-10
            for invalid in (None, 0., -1., float('nan')):
                geometry = replace(payload.view.geometry,
                                   poni={**payload.view.geometry.poni, 'wavelength': invalid})
                fallback = replace(payload, view=replace(payload.view, geometry=geometry), wavelength_m=1e-10)
                assert fallback.wavelength_m == 1e-10
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, 'Overlay'))
        assert controller.select_navigation(keys[-1], tuple(keys))
        deadline = time.monotonic() + 30.
        while time.monotonic() < deadline:
            qapp.processEvents(); page._refresh_shell()
            if len(page._last_scientific_projection.traces) == len(keys):
                break
            time.sleep(.005)
        assert controller.navigation.selected == tuple(keys)
        assert len(page._last_scientific_projection.traces) == len(keys)
        np.testing.assert_allclose(view._phase_wavelengths, sorted(expected), rtol=1e-14, atol=0.)
