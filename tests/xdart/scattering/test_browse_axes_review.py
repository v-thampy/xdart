"""Production record kinds that share Int's Browse axis-resolution branch."""
from dataclasses import replace

import h5py
import numpy as np
import pytest

from tests.core.v2_fixture_factory import current_entry
from tests.xdart.scattering.test_browse_axes_after_gi import _open, workspace, qapp
from tests.xdart.scattering.test_gi_browse_preview_modes import _write_named_gi
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.core import FrameRecord, FrameView, axis_from_unit
from xrd_tools.io import write_frame_records


@pytest.mark.parametrize("unit,mode", [
    ("qip_A^-1", "q_ip"), ("chi_deg", "default"), ("2th_deg", "default"),
])
@pytest.mark.parametrize("share", [False, True])
def test_native_one_dimensional_record_after_gi(workspace, tmp_path, unit, mode, share):
    page = workspace
    gi = tmp_path / "gi.nexus"
    _write_named_gi(gi)
    _open(page, gi)
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_SHARE_AXIS, share))
    target = tmp_path / "native_int1d.nexus"
    values = np.array([2., 4., 8.])
    with h5py.File(target, "w") as handle:
        write_frame_records(current_entry(handle), [FrameRecord.from_view(FrameView(
            1, axis_1d=axis_from_unit(unit, np.array([.1, .2, .3])), intensity_1d=values,
        ), mode_1d=mode)])
    state = _open(page, target)
    np.testing.assert_array_equal(state.traces[0].intensity, values)
    assert state.traces[0].axis.unit == unit
    expected = {"qip_A^-1": "q_ip", "chi_deg": "chi", "2th_deg": "2theta"}[unit]
    assert page._preferences.plot_axis == expected
    assert page._shell.scientific.plot_axis.currentData() == expected
    assert page._preferences.share_axis is share


@pytest.mark.parametrize("share", [False, True])
def test_standard_stitch_map_after_gi_keeps_trace_and_shared_axis(workspace, tmp_path, share):
    from tests.core.test_stitch_operation import _native_form
    from tests.xdart.scattering.test_p34_reintegrate_operation import _wait
    from xdart.gui.tools.stitch_values import prepare_stitch_tool
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.stitch_operation import run_stitch_operation
    from xrd_tools.io.frame_view import read_frame_view

    page = workspace
    gi = tmp_path / "gi.nexus"
    _write_named_gi(gi)
    _open(page, gi)
    page._handle_shell_command(ShellCommand(ShellCommandKind.SET_SHARE_AXIS, share))
    form = replace(_native_form(tmp_path, "multigeometry", scans="1", unit="2th_deg"),
                   mode="2d", map_unit="2th_deg", npt_1d=37, npt_rad_2d=23, npt_azim_2d=19)
    request = prepare_stitch_tool(form).request
    assert run_stitch_operation(request).terminal.disposition is ModuleDisposition.COMMITTED
    target = request.module.output.target
    saved = read_frame_view(target, 1, include_thumbnail=False)
    _open(page, target)
    _wait(lambda: True if (page._last_scientific_projection.heavy is not None
                          and page._last_scientific_projection.heavy.cake is not None) else None)
    state = page._last_scientific_projection
    shared = share and page._intents.snapshot().thaw().processing_mode == "Int 2D"
    expected = 4.0 * np.pi * np.sin(np.deg2rad(saved.axis_1d.values) / 2) if shared else saved.axis_1d.values
    assert state.traces[0].axis.unit == ("q_A^-1" if shared else "2th_deg")
    np.testing.assert_allclose(state.traces[0].axis.values, expected, rtol=1e-6)
    np.testing.assert_allclose(state.traces[0].intensity, saved.intensity_1d, equal_nan=True)
    assert state.share_axis is share
