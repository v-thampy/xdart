"""Deferred item 10 (the maintainer's list, 2026-09-30): a GI Stitch record opened in Int's Browse is a GI frame,
as Int's own are. Int read it as Standard: a merged record has no one incidence angle, and Int's test was the
angle or a non-(χ, q) primary map. A record whose modes are GI's is GI, so Int offers the q_ip-q_oop map beside
the (χ, q) one, and its cuts on the plot axis."""
from dataclasses import replace

import numpy as np
import pytest

from tests.xdart.scattering.test_p34_reintegrate_operation import _page, _wait
from tests.xdart.tools.test_stitch_native_gi import _SAMPLE, _gi_form
from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xdart.gui.tools.stitch_values import prepare_stitch_tool
from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.stitch_operation import run_stitch_operation
from xrd_tools.integrate.stitch_gi import StitchGI
from xrd_tools.io.analysis_artifact import read_analysis_artifact


def _painted(page, check=lambda state: True):
    def ready():
        page._drain_executor()
        value = page._last_scientific_projection
        return value if value is not None and value.traces and check(value) else None
    return _wait(ready)


@pytest.mark.parametrize("mode", ["1d", "2d"])
def test_a_gi_stitch_record_is_a_gi_frame_in_int_browse(tmp_path, monkeypatch, _xdart_qt_harness, mode):
    form = _gi_form(tmp_path, "multigeometry", StitchGI(("q_ip", "q_oop"), incidence_motor="eta",
                                                        tilt_motor="chi", sample_motors=_SAMPLE))
    if mode == "2d":
        form = replace(form, mode="2d", npt_rad_2d=48, npt_azim_2d=32)
    request, = prepare_stitch_tool(form).requests
    assert run_stitch_operation(request).terminal.disposition is ModuleDisposition.COMMITTED
    target = request.module.output.target
    payload = read_analysis_artifact(target)
    (tmp_path / "int").mkdir()
    page, _store = _page(tmp_path / "int", monkeypatch)
    try:
        browse = page._context_controller.begin_browse(str(target), source_root=str(tmp_path))
        outcome = _wait(page._context_controller.poll_browse)
        assert outcome.request is browse and outcome.status is BrowseLoadStatus.READY, outcome
        page.resize(1200, 800)
        page.show()
        page._refresh_shell()
        # Sparse traces can arrive before the independently hydrated 2D map.
        state = _painted(
            page,
            lambda value: mode == "1d"
            or (value.heavy is not None and value.heavy.cake is not None),
        )
        assert state.measurement_mode == "GI" and state.gi_mode_1d == "q_total"
        np.testing.assert_array_equal(state.traces[0].intensity, (payload.profile if mode == "2d" else payload).intensity)
        if mode == "1d":
            return
        assert state.gi_maps == ("q_chi", "qip_qoop") and state.gi_mode_2d == "q_chi"
        np.testing.assert_array_equal(state.heavy.cake, np.asarray(payload.intensity.T, dtype=float))
        assert ScatteringWorkspace._edit_scientific_preference(
            page, ShellCommand(ShellCommandKind.SET_IMAGE_AXIS, "qip_qoop"))
        page._refresh_shell()
        state = _painted(page, lambda value: value.image_axis == "qip_qoop")
        np.testing.assert_array_equal(state.heavy.cake, np.asarray(payload.gi_map.intensity.T, dtype=float))
        choices = [page._shell.scientific.plot_axis.itemData(row)
                   for row in range(page._shell.scientific.plot_axis.count())]
        assert {"q_ip", "q_oop"} <= set(choices)  # its cuts, from the q_ip-q_oop map
    finally:
        page.close()
        page.deleteLater()
