"""One plot-option table (STITCH-2D-20260930 C13; Codex general audit AUD-8): the view emits, and Int's and
Stitch's pages apply, every 1-D plot option through ``PLOT_OPTION_BINDINGS`` and ``plot_option_update``."""
from dataclasses import fields

from xdart.gui.pages.services import empty_host_services
from xdart.gui.pages.values import PageCleanup
from xdart.gui.tabs.scattering.scientific_plot_options import PLOT_OPTION_BINDINGS, plot_option_update
from xdart.gui.tabs.scattering.shell_values import ScientificPlotOptions, ShellCommand, ShellCommandKind
from xdart.gui.tabs.stitch.page import StitchPage
from tests.xdart.pages.test_stitch_page import _Status, qapp  # noqa: F401

#: A value each option takes, and one it refuses.
_TAKEN = {"waterfall_y_axis": "Time (s)", "waterfall_start": 3, "waterfall_stop": 9, "waterfall_step": 2,
          "overlay_offset": 1.5, "show_legend": False, "intensity_scale": "Log"}
_REFUSED = {"waterfall_y_axis": "", "waterfall_start": 0, "waterfall_stop": -1, "waterfall_step": 0,
            "overlay_offset": float("nan"), "show_legend": 1, "intensity_scale": "Cube"}


def test_one_table_binds_every_plot_option_to_its_command():
    assert [field for _path, field in PLOT_OPTION_BINDINGS] == [item.name for item in fields(ScientificPlotOptions)]
    assert len({path for path, _field in PLOT_OPTION_BINDINGS}) == len(PLOT_OPTION_BINDINGS)
    for path, field in PLOT_OPTION_BINDINGS:
        assert getattr(plot_option_update(ScientificPlotOptions(), path, _TAKEN[field]), field) == _TAKEN[field]
        assert plot_option_update(ScientificPlotOptions(), path, _REFUSED[field]) is None
    assert plot_option_update(ScientificPlotOptions(), ("overlay", "offset"), 2).overlay_offset == 2.0
    assert plot_option_update(ScientificPlotOptions(), ("other", "unknown"), True) is None


def test_the_stitch_page_applies_every_plot_option_as_int_does(qapp):
    page = StitchPage(empty_host_services(_Status()))
    try:
        for path, field in PLOT_OPTION_BINDINGS:
            page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, _TAKEN[field], path))
            assert getattr(page._view.plot_options, field) == _TAKEN[field]
            page._view_command(ShellCommand(ShellCommandKind.SET_PLOT_OPTION, _REFUSED[field], path))
            assert getattr(page._view.plot_options, field) == _TAKEN[field]  # refused, as Int refuses it
    finally:
        assert page.shutdown().status is PageCleanup.CLEAN
        page.deleteLater()
        qapp.processEvents()
