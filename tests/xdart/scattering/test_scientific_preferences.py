"""The plot-command reducer Int's workspace and the Stitch page share: one set of rules for the view's
preferences (Stitch vs Int 2D audit, 2026-09-30: Stitch kept a copy of Int's)."""
from dataclasses import replace

from xdart.gui.tabs.scattering.scientific_preferences import scientific_preference_updates
from xdart.gui.tabs.scattering.shell_projection import ScientificPreferences
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind


def _updates(preferences, kind, value=None, path=(), **kwargs):
    return scientific_preference_updates(preferences, ShellCommand(kind, value, path), **kwargs)


def test_the_shared_rules():
    pinned = replace(ScientificPreferences(plot_mode="Overlay"), slice_pins=("pin",))
    # a pin lives in Overlay and Waterfall only
    assert _updates(pinned, ShellCommandKind.SET_PLOT_MODE, "Single") == {"plot_mode": "Single", "slice_pins": ()}
    assert _updates(pinned, ShellCommandKind.SET_PLOT_MODE, "Waterfall") == {"plot_mode": "Waterfall"}
    assert _updates(pinned, ShellCommandKind.SET_PLOT_MODE, "Stack") is None
    # Share axis follows the map shown, and an image axis it cannot share turns it off
    shared = ScientificPreferences(share_axis=True, plot_axis="Q")
    assert _updates(shared, ShellCommandKind.SET_IMAGE_AXIS, "2Th-Chi")["plot_axis"] == "2theta"
    assert _updates(shared, ShellCommandKind.SET_IMAGE_AXIS, "nonsense") is None
    assert _updates(ScientificPreferences(), ShellCommandKind.SET_SHARE_AXIS, True,
                    shown_image_axis="2Th-Chi")["share_axis"] is True
    # the log scale is the image's and the curve's
    log = _updates(ScientificPreferences(), ShellCommandKind.SET_LOG_SCALE, True)
    assert log["log_scale"] is True and log["plot_options"].intensity_scale == "Log"
    assert _updates(ScientificPreferences(), ShellCommandKind.SET_RANGE, 2.5, ("q", "high"))["q_range"][1] == 2.5
    assert _updates(ScientificPreferences(), ShellCommandKind.CLEAR_1D) is None  # a page's own command
