"""C7 terminal review: legacy multi-selection must settle without cache churn."""
from tests.xdart.scattering.test_angular_support_strip import (
    _command, _curves, _enable, _fresh, _legacy_copy, _masked_series, _wait)
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _reopen
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401


def test_nine_legacy_profiles_finish_reconstruction_together(qapp, tmp_path):
    from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
    output = _fresh(qapp, tmp_path / "run", energies=(16,) * 9)
    legacy = _legacy_copy(output, tmp_path / "legacy.nexus")
    with _page(qapp, _masked_series(tmp_path / "browse")) as (page, view, _observed):
        _enable(qapp, view)
        _command(qapp, page, ShellCommandKind.SET_PLOT_MODE, "Overlay")
        _reopen(qapp, page, legacy)
        keys = page._context_controller.frame_keys
        assert len(keys) == 9
        assert page._context_controller.select_navigation(keys[0], tuple(keys))
        page._refresh_shell()
        _wait(qapp, lambda: len(_curves(view)) == 9 and not view.angular_support_owner.computing)
