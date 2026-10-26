"""Total I(q) compares GI theta-scan frames on the unchanged radial grid."""
import numpy as np

from tests.xdart.scattering.test_angular_support_strip import _masked_series, _wait
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _run, _browsing
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.run_configuration import GIIntent


def test_gi_total_q_overlay_keeps_frames_with_different_incidence_angles(qapp, tmp_path):
    intent = _masked_series(tmp_path, unit='q_A^-1')
    intent.gi = GIIntent(enabled=True, incidence_motor='th', mode_1d='q_total')
    sidecars = sorted(tmp_path.glob('*.txt'))
    assert len(sidecars) == 2
    for path, angle in zip(sidecars, (.2, 2.)):
        path.write_text(path.read_text().replace('# Motors\n', f'# Motors\nth = {angle}\n'))
    with _page(qapp, intent) as (page, view, observed):
        page._handle_shell_command(ShellCommand(ShellCommandKind.SET_PLOT_MODE, 'Overlay'))
        _run(qapp, page, observed)
        _browsing(qapp, page)
        controller = page._context_controller
        frames = tuple(controller.frame_keys)
        assert len(frames) == 2 and controller.select_navigation(frames[-1], frames)
        page._refresh_shell()
        _wait(qapp, lambda: view.trace_history_keys == frames and len(view.curve.listDataItems()) == 2)
        assert view.trace_history_keys == frames
        first, last = (item.getData() for item in view.curve.listDataItems())
        np.testing.assert_allclose(first[0], last[0], rtol=1e-9)
        # Compare scientific intensities before the view's configured vertical
        # Overlay offset is applied to the second drawn curve.
        native = page._last_scientific_projection.traces
        assert len(native) == 2
        np.testing.assert_allclose(native[0].intensity, native[1].intensity, rtol=1e-9, equal_nan=True)
