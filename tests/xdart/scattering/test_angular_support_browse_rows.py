"""COVERAGE-SUPPORT-20261013 step-7 regression row (Claude, October 13; not part of the frozen oracle).

Sabotage S4 (ignore generation/selection: an earlier Browse output's support shown for a later one) survived the
frozen D4/D5, whose Single-mode Browse frames come from the reader's own publication, not Browse's 1-D cache rows.
Overlay draws those rows; this row reopens two outputs with different masks at the same frame coordinates on one page
and checks every drawn profile against its own output's independent expectation.
"""
import os

import numpy as np

from tests.core.test_angular_support import TOL
from tests.xdart.scattering.test_angular_support_strip import BAND, _analytic, _command, _enable, _fresh, \
    _masked_series, _wait
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _pump, _reopen
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401


def test_g5_overlay_rows_carry_each_outputs_own_support(qapp, tmp_path):
    from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
    from xrd_tools.core.angular_support import SupportOrigin
    outputs = ((_fresh(qapp, tmp_path / 'first', energies=(16, 16)), BAND),
               (_fresh(qapp, tmp_path / 'second', energies=(16, 16), band=(100, 200)), (100, 200)))
    with _page(qapp, _masked_series(tmp_path / 'browse')) as (page, view, _observed):
        _enable(qapp, view)
        _command(qapp, page, ShellCommandKind.SET_PLOT_MODE, 'Overlay')
        bands = {}
        expected_keys = []
        for output, band in outputs:
            assert _pump(qapp, lambda: not page._context_controller.browse_preview_polling_needed, 30.)
            _reopen(qapp, page, output)
            keys = page._context_controller.frame_keys
            assert page._context_controller.select_navigation(keys[0], tuple(keys))
            page._refresh_shell()
            bands[os.path.normcase(os.path.abspath(output))] = band
            expected_keys.extend(keys)

            def drawn():
                return [item for item in view.angular_support_plot.getPlotItem().listDataItems()
                        if item.getData()[0] is not None]

            def settled():
                # Overlay retains the earlier output's compatible curves (1.2.2): wait until this output's own
                # frames are among the rows and every row's strip curve is drawn.
                rows, items = view.trace_history_projections, drawn()
                return (len(keys) == 2 and len(rows) == len(expected_keys)
                        and all(row.frame is key for row, key in zip(rows, expected_keys, strict=True))
                        and len(items) == len(rows)
                        and all(item.angular_support is row.angular_support
                                for item, row in zip(items, rows, strict=True)))

            _wait(qapp, settled)
            for item, row in zip(drawn(), view.trace_history_projections, strict=True):
                profile = item.angular_support
                assert profile is row.angular_support
                assert profile.origin is SupportOrigin.SAVED and profile.available
                np.testing.assert_allclose(profile.supported_deg, _analytic(profile, bands[row.frame.artifact]),
                                           atol=TOL * 360, err_msg=f"{row.frame.artifact}: another output's support is shown")
