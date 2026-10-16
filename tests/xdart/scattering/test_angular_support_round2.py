"""COVERAGE-SUPPORT-20261013 correction round 2 GUI rows (Claude, October 14): frozen before implementation.

Contract B through the real Run click and the real Reintegrate successor: an Append keeps the output's recorded choice
and says so when it differs from the pill, whose tooltip states the rule; a successor keeps its artifact's choice.
Codex's B4 beside it: a shown legacy selection is one request, so a changed or reordered selection reuses what is
still shown, and off and close retire the owner. Spies wrap the real kernel and reconstruction.
"""
from pathlib import Path
from unittest import mock

import h5py

from tests.xdart.scattering._output_slots import written
from tests.xdart.scattering.test_angular_support_recording import _run_off
from tests.xdart.scattering.test_angular_support_strip import (
    _command, _curves, _enable, _fresh, _legacy_copy, _masked_series, _wait)
from tests.xdart.scattering.test_browse_reference_rings_run import _page, _reopen, _run
from tests.xdart.scattering.test_coverage_control_clarity import _int_recording
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401


def test_g6_append_keeps_an_off_outputs_choice_and_says_so(qapp, tmp_path):
    from xrd_tools.analysis import angular_support as kernel
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.io.frame_view import FrameViewReader
    _run_off(qapp, tmp_path)  # the output is created with recording off
    with _page(qapp, _masked_series(tmp_path, energies=(16, 16, 16), output_mode='Append')) as (page, view, observed):
        # LF-007: Int's pill lives in Advanced → Integrate 1D; each read opens the real modal action again
        if not _int_recording(qapp, page).isChecked():
            _int_recording(qapp, page, toggle=True)
        assert _int_recording(qapp, page).isChecked(), 'the pill asks for recording'
        with mock.patch.object(kernel, 'support_footprint', wraps=kernel.support_footprint) as footprints:
            _run(qapp, page, observed)
        assert footprints.call_count == 0, 'the Append Run inherits off and computes nothing'
        details = ' '.join(str(getattr(event, 'detail', '') or '') for event in observed)
        assert 'angular coverage' in details.lower() and 'off' in details.lower(), details
        control = _int_recording(qapp, page)
        assert control.isChecked(), 'the pill keeps the preference for new outputs'
        assert 'Append' in control.toolTip(), 'the pill explains that Append keeps an output’s choice'
    with FrameViewReader(written(tmp_path / 'run.nexus', 'Int 1D')) as reader:
        labels = reader.labels()
        assert len(labels) >= 3
        assert all(reader.read(label).angular_support.reason is SupportReason.RECORDING_DISABLED for label in labels)


def test_g7_a_reintegrate_successor_keeps_an_off_outputs_choice(qapp, tmp_path):
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.io import open_scan
    from xrd_tools.io.frame_view import FrameViewReader
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    artifact = _run_off(qapp, tmp_path)
    plan = ReintegrateSuccessorPlan.from_artifact(
        artifact, entry='entry', dimension='1d', source_root=str(tmp_path),
        expected_target_snapshot=capture_target_snapshot(artifact),
        expected_labels=tuple(int(v) for v in open_scan(artifact).frame_indices),
        preparation=ScatteringWorkspace._reintegrate_preparation(_masked_series(tmp_path, one={'npt': 48}), '1d'))
    result = run_reintegrate_successor(plan)
    assert result.disposition == 'COMMITTED', result
    successor = Path(result.output_artifact)
    with FrameViewReader(successor) as reader:
        saved = [reader.read(label).angular_support for label in reader.labels()]
    assert saved and all(profile.reason is SupportReason.RECORDING_DISABLED for profile in saved), \
        [profile.text for profile in saved]
    with h5py.File(successor, 'r', locking=False) as handle:
        assert not any(name in handle['entry/integrated_1d'] for name in ANGULAR_SUPPORT_DATASETS)


def test_b4_a_shown_legacy_set_is_one_request_through_changes_off_and_close(qapp, tmp_path):
    from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
    from xrd_tools.analysis.angular_support import SavedSupportBasis
    output = _fresh(qapp, tmp_path / 'run', energies=(16,) * 9)
    legacy = _legacy_copy(output, tmp_path / 'legacy.nexus')
    with _page(qapp, _masked_series(tmp_path / 'browse')) as (page, view, _observed):
        _enable(qapp, view)
        _command(qapp, page, ShellCommandKind.SET_PLOT_MODE, 'Overlay')
        _reopen(qapp, page, legacy)
        controller, owner = page._context_controller, view.angular_support_owner
        keys = tuple(controller.frame_keys)
        assert len(keys) == 9

        def show(selection, count):
            assert controller.select_navigation(selection[0], tuple(selection))
            page._refresh_shell()
            _wait(qapp, lambda: len(_curves(view)) == count and not owner.computing)

        with mock.patch.object(SavedSupportBasis, 'profile', autospec=True,
                               side_effect=SavedSupportBasis.profile) as rebuilt:
            show(keys, 9)
            settled = rebuilt.call_count
            show(keys[5::-1], 6)  # fewer frames, reordered: every one still shown is reused
            assert rebuilt.call_count == settled
            # all nine again: only the three let go are rebuilt, in one job. Test-only repair at implementation: Browse
            # displayed six traces for (k5…k0, k8) and for (k6…k0) after the shrink (probe log), so the row reselects
            # every frame in navigation order instead.
            show(keys, 9)
            assert rebuilt.call_count == settled + 3
        _enable(qapp, view, on=False)
        assert not owner.computing and not _curves(view)
        assert owner.shutdown().status.name == 'CLEAN'
