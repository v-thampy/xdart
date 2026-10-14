"""COVERAGE-SUPPORT-20261013 decision-6 frozen GUI rows (Claude, October 13, before the setting is implemented).

The maintainer's addition at the step-3 disposition: “Record angular coverage”, default on, in Int and Stitch through
the existing shared settings/editor owners; the preference persists and is frozen at Run start; Off skips the
computation and coverage arrays, records an explicit disabled reason and prevents automatic Browse reconstruction of
those results; previously saved coverage remains displayable. Rows G1-G4 click the real control, run through the real
executor, close and restore the real session, reopen in Browse and drive the real Stitch page. New APIs are imported
inside each row, so a missing API fails only its own row.
"""
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._output_slots import written
from tests.xdart.scattering.test_angular_support_strip import _curves, _enable, _fresh, _masked_series, _wait
from tests.xdart.scattering.test_browse_reference_rings_run import _browsing, _page, _pump, _reopen, _run
from tests.xdart.scattering.test_predicted_ring_overlays import qapp  # noqa: F401
from tests.xdart.pages.test_scattering_page import isolated_settings  # noqa: F401

LABEL = "Record angular coverage"


def _control(widget):
    """The shared control as a user finds it: the one checkable button labelled “Record angular coverage”."""
    found = [button for button in widget.findChildren(QtWidgets.QAbstractButton)
             if button.isCheckable() and button.text() == LABEL]
    assert len(found) == 1, f"{len(found)} “{LABEL}” controls"
    return found[0]


def _run_off(qapp, root):
    """A Run with the control turned off through its real button; returns the output path."""
    with _page(qapp, _masked_series(root)) as (page, view, observed):
        _control(page).click()
        qapp.processEvents()
        _run(qapp, page, observed)
        _browsing(qapp, page)
    return written(root / 'run.nexus', 'Int 1D')


# ---------------------------------------------------------------------------------------------------- G1

def test_g1_int_control_is_frozen_at_run_start_and_off_records_its_reason(qapp, tmp_path, monkeypatch):
    import xdart.gui.tabs.scattering.adapters.dynamic_output as dynamic_output
    from xrd_tools.analysis.angular_support import AngularSupportPolicy, DEFAULT_RECORD_ANGULAR_SUPPORT
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.io.frame_view import FrameViewReader
    with _page(qapp, _masked_series(tmp_path)) as (page, view, observed):
        control = _control(page)
        assert control.isChecked() is DEFAULT_RECORD_ANGULAR_SUPPORT, 'the default is the one constant'
        assert 'angular' in control.toolTip().lower(), 'the control explains itself'
        control.click()
        qapp.processEvents()
        assert page._intents.snapshot().thaw().run_options['record_angular_support'] is False
        opened = []
        real = dynamic_output.open_headless_scan_session

        def admitted(*args, **kwargs):
            opened.append(kwargs['policy'])
            return real(*args, **kwargs)
        monkeypatch.setattr(dynamic_output, 'open_headless_scan_session', admitted)
        _run(qapp, page, observed)
        assert [policy.angular_support for policy in opened] == [AngularSupportPolicy(enabled=False)], \
            'the Run carries the value frozen at its start'
        _browsing(qapp, page)
    with FrameViewReader(written(tmp_path / 'run.nexus', 'Int 1D')) as reader:
        reasons = {reader.read(label).angular_support.reason for label in reader.labels()}
    assert reasons == {SupportReason.RECORDING_DISABLED}


# ---------------------------------------------------------------------------------------------------- G2

def test_g2_the_int_preference_persists_across_sessions(qapp, isolated_settings, tmp_path, monkeypatch):
    from xdart._gui_main import Main
    monkeypatch.delenv('XDART_SESSION_FRESH', raising=False)
    window = Main()
    try:
        page = window.page_handle.widget
        control = _control(page)
        assert control.isChecked()
        control.click()
        qapp.processEvents()
        assert page._intents.snapshot().thaw().run_options['record_angular_support'] is False
        window.close()
        qapp.processEvents()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
    restored = Main()
    try:
        page = restored.page_handle.widget
        assert page._intents.snapshot().thaw().run_options['record_angular_support'] is False
        assert not _control(page).isChecked(), 'the restored session shows it off'
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------------------------------- G3

def test_g3_browse_names_disabled_results_and_keeps_saved_coverage(qapp, tmp_path):
    from xrd_tools.core.angular_support import REASON_TEXT, SupportOrigin, SupportReason
    recorded = _fresh(qapp, tmp_path / 'on')
    disabled = _run_off(qapp, tmp_path / 'off')
    with _page(qapp, _masked_series(tmp_path / 'browse')) as (page, view, _observed):
        _control(page).click()  # recording is off in this session; it never governs what is displayed
        qapp.processEvents()
        _enable(qapp, view)
        _reopen(qapp, page, recorded)
        _wait(qapp, lambda: bool(_curves(view)))
        (_data, profile), = _curves(view).values()
        assert profile.origin is SupportOrigin.SAVED and profile.available, 'previously saved coverage displays'
        _reopen(qapp, page, disabled)
        _wait(qapp, lambda: REASON_TEXT[SupportReason.RECORDING_DISABLED] in view.angular_support_message())
        owner = view.angular_support_owner
        _pump(qapp, lambda: False, 1.0)
        assert not _curves(view) and not owner.computing and owner.completed_requests() == 0, \
            'an intentionally disabled result is never reconstructed'


# ---------------------------------------------------------------------------------------------------- G4

def test_g4_stitch_setting_persists_is_frozen_at_run_start_and_off_records_its_reason(qapp, tmp_path, monkeypatch):
    from dataclasses import replace
    from tests.core.test_stitch_operation import _native_form
    from tests.xdart.pages.test_stitch_page import _Status, _wait as _stitch_wait
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.pages.values import PageCleanup
    from xdart.gui.tabs.stitch.page import StitchPage
    from xdart.gui.tabs.stitch.settings import apply_page_settings, page_settings
    from xrd_tools.analysis.angular_support import AngularSupportPolicy, DEFAULT_RECORD_ANGULAR_SUPPORT
    from xrd_tools.analysis.stitch_operation import StitchOperationExecution
    from xrd_tools.core.angular_support import REASON_TEXT, SupportReason
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    page = StitchPage(empty_host_services(_Status()))
    try:
        control = _control(page)
        assert control.isChecked() is DEFAULT_RECORD_ANGULAR_SUPPORT and 'angular' in control.toolTip().lower()
        page.load_form(replace(_native_form(tmp_path, 'multigeometry', scans='1'), mode='1d'))
        _stitch_wait(page, qapp)
        control.click()
        qapp.processEvents()
        document = page_settings(page)
        assert document['fields']['record_angular_support'] is False
        policies = []
        real = StitchOperationExecution.run

        def run(self, *args, **kwargs):
            policies.append(kwargs.get('angular_support'))
            return real(self, *args, **kwargs)
        monkeypatch.setattr(StitchOperationExecution, 'run', run)
        page.run_bar.startButton.click()
        _stitch_wait(page, qapp)
        assert policies == [AngularSupportPolicy(enabled=False)], 'the Run carries the value frozen at its start'
        target = page._owner.prepared.request.module.output.target
        assert read_analysis_artifact(target).angular_support.reason is SupportReason.RECORDING_DISABLED
        view = page.scientific
        _enable(qapp, view)
        _wait(qapp, lambda: REASON_TEXT[SupportReason.RECORDING_DISABLED] in view.angular_support_message())
        assert not _curves(view)
        assert page.shutdown().status is PageCleanup.CLEAN
    finally:
        page.shutdown()
        page.deleteLater()
        qapp.processEvents()
    fresh = StitchPage(empty_host_services(_Status()))
    try:
        assert _control(fresh).isChecked()
        apply_page_settings(fresh, document)
        qapp.processEvents()
        assert not _control(fresh).isChecked(), 'the saved settings restore it off'
    finally:
        fresh.shutdown()
        fresh.deleteLater()
        qapp.processEvents()
