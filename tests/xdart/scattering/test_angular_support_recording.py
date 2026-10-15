"""COVERAGE-SUPPORT-20261013 decision-6 frozen GUI rows (Claude, October 13, before the setting is implemented).

The maintainer's addition at the step-3 disposition: “Record angular coverage”, default on, in Int and Stitch through
the existing shared settings/editor owners; the preference persists and is frozen at Run start; Off skips the
computation and coverage arrays and records an explicit disabled reason; previously saved coverage remains displayable.
LIVE-TEST-FIXES-20261015 allows requested Browse reconstruction of qualifying Standard Int Off results. Stitch still
requires recorded coverage. Rows G1-G4 click the real control, run through the real
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
    """The shared control as a user finds it: the one checkable button named “Record angular coverage”, whose text
    says its state (COVERAGE-EXPORT-FOLLOWUP-20261014 Stage C: “Record angular coverage: On” or “… Off”)."""
    found = [button for button in widget.findChildren(QtWidgets.QAbstractButton)
             if button.isCheckable() and button.accessibleName() == LABEL]
    assert len(found) == 1, f"{len(found)} “{LABEL}” controls"
    assert found[0].text() == f"{LABEL}: {'On' if found[0].isChecked() else 'Off'}", found[0].text()
    return found[0]


def _run_off(qapp, root, *, mode='Int 1D'):
    """A Run with the control turned off through its real button; returns the output path."""
    with _page(qapp, _masked_series(root, mode=mode)) as (page, view, observed):
        _control(page).click()
        qapp.processEvents()
        _run(qapp, page, observed)
        _browsing(qapp, page)
    return written(root / 'run.nexus', mode)


# ---------------------------------------------------------------------------------------------------- G1

def test_g1_int_control_is_frozen_at_run_start_and_off_records_its_reason(qapp, tmp_path, monkeypatch):
    import xdart.gui.tabs.scattering.adapters.dynamic_output as dynamic_output
    from xrd_tools.analysis.angular_support import AngularSupportPolicy, DEFAULT_RECORD_ANGULAR_SUPPORT
    from xrd_tools.core.angular_support import SupportReason
    from xrd_tools.io.frame_view import FrameViewReader
    with _page(qapp, _masked_series(tmp_path)) as (page, view, observed):
        control = _control(page)
        # the series asks for recording explicitly (the default itself is pinned by A1, Off since
        # INT-PERF-REGRESSION-20261014); an Off equal to the default is stored as no option
        assert control.isChecked(), 'the series asks for recording'
        assert 'angular' in control.toolTip().lower(), 'the control explains itself'
        control.click()
        qapp.processEvents()
        run_options = page._intents.snapshot().thaw().run_options
        assert run_options.get('record_angular_support', DEFAULT_RECORD_ANGULAR_SUPPORT) is False
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
        # INT-PERF-REGRESSION-20261014: a fresh session is Off; a deliberate On is the choice that must persist
        assert not control.isChecked(), 'a fresh session records nothing'
        control.click()
        qapp.processEvents()
        assert page._intents.snapshot().thaw().run_options['record_angular_support'] is True
        window.close()
        qapp.processEvents()
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
    restored = Main()
    try:
        page = restored.page_handle.widget
        assert page._intents.snapshot().thaw().run_options['record_angular_support'] is True
        assert _control(page).isChecked(), 'the restored session shows it on'
    finally:
        restored.close()
        restored.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------------------------------- G3

@pytest.mark.parametrize('mode', ['Int 1D', 'Int 2D'])
def test_g3_browse_reconstructs_disabled_int_and_keeps_saved_coverage(qapp, tmp_path, mode):
    import hashlib
    from xrd_tools.core.angular_support import SupportOrigin
    from tests.xdart.scattering.test_angular_support_strip import _analytic, TOL
    recorded = _fresh(qapp, tmp_path / 'on')
    disabled = _run_off(qapp, tmp_path / 'off', mode=mode)
    before = hashlib.sha256(disabled.read_bytes()).digest()
    with _page(qapp, _masked_series(tmp_path / 'browse', band=(20, 120), one={'npt': 64})) as (page, view, _observed):
        _control(page).click()  # recording is off in this session; it never governs what is displayed
        qapp.processEvents()
        _enable(qapp, view)
        _reopen(qapp, page, recorded)
        _wait(qapp, lambda: bool(_curves(view)))
        (_data, profile), = _curves(view).values()
        assert profile.origin is SupportOrigin.SAVED and profile.available, 'previously saved coverage displays'
        # Test-only oracle repair: a reopen replaces the shown Browse only once the page's poll has consumed its last
        # repaint (else "previous Browse cleanup is pending", by design and with the strip off too, as at cf49f3cb)
        assert _pump(qapp, lambda: not page._context_controller.browse_preview_polling_needed, 30.)
        _reopen(qapp, page, disabled)
        owner = view.angular_support_owner
        _wait(qapp, lambda: bool(_curves(view)) and not owner.computing, seconds=10.)
        (_data, profile), = _curves(view).values()
        assert profile.origin is SupportOrigin.RECONSTRUCTED and profile.available
        np.testing.assert_allclose(profile.supported_deg, _analytic(profile), atol=TOL * 360)
        computed = owner.completed_requests()
        assert computed > 0
        page._refresh_shell()
        qapp.processEvents()
        assert owner.completed_requests() == computed, 'unchanged display reuses the reconstruction'
        _enable(qapp, view, on=False)
        assert not _curves(view) and owner.retained_keys() == 0
    assert hashlib.sha256(disabled.read_bytes()).digest() == before, 'display leaves the saved scan untouched'


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
        assert not control.isChecked(), 'Off by default (INT-PERF-REGRESSION-20261014)'
        page.load_form(replace(_native_form(tmp_path, 'multigeometry', scans='1'), mode='1d'))
        _stitch_wait(page, qapp)
        control.click()  # explicitly On, then the user's Off below is the value the Run must freeze
        qapp.processEvents()
        assert control.isChecked()
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
        _control(fresh).click()  # On first, so restoring the saved Off is a real change
        qapp.processEvents()
        assert _control(fresh).isChecked()
        apply_page_settings(fresh, document)
        qapp.processEvents()
        assert not _control(fresh).isChecked(), 'the saved settings restore it off'
    finally:
        fresh.shutdown()
        fresh.deleteLater()
        qapp.processEvents()


# ---------------------------------------------------------------------------------------------------- A1, A3

def test_a1_a_fresh_int_page_records_nothing_and_skips_capture(qapp, tmp_path, monkeypatch):
    """INT-PERF-REGRESSION-20261014 A1: with no recording option, Int's control shows Off and the frozen Run records
    nothing. Its output carries the Off marker and no coverage datasets, and the real capture is never asked."""
    import h5py
    from xrd_tools.analysis.angular_support import RunSupportCapture
    from xrd_tools.io.schema import ANGULAR_SUPPORT_DATASETS, ANGULAR_SUPPORT_RECORDING_ATTR
    intent = _masked_series(tmp_path)
    del intent.run_options['record_angular_support']  # the default itself
    asked = []
    real = RunSupportCapture.profile

    def profile(self, *args, **kwargs):  # the real capture, only counted
        asked.append(1)
        return real(self, *args, **kwargs)
    monkeypatch.setattr(RunSupportCapture, 'profile', profile)
    with _page(qapp, intent) as (page, view, observed):
        control = _control(page)
        assert not control.isChecked() and control.text() == f'{LABEL}: Off', 'Off by default'
        assert 'record_angular_support' not in page._intents.snapshot().thaw().run_options, 'the default is no option'
        _run(qapp, page, observed)
        _browsing(qapp, page)
    assert not asked, 'recording Off asks the capture for nothing'
    with h5py.File(written(tmp_path / 'run.nexus', 'Int 1D'), 'r') as handle:
        assert handle['entry'].attrs[ANGULAR_SUPPORT_RECORDING_ATTR] == 'off'
        assert not any(name in handle['entry/integrated_1d'] for name in ANGULAR_SUPPORT_DATASETS)


def test_a3_missing_true_and_false_round_trip_in_int_and_stitch(qapp, tmp_path):
    """INT-PERF-REGRESSION-20261014 A3: a missing recording choice follows the Off default and a represented True or
    False wins, through Int's profile codec and Stitch's saved settings. No earlier intent is inferred."""
    from tests.xdart.pages.test_stitch_page import _Status
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage
    from xdart.gui.tabs.stitch.settings import apply_page_settings, page_settings
    from xrd_tools.session import dump_run_intent_profile, load_run_intent_profile
    for value, expected in ((None, False), (True, True), (False, False)):
        intent = _masked_series(tmp_path / f'int-{value}')
        del intent.run_options['record_angular_support']
        if value is not None:
            intent.run_options['record_angular_support'] = value
        restored = load_run_intent_profile(dump_run_intent_profile(intent))
        assert restored.run_options == intent.run_options, 'profile preserves missing versus represented choices'
        with _page(qapp, restored) as (page, _view, _observed):
            assert _control(page).isChecked() is expected, f'Int with {value!r}'
    page = StitchPage(empty_host_services(_Status()))
    try:
        document = page_settings(page)
    finally:
        page.shutdown()
        page.deleteLater()
        qapp.processEvents()
    for value, expected in ((None, False), (True, True), (False, False)):
        fields = {key: item for key, item in document['fields'].items() if key != 'record_angular_support'}
        if value is not None:
            fields['record_angular_support'] = value
        fresh = StitchPage(empty_host_services(_Status()))
        try:
            if value is False:
                _control(fresh).click()  # On first, so the saved False is a real change
                qapp.processEvents()
            apply_page_settings(fresh, {**document, 'fields': fields})
            qapp.processEvents()
            assert _control(fresh).isChecked() is expected, f'Stitch with {value!r}'
        finally:
            fresh.shutdown()
            fresh.deleteLater()
            qapp.processEvents()
