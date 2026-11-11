"""Config ▸ Performance Diagnostics must never stop a Run from processing its scans (1.2.10 review H3).

In 1.2.9, Apply stored private pipeline options in the Run options whatever the Run mode, the session kept them
across restarts, and a Live, Batch or Append Run then skipped every scan ("post-G2 pipeline V2 requires a non-Live
run [SCAN-SKIPPED]"). The rows drive the real page and dialog for Apply, the real session profile codec and the real
StandardRunExecutor for the Run; none replaces the seam being fixed.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering.test_e1b2_page_command_boundaries import _Executor, _Sources, _dispose
from tests.xdart.scattering.test_live_owner_transitions import _intent
from tests.xdart.scattering.test_live_qualify_refusal_run import _frame, _labels, _said, _stop, _terminals, _wait
from tests.xdart.scattering.test_p1b_output_graph import _run_to_terminal, _start
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.performance_diagnostics import PerformanceDiagnosticsDialog
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_intent_profile import dump_run_intent_profile, load_run_intent_profile

_PRIVATE = ("_post_g2_pipeline_v2", "_post_g2_output_diagnostics_v1", "_post_g2_unfunded_staging_diagnostic_v1")
_STAGING_MARKER = {"mode": "UNSAFE_UNFUNDED", "checkpoint": 10_000,
                   "staging_frame_cap": 10_008, "max_frames": 3_621}
#: The Run kinds that cannot take a pipeline choice, as (live, output mode, batch, processing mode). Apply already
#: drops the private options for XYE-only processing, so that kind is reached only through a stored choice.
_INCOMPATIBLE = {
    "live-append": (True, "Append", False, "Int 1D"),
    "folder-append": (False, "Append", False, "Int 1D"),
    "folder-batch": (False, "Overwrite", True, "Int 1D"),
}
_STORED = {**_INCOMPATIBLE, "folder-xye": (False, "Overwrite", False, "Int 1D (XYE)")}
#: A profile exactly as 1.2.9's session wrote it after Apply with the dialog's values (bytes from the parent's
#: dump_run_intent_profile at 8763203d; the lane's leaves/a1-session-129-profile.txt).
_SESSION_129_PROFILE = """{
  "schema": "xdart.run-intent-profile",
  "version": 1,
  "intent": {
    "source_spec": null,
    "processing_mode": "Int 1D",
    "output_mode": "Overwrite",
    "live_mode": false,
    "batch_mode": false,
    "max_cores": 2,
    "bai_1d_args": {},
    "bai_2d_args": {},
    "gi": {
      "enabled": false,
      "incidence_motor": "Manual",
      "th_val": 0.1,
      "sample_orientation": 4,
      "tilt_angle": 0.0,
      "mode_1d": "q_total",
      "mode_2d": "qip_qoop",
      "gi_exit_angle_convention": "xdart_reflection_qoop_v1"
    },
    "threshold": {
      "apply_threshold": false,
      "threshold_min": null,
      "threshold_max": null,
      "mask_saturation": true
    },
    "poni_file": "",
    "poni_values": null,
    "mask_file": "",
    "background": {
      "version": 1,
      "mode": "None"
    },
    "project_root": "/data/project",
    "save_path": "",
    "run_options": {
      "_post_g2_output_diagnostics_v1": {
        "durable_fsync": true,
        "save_xye": true
      },
      "_post_g2_pipeline_v2": {
        "nexus_record_batch_size": 8,
        "reduction_inflight": 8,
        "semantic_checkpoint_frame_cap": 16,
        "staging_frame_cap": 64,
        "writer_settlement_batch_size": 1
      },
      "heavy_window": 32
    }
  }
}
"""


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _click_apply_when_shown(holder, *, edit=None, timeout_ms=10_000):
    """Press the real dialog's Apply once its modal loop shows it (``holder()`` returns the dialog or None)."""
    timer = QtCore.QElapsedTimer()
    timer.start()

    def click():
        dialog = holder()
        if dialog is None or not dialog.isVisible():
            if timer.elapsed() < timeout_ms:
                QtCore.QTimer.singleShot(10, click)
            return
        if edit is not None:
            edit(dialog)
        box = dialog.findChild(QtWidgets.QDialogButtonBox)
        box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()

    QtCore.QTimer.singleShot(0, click)


def _applied_intent(qapp, intent, monkeypatch, *, edit=None, plot_interval_ms=None):
    """The intent the real page holds after Config ▸ Performance Diagnostics… is accepted with its loaded values."""
    monkeypatch.setenv("XDART_PERF", "0")
    monkeypatch.setenv("XDART_PERF_QUARTILES", "0")
    page = ScatteringWorkspace(intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
                               sources=_Sources(), executor=_Executor())
    try:
        _click_apply_when_shown(lambda: page._performance_diagnostics_dialog, edit=edit)
        page._handle_shell_command(ShellCommand(ShellCommandKind.MENU, "Config:Performance Diagnostics…"))
        assert os.environ["XDART_PERF"] == "1"
        if plot_interval_ms is not None:
            assert page._live_plot_interval_ms == plot_interval_ms
        return page._intents.snapshot().thaw()
    finally:
        _dispose(page, qapp)


def _with_mode(intent, kind):
    live, output, batch, processing = _STORED[kind]
    intent.live_mode, intent.output_mode, intent.batch_mode = live, output, batch
    intent.processing_mode = processing
    return intent


def _run(tmp_path, raw, intent, *, scans):
    """Run ``intent`` through the real executor; returns (frames each scan's output reported ready, terminal)."""
    if intent.live_mode:
        executor = StandardRunExecutor(join_timeout=2.0)
        identity = _start(executor, intent, request_value=1)
        events: list = []
        try:
            _wait(executor, events, lambda seen: all(len(set(_labels(seen, stem))) == 2 for stem in scans), 25.)
        except AssertionError:
            pass  # a Run that skips its scans never reaches its frames; the assertions below say so
        terminals = _stop(executor, identity, events)
        return {stem: sorted(set(_labels(events, stem))) for stem in scans}, terminals, events
    executor, identity, events = _run_to_terminal(intent, request_value=1)
    executor.close(identity)
    return None, _terminals(events), list(events)


def _scans(raw, names):
    for stem in names:
        for index in (1, 2):
            _frame(raw, stem, index, "txt", float(index))


@pytest.mark.parametrize("route", ["applied", "session"])
@pytest.mark.parametrize("kind", sorted(_INCOMPATIBLE))
def test_apply_with_the_dialog_values_then_an_incompatible_run_processes_every_scan(
    qapp, tmp_path, monkeypatch, kind, route,
):
    """Apply with the values the dialog loads, then a Live, folder Append or Batch Run, directly and after the
    session profile's round trip: every scan is processed and none is skipped for a pipeline choice."""
    raw = tmp_path / "raw"
    raw.mkdir()
    scans = ("scana", "scanb", "scanc")
    _scans(raw, scans)
    intent = _with_mode(_intent(tmp_path, raw, live=False, output="Overwrite"), kind)
    applied = _applied_intent(qapp, intent, monkeypatch)
    if route == "session":
        applied = load_run_intent_profile(dump_run_intent_profile(applied))
    ready, terminals, events = _run(tmp_path, raw, applied, scans=scans)
    _assert_every_scan(ready, terminals, scans)
    detail = terminals[-1][1]
    assert not _said(events, "post-G2"), terminals
    assert "skipped" not in detail.replace("0 skipped", ""), terminals


def _assert_every_scan(ready, terminals, scans):
    detail = terminals[-1][1] if terminals else ""
    if ready is not None:
        assert ready == {stem: [1, 2] for stem in scans}, (ready, terminals)
        assert terminals and terminals[-1][0] == "STOPPED" and "6 processed" in detail, terminals
    else:
        assert terminals and terminals[-1][0] == "FINISHED" and detail.startswith("Complete · 6 processed"), terminals


@pytest.mark.parametrize("kind", sorted(_STORED))
def test_the_same_runs_with_no_stored_choice_process_every_scan(qapp, tmp_path, kind, caplog):
    """The control for the rows below (green on 8763203d too): with no private option stored, each of these Runs
    processes all six frames of its three scans through the same harness, and says nothing about diagnostics."""
    raw = tmp_path / "raw"
    raw.mkdir()
    scans = ("scana", "scanb", "scanc")
    _scans(raw, scans)
    intent = _with_mode(_intent(tmp_path, raw, live=False, output="Overwrite"), kind)
    caplog.set_level(logging.INFO)
    ready, terminals, _events = _run(tmp_path, raw, intent, scans=scans)
    _assert_every_scan(ready, terminals, scans)
    assert not [record for record in caplog.records if "[PERF-DIAGNOSTICS]" in record.getMessage()]


@pytest.mark.parametrize("kind", sorted(_STORED))
def test_a_stored_pipeline_choice_is_ignored_once_by_a_run_that_cannot_take_it(qapp, tmp_path, kind, caplog):
    """A choice that differs from the defaults (record batch 4, fsync on) stays stored for a later Run that can take
    it; a Live, folder Append, Batch or XYE-only Run over three scans processes all of them and says once that it
    ignored it."""
    raw = tmp_path / "raw"
    raw.mkdir()
    scans = ("scana", "scanb", "scanc")
    _scans(raw, scans)
    intent = _with_mode(_intent(tmp_path, raw, live=False, output="Overwrite"), kind)
    intent.run_options["_post_g2_pipeline_v2"] = {
        "writer_settlement_batch_size": 1, "nexus_record_batch_size": 4, "reduction_inflight": 8,
        "semantic_checkpoint_frame_cap": 16, "staging_frame_cap": 64}
    intent.run_options["_post_g2_output_diagnostics_v1"] = {"save_xye": True, "durable_fsync": True}
    caplog.set_level(logging.INFO)
    ready, terminals, _events = _run(tmp_path, raw, intent, scans=scans)
    _assert_every_scan(ready, terminals, scans)
    notices = [record.getMessage() for record in caplog.records if "[PERF-DIAGNOSTICS]" in record.getMessage()]
    assert len(notices) == 1, notices
    assert any(wording in notices[0].casefold() for wording in ("ignored", "not used")), notices


def test_apply_with_the_default_pipeline_values_stores_no_private_options(qapp, tmp_path, monkeypatch):
    """Apply for timing only (the dialog's values unchanged) leaves the Run options as they were, so no later Run
    has a pipeline choice to ignore; the plot interval and timing still apply."""
    raw = tmp_path / "raw"
    raw.mkdir()
    intent = _intent(tmp_path, raw, live=True)
    intent.run_options["heavy_window"] = 32
    def change_plot_interval(dialog):
        editor = getattr(dialog.plot_interval, "editor", dialog.plot_interval)
        if isinstance(editor, QtWidgets.QSpinBox):
            editor.setValue(625)
        else:
            editor.setText("625")

    applied = _applied_intent(qapp, intent, monkeypatch, edit=change_plot_interval, plot_interval_ms=625)
    assert applied.run_options == {"heavy_window": 32}


def test_apply_keeps_an_explicit_output_choice_for_a_compatible_run(qapp, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    scans = ("scana",)
    _scans(raw, scans)
    intent = _intent(tmp_path, raw, live=False, output="Overwrite")

    def turn_off_xye(dialog):
        editor = getattr(dialog.save_xye, "editor", dialog.save_xye)
        editor.setChecked(False)

    applied = _applied_intent(qapp, intent, monkeypatch, edit=turn_off_xye)
    assert applied.run_options["_post_g2_output_diagnostics_v1"]["save_xye"] is False
    _, terminals, _ = _run(tmp_path, raw, applied, scans=scans)
    assert terminals and terminals[-1][0] == "FINISHED", terminals
    assert terminals[-1][1].startswith("Complete · 2 processed"), terminals
    output = Path(applied.save_path)
    assert any(output.rglob("*.nexus")), terminals
    assert not list(output.rglob("*.xye"))


def test_a_lone_staging_marker_is_ignored_in_each_incompatible_run(qapp, tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC", raising=False)
    caplog.set_level(logging.INFO)
    scans = ("scana", "scanb", "scanc")
    for run_number in (1, 2):
        run_path = tmp_path / str(run_number)
        raw = run_path / "raw"
        raw.mkdir(parents=True)
        _scans(raw, scans)
        intent = _with_mode(_intent(run_path, raw, live=False, output="Overwrite"), "folder-batch")
        intent.run_options[_PRIVATE[2]] = _STAGING_MARKER.copy()
        caplog.clear()
        ready, terminals, _ = _run(run_path, raw, intent, scans=scans)
        _assert_every_scan(ready, terminals, scans)
        notices = [record for record in caplog.records if "[PERF-DIAGNOSTICS]" in record.getMessage()]
        assert len(notices) == 1, notices


def test_the_dialog_loads_the_adapters_defaults_and_names_the_fsync_default(qapp):
    """With nothing stored, the dialog shows what a Run uses without it: the adapter's pipeline tuple and durable fsync
    off (the adapter's buffered default since 9922678d), and its text says that fsync is off by default."""
    from xdart.gui.tabs.scattering.adapters.dynamic_output import (
        _PostG2OutputDiagnosticsChoice, _PostG2PipelineV2Choice,
    )
    from xrd_tools.session.run_configuration import RunIntent

    dialog = PerformanceDiagnosticsDialog()
    try:
        _click_apply_when_shown(lambda: dialog)
        values = dialog.edit(RunIntentStore(RunIntent()).snapshot(), 375)
        assert values is not None
        default = _PostG2OutputDiagnosticsChoice()
        assert (values.save_xye, values.durable_fsync) == (default.save_xye, default.durable_fsync) == (True, False)
        pipeline = _PostG2PipelineV2Choice(1, 8, 8, 16, 64)
        assert values.pipeline_mapping() == {
            "writer_settlement_batch_size": pipeline.writer_settlement_batch_size,
            "nexus_record_batch_size": pipeline.nexus_record_batch_size,
            "reduction_inflight": pipeline.reduction_inflight,
            "semantic_checkpoint_frame_cap": pipeline.semantic_checkpoint_frame_cap,
            "staging_frame_cap": pipeline.staging_frame_cap,
        }
        texts = [label.text() for label in dialog.findChildren(QtWidgets.QLabel)]
        texts += [widget.toolTip() for widget in dialog.findChildren(QtWidgets.QWidget)]
        assert any("off by default" in text for text in texts), texts
    finally:
        dialog.deleteLater()
        qapp.processEvents()


def test_the_dialog_mounts_the_shared_advanced_dialog_and_rows_with_tooltips(qapp):
    """Hard rule 14: the dialog is the shared AdvancedDialog with FormRows, builds no spin box or check box of its
    own, gives every row a tooltip, and refuses Apply while a whole-number row holds refused text."""
    from xdart.gui.tabs.scattering.advanced_editor import AdvancedDialog
    from xdart.gui.widgets.controls_panel import FormRow
    from xrd_tools.session.run_configuration import RunIntent

    dialog = PerformanceDiagnosticsDialog()
    try:
        assert isinstance(dialog, AdvancedDialog)
        assert not dialog.findChildren(QtWidgets.QSpinBox)
        assert not dialog.findChildren(QtWidgets.QCheckBox)
        rows = dialog.findChildren(FormRow)
        assert len(rows) == 9
        assert all(row.editor.toolTip().strip() for row in rows), [row.label.text() for row in rows]
        record = next(row for row in rows if row.label.text() == "NeXus record batch")
        box = dialog.findChild(QtWidgets.QDialogButtonBox)
        attempts = []

        def try_invalid_apply():
            try:
                record.editor.setText("abc")
                box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).click()
                attempts.append((dialog.isVisible(), record.refusal()))
            finally:
                dialog.reject()

        QtCore.QTimer.singleShot(0, try_invalid_apply)
        assert dialog.edit(RunIntentStore(RunIntent()).snapshot(), 375) is None
        assert len(attempts) == 1 and all(attempts[0]), attempts
    finally:
        dialog.deleteLater()
        qapp.processEvents()


def test_loading_an_old_profile_retires_all_private_options():
    """Config ▸ Load and notebook replay both drop old session-only choices before the next Run."""
    legacy = json.loads(_SESSION_129_PROFILE)
    legacy["intent"]["run_options"][_PRIVATE[2]] = _STAGING_MARKER
    loaded = load_run_intent_profile(json.dumps(legacy))
    assert loaded.run_options == {"heavy_window": 32}


def test_saving_a_profile_excludes_all_private_options_without_mutating_the_intent():
    from xrd_tools.session.run_configuration import RunIntent

    options = json.loads(_SESSION_129_PROFILE)["intent"]["run_options"]
    options[_PRIVATE[2]] = _STAGING_MARKER.copy()
    intent = RunIntent(run_options=options)
    saved = json.loads(dump_run_intent_profile(intent))
    assert saved["intent"]["run_options"] == {"heavy_window": 32}
    assert all(key in intent.run_options for key in _PRIVATE)


def test_a_session_saved_by_129_with_the_private_options_starts_clean(qapp, tmp_path, monkeypatch):
    """A 1.2.9 session whose profile carries the private options (literal bytes) restores without them; the
    ordinary Run options it holds are kept, and the next session save carries none of them."""
    from xdart._gui_main import Main
    from xdart.utils.session import load_session, save_session

    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    save_session({"scattering_run_profile": _SESSION_129_PROFILE})
    window = Main()
    try:
        restored = window.page_handle.widget._intents.snapshot().thaw()
        assert restored.run_options == {"heavy_window": 32}
        assert restored.project_root == "/data/project"
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
    saved = load_session()["scattering_run_profile"]
    assert not any(key in saved for key in _PRIVATE), saved
    assert Path(tmp_path / "session.json").is_file()
