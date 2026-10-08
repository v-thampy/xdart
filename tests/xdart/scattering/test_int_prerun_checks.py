"""P2 frozen oracle: real Int readiness, construction and notice bindings."""

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_gi_motorless_source import _close, _wait
from tests.xdart.scattering.test_p1b_output_graph import _intent, _write_tiff, _written
from xdart.gui.pages.scattering_workspace import _WorkspaceRunControl
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.controls_inventory import MASK_FILE, PONI_FILE, SOURCE_FILE
from xdart.gui.tabs.scattering.controls_projection import GI_MOTOR
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.core.energy import energy_eV_to_wavelength_m
from xrd_tools.core.provenance import read_provenance
from xrd_tools.io import open_scan
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.sources.selection import image_series_spec


ENERGY = "Using the scan energy, 16.000 keV; the PONI says 12.398 keV."
AUTO = "GI Auto selected incidence motor eta."


def _page(root, *, mode="Int 1D", missing=False, energy=None, gi=False,
          motor="eta", initial_motor="Manual", wavelength=1e-10, unit="q_A^-1"):
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "raw_0001.tif"
    _write_tiff(raw, 1)
    has_metadata = energy is not None or (gi and motor is not None)
    if has_metadata:
        motors = [f"{motor} = 0.2"] if gi and motor else []
        if energy is not None:
            motors.append(f"energy = {energy}")
        raw.with_suffix(".txt").write_text(
            "# Counters\nI0 = 1\n# Motors\n" + ", ".join(motors) + "\n"
            "User: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n")
    poni = root / "cal.poni"
    if not missing:
        write_poni(poni)
        if wavelength != 1e-10:
            poni.write_text(poni.read_text().replace("1.0e-10", repr(wavelength)))
    intent = _intent(raw, root / "result.nexus", poni, processing_mode=mode)
    intent.source_spec = image_series_spec(raw, metadata_format="txt" if has_metadata else None)
    intent.gi.enabled = gi
    intent.gi.incidence_motor = initial_motor
    intent.gi.th_val = 0.37
    intent.bai_1d_args["unit"] = unit
    page = ScatteringWorkspace(
        intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(), executor=StandardRunExecutor(join_timeout=2),
    )
    page.resize(1280, 900)
    page.show()
    return page, poni, _written(root / "result.nexus", mode)


def _edit(page, path, value):
    page._handle_shell_command(ShellCommand(ShellCommandKind.CONTROL_EDIT, path=path, value=value))


def _activate(page):
    QtWidgets.QApplication.sendEvent(page, QtCore.QEvent(QtCore.QEvent.WindowActivate))


def _record_notices(page):
    notices = []
    def record(text):
        label = page._shell.run_controls.readinessLabel
        notices.append((text, label.text(), label.toolTip()))
    page.noticeChanged.connect(record)
    return notices


def _run(page, app):
    _wait(app, lambda: page._source_selection.observation is not None)
    page._shell.run_controls.startButton.click()
    _wait(app, lambda: page._progress.terminal
          or page._notice_text.startswith("Output admission failed:"))


def _brief(page, notices, clause):
    matches = [row for row in notices if clause in row[0]]
    assert len(matches) == 1, notices
    assert clause in matches[0][2], matches
    assert clause not in page._shell.run_controls.readinessLabel.toolTip()
    assert clause not in page._shell.scientific.status.text()
    assert clause not in page._notice_text


@pytest.mark.parametrize("mode", ("Int 1D", "Int 2D", "Int 1D (XYE)"))
def test_int_run_requires_current_readable_poni(tmp_path, monkeypatch, mode):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page, poni, artifact = _page(tmp_path, mode=mode, missing=True)
    try:
        _wait(app, lambda: page._source_selection.observation is not None)
        assert not page._shell.run_controls.startButton.isEnabled()
        label = page._shell.run_controls.readinessLabel
        assert "PONI" in label.toolTip()
        assert page._shell.controls.experiment_card.valid_marker.isHidden()
        assert "readable" in page._shell.controls.experiment_card.valid_marker.toolTip()

        write_poni(poni)
        original = poni.read_bytes()
        _activate(page)  # Real idle activation refreshes a restored same-path asset.
        assert page._shell.run_controls.startButton.isEnabled()
        assert not page._shell.controls.experiment_card.valid_marker.isHidden()

        # Observe the existing real stat/parse owner; never replace its result.
        from xdart.gui.tabs.scattering import detector_projection
        checks = []
        owner = detector_projection._detector_projection
        def checked(poni_file, mask_file):
            checks.append(poni_file)
            return owner(poni_file, mask_file)
        monkeypatch.setattr(detector_projection, "_detector_projection", checked)
        for _ in range(4):
            page._drain_executor()
            page._refresh_shell()
        assert not checks, "heartbeat/projection re-probed PONI"

        for mutation in ("delete", "malformed", "directory"):
            if poni.is_dir():
                poni.rmdir()
            elif poni.exists():
                poni.unlink()
            if mutation == "malformed":
                poni.write_text("not a calibration\n")
            elif mutation == "directory":
                poni.mkdir()
            checks.clear()
            _WorkspaceRunControl(page).run_pause()  # Real host/shortcut route.
            assert checks, "Run entry did not revalidate PONI"
            assert not artifact.exists()
            assert not page._shell.run_controls.startButton.isEnabled()
            assert "PONI" in label.toolTip()
            assert page._shell.controls.experiment_card.valid_marker.isHidden()
            if poni.is_dir():
                poni.rmdir()
            poni.write_bytes(original)
            _activate(page)
            assert page._shell.run_controls.startButton.isEnabled()

        permissions = poni.stat().st_mode & 0o777
        try:
            poni.chmod(0)
            _WorkspaceRunControl(page).run_pause()
            assert not page._shell.run_controls.startButton.isEnabled()
            assert not artifact.exists()
        finally:
            poni.chmod(permissions)
        _activate(page)
        assert page._shell.run_controls.startButton.isEnabled()

        _edit(page, PONI_FILE, "")
        assert "Needs PONI" in label.toolTip()
        _edit(page, PONI_FILE, str(poni))  # Real field edit refresh.
        assert page._shell.run_controls.startButton.isEnabled()
        _edit(page, MASK_FILE, str(tmp_path / "absent-mask.edf"))
        assert page._shell.run_controls.startButton.isEnabled()
        assert "mask" not in label.toolTip().lower()
    finally:
        _close(page, app)


@pytest.mark.parametrize("unit", ("q_A^-1", "2th_deg"))
def test_int_energy_notice_is_brief_and_keeps_native_selection(tmp_path, monkeypatch, unit):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page, poni, artifact = _page(tmp_path / "mismatch", energy=16000, unit=unit)
    from xrd_tools.sources import image
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    reads = []
    native_read = image.TiffSeriesSource._read_path
    metadata_read = image.read_image_metadata_observed
    def read_image(source, path):
        reads.append("image")
        return native_read(source, path)
    def read_metadata(*args, **kwargs):
        reads.append("metadata")
        return metadata_read(*args, **kwargs)
    monkeypatch.setattr(image.TiffSeriesSource, "_read_path", read_image)
    monkeypatch.setattr(image, "read_image_metadata_observed", read_metadata)
    delivered = []
    drain = page._run_executor.drain_events
    def observed_drain():
        events = drain()
        delivered.extend(events)
        return events
    monkeypatch.setattr(page._run_executor, "drain_events", observed_drain)
    before = poni.read_bytes()
    notices = _record_notices(page)
    try:
        _wait(app, lambda: page._source_selection.observation is not None)
        reads.clear()
        _run(page, app)
        assert len([row for row in notices if ENERGY in row[0]]) == 1, notices
        _brief(page, notices, ENERGY)
        assert poni.read_bytes() == before
        selection = read_provenance(artifact)["config"]["energy_selection"]
        assert selection["source"] == "metadata"
        assert selection["disagreement"] is True
        assert selection["energy_eV"] == 16000
        assert selection["wavelength_m"] == energy_eV_to_wavelength_m(16000)
        selected = open_scan(artifact)
        assert selected.metadata["wavelength_A"] == pytest.approx(selection["wavelength_m"] * 1e10)
        result = selected.get_1d(1)
        axis, intensity = result.q.copy(), result.intensity.copy()
        mismatch_reads = tuple(reads)
        actual_context = next(event for event in delivered if event.kind is StandardEventKind.CONTEXT_READY)
        page._notice("Current error is retained")
        page._run_executor._events.put(actual_context)  # Actual old run event, now stale.
        page._drain_executor()
        assert page._notice_text == "Current error is retained"
        assert len([row for row in notices if ENERGY in row[0]]) == 1
    finally:
        _close(page, app)

    # Same science with a matching effective PONI: axes/intensities are identical,
    # but no warning. Missing and within-threshold metadata also remain silent.
    for label, energy, wavelength in (
        ("matching", 16000, energy_eV_to_wavelength_m(16000)),
        ("missing", None, 1e-10),
        ("within", 12400, 1e-10),
    ):
        page, _, target = _page(tmp_path / label, energy=energy, wavelength=wavelength, unit=unit)
        reads.clear()
        notices = _record_notices(page)
        try:
            _wait(app, lambda: page._source_selection.observation is not None)
            reads.clear()
            _run(page, app)
            assert not [row for row in notices if "Using the scan energy" in row[0]]
            if label == "matching":
                match = open_scan(target).get_1d(1)
                np.testing.assert_array_equal(match.q, axis)
                np.testing.assert_array_equal(match.intensity, intensity)
                assert tuple(reads) == mismatch_reads, "mismatch notice added scientific reads"
        finally:
            _close(page, app)

    page, _, target = _page(tmp_path / "invalid", energy=-16000, unit=unit)
    notices = _record_notices(page)
    try:
        _wait(app, lambda: page._source_selection.observation is not None)
        page._shell.run_controls.startButton.click()
        _wait(app, lambda: page._notice_text.startswith("Output admission failed:")
              or page._lifecycle.phase.name == "FAILED")
        assert not target.exists()
        assert not [row for row in notices if "Using the scan energy" in row[0]]
    finally:
        _close(page, app)


def test_int_gi_auto_notice_names_only_the_owned_launch_choice(tmp_path, monkeypatch):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page, _, artifact = _page(tmp_path / "auto", gi=True, mode="Int 2D", energy=16000)
    notices = _record_notices(page)
    try:
        # Initial observation can precede the real motor preview. Exercise an
        # actual owner-selected Auto launch, rather than racing a Manual Run.
        _wait(app, lambda: page._intents.snapshot().thaw().gi.incidence_motor == "eta")
        _run(page, app)
        assert len([row for row in notices if AUTO in row[0]]) == 1, notices
        _brief(page, notices, AUTO)
        _brief(page, notices, ENERGY)
        assert len([row for row in notices if AUTO in row[0] and ENERGY in row[0]]) == 1
        saved = read_provenance(artifact)["config"]["run_configuration"]
        assert saved["gi"]["incidence_motor"] == "eta"
    finally:
        _close(page, app)

    for label, motor, initial, edit in (
        ("explicit", "eta", "eta", None),
        ("manual", "eta", "Manual", "Manual"),
        ("motorless", None, "Manual", None),
        ("replaced", "eta", "Manual", "eta"),
    ):
        page, _, artifact = _page(tmp_path / label, gi=True, mode="Int 2D",
                                  motor=motor, initial_motor=initial)
        notices = _record_notices(page)
        try:
            _wait(app, lambda: page._source_selection.observation is not None)
            if edit is not None:
                _edit(page, GI_MOTOR, edit)  # Real owner edit invalidates automatic provenance.
            _run(page, app)
            assert not [row for row in notices if "GI Auto selected" in row[0]]
            saved = read_provenance(artifact)["config"]["run_configuration"]["gi"]
            assert saved["incidence_motor"] == (edit or initial)
            assert saved["th_val"] == 0.37
        finally:
            _close(page, app)

    # The real focused source editor is captured at Run, before its normal
    # editingFinished signal. An old source's automatic eta must not leak.
    from tests.xdart.scattering.test_e4_run_action_focused_edit import _row, _focus
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    page, _, artifact = _page(tmp_path / "focused", gi=True, mode="Int 2D")
    notices = _record_notices(page)
    frozen = []
    drain = page._run_executor.drain_events

    def capture_launch():
        events = drain()
        for event in events:
            if event.kind is StandardEventKind.CONTEXT_READY:
                context = page._run_executor.acquisition_context(event.run_identity)
                if context is not None:
                    frozen.append(context.run_configuration)
        return events

    monkeypatch.setattr(page._run_executor, "drain_events", capture_launch)
    try:
        _wait(app, lambda: page._intents.snapshot().thaw().gi.incidence_motor == "eta")
        replacement = tmp_path / "focused" / "replacement.tif"
        _write_tiff(replacement, 1)
        replacement.with_suffix(".txt").write_text(
            "# Counters\nI0 = 1\n# Motors\nomega = 0.2\n"
            "User: energy, time: Mon Jan 15 10:30:00 2024  # Temp\n")
        row = _row(page, SOURCE_FILE)
        _focus(page, row.editor, app)
        row.editor.setText(str(replacement))
        row.editor.textEdited.emit(str(replacement))
        _run(page, app)
        assert not [row for row in notices if AUTO in row[0]]
        saved = read_provenance(artifact)["config"]["run_configuration"]["gi"]
        assert saved["incidence_motor"] == frozen[0].gi.incidence_motor
        assert saved["th_val"] == 0.37
    finally:
        _close(page, app)
