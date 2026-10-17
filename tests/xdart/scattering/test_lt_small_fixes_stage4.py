"""LT-SMALL-FIXES-20261014 stage 4: frozen oracle rows S1, S2, V1, H1, H2 and G1 (Claude, October 14).

S1 ENERGY-SCAN-V3-PARALLAX-OFF-REFUSAL: an energy scan with parallax turned off is reduced, frame by frame at its own
energy, through the page's real Advanced override and through the intent owner; parallax on still refuses by name.
S2 FINITE-SPEC-INVALID-LINE-SHIFT: a selected SPEC scan holding a malformed row gives no metadata to any public read,
with one warning per read, never a shifted row. V1: the promotion policy's two stale selectors collect their members,
17 in all, with its corpus pins unchanged. H1 HELP-RETIRED-ADD-PAYLOAD: a closed Phase Fitter's finished Add CIF lets
its parse go on an ordinary display refresh, the page still open. H2 HELP-HOVER-ENDPOINT-REACH: a reference line
answers a hover within 4 px of its drawn stick, not at a 4.24 px corner. G1 (maintainer, October 14, "New outputs
only"): a new constant-energy GI Run Reintegrates in 1-D and 2-D, automatic and explicit ranges.

Real pages and the real Advanced dialog, the production executor, the public metadata reader, a real pytest
collection, a real Int page's Phase Fitter, real mouse moves and real Reintegrate operations. Rows marked
preservation were green on the parent; every other row failed there.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

ROOT = Path(__file__).resolve().parents[3]
_V3_PONI = ('poni_version: 3.0\nDetector: Pilatus100k\n'
            'Detector_config: {"orientation":3,"sensor":{"material":"Si","thickness":0.00045}}\n'
            "Distance: 0.1234\nPoni1: 0.05\nPoni2: 0.06\nRot1: 0.0\nRot2: 0.0\nRot3: 0.0\n"
            "Wavelength: 1.0e-10\nParallax: PARALLAX\n")


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _press(widget):
    QtTest.QTest.mouseClick(widget, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()


def _tick(box):
    """Click a check box where a person does: on its indicator."""
    option = QtWidgets.QStyleOptionButton()
    option.initFrom(box)
    option.text = box.text()
    rect = box.style().subElementRect(QtWidgets.QStyle.SubElement.SE_CheckBoxIndicator, option, box)
    QtTest.QTest.mouseClick(box, QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier, rect.center())
    QtWidgets.QApplication.processEvents()


def _settle(seconds=.06):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        QtWidgets.QApplication.processEvents()
        time.sleep(.005)


# S1 · an energy scan with parallax turned off ----------------------------------------------------------------------

def _escan_intent(root, poni_text):
    from xrd_tools.sources import image_series_spec
    from tests.xdart.scattering.test_energy_scan_run import _series
    from tests.xdart.scattering.test_p1b_output_graph import _intent, write_poni
    root.mkdir(parents=True, exist_ok=True)
    raw = _series(root, (16, 17))
    poni = root / "cal.poni"
    if poni_text is None:
        write_poni(poni)
    else:
        poni.write_text(poni_text, encoding="utf-8")
    intent = _intent(raw, root / "escan.nexus", poni)
    intent.source_spec = image_series_spec(raw, metadata_format="txt")
    intent.bai_1d_args["unit"] = "q_A^-1"
    return intent


def _advanced_parallax(qapp, page, *, parallax):
    """The page's real Advanced dialog, as a person sets it: Override selected PONI sensor settings, Enable parallax
    checked or not, OK."""
    from xdart.gui.widgets.controls_panel import ActionButton
    from xrd_tools.session.readiness import ControlAction
    done = []

    def act():
        dialog = page._advanced_dialog
        if dialog is None or not dialog.isVisible():
            QtCore.QTimer.singleShot(20, act)
            return
        _tick(dialog.poni_v3_override)
        if dialog.poni_v3_parallax.isChecked() is not parallax:
            _tick(dialog.poni_v3_parallax)
        box = dialog.findChild(QtWidgets.QDialogButtonBox)
        _press(box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok))
        done.append(True)
    QtCore.QTimer.singleShot(0, act)
    buttons = [button for button in page.findChildren(ActionButton)
               if button.spec.action is ControlAction.ADVANCED_PROCESSING]
    assert len(buttons) == 1
    _press(buttons[0])
    assert done, 'the Advanced dialog did not open'
    override = page._intents.snapshot().thaw().poni_v3_override
    assert override is not None and override.parallax is parallax


def _escan_page(qapp, monkeypatch, root, poni_text, *, parallax, override=None):
    """Click Run on a txt-energy TIFF series (16, 17 keV), the parallax override first set in Advanced (``parallax``)
    or already on the intent (``override``, ``parallax`` None); the terminal events."""
    from xrd_tools.session.intent_store import RunIntentStore
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
    from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
    intent = _escan_intent(root, poni_text)
    intent.poni_v3_override = override
    executor = StandardRunExecutor(join_timeout=2.0)
    page = ScatteringWorkspace(intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
                               sources=FilesystemSourceAdapter(), executor=executor)
    observed = []
    real_drain = executor.drain_events

    def observe():
        events = real_drain()
        observed.extend(events)
        return events
    monkeypatch.setattr(executor, "drain_events", observe)
    terminal = {StandardEventKind.FINISHED, StandardEventKind.FAILED}
    try:
        page.show()
        if parallax is not None:
            _advanced_parallax(qapp, page, parallax=parallax)
        page.findChild(ScatteringWorkspaceShell).run_controls.startButton.click()
        until = time.monotonic() + 60
        while time.monotonic() < until and not any(event.kind in terminal for event in observed):
            qapp.processEvents()
            time.sleep(0.01)
        return [event for event in observed if event.kind in terminal]
    finally:
        for box in page.findChildren(QtWidgets.QMessageBox):
            box.close()
        page.close()
        page.deleteLater()
        qapp.processEvents()


def test_s1_preservation_parallax_on_through_the_real_advanced_override_refuses_by_name(qapp, monkeypatch, tmp_path):
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    events = _escan_page(qapp, monkeypatch, tmp_path / "on", None, parallax=True)
    assert [event.kind for event in events] == [StandardEventKind.FAILED]
    primary = events[0].primary
    assert (primary.type_module, primary.type_qualname) == ("xrd_tools.session.readiness", "EnergyScanRefused")
    assert not (tmp_path / "on" / "escan.nexus").exists()  # refused before any frame was reduced


def test_s1_parallax_off_through_the_real_advanced_override_reduces_each_frame_at_its_energy(
        qapp, monkeypatch, tmp_path):
    from tests.xdart.scattering._output_slots import written
    from tests.xdart.scattering.test_energy_scan_run import _run
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io import open_scan
    # a sensor-equipped PONI v3 with parallax on, overridden off in Advanced
    events = _escan_page(qapp, monkeypatch, tmp_path / "off", _V3_PONI.replace("PARALLAX", "True"), parallax=False)
    assert [event.kind for event in events] == [StandardEventKind.FINISHED], [event.primary for event in events]
    artifact = written(tmp_path / "off" / "escan.nexus", "Int 1D")
    selection = read_provenance(artifact)["config"]["energy_selection"]
    assert selection["frame_energies_eV"] == pytest.approx([16000., 17000.], rel=1e-15, abs=0)
    # the existing energy science: each frame exactly as the same scan without any sensor (PONI v2)
    _intent, reference = _run(tmp_path / "reference", (16, 17), "q_A^-1")
    scan, expected = open_scan(artifact), open_scan(reference)
    assert [int(label) for label in scan.frame_indices] == [int(label) for label in expected.frame_indices]
    for label in scan.frame_indices:
        np.testing.assert_array_equal(scan.get_1d(int(label)).q, expected.get_1d(int(label)).q)
        np.testing.assert_array_equal(scan.get_1d(int(label)).intensity, expected.get_1d(int(label)).intensity)


@pytest.mark.parametrize("case", ["intent override off", "PONI v3 off"])
def test_s1_the_intent_owners_parallax_off_is_admitted(qapp, monkeypatch, tmp_path, case):
    from tests.xdart.scattering._output_slots import written
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.session.run_configuration import PoniV3OverrideIntent
    root = tmp_path / "scan"
    if case == "intent override off":
        events = _escan_page(qapp, monkeypatch, root, None, parallax=None,
                             override=PoniV3OverrideIntent("Si", 450e-6, False))
    else:
        events = _escan_page(qapp, monkeypatch, root, _V3_PONI.replace("PARALLAX", "False"), parallax=None)
    assert [event.kind for event in events] == [StandardEventKind.FINISHED], [event.primary for event in events]
    selection = read_provenance(written(root / "escan.nexus", "Int 1D"))["config"]["energy_selection"]
    assert selection["frame_energies_eV"] == pytest.approx([16000., 17000.], rel=1e-15, abs=0)


# S2 · a malformed selected SPEC scan ------------------------------------------------------------------------------

def _spec_fixture(root):
    """Scan 1 with a non-number in its second row (silx skips it, so later rows would shift); scan 2 valid."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sample").write_text(
        "#F sample\n#E 1\n#D today\n#O0 th  chi\n"
        "#S 1 ascan th 0 3 3 1\n#D today\n#P0 0 0\n#N 3\n#L del  nu  I0\n0 0 10\n1 0 1x1\n2 0 12\n3 0 13\n\n"
        "#S 2 ascan th 0 2 2 1\n#D today\n#P0 0 0\n#N 3\n#L del  nu  I0\n0 0 20\n1 0 21\n2 0 22\n")
    first = [root / f"sample_scan1_{index:04d}.raw" for index in range(4)]
    second = [root / f"sample_scan2_{index:04d}.raw" for index in (0, 1, 2, 5)]
    for path in (*first, *second):
        path.write_bytes(b"")
    return first, second


def _reads():
    from xrd_tools.io.metadata import read_image_metadata, read_image_metadata_observed
    return {
        "direct": lambda path: dict(read_image_metadata(path, "spec")),
        "full": lambda path: dict(read_image_metadata(path, "spec", max_input_bytes=1 << 20)),
        "point": lambda path: dict(read_image_metadata_observed(path, "spec", max_input_bytes=1 << 20,
                                                                spec_point_snapshot=True).values),
    }


@pytest.mark.parametrize("variant", ["direct", "full", "point"])
def test_s2_a_malformed_selected_scan_gives_no_row_to_any_public_read_and_says_so_once(tmp_path, caplog, variant):
    first, _second = _spec_fixture(tmp_path)
    read = _reads()[variant]
    for path in first:
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="xrd_tools.io.metadata"):
            values = read(path)
        assert values == {}, (path.name, values)
        warnings = [record for record in caplog.records
                    if record.name == "xrd_tools.io.metadata" and record.levelno == logging.WARNING]
        assert len(warnings) == 1, (path.name, [record.getMessage() for record in warnings])


@pytest.mark.parametrize("variant", ["direct", "full", "point"])
def test_s2_preservation_valid_rows_clamp_revision_and_scratch_cleanup(tmp_path, monkeypatch, variant):
    import tempfile
    from xrd_tools.io.metadata import read_image_metadata_observed
    _first, second = _spec_fixture(tmp_path / "data")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    read = _reads()[variant]
    for path, counts in zip(second, (20., 21., 22., 22.)):  # image 5 of three rows: the last row
        values = read(path)
        assert values["I0"] == counts and values["nu"] == 0., (path.name, values)
    observed = read_image_metadata_observed(second[1], "spec", max_input_bytes=1 << 20,
                                            spec_point_snapshot=variant == "point")
    assert observed.source_path == tmp_path / "data" / "sample" and observed.source_revision is not None
    assert list(scratch.iterdir()) == []


def test_s2_preservation_finite_and_live_series_qualification_still_refuse_the_scan(tmp_path):
    from xrd_tools.io.metadata import SpecRowInvalid, read_spec_points
    first, second = _spec_fixture(tmp_path)
    assert read_spec_points(first) is None  # finite: the scan holds an invalid row
    with pytest.raises(SpecRowInvalid):
        read_spec_points(first, published=0)
    points = read_spec_points(second[:3])
    assert points is not None


# V1 · the promotion policy's selectors ----------------------------------------------------------------------------

#: the policy's corpus sections (corpus_id, files, file_groups), unchanged by the selector repair
_CORPUS_SHA256 = "afa718e00880c84f9344f39ac436381a4951ff6e88a5e17efd60e6d8a9d82b6d"


def test_v1_the_promotion_policy_collects_exactly_its_17_members_with_its_corpus_unchanged():
    policy = json.loads((ROOT / "tests" / "promotion" / "real_data_gate_v1.json").read_text())
    corpus = json.dumps({key: policy[key] for key in ("corpus_id", "files", "file_groups")}, sort_keys=True)
    assert hashlib.sha256(corpus.encode()).hexdigest() == _CORPUS_SHA256
    assert policy["expected_tests"] == 17
    if not os.environ.get("XDART_TEST_DATA"):
        pytest.skip("the members are parametrized from the corpus: XDART_TEST_DATA must name it")
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                             *policy["pytest_nodes"]], cwd=ROOT, capture_output=True, text=True, timeout=600)
    collected = [line.strip() for line in result.stdout.splitlines() if "::" in line]
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert len(collected) == 17 == len(set(collected)), collected
    counts = {node: sum(line.startswith(node) for line in collected) for node in policy["pytest_nodes"]}
    assert all(count >= 1 for count in counts.values()), counts
    explicit = [node for node in policy["pytest_nodes"] if "test_e2sci_saturation_mask.py" in node]
    assert [counts[node] for node in explicit] == [4, 5], counts


# H1 · a closed Phase Fitter's finished Add CIF ---------------------------------------------------------------------

def test_h1_a_closed_fitters_finished_add_cif_lets_its_parse_go_on_an_ordinary_refresh(qapp, tmp_path, monkeypatch):
    import xdart.gui.analysis.phase_source as phase_source
    from tests.xdart.scattering.test_phase_reference_overlays import _cif, _navigation
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))
    started, gate = threading.Event(), threading.Event()
    real = phase_source._add_cifs

    def held(request, cancel, progress):
        started.set()
        gate.wait(30)
        return real(request, cancel, progress)
    monkeypatch.setattr(phase_source, "_add_cifs", held)
    try:
        with run_page(qapp, tmp_path / "int", energies=(16, 16)) as (page, view):
            page._shell.commandRequested.emit(ShellCommand(ShellCommandKind.ANALYSIS_ACTION, "phase_fit"))
            qapp.processEvents()
            dialog = page._phase_dialog
            assert dialog is not None and dialog.isVisible()
            cif = _cif(tmp_path, "phaseA", 4.1)
            monkeypatch.setattr(phase_source, "browse", lambda *args, **kwargs: [str(cif)])
            _press(dialog.phase_source.add_cif_btn)
            assert started.wait(10)
            worker = dialog.phase_source.add_worker
            dialog.close()
            qapp.processEvents()
            assert view._retired_phase_workers == [worker] and worker.busy  # still reading: kept, as before
            gate.set()
            thread = worker._worker  # the parse's thread: the deleted Fitter no longer polls its owner
            until = time.monotonic() + 20
            while thread.is_alive() and time.monotonic() < until:
                qapp.processEvents()
                time.sleep(0.01)
            assert not thread.is_alive() and view._retired_phase_workers == [worker]
            # an ordinary refresh of the retained display, the page still open
            state, navigation = page._last_scientific_projection, _navigation(page)
            view.reconcile(replace(state, heavy=None, retain_display=True), navigation, completed=0, total=0,
                           detail='')
            assert view._retired_phase_workers == [] and worker._terminal is None
    finally:
        gate.set()


# H2 · a reference line's hover reach ------------------------------------------------------------------------------

def test_h2_a_reference_line_answers_a_hover_within_four_pixels_of_its_drawn_stick(qapp):
    import pyqtgraph as pg
    from xdart.gui.tabs.scattering.scientific_curves import draw_reference_lines, reference_entry
    from xdart.gui.widgets.scientific_plot import PlotCursorReadout
    from xrd_tools.analysis.phase_overlays import PhaseLineValue
    widget = pg.PlotWidget()
    try:
        widget.resize(640, 420)
        widget.show()
        readout = PlotCursorReadout(widget)
        value = PhaseLineValue(q=2.0, d_spacing=3.1416, intensity=50., hkls=((1, 1, 1),))
        item = draw_reference_lines(widget, (2.0,), color=(10, 20, 30), family="phase:x", values=(value,))
        widget.setXRange(0., 4., padding=0)
        widget.setYRange(0., 10., padding=0)
        _settle(.1)
        expected = reference_entry(value, item.color)
        top = widget.mapFromScene(widget.getViewBox().mapViewToScene(QtCore.QPointF(2.0, 5.0)))  # the stick's top

        def hover(dx, dy):
            QtTest.QTest.mouseMove(widget.viewport(), QtCore.QPoint(top.x() + dx, top.y() + dy))
            _settle(.08)  # the readout's rate-limited proxy
            return readout.hover.text
        assert hover(0, -3) == expected  # 3 px above the top (preservation)
        assert hover(3, 60) == expected  # 3 px beside the stick (preservation)
        assert hover(3, -3) == ""        # 4.24 px from the top: beyond the 4 px reach
        assert hover(6, 60) == ""        # 6 px beside it (preservation)
    finally:
        widget.deleteLater()  # the Qt test boundary retires (closes) this top level once
        qapp.processEvents()


# G1 · a new constant-energy GI Run Reintegrates -------------------------------------------------------------------

@pytest.mark.parametrize("radial", [None, (1.0, 4.0)], ids=["automatic", "explicit"])
def test_g1_a_new_gi_run_reintegrates_both_dimensions_and_again(tmp_path, radial):
    from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _intent, _run_to_terminal, _write_tiff
    from tests.xdart.scattering.test_p1b_output_graph import write_poni
    from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xdart.gui.tabs.scattering.page import ScatteringWorkspace
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io import open_scan
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
    raw = tmp_path / "sample_0001.tif"
    _write_tiff(raw, 50)
    _write_tiff(tmp_path / "sample_0002.tif", 60)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    intent = _intent(raw, tmp_path / "out", poni, processing_mode="Int 2D")
    intent.gi.enabled, intent.gi.incidence_motor, intent.gi.th_val = True, "Manual", 0.2  # as the GI controls set it
    if radial is not None:
        intent.bai_1d_args["radial_range"] = radial
    executor, identity, events = _run_to_terminal(intent, request_value=9611)
    try:
        terminal = next(event for event in events if event.kind in _TERMINAL)
        assert terminal.kind is StandardEventKind.FINISHED, terminal.primary
    finally:
        executor.close(identity)
    artifact = next((tmp_path / "out").rglob("*.nexus"))
    config = read_provenance(artifact)["config"]
    run = config["run_configuration"]
    # the settings the Run used: its GI modes and the ranges its GI freeze fixed; a requested range as requested
    assert (run["bai_1d_args"]["gi_mode_1d"], run["bai_2d_args"]["gi_mode_2d"]) == ("q_total", "qip_qoop")
    assert run["bai_1d_args"]["radial_range"] == (list(radial) if radial else config["bai_1d_args"]["radial_range"])
    for key in ("x_range", "y_range"):
        assert run["bai_2d_args"][key] == config["bai_2d_args"][key]
    scan = open_scan(artifact)
    labels = tuple(int(label) for label in scan.frame_indices)
    # the Run's own values, copied before any Reintegrate writes
    one_d = {label: (np.array(scan.get_1d(label).q), np.array(scan.get_1d(label).intensity)) for label in labels}
    two_d = {label: np.array(scan.get_2d(label).intensity) for label in labels}

    def reintegrate(target, dimension):
        plan = ReintegrateSuccessorPlan.from_artifact(
            target, entry="entry", dimension=dimension, source_root=str(tmp_path),
            expected_target_snapshot=capture_target_snapshot(target), expected_labels=labels,
            preparation=ScatteringWorkspace._reintegrate_preparation(intent, dimension))
        result = run_reintegrate_successor(plan)
        assert result.disposition == "COMMITTED", result
        return Path(result.output_artifact)

    def same_1d(path):
        output = open_scan(path)
        for label in labels:
            np.testing.assert_array_equal(output.get_1d(label).q, one_d[label][0])
            np.testing.assert_array_equal(output.get_1d(label).intensity, one_d[label][1])
    first = reintegrate(artifact, "1d")
    same_1d(first)
    output = open_scan(reintegrate(artifact, "2d"))
    for label in labels:
        np.testing.assert_array_equal(np.asarray(output.get_2d(label).intensity), two_d[label])
    # Reintegrating the Run's output again replaces its 1-D slot with the same values (a Reintegrate output itself is
    # not reintegrated: stable operation slots refuse chaining, ADR-0010)
    same_1d(reintegrate(artifact, "1d"))
