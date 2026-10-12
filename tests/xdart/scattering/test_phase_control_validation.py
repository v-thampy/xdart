"""Direct shared-owner checks for disabled phase help and refused floor drafts.

REFERENCE-PHASES-20261011 successors (declared in its kickoff): the floor rows moved from Plot Options to the
Reference Phases window, whose global minimum is a decimal spinbox that cannot commit an invalid value.
"""
import math

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from xdart.gui.analysis.phase_source import PhaseSourceWidget
from xdart.gui.tabs.scattering.scientific_plot_options import PlotOptionsDialog
from xrd_tools.analysis.phase_references import PhaseLibrary


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_disabled_phase_reason_survives_resolution_on_every_shared_control(qapp, tmp_path):
    source = PhaseSourceWidget(library=PhaseLibrary(tmp_path / "library"))
    controls = [source, source.browser, source.browser.scans, source.browser.scans_label,
                source.browser.show_all, source.browser.metadata, source.browser.auto_last,
                source.browser.date_sort, source.browser.refresh, source.add_cif_btn,
                source.add_lattice_btn, source.remove_btn, source.lattice_form,
                source.add_lattice_commit, source.status]
    for row in source.lattice_rows.values():
        controls.extend((row, row.label, row.editor))
    purposes = {control: control.toolTip() for control in controls}
    reason = "Phase sources need pymatgen; install xdart[fitting]."
    try:
        source.set_available(False, reason)
        assert source.selected_entries() == ()
        for control in controls:
            assert reason in control.toolTip(), (control.objectName(), control.toolTip())
            assert purposes[control] in control.toolTip()
        assert not source.isEnabled()
        source.set_available(True)
        assert source.selected_entries() == ()
        for control in controls:
            assert control.toolTip() == purposes[control]
    finally:
        source.close()
        source.deleteLater()
        qapp.processEvents()


def _cutoff_source(tmp_path):
    source = PhaseSourceWidget(library=PhaseLibrary(tmp_path / "library"), checkable=True, show_library=False,
                               cutoffs=True)
    source.show()
    return source


def _type(qapp, editor, text):
    editor.window().activateWindow()
    editor.setFocus()
    qapp.processEvents()
    editor.selectAll()
    QtTest.QTest.keyClicks(editor, text)
    QtTest.QTest.keyClick(editor, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()


@pytest.mark.parametrize("draft", ("", "nan", "inf", "-1", "101"))
def test_reference_minimum_cannot_commit_an_invalid_draft_then_commits_a_valid_edit(qapp, tmp_path, draft):
    source = _cutoff_source(tmp_path)
    changes = []
    source.cutoffsChanged.connect(changes.append)
    try:
        editor = source.global_row.editor
        tip = editor.toolTip().lower()
        assert "0" in tip and "100" in tip, tip
        _type(qapp, editor, draft)
        values = [change.global_value for change in changes] + [source.cutoffs.global_value]
        assert all(math.isfinite(value) and 0. <= value <= 100. for value in values), values
        _type(qapp, editor, "12.5")
        assert source.cutoffs.global_value == 12.5 and changes[-1].global_value == 12.5
    finally:
        source.close()
        source.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("floor", ("0", "100"))
def test_reference_minimum_accepts_both_boundaries(qapp, tmp_path, floor):
    source = _cutoff_source(tmp_path)
    changes = []
    source.cutoffsChanged.connect(changes.append)
    try:
        _type(qapp, source.global_row.editor, floor)
        assert source.cutoffs.global_value == float(floor) and changes[-1].global_value == float(floor)
    finally:
        source.close()
        source.deleteLater()
        qapp.processEvents()


def test_reconciling_plot_options_reads_no_phase_library_files(qapp, tmp_path, monkeypatch):
    """A selection resolves once; every later projection reconcile reuses it."""
    from dataclasses import replace
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    library = PhaseLibrary(tmp_path / "library")
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(library.path))
    path = tmp_path / "CsCl.cif"
    path.write_text(str(CifWriter(Structure(Lattice.cubic(4.1), ["Cs", "Cl"], [[0, 0, 0], [.5, .5, .5]]))))
    cif = library.add_cif(path, name="CsCl")
    lattice = library.add_lattice("cell", a=4.2, space_group="Pm-3m")
    from xdart.gui.tabs.scattering.reference_phases import ReferencePhasesWindow
    from xdart.gui.tabs.scattering.shell_values import ScientificPlotOptions
    dialog = ReferencePhasesWindow()
    try:
        options = replace(ScientificPlotOptions(), phase_references=((cif.id, True), (lattice.id, True)))
        dialog.reconcile(options)
        assert tuple(entry.id for entry in dialog.phase_source.selected_entries()) == (cif.id, lattice.id)
        reads = []
        for name in ("resolve", "entries"):
            actual = getattr(PhaseLibrary, name)
            monkeypatch.setattr(PhaseLibrary, name, lambda self, *args, _actual=actual, _name=name, **kwargs:
                                (reads.append(_name), _actual(self, *args, **kwargs))[1])
        for _ in range(3):
            dialog.reconcile(options)
            dialog.set_available(True)
            assert tuple(entry.id for entry in dialog.phase_source.selected_entries()) == (cif.id, lattice.id)
        assert reads == [], "a projection reconcile must not read the phase library on the GUI thread"
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_the_int_workspace_starts_without_importing_pymatgen(tmp_path):
    """Phase availability is probed without loading the optional backend on the GUI thread."""
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    code = textwrap.dedent("""
        import os, sys
        from pyqtgraph.Qt import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from xdart.gui.pages.scattering_workspace import build_scattering_workspace
        from xdart.gui.pages.services import empty_host_services
        from tests.xdart.pages.test_stitch_page import _Status
        handle = build_scattering_workspace(empty_host_services(_Status()), None)
        app.processEvents()
        print("pymatgen" if "pymatgen" in sys.modules else "clean", flush=True)
        handle.close()
        os._exit(0)
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", XDART_PHASE_LIBRARY=str(tmp_path / "library"),
               XDART_SESSION_FILE=str(tmp_path / "session.json"))
    env["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root), env.get("PYTHONPATH", "")))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          cwd=root, timeout=300)
    assert proc.stdout.strip().splitlines()[-1:] == ["clean"], (proc.stdout, proc.stderr[-2000:])


def test_okay_and_cancel_stay_the_last_plot_options_row(qapp):
    """Every host row is built above the dialog's buttons; reference phases live in their own window now."""
    dialog = PlotOptionsDialog()
    try:
        dialog.add_choice("Host choice", ("a", "b"), "a", tooltip="A host's own choice.")
        dialog.mount_display_scale(QtWidgets.QWidget())
        layout = dialog.layout()
        last = layout.itemAt(layout.count() - 1).layout()
        assert last is not None and last.indexOf(dialog.accept_button) >= 0, "Okay/Cancel must stay last"
        assert not [group for group in dialog.findChildren(QtWidgets.QGroupBox)
                    if group.title() == "Reference phases"]
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_each_phase_legend_entry_explains_its_family(qapp, tmp_path, monkeypatch):
    """The visible legend sample and label carry the family's tooltip, not only a hidden proxy."""
    from tests.xdart.scattering.test_predicted_ring_overlays import run_page
    from tests.xdart.scattering.test_phase_reference_overlays import _edit, _entries
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(tmp_path / "phase-library"))
    with run_page(qapp, tmp_path / "legend", energies=(16, 16)) as (_page, view):
        entry, = _entries(tmp_path)
        _edit(qapp, view, (entry,))
        shown = [(sample, label) for sample, label in view.curve.getPlotItem().legend.items
                 if label.text == entry.name]
        assert shown, "the phase must have a legend entry"
        for sample, label in shown:
            for item in (sample, label):
                assert entry.name in item.toolTip() and "omitted" in item.toolTip().lower(), item.toolTip()


def test_reference_lines_are_one_passive_item_that_follows_the_view(qapp):
    """The shared joined item is never trace data, and its lines span every pan, zoom and log scale."""
    import numpy as np
    import pyqtgraph as pg
    from xdart.gui.tabs.scattering.scientific_curves import draw_reference_lines
    widget = pg.PlotWidget()
    try:
        widget.show()
        trace = widget.plot([1., 2., 3.], [10., 100., 1000.])
        item = draw_reference_lines(widget, (1.5, 2.5), color=(10, 20, 30), family="phase:x", tooltip="tip")
        assert widget.getPlotItem().listDataItems() == [trace], "reference lines must not join the traces"
        assert item.data(0) == "phase:x" and item.reference_positions == (1.5, 2.5)
        assert item.toolTip() == item.lines.toolTip() == "tip"
        for log, span in ((False, (-50., 2000.)), (True, (1., 3.))):
            widget.getPlotItem().setLogMode(False, log)
            widget.setYRange(*span, padding=0)
            qapp.processEvents()
            low, high = widget.getViewBox().viewRange()[1]
            x, y = item.lines.getData()
            np.testing.assert_allclose(x[~np.isnan(x)], (1.5, 1.5, 2.5, 2.5))
            np.testing.assert_allclose(y[~np.isnan(y)], (low, high, low, high))
    finally:
        widget.deleteLater()  # the Qt test boundary retires (closes) this top level once
