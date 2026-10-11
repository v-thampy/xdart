"""Direct shared-owner checks for disabled phase help and refused floor drafts."""

import pytest
from pyqtgraph.Qt import QtWidgets

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


@pytest.mark.parametrize("draft", ("", "nan", "inf", "-1", "101"))
def test_plot_options_keep_invalid_phase_floor_open_with_reason_then_accept_valid_edit(qapp, draft):
    dialog = PlotOptionsDialog()
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(dialog.edited_options()))
    original = dialog.projection
    try:
        dialog.show()
        dialog.phase_floor.editor.setText(draft)
        dialog.accept_button.click()
        assert dialog.isVisible() and accepted == []
        assert dialog.projection is original
        assert dialog.phase_floor.editor.text() == draft
        reason = dialog.phase_floor.editor.toolTip().lower()
        assert "finite" in reason and "0" in reason and "100" in reason
        assert dialog.phase_floor.label.toolTip() == dialog.phase_floor.editor.toolTip()
        dialog.phase_floor.editor.setText("12.5")
        dialog.accept_button.click()
        assert not dialog.isVisible() and len(accepted) == 1
        assert accepted[0].phase_min_intensity == 12.5
        assert "finite" not in dialog.phase_floor.editor.toolTip().lower()
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("floor", ("0", "100"))
def test_plot_options_accept_both_phase_floor_boundaries(qapp, floor):
    dialog = PlotOptionsDialog()
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(dialog.edited_options()))
    try:
        dialog.show()
        dialog.phase_floor.editor.setText(floor)
        dialog.accept_button.click()
        assert not dialog.isVisible() and len(accepted) == 1
        assert accepted[0].phase_min_intensity == float(floor)
    finally:
        dialog.close()
        dialog.deleteLater()
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
    dialog = PlotOptionsDialog()
    try:
        options = replace(dialog.projection, phase_references=((cif.id, True), (lattice.id, True)))
        dialog.reconcile(options)
        assert tuple(entry.id for entry in dialog.phase_source.selected_entries()) == (cif.id, lattice.id)
        reads = []
        for name in ("resolve", "entries"):
            actual = getattr(PhaseLibrary, name)
            monkeypatch.setattr(PhaseLibrary, name, lambda self, *args, _actual=actual, _name=name, **kwargs:
                                (reads.append(_name), _actual(self, *args, **kwargs))[1])
        for _ in range(3):
            dialog.reconcile(options)
            dialog.set_phase_available(True)
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
