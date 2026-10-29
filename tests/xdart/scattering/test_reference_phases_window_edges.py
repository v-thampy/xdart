"""REFERENCE-PHASES-20261011 self-review rows SR1-SR3 (Claude, October 11), added after the frozen oracle.

Claude's pre-handback adversarial pass found three defects that no frozen row drove: the window built its own label
for the rings notice (Rule 14), the window outlived its page's shutdown, and a CIF minimum edit rebuilt calibrant and
lattice families that no minimum applies to. The original three rows fail on 0b8b6c01 and pass with the fix.
The maintainer requested the additional SR2 Stitch shutdown preservation row before integration.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering.test_phase_reference_overlays import _cif, _settle, _stitch
from tests.xdart.scattering.test_predicted_ring_overlays import run_page
from tests.xdart.scattering.test_reference_phases_window import (LAB6, _ids, _marker, _options, _pane_items, _same,
                                                                 _type, _window)


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_library(tmp_path, monkeypatch):
    monkeypatch.setenv('XDART_PHASE_LIBRARY', str(tmp_path / 'phase-library'))


def test_sr1_the_rings_notice_is_the_shared_sources_note_line(qapp):
    """PHASE-ANALYSIS-128: the binding (the retired window's code) shows the view's rings notice on its source's note
    line and builds no display of its own."""
    from xdart.gui.analysis.phase_source import PhaseSourceWidget
    from xdart.gui.tabs.scattering.reference_phases import RINGS_TOOLTIP, ReferencePhaseBinding
    source = PhaseSourceWidget(checkable=True, show_library=False, calibrants=True, cutoffs=True,
                               status_visible=False)
    binding = ReferencePhaseBinding(source)
    try:
        assert not binding.findChildren(QtWidgets.QWidget), 'the binding builds no display of its own'
        assert binding.rings_note is source.note
        source.show()
        qapp.processEvents()
        assert source.note.isHidden() and source.note.toolTip() == RINGS_TOOLTIP
        binding.set_rings_reason('Rings need a Run frame.')
        assert source.note.isVisibleTo(source) and source.note.text() == 'Rings need a Run frame.'
        assert source.note.toolTip() == RINGS_TOOLTIP + ' Rings need a Run frame.'
        binding.set_rings_reason('')
        assert source.note.isHidden() and not source.note.text()
    finally:
        source.close()
        source.deleteLater()
        qapp.processEvents()


def test_sr2_closing_the_page_closes_its_reference_phases_window(qapp, tmp_path):
    from xdart.gui.tabs.scattering.events import CleanupStatus
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        window = _window(qapp, view)
        window.phase_source.add_calibrant('LaB6')
        _settle(qapp, view)
        assert page.close_workspace().cleanup_status is CleanupStatus.CLEANED
        qapp.processEvents()
        assert page._phase_dialog is None and view.phase_list is None, 'the window does not outlive its page'


def test_sr2_stitch_shutdown_closes_its_reference_phases_window(qapp, tmp_path):
    from xdart.gui.pages.values import PageCleanup
    with _stitch(qapp, tmp_path / 'stitch') as (page, view, _payload):
        window = _window(qapp, view)
        window.phase_source.add_calibrant('LaB6')
        _settle(qapp, view)
        assert page.shutdown().status is PageCleanup.CLEAN
        qapp.processEvents()
        assert not window.isVisible() and view.phase_list is None, 'the window does not outlive its Stitch page'


def test_sr3_a_cif_minimum_edit_leaves_calibrant_and_lattice_families_identical(qapp, tmp_path):
    with run_page(qapp, tmp_path / 'int', energies=(16, 16)) as (page, view):
        source = _window(qapp, view).phase_source
        source.add_cif_paths((str(_cif(tmp_path, 'phaseA', 4.1)),))
        source.add_calibrant('LaB6')
        lattice = source.add_lattice('cell', a=4.1, space_group=221)
        _settle(qapp, view)
        cif = _ids(source)[0]
        families = dict(view.phase_families)
        markers = {identifier: _marker(view, identifier) for identifier in (cif, LAB6, lattice.id)}
        rings = _pane_items(view.raw, LAB6)
        assert all(markers.values()) and rings and set(families) == set(markers)
        _type(qapp, source.global_row.editor, '3')
        _settle(qapp, view)
        assert _options(page).phase_min_intensity == 3. and _options(page).phase_global_floor is True
        assert view.phase_families[cif] is not families[cif], 'the CIF phase is re-cut'
        for identifier in (LAB6, lattice.id):
            assert view.phase_families[identifier] is families[identifier], identifier
            assert _marker(view, identifier) is markers[identifier], identifier
        assert _same(_pane_items(view.raw, LAB6), rings)
