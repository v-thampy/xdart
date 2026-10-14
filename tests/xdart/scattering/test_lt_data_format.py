"""LT-SMALL-FIXES-20261014 stage 2: frozen oracle rows F1–F4 (Claude, October 14).

LT-004–LT-006 under the maintainer's decisions: one Data format per page (D1 (b): Stitch and RSM each remember their
own), SPEC + Image Series / SPEC + Eiger / NeXus, driving that page's Source rows, its calibration window's
calibrant and direct-beam sources and Quick; a deterministic legacy default (D2) with conflicts left to the user;
format-agnostic Directory discovery and unchanged source identity (D3); Int unchanged. Real pages, the real
calibration window each page opens, real settings documents and the real shared composer. A calibration job is
held only by delaying the window's real job dispatcher, never by faking its busy state.
"""
import dataclasses
import threading
import time

import pytest
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets

from tests.core.test_spec_source import _spec as _spec_file
from tests.xdart.scattering.test_calibration_quick_chain import _close
from tests.xdart.scattering.test_stitch_calibration_fit_dialog import _wait
from xdart.gui.widgets.controls_panel import FormRow
from xdart.gui.widgets.file_dialogs import NEXUS_FILTER

FORMATS = ('spec_series', 'spec_container', 'nexus')
LABELS = ('SPEC + Image Series', 'SPEC + Eiger', 'NeXus')
#: Data format -> (calibrant kind, direct-beam kind), the approved table
ROLE_KINDS = {'spec_series': ('spec', 'spec_series'), 'spec_container': ('spec_container', 'spec_container'),
              'nexus': ('nexus_stack', 'nexus')}
#: each page's stored scan-mode tokens (unchanged) under the visible All / Select
SCAN_TOKENS = {'Stitch': ('Stitch All', 'Select Scans'), 'RSM': ('All Scans', 'Select Scans')}


@pytest.fixture(scope='module')
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _page(name, tmp_path):
    from xdart.gui.pages.services import empty_host_services
    if name == 'Stitch':
        from tests.xdart.pages.test_stitch_page import _Status
        from xdart.gui.tabs.stitch.page import StitchPage
        page = StitchPage(empty_host_services(_Status()))
    else:
        from tests.xdart.pages.test_rsm_page import _Status
        from xdart.gui.tabs.rsm.page import RSMPage
        page = RSMPage(empty_host_services(_Status()))
    page.set_field('project_root', str(tmp_path))
    return page


def _dialog(name, page):
    if name == 'Stitch':
        page._calibrate()
    else:
        from xrd_tools.session.readiness import ControlAction
        page._control_action(ControlAction.CALIBRATE)
    dialog = page._calibration_dialog
    _wait(dialog)
    return dialog


def _settings(name):
    if name == 'Stitch':
        from xdart.gui.tabs.stitch import settings
    else:
        from xdart.gui.tabs.rsm import settings
    return settings.page_settings, settings.apply_page_settings


def _choose(qapp, row, token):
    """A user's choice in a combo row: real arrow keys on the closed menu, Qt's activation, which FormRow commits
    on the next event-loop turn (it ignores a programmatic index change). The shown item needs no key."""
    editor = row.editor
    index = editor.findData(token)
    assert index >= 0, (token, [editor.itemData(i) for i in range(editor.count())])
    key = QtCore.Qt.Key.Key_Down if index > editor.currentIndex() else QtCore.Qt.Key.Key_Up
    for _ in range(editor.count()):
        if editor.currentIndex() == index:
            break
        QtTest.QTest.keyClick(editor, key)
    assert editor.currentIndex() == index
    qapp.processEvents()
    qapp.processEvents()


def _shown(widget, root):
    return widget.isVisibleTo(root)


def _readiness(name, page):
    return page._readiness[0] if name == 'Stitch' else page._readiness_now()[0]


def _shut(qapp, page, dialog=None):
    if dialog is not None:
        _close(qapp, page, dialog)
        return
    from xdart.gui.pages.values import PageCleanup
    close = getattr(page, 'close_page', None) or page.shutdown
    until = time.monotonic() + 10
    receipt = close()
    while receipt.status is PageCleanup.PENDING and time.monotonic() < until:
        qapp.processEvents()
        time.sleep(.005)
        receipt = close()
    assert receipt.status is PageCleanup.CLEAN
    page.deleteLater()
    qapp.processEvents()


@pytest.mark.parametrize('name', ('Stitch', 'RSM'))
def test_f1_the_page_data_format_drives_its_source_rows(qapp, tmp_path, name):
    page = _page(name, tmp_path)
    try:
        rows = page._rows
        row = rows['data_format']
        assert page.controls.project_card.isAncestorOf(row) and row.label.text() == 'Data format'
        assert [row.editor.itemData(i) for i in range(row.editor.count())] == list(FORMATS)
        assert [row.editor.itemText(i) for i in range(row.editor.count())] == list(LABELS)
        assert row.editor.toolTip() and rows['input_mode'].editor.toolTip()
        assert page._value('data_format') == 'spec_series' and page._value('input_mode') == 'files'
        mode = rows['input_mode']
        assert [mode.editor.itemText(i) for i in range(mode.editor.count())] == ['Files', 'Folder']
        shown_labels = [r.label.text() for r in page.controls.source_card.findChildren(FormRow)
                        if _shown(r, page) and _shown(r.label, page)]
        assert 'Source' not in shown_labels, shown_labels  # the redundant Source-kind row is gone
        for token, folder, kind, label in (('spec_series', False, 'Spec', 'SPEC File'),
                                           ('spec_container', False, 'Spec', 'SPEC File'),
                                           ('nexus', False, 'NeXus', 'NeXus File'),
                                           ('spec_series', True, 'Directory', 'Folder'),
                                           ('nexus', True, 'Directory', 'Folder')):
            _choose(qapp, row, token)
            _choose(qapp, mode, 'directory' if folder else 'files')
            spec_files = token != 'nexus' and not folder
            assert page._value('source_kind') == kind, (token, folder)
            assert mode.label.text() == label, (token, folder, mode.label.text())
            assert _shown(rows['spec_path'], page)
            assert _shown(rows['subdirs'], page) is folder and _shown(rows['name_filter'], page) is folder
            images = folder or token != 'nexus'
            assert _shown(rows['image_dir'], page) is images and _shown(rows['image_stem'], page) is images
            assert rows['image_dir'].label.text() == 'Image Path' and rows['image_stem'].label.text() == 'Image Stem'
            assert _shown(rows['scan_mode'], page) is spec_files, (token, folder)
    finally:
        _shut(qapp, page)


@pytest.mark.parametrize('name', ('Stitch', 'RSM'))
def test_f1_the_page_format_sets_its_calibration_roles_and_quick_rows(qapp, tmp_path, name):
    page = _page(name, tmp_path)
    dialog = _dialog(name, page)
    try:
        spec = tmp_path / 'sample.spec'
        spec.write_text('')
        page.set_sources((str(spec),))
        for key, text in (('source', str(tmp_path / 'calibrant.spec')), ('beam_source', str(tmp_path / 'beam.spec'))):
            dialog._fields[key].setText(text)
            dialog._fields[key].editingFinished.emit()
        qapp.processEvents()
        _wait(dialog)
        assert dialog.rough_mode.currentData() == 'scan'
        for token in ('spec_container', 'nexus', 'spec_series'):
            _choose(qapp, page._rows['data_format'], token)
            _wait(dialog)
            calibrant, beam = ROLE_KINDS[token]
            assert (dialog.source_kind.currentData(), dialog.beam_source_kind.currentData()) == (calibrant, beam)
            for combo in (dialog.source_kind, dialog.beam_source_kind):
                assert not combo.isEnabled() and 'Data format' in combo.toolTip(), combo.toolTip()
            spec_rows, entry = token != 'nexus', token != 'spec_series'
            for prefix, role in (('', 'Calibrant'), ('beam_', 'Direct-beam')):
                quick = dialog._quick_rows
                assert _shown(quick[prefix + 'scan'], dialog) is spec_rows, (token, prefix)
                assert _shown(quick[prefix + 'entry'], dialog) is entry, (token, prefix)
                for key in ('images', 'stem'):
                    assert _shown(quick[prefix + key], dialog) is spec_rows, (token, prefix, key)
                assert quick[prefix + 'source'].label.text() == f"{role} {'SPEC' if spec_rows else 'NeXus'} file"
                assert dialog._choosers[prefix + 'source']() == ('spec' if spec_rows else NEXUS_FILTER)
            # no role borrows another's inputs (the page's SPEC files stay its own; NeXus shows NeXus files)
            assert page._paths_by_kind.get('Spec') == (str(spec),)
            assert dialog._text('source') == str(tmp_path / 'calibrant.spec')
            assert dialog._text('beam_source') == str(tmp_path / 'beam.spec')
    finally:
        _shut(qapp, page, dialog)


@pytest.mark.parametrize('name', ('Stitch', 'RSM'))
def test_f2_each_page_remembers_its_own_format(qapp, tmp_path, name):
    document_of, apply = _settings(name)
    other_name = 'RSM' if name == 'Stitch' else 'Stitch'
    page = _page(name, tmp_path)
    dialog = _dialog(name, page)
    other = _page(other_name, tmp_path)  # the same Project folder: still its own choice
    try:
        _choose(qapp, page._rows['data_format'], 'nexus')
        _wait(dialog)
        document = document_of(page)
        assert document['fields']['data_format'] == 'nexus' and document['fields']['input_mode'] == 'files'
        assert 'source_kind' not in document['fields']
        choices = document['calibration']['choices']
        assert 'source_kind' not in choices and 'beam_source_kind' not in choices, sorted(choices)
        assert other._value('data_format') == 'spec_series', 'one page never changes the other'
        restored = _page(name, tmp_path / 'restored')
        try:
            apply(restored, document)
            assert restored._value('data_format') == 'nexus' and restored._value('source_kind') == 'NeXus'
        finally:
            _shut(qapp, restored)
        assert other._value('data_format') == 'spec_series'
    finally:
        _shut(qapp, other)
        _shut(qapp, page, dialog)


_LEGACY = (  # (page kind, calibrant choice, its source, expected format, mode, conflict)
    ('Spec', None, '', 'spec_series', 'files', False),
    ('Directory', None, '', 'spec_series', 'directory', False),
    ('NeXus', None, '', 'nexus', 'files', False),
    ('Spec', 'spec_container', 'cal.spec', 'spec_container', 'files', False),
    ('Spec', 'nexus_stack', 'cal.nxs', 'spec_series', 'files', True),
)


@pytest.mark.parametrize('name', ('Stitch', 'RSM'))
@pytest.mark.parametrize(('kind', 'calibrant', 'source', 'expected', 'mode', 'conflict'), _LEGACY)
def test_f2_legacy_settings_migrate_before_validation(qapp, tmp_path, name, kind, calibrant, source, expected, mode,
                                                      conflict):
    document_of, apply = _settings(name)
    template = _page(name, tmp_path / 'template')
    page = _page(name, tmp_path / 'page')
    try:
        document = document_of(template)
        fields = document['fields']
        fields.pop('data_format', None)
        fields.pop('input_mode', None)
        fields.update(source_kind=kind, image_stem='legacy_stem', scan='3')
        document['sources'] = {'Spec': [str(tmp_path / 'a.spec')], 'NeXus': [str(tmp_path / 'b.nxs')],
                               'Directory': [str(tmp_path)]}
        if calibrant is not None:
            text = {'spec_container': 'SPEC + Eiger / NeXus', 'nexus_stack': 'NeXus images + metadata'}[calibrant]
            document['calibration'] = {'fields': {'source': str(tmp_path / source)},
                                       'choices': {'source_kind': {'data': calibrant, 'text': text}}}
        apply(page, document)
        qapp.processEvents()
        assert (page._value('data_format'), page._value('input_mode')) == (expected, mode)
        assert page._value('image_stem') == 'legacy_stem' and page._value('scan') == '3'
        assert page._paths_by_kind == {'Spec': (str(tmp_path / 'a.spec'),), 'NeXus': (str(tmp_path / 'b.nxs'),),
                                       'Directory': (str(tmp_path),)}
        if conflict:
            assert page._format_choice and 'Data format' in _readiness(name, page)
            assert page._format_choice in page._rows['data_format'].editor.toolTip()
            assert 'NeXus' in page._format_choice, 'the reason names the conflicting role format'
            page._rows['data_format'].editor.activated.emit(page._rows['data_format'].editor.currentIndex())
            qapp.processEvents()
            assert not page._format_choice, 'choosing the format (even the shown one) settles it'
        else:
            assert not page._format_choice
    finally:
        _shut(qapp, template)
        _shut(qapp, page)


def test_f2_source_identity_and_int_composition_are_unchanged(qapp, tmp_path):
    from xdart.gui.tabs.scattering.controls_inventory import (SOURCE_DIRECTORY, SOURCE_FILE, SOURCE_FILTER,
                                                              SOURCE_META, SOURCE_RECURSIVE, SOURCE_SUFFIX,
                                                              SOURCE_TYPE)
    from xdart.gui.widgets.controls_panel import ControlsPanel
    from xrd_tools.analysis.stitch_sources import StitchSourceSelection
    from xrd_tools.session.readiness import ControlFieldKind, ControlFormField, SectionId
    # the persisted selection identity keeps its tokens and its fields (no format element)
    assert [field.name for field in dataclasses.fields(StitchSourceSelection)] == [
        'kind', 'paths', 'scans', 'stitch_all', 'image_dir', 'image_stem', 'recursive', 'name_filter',
        'point_image_choices']
    page = _page('Stitch', tmp_path)
    try:
        spec = tmp_path / 'sample.spec'
        spec.write_text('')
        for kind, paths in (('Spec', (str(spec),)), ('NeXus', (str(tmp_path / 'b.nxs'),)), ('Directory', (str(tmp_path),))):
            page.set_field('source_kind', kind)
            page.set_sources(paths)
            selection = page._source_selection()
            assert selection.kind == kind and selection.fingerprint_value[0] == kind
    finally:
        _shut(qapp, page)
    # Int's own source rows: the composer without the opt-in groups lays them out exactly as before
    panel = ControlsPanel(show_section_numbers=False)
    try:
        fields = (ControlFormField(SectionId.SOURCE, 'Source', SOURCE_TYPE, 'Image Series', ControlFieldKind.COMBO,
                                   ('Image Series', 'Image Directory', 'Single Image')),
                  ControlFormField(SectionId.SOURCE, 'Subdirs', SOURCE_RECURSIVE, False, ControlFieldKind.BOOL),
                  ControlFormField(SectionId.SOURCE, 'Directory', SOURCE_DIRECTORY, '', ControlFieldKind.LINE, (), True),
                  ControlFormField(SectionId.SOURCE, 'Image File', SOURCE_FILE, '', ControlFieldKind.LINE, (), True),
                  ControlFormField(SectionId.SOURCE, 'File Type', SOURCE_SUFFIX, 'tif', ControlFieldKind.COMBO,
                                   ('tif', 'raw')),
                  ControlFormField(SectionId.SOURCE, 'Meta Type', SOURCE_META, 'txt', ControlFieldKind.COMBO,
                                   ('txt', 'pdi')),
                  ControlFormField(SectionId.SOURCE, 'Filter', SOURCE_FILTER, ''))
        built = panel.render_source_rows(panel.source_card, fields)
        parents = {path: row.parentWidget() for path, row in built.items()}
        singles = (SOURCE_DIRECTORY, SOURCE_FILE, SOURCE_FILTER)
        assert len({parents[path] for path in singles}) == 1, 'single rows sit on the card itself'
        body = parents[SOURCE_DIRECTORY]
        assert parents[SOURCE_TYPE] is parents[SOURCE_RECURSIVE] and parents[SOURCE_TYPE] is not body
        assert parents[SOURCE_SUFFIX] is parents[SOURCE_META] and parents[SOURCE_SUFFIX] not in (body, parents[SOURCE_TYPE])
        assert [built[path].label.text() for path in (SOURCE_TYPE, *singles)] == [
            'Source', 'Directory', 'Image File', 'Filter']
    finally:
        panel.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize('name', ('Stitch', 'RSM'))
def test_f3_format_edits_follow_the_effective_source_and_wait_for_busy_jobs(qapp, tmp_path, name, monkeypatch):
    from xdart.gui.tabs.calibration import dialog as dialog_module
    page = _page(name, tmp_path)
    dialog = _dialog(name, page)
    try:
        spec = _spec_file(tmp_path)
        for key, value in (('beam_source', str(spec)), ('beam_scan', '1'), ('source', str(spec)), ('scan', '1')):
            dialog._fields[key].setText(value)
            dialog._fields[key].editingFinished.emit()
        qapp.processEvents()
        _wait(dialog)
        row = page._rows['data_format']
        counter = '_revision' if name == 'Stitch' else '_form_revision'
        revision = getattr(page, counter)
        before = (dialog._spec(beam=True), dialog._spec())
        _choose(qapp, row, 'spec_container')
        _wait(dialog)
        assert getattr(page, counter) > revision, 'a format edit reconciles the page revision'
        assert (dialog._spec(beam=True), dialog._spec()) == before, 'a SPEC variant keeps the same effective source'
        _choose(qapp, row, 'nexus')
        _wait(dialog)
        assert dialog._spec(beam=True).kind.value == 'nexus_stack' and dialog._spec().kind.value == 'nexus_stack'
        _choose(qapp, row, 'spec_series')
        _wait(dialog)
        # a real calibration job, held at the window's own dispatcher, keeps the format until it ends
        gate = threading.Event()
        real = dialog_module._dispatch_job

        def held(request, cancelled, publish):
            gate.wait(20)
            return real(request, cancelled, publish)
        monkeypatch.setattr(dialog_module, '_dispatch_job', held)
        dialog._fields['scan'].setText('2')
        dialog._fields['scan'].editingFinished.emit()
        until = time.monotonic() + 10
        while not dialog._owner.busy and time.monotonic() < until:
            qapp.processEvents()
            time.sleep(.005)
        assert dialog._owner.busy
        qapp.processEvents()
        assert not row.isEnabled() and 'calibration' in row.editor.toolTip().lower(), row.editor.toolTip()
        gate.set()
        _wait(dialog)
        qapp.processEvents()
        assert row.isEnabled() and page._value('data_format') == 'spec_series'
    finally:
        gate_set = locals().get('gate')
        if gate_set is not None:
            gate_set.set()
        _shut(qapp, page, dialog)


@pytest.mark.parametrize('name', ('Stitch', 'RSM'))
def test_f4_scans_is_one_row_with_a_wide_selection_editor(qapp, tmp_path, name):
    page = _page(name, tmp_path)
    try:
        page.resize(1400, 900)
        page.show()
        qapp.processEvents()
        menu, selection = page._rows['scan_mode'], page._rows['scan']
        all_token, select_token = SCAN_TOKENS[name]
        assert [menu.editor.itemData(i) for i in range(menu.editor.count())] == [all_token, select_token]
        assert [menu.editor.itemText(i) for i in range(menu.editor.count())] == ['All', 'Select']
        assert menu.label.text() == 'Scans' and selection.editor.toolTip()
        _choose(qapp, menu, select_token)
        assert _shown(selection, page) and not _shown(selection.label, page), 'no separate Selection label'
        menu_box = QtCore.QRect(menu.editor.mapTo(page, QtCore.QPoint(0, 0)), menu.editor.size())
        line = QtCore.QRect(selection.editor.mapTo(page, QtCore.QPoint(0, 0)), selection.editor.size())
        assert abs(menu_box.center().y() - line.center().y()) <= 4, (menu_box, line)  # one row
        assert line.left() > menu_box.left() and selection.editor.width() > menu.editor.width()
        assert menu.editor.width() >= menu.editor.sizeHint().width(), 'the menu is not clipped'
        selection.editor.setFocus()
        selection.editor.selectAll()
        QtTest.QTest.keyClicks(selection.editor, '2,[4-6]')
        QtTest.QTest.keyClick(selection.editor, QtCore.Qt.Key.Key_Return)
        qapp.processEvents()
        _choose(qapp, menu, all_token)
        assert page._value('scan_mode') == all_token and menu.editor.currentText() == 'All'
        assert not _shown(selection, page) and _shown(menu, page), 'All hides the editor, not the row'
        _choose(qapp, menu, select_token)
        assert _shown(selection, page) and page._value('scan') == '2,[4-6]', 'the selection is retained'
    finally:
        page.hide()
        _shut(qapp, page)
