"""The SPEC file picker: the platform's own dialog, keeping the SPEC files picked."""
import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.widgets import file_dialogs
from xdart.gui.widgets.file_dialogs import ANY_FILTER, NEXUS_FILTER, browse_spec


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("many", (True, False))
def test_the_spec_picker_is_the_platform_dialog_and_refuses_what_is_not_spec(qapp, tmp_path, monkeypatch, many):
    """Maintainer, 2026-09-30: "choosing the spec file still brings up this dialog. all others bring up
    the native mac os style dialog". A beamline folder: the SPEC file beside its images and notes."""
    spec = tmp_path / "direct_beam"
    spec.write_text("#F direct_beam\n#E 1\n#S 1 ascan del 0 1 1 1\n")
    image, notes = tmp_path / "direct_beam_scan1_0000.raw", tmp_path / "direct_beam_scan1_0000.raw.pdi"
    image.write_bytes(b"\x00\x01" * 8)
    notes.write_text("All Counters;del=1\n")
    opened, warned = [], []
    picks = [str(spec), str(image), str(notes)] if many else [str(image)]

    def static(parent, title, start, file_filter):
        opened.append((title, start, file_filter))
        return (picks, file_filter) if many else (picks[0], file_filter)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames" if many else "getOpenFileName",
                        staticmethod(static))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *args: warned.append(args[2])))
    kept = browse_spec(None, "Choose SPEC files", many=many, fallback=str(tmp_path))
    # the static (native) dialog, listing every file; where it opens is browse()'s rule (the last pick)
    assert [(title, file_filter) for title, _start, file_filter in opened] == [("Choose SPEC files", ANY_FILTER)]
    if many:
        assert kept == [str(spec)]
        assert warned == [f"Not a SPEC file, so not added: {image.name}, {notes.name}"]
    else:
        assert kept == [] and warned == [f"Not a SPEC file, so not added: {image.name}"]
    warned.clear()
    picks[:] = [str(spec)]
    assert browse_spec(None, "Choose SPEC files", many=many) == [str(spec)] and not warned


def test_stitch_spec_sources_use_the_spec_picker_and_keep_every_pick(qapp, tmp_path, monkeypatch):
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch.page import StitchPage

    class Status:
        def show(self, text, timeout_ms=0):
            pass
    calls = []
    picks = [str(tmp_path / "one"), str(tmp_path / "two")]
    monkeypatch.setattr(file_dialogs, "browse_spec", lambda *a, **k: calls.append(("spec", k)) or picks)
    monkeypatch.setattr(file_dialogs, "browse", lambda *a, **k: calls.append(("any", k)) or picks)
    page = StitchPage(empty_host_services(Status()))
    try:
        page.set_field("source_kind", "Spec")
        page._browse("spec_path")
        assert calls[-1][0] == "spec" and page._source_paths == tuple(picks)  # both files kept
        page.set_field("source_kind", "NeXus")
        page._browse("spec_path")
        kind, options = calls[-1]
        assert kind == "any" and options["file_filter"] == NEXUS_FILTER
    finally:
        page.shutdown()


def test_calibration_source_fields_pick_spec_or_nexus_by_their_source_type(qapp, tmp_path, monkeypatch):
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    calls = []
    monkeypatch.setattr(file_dialogs, "browse_spec", lambda *a, **k: calls.append("spec") or [])
    monkeypatch.setattr(file_dialogs, "browse", lambda *a, **k: calls.append(k.get("file_filter")) or [])
    dialog = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    try:
        def click(key):
            dialog._rows[key][1].findChild(QtWidgets.QAbstractButton).click()
            return calls[-1]
        for key, combo, spec_kind, nexus_kind in (("source", dialog.source_kind, "spec", "nexus_stack"),
                                                  ("beam_source", dialog.beam_source_kind, "spec_series", "nexus")):
            combo.setCurrentIndex(combo.findData(spec_kind))
            assert click(key) == "spec"
            combo.setCurrentIndex(combo.findData(nexus_kind))
            assert click(key) == NEXUS_FILTER
    finally:
        dialog.close()
        dialog.deleteLater()


def test_the_save_mode_offers_its_name_in_the_last_folder_and_remembers_only_a_pick(qapp, tmp_path, monkeypatch):
    """`file_dialogs.browse(save=True)`: the shared rule's start folder with the default file name (an absolute
    default, a managed location, is offered as it is); a pick is remembered as the last folder, a cancel
    remembers nothing (Codex's Stitch/Int reuse audit R5)."""
    from xdart.utils.browse import last_browse_dir, remember_browse_path
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))  # this test's own last folder
    last, other = tmp_path / "last", tmp_path / "other"
    last.mkdir()
    other.mkdir()
    remember_browse_path(last)
    offered, answer = [], [""]

    def save(parent, title, start, file_filter):
        offered.append((title, start, file_filter))
        return answer[0], file_filter
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", staticmethod(save))
    assert file_dialogs.browse(None, "Save it", save=True, default_name="a.json") == []  # cancelled
    assert offered[-1] == ("Save it", str(last / "a.json"), ANY_FILTER) and last_browse_dir() == str(last)
    managed = str(other / "managed.json")
    answer[0] = managed
    assert file_dialogs.browse(None, "Save it", save=True, default_name=managed) == [managed]
    assert offered[-1][1] == managed and last_browse_dir() == str(other)


def test_every_stitch_and_rsm_chooser_goes_through_the_shared_rule(qapp, tmp_path, monkeypatch):
    """Codex's Stitch/Int reuse audit R5: the mask's frame chooser, the calibration save, Stitch's settings and
    RSM's Output called Qt's dialog directly, so a pick never became the shared last folder and the mask chooser
    had no file types. Through ``file_dialogs`` each remembers its pick, and a cancel changes nothing."""
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.rsm.page import RSMPage
    from xdart.gui.tabs.calibration.dialog import CalibrationContext, CalibrationDialog
    from xdart.gui.tabs.calibration.mask import MaskDialog
    from xdart.gui.tabs.stitch.page import StitchPage
    from xdart.utils.browse import last_browse_dir

    class Status:
        def show(self, text, timeout_ms=0):
            pass
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))  # no last folder yet
    asked, answer = [], [""]

    def dialog(kind):
        def static(parent, title, start, file_filter=""):
            asked.append((kind, title, start, file_filter))
            return answer[0], file_filter
        return staticmethod(static)
    for kind in ("getSaveFileName", "getOpenFileName"):
        monkeypatch.setattr(QtWidgets.QFileDialog, kind, dialog(kind))
    picked = tmp_path / "picked"
    picked.mkdir()
    (tmp_path / "calibration").mkdir()
    mask = MaskDialog(tmp_path)
    calibration = CalibrationDialog(CalibrationContext('Stitch', tmp_path))
    page = StitchPage(empty_host_services(Status()))
    rsm = RSMPage(empty_host_services(Status()))
    try:
        mask._choose()  # cancelled: no source, no folder remembered
        assert asked[-1][0] == "getOpenFileName" and asked[-1][3] == file_dialogs.DETECTOR_SOURCE_FILTER
        assert mask._source is None and last_browse_dir() == ""
        assert calibration._save_path() is None and asked[-1][2] == str(calibration._default_name(False))
        assert page._choose_settings_path("save", str(tmp_path)) is None
        assert asked[-1][1:] == ("Save Stitch settings", str(tmp_path), file_dialogs.SETTINGS_FILTER)
        rsm.set_field("output_path", str(tmp_path / "old_rsm.nexus"))
        rsm._browse("output_path")
        assert rsm._value("output_path") == str(tmp_path / "old_rsm.nexus")  # a cancel changes nothing
        answer[0] = str(picked / "new_rsm.nexus")
        rsm._browse("output_path")
        assert rsm._value("output_path") == answer[0] and last_browse_dir() == str(picked)
        assert asked[-1][2] == str(tmp_path / "old_rsm.nexus")  # its name, in its own folder (no last yet)
        answer[0] = str(picked / "settings.json")
        assert page._choose_settings_path("load", "") == answer[0] and asked[-1][0] == "getOpenFileName"
        saved = tmp_path / "calibration" / "geometry.json"
        answer[0] = str(saved)
        assert calibration._save_path() == saved.resolve() and last_browse_dir() == str(saved.parent)
    finally:
        for widget in (mask, calibration):
            widget.close()
            widget.deleteLater()
        page.shutdown()
        rsm.close_page()
        rsm.deleteLater()


def test_ask_opens_the_dialog_as_given_and_remembers_nothing(qapp, tmp_path, monkeypatch):
    """`file_dialogs.ask` is the one place a dialog opens (Codex's Stitch/Int reuse audit R5): the start folder
    and filter exactly as given, [] when cancelled, and no last folder remembered. `browse` keeps the shared
    rule around it."""
    from xdart.utils.browse import last_browse_dir, remember_browse_path
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    last = tmp_path / "last"
    last.mkdir()
    remember_browse_path(last)
    asked, answer = [], [str(tmp_path / "picked.tif")]

    def open_file(parent, title, start, file_filter):
        asked.append(("open", title, start, file_filter))
        return answer[0], file_filter

    def folder(parent, title, start):
        asked.append(("folder", title, start))
        return answer[0]
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(open_file))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", staticmethod(folder))
    assert file_dialogs.ask(None, "Pick", start="/given", file_filter=NEXUS_FILTER) == answer
    assert asked[-1] == ("open", "Pick", "/given", NEXUS_FILTER) and last_browse_dir() == str(last)
    assert file_dialogs.ask(None, "Folder", start="/given", folder=True) == answer
    assert asked[-1] == ("folder", "Folder", "/given")
    answer[0] = ""
    assert file_dialogs.ask(None, "Pick", start="/given") == [] and last_browse_dir() == str(last)


def test_ints_own_choosers_open_through_the_shared_dialog_owner(qapp, monkeypatch):
    """Int's workspace choosers (Controls paths, standalone Calibrate / Make Mask sources, the Live source) open
    their dialogs through `file_dialogs.ask`, with their own titles, the page's start folder and their own
    filters as before. The page keeps the Browse rule itself, so they remember nothing."""
    from xdart.gui.pages import scattering_workspace as workspace
    from xdart.gui.tabs.scattering.controls_projection import (
        BACKGROUND_DIRECTORY, BACKGROUND_FILE, MASK_FILE, PONI_FILE, PROJECT_ROOT, SAVE_PATH, SOURCE_DIRECTORY,
    )
    from xrd_tools.sources.selection import DirectorySourceSpec
    asked = []

    def ask(parent, title, **options):
        asked.append((title, options))
        return ["/picked/frame.tif"]
    monkeypatch.setattr(file_dialogs, "ask", ask)
    paths = workspace._control_path_chooser(None)
    for path in (PROJECT_ROOT, SOURCE_DIRECTORY, BACKGROUND_DIRECTORY):
        assert paths(path, "", "/start") == "/picked/frame.tif"
        assert asked[-1] == ("Choose folder", {"start": "/start", "folder": True})
    assert paths(SAVE_PATH, "", "/start") == "/picked/frame.tif"
    assert asked[-1] == ("Choose processed-data folder", {"start": "/start", "folder": True})
    for path, title, file_filter in ((BACKGROUND_FILE, "Choose Background detector image",
                                      file_dialogs.BACKGROUND_IMAGE_FILTER),
                                     (PONI_FILE, "Choose PONI calibration", file_dialogs.PONI_FILTER),
                                     (MASK_FILE, "Choose detector mask", file_dialogs.MASK_FILTER)):
        assert paths(path, "", "/start") == "/picked/frame.tif"
        assert asked[-1] == (title, {"start": "/start", "file_filter": file_filter})
    sources = workspace._authoring_source_chooser(None)
    assert sources("poni", "/start") == "/picked/frame.tif"
    assert asked[-1][0] == "Choose calibration source image" and asked[-1][1]["start"] == "/start"
    assert sources("mask", "/start") == "/picked/frame.tif"  # a TIFF: no HDF frame dialog
    assert asked[-1] == ("Choose TIFF or HDF5/NeXus source for mask", {
        "start": "/start", "file_filter": "Mask sources (*.tif *.tiff *.h5 *.hdf5 *.nxs *.nexus)"})
    live = workspace._source_selection_chooser(None)
    chosen = live(None, "Image Directory", "/start")
    assert type(chosen) is DirectorySourceSpec and str(chosen.root) == "/picked/frame.tif"
    assert asked[-1] == ("Choose image directory", {"start": "/start", "folder": True})
    live(None, "Single Image", "/start")
    assert asked[-1][0] == "Choose one detector image" and asked[-1][1]["start"] == "/start"
    assert len(asked) == 11  # all seven of the module's dialog call sites, each through the owner
