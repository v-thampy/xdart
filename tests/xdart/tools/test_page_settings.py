"""The shared settings owner (``tools/page_settings``, G12), Qt-free: a page's settings document, its File ▸ Save and
Load, and its last session (the maintainer, 2026-09-24: "File ▸ Save/Load Settings and last-session restore
covering every input, the Advanced options and any calibration panel, never per-use confirmations, and a fresh
start neither reading nor writing it"). Stitch's and RSM's own rows are in their pages' settings tests."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from xdart.gui.pages.values import ActionCompleted, ActionRefused, CloseReceipt, PageCleanup
from xdart.gui.tools.page_settings import PageSettings, SettingsKind, plain
from xrd_tools.session.threshold_editing import ThresholdState

_KIND = SettingsKind("xdart-demo-settings", 2, "demo_settings", "Demo", sections=("extra",))


class _Page:
    """The surface the owner reads on a page (Stitch's and RSM's), without Qt."""

    def __init__(self, project=""):
        self.values = {"project_root": project, "count": "1"}
        self.statuses = []
        self.asked = []
        self.pick = None
        self.busy = False
        self.receipts = []
        self._readiness = ("", True)
        self._closing = False
        self._settings_path_chooser = self._choose

    def _choose(self, action, start):
        self.asked.append((action, start))
        return self.pick

    def _status(self, text, *, ready=True):
        self.statuses.append((text, ready))

    def _value(self, key):
        return self.values[key]


def _document(page):
    return {"kind": _KIND.kind, "version": _KIND.version, "fields": dict(page.values)}


def _apply(page, document):
    """The page's own part: check, then apply field by field; "boom" fails midway, after a field is set."""
    document = _KIND.validate(document)
    for key, value in document.get("fields", {}).items():
        page.values[key] = value
        if value == "boom":
            raise RuntimeError("no such count")


def _owner(page):
    def close():
        page._closing = True
        return page.receipts.pop(0)
    return PageSettings(page, _KIND, document=_document, apply=_apply, busy=lambda: page.busy, close=close)


@pytest.fixture
def session(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    monkeypatch.setenv("XDART_SESSION_FILE", str(path))
    monkeypatch.delenv("XDART_SESSION_FRESH", raising=False)
    return path


def test_the_document_checks_its_kind_version_and_sections_and_names_the_page():
    good = {"kind": "xdart-demo-settings", "version": 2, "fields": {"a": 1}, "extra": {}}
    assert _KIND.outcome == "demo-settings"
    for document in (good, {**good, "version": 1}, {**good, "version": 0}, {**good, "calibration": None},
                     {**good, "calibration": {"fields": {}, "choices": {}, "camera": ["x-", "z+"]}}):
        assert _KIND.validate(document) is document
    for document in ([good], {**good, "kind": "xdart-stitch-settings"}, {**good, "version": "2"},
                     {**good, "version": None}, {key: value for key, value in good.items() if key != "version"}):
        with pytest.raises(ValueError, match="^not an xdart Demo settings file$"):
            _KIND.validate(document)
    with pytest.raises(ValueError, match="^these Demo settings come from a newer xdart$"):
        _KIND.validate({**good, "version": 3})
    for key in ("fields", "sources", "threshold", "extra"):
        with pytest.raises(ValueError, match=f"^Demo settings {key} are malformed$"):
            _KIND.validate({**good, key: []})
    for calibration in ([], {"fields": ["rms"]}, {"choices": "x"}):
        with pytest.raises(ValueError, match="^Demo settings calibration is malformed$"):
            _KIND.validate({**good, "calibration": calibration})
    text = _KIND.dump({**good, "b": 2})
    assert text == json.dumps({**good, "b": 2}, indent=2, sort_keys=True) and _KIND.load(text) == {**good, "b": 2}
    for text in ("{not json", ""):
        with pytest.raises(ValueError, match="^not an xdart Demo settings file$"):
            _KIND.load(text)
    with pytest.raises(ValueError, match="^not an xdart Demo settings file$"):
        _KIND.dump({"kind": "other"})


def test_the_document_is_the_shared_sections_then_the_pages_own_in_order():
    page = _Page()
    page.values.update(a=1.5, b=math.inf, c=Path("x") / "y", d=True)
    page._fields = dict.fromkeys(page.values)
    page._paths_by_kind = {"Spec": ("one.spec", Path("two.spec")), "NeXus": ()}
    page._threshold_state = ThresholdState(True, 2.0, math.nan, False)
    page._calibration_settings_now = lambda: {"fields": {"rms": ".02"}, "choices": {}}
    document = _KIND.document(page, ["a", "b", "c", "d"], extra={"on": True}, write_mode="Append")
    assert list(document) == ["kind", "version", "fields", "sources", "threshold", "calibration", "extra",
                              "write_mode"]
    assert document == {
        "kind": "xdart-demo-settings", "version": 2,
        "fields": {"a": 1.5, "b": None, "c": str(Path("x") / "y"), "d": True},
        "sources": {"Spec": ["one.spec", "two.spec"]},  # a kind with no files is left out
        "threshold": {"apply_threshold": True, "threshold_min": 2.0, "threshold_max": None, "mask_saturation": False},
        "calibration": {"fields": {"rms": ".02"}, "choices": {}}, "extra": {"on": True}, "write_mode": "Append"}
    assert [plain(value) for value in (None, 3, "s", False, -0.5, math.nan)] == [None, 3, "s", False, -0.5, None]


def test_save_and_load_through_the_chooser_never_confirm_and_roll_back_what_cannot_apply(tmp_path, session):
    project = tmp_path / "project"
    project.mkdir()
    page = _Page(str(project))
    owner = _owner(page)
    page.pick = None
    assert owner.save() == ActionCompleted("demo-settings-save-cancelled") and not page.statuses
    assert page.asked == [("save", str(project))]  # the Project without a remembered folder
    page.pick = str(tmp_path / "picked" / "demo")  # a bare name: .json added
    (tmp_path / "picked").mkdir()
    assert owner.save() == ActionCompleted("demo-settings-saved")
    saved = tmp_path / "picked" / "demo.json"
    assert json.loads(saved.read_text(encoding="utf-8")) == _document(page)
    assert saved.read_text(encoding="utf-8") == _KIND.dump(_document(page))
    assert sorted(path.name for path in saved.parent.iterdir()) == ["demo.json"]  # no temporary left
    assert page.statuses[-1] == ("Demo settings saved: demo.json", True)
    page.pick = None
    owner.load()
    assert page.asked[-1] == ("load", str(saved.parent))  # the folder remembered from the save
    page.pick = str(tmp_path / "missing" / "demo.json")
    refused = owner.save()
    assert type(refused) is ActionRefused and "No such file or directory" in refused.reason
    assert page.statuses[-1] == (f"Could not save the Demo settings: {refused.reason}", False)

    other = _Page(str(project))
    loader = _owner(other)
    other.pick = page.pick = str(saved)
    page.values["count"] = "7"
    assert owner.save() == ActionCompleted("demo-settings-saved")
    other._readiness = ("Needs a Project folder", False)
    assert loader.load() == ActionCompleted("demo-settings-loaded") and other.values["count"] == "7"
    assert other.statuses[-1] == ("Demo settings loaded: demo.json", False)  # the page's readiness
    other.busy = True
    asked = len(other.asked)
    assert loader.load() == ActionRefused("Demo is busy; load settings when it is idle")
    assert len(other.asked) == asked  # refused before any chooser
    other.busy = False
    before = dict(other.values)
    for document, reason in (({"kind": "xdart-run-intent-profile"}, "not an xdart Demo settings file"),
                             ({**_document(other), "version": 9}, "these Demo settings come from a newer xdart"),
                             ({**_document(other), "fields": {"count": "boom"}},
                              "these Demo settings cannot be applied here (RuntimeError: no such count)")):
        saved.write_text(json.dumps(document), encoding="utf-8")
        assert loader.load() == ActionRefused(reason)
        assert other.values == before  # rolled back, or never applied
        assert other.statuses[-1] == (f"Could not load the Demo settings: {reason}", False)
    other.pick = None
    assert loader.load() == ActionCompleted("demo-settings-load-cancelled") and other.values == before


def test_the_last_session_is_restored_and_remembered_once_clean_and_a_fresh_start_neither_reads_nor_writes(
        tmp_path, session, monkeypatch):
    page = _Page()
    owner = _owner(page)
    owner.restore_last()  # no session file: nothing
    assert page.values == {"project_root": "", "count": "1"} and not page.statuses
    page.values["count"] = "5"
    page.receipts = [CloseReceipt(PageCleanup.PENDING, "a worker"), CloseReceipt(PageCleanup.CLEAN)]
    assert owner.close_and_remember().status is PageCleanup.PENDING and not session.exists()
    page.values["count"] = "6"  # after the first close: the inputs captured then are the ones kept
    assert owner.close_and_remember().status is PageCleanup.CLEAN
    assert json.loads(session.read_text())["demo_settings"]["fields"]["count"] == "5"
    remembered = session.read_bytes()
    page.receipts = [CloseReceipt(PageCleanup.CLEAN)]
    owner.close_and_remember()
    assert session.read_bytes() == remembered  # once

    restored = _Page()
    _owner(restored).restore_last()
    assert restored.values["count"] == "5" and not restored.statuses
    document = json.loads(session.read_text())
    document["demo_settings"]["fields"] = {"count": "boom"}
    session.write_text(json.dumps(document))
    broken = _Page()
    _owner(broken).restore_last()
    assert broken.values == {"project_root": "", "count": "1"}  # rolled back
    assert broken.statuses == [("Could not restore the last Demo settings", False)]

    document["demo_settings"]["fields"] = {"count": "9"}
    session.write_text(json.dumps(document))
    before = session.read_bytes()
    monkeypatch.setenv("XDART_SESSION_FRESH", "1")
    fresh = _Page(str(tmp_path))
    owner = _owner(fresh)
    owner.restore_last()
    assert fresh.values["count"] == "1"  # not read
    fresh.pick = str(tmp_path / "fresh.json")
    assert owner.save() == ActionCompleted("demo-settings-saved")  # a file the user names is still written
    fresh.receipts = [CloseReceipt(PageCleanup.CLEAN)]
    assert owner.close_and_remember().status is PageCleanup.CLEAN
    assert session.read_bytes() == before  # neither the settings nor the remembered folder written


def test_a_saved_field_its_row_cannot_hold_is_refused_naming_the_row():
    """``check_fields`` (RSM's, G12 unit 2): a choice among its row's choices unless they are a source's own
    names, a toggle true or false, and text or a number in a text row; else a ValueError naming the row."""
    from types import SimpleNamespace
    from xdart.gui.tools.page_settings import check_fields
    from xrd_tools.session.readiness import ControlFieldKind, ControlFormField, SectionId

    def field(label, kind, choices=()):
        return ControlFormField(SectionId.PROCESSING, label, ("Demo", label), "", kind, choices)
    page = SimpleNamespace(_fields={
        "grid": field("Grid", ControlFieldKind.COMBO, ("Quick", "Full")),
        "monitor": field("Monitor", ControlFieldKind.COMBO, ("None",)),
        "subdirs": field("Subdirs", ControlFieldKind.BOOL),
        "scan": field("Selection", ControlFieldKind.LINE), "rows": field("Rows", ControlFieldKind.INT)})
    check_fields(page, {"grid": "Full", "monitor": "I0", "subdirs": False, "scan": "43", "rows": 195},
                 open_choices={"monitor"})
    check_fields(page, {"scan": None, "rows": 1.5})
    for fields, reason in (({"grid": "Huge"}, "Grid: 'Huge' is not one of its choices"),
                           ({"monitor": "I0"}, "Monitor: 'I0' is not one of its choices"),
                           ({"subdirs": "false"}, "Subdirs: 'false' is not on or off"),
                           ({"subdirs": 0}, "Subdirs: 0 is not on or off"),
                           ({"scan": [43]}, "Selection: [43] is not text or a number"),
                           ({"rows": True}, "Rows: True is not text or a number"),
                           ({"rows": {"n": 1}}, "Rows: {'n': 1} is not text or a number")):
        with pytest.raises(ValueError) as refused:
            check_fields(page, fields)
        assert str(refused.value) == reason


def test_stitch_keeps_its_settings_on_the_shared_owner(qapp):
    """Stitch's document is the shared owner's (G12 unit 1): its kind, and the page's owner."""
    from xdart.gui.pages.services import empty_host_services
    from xdart.gui.tabs.stitch import settings as stitch
    from xdart.gui.tabs.stitch.page import StitchPage
    from tests.xdart.pages.test_stitch_page import _Status

    assert stitch.STITCH_SETTINGS == SettingsKind("xdart-stitch-settings", 1, "stitch_settings", "Stitch", ("gi",))
    for function in (stitch.validate_settings, stitch.dump_settings, stitch.load_settings):
        assert function.__self__ is stitch.STITCH_SETTINGS
    page = StitchPage(empty_host_services(_Status()))
    try:
        assert type(page._saved_settings) is PageSettings and page._saved_settings.kind is stitch.STITCH_SETTINGS
    finally:
        page.shutdown()
        page.deleteLater()


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
