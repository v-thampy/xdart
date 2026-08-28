"""O-1a-T2.8b (§19.9 / §19.8) — GUI input authority + recursively immutable journal.

§19.9: the Controls transaction must NOT infer user intent by polling every
visible editor.  A programmatic ``setText`` on a non-focused row is a projection,
not an edit; only an explicit draft signal (``textEdited`` → ``draftChanged``) or
the single editor the user is CURRENTLY editing may become journal intent.

§19.8: ``JournalEntry`` stores a recursively immutable representation, so a
caller cannot reach the stored value through any exposed reference — including
the private one — and ``.value`` reconstructs a fresh mutable copy.

Production-wired: real ``staticWidget``, real Qt widgets/rows, real draft slots.
The two amended §12 form-harvest cases are committed here in-tree (§19.9 req 4)
so this contract no longer depends on a mutable temporary oracle.
"""

from __future__ import annotations

import pytest
from pyqtgraph.Qt import QtTest, QtWidgets


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setenv("XDART_CONTROLS_PANEL_V2", "1")
    from xdart.gui.tabs.static_scan.static_scan_widget import staticWidget

    value = staticWidget()
    value._refresh_controls_v2_profile_now()
    try:
        yield value
    finally:
        value.close()
        value.deleteLater()
        qapp.processEvents()


def _form_row(widget, path):
    from xdart.gui.widgets.controls_panel import FormRow

    return next(
        row
        for row in widget.controls_v2.findChildren(FormRow)
        if tuple(row.path) == tuple(path)
    )


# ---------------------------------------------------------------------------
# §19.9 — the no-focus full-form sweep is deleted
# ---------------------------------------------------------------------------

def test_nonfocused_programmatic_text_never_becomes_a_user_edit(widget, qapp):
    """A projection/programmatic ``setText`` on an unfocused row is not intent."""
    path = ("Int1D", "points")
    row = _form_row(widget, path)
    row.editor.clearFocus()
    qapp.processEvents()
    assert not row.editor.hasFocus()
    widget._controls_v2_edit_journal_dict().pop(path, None)

    row.editor.setText("777")

    winners = dict(widget._controls_v2_collect_pending_edits())
    assert path not in winners
    assert path not in widget._controls_v2_edit_journal_dict()


def test_nonfocused_programmatic_text_does_not_reach_a_non_run_commit(widget, qapp):
    """The same authority rule at the NON-RUN action seam (reintegrate /
    advanced processing): an unfocused programmatic value is never applied."""
    path = ("Int1D", "points")
    row = _form_row(widget, path)
    row.editor.clearFocus()
    qapp.processEvents()
    widget._controls_v2_edit_journal_dict().pop(path, None)
    before = widget._controls_v2_field_values(overlay_pending=False)[path]

    row.editor.setText("888")
    assert widget._commit_controls_v2_pending_edits() is None

    after = widget._controls_v2_field_values(overlay_pending=False)[path]
    assert after == before
    assert str(after) != "888"


def test_full_form_sweep_is_not_consulted_by_the_transaction(widget, monkeypatch):
    """The transaction must not call the full visible-form snapshot at all —
    a sweep that merely 'looks but does not journal' is still an authority."""
    calls = []
    real = widget.controls_v2.current_form_edits
    monkeypatch.setattr(
        widget.controls_v2,
        "current_form_edits",
        lambda: (calls.append(1), real())[1],
    )

    widget._controls_v2_collect_pending_edits()

    assert calls == []


def test_focused_editor_flush_is_preserved(widget, qapp):
    """§19.9 req 2: the ONE focused editor still flushes — the value the user is
    mid-editing may post-date its last draft signal (click/focus-transition)."""
    path = ("Int1D", "points")
    row = _form_row(widget, path)
    widget._controls_v2_edit_journal_dict().pop(path, None)
    widget.show()
    qapp.processEvents()
    row.editor.setFocus()
    qapp.processEvents()
    assert row.editor.hasFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, "333")
    qapp.processEvents()

    assert dict(widget._controls_v2_collect_pending_edits())[path] == "333"


def test_focused_harvest_failure_refuses_preparation(widget, qapp, monkeypatch):
    """§12 test 9's contract, re-seated on the SURVIVING harvest seam: a harvest
    failure is a typed refusal, never fail-open."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsTransactionError,
        DeferredRunEditsPendingError,
    )

    widget._pending_controls_v2_run_configuration = None
    widget._enter_run_state()
    widget._on_controls_v2_field_changed(("GI", "Grazing"), True)
    widget.wrangler.finished.emit()
    qapp.processEvents()

    def _boom():
        raise RuntimeError("injected focused-editor harvest failure")

    monkeypatch.setattr(widget.controls_v2, "focused_form_edit", _boom)

    with pytest.raises(ControlsTransactionError):
        widget._controls_v2_collect_pending_edits()
    with pytest.raises(DeferredRunEditsPendingError):
        widget._prepare_controls_v2_run_configuration()
    assert getattr(widget, "_pending_controls_v2_run_configuration", None) is None


# -- the two amended §12 form-harvest cases, committed in-tree (§19.9 req 4) --

def test_newer_uncommitted_form_edit_wins_over_older_journal_entry(widget, qapp):
    """Amended §12 test: a real focused keystroke edit beats an older journal
    entry by revision (shown widget, real focus, real QTest.keyClicks)."""
    path = ("Int1D", "points")
    widget._controls_v2_record_edit(path, "111", origin="deferred")
    row = _form_row(widget, path)
    widget.show()
    qapp.processEvents()
    row.editor.setFocus()
    qapp.processEvents()
    assert row.editor.hasFocus()
    row.editor.selectAll()
    QtTest.QTest.keyClicks(row.editor, "222")
    qapp.processEvents()

    assert dict(widget._controls_v2_collect_pending_edits())[path] == "222"


def test_failed_uncommitted_form_edit_is_retained_in_journal(widget, qapp):
    """Amended §12 test: the production user-input seam is the draft signal; an
    invalid draft refuses preparation and is RETAINED at draft origin."""
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        DeferredRunEditsPendingError,
    )

    path = ("BG", "Scale")
    widget._on_controls_v2_field_draft_changed(path, "not-a-number")

    with pytest.raises(DeferredRunEditsPendingError):
        widget._prepare_controls_v2_run_configuration()

    entry = widget._controls_v2_edit_journal_dict().get(path)
    assert entry is not None
    assert entry["value"] == "not-a-number"
    assert entry["origin"] == "draft"


# ---------------------------------------------------------------------------
# §19.8 — JournalEntry is recursively immutable
# ---------------------------------------------------------------------------

def _entry(value, revision=1, origin="draft"):
    from xdart.gui.tabs.static_scan.static_scan_widget import JournalEntry

    return JournalEntry(value, revision, origin)


def test_list_ingress_alias_mutation_cannot_reach_the_entry():
    source = ["a"]
    entry = _entry(source)
    source.append("b")
    assert entry.value == ["a"]


def test_nested_ingress_alias_mutation_cannot_reach_the_entry():
    inner = {"k": [1, 2]}
    source = {"outer": inner}
    entry = _entry(source)
    inner["k"].append(3)
    inner["new"] = "x"
    source["another"] = 1
    assert entry.value == {"outer": {"k": [1, 2]}}


def test_private_storage_cannot_be_mutated_in_place():
    """§19.8: mutating the private reference must not change future reads."""
    entry = _entry(["a"])
    entry._value.append("b")
    assert entry.value == ["a"]
    assert entry["value"] == ["a"]


def test_private_storage_cannot_be_rebound():
    entry = _entry(["a"])
    with pytest.raises((AttributeError, TypeError)):
        entry._value = "hijacked"
    with pytest.raises((AttributeError, TypeError)):
        entry.revision = 99
    with pytest.raises((AttributeError, TypeError)):
        entry.origin = "deferred"
    assert entry.value == ["a"]


def test_nested_outward_mutation_cannot_reach_the_entry():
    entry = _entry({"a": [1]})
    got = entry.value
    got["a"].append(2)
    got["b"] = 3
    assert entry.value == {"a": [1]}


def test_scalar_and_tuple_semantics_are_preserved():
    assert _entry(7).value == 7
    assert _entry("7").value == "7"
    assert _entry(True).value is True
    assert _entry(None).value is None
    tup = _entry((1, "a")).value
    assert tup == (1, "a") and isinstance(tup, tuple)
    lst = _entry([1, "a"]).value
    assert lst == [1, "a"] and isinstance(lst, list)


def test_cloned_entry_shares_no_mutable_state():
    import copy as _copy

    entry = _entry([{"a": 1}])
    clone = _copy.deepcopy(entry)
    assert clone.value == [{"a": 1}]
    assert clone.revision == entry.revision and clone.origin == entry.origin
    clone.value[0]["a"] = 999
    assert entry.value == [{"a": 1}]


def test_last_write_wins_replacement_keeps_monotonic_revisions(widget):
    path = ("Signal", "mask_file")
    first = widget._controls_v2_record_edit(path, "/tmp/one.edf", origin="draft")
    second = widget._controls_v2_record_edit(path, "/tmp/two.edf", origin="idle")
    entry = widget._controls_v2_edit_journal_dict()[path]
    assert second > first
    assert entry["value"] == "/tmp/two.edf"
    assert entry["revision"] == second
    assert entry["origin"] == "idle"


def test_exact_revision_clearing_is_preserved(widget):
    """§19.8 req 5: a newer same-path revision recorded during the commit is NOT
    cleared by the transaction that consumed the older one."""
    path = ("Signal", "mask_file")
    widget._controls_v2_record_edit(path, "/tmp/old.edf", origin="deferred")
    original_commit = widget.commit_controls_transaction

    def commit_then_record_newer(staged):
        result = original_commit(staged)
        assert result.ok
        widget._controls_v2_record_edit(path, "/tmp/new.edf", origin="draft")
        return result

    widget.commit_controls_transaction = commit_then_record_newer
    try:
        assert widget._controls_v2_fold_deferred_edits_into_intent() is None
    finally:
        widget.commit_controls_transaction = original_commit

    assert widget._controls_v2_edit_journal_dict()[path]["value"] == "/tmp/new.edf"
