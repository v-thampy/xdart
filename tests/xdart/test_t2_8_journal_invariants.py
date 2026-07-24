"""T-2.8 (Correction E, §15.12-E): journal invariants + §17.8 purity.

Frozen JournalEntry (E.1 — stack #5), exact-revision interleaving regression
guard (E.4), and the §17.8 purity sentinel for _controls_v2_threshold_config.
Production-wired: real staticWidget.
"""
from __future__ import annotations

import copy

import pytest
from pyqtgraph.Qt import QtWidgets


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


# -- E.1 : truly frozen, deep-copied journal entry --------------------------

def test_journal_entry_metadata_is_immutable(widget):
    path = ("Signal", "mask_file")
    revision = widget._controls_v2_record_edit(path, "/tmp/e.edf", origin="draft")
    entry = widget._controls_v2_edit_journal_dict()[path]
    with pytest.raises((AttributeError, TypeError)):
        entry.revision = revision + 100
    with pytest.raises((AttributeError, TypeError)):
        entry.origin = "deferred"
    with pytest.raises((AttributeError, TypeError)):
        entry._value = "hijacked"


def test_journal_entry_value_is_deep_copied_out(widget):
    path = ("Int1D", "points")
    widget._controls_v2_record_edit(path, ["a"], origin="draft")
    entry = widget._controls_v2_edit_journal_dict()[path]
    got = entry.value
    got.append("b")
    # mutating the projected value must NOT change the stored entry
    assert widget._controls_v2_edit_journal_dict()[path].value == ["a"]


# -- E.4 : exact-revision interleaving regression guard ---------------------

def test_newer_same_path_revision_survives_successful_clear(widget):
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

    entry = widget._controls_v2_edit_journal_dict()[path]
    assert entry["value"] == "/tmp/new.edf"


# -- §17.8 : _controls_v2_threshold_config is a pure reader ------------------

def test_threshold_config_does_not_mutate_intent_or_store_state(widget):
    intent = widget._controls_v2_ensure_run_intent()
    widget._controls_v2_threshold_state = None
    before_ref = intent.threshold
    before_val = copy.deepcopy(intent.threshold)

    cfg = widget._controls_v2_threshold_config()

    # A config getter is passive: no rewrite of intent.threshold (identity kept),
    # no lazy-store of _controls_v2_threshold_state.
    assert intent.threshold is before_ref
    assert intent.threshold == before_val
    assert widget._controls_v2_threshold_state is None
    assert cfg is not None
