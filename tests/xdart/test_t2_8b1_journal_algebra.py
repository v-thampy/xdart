"""O-1a-T2.8b.1 (§21.6) — the closed immutable journal value algebra.

The journal is the revisioned AUTHORITY for user intent.  T-2.8b froze the
built-in containers but stored every other value as ``("atom", deepcopy(value))``
— and that copy stays reachable and MUTABLE through ``entry._frozen``, so a
``bytearray``/NumPy array/mutable object could change journal authority in place
with no new revision.

These adversaries attack the REAL ``_frozen`` storage, round-trip every supported
leaf and container, and prove that unsupported mutable objects are refused at
ingress with the typed refusal rather than hidden behind a deepcopy.

Production-wired: the real ``staticWidget``, its real journal, and the real Qt
draft/deferred slots.
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


def _entry(value, revision=1, origin="draft"):
    from xdart.gui.tabs.static_scan.static_scan_widget import JournalEntry

    return JournalEntry(value, revision, origin)


def _error():
    from xdart.gui.tabs.static_scan.static_scan_widget import (
        ControlsTransactionError,
    )

    return ControlsTransactionError


# ---------------------------------------------------------------------------
# §21.6 req 6 — adversaries against the REAL `_frozen` payload
# ---------------------------------------------------------------------------

def test_bytearray_payload_cannot_be_mutated_through_frozen_storage():
    """Codex's probe: `entry._frozen[1].extend(b"b")` changed journal authority
    without a revision."""
    entry = _entry(bytearray(b"a"))
    assert entry.value == bytearray(b"a")

    payload = entry._frozen[1]
    assert isinstance(payload, bytes)          # immutably ENCODED, not copied
    with pytest.raises(AttributeError):
        payload.extend(b"b")

    assert entry.value == bytearray(b"a")


def test_bytearray_round_trips_as_an_independent_bytearray():
    entry = _entry(bytearray(b"ab"))
    first = entry.value
    assert isinstance(first, bytearray) and first == bytearray(b"ab")
    first.extend(b"c")                         # outward mutation is isolated
    assert entry.value == bytearray(b"ab")


def test_mutable_custom_object_is_refused_at_ingress():
    class Mutable:
        def __init__(self):
            self.items = []

    with pytest.raises(_error()):
        _entry(Mutable())


def test_numpy_array_is_refused_at_ingress():
    np = pytest.importorskip("numpy")

    with pytest.raises(_error()):
        _entry(np.array([1, 2, 3]))


def test_numpy_scalars_are_normalized_to_python_leaves():
    np = pytest.importorskip("numpy")

    assert _entry(np.int64(7)).value == 7
    assert type(_entry(np.int64(7)).value) is int
    assert _entry(np.float64(1.5)).value == 1.5
    assert type(_entry(np.float64(1.5)).value) is float
    assert _entry(np.bool_(True)).value is True


def test_a_container_holding_an_unsupported_value_is_refused():
    class Mutable:
        pass

    with pytest.raises(_error()):
        _entry([1, Mutable()])
    with pytest.raises(_error()):
        _entry({"k": {"nested": Mutable()}})


# ---------------------------------------------------------------------------
# §21.6 req 6 — round trips for every supported leaf and container
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, 7, -3, 1.5, 2 + 3j, "", "text", b"bytes"],
)
def test_supported_leaves_round_trip_exactly(value):
    got = _entry(value).value
    assert got == value
    assert type(got) is type(value)


@pytest.mark.parametrize(
    "value",
    [
        (),
        (1, "a", None),
        [],
        [1, [2, 3], {"k": "v"}],
        {},
        {"a": 1, "b": [1, 2]},
        set(),
        {1, 2, 3},
        frozenset({1, 2}),
        {"nested": {"deep": (1, [2, {"x": b"y"}])}},
    ],
)
def test_supported_containers_round_trip_exactly(value):
    got = _entry(value).value
    assert got == value
    assert type(got) is type(value)


def test_mapping_equality_is_order_independent():
    """§21.6 req 5: equality must be value-correct, not insertion-order bound."""
    first = _entry({"a": 1, "b": 2})
    second = _entry({"b": 2, "a": 1})
    assert first.value == second.value


def test_set_equality_is_order_independent():
    assert _entry({1, 2, 3}) == _entry({3, 1, 2})
    assert _entry(frozenset({1, 2})) == _entry(frozenset({2, 1}))


def test_journal_entry_is_not_hashable():
    """§21.6 req 5: __hash__ is removed — no consumer needs it and the old
    unconditional implementation was invalid for unhashable payloads."""
    with pytest.raises(TypeError):
        hash(_entry("x"))


def test_nested_ingress_alias_and_outward_mutation_are_still_isolated():
    inner = {"k": [1, 2]}
    entry = _entry({"outer": inner})
    inner["k"].append(3)
    assert entry.value == {"outer": {"k": [1, 2]}}
    got = entry.value
    got["outer"]["k"].append(9)
    assert entry.value == {"outer": {"k": [1, 2]}}


def test_clone_shares_no_mutable_state():
    entry = _entry([{"a": 1}])
    clone = copy.deepcopy(entry)
    clone.value[0]["a"] = 999
    assert entry.value == [{"a": 1}]
    assert clone == entry


# ---------------------------------------------------------------------------
# §21.6 req 4 — ingress rejection is a VISIBLE typed refusal at the Qt slots
# ---------------------------------------------------------------------------

def test_unsupported_draft_value_does_not_crash_the_qt_slot(widget, qapp):
    class Mutable:
        pass

    path = ("Int1D", "points")
    widget._controls_v2_edit_journal_dict().pop(path, None)
    messages = []
    widget.wrangler.showLabel.connect(messages.append)

    widget._on_controls_v2_field_draft_changed(path, Mutable())   # must not raise
    qapp.processEvents()

    assert path not in widget._controls_v2_edit_journal_dict()    # not stored
    assert messages                                              # not silent


def test_unsupported_deferred_value_does_not_crash_the_qt_slot(widget, qapp):
    class Mutable:
        pass

    path = ("BG", "Scale")
    widget._controls_v2_edit_journal_dict().pop(path, None)
    messages = []
    widget.wrangler.showLabel.connect(messages.append)

    widget._controls_v2_defer_field_edit(path, Mutable())         # must not raise
    qapp.processEvents()

    assert path not in widget._controls_v2_edit_journal_dict()
    assert messages


def test_supported_values_still_journal_through_the_real_slots(widget, qapp):
    path = ("Int1D", "points")
    widget._controls_v2_edit_journal_dict().pop(path, None)

    widget._on_controls_v2_field_draft_changed(path, "444")

    entry = widget._controls_v2_edit_journal_dict()[path]
    assert entry["value"] == "444"
    assert entry["origin"] == "draft"


# ---------------------------------------------------------------------------
# §21.6 req 6 — the exact-revision interleaving case still holds
# ---------------------------------------------------------------------------

def test_exact_revision_clearing_is_preserved(widget):
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
