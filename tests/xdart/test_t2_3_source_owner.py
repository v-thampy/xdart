"""In-tree acceptance tests for O-1a-T2.3 (single source owner: idle reconcile +
build-aside swap).

§14.11.E-8 full coverage for the idle path: an idle unrelated edit, an
energy-preference-only edit, and a net-zero source edit perform ZERO source
reconciliation/poll; exactly ONE reconciliation happens for a real
source-selection change, including across an idle change → Start; and a GENUINE
direct-in-tree parameter edit STILL reconciles (the echo reentrancy guard does
not over-block).  Driven through the real ``_on_controls_v2_field_changed`` /
``_on_controls_v2_source_tree_changed`` handlers."""

from __future__ import annotations

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


def _count_reconciles(widget, monkeypatch):
    calls = []
    monkeypatch.setattr(
        widget, "_sync_controls_v2_source_index", lambda: calls.append(1))
    return calls


def test_idle_unrelated_edit_does_not_reconcile(widget, monkeypatch):
    calls = _count_reconciles(widget, monkeypatch)
    widget._on_controls_v2_field_changed(("Signal", "mask_file"), "/tmp/t23-mask.edf")
    assert calls == []


def test_energy_preference_only_edit_clears_energy_cache_only(widget, monkeypatch):
    calls = _count_reconciles(widget, monkeypatch)
    energy_sentinel = ("src", 12.0)
    probe_sentinel = ("src", "probe")
    widget._controls_v2_source_energy_cache = energy_sentinel
    widget._controls_v2_metadata_probe_cache = probe_sentinel

    widget._on_controls_v2_field_changed(
        ("Source", "energy_preference"), "metadata")

    # §14.11.D.4: energy-preference invalidation is LOCAL to the energy cache.
    assert widget._controls_v2_source_energy_cache is None
    assert widget._controls_v2_metadata_probe_cache is probe_sentinel
    # ... and it performs ZERO source reconciliation.
    assert calls == []


def test_net_zero_source_edit_does_not_reconcile(widget, monkeypatch):
    # A source-selection path recorded at its CURRENT (committed) value is a
    # net-zero edit — the effective selection is unchanged, so the fold reconciles
    # nothing (this is the A->B->A history collapsed to its winning value).
    path = ("Signal", "include_subdir")
    current = widget._controls_v2_param(path).value()
    widget._controls_v2_record_edit(path, current, origin="deferred")
    calls = _count_reconciles(widget, monkeypatch)

    assert widget._controls_v2_fold_deferred_edits_into_intent() is None
    assert calls == []


def test_exactly_one_reconcile_across_idle_change_then_start(widget, monkeypatch, tmp_path):
    calls = _count_reconciles(widget, monkeypatch)
    new_dir = tmp_path / "scandir"
    new_dir.mkdir()

    # An IDLE source-selection change reconciles exactly once (the echo of the
    # programmatic apply is guarded; the field handler owns the single reconcile).
    widget._on_controls_v2_field_changed(("Signal", "img_dir"), str(new_dir))
    assert calls == [1]

    # Its still-present journal entry must NOT cause a SECOND reconcile at Start:
    # the effective selection already matches the committed live value.
    assert widget._controls_v2_fold_deferred_edits_into_intent() is None
    assert calls == [1]


def test_genuine_in_tree_edit_still_reconciles(widget, monkeypatch):
    calls = _count_reconciles(widget, monkeypatch)
    # A direct parameter-tree edit (NOT via _on_controls_v2_field_changed, so the
    # reentrancy guard is not set) must STILL reconcile.
    assert not getattr(widget, "_controls_v2_applying_field", False)
    widget._on_controls_v2_source_tree_changed(
        None, [(object(), "value", "x")])
    assert calls == [1]
