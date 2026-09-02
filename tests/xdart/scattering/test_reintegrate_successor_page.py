"""Focused page-level immutable Reintegration successor lifecycle tests."""

from __future__ import annotations

from dataclasses import replace
from threading import Event
import time

import pytest

from tests.core.test_vnext_p34_existing_replacement import _seed_existing
from tests.xdart.scattering.test_p34_reintegrate_operation import _loaded_page
from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.operation_values import OperationIdentity
from xdart.gui.tabs.scattering.processed_browser import (
    ReintegrateSuccessorDirective,
    ReintegrateSuccessorPhase,
)
from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
from xrd_tools.session.readiness import ControlAction, SectionId


@pytest.fixture
def qapp():
    from pyqtgraph.Qt import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _directive(page, successor) -> ReintegrateSuccessorDirective:
    capture = page._capture_current_loaded_browse()
    assert capture is not None
    return ReintegrateSuccessorDirective(
        capture,
        str(successor.target.resolve()),
        capture.entry,
        capture.labels,
        successor.terminal.commit_identity,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        OperationIdentity(90),
        page._operation_context_stamp(),
    )


def _wait(call, *, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = call()
        if value is not None:
            return value
        time.sleep(0.005)
    raise AssertionError("successor page lifecycle did not settle")


def test_page_selects_only_exact_ready_successor(
    tmp_path, monkeypatch, qapp,
) -> None:
    page, _store, predecessor, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    successor = _seed_existing(
        predecessor.target.parent, name="page-successor",
    )
    target = str(successor.target.resolve())
    try:
        adoption = page._processed_browser.adopt_reintegrate_successor(
            _directive(page, successor)
        )
        assert page._begin_pending_reintegrate_successor(adoption)

        def selected():
            page._drain_executor()
            capture = page._capture_current_loaded_browse()
            return capture if capture is not None and capture.target == target else None

        capture = _wait(selected)
        assert capture.request.terminal_commit_identity is (
            successor.terminal.commit_identity
        )
        assert page._processed_browser.pending_reintegrate_successor is None
        assert "now viewing" in page._notice_text
    finally:
        page.close_workspace()


def test_refresh_cancels_successor_load_without_switching(
    tmp_path, monkeypatch, qapp,
) -> None:
    page, _store, predecessor, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    successor = _seed_existing(
        predecessor.target.parent, name="refresh-successor",
    )
    successor_target = str(successor.target.resolve())
    predecessor_target = str(predecessor.target.resolve())
    entered, release = Event(), Event()
    real_read = BrowseLoader._read_context

    def gated(self, request, *args, **kwargs):
        if request.source_path == successor_target:
            entered.set()
            assert release.wait(10.0)
        return real_read(self, request, *args, **kwargs)

    monkeypatch.setattr(BrowseLoader, "_read_context", gated)
    try:
        adoption = page._processed_browser.adopt_reintegrate_successor(
            _directive(page, successor)
        )
        assert page._begin_pending_reintegrate_successor(adoption)
        assert entered.wait(5.0)
        page._handle_shell_command(ShellCommand(
            ShellCommandKind.REFRESH_BROWSER
        ))
        cancelling = page._processed_browser.pending_reintegrate_successor
        assert cancelling is not None
        assert cancelling.phase is ReintegrateSuccessorPhase.CANCELLING
        release.set()

        def settled():
            page._drain_executor()
            return (
                True
                if page._processed_browser.pending_reintegrate_successor is None
                else None
            )

        assert _wait(settled)
        capture = page._capture_current_loaded_browse()
        assert capture is not None and capture.target == predecessor_target
        assert capture.target != successor_target
    finally:
        release.set()
        page.close_workspace()


def test_selection_waits_for_successor_cleanup_then_replays_latest_target(
    tmp_path, monkeypatch, qapp,
) -> None:
    page, _store, predecessor, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    successor = _seed_existing(
        predecessor.target.parent, name="blocked-successor",
    )
    selected = _seed_existing(
        predecessor.target.parent, name="deferred-selection",
    )
    successor_target = str(successor.target.resolve())
    selected_target = str(selected.target.resolve())
    entered, release = Event(), Event()
    real_read = BrowseLoader._read_context

    def gated(self, request, *args, **kwargs):
        if request.source_path == successor_target:
            entered.set()
            assert release.wait(10.0)
        return real_read(self, request, *args, **kwargs)

    monkeypatch.setattr(BrowseLoader, "_read_context", gated)
    try:
        adoption = page._processed_browser.adopt_reintegrate_successor(
            _directive(page, successor)
        )
        assert page._begin_pending_reintegrate_successor(adoption)
        assert entered.wait(5.0)
        page._select_scan(selected_target)
        deferred = page._deferred_browse_selection
        assert deferred is not None and deferred.value == selected_target
        assert page._context_controller._browse_request is None
        release.set()

        def selected_after_cleanup():
            page._drain_executor()
            capture = page._capture_current_loaded_browse()
            return (
                capture
                if capture is not None and capture.target == selected_target
                else None
            )

        assert _wait(selected_after_cleanup)
        assert page._deferred_browse_selection is None
        assert page._processed_browser.pending_reintegrate_successor is None
    finally:
        release.set()
        page.close_workspace()


def test_wrong_successor_inventory_retires_without_switching_or_staying_busy(
    tmp_path, monkeypatch, qapp,
) -> None:
    page, _store, predecessor, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    successor = _seed_existing(
        predecessor.target.parent,
        labels=(2, 5),
        name="wrong-inventory-successor",
    )
    predecessor_target = str(predecessor.target.resolve())
    successor_target = str(successor.target.resolve())
    try:
        adoption = page._processed_browser.adopt_reintegrate_successor(
            _directive(page, successor)
        )
        assert page._begin_pending_reintegrate_successor(adoption)

        def rejected():
            page._drain_executor()
            return (
                True
                if page._processed_browser.pending_reintegrate_successor
                is None
                else None
            )

        assert _wait(rejected)
        capture = page._capture_current_loaded_browse()
        assert capture is not None and capture.target == predecessor_target
        assert capture.target != successor_target
        assert not page._processed_browser.busy
        assert "ended without switching" in page._notice_text
    finally:
        page.close_workspace()


def test_post_consume_refusal_releases_candidate_and_retires_adoption(
    tmp_path, monkeypatch, qapp,
) -> None:
    page, _store, predecessor, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    successor = _seed_existing(
        predecessor.target.parent,
        name="post-consume-refusal-successor",
    )
    predecessor_target = str(predecessor.target.resolve())
    successor_target = str(successor.target.resolve())
    real_consume = BrowseLoader.consume

    def refuse_after_transfer(self, outcome):
        candidate = real_consume(self, outcome)
        if outcome.request.source_path == successor_target:
            assert candidate is not None
            return None
        return candidate

    monkeypatch.setattr(BrowseLoader, "consume", refuse_after_transfer)
    try:
        adoption = page._processed_browser.adopt_reintegrate_successor(
            _directive(page, successor)
        )
        assert page._begin_pending_reintegrate_successor(adoption)

        def rejected():
            page._drain_executor()
            return (
                True
                if page._processed_browser.pending_reintegrate_successor
                is None
                else None
            )

        assert _wait(rejected)
        capture = page._capture_current_loaded_browse()
        assert capture is not None and capture.target == predecessor_target
        assert capture.target != successor_target
        assert not page._processed_browser.busy
        assert not page._context_controller.browse_pending
        assert "ended without switching" in page._notice_text
    finally:
        page.close_workspace()


def test_foreign_candidate_path_retires_adoption_without_stranding_ui(
    tmp_path, monkeypatch, qapp,
) -> None:
    page, _store, predecessor, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    successor = _seed_existing(
        predecessor.target.parent,
        name="foreign-candidate-path-successor",
    )
    predecessor_target = str(predecessor.target.resolve())
    successor_target = str(successor.target.resolve())
    real_read = BrowseLoader._read_context

    def foreign_path(self, request, *args, **kwargs):
        candidate = real_read(self, request, *args, **kwargs)
        if request.source_path == successor_target:
            assert candidate is not None
            return replace(
                candidate,
                requested_path="/foreign/not-the-successor.nexus",
            )
        return candidate

    monkeypatch.setattr(BrowseLoader, "_read_context", foreign_path)
    try:
        adoption = page._processed_browser.adopt_reintegrate_successor(
            _directive(page, successor)
        )
        assert page._begin_pending_reintegrate_successor(adoption)

        def rejected():
            page._drain_executor()
            return (
                True
                if page._processed_browser.pending_reintegrate_successor
                is None
                else None
            )

        assert _wait(rejected)
        capture = page._capture_current_loaded_browse()
        assert capture is not None and capture.target == predecessor_target
        assert capture.target != successor_target
        assert not page._processed_browser.busy
        assert not page._context_controller.browse_pending
    finally:
        page.close_workspace()


def test_pending_retired_browse_cleanup_fences_projection_and_dispatch(
    tmp_path, monkeypatch, qapp,
) -> None:
    page, store, _predecessor, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    controller = page._context_controller
    assert not controller.browse_pending
    controller._retired_browse_cleanup = object()
    try:
        actions = {
            action.action: action
            for action in page._project_controls(store.snapshot()).actions_for(
                SectionId.PROCESSING
            )
        }
        assert not actions[ControlAction.REINTEGRATE_1D].enabled
        assert not actions[ControlAction.REINTEGRATE_2D].enabled
        page._reintegrate_action("1d")
        assert page._workspace_operations.reintegrate_identity is None
        assert page._notice_text == (
            "Reintegrate is unavailable while Browse cleanup is pending."
        )
    finally:
        controller._retired_browse_cleanup = None
        page.close_workspace()
