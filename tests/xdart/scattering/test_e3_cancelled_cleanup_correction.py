"""Frozen oracle for reusable cancellation cleanup at the E3 join."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
from pathlib import Path
from threading import Event, Thread, current_thread
import textwrap
import time
from typing import get_type_hints

import pytest
from pyqtgraph.Qt import QtCore

from xdart.gui.tabs.scattering.adapters import browse_loader as loader_module
from xdart.gui.tabs.scattering.adapters.browse_loader import (
    BrowseLoader,
    _BrowseOperation,
)
from xdart.gui.tabs.scattering.browse_values import (
    BrowseCleanupReceipt,
    BrowseLoadOutcome,
    BrowseLoadRequest,
    BrowseLoadStatus,
)
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.display_retirement import (
    DisplayRetirementReceipt,
)
from xdart.gui.tabs.scattering.events import CleanupStatus, detach_exception
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.modules.display_context import BrowseContext
from xrd_tools.io import FrameScalarCatalog

from tests.xdart.scattering.test_e3_context_contract import (
    _browse,
    _running_controller,
    _select_browse,
)
from tests.xdart.scattering.test_e3_join_oracle import (
    _mount,
    _pause,
    _produce_browse_artifact,
    _run,
    _wait,
)


class _EmptyScalarReader:
    def __init__(self, source, *, resolve_source, callback=None):
        assert resolve_source is False
        self._path = str(Path(source).resolve())
        self._callback = callback

    def __enter__(self):
        return self

    def read_scalar_catalog(self, *, cancelled):
        if self._callback is not None:
            self._callback(self._path, cancelled)
        if cancelled():
            raise InterruptedError("Browse scalar catalog read cancelled")
        return FrameScalarCatalog(self._path, "entry", ())

    def __exit__(self, _exc_type, _exc, _tb):
        return None


def _empty_reader_factory(callback=None):
    def open_reader(source, *, resolve_source):
        return _EmptyScalarReader(
            source,
            resolve_source=resolve_source,
            callback=callback,
        )

    return open_reader


def _open_catalog_row(rig, monkeypatch, artifact: Path):
    monkeypatch.setattr(
        rig.page,
        "_browser_directory_chooser",
        lambda _current, _start_directory: str(artifact.parent),
    )
    rig.command(ShellCommand(ShellCommandKind.MENU, "File:Open Folder"))

    def catalog_row():
        scans = rig.shell.browser.scans
        return next(
            (
                scans.item(index)
                for index in range(scans.count())
                if scans.item(index).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                == str(artifact)
            ),
            None,
        )

    _wait(
        rig.app,
        lambda: catalog_row() is not None,
        diagnostic=lambda: f"{artifact} never entered the browser catalog",
    )
    row = catalog_row()
    assert row is not None
    return row


def _ready_operation(
    request: BrowseLoadRequest,
    context: BrowseContext,
) -> _BrowseOperation:
    operation = _BrowseOperation(request, Event())
    operation.context = context
    operation.outcome = BrowseLoadOutcome(
        request, BrowseLoadStatus.READY
    )
    operation.terminal = True
    return operation


def _controller_with_ready_c():
    controller, lifecycle, executor, _port, acquisition = (
        _running_controller()
    )
    controller.pause()
    request = controller.begin_browse("/processed/browse.c.nxs")
    _, context = _browse(
        request.token,
        request.load_generation,
        scan_key="browse.c",
        request=request,
    )
    operation = _ready_operation(request, context)
    loader = BrowseLoader()
    loader._active = operation
    controller._browse_loader = loader
    return (
        controller,
        lifecycle,
        executor,
        loader,
        acquisition,
        request,
        context,
        operation,
    )


def _wait_without_qt(predicate, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    assert predicate()


def test_cancelled_cleanup_has_one_typed_nonterminal_architecture() -> None:
    assert get_type_hints(BrowseLoader.cancel)["return"] is BrowseCleanupReceipt

    def class_tree(owner):
        return ast.parse(textwrap.dedent(inspect.getsource(owner))).body[0]

    def self_assignments(owner):
        return {
            node.attr
            for node in ast.walk(class_tree(owner))
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and isinstance(node.ctx, ast.Store)
        }

    loader_fields = self_assignments(BrowseLoader)
    assert {"_active", "_queued", "_close"} <= loader_fields
    assert not {name for name in loader_fields if "cleanup" in name}

    controller_fields = self_assignments(ContextController)
    assert {name for name in controller_fields if "cleanup" in name} == {
        "_cleanup_receipt"
    }
    controller_calls = []
    for method in class_tree(ContextController).body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for call in ast.walk(method):
            if not isinstance(call, ast.Call):
                continue
            function = call.func
            if (
                isinstance(function, ast.Attribute)
                and function.attr in {"cancel", "close"}
                and isinstance(function.value, ast.Attribute)
                and isinstance(function.value.value, ast.Name)
                and function.value.value.id == "self"
                and function.value.attr == "_browse_loader"
            ):
                controller_calls.append((method.name, function.attr))
    assert controller_calls == [
        ("close", "close"),
        ("_retain_cancel", "cancel"),
    ]

    page_uses = {}
    for method in class_tree(ScatteringWorkspace).body:
        if isinstance(method, ast.FunctionDef):
            if any(
                isinstance(node, ast.Attribute)
                and node.attr == "browse_pending"
                for node in ast.walk(method)
            ):
                page_uses[method.name] = True
    assert set(page_uses) == {
        "_classify_deferred_metadata",
        "_retry_pending_average_reload",
        "_select_scan",
        "_drain_executor",
        "_polling_needed",
        "_refresh_shell",
        "_start_permitted",
    }


@pytest.mark.parametrize("command", ("resume", "stop"))
def test_resume_or_stop_waits_for_exact_cancelled_c_cleanup_before_a_retirement(
    command: str,
    monkeypatch,
) -> None:
    (
        controller,
        _lifecycle,
        _executor,
        loader,
        acquisition,
        request_c,
        context_c,
        operation_c,
    ) = _controller_with_ready_c()
    records_c = context_c.record_store
    publications_c = context_c.publication_store
    release = BrowseContext.release
    attempts = 0

    def fail_once(context):
        nonlocal attempts
        if context is context_c:
            attempts += 1
            if attempts == 1:
                raise RuntimeError("cancelled C cleanup must be retried")
        return release(context)

    monkeypatch.setattr(BrowseContext, "release", fail_once)
    proof = DisplayRetirementReceipt(
        controller.run_identity, CleanupStatus.CLEANED
    )
    try:
        getattr(controller, command)()
        assert attempts == 1
        assert controller._cleanup_receipt is not None
        assert controller._cleanup_receipt.request is request_c
        assert loader._active is operation_c
        assert operation_c.context is context_c
        assert len(records_c) == len(publications_c) == 1

        # A clean acquisition proof cannot erase A while the exact cancelled
        # C operation still owns stores.  The cleanup retry is a separate,
        # page-driven prerequisite rather than a side effect of retirement.
        assert controller.apply_display_retirement(proof) is False
        assert attempts == 1
        assert controller.acquisition_context is acquisition
        assert loader._active is operation_c

        # Keep the existing controller poll as the one page-facing progress
        # surface.  No prospective API is required to reach the intended red.
        controller.poll_browse()
        assert attempts == 2
        assert len(records_c) == len(publications_c) == 0
        assert loader._active is None
        assert loader._queued is None
        assert loader._worker is None
        assert controller._cleanup_receipt is None
        assert controller.browse_pending is False

        assert controller.apply_display_retirement(proof) is True
        assert controller.acquisition_context is None
    finally:
        loader.close(request_c)


def test_page_polls_cancelled_c_refreshes_blocker_and_only_then_allows_next_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_c = _produce_browse_artifact(monkeypatch, tmp_path / "c")
    rig = _mount(
        monkeypatch,
        tmp_path / "a",
        labels=tuple(range(1, 120)),
        reduction_delay=0.001,
    )
    allow_release = Event()
    terminal_close = None
    try:
        _run(rig)
        _pause(rig)
        old_identity = rig.controller.run_identity
        acquisition = rig.controller.acquisition_context
        assert old_identity is not None
        assert acquisition is not None
        row_c = _open_catalog_row(rig, monkeypatch, artifact_c)

        # Freeze C at READY in the real loader while keeping the page from
        # consuming it.  Resume/Stop then exercises cancellation, not load.
        ensure_timer = rig.page._ensure_timer
        rig.page._run_timer.stop()
        monkeypatch.setattr(rig.page, "_ensure_timer", lambda: None)
        row_c.setSelected(True)
        rig.app.processEvents()
        _wait_without_qt(
            lambda: (
                rig.loader._active is not None
                and rig.loader._active.terminal
                and rig.loader._active.context is not None
            )
        )
        monkeypatch.setattr(rig.page, "_ensure_timer", ensure_timer)

        operation_c = rig.loader._active
        assert operation_c is not None
        request_c = operation_c.request
        context_c = operation_c.context
        assert context_c is not None
        records_c = context_c.record_store
        publications_c = context_c.publication_store
        catalog_c = context_c.scalar_catalog
        cache_c = context_c.browse_1d_cache
        assert type(catalog_c) is FrameScalarCatalog
        labels_c = catalog_c.labels
        assert labels_c
        assert context_c.frame_ids is labels_c
        assert context_c.loaded_labels is labels_c
        assert cache_c is not None
        real_release = rig.loader.release_context
        release_attempts: list[BrowseContext] = []

        def gated_release(context):
            if context is context_c:
                release_attempts.append(context)
                if not allow_release.is_set():
                    return BrowseCleanupReceipt(
                        request_c, CleanupStatus.CLEANUP_PENDING
                    )
            return real_release(context)

        monkeypatch.setattr(
            rig.loader, "release_context", gated_release
        )

        # Resume is the public cancellation command for paused C.  Stop only
        # after acquisition is running again; the real executor deliberately
        # rejects new submissions while its session remains paused.
        rig.shell.run_controls.startButton.click()
        _wait(
            rig.app,
            lambda: rig.lifecycle.phase is RunPhase.RUNNING,
            timeout=30.0,
        )
        rig.shell.run_controls.stopButton.click()
        _wait(
            rig.app,
            lambda: rig.lifecycle.phase
            in {RunPhase.IDLE, RunPhase.FAILED},
            timeout=30.0,
        )
        assert rig.lifecycle.phase is RunPhase.IDLE, (
            rig.page._notice_text,
            rig.page._progress,
        )
        assert release_attempts
        assert rig.controller._cleanup_receipt is not None
        assert rig.controller._cleanup_receipt.request is request_c
        assert rig.loader._active is operation_c
        assert len(records_c) == len(publications_c) == 0
        assert context_c.scalar_catalog is catalog_c
        assert context_c.browse_1d_cache is cache_c
        assert context_c.frame_ids is labels_c
        assert context_c.loaded_labels is labels_c
        assert context_c.released is False

        # Use a fresh target for both next-Run attempts so a mutant cannot red
        # on output-state admission instead of the cleanup prerequisite.
        snapshot = rig.page._intents.snapshot()
        intent = snapshot.thaw()
        intent.save_path = str(rig.output.with_name("run.second.nexus"))
        committed = rig.page._intents.commit(
            intent, expected_revision=snapshot.revision
        )
        assert committed.revision == snapshot.revision + 1
        rig.page._refresh_shell()

        # Exercise the page admission boundary even if the corrected shell
        # chooses to disable its Run button while cleanup is pending.
        rig.page._begin_run()
        _wait(
            rig.app,
            lambda: (
                rig.page._admission is None
                and rig.lifecycle.phase
                in {RunPhase.IDLE, RunPhase.RUNNING, RunPhase.FAILED}
            ),
            timeout=30.0,
        )
        assert rig.lifecycle.phase is RunPhase.IDLE
        assert rig.controller.run_identity is old_identity
        assert rig.controller.acquisition_context is acquisition
        assert rig.loader._active is operation_c

        rig.page._refresh_shell()
        assert rig.controller.browse_pending is True
        assert rig.page._run_timer.isActive()
        assert not rig.shell.run_controls.startButton.isEnabled()
        # A retained terminal progress summary cannot hide the semantic reason
        # why the next Run is disabled once the lifecycle is idle again.
        readiness = rig.shell.run_controls.readinessLabel
        assert readiness.full_text() == "Browse cleanup remains pending"
        assert readiness.toolTip() == "Browse cleanup remains pending"
        assert rig.page._start_permitted() == (
            False,
            "Browse cleanup remains pending",
        )

        attempts_before_release = len(release_attempts)
        allow_release.set()
        # From here the test calls no controller or loader method.  Ordinary
        # Qt delivery must retry cleanup and refresh the stale Run blocker.
        _wait(
            rig.app,
            lambda: (
                not rig.controller.browse_pending
                and not rig.loader.owns_request(request_c)
                and rig.shell.run_controls.startButton.isEnabled()
            ),
            timeout=30.0,
        )
        assert len(release_attempts) > attempts_before_release
        assert len(records_c) == len(publications_c) == 0
        assert context_c.scalar_catalog is None
        assert context_c.browse_1d_cache is None
        assert context_c.frame_ids == ()
        assert context_c.loaded_labels == ()
        assert context_c.released is True
        assert rig.loader._active is None
        assert rig.loader._queued is None
        assert rig.loader._worker is None
        assert rig.loader._close is None

        rig.shell.run_controls.startButton.click()
        _wait(
            rig.app,
            lambda: (
                rig.controller.run_identity is not None
                and rig.controller.run_identity is not old_identity
                and rig.controller.acquisition_context is not None
                and rig.controller.acquisition_context is not acquisition
            ),
            timeout=30.0,
            diagnostic=lambda: (
                f"phase={rig.lifecycle.phase.value}; "
                f"notice={rig.page._notice_text!r}; "
                f"admission={rig.page._admission!r}; "
                f"browse_pending={rig.controller.browse_pending}"
            ),
        )
    finally:
        allow_release.set()
        terminal_close = rig.close()
        assert terminal_close.cleanup_status is CleanupStatus.CLEANED
        rig.page.deleteLater()
        rig.app.processEvents()


def test_select_scan_b_release_failure_rearms_page_cleanup_polling(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_b = _produce_browse_artifact(
        monkeypatch, tmp_path / "browse-b"
    )
    artifact_c = _produce_browse_artifact(
        monkeypatch, tmp_path / "browse-c"
    )
    rig = _mount(
        monkeypatch,
        tmp_path / "acquisition-a",
        labels=tuple(range(1, 80)),
        reduction_delay=0.001,
    )
    try:
        _run(rig)
        _pause(rig)
        row_b = _open_catalog_row(rig, monkeypatch, artifact_b)
        row_b.setSelected(True)
        rig.app.processEvents()
        _wait(
            rig.app,
            lambda: (
                rig.controller.browse_context is not None
                and rig.controller.browse_context.requested_path
                == str(artifact_b)
            ),
            timeout=30.0,
        )
        context_b = rig.controller.browse_context
        assert context_b is not None
        records_b = context_b.record_store
        publications_b = context_b.publication_store
        catalog_b = context_b.scalar_catalog
        assert type(catalog_b) is FrameScalarCatalog
        labels_b = catalog_b.labels
        assert labels_b
        assert context_b.frame_ids is labels_b
        assert context_b.loaded_labels is labels_b
        row_c = _open_catalog_row(rig, monkeypatch, artifact_c)
        assert rig.controller.browse_context is context_b
        assert rig.controller.selection.names(context_b)
        assert context_b.scalar_catalog is catalog_b
        _wait(
            rig.app,
            lambda: (
                not rig.controller.browse_preview_polling_needed
                and rig.page._browse_1d_release_debt is None
                and not rig.page._scientific_repaint_pending
            ),
            timeout=30.0,
        )
        record_labels_b = records_b.labels()
        publication_labels_b = publications_b.labels()
        release = BrowseContext.release
        context_c: BrowseContext | None = None
        b_failures = c_failures = 0

        def fail_b_then_c_once(context):
            nonlocal context_c, b_failures, c_failures
            if context is context_b and b_failures == 0:
                b_failures += 1
                _wait_without_qt(
                    lambda: (
                        rig.loader._active is not None
                        and rig.loader._active.terminal
                        and rig.loader._active.context is not None
                    )
                )
                operation = rig.loader._active
                assert operation is not None
                context_c = operation.context
                raise RuntimeError("B release failed after C admission")
            if context is context_c and c_failures == 0:
                c_failures += 1
                raise RuntimeError("C cancellation cleanup failed once")
            return release(context)

        monkeypatch.setattr(
            BrowseContext, "release", fail_b_then_c_once
        )
        rig.page._run_timer.stop()
        row_c.setSelected(True)
        rig.app.processEvents()
        _wait(
            rig.app,
            lambda: b_failures == c_failures == 1,
            timeout=30.0,
            diagnostic=lambda: (
                f"b_failures={b_failures}; c_failures={c_failures}; "
                f"row_selected={row_c.isSelected()}; "
                f"browse_pending={rig.controller.browse_pending}; "
                f"active={rig.loader._active!r}; "
                f"notice={rig.page._notice_text!r}"
            ),
        )

        assert b_failures == c_failures == 1
        assert context_c is not None
        request_c = context_c.load_request
        assert rig.controller._cleanup_receipt is not None
        assert rig.controller._cleanup_receipt.request is request_c
        assert rig.controller.browse_context is context_b
        assert records_b.labels() == record_labels_b
        assert publications_b.labels() == publication_labels_b
        assert context_b.scalar_catalog is catalog_b
        assert context_b.frame_ids is labels_b
        assert context_b.loaded_labels is labels_b
        assert context_b.released is False
        assert rig.page._run_timer.isActive()

        records_c = context_c.record_store
        publications_c = context_c.publication_store
        catalog_c = context_c.scalar_catalog
        assert type(catalog_c) is FrameScalarCatalog
        labels_c = catalog_c.labels
        assert labels_c
        assert context_c.frame_ids is labels_c
        assert context_c.loaded_labels is labels_c
        _wait(
            rig.app,
            lambda: (
                not rig.controller.browse_pending
                and not rig.loader.owns_request(request_c)
            ),
            timeout=30.0,
        )
        assert c_failures == 1
        assert len(records_c) == len(publications_c) == 0
        assert context_c.scalar_catalog is None
        assert context_c.browse_1d_cache is None
        assert context_c.frame_ids == ()
        assert context_c.loaded_labels == ()
        assert context_c.released is True
        assert rig.controller.browse_context is context_b
        assert rig.controller.selection.names(context_b)
        assert records_b.labels() == record_labels_b
        assert publications_b.labels() == publication_labels_b
        assert context_b.scalar_catalog is catalog_b
        assert context_b.frame_ids is labels_b
        assert context_b.loaded_labels is labels_b
        assert context_b.released is False
    finally:
        receipt = rig.close()
        assert receipt.cleanup_status is CleanupStatus.CLEANED
        rig.page.deleteLater()
        rig.app.processEvents()


def test_b_release_failure_retains_c_cleanup_blocks_d_and_preserves_generation(
    monkeypatch,
) -> None:
    controller, _lifecycle, _executor, port, acquisition = (
        _running_controller()
    )
    controller.pause()
    request_b, context_b = _select_browse(
        controller, port, scan_key="browse.b"
    )
    records_b = context_b.record_store
    publications_b = context_b.publication_store

    loader = BrowseLoader()
    controller._browse_loader = loader
    admitted: list[tuple[BrowseLoadRequest, BrowseContext]] = []

    def launch_ready(
        request,
        *,
        propagate_failure=True,
        operation=None,
    ):
        assert operation is None
        _, context = _browse(
            request.token,
            request.load_generation,
            scan_key=Path(request.source_path).stem,
            request=request,
        )
        loader._active = _ready_operation(request, context)
        admitted.append((request, context))

    monkeypatch.setattr(loader, "_launch", launch_ready)
    release = BrowseContext.release
    attempts: dict[int, int] = {}

    def fail_first_b_and_c(context):
        key = id(context)
        attempts[key] = attempts.get(key, 0) + 1
        if context is context_b and attempts[key] == 1:
            raise RuntimeError("B release failed once")
        if (
            admitted
            and context is admitted[0][1]
            and attempts[key] == 1
        ):
            raise RuntimeError("cancelled C release failed once")
        return release(context)

    monkeypatch.setattr(
        BrowseContext, "release", fail_first_b_and_c
    )
    try:
        with pytest.raises(
            RuntimeError, match="previous Browse cleanup is pending"
        ):
            controller.begin_browse("/processed/browse.c.nxs")
        assert len(admitted) == 1
        request_c, context_c = admitted[0]
        operation_c = loader._active
        assert operation_c is not None
        assert operation_c.request is request_c
        assert operation_c.context is context_c

        # The failed call still admitted exact C.  Losing that identity is the
        # rejected-tip defect; generation C must never be silently reused.
        assert controller._cleanup_receipt is not None
        assert controller._cleanup_receipt.request is request_c
        assert controller.browse_pending is True
        assert controller._load_generation == request_c.load_generation
        assert request_c.load_generation == request_b.load_generation + 1
        assert controller.browse_context is context_b
        assert controller.selection.names(context_b)
        assert len(records_b) == len(publications_b) == 1
        assert len(context_c.record_store) == 1
        assert len(context_c.publication_store) == 1

        with pytest.raises(RuntimeError, match="cleanup"):
            controller.begin_browse("/processed/browse.d.nxs")
        assert len(admitted) == 1
        assert loader._active is operation_c
        assert loader._queued is None
        assert controller._load_generation == request_c.load_generation

        proof = DisplayRetirementReceipt(
            controller.run_identity, CleanupStatus.CLEANED
        )
        assert controller.apply_display_retirement(proof) is False
        assert controller.acquisition_context is acquisition
        assert loader._active is operation_c

        controller.poll_browse()
        assert len(context_c.record_store) == 0
        assert len(context_c.publication_store) == 0
        assert loader._active is None
        assert controller._cleanup_receipt is None
        assert controller.browse_pending is False

        request_d = controller.begin_browse(
            "/processed/browse.d.nxs"
        )
        assert request_d.load_generation == (
            request_c.load_generation + 1
        )
        assert len(admitted) == 2
        assert admitted[1][0] is request_d
        assert loader.owns_request(request_d)
        controller._invalidate_browse_request()
        assert not loader.owns_request(request_d)
        assert controller.browse_pending is False
        assert loader._close is None

        assert controller.apply_display_retirement(proof) is True
        assert len(records_b) == len(publications_b) == 0
        assert controller.acquisition_context is None
    finally:
        expected = (
            None if loader._active is None else loader._active.request
        )
        loader.close(expected)


def test_latest_queued_d_cleanup_drains_active_c_without_d_io_and_foreign_is_inert(
    monkeypatch,
) -> None:
    request_c = BrowseLoadRequest(
        "browse-c", 1, "/processed/browse.c.nxs"
    )
    _, context_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="browse.c",
        request=request_c,
    )
    operation_c = _ready_operation(request_c, context_c)
    io_facts: list[tuple[str, str]] = []

    def read_catalog(source, _cancelled):
        io_facts.append(("read", str(source)))

    loader = BrowseLoader(
        open_scan=lambda source: io_facts.append(
            ("open", str(source))
        ),
        open_reader=_empty_reader_factory(read_catalog),
    )
    loader._active = operation_c
    request_d = BrowseLoadRequest(
        "browse-d", 2, "/processed/browse.d.nxs"
    )
    real_release = loader.release_context
    hold_c = True
    failure = detach_exception(
        RuntimeError("queued predecessor cleanup retry"),
        "browse.release",
    )

    def gated_release(context):
        if context is context_c and hold_c:
            return BrowseCleanupReceipt(
                request_c,
                CleanupStatus.CLEANUP_PENDING,
                (failure,),
            )
        return real_release(context)

    monkeypatch.setattr(loader, "release_context", gated_release)
    try:
        assert loader.begin(request_d) is request_d
        assert loader._active is operation_c
        assert loader._queued is not None
        assert loader._queued.request is request_d
        assert io_facts == []

        for foreign in (
            replace(request_d),
            BrowseLoadRequest(
                "browse-foreign",
                request_d.load_generation,
                request_d.source_path,
            ),
        ):
            foreign_receipt = loader.cancel(foreign)
            assert type(foreign_receipt) is BrowseCleanupReceipt
            assert foreign_receipt.request is foreign
            assert (
                foreign_receipt.cleanup_status
                is CleanupStatus.CLEANUP_PENDING
            )
            assert loader._active is operation_c
            assert loader._queued is not None
            assert loader._queued.request is request_d
            assert loader.owns_request(request_d)
            assert io_facts == []

        pending = loader.cancel(request_d)
        assert pending.request is request_d
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_failures == (failure,)
        assert loader._active is operation_c
        assert loader._queued is not None
        assert loader._queued.request is request_d
        assert io_facts == []

        hold_c = False
        receipt = loader.cancel(request_d)
        # Exact latest D owns the nonterminal cancellation receipt even though
        # the resources being retired belong to active C.  D must never launch.
        assert loader._active is None
        assert loader._queued is None
        assert loader._worker is None
        assert len(context_c.record_store) == 0
        assert len(context_c.publication_store) == 0
        assert io_facts == []
        assert type(receipt) is BrowseCleanupReceipt
        assert receipt.request is request_d
        assert receipt.cleanup_status is CleanupStatus.CLEANED
        assert loader._close is None
    finally:
        hold_c = False
        expected = (
            request_d
            if loader.owns_request(request_d)
            else request_c
            if loader.owns_request(request_c)
            else None
        )
        loader.close(expected)


def test_cancel_d_after_promotion_check_performs_no_d_io(
    monkeypatch,
) -> None:
    request_c = BrowseLoadRequest(
        "browse-c", 1, "/processed/browse.c.nxs"
    )
    request_d = BrowseLoadRequest(
        "browse-d", 2, "/processed/browse.d.nxs"
    )
    operation_c = _BrowseOperation(request_c, Event())
    operation_c.cancelled.set()
    operation_c.outcome = BrowseLoadOutcome(
        request_c, BrowseLoadStatus.CANCELLED
    )
    operation_c.terminal = True
    io_facts: list[tuple[str, str]] = []

    def read_catalog(source, _cancelled):
        io_facts.append(("read", str(source)))

    loader = BrowseLoader(
        open_scan=lambda source: (
            io_facts.append(("open", str(source))) or object()
        ),
        open_reader=_empty_reader_factory(read_catalog),
    )
    loader._active = operation_c
    loader._queued = _BrowseOperation(request_d, Event())
    launch_entered = Event()
    continue_launch = Event()
    original_launch = loader._launch

    class TrackingPath:
        def __init__(self, source):
            self.source = str(source)

        def is_file(self):
            io_facts.append(("path", self.source))
            return True

    def canonical(source):
        io_facts.append(("canonical", str(source)))
        return "browse.d"

    def gated_launch(request, **kwargs):
        assert request is request_d
        launch_entered.set()
        if not continue_launch.wait(timeout=10.0):
            raise TimeoutError("post-promotion launch gate timed out")
        return original_launch(request, **kwargs)

    monkeypatch.setattr(
        loader_module, "canonical_browse_scan_key", canonical
    )
    monkeypatch.setattr(loader_module, "Path", TrackingPath)
    monkeypatch.setattr(loader, "_launch", gated_launch)
    promotion = Thread(target=loader._progress)
    promotion.start()
    try:
        assert launch_entered.wait(timeout=5.0)
        pending = loader.cancel(request_d)
        assert pending.request is request_d
        assert (
            pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        )

        continue_launch.set()
        promotion.join(timeout=5.0)
        assert not promotion.is_alive()
        worker = loader._worker
        if worker is not None:
            worker.join(timeout=5.0)

        # Cancellation happened after promotion's cancellation check but
        # before _launch.  No part of D's load path may execute afterward.
        assert io_facts == []
    finally:
        continue_launch.set()
        promotion.join(timeout=5.0)
        worker = loader._worker
        if worker is not None:
            worker.join(timeout=5.0)
        if loader.owns_request(request_d):
            loader.cancel(request_d)
        expected = (
            request_d
            if loader.owns_request(request_d)
            else request_c
            if loader.owns_request(request_c)
            else None
        )
        loader.close(expected)


def test_cleaned_d_waits_for_terminal_c_worker_epilogue_and_retry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path_c = tmp_path / "c.nexus"
    path_d = tmp_path / "d.nexus"
    path_e = tmp_path / "e.nexus"
    path_c.write_bytes(b"c")
    path_d.write_bytes(b"d")
    path_e.write_bytes(b"e")
    load_entered = Event()
    finish_load = Event()
    epilogue_entered = Event()
    finish_epilogue = Event()

    def blocked_catalog(source, _cancelled):
        if Path(source) == path_c:
            load_entered.set()
            if not finish_load.wait(timeout=10.0):
                raise TimeoutError("C load gate timed out")

    monkeypatch.setattr(
        loader_module,
        "canonical_browse_scan_key",
        lambda source: Path(source).stem,
    )

    loader = BrowseLoader(
        open_scan=lambda _source: object(),
        open_reader=_empty_reader_factory(blocked_catalog),
    )
    request_c = BrowseLoadRequest("browse-c", 1, str(path_c))
    request_d = BrowseLoadRequest("browse-d", 2, str(path_d))
    request_e = BrowseLoadRequest("browse-e", 3, str(path_e))
    original_progress = loader._progress
    worker_c = None

    def gated_progress(*, retire_cancelled=False):
        if current_thread() is worker_c:
            epilogue_entered.set()
            if not finish_epilogue.wait(timeout=10.0):
                raise TimeoutError("C worker epilogue gate timed out")
        return original_progress(retire_cancelled=retire_cancelled)

    monkeypatch.setattr(loader, "_progress", gated_progress)
    loader.begin(request_c)
    worker_c = loader._worker
    assert worker_c is not None
    assert load_entered.wait(timeout=5.0)
    loader.begin(request_d)
    first = loader.cancel(request_d)
    assert first.request is request_d
    assert first.cleanup_status is CleanupStatus.CLEANUP_PENDING
    finish_load.set()
    assert epilogue_entered.wait(timeout=5.0)
    assert worker_c.is_alive()

    accepted_e = None
    blocked_e = False
    try:
        second = loader.cancel(request_d)
        try:
            accepted_e = loader.begin(request_e)
        except RuntimeError as error:
            blocked_e = "cleanup" in str(error).lower()

        # CLEANED is a quiescence proof: it cannot precede C's worker exit,
        # clear D's exact cleanup identity, or admit a replacement E.
        assert (
            second.request is request_d,
            second.cleanup_status,
            worker_c.is_alive(),
            loader.owns_request(request_d),
            blocked_e,
            accepted_e,
        ) == (
            True,
            CleanupStatus.CLEANUP_PENDING,
            True,
            True,
            True,
            None,
        )

        finish_epilogue.set()
        worker_c.join(timeout=5.0)
        assert not worker_c.is_alive()
        cleaned = loader.cancel(request_d)
        assert cleaned.request is request_d
        assert cleaned.cleanup_status is CleanupStatus.CLEANED
        assert loader.begin(request_e) is request_e
    finally:
        finish_load.set()
        finish_epilogue.set()
        worker_c.join(timeout=5.0)
        worker = loader._worker
        if worker is not None and worker is not worker_c:
            worker.join(timeout=5.0)
        expected = (
            loader._queued.request
            if loader._queued is not None
            else (
                None
                if loader._active is None
                else loader._active.request
            )
        )
        loader.close(expected)


def test_close_cleans_pending_c_before_live_b_and_caches_exact_c_identity(
    monkeypatch,
) -> None:
    controller, _lifecycle, _executor, port, _acquisition = (
        _running_controller()
    )
    controller.pause()
    _request_b, context_b = _select_browse(
        controller, port, scan_key="browse.b"
    )
    records_b = context_b.record_store
    publications_b = context_b.publication_store

    request_c = BrowseLoadRequest(
        "browse-c",
        context_b.load_generation + 1,
        "/processed/browse.c.nxs",
    )
    _, context_c = _browse(
        request_c.token,
        request_c.load_generation,
        scan_key="browse.c",
        request=request_c,
    )
    loader = BrowseLoader()
    loader._active = _ready_operation(request_c, context_c)
    controller._browse_loader = loader
    controller._browse_request = request_c
    controller._load_generation = request_c.load_generation

    release_context = loader.release_context
    release_order: list[BrowseContext] = []

    def record_release(context):
        release_order.append(context)
        return release_context(context)

    monkeypatch.setattr(loader, "release_context", record_release)
    try:
        first = controller.close()
        assert first.request is request_c
        assert first.cleanup_status is CleanupStatus.CLEANED
        assert release_order == [context_c, context_b]
        assert len(context_c.record_store) == 0
        assert len(context_c.publication_store) == 0
        assert len(records_b) == len(publications_b) == 0
        assert controller.selection is None

        duplicate = controller.close()
        assert duplicate is first
        assert duplicate.request is request_c
        assert release_order == [context_c, context_b]
    finally:
        controller.close()
