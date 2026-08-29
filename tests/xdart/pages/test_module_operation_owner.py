from __future__ import annotations

import ast
import copy
from pathlib import Path
from threading import Event
import threading
import time

import pytest

import xdart.gui.pages.operation_owner as owner_api
from xdart.gui.pages.operation_owner import (
    OperationCancelled,
    OperationIdentity,
    OperationTerminalStatus,
    SingleWorkerOwner,
)
from xdart.gui.pages.values import PageCleanup


def _poll_until(owner, identity, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        update = owner.poll(identity)
        if update is not None:
            return update
        time.sleep(0.001)
    raise AssertionError("operation update did not arrive")


def test_owner_is_stdlib_only_and_page_architecture_safe():
    path = Path(__file__).resolve().parents[3] / "src/xdart/gui/pages/operation_owner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        name.startswith(("PySide6", "xrd_tools", "xdart.gui.tabs"))
        for name in imports
    )


def test_owner_refuses_lifecycle_aliases():
    owner = SingleWorkerOwner(lambda *_args: object())
    with pytest.raises(TypeError, match="not copyable"):
        copy.copy(owner)
    with pytest.raises(TypeError, match="not copyable"):
        copy.deepcopy(owner)
    assert owner.close().status is PageCleanup.CLEAN


@pytest.mark.parametrize(
    "join_timeout",
    (float("nan"), float("inf"), 1e20),
)
def test_owner_refuses_nonfinite_join_timeout(join_timeout):
    with pytest.raises(TypeError, match="nonnegative number"):
        SingleWorkerOwner(lambda *_args: object(), join_timeout=join_timeout)


def test_exact_identity_latest_progress_terminal_once_and_reuse():
    entered = Event()
    release = Event()
    calls = []
    result = object()

    def runner(request, cancel, publish):
        calls.append((request, cancel))
        assert publish("prepare", 0, 3)
        assert publish("prepare", 1, 3)
        assert not publish("bad-total", 2, 4)
        assert not publish("regression", 0, 3)
        assert publish("compute", 2, 3)
        entered.set()
        assert release.wait(2)
        assert publish("commit", 3, 3)
        return result

    owner = SingleWorkerOwner(runner)
    request = object()
    identity = owner.begin(request)
    assert type(identity) is OperationIdentity
    assert identity.request is request
    assert owner.begin(object()) is None
    assert entered.wait(2)

    foreign = OperationIdentity(identity.serial, request)
    assert owner.poll(foreign) is None
    assert owner.cancel(foreign) is False
    progress = _poll_until(owner, identity).progress
    assert progress is not None
    assert (progress.revision, progress.stage, progress.completed, progress.total) == (
        3, "compute", 2, 3,
    )
    assert owner.poll(identity) is None

    release.set()
    terminal = _poll_until(owner, identity).terminal
    assert terminal is not None
    assert terminal.identity is identity
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert terminal.payload is result
    assert owner.poll(identity) is None
    assert owner.busy is False
    assert calls[0][0] is request

    next_identity = owner.begin(object())
    assert next_identity.serial == identity.serial + 1
    release.set()
    _poll_until(owner, next_identity)
    assert owner.close().status is PageCleanup.CLEAN


def test_cancel_is_exact_once_and_return_wins_a_cancel_race():
    entered = Event()
    release = Event()
    result = object()

    def runner(_request, cancel, _publish):
        entered.set()
        assert release.wait(2)
        assert cancel.is_set()
        return result

    owner = SingleWorkerOwner(runner)
    identity = owner.begin(object())
    assert entered.wait(2)
    assert owner.cancel(identity) is True
    assert owner.cancel(identity) is False
    release.set()
    terminal = _poll_until(owner, identity).terminal
    assert terminal.status is OperationTerminalStatus.RETURNED
    assert terminal.payload is result
    assert owner.close().status is PageCleanup.CLEAN


def test_cooperative_cancel_and_failures_are_detached_terminals():
    def cancelled(_request, _cancel, _publish):
        raise OperationCancelled("stop")

    owner = SingleWorkerOwner(cancelled)
    identity = owner.begin(object())
    terminal = _poll_until(owner, identity).terminal
    assert terminal.status is OperationTerminalStatus.CANCELLED
    assert terminal.payload is None

    def failed(_request, _cancel, _publish):
        raise SystemExit("detached")

    failing = SingleWorkerOwner(failed)
    identity = failing.begin(object())
    terminal = _poll_until(failing, identity).terminal
    assert terminal.status is OperationTerminalStatus.FAILED
    assert terminal.payload is None
    assert terminal.failure_module == "builtins"
    assert terminal.failure_type == "SystemExit"
    assert terminal.failure_message == "detached"
    assert owner.close().status is PageCleanup.CLEAN
    assert failing.close().status is PageCleanup.CLEAN


def test_close_is_retryable_join_and_seals_polling():
    entered = Event()
    release = Event()

    def runner(_request, cancel, publish):
        assert publish("held", 0, 1)
        entered.set()
        assert release.wait(2)
        assert cancel.is_set()
        raise OperationCancelled()

    owner = SingleWorkerOwner(runner)
    identity = owner.begin(object())
    assert entered.wait(2)
    first = owner.close()
    assert first.status is PageCleanup.PENDING
    assert owner.poll(identity) is None
    assert owner.cancel(identity) is False
    assert owner.begin(object()) is None
    second = owner.close()
    assert second.status is PageCleanup.PENDING
    release.set()
    deadline = time.monotonic() + 2
    while True:
        final = owner.close()
        if final.status is PageCleanup.CLEAN:
            break
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert owner.close() is final
    assert owner.busy is False


def test_close_during_thread_start_cannot_hand_back_an_unowned_worker(
    monkeypatch: pytest.MonkeyPatch,
):
    start_entered = Event()
    start_release = Event()
    runner_entered = Event()

    class HeldStartThread(threading.Thread):
        def start(self):
            start_entered.set()
            assert start_release.wait(2)
            return super().start()

    monkeypatch.setattr(owner_api, "Thread", HeldStartThread)

    def runner(_request, cancel, _publish):
        runner_entered.set()
        assert cancel.is_set()
        raise OperationCancelled()

    owner = SingleWorkerOwner(runner)
    identities = []
    caller = threading.Thread(
        target=lambda: identities.append(owner.begin(object())),
        daemon=False,
    )
    caller.start()
    assert start_entered.wait(2)
    assert owner.close().status is PageCleanup.PENDING
    start_release.set()
    caller.join(2)
    assert not caller.is_alive()
    assert type(identities[0]) is OperationIdentity
    assert runner_entered.wait(2)

    deadline = time.monotonic() + 2
    while owner.close().status is PageCleanup.PENDING:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert owner.close().status is PageCleanup.CLEAN


def test_thread_start_failure_delivers_one_failed_terminal_and_closes(
    monkeypatch: pytest.MonkeyPatch,
):
    class FailingStartThread(threading.Thread):
        def start(self):
            raise RuntimeError("start refused")

    monkeypatch.setattr(owner_api, "Thread", FailingStartThread)
    owner = SingleWorkerOwner(lambda *_args: object())
    identity = owner.begin(object())
    terminal = owner.poll(identity).terminal
    assert terminal.status is OperationTerminalStatus.FAILED
    assert terminal.failure_type == "RuntimeError"
    assert terminal.failure_message == "start refused"
    assert owner.poll(identity) is None
    assert owner.close().status is PageCleanup.CLEAN


def test_thread_start_then_raise_retains_and_joins_the_live_worker(
    monkeypatch: pytest.MonkeyPatch,
):
    entered = Event()
    release = Event()

    class StartedThenRaisedThread(threading.Thread):
        def start(self):
            super().start()
            raise RuntimeError("wrapper failed after start")

    monkeypatch.setattr(owner_api, "Thread", StartedThenRaisedThread)

    def runner(_request, cancel, _publish):
        entered.set()
        assert release.wait(2)
        assert cancel.is_set()
        raise OperationCancelled()

    owner = SingleWorkerOwner(runner)
    identity = owner.begin(object())
    assert type(identity) is OperationIdentity
    assert entered.wait(2)
    assert owner.close().status is PageCleanup.PENDING
    assert owner.busy is True
    release.set()
    deadline = time.monotonic() + 2
    while owner.close().status is PageCleanup.PENDING:
        assert time.monotonic() < deadline
        time.sleep(0.001)
    assert owner.close().status is PageCleanup.CLEAN
    assert owner.busy is False


def test_concurrent_close_callers_share_one_exact_clean_receipt(
    monkeypatch: pytest.MonkeyPatch,
):
    joins = threading.Barrier(2)

    class PairedJoinThread(threading.Thread):
        def join(self, timeout=None):
            joins.wait(timeout=2)
            return super().join(timeout)

    monkeypatch.setattr(owner_api, "Thread", PairedJoinThread)

    def runner(_request, cancel, _publish):
        assert cancel.wait(2)
        raise OperationCancelled()

    owner = SingleWorkerOwner(runner, join_timeout=2.0)
    assert type(owner.begin(object())) is OperationIdentity
    receipts = []
    callers = [
        threading.Thread(target=lambda: receipts.append(owner.close()))
        for _index in range(2)
    ]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(2)
        assert not caller.is_alive()
    assert len(receipts) == 2
    assert receipts[0] is receipts[1]
    assert receipts[0].status is PageCleanup.CLEAN
