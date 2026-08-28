"""Focused lifecycle oracle for the one P3 page-owned operation slot."""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, is_dataclass
import os
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.operation_values import (
    OperationCleanupReceipt,
    OperationContextStamp,
    OperationIdentity,
    OperationPending,
    OperationProgress,
    OperationTerminal,
    OperationTerminalStatus,
    OperationUpdate,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.presentation_background import PresentationBackgroundOwner
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.readiness import ControlAction, SectionId
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.reduction import DisplayBackgroundPlan


@dataclass(frozen=True, slots=True)
class _Job:
    label: str = "job"


@dataclass
class _MutablePayload:
    value: int = 1


class _HeldOperationOwner:
    def __init__(self) -> None:
        self.held = True
        self.close_calls = 0

    def close(self) -> SimpleNamespace:
        self.close_calls += 1
        return SimpleNamespace(cleanup_status=(
            CleanupStatus.CLEANUP_PENDING
            if self.held else CleanupStatus.CLEANED
        ))


def _page() -> ScatteringWorkspace:
    return ScatteringWorkspace(
        intents=RunIntentStore(RunIntent()),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )


def _held_body(entered: Event, release: Event, seen=None):
    def body(job, identity, _cancel, _publish):
        if seen is not None:
            seen.value = job
        entered.set()
        if not release.wait(2):
            return OperationTerminal(
                identity, OperationTerminalStatus.FAILED, "release timeout"
            )
        return OperationTerminal(identity, OperationTerminalStatus.RETURNED)
    return body


def _start_held(slot: OperationSlot, job: _Job | None = None):
    entered, release = Event(), Event()
    identity = slot._begin(
        _Job() if job is None else job,
        OperationContextStamp(0),
        _held_body(entered, release),
    )
    assert type(identity) is OperationIdentity
    assert entered.wait(2)
    return identity, release


def _release_and_poll(slot: OperationSlot, identity, release):
    worker = slot._worker
    assert worker is not None
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    update = slot.poll(identity)
    assert update is not None
    assert update.terminal.status is OperationTerminalStatus.RETURNED
    return update


def _background():
    axis = np.arange(3.0)
    values = np.array([1.0, 2.0, 3.0])
    plan = DisplayBackgroundPlan(
        "integrated_1d", ("source",), ((3,),), (((3,),),), (("q",),))
    owner = PresentationBackgroundOwner(capacity_bytes=536_870_912)
    reservation = owner.reserve(
        plan, ((values, axis),), stamp=OperationContextStamp(0, "context", 1),
        active_key=("context", 1, "integrated_1d"),
        projection_keys=(1,), projection_indices=(0,))
    assert reservation == 1
    return owner, plan, reservation


def _arrays_free(value) -> bool:
    if isinstance(value, np.ndarray): return False
    if is_dataclass(value) and not isinstance(value, type):
        return all(_arrays_free(getattr(value, member.name)) for member in fields(value))
    if isinstance(value, (tuple, list, dict)):
        items = value.items() if isinstance(value, dict) else value
        return all(_arrays_free(item) for item in items)
    return True


def test_page_close_waits_for_held_operation_owner() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = _page()
    held = _HeldOperationOwner()
    page._workspace_operations._slot = held
    try:
        pending = page.close_workspace()
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert held.close_calls == 1
        held.held = False
        cleaned = page.close_workspace()
        assert cleaned.cleanup_status is CleanupStatus.CLEANED
        assert held.close_calls == 2
    finally:
        held.held = False
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_held_runner_returns_immediately_and_uses_one_common_worker() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    heartbeat = Event()
    QtCore.QTimer.singleShot(0, heartbeat.set)
    slot = OperationSlot()
    identity, release = _start_held(slot)
    worker = slot._worker
    peer = slot._begin(_Job("peer"), OperationContextStamp(0), _held_body(Event(), release))
    assert peer is None and slot._worker is worker
    assert slot._worker is not None and slot._worker.is_alive()
    assert not release.is_set()
    qapp.processEvents()
    assert heartbeat.is_set()
    _release_and_poll(slot, identity, release)


def test_common_worker_is_non_daemon_joinable_and_has_no_peer_owner() -> None:
    slot = OperationSlot()
    identity, release = _start_held(slot)
    worker = slot._worker
    workers = [value for value in vars(slot).values()
               if isinstance(value, Thread)]
    assert workers == [worker]
    assert worker is not None and worker.daemon is False and worker.is_alive()
    _release_and_poll(slot, identity, release)
    worker.join(0)


def test_worker_receives_identical_frozen_object() -> None:
    slot, seen, entered, release = (
        OperationSlot(), SimpleNamespace(value=None), Event(), Event()
    )
    job = _Job("same-object")
    identity = slot._begin(
        job, OperationContextStamp(0), _held_body(entered, release, seen)
    )
    assert type(identity) is OperationIdentity and entered.wait(2)
    assert seen.value is job
    _release_and_poll(slot, identity, release)


def test_progress_is_replace_in_place_arrays_free_and_identity_qualified() -> None:
    slot = OperationSlot()
    first, proceed, second, release = Event(), Event(), Event(), Event()
    def body(_job, identity, _cancel, publish):
        publish("", 0, 0); publish("read", -1, 3); publish("read", 4, 3)
        publish("read", 1, 3); first.set(); proceed.wait(2)
        publish("read", 2, 3); second.set(); release.wait(2)
        return OperationTerminal(identity, OperationTerminalStatus.RETURNED)
    identity = slot._begin(_Job(), OperationContextStamp(0), body)
    assert type(identity) is OperationIdentity and first.wait(2)
    progress1 = slot._progress
    proceed.set(); assert second.wait(2)
    progress2 = slot._progress
    assert progress2 is not progress1 and slot._progress is progress2
    assert progress2.identity is identity and progress2.revision == 2
    assert all(type(getattr(progress2, item.name)) in {OperationIdentity, str, int}
               for item in fields(progress2))
    assert not any(isinstance(value, (list, dict, set))
                   for value in vars(slot).values())
    update = slot.poll(identity)
    assert update is not None and update.progress is progress2
    assert slot.poll(identity) is None
    _release_and_poll(slot, identity, release)


def test_progress_rejects_regression_and_reused_revision() -> None:
    slot, ready, release, snapshots = OperationSlot(), Event(), Event(), []
    def body(_job, identity, _cancel, publish):
        publish("read", 2, 4); snapshots.append(slot._progress)
        publish("read", 1, 4); snapshots.append(slot._progress)
        publish("read", 2, 4); snapshots.append(slot._progress)
        ready.set(); release.wait(2)
        return OperationTerminal(identity, OperationTerminalStatus.RETURNED)
    identity = slot._begin(_Job(), OperationContextStamp(0), body)
    assert type(identity) is OperationIdentity and ready.wait(2)
    assert snapshots[1] is snapshots[0]
    assert [snapshots[0].revision, snapshots[2].revision] == [1, 2]
    assert snapshots[2].completed == 2
    _release_and_poll(slot, identity, release)


def test_foreign_and_late_poll_cancel_are_inert() -> None:
    slot, entered, seen = OperationSlot(), Event(), SimpleNamespace(event=None)
    def body(_job, identity, cancel_event, _publish):
        seen.event = cancel_event; entered.set()
        if not cancel_event.wait(2):
            return OperationTerminal(
                identity, OperationTerminalStatus.FAILED, "cancel timeout")
        return OperationTerminal(identity, OperationTerminalStatus.CANCELLED)
    identity = slot._begin(_Job(), OperationContextStamp(0), body)
    assert type(identity) is OperationIdentity and entered.wait(2)
    foreign = OperationIdentity(identity.serial)
    assert foreign == identity and foreign is not identity
    assert slot.poll(foreign) is None and slot.cancel(foreign) is False
    assert seen.event is slot._cancel_event and not seen.event.is_set()
    assert slot.cancel(identity) is True and slot.cancel(identity) is False
    worker = slot._worker; assert worker is not None
    worker.join(2); update = slot.poll(identity)
    assert update.terminal.status is OperationTerminalStatus.CANCELLED
    assert slot.poll(identity) is None and slot.cancel(identity) is False


def test_publication_seal_linearizes_cancel_and_close_on_the_common_lock() -> None:
    slot = OperationSlot()
    identity, release = _start_held(slot)
    assert slot._seal_publication(OperationIdentity(identity.serial)) is False
    assert slot._seal_publication(identity) is True
    assert slot._seal_publication(identity) is False
    assert slot.cancel(identity) is False
    pending = slot.close()
    assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert pending.cancel_accepted is False
    worker = slot._worker; release.set(); worker.join(2)
    assert slot.close().terminal.status is OperationTerminalStatus.RETURNED


def test_terminal_payload_is_one_detached_frozen_dataclass() -> None:
    identity = OperationIdentity(1)
    payload = _Job("proof")
    assert OperationTerminal(
        identity, OperationTerminalStatus.RETURNED, payload=payload
    ).payload is payload
    for invalid in (_MutablePayload(), _Job, object()):
        with pytest.raises((TypeError, ValueError)):
            OperationTerminal(
                identity, OperationTerminalStatus.RETURNED, payload=invalid
            )


def test_context_aba_latches_stale_and_preserves_terminal_truth() -> None:
    slot = OperationSlot()
    identity, release = _start_held(slot)
    original = OperationContextStamp(0)
    slot.observe_stamp(OperationContextStamp(0, "browse-token", 1))
    slot.observe_stamp(original)
    update = _release_and_poll(slot, identity, release)
    assert update.stale is True
    assert update.terminal.status is OperationTerminalStatus.RETURNED


def test_close_reuses_same_worker_until_clean() -> None:
    slot = OperationSlot()
    identity, release = _start_held(slot)
    worker, cancel_event = slot._worker, slot._cancel_event
    pending1 = slot.close(); pending2 = slot.close()
    assert pending1.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert pending2.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert pending1.identity is identity and pending1.cancel_accepted is True
    assert slot._worker is worker and slot._cancel_event is cancel_event
    release.set(); worker.join(2)
    cleaned = slot.close()
    assert cleaned.cleanup_status is CleanupStatus.CLEANED
    assert cleaned.worker_identity == id(worker)
    assert cleaned.terminal.status is OperationTerminalStatus.RETURNED
    assert slot.close() is cleaned and slot.owned is False


def test_terminal_delivery_and_effect_are_exactly_once_after_close() -> None:
    def returned(_job, identity, _cancel, _publish):
        return OperationTerminal(identity, OperationTerminalStatus.RETURNED)
    polled = OperationSlot()
    identity = polled._begin(_Job(), OperationContextStamp(0), returned)
    worker = polled._worker; worker.join(2)
    update = polled.poll(identity)
    assert update.terminal is not None and polled.poll(identity) is None
    assert polled.close().terminal is None
    closed = OperationSlot()
    identity2 = closed._begin(_Job(), OperationContextStamp(0), returned)
    worker2 = closed._worker; worker2.join(2)
    receipt = closed.close()
    assert receipt.terminal is not None and receipt.identity is identity2
    assert closed.close() is receipt and closed.poll(identity2) is None


def test_background_slot_values_are_array_free_and_stage_exact_result() -> None:
    owner, plan, reservation = _background()
    slot = OperationSlot()
    identity = slot.begin_background(
        plan, OperationContextStamp(0, "context", 1), owner, reservation)
    assert type(identity) is OperationIdentity
    worker = slot._worker; assert worker is not None
    worker.join(2); assert not worker.is_alive()
    assert owner.phase == "STAGED" and owner._result is not None
    assert _arrays_free(slot._frozen) and _arrays_free(slot._progress)
    assert _arrays_free(slot._terminal) and _arrays_free(slot._abort_fact)
    staged = owner._result
    update = slot.poll(identity)
    assert update is not None and update.terminal.payload.result_identity == staged.result_identity
    assert owner.phase == "STAGED" and slot.owned is False
    assert _arrays_free(update) and _arrays_free(slot.close())
    owner.abort(reservation, "TEST_RELEASE")


def test_clear_while_reserved_remains_pending_until_worker_relinquishes(
    monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering import presentation_background as owner_module

    owner, plan, reservation = _background()
    entered, release = Event(), Event()
    runner = owner_module.run_display_background
    def held(*args, **kwargs):
        entered.set(); assert release.wait(2); return runner(*args, **kwargs)
    monkeypatch.setattr(owner_module, "run_display_background", held)
    slot = OperationSlot()
    identity = slot.begin_background(
        plan, OperationContextStamp(0, "context", 1), owner, reservation)
    assert type(identity) is OperationIdentity and entered.wait(2)
    roots = owner._contributors
    assert not owner.release() and owner.phase == "CLEANUP_PENDING"
    assert owner._contributors is roots and slot.cancel(identity)
    release.set(); worker = slot._worker; worker.join(2)
    update = slot.poll(identity)
    assert update.terminal.status is OperationTerminalStatus.CANCELLED
    assert owner.phase == "RELEASED" and owner._contributors == ()


def test_post_stage_terminal_construction_failure_finalizes_outside_lock(
    monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.adapters import external_operation as slot_module

    owner, plan, reservation = _background()
    slot, finalized = OperationSlot(), []
    real_finalize = owner.finalize
    def observe(token, outcome):
        outside = slot._lock.acquire(blocking=False)
        if outside: slot._lock.release()
        finalized.append((token, outcome, outside)); real_finalize(token, outcome)
    owner.finalize = observe
    def refuse_terminal(*_args, **_kwargs):
        raise MemoryError("terminal allocation failed")
    monkeypatch.setattr(slot_module, "OperationTerminal", refuse_terminal)
    identity = slot.begin_background(
        plan, OperationContextStamp(0, "context", 1), owner, reservation)
    worker = slot._worker; worker.join(2)
    assert owner.phase == "STAGED"
    assert slot._terminal is None and slot._abort_fact[0] == "ABORTED_WITHOUT_TERMINAL"
    assert slot.poll(identity) is None
    assert finalized == [(reservation, "ABORTED_WITHOUT_TERMINAL", True)]
    assert owner.phase == "RELEASED" and not slot.owned


def test_real_page_timer_polling_and_close_include_operation_owner() -> None:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page = _page()
    try:
        entered, release = Event(), Event()
        identity = page._begin_operation(_Job(), _held_body(entered, release))
        assert type(identity) is OperationIdentity and entered.wait(2)
        assert page._run_timer.isActive() and page._polling_needed()
        worker = page._workspace_operations._slot._worker
        release.set(); worker.join(2); page._drain_executor()
        assert page._workspace_operations.owned is False
        assert not page._run_timer.isActive()
        entered2, release2 = Event(), Event()
        page._begin_operation(_Job("close"), _held_body(entered2, release2))
        assert entered2.wait(2)
        assert page.close_workspace().cleanup_status is CleanupStatus.CLEANUP_PENDING
        worker2 = page._workspace_operations._slot._worker
        release2.set(); worker2.join(2)
        assert page.close_workspace().cleanup_status is CleanupStatus.CLEANED
    finally:
        page.close_workspace(); page.deleteLater(); qapp.processEvents()


def test_operation_surface_and_owner_censuses_remain_bounded() -> None:
    root = Path(__file__).parents[3]
    slot_path = root / "src/xdart/gui/tabs/scattering/adapters/external_operation.py"
    page_path = root / "src/xdart/gui/tabs/scattering/page.py"
    owner_path = root / "src/xdart/gui/tabs/scattering/workspace_operations.py"
    values_path = root / "src/xdart/gui/tabs/scattering/operation_values.py"
    slot_text, page_text, owner_text, values_text = (
        slot_path.read_text(), page_path.read_text(), owner_path.read_text(),
        values_path.read_text(),
    )
    tree = ast.parse(slot_text)
    calls = [node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    methods = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert calls.count("Thread") == calls.count("Event") == 1
    assert calls.count("Queue") == 1  # Average cleanup command handoff
    assert not ({"Timer", "ThreadPoolExecutor", "Process"} & set(calls))
    assert "_begin" in methods and "begin" not in methods
    assert page_text.count("OperationSlot()") == 1  # analysis only
    assert owner_text.count("OperationSlot()") == 1  # experiment only
    assert page_text.count("QtCore.QTimer(") == 3
    assert page_text.count("ThreadPoolExecutor(max_workers=1)") == 2
    assert page_text.count("deque(maxlen=1)") == 1
    assert page_text.count("ScatteringWorkspace._observe_operation_stamp") == 4
    assert page_text.count("begin_calibrate(") == page_text.count("begin_mask(") == 1
    slot_owner = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "OperationSlot")
    slot_imports = {alias.name.split(".", 1)[0]
                    for node in ast.walk(slot_owner) if isinstance(node, ast.Import)
                    for alias in node.names}
    slot_imports.update(node.module.split(".", 1)[0]
                        for node in ast.walk(slot_owner)
                        if isinstance(node, ast.ImportFrom) and node.module)
    assert not ({"numpy", "h5py", "pyFAI"} & slot_imports)
    value_types = (OperationCleanupReceipt, OperationContextStamp,
                   OperationIdentity, OperationPending, OperationProgress,
                   OperationTerminal, OperationUpdate)
    assert values_text.count("@dataclass(frozen=True, slots=True") == 7
    assert all(value.__dataclass_params__.frozen and hasattr(value, "__slots__")
               for value in value_types)
    state = project_controls(
        RunIntentStore(RunIntent()).snapshot(), None, RunPhase.IDLE
    )
    actions = state.actions_for(SectionId.EXPERIMENT)
    assert tuple(action.action for action in actions) == (
        ControlAction.CALIBRATE, ControlAction.MAKE_MASK
    )
    assert all(not action.enabled and action.reason for action in actions)
