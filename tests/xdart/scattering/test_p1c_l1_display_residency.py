"""Finite P1-C L1 oracles for canonical projection and display residency."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from pyqtgraph import QtWidgets

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_comprehensive_live import (
    _batch_intent,
    _write_stack,
)
from tests.xdart.scattering.test_p1b_output_graph import (
    _TERMINAL,
    _drain_until,
    _start,
)
from xdart.gui.tabs.scattering.adapters import dynamic_output
from xdart.gui.tabs.scattering.adapters.dynamic_output import DynamicOutputAdapter
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.display_residency import DisplayResidencyLimits
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.modules.frame_publication import FramePublication, PublicationStore
from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
from xrd_tools.session import (
    DynamicRunAccounting,
    FrameRecordStore,
    Light1DCustodyState,
    Light1DLeaseState,
    SessionResourceAuthority,
)
from xrd_tools.session.intent_store import RunIntentStore


def _owner(state: RunDisplayState):
    return state.add_artifact(
        Path("/run/p1c-l1.nexus"),
        "p1c-l1",
        mask=None,
        mask_saturation=False,
        measurement_mode="Standard",
    )


def _record(label: int) -> tuple[FrameRecord, FramePublication, str]:
    source = f"/source/p1c_{label:04d}.tif"
    view = FrameView(
        label=label,
        axis_1d=Axis("q", "1/angstrom", np.linspace(0.1, 1.0, 8)),
        intensity_1d=np.arange(8, dtype=float) + label,
        axis_2d_x=Axis("q", "1/angstrom", np.linspace(0.1, 1.0, 4)),
        axis_2d_y=Axis("chi", "degree", np.linspace(-1.0, 1.0, 3)),
        intensity_2d=np.arange(12, dtype=float).reshape(3, 4) + label,
        two_d_kind=TwoDKind.Q_CHI,
        raw=np.full((4, 4), label, dtype=np.uint16),
        thumbnail=np.full((2, 2), label, dtype=float),
        source_path=source,
        source_frame_index=label,
    )
    record = FrameRecord.from_view(view)
    source_identity = f"{source}#{label}"
    return (
        record,
        FramePublication(
            view,
            record=record,
            source_identity=source_identity,
            scan_key="p1c-l1",
        ),
        source_identity,
    )


def _publish(
    state: RunDisplayState,
    owner,
    label: int,
    *,
    hydratable: tuple[tuple[str, str], ...],
    durable: tuple[tuple[str, str], ...],
    dropped: tuple[tuple[str, str], ...] = (),
):
    record, publication, source_identity = _record(label)
    owner.records.upsert(record, source_identity=source_identity)
    owner.records.replace_projection(
        label,
        hydratable=hydratable,
        durable=durable,
        dropped=dropped,
    )
    delta = state.append_navigation(owner.source_scan, str(owner.artifact), label)
    state.retain_frame(
        owner,
        delta.appended,
        record,
        publication,
        source_identity=source_identity,
        frame_mask_qualified=False,
    )
    return record, delta.appended


def _p2_construction_case(monkeypatch, root, case_id, terminal_run):
    """Run one independent P2 construction seam and return its full ledger."""

    cases = {
        "store-hooks": (RunDisplayState, "bind_light_1d", False, True),
        "display-lease": (dynamic_output, "Light1DCustodySlot", False, True),
        "slot-pending": (RunDisplayState, "bind_light_custody", False, True),
        "custody": (RunDisplayState, "bind_light_custody", True, False),
        "accounting": (DynamicRunAccounting, "bind_light_1d", True, False),
        "subscription-before": (
            RunDisplayState, "bind_light_subscription", False, False,
        ),
        "subscription-after": (
            RunDisplayState, "bind_light_subscription", True, False,
        ),
    }
    target_owner, method, after, pre_slot = cases[case_id]
    with monkeypatch.context() as seam:
        seen = {}
        real_acquire = dynamic_output.acquire_light_1d_retention
        real_open = dynamic_output.open_headless_scan_session
        real_hooks = PublicationStore.light_1d_cleanup_hooks
        real_slot = dynamic_output.Light1DCustodySlot
        real_transfer = DynamicOutputAdapter._release_construction_custody
        real_release = SessionResourceAuthority._release
        original = getattr(target_owner, method)
        cleanup_blocked = [pre_slot]

        def capture_acquire(*args, **kwargs):
            seen["lease"] = real_acquire(*args, **kwargs)
            return seen["lease"]

        def capture_open(*args, **kwargs):
            seen["session"] = real_open(*args, **kwargs)
            seen["accounting"] = kwargs["accounting"]
            return seen["session"]

        def capture_hooks(self, *args, **kwargs):
            seen["hooks"] = real_hooks(self, *args, **kwargs)
            return seen["hooks"]

        def capture_slot(*args, **kwargs):
            seen["slot"] = real_slot(*args, **kwargs)
            return seen["slot"]

        def capture_transfer(self, *args, **kwargs):
            seen["adapter"] = self
            return real_transfer(self, *args, **kwargs)

        def fail_boundary(*args, **kwargs):
            if after:
                original(*args, **kwargs)
            adapter = seen.get("adapter")
            seen["graph_at_failure"] = bool(
                adapter is not None and adapter._graphs
            )
            raise RuntimeError(f"injected construction seam {case_id}")

        def block_cleanup(self, *args, **kwargs):
            if cleanup_blocked[0]:
                raise RuntimeError("injected construction cleanup")
            return real_release(self, *args, **kwargs)

        seam.setattr(dynamic_output, "acquire_light_1d_retention", capture_acquire)
        seam.setattr(dynamic_output, "open_headless_scan_session", capture_open)
        seam.setattr(PublicationStore, "light_1d_cleanup_hooks", capture_hooks)
        seam.setattr(
            DynamicOutputAdapter, "_release_construction_custody",
            capture_transfer,
        )
        if method != "Light1DCustodySlot":
            seam.setattr(dynamic_output, "Light1DCustodySlot", capture_slot)
        seam.setattr(target_owner, method, fail_boundary)
        seam.setattr(SessionResourceAuthority, "_release", block_cleanup)
        executor, identity, run, artifact, _target, terminal = terminal_run(
            root / case_id
        )
        session, lease = seen["session"], seen["lease"]
        terminal_truth = terminal.cleanup_status.value, run.closed
        try:
            first_value = executor.close(identity).cleanup_status.value
        except BaseException as error:
            first_value = f"{type(error).__name__}: {error}"
        token = None if lease.cleanup_receipt is None \
            else lease.cleanup_receipt.retry_token
        retained = (
            artifact.light_lease is lease,
            artifact.light_hooks is seen.get("hooks"),
            seen.get("slot") is not None
            and artifact.light_slot is seen["slot"],
            artifact.light_retry_token is token and token is not None,
        )
        cleanup_blocked[0] = False
        try:
            second_value = executor.close(identity).cleanup_status.value
        except BaseException as error:
            second_value = f"{type(error).__name__}: {error}"
        result = {
            "terminal": terminal.kind.value,
            "session_terminal": not session.is_running,
            "accounting": seen["accounting"].snapshot().state.value,
            "sink_terminal": session.terminal_result is not None,
            "graph_at_failure": bool(seen.get("graph_at_failure")),
            "terminal_truth": terminal_truth,
            "first": first_value,
            "retained": retained,
            "second": second_value,
            "released": lease.state.value,
            "reservation": lease.authority.snapshot().reservation_count,
            "frame_callbacks": len(session._frame_cbs),
            "detached": artifact.publications.allocation is None
            and artifact.publications._light_1d is None,
            "cleared": artifact.light_lease is artifact.light_slot is None
            and artifact.light_hooks is artifact.light_retry_token is None
            and artifact.light_unsubscribe is None and not session._frame_cbs,
        }
        try:
            executor.close(identity)
        except BaseException:
            pass
        slot = seen.get("slot")
        assert slot is None or slot.state in {
            Light1DCustodyState.CANCELLED,
            Light1DCustodyState.RELEASED,
        }
        return result


def _p2_wait_transport_idle(transport, gate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not transport.retains_gate(gate):
            return
        time.sleep(0.01)
    raise AssertionError("hydration transport did not settle")


def _p2_demote_combined(store):
    base = store._items[0]
    if base.view.thumbnail is None:
        source = base.view.raw
        if source is None:
            source = base.view.intensity_2d
        assert source is not None
        store.upsert(replace(base, view=replace(
            base.view, thumbnail=np.asarray(source)[::8, ::8],
        )))
    assert store.evict_heavy(0)


def _heavy_case(identity: str, labels: tuple[int, ...]):
    state = RunDisplayState(RunIdentity(1, identity), max_payload_items=2)
    state.configure(partition_count=1, npt=8, frame_bytes=32)
    owner = _owner(state)
    modes = (("1d", "default"), ("2d", "default"))
    keys = tuple(
        _publish(state, owner, label, hydratable=modes, durable=())[1]
        for label in labels
    )
    return state, owner, modes, keys


def test_blocked_heavy_candidates_probe_once_until_owner_rearm(
    monkeypatch,
) -> None:
    state, owner, modes, keys = _heavy_case("p1c-candidates", (1, 2))
    state._residency.limits = DisplayResidencyLimits(1, 8, 8, 8)
    attempts = []
    real_evict = state._residency._evict_heavy

    def counted_evict(key):
        attempts.append(key)
        return real_evict(key)

    monkeypatch.setattr(state._residency, "_evict_heavy", counted_evict)
    state._residency.enforce()
    assert attempts == list(keys)
    state._residency.enforce()
    state._residency.enforce()
    assert attempts == list(keys)

    for label in (1, 2):
        owner.records.replace_projection(
            label, hydratable=modes, durable=modes
        )
    state.mark_durable(owner, (1, 2))
    assert attempts == [*keys, keys[0]]
    assert tuple(state._residency._heavy) == (keys[1],)


def test_owner_rearm_uses_global_fifo_and_includes_dropped_projection(
    monkeypatch,
) -> None:
    state = RunDisplayState(RunIdentity(1, "p1c-owner-fifo"), max_payload_items=2)
    state.configure(partition_count=2, npt=8, frame_bytes=32)
    owners = [
        state.add_artifact(
            Path(f"/run/p1c-{name}.nexus"),
            "p1c-l1",
            mask=None,
            mask_saturation=False,
            measurement_mode="Standard",
        )
        for name in ("a", "b")
    ]
    modes = (("1d", "default"), ("2d", "default"))
    keys = [
        _publish(state, owner, label, hydratable=modes, durable=())[1]
        for owner in owners
        for label in (1, 2)
    ]
    residency = state._residency
    residency.limits = DisplayResidencyLimits(2, 8, 8, 8)
    residency.enforce()
    residency.observe(
        keys[3],
        records=owners[1].records,
        publications=owners[1].publications,
    )
    assert tuple(residency._heavy_candidates) == (keys[3],)

    owners[0].records.replace_projection(
        1, hydratable=modes, durable=modes
    )
    owners[0].records.replace_projection(
        2,
        hydratable=(("1d", "default"),),
        durable=(("1d", "default"),),
        dropped=(("2d", "default"),),
    )
    attempts = []
    real_evict = residency._evict_heavy

    def counted_evict(key):
        attempts.append(key)
        return real_evict(key)

    monkeypatch.setattr(residency, "_evict_heavy", counted_evict)
    state.mark_durable(owners[0], (1,))

    assert attempts == keys[:2]
    assert tuple(residency._heavy) == tuple(keys[2:])
    assert tuple(residency._heavy_candidates) == (keys[3],)
    assert owners[0].records.dropped_modes(2) == {("2d", "default")}
    assert not owners[0].records.has_heavy_payload(2)
    assert owners[1].records.has_heavy_payload(1)


def test_heavy_eviction_failure_is_retryable(
    monkeypatch,
) -> None:
    """A failed eviction remains retryable without residency snapshot undo."""
    state, owner, modes, keys = _heavy_case("p1c-retry", (1, 2, 3))
    residency = state._residency

    owner.records.replace_projection(1, hydratable=modes, durable=modes)
    residency.limits = DisplayResidencyLimits(2, 8, 8, 8)
    attempts = []
    real_evict = residency._evict_heavy

    def fail_once(key):
        attempts.append(key)
        if len(attempts) == 1:
            raise RuntimeError("injected heavy eviction failure")
        return real_evict(key)

    monkeypatch.setattr(residency, "_evict_heavy", fail_once)
    with pytest.raises(RuntimeError, match="injected heavy eviction failure"):
        residency.enforce()
    assert attempts == [keys[0]]
    assert tuple(residency._heavy_candidates) == keys
    assert tuple(residency._heavy) == keys

    residency.enforce()
    assert attempts == [keys[0], keys[0]]
    assert tuple(residency._heavy_candidates) == keys[1:]
    assert tuple(residency._heavy) == keys[1:]


def test_display_retain_does_not_rewrite_session_projection() -> None:
    state = RunDisplayState(RunIdentity(1, "p1c-projection"), max_payload_items=2)
    state.configure(partition_count=1, npt=8, frame_bytes=32)
    owner = _owner(state)
    one_d = (("1d", "default"),)
    two_d = (("2d", "default"),)

    record = _record(1)[0]
    owner.records.upsert(record, source_identity="/source/p1c_0001.tif#1")
    owner.records.replace_projection(
        1,
        hydratable=one_d,
        durable=one_d,
        dropped=two_d,
    )
    before = (
        owner.records._revisions[1],
        owner.records.hydratable_modes(1),
        owner.records.durable_modes(1),
        owner.records.dropped_modes(1),
    )
    publication = _record(1)[1]
    delta = state.append_navigation(owner.source_scan, str(owner.artifact), 1)
    state.retain_frame(
        owner,
        delta.appended,
        record,
        publication,
        source_identity="/source/p1c_0001.tif#1",
        frame_mask_qualified=False,
    )
    state.mark_durable(owner, (1,))

    assert (
        owner.records._revisions[1],
        owner.records.hydratable_modes(1),
        owner.records.durable_modes(1),
        owner.records.dropped_modes(1),
    ) == before
    assert state._residency._evict_live(delta.appended) is True


def test_nine_durable_frames_cross_real_heavy_cap_and_cleanly_demote(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    state = RunDisplayState(RunIdentity(1, "p1c-cap"), max_payload_items=2)
    state.configure(partition_count=1, npt=8, frame_bytes=32)
    owner = _owner(state)
    modes = (("1d", "default"), ("2d", "default"))

    for label in range(1, 10):
        _publish(
            state,
            owner,
            label,
            hydratable=modes,
            durable=modes,
        )
        state.mark_durable(owner, (label,))

    snapshot = state.residency_snapshot()
    assert snapshot.heavy == snapshot.limits.heavy == 8
    assert owner.records.has_heavy_payload(1) is False
    assert owner.publications.get(1).raw_status == "thumbnail"
    assert not hasattr(owner, "light_records")
    assert owner.publications.get(1) is not None


def test_owed_live_retirement_is_unchanged_then_durable_retry_is_total() -> None:
    state = RunDisplayState(RunIdentity(1, "p1c-retire"), max_payload_items=2)
    state.configure(partition_count=1, npt=8, frame_bytes=32)
    state._residency.limits = DisplayResidencyLimits(8, 8, 1, 1)
    owner = _owner(state)
    modes = (("1d", "default"), ("2d", "default"))
    record, key = _publish(
        state,
        owner,
        1,
        hydratable=modes,
        durable=(),
    )
    before = (
        owner.records.get(1),
        owner.publications.get(1),
        state.residency_snapshot(),
    )

    assert state._residency._evict_live(key) is False
    assert (
        owner.records.get(1),
        owner.publications.get(1),
        state.residency_snapshot(),
    ) == before

    owner.records.replace_projection(
        1, hydratable=modes, durable=modes,
    )
    state.mark_durable(owner, (1,))
    assert state._residency._evict_live(key) is True
    assert owner.records.get(1) is None
    assert owner.publications.get(1) is None
    assert key not in state._residency._stores
    assert key not in state._residency._heavy
    assert key not in state._residency._thumbnails
    assert key not in state._residency._browse
    assert key not in state._residency._live
    assert record.label == 1


def test_removed_store_calls_and_canonical_projection_writes_stay_absent() -> None:
    root = Path(__file__).resolve().parents[3]
    offenders = []
    for relative in (
        "src/xdart/gui/tabs/scattering/display_runtime.py",
        "src/xdart/gui/tabs/scattering/display_residency.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            ):
                continue
            receiver = ast.unparse(node.func.value)
            method = node.func.attr
            canonical = receiver.endswith(".records")
            if (
                canonical
                and method in {
                    "upsert",
                    "mark_persisted",
                    "replace_projection",
                    "evict_heavy",
                    "discard",
                }
            ):
                offenders.append((relative, receiver, method, node.lineno))

    assert offenders == []
    assert all(
        "light_records" not in (root / relative).read_text(encoding="utf-8")
        for relative in (
            "src/xdart/gui/tabs/scattering/display_runtime.py",
            "src/xdart/gui/tabs/scattering/display_residency.py",
        )
    )
    assert not hasattr(FrameRecordStore, "discard")
    assert not hasattr(FrameRecordStore, "evict_heavy")


def test_real_nine_frame_container_finishes_and_executor_closes(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    source = tmp_path / "nine.nxs"
    target = tmp_path / "nine-processed.nexus"
    poni = tmp_path / "cal.poni"
    _write_stack(source, 9)
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        _batch_intent(source, target, poni),
        request_value=9101,
    )
    events = _drain_until(
        executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
        timeout=90.0,
    )
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FINISHED
    assert terminal.completed == terminal.total == 9
    run = executor._exact_run(identity)
    assert run is not None
    assert run.completed == 9
    snapshot = run.display.residency_snapshot()
    fact, = (
        value for value in run.resource_facts
        if isinstance(value, dynamic_output.HeavyResidencyFact)
    )
    owner, = run.display.artifacts.values()
    publications = owner.publications
    labels = publications.labels()
    assert labels == tuple(range(9))
    assert publications.allocation is not None
    assert (fact.choice, fact.resolution_source,
            fact.requested_heavy_bound) == ("auto", "environment", 8)
    assert fact.granted_publication_heavy_count == publications._max_heavy_items
    assert len(publications._heavy_labels) == min(
        len(labels), fact.granted_publication_heavy_count)
    thumbnail_count = sum(map(publications.has_thumbnail, labels))
    assert thumbnail_count == len(publications._thumb_labels) == min(
        len(labels), publications._max_thumbnail_items)
    assert snapshot.limits.heavy == fact.effective_display_count == min(
        fact.granted_record_heavy_count, fact.granted_publication_heavy_count)
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED


def test_real_host_exit_action_closes_after_nine_frame_run(
    monkeypatch,
    tmp_path,
) -> None:
    from xdart import _gui_main
    from xdart.gui.pages.catalog import SCATTERING_WORKSPACE_PAGE
    from xdart.gui.pages.services import (
        DiagnosticIdentity,
        ExecutionProfile,
        HostServices,
    )
    from xdart.gui.pages.values import PageCleanup
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter

    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    source = tmp_path / "host-nine.nxs"
    target = tmp_path / "host-nine-processed.nexus"
    poni = tmp_path / "host-cal.poni"
    _write_stack(source, 9)
    write_poni(poni)
    intent_store = RunIntentStore(_batch_intent(source, target, poni))
    executor = StandardRunExecutor(join_timeout=2.0)
    source_port = FilesystemSourceAdapter()

    class _Status:
        def show(self, _text, timeout_ms=0):
            del timeout_ms

    class _Intents:
        def store_for(self, _key):
            return intent_store

    class _Execution:
        def executor_for(self, _key):
            return executor

    class _Sources:
        def source_port_for(self, _key):
            return source_port

    services = HostServices(
        status=_Status(),
        run_intents=_Intents(),
        execution=_Execution(),
        sources=_Sources(),
        execution_profile=ExecutionProfile.TEST,
        diagnostics=DiagnosticIdentity("tests.p1c-l1-host-close"),
    )
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _gui_main.Main(
        page_descriptors=(SCATTERING_WORKSPACE_PAGE,),
        host_services=services,
        selected_page_key=SCATTERING_WORKSPACE_PAGE.key,
    )
    page = window.main_widget
    terminated = []
    monkeypatch.setattr(
        window,
        "_terminate_process",
        lambda: terminated.append(True),
    )
    window.show()

    def wait_for(predicate, timeout=90.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            qapp.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return False

    try:
        window.page_handle.run_control.run_pause()
        assert wait_for(lambda: page._lifecycle.phase is RunPhase.RUNNING)
        assert wait_for(
            lambda: page._lifecycle.phase is RunPhase.IDLE
            and bool(page._shell.scientific._trace_history_by_identity)
            and bool(page._shell.scientific.curve.listDataItems())
        )
        run = executor._active
        assert run is not None and run.closed
        owner = next(iter(run.display.artifacts.values()))
        lease, slot = owner.light_lease, owner.light_slot
        assert lease is not None and lease.state is Light1DLeaseState.ACTIVE
        assert slot is not None and slot.state is Light1DCustodyState.RETAINED
        window.ui.actionExit.trigger()
        assert wait_for(
            lambda: bool(terminated) and not window.isVisible(), timeout=5.0
        )
        assert window.page_handle.close().status is PageCleanup.CLEAN
        assert page._context_controller.acquisition_context is None
        assert lease.state is Light1DLeaseState.RELEASED
        assert slot.state is Light1DCustodyState.RELEASED
        assert lease.authority.snapshot().reservation_count == 0
        assert wait_for(
            lambda: not any(
                thread.is_alive() and thread.name.startswith("scattering-")
                for thread in threading.enumerate()
            ),
            timeout=5.0,
        )
    finally:
        window._process_exit_requested = False
        window.close()
        window.deleteLater()
        qapp.processEvents()
