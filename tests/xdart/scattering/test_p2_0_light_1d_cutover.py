"""Finite P2-0 oracle for the light-1D display ownership cutover."""

from __future__ import annotations

import ast
from pathlib import Path
from threading import Barrier, Event, Lock, Thread
from time import monotonic, sleep

import h5py
import numpy as np

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_comprehensive_live import (
    _batch_intent, _grow_stack, _write_stack,
)
from tests.xdart.scattering.test_p1b_output_graph import _TERMINAL, _drain_until, _start
from tests.xdart.scattering.test_p1c_l1_display_residency import (
    _p2_demote_combined as _demote_combined,
    _p2_construction_case,
    _p2_wait_transport_idle as _wait_transport_idle,
)
from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor
from xdart.gui.tabs.scattering.adapters.dynamic_output import DynamicOutputAdapter
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_runtime import (
    DetectorHydrationOutcome, DisplayArtifact, RunDisplayState,
)
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.modules.display_context import HydrationRequest
from xdart.modules.frame_publication import PublicationStore
from xrd_tools.session import (
    DynamicRunState, HydrationPurpose, HydrationReadKey, HydrationScope,
    HydrationToken, Light1DCleanupPending, Light1DCustodyState, Light1DLeaseState,
    Light1DRetentionLease, SessionResourceAuthority,
)
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.session.intent_store import RunIntent, RunIntentStore
from xrd_tools.sources.selection import DirectorySourceSpec
ROOT = Path(__file__).resolve().parents[3]
DISPLAY_RUNTIME = ROOT / "src/xdart/gui/tabs/scattering/display_runtime.py"
DISPLAY_RESIDENCY = ROOT / "src/xdart/gui/tabs/scattering/display_residency.py"
RUN_EXECUTOR = ROOT / "src/xdart/gui/tabs/scattering/adapters/run_executor.py"
DYNAMIC_OUTPUT = ROOT / "src/xdart/gui/tabs/scattering/adapters/dynamic_output.py"
def _terminal_events(executor):
    return _drain_until(executor, lambda values: any(event.kind in _TERMINAL for event in values), timeout=90.0)
def _terminal_event(executor):
    return next(event for event in _terminal_events(executor) if event.kind in _TERMINAL)
def _terminal_run(tmp_path: Path, *, frames=1, processing_mode="Int 1D"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.nxs"
    target = tmp_path / "processed.nexus"
    poni = tmp_path / "cal.poni"
    _write_stack(source, frames)
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=3.0)
    intent = _batch_intent(source, target, poni)
    intent.processing_mode = processing_mode
    identity = _start(executor, intent, request_value=20_000 + frames)
    terminal = _terminal_event(executor)
    run = executor._exact_run(identity)
    assert run is not None
    assert len(run.display.artifacts) == 1
    artifact = next(iter(run.display.artifacts.values()))
    return executor, identity, run, artifact, target, terminal
def _directory_intent(
    raw: Path, target: Path, poni: Path, *, live: bool = False,
) -> RunIntent:
    return RunIntent(
        source_spec=DirectorySourceSpec(raw, suffixes=(".nxs",), metadata_format=None),
        poni_file=str(poni),
        project_root=str(raw),
        save_path=str(target),
        output_mode="Overwrite",
        processing_mode="Int 1D",
        live_mode=live,
        max_cores=1,
        bai_1d_args={"npt": 8, "method": "numpy"},
        bai_2d_args={"npt_rad": 8, "npt_azim": 4, "method": "numpy"},
    )
def _function_tree(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _attribute_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        child for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == name
    ]


def _light_lock_descendants(function: ast.FunctionDef) -> set[int]:
    locks = [
        node for node in ast.walk(function)
        if isinstance(node, ast.With)
        and any(
            ast.unparse(item.context_expr) == "self._light_admission_lock"
            for item in node.items
        )
    ]
    assert locks
    return {id(child) for lock in locks for child in ast.walk(lock)}


def test_p2_0_constructs_exact_light_graph_and_unwinds_every_boundary(monkeypatch, tmp_path) -> None:
    executor, identity, _run, artifact, _target, terminal = _terminal_run(tmp_path)
    lease = artifact.light_lease
    slot = artifact.light_slot
    try:
        assert terminal.kind is StandardEventKind.FINISHED
        assert type(artifact) is DisplayArtifact
        assert not hasattr(artifact, "light_records")
        assert type(lease) is Light1DRetentionLease
        assert lease.state is Light1DLeaseState.ACTIVE
        assert slot.state is Light1DCustodyState.RETAINED
        assert artifact.publications._light_1d is lease
        assert lease.authority.parent_allocation is artifact.publications.allocation
        assert lease.authority.snapshot().reservation_count == 1
    finally:
        receipt = executor.close(identity)
    assert receipt.cleanup_status is CleanupStatus.CLEANED
    assert lease.state is Light1DLeaseState.RELEASED
    assert lease.authority.snapshot().reservation_count == 0

    case_ids = (
        "store-hooks", "display-lease", "slot-pending", "custody",
        "accounting", "subscription-before", "subscription-after",
    )
    ledger = {
        case_id: _p2_construction_case(
            monkeypatch, tmp_path, case_id, _terminal_run,
        )
        for case_id in case_ids
    }
    assert tuple(ledger) == case_ids
    failures = []
    terminal_states = {state.value for state in (
        DynamicRunState.FINISHED, DynamicRunState.STOPPED,
        DynamicRunState.ABORTED,
    )}
    for case_id in case_ids:
        case = ledger[case_id]
        pre_slot = case_id in case_ids[:3]
        expected_first = "cleanup_pending" if pre_slot else "cleaned"
        expected_graph = case_id.startswith("subscription-")
        expected_terminal = ("cleanup_pending", False) if pre_slot \
            else ("cleaned", True)
        if not (
            case["terminal"] == "failed"
            and case["session_terminal"]
            and case["accounting"] in terminal_states
            and case["sink_terminal"]
            and case["graph_at_failure"] is expected_graph
            and case["terminal_truth"] == expected_terminal
            and case["first"] == expected_first
            and (not pre_slot or case["retained"] == (
                True, True, case_id == "slot-pending", True,
            ))
            and case["second"] == "cleaned"
            and case["released"] == "released"
            and case["reservation"] == 0
            and case["frame_callbacks"] == 0
            and case["detached"] and case["cleared"]
        ):
            failures.append((case_id, case))
    assert failures == []
def test_p2_0_synchronous_pair_precedes_finish_and_prefix_stop(monkeypatch, tmp_path) -> None:
    order: list[str] = []
    guard = Lock()
    original_publish = PublicationStore.publish_gui_light_1d
    original_finish = DynamicOutputAdapter.finish_current

    def publish(self, publication, light_record):
        with guard:
            order.append("light")
        return original_publish(self, publication, light_record)

    def finish(self, *args, **kwargs):
        result = original_finish(self, *args, **kwargs)
        with guard:
            order.append("finish")
        return result

    monkeypatch.setattr(PublicationStore, "publish_gui_light_1d", publish)
    monkeypatch.setattr(DynamicOutputAdapter, "finish_current", finish)
    executor, identity, _run, artifact, _target, terminal = _terminal_run(
        tmp_path, frames=3
    )
    try:
        assert terminal.kind is StandardEventKind.FINISHED
        assert order.count("light") == 3
        assert "finish" in order
        assert max(index for index, value in enumerate(order) if value == "light") < order.index("finish")
        assert tuple(artifact.publications._light_1d_items) == (0, 1, 2)
    finally:
        assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED
def test_p2_0_terminal_matrix_adopts_only_canonical_prefix(monkeypatch, tmp_path) -> None:
    original = PublicationStore.publish_gui_light_1d
    calls = 0

    def fail_first(self, publication, light_record):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected synchronous light failure")
        return original(self, publication, light_record)

    with monkeypatch.context() as injected:
        injected.setattr(
            PublicationStore, "publish_gui_light_1d", fail_first
        )
        executor, identity, _run, artifact, target, terminal = _terminal_run(
            tmp_path / "callback-failure", frames=3
        )
        lease = artifact.light_lease
        slot = artifact.light_slot
        try:
            assert terminal.kind is StandardEventKind.FAILED
            assert calls == 1
            with h5py.File(target, "r") as handle:
                assert tuple(
                    handle["entry/integrated_1d/frame_index"][()]
                ) == (0, 1, 2)
            assert slot.state is Light1DCustodyState.RETAINED
            assert slot.custody_receipt.retained_rows == 0
            assert lease.keys() == ()
        finally:
            assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED

    missing_calls = 0
    semantic_failures: list[object] = []
    original_get = FrameRecordStore.get

    def miss_first_record(self, label):
        nonlocal missing_calls
        missing_calls += 1
        if missing_calls == 1:
            return None
        return original_get(self, label)

    with monkeypatch.context() as missing:
        missing.setattr(FrameRecordStore, "get", miss_first_record)
        executor, identity, _run, _artifact, target, terminal = _terminal_run(
            tmp_path / "missing-record", frames=3
        )
        try:
            if (
                terminal.kind is not StandardEventKind.FAILED
                or missing_calls != 1
            ):
                semantic_failures.append((
                    "missing-record", terminal.kind, missing_calls,
                ))
            with h5py.File(target, "r") as handle:
                assert tuple(
                    handle["entry/integrated_1d/frame_index"][()]
                ) == (0, 1, 2)
        finally:
            assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED

    source = tmp_path / "append" / "source.nxs"
    target = tmp_path / "append" / "processed.nexus"
    poni = tmp_path / "append" / "cal.poni"
    source.parent.mkdir()
    _write_stack(source, 2)
    write_poni(poni)
    seed = StandardRunExecutor(join_timeout=3.0)
    seed_identity = _start(
        seed, _batch_intent(source, target, poni), request_value=20_031,
    )
    assert _terminal_event(seed).kind is StandardEventKind.FINISHED
    assert seed.close(seed_identity).cleanup_status is CleanupStatus.CLEANED
    _grow_stack(source, 3, value=3)
    append_intent = _batch_intent(source, target, poni)
    append_intent.output_mode = "Append"
    observed: list[tuple[tuple[int, bool], ...]] = []
    real_open = dynamic_output.open_headless_scan_session

    def observed_open(scan, plan, **kwargs):
        observed.append(tuple(
            (int(frame.index), frame.image is None) for frame in scan.frames
        ))
        return real_open(scan, plan, **kwargs)

    monkeypatch.setattr(
        dynamic_output, "open_headless_scan_session", observed_open,
    )
    append = StandardRunExecutor(join_timeout=3.0)
    append_identity = _start(append, append_intent, request_value=20_032)
    append_terminal = _terminal_event(append)
    try:
        if (
            append_terminal.kind is not StandardEventKind.FINISHED
            or observed != [((2, False),)]
        ):
            semantic_failures.append((
                "descriptor-append", append_terminal.kind, observed,
            ))
    finally:
        assert append.close(append_identity).cleanup_status is CleanupStatus.CLEANED
    assert semantic_failures == []
def test_p2_0_releases_a_before_b_open_reuses_local_zero_and_shared_hydration_lane(monkeypatch, tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_stack(raw / "a.nxs", 1)
    _write_stack(raw / "b.nxs", 1)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    acquired: list[Light1DRetentionLease] = []
    sessions = []
    slots = []
    drop_states = []
    predecessor_states: list[tuple[Light1DLeaseState, ...]] = []
    source_open_states: list[tuple[Light1DLeaseState, ...]] = []
    original_acquire = dynamic_output.acquire_light_1d_retention
    original_session = dynamic_output.open_headless_scan_session
    original_bind = RunDisplayState.bind_light_custody
    original_drop = DynamicOutputAdapter.drop_released_predecessor
    original_open = run_executor.open_container_cursor
    original_materialize = run_executor.materialize_live_directory_group
    original_drain = StandardRunExecutor._drain_light_lineage
    cleanup_browse = Barrier(2)
    cleanup_entered = Event()
    browse_settled = Event()
    first_drain = True
    materialization_states: list[tuple[Light1DLeaseState, ...]] = []
    a_graph = []
    output_ref = []
    run_ref = []
    first_effect = []

    def acquire(*args, **kwargs):
        predecessor_states.append(tuple(lease.state for lease in acquired))
        lease = original_acquire(*args, **kwargs)
        acquired.append(lease)
        return lease

    def open_session(*args, **kwargs):
        session = original_session(*args, **kwargs)
        sessions.append(session)
        return session

    def bind_custody(self, owner, hooks, slot):
        slots.append(slot)
        return original_bind(self, owner, hooks, slot)

    def drop(self, owner):
        graph = self._current
        a_graph.append(graph)
        output_ref.append(self)
        state = (
            graph["session"].is_running,
            graph["accounting"].snapshot().state.value,
            slots[0].state,
            graph["projection_pending"],
            self._durable_labels_for(graph),
            tuple(sorted(graph["settled_labels"])),
        )
        value = original_drop(self, owner)
        drop_states.append(state + (
            graph not in self._graphs.values(), self._current is None,
        ))
        return value

    def open_cursor(*args, **kwargs):
        source_open_states.append(tuple(lease.state for lease in acquired))
        return original_open(*args, **kwargs)

    def materialize(*args, **kwargs):
        materialization_states.append(tuple(
            lease.state for lease in acquired
        ))
        if a_graph:
            graph = a_graph[0]
            first_effect.append((
                graph["session"] is sessions[0], graph["sink"] is not None,
                graph["accounting"] is not None,
                graph["display_owner"].light_slot is None,
                graph["display_owner"].light_lease is None,
                graph not in output_ref[0]._graphs.values(),
                output_ref[0]._current is None,
                run_ref[0].current_published,
            ))
        return original_materialize(*args, **kwargs)

    def drain(self, run, owner):
        nonlocal first_drain
        if first_drain:
            first_drain = False
            cleanup_entered.set()
            cleanup_browse.wait(timeout=10.0)
            assert browse_settled.wait(timeout=10.0)
        return original_drain(self, run, owner)

    monkeypatch.setattr(dynamic_output, "acquire_light_1d_retention", acquire)
    monkeypatch.setattr(dynamic_output, "open_headless_scan_session", open_session)
    monkeypatch.setattr(RunDisplayState, "bind_light_custody", bind_custody)
    monkeypatch.setattr(DynamicOutputAdapter, "drop_released_predecessor", drop)
    monkeypatch.setattr(run_executor, "open_container_cursor", open_cursor)
    monkeypatch.setattr(
        run_executor, "materialize_live_directory_group", materialize,
    )
    monkeypatch.setattr(StandardRunExecutor, "_drain_light_lineage", drain)
    executor = StandardRunExecutor(join_timeout=3.0)
    identity = _start(
        executor,
        _directory_intent(raw, tmp_path / "processed", poni, live=True),
        request_value=20_010,
    )
    if not cleanup_entered.wait(timeout=20.0):
        executor.stop(identity)
        executor.close(identity)
        raise AssertionError("Live predecessor never reached terminal drain")
    run = executor._exact_run(identity)
    assert run is not None and run.context_runtime is not None
    run_ref.append(run)
    assert len(acquired) == 1
    a_owner = next(iter(run.display.artifacts.values()))
    a_lease = acquired[0]
    gate = run.context_runtime.context.commit_gate
    hydration_owner = run.context_runtime.context.hydration_owner
    transport = run.display.transport
    assert a_owner.light_lease is a_lease
    assert a_lease.state is Light1DLeaseState.FENCED
    assert source_open_states == [()]

    browse_store = PublicationStore(max_items=2, max_heavy_items=1)
    read_key = HydrationReadKey(
        HydrationScope(*hydration_owner.as_tuple()),
        str(a_owner.artifact),
        0,
        HydrationPurpose.PREVIEW,
    )
    browse_request = HydrationRequest(
        0,
        HydrationPurpose.PREVIEW,
        51,
        hydration_owner,
        (browse_store,),
        gate,
        read_key=read_key,
        token=HydrationToken(read_key, 51),
    )
    detached_calls: list[HydrationRequest] = []
    original_submit_detached = transport.submit_detached
    original_dispatch_detached = transport.dispatch_detached
    original_derive = transport._derive
    derive_entered = Event()
    derive_release = Event()
    browse_errors: list[BaseException] = []

    def counted_capture(request, *, closed=False):
        if request is browse_request:
            assert run.display._light_admission_lock.acquire(blocking=False)
            run.display._light_admission_lock.release()
            return original_submit_detached(request, closed=closed)
        assert not run.display._light_admission_lock.acquire(blocking=False)
        detached_calls.append(request)
        return original_submit_detached(request, closed=closed)

    def checked_dispatch(mutation):
        assert run.display._light_admission_lock.acquire(blocking=False)
        run.display._light_admission_lock.release()
        return original_dispatch_detached(mutation)

    def held_derive(request):
        if request is browse_request:
            cleanup_browse.wait(timeout=10.0)
            derive_entered.set()
            assert derive_release.wait(timeout=10.0)
        return original_derive(request)

    monkeypatch.setattr(transport, "submit_detached", counted_capture)
    monkeypatch.setattr(transport, "dispatch_detached", checked_dispatch)
    monkeypatch.setattr(transport, "_derive", held_derive)

    def submit_browse() -> None:
        try:
            assert transport.submit(browse_request) is not None
            _wait_transport_idle(transport, gate)
        except BaseException as error:
            browse_errors.append(error)
        finally:
            browse_settled.set()

    browse_thread = Thread(target=submit_browse)
    browse_thread.start()
    assert derive_entered.wait(timeout=10.0)
    a_key = run.display.catalog_snapshot().entries[0]
    preview_done = Event()

    def attempt_fenced_preview() -> None:
        run.display._request_preview(
            a_owner, a_key, 52, True, hydration_owner, gate,
        )
        preview_done.set()

    preview_thread = Thread(target=attempt_fenced_preview)
    preview_thread.start()
    assert preview_done.wait(timeout=2.0)
    assert detached_calls == []
    derive_release.set()
    preview_thread.join(timeout=10.0)
    browse_thread.join(timeout=10.0)
    assert not preview_thread.is_alive() and not browse_thread.is_alive()
    assert browse_errors == []

    deadline = monotonic() + 90.0
    while monotonic() < deadline \
            and len(executor.processed_live_revisions(identity)) < 2:
        sleep(0.01)
    assert len(executor.processed_live_revisions(identity)) == 2
    executor.stop(identity)
    terminal = _terminal_event(executor)
    semantic_failures: list[object] = []
    try:
        assert terminal.kind is StandardEventKind.STOPPED
        assert drop_states == [(
            False, "finished", Light1DCustodyState.RELEASED, False,
            (0,), (0,), True, True,
        )]
        assert first_effect == [(
            True, True, True, True, True, True, True, 1,
        )]
        assert len(acquired) == 2
        assert predecessor_states[1] == (Light1DLeaseState.RELEASED,)
        if materialization_states != [
            (), (Light1DLeaseState.RELEASED,),
        ]:
            semantic_failures.append((
                "deferred-pre-probe", materialization_states,
            ))
        assert source_open_states[1] == (Light1DLeaseState.RELEASED,)
        assert acquired[0].state is Light1DLeaseState.RELEASED
        assert acquired[1].state is Light1DLeaseState.ACTIVE
        keys = run.display.catalog_snapshot().entries
        assert tuple(key.local_frame_label for key in keys) == (0, 0)
        assert run.display.transport is transport
        assert run.context_runtime.context.commit_gate is gate
        assert gate.cancelled is False
        b_owner = run.display.artifacts[str(run.artifact)]
        b_key = run.display.catalog.resolve_exact(str(run.artifact), 0)
        assert b_key is not None and b_owner.publications.discard(0)
        detached_calls.clear()
        assert run.display.project(
            b_key,
            53,
            closed=True,
            owner=run.context_runtime.context.hydration_owner,
            commit_gate=gate,
        ) is None
        assert detached_calls and detached_calls[-1].stores == (
            b_owner.records, b_owner.publications,
        )
        _wait_transport_idle(transport, gate)
        hydrated = b_owner.publications.get(0)
        assert hydrated is not None and hydrated.view.intensity_1d is not None
        del hydrated

        b_lease = acquired[1]
        race_states: list[Light1DLeaseState] = []
        submit_entered = Event()
        release_started = Event()

        def barrier_capture(_request, *, closed=False):
            assert not run.display._light_admission_lock.acquire(blocking=False)
            submit_entered.set()
            assert release_started.wait(timeout=2.0)
            deadline = monotonic() + 0.25
            while b_lease.state is Light1DLeaseState.ACTIVE \
                    and monotonic() < deadline:
                sleep(0.005)
            race_states.append(b_lease.state)

        monkeypatch.setattr(transport, "submit_detached", barrier_capture)
        race_preview = Thread(target=lambda: run.display._request_preview(
            b_owner, b_key, 54, True, hydration_owner, gate,
        ))
        release_results: list[bool] = []

        def release_b() -> None:
            release_started.set()
            release_results.append(run.display.release_light_1d(
                b_owner, reason="atomic-race",
            ))

        race_release = Thread(target=release_b)
        race_preview.start()
        assert submit_entered.wait(timeout=2.0)
        race_release.start()
        race_preview.join(timeout=5.0)
        race_release.join(timeout=5.0)
        assert not race_preview.is_alive() and not race_release.is_alive()
        if race_states != [Light1DLeaseState.ACTIVE]:
            semantic_failures.append(("atomic-admission", race_states))
        assert release_results == [True]
    finally:
        cleanup_status = executor.close(identity).cleanup_status
    assert cleanup_status is CleanupStatus.CLEANED
    assert semantic_failures == []
def test_p2_0_removes_light_records_and_second_ndarray_owner() -> None:
    for path in (DISPLAY_RUNTIME, DISPLAY_RESIDENCY):
        source = path.read_text(encoding="utf-8")
        assert "light_records" not in source
    tree = ast.parse(DISPLAY_RUNTIME.read_text(encoding="utf-8"))
    constructions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_record_store_factory"
    ]
    assert len(constructions) == 1
    assert "Light1DCustodySlot" in DYNAMIC_OUTPUT.read_text(encoding="utf-8")
    artifact = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DisplayArtifact"
    )
    assert "light_admission_lock" not in {
        node.target.id
        for node in artifact.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
    }
    init = _function_tree(DISPLAY_RUNTIME, "__init__")
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Lock"
        for node in ast.walk(init)
    ) == 1
    for name in ("cancel_light_1d", "_request_preview"):
        function = _function_tree(DISPLAY_RUNTIME, name)
        source = ast.unparse(function)
        assert "self._light_admission_lock" in source
        assert "with self._lock" not in source
        calls = {
            node.func.attr
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not {"submit", "cancel_gate"} & calls
        capture_name = ("submit_detached" if name == "_request_preview"
                        else "cancel_gate_detached")
        capture = _attribute_calls(function, capture_name)
        dispatch = _attribute_calls(function, "dispatch_detached")
        locked = _light_lock_descendants(function)
        assert len(capture) == 1 and id(capture[0]) in locked
        assert dispatch and all(id(call) not in locked for call in dispatch)
    assert "_light_admission_lock" not in ast.unparse(
        _function_tree(DISPLAY_RUNTIME, "commit_preview")
    )
    construct = ast.unparse(_function_tree(RUN_EXECUTOR, "_construct"))
    assert construct.index("_settle_predecessor") < construct.index(
        "validate_planned_source"
    ) < construct.index("open_source")
    assert "commit_gate.cancel" not in construct
def test_p2_0_projection_queue_is_array_free_and_preserves_pair(monkeypatch, tmp_path) -> None:
    function = _function_tree(RUN_EXECUTOR, "_frame_ready")
    queued = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "put"
    ]
    assert queued
    assert [ast.unparse(call.args[0]) for call in queued] == [
        "(label, image, session)"
    ]
    executor, identity, _run, artifact, _target, terminal = _terminal_run(
        tmp_path
    )
    try:
        assert terminal.kind is StandardEventKind.FINISHED
        base = artifact.publications._items[0]
        pair = artifact.publications.get(0)
        cached = next(iter(_run.display.payloads.values()))
        assert base.view.intensity_1d is None
        assert cached.view.intensity_1d is None
        assert all(
            view.axis_1d is None
            and view.intensity_1d is None
            and view.sigma_1d is None
            for view in base.record.results_1d.values()
        )
        assert pair.view.intensity_1d is not None
        assert pair.record.results_1d
        del pair, base, cached
    finally:
        assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED

    raw = tmp_path / "two-d" / "raw"
    target = tmp_path / "two-d" / "processed"
    poni = tmp_path / "two-d" / "cal.poni"
    raw.mkdir(parents=True)
    _write_stack(raw / "a.nxs", 1)
    _write_stack(raw / "b.nxs", 1)
    write_poni(poni)
    real_plan = run_executor.execution_plan_values
    real_open = dynamic_output.open_headless_scan_session
    sessions = []

    def two_d_only(configuration, detector_mask=None):
        one, two, values = real_plan(configuration, detector_mask)
        return one, two, {
            **values, "integrate_1d": False, "integrate_2d": True,
        }

    def capture_session(*args, **kwargs):
        session = real_open(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(run_executor, "execution_plan_values", two_d_only)
    monkeypatch.setattr(
        dynamic_output, "_mode_tokens", lambda _configuration: ("2d:default",),
    )
    monkeypatch.setattr(dynamic_output, "open_headless_scan_session", capture_session)
    intent = _directory_intent(raw, target, poni, live=True)
    intent.processing_mode = "Int 2D"
    executor = StandardRunExecutor(join_timeout=3.0)
    identity = _start(executor, intent, request_value=20_041)
    deadline = monotonic() + 90.0
    while monotonic() < deadline \
            and len(executor.processed_live_revisions(identity)) < 2:
        sleep(0.01)
    executor.stop(identity)
    events = _terminal_events(executor)
    run = executor._exact_run(identity)
    assert run is not None
    owners = tuple(run.display.artifacts.values())
    try:
        assert next(
            event for event in events if event.kind in _TERMINAL
        ).kind is StandardEventKind.STOPPED
        assert len(executor.processed_live_revisions(identity)) == 2
        frames = tuple(
            event for event in events
            if event.kind is StandardEventKind.FRAME_READY
        )
        assert tuple(event.frame_key.local_frame_label for event in frames) \
            == (0, 0)
        assert all(owner.light_lease is owner.light_slot is None for owner in owners)
        assert tuple(
            key.local_frame_label
            for key in run.display.catalog_snapshot().entries
        ) == (0, 0)
        assert all(owner.publications.get(0).view.intensity_2d is not None for owner in owners)
    finally:
        receipt = executor.close(identity)
        leaked = tuple(session for session in sessions if session.is_running)
        for session in leaked:
            session.stop()
            session.finish(raise_on_failure=False)
    assert receipt.cleanup_status is CleanupStatus.CLEANED
    assert leaked == ()


def test_p2_0_acquisition_payload_detaches_light_1d_before_historical_retirement(
    tmp_path,
) -> None:
    executor, identity, run, artifact, _target, terminal = _terminal_run(
        tmp_path
    )
    projected = pair = None
    closed = False
    try:
        assert terminal.kind is StandardEventKind.FINISHED
        key = run.display.catalog_snapshot().entries[0]
        pair = artifact.publications.get(0)
        projected = run.display.project(
            key,
            0,
            closed=True,
            require_complete=False,
        )
        assert pair is not None
        assert projected is not None
        assert projected.view.axis_1d is not None
        assert projected.view.intensity_1d is not None
        expected_axis = np.array(pair.view.axis_1d.values, copy=True)
        expected_intensity = np.array(
            pair.view.intensity_1d,
            copy=True,
        )
        axis_detached = not np.shares_memory(
            projected.view.axis_1d.values,
            pair.view.axis_1d.values,
        )
        intensity_detached = not np.shares_memory(
            projected.view.intensity_1d,
            pair.view.intensity_1d,
        )
        lease = artifact.light_lease
        assert lease is not None
        del pair
        pair = None

        receipt = executor.close(identity)
        closed = receipt.cleanup_status is CleanupStatus.CLEANED
        assert closed
        assert axis_detached
        assert intensity_detached
        assert lease.state is Light1DLeaseState.RELEASED
        assert lease.authority.snapshot().reservation_count == 0
        assert not projected.view.axis_1d.values.flags.writeable
        assert not projected.view.intensity_1d.flags.writeable
        assert np.array_equal(
            projected.view.axis_1d.values,
            expected_axis,
        )
        assert np.array_equal(
            projected.view.intensity_1d,
            expected_intensity,
        )
    finally:
        del projected, pair
        if not closed:
            assert (
                executor.close(identity).cleanup_status
                is CleanupStatus.CLEANED
            )


def test_p2_0_preview_preserves_pair_and_light_miss_stays_array_free(monkeypatch, tmp_path) -> None:
    executor, identity, run, artifact, target, terminal = _terminal_run(
        tmp_path, processing_mode="Int 2D")
    store = artifact.publications
    gate = run.context_runtime.context.commit_gate
    transport = run.display.transport
    submitted = []
    real_submit = transport.submit_detached

    def count_submit(request, *, closed=False):
        assert not run.display._light_admission_lock.acquire(blocking=False)
        submitted.append(request)
        return real_submit(request, closed=closed)

    monkeypatch.setattr(transport, "submit_detached", count_submit)
    ledger: dict[str, object] = {}
    try:
        assert terminal.kind is StandardEventKind.FINISHED
        key = run.display.catalog_snapshot().entries[0]
        pair_identity = store._light_1d_items[0].shell.source_identity
        assert Path(pair_identity.rsplit("#", 1)[0]).is_absolute()
        assert store.get(0).view.intensity_2d is not None
        with h5py.File(target, "a") as handle:
            del handle["entry/frames/frame_0000/thumbnail"]
        _demote_combined(store)
        (tmp_path / "source.nxs").unlink()
        assert run.display.project(
            key, 41, closed=True,
            owner=run.context_runtime.context.hydration_owner,
            commit_gate=gate,
        ) is None
        _wait_transport_idle(transport, gate)
        first = store.get(0)
        ledger["detector-baseline"] = (
            first.view.intensity_2d is not None,
            run.display._detector_outcomes.get(key),
        )
        _demote_combined(store)
        before = len(submitted)
        second_miss = run.display.project(
            key, 42, closed=True,
            owner=run.context_runtime.context.hydration_owner,
            commit_gate=gate,
        ) is None
        if second_miss:
            _wait_transport_idle(transport, gate)
        recovered = store.get(0)
        after = len(submitted)
        immediate = run.display.project(key, 43, closed=True)
        ledger["demotion"] = (
            second_miss,
            after - before,
            recovered.view.intensity_2d is not None,
            immediate is not None,
            len(submitted) == after,
        )
        direct_before = len(submitted)
        run.display._request_preview(
            artifact, key, 44, True,
            run.context_runtime.context.hydration_owner, gate,
        )
        _wait_transport_idle(transport, gate)
        direct = store.get(0)
        ledger["identity"] = (
            len(submitted) == direct_before + 1,
            store._light_1d_items[0].shell.source_identity == pair_identity,
            not Path(direct.view.source_path).is_absolute(),
            artifact.light_lease is not None,
        )
        del first, recovered, immediate, direct
    finally:
        assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED
    assert tuple(ledger) == ("detector-baseline", "demotion", "identity")
    assert ledger == {
        "detector-baseline": (True, DetectorHydrationOutcome.DETECTOR_UNAVAILABLE),
        "demotion": (True, 1, True, True, True),
        "identity": (True, True, True, True),
    }
def test_p2_0_close_and_replacement_retain_exact_cleanup_custody(monkeypatch, tmp_path) -> None:
    ledger: dict[str, object] = {}
    with monkeypatch.context() as staged_case:
        root = tmp_path / "staged"
        source, target, poni = (
            root / "source.nxs", root / "processed.nexus", root / "cal.poni",
        )
        root.mkdir()
        _write_stack(source, 1)
        write_poni(poni)
        staged, resume = Event(), Event()
        seen: dict[str, object] = {}
        real_open = dynamic_output.open_headless_scan_session

        def capture_open(*args, **kwargs):
            seen["session"] = real_open(*args, **kwargs)
            return seen["session"]

        def pause_before_slot(*_args, **_kwargs):
            staged.set()
            assert resume.wait(timeout=10.0)
            seen["slot_lease_state"] = lease.state
            raise RuntimeError("injected staged construction stop")

        staged_case.setattr(dynamic_output, "open_headless_scan_session", capture_open)
        staged_case.setattr(dynamic_output, "Light1DCustodySlot", pause_before_slot)
        executor = StandardRunExecutor(join_timeout=0.25)
        identity = _start(executor, _batch_intent(source, target, poni),
                          request_value=20_051)
        assert staged.wait(timeout=20.0)
        run = executor._exact_run(identity)
        assert run is not None and len(run.display.artifacts) == 1
        artifact = next(iter(run.display.artifacts.values()))
        lease = artifact.light_lease
        first = executor.close(identity)
        staged_truth = (
            first.cleanup_status, lease.state, artifact.light_lease is lease,
            artifact.light_slot, artifact.light_release_reason,
            lease.authority.snapshot().reservation_count,
            seen["session"].is_running,
        )
        resume.set()
        terminal = _terminal_event(executor)
        second = executor.close(identity)
        ledger["staged"] = (
            staged_truth, seen["slot_lease_state"], terminal.kind,
            not seen["session"].is_running, second.cleanup_status, lease.state,
            lease.authority.snapshot().reservation_count, artifact.light_slot,
        )

    with monkeypatch.context() as release_case:
        executor, identity, run, artifact, _target, terminal = _terminal_run(
            tmp_path / "single-flight")
        lease, slot = artifact.light_lease, artifact.light_slot
        real_authority_release = SessionResourceAuthority._release
        real_slot_release = type(slot).release
        cleanup_pending, publish_token = Event(), Event()
        authority_calls: list[str] = []

        def fail_once(self, *args, **kwargs):
            authority_calls.append("release")
            if len(authority_calls) == 1:
                raise RuntimeError("injected single-flight cleanup")
            return real_authority_release(self, *args, **kwargs)

        def pause_before_publication(self, *args, **kwargs):
            try:
                return real_slot_release(self, *args, **kwargs)
            except Light1DCleanupPending:
                cleanup_pending.set()
                assert publish_token.wait(timeout=10.0)
                raise

        release_case.setattr(SessionResourceAuthority, "_release", fail_once)
        release_case.setattr(type(slot), "release", pause_before_publication)
        winner, loser = [], []

        def release_into(values):
            try:
                values.append(run.display.release_light_1d(artifact,
                    reason="single-flight"))
            except BaseException as error:
                values.append(f"{type(error).__name__}: {error}")

        winning_thread = Thread(target=release_into, args=(winner,))
        winning_thread.start()
        assert cleanup_pending.wait(timeout=10.0)
        losing_thread = Thread(target=release_into, args=(loser,))
        losing_thread.start()
        losing_thread.join(timeout=10.0)
        assert not losing_thread.is_alive()
        publish_token.set()
        winning_thread.join(timeout=10.0)
        assert not winning_thread.is_alive()
        token = lease.cleanup_receipt.retry_token
        before_retry = (tuple(authority_calls),
                        artifact.light_retry_token is token and token is not None)
        retried = run.display.release_light_1d(artifact, reason="single-flight")
        ledger["single-flight"] = (
            terminal.kind, winner, loser, before_retry,
            artifact.light_retry_token is None, retried, tuple(authority_calls),
            lease.state, slot.state, lease.authority.snapshot().reservation_count,
            executor.close(identity).cleanup_status,
        )

    assert tuple(ledger) == ("staged", "single-flight")
    assert (ledger["staged"], ledger["single-flight"]) == (
        ((CleanupStatus.CLEANUP_PENDING, Light1DLeaseState.ACTIVE, True, None,
          None, 1, True), Light1DLeaseState.ACTIVE, StandardEventKind.FAILED,
         True, CleanupStatus.CLEANED, Light1DLeaseState.RELEASED, 0, None),
        (StandardEventKind.FINISHED, [False], [False], (("release",), True),
         True, True, ("release", "release"), Light1DLeaseState.RELEASED,
         Light1DCustodyState.RELEASED, 0, CleanupStatus.CLEANED),
    )
