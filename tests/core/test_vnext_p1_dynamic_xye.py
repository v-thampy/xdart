"""P1-A transaction-backed dynamic XYE acceptance discriminators.

The tests deliberately enter through the public xdart session facade and keep
the existing H10 accounting and H23 XYE transaction as the only authorities.
The parent has no ``TransactionalXYESink`` yet; the fallback to the legacy
``XYESink`` makes that absence fail at the current truthful typed admission
door instead of failing import or collection.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading

import h5py
import numpy as np
import pytest

from tests.core._vnext_p0_c2_bridge_support import append_intent


class _CountingIntegrator:
    detector = None

    def __init__(self) -> None:
        self.calls = 0

    def integrate1d(self, image, npt, *, unit="q_A^-1", **_kwargs):
        self.calls += 1
        value = float(np.asarray(image).sum())
        return SimpleNamespace(
            radial=np.linspace(0.0, 1.0, int(npt)),
            intensity=np.full(int(npt), value),
            sigma=np.full(int(npt), value / 10.0),
            unit=unit,
        )


class _ForeignReceiptBoundary:
    @property
    def output_receipt_capabilities(self):
        from xrd_tools.io import OutputReceiptCapability

        return frozenset({OutputReceiptCapability.DURABLE_XYE})


def _plan():
    from xrd_tools.reduction import Integration1DPlan, ReductionPlan

    return ReductionPlan(
        integration_1d=Integration1DPlan(npt=8),
        integration_2d=None,
    )


def _live_frames(tmp_path: Path, count: int, integrator):
    return tuple(
        SimpleNamespace(
            idx=index,
            map_raw=np.full((2, 2), index + 1.0),
            bg_raw=None,
            scan_info={},
            source_file=str(tmp_path / f"source-{index}.tif"),
            source_frame_idx=0,
            mask=None,
            poni=None,
            integrator=integrator,
        )
        for index in range(int(count))
    )


def _targets(tmp_path: Path, name: str, *, nexus: bool):
    xye_directory = (tmp_path / f"{name}-xye").resolve()
    xye_target = f"xye:{xye_directory}"
    nexus_path = (tmp_path / f"{name}.nexus").resolve() if nexus else None
    nexus_target = None if nexus_path is None else f"nexus:{nexus_path}"
    return xye_directory, xye_target, nexus_path, nexus_target


def _accounting(*targets: str, max_outstanding: int = 8):
    from xrd_tools.session import (
        DynamicAccountingLimits,
        DynamicRunAccounting,
        ResultMode,
        StageLedger,
    )

    mode = ResultMode.one_d()
    ledger = StageLedger(
        required_modes=(mode,),
        targets_by_mode={mode: tuple(targets)},
    )
    accounting = DynamicRunAccounting(
        ledger,
        run_generation=1,
        limits=DynamicAccountingLimits(2, 3, int(max_outstanding)),
    )
    return ledger, accounting, mode


def _arm(accounting, label: int, *, revision: int = 1):
    from xrd_tools.session import DynamicFrameIdentity

    label = int(label)
    key = DynamicFrameIdentity("p1-a/source", label)
    accounting.discover(
        key, group="scan", ordinal=label, output_label=label,
    )
    token = accounting.begin_attempt(key, source_revision=int(revision))
    accounting.record_enqueued(token)
    return key, token


def _transactional_sink(directory: Path, *, stale_paths=()):
    """Use the future exact built-in, or today's truthful legacy refusal."""
    import xrd_tools.reduction as reduction

    sink_type = getattr(reduction, "TransactionalXYESink", None)
    if sink_type is None:
        return reduction.XYESink(directory)
    return sink_type(directory, stale_paths=tuple(stale_paths))


def _open_case(
    tmp_path: Path,
    name: str,
    *,
    frame_count: int,
    nexus: bool,
    integrator=None,
    stale_paths=(),
    first_intent=None,
):
    from xdart.modules.reduction import open_live_scan_session
    from xrd_tools.reduction import CompositeSink, NexusSink

    integrator = integrator or _CountingIntegrator()
    xye_directory, xye_target, nexus_path, nexus_target = _targets(
        tmp_path, name, nexus=nexus,
    )
    target_values = (xye_target,) if nexus_target is None else (
        nexus_target, xye_target,
    )
    ledger, accounting, mode = _accounting(*target_values)
    xye = _transactional_sink(xye_directory, stale_paths=stale_paths)
    if nexus:
        nexus_sink = NexusSink(
            nexus_path,
            overwrite=True,
            atomic=False,
            flush_every=None,
            same_run_intent=first_intent,
        )
        # XYE-first input order is intentional: the exact binder/session must
        # still impose NeXus-first settlement rather than trust tuple order.
        sink = CompositeSink((xye, nexus_sink))
    else:
        nexus_sink = None
        sink = xye
    session = open_live_scan_session(
        _live_frames(tmp_path, frame_count, integrator),
        _plan(),
        scan_name=name,
        sink=sink,
        executor=1,
        accounting=accounting,
        nexus_target=nexus_target,
        xye_target=xye_target,
        xye_receipt_boundary=xye,
    )
    return SimpleNamespace(
        session=session,
        ledger=ledger,
        accounting=accounting,
        mode=mode,
        xye=xye,
        xye_directory=xye_directory,
        xye_target=xye_target,
        nexus=nexus_sink,
        nexus_path=nexus_path,
        nexus_target=nexus_target,
        integrator=integrator,
    )


def _xye_files(directory: Path):
    return tuple(sorted(directory.glob("*.xye")))


def _nexus_rows(path: Path):
    with h5py.File(path, "r") as handle:
        return tuple(
            int(value)
            for value in handle["entry/integrated_1d/frame_index"][()]
        )


def _assert_durable(case, key, token, target: str):
    assert case.accounting.snapshot().durable_attempts[
        (key, case.mode, target)
    ] is token


def test_transactional_xye_perf_snapshot_cannot_observe_half_worker_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import xrd_tools.reduction.core as reduction_core
    from xrd_tools.reduction import TransactionalXYESink

    half_update = threading.Event()
    release_update = threading.Event()

    class InterleavingValues(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            if key == "sink_xye_worker_format" and value > 0.0:
                half_update.set()
                assert release_update.wait(2.0)

    sink = TransactionalXYESink(tmp_path / "race-safe-xye")
    sink._perf_enabled = True
    sink._perf_values = InterleavingValues({
        "sink_xye_write": 0.0,
        "sink_xye_worker_format": 0.0,
        "sink_xye_enqueue_wait": 0.0,
        "sink_xye_queue_high_water": 0,
        "sink_xye_drain": 0.0,
        "sink_xye_promotion": 0.0,
        "sink_xye_generated": 0,
    })
    monkeypatch.setattr(reduction_core, "write_xye", lambda *_args: None)
    failures: list[BaseException] = []
    snapshots: list[dict[str, float]] = []

    def format_stage() -> None:
        try:
            values = np.arange(4, dtype=float)
            sink._format_stage(1, values, values, values, worker=True)
        except BaseException as error:  # pragma: no cover - thread handoff
            failures.append(error)

    worker = threading.Thread(target=format_stage)
    worker.start()
    assert half_update.wait(2.0)
    snapshot_worker = threading.Thread(
        target=lambda: snapshots.append(sink.perf_snapshot())
    )
    snapshot_worker.start()
    snapshot_worker.join(0.1)
    release_update.set()
    worker.join(2.0)
    snapshot_worker.join(2.0)

    assert not failures
    assert not worker.is_alive()
    assert not snapshot_worker.is_alive()
    assert len(snapshots) == 1
    assert snapshots[0]["sink_xye_worker_format"] == pytest.approx(
        snapshots[0]["sink_xye_write"]
    )


def test_quartile_env_alone_activates_reducer_nexus_and_xye_counters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("XDART_PERF", raising=False)
    monkeypatch.setenv("XDART_PERF_QUARTILES", "1")
    case = _open_case(
        tmp_path, "quartile-env-only", frame_count=2, nexus=True,
    )
    for label, frame in enumerate(case.session.scan.frames):
        _key, token = _arm(case.accounting, label)
        assert case.session.submit(frame, attempt_token=token)
    assert case.session.finish().failed is False

    snapshot = case.session.perf_snapshot()
    assert case.xye._perf_enabled is True
    assert case.nexus._perf_nexus_enabled is True
    assert {
        "reducer_compute",
        "reducer_compute_count",
        "sink_nexus_write",
        "sink_nexus_flush",
        "sink_xye_write",
        "sink_xye_promotion",
    } <= set(snapshot)
    assert snapshot["reducer_compute_count"] == 2.0


def _observe_terminal_callbacks(monkeypatch, boundary, events=None):
    """Observe the real boundary callbacks while delegating unchanged."""
    events = [] if events is None else events
    boundary_type = type(boundary)
    originals = {
        "session_finished": boundary_type.session_finished,
        "session_stopped": boundary_type.session_stopped,
        "epoch_aborted": boundary_type.epoch_aborted,
    }

    def observe(name):
        original = originals[name]

        def delegated(owner, *args, **kwargs):
            events.append((name, owner))
            return original(owner, *args, **kwargs)

        return delegated

    for name in originals:
        monkeypatch.setattr(boundary_type, name, observe(name))
    return events


def test_transactional_xye_formats_hidden_stage_on_owned_worker_before_publication(
    tmp_path, monkeypatch,
):
    import xrd_tools.reduction.core as reduction_core

    entered, release = threading.Event(), threading.Event()
    calls = []
    original_write = reduction_core.write_xye

    def observe_write(path, radial, intensity, sigma=None):
        value = original_write(path, radial, intensity, sigma)
        calls.append((threading.current_thread().name, Path(path)))
        entered.set()
        assert release.wait(timeout=5.0)
        return value

    monkeypatch.setattr(reduction_core, "write_xye", observe_write)
    case = _open_case(
        tmp_path, "worker-stage", frame_count=1, nexus=False,
    )
    _key, token = _arm(case.accounting, 0)
    assert case.session.submit(case.session.scan.frames[0], attempt_token=token)
    entered_before_finish = entered.wait(timeout=2.0)
    observed = tuple(calls)
    hidden_before = tuple(case.xye_directory.glob(".xdart-xye-*.tmp"))
    canonical_before = _xye_files(case.xye_directory)
    release.set()
    result = case.session.finish()

    # Parent RED reaches and cleans finish before this first new expectation.
    assert entered_before_finish is True
    assert len(observed) == 1
    assert observed[0][0] == "xrd-tools-xye-writer"
    assert observed[0][1] == hidden_before[0]
    assert observed[0][1].name.startswith(".xdart-xye-")
    assert canonical_before == ()
    assert result.failed is False
    assert len(_xye_files(case.xye_directory)) == 1
    assert tuple(case.xye_directory.glob(".xdart-xye-*.tmp")) == ()
    assert case.xye._worker is None
    assert case.xye._pending is None


def test_xye_only_epoch_then_finish_uses_one_transaction_owner(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.output_transaction import (
        OutputTransactionCoordinator,
        XyeOutputTransaction,
    )
    from xrd_tools.io.export import read_xye
    from xrd_tools.session import DynamicAttemptToken, DynamicFrameIdentity

    prepared, publications, promotions = [], [], []
    original_prepare = OutputTransactionCoordinator.prepare_xye
    original_publish_epoch = XyeOutputTransaction.publish_epoch

    def observe_prepare(owner, directory, *, run_owner):
        transaction = original_prepare(owner, directory, run_owner=run_owner)
        prepared.append((run_owner, transaction))
        return transaction

    def observe_publish_epoch(owner, **kwargs):
        publications.append(owner)
        return original_publish_epoch(owner, **kwargs)

    monkeypatch.setattr(
        OutputTransactionCoordinator, "prepare_xye", observe_prepare,
    )
    monkeypatch.setattr(
        XyeOutputTransaction, "publish_epoch", observe_publish_epoch,
    )
    case = _open_case(
        tmp_path, "xye-only-epoch", frame_count=2, nexus=False,
    )
    boundary_type = type(case.accounting.writer_boundary)
    original_promote = boundary_type.epoch_committed

    def observe_promote(owner, *args, **kwargs):
        if owner is case.accounting.writer_boundary:
            promotions.append(owner)
        return original_promote(owner, *args, **kwargs)

    monkeypatch.setattr(boundary_type, "epoch_committed", observe_promote)
    key0, token0 = _arm(case.accounting, 0)
    assert case.session.submit(case.session.scan.frames[0], attempt_token=token0)
    epoch = case.session.commit_epoch()
    assert epoch is not None
    assert case.session.commit_epoch() is epoch
    assert len(prepared) == 1
    assert publications == [prepared[0][1]]
    assert promotions == [case.accounting.writer_boundary]
    _assert_durable(case, key0, token0, case.xye_target)
    assert case.accounting.snapshot().state.value == "active"
    assert len(_xye_files(case.xye_directory)) == 1

    before_accounting = case.accounting.snapshot()
    before_transaction = prepared[0][1].snapshot()
    before_files = _xye_files(case.xye_directory)
    foreign = DynamicAttemptToken(
        DynamicFrameIdentity("p1-a/foreign", 1), 1, 1, 2,
    )
    with pytest.raises(ValueError, match="foreign attempt token"):
        case.session.submit(case.session.scan.frames[1], attempt_token=foreign)
    assert case.accounting.snapshot() == before_accounting
    assert prepared[0][1].snapshot() == before_transaction
    assert case.session.commit_epoch() is epoch
    assert publications == [prepared[0][1]]
    assert promotions == [case.accounting.writer_boundary]
    assert _xye_files(case.xye_directory) == before_files

    key1, token1 = _arm(case.accounting, 1, revision=2)
    assert case.session.submit(case.session.scan.frames[1], attempt_token=token1)
    result = case.session.finish()
    assert result.failed is False
    _assert_durable(case, key1, token1, case.xye_target)
    assert case.accounting.snapshot().state.value == "finished"
    assert len(prepared) == 1
    files = _xye_files(case.xye_directory)
    assert len(files) == 2
    representative = next(path for path in files if "0001" in path.name)
    assert representative.name == "xye-only-epoch_0001.xye"
    radial, intensity, _sigma = read_xye(representative)
    assert radial.shape == intensity.shape == (8,)
    np.testing.assert_allclose(intensity, np.full(8, 8.0))
    transaction = prepared[0][1].snapshot()
    assert transaction.complete is True
    assert transaction.retryable is False
    assert transaction.staged_indices == ()
    assert case.session not in case.accounting.owner_census()


def test_nexus_xye_epoch_settles_nexus_then_xye_and_blocks_early_extend(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.output_transaction import XyeOutputTransaction
    from xrd_tools.reduction import NexusSink, TransactionalXYESink
    from xrd_tools.session import DynamicAttemptToken, DynamicFrameIdentity

    first = append_intent(
        tmp_path,
        extent=1,
        labels=(0,),
        generation=0,
        source_identity="p1-a/nexus-xye-epoch",
    )
    second = append_intent(
        tmp_path,
        extent=2,
        labels=(0, 1),
        generation=1,
        source_identity="p1-a/nexus-xye-epoch",
    )
    events, early, nexus_calls = [], [], []
    fault = OSError("forced NeXus epoch settlement fault")
    failures = [fault]
    original_nexus_epoch = NexusSink.commit_epoch
    original_xye_epoch = XyeOutputTransaction.publish_epoch
    original_publish_values = TransactionalXYESink._publish_values
    holder = {}

    def observe_nexus_epoch(owner, result):
        nexus_calls.append(owner)
        if owner is holder.get("nexus") and failures:
            raise failures.pop()
        events.append("nexus-physical")
        return original_nexus_epoch(owner, result)

    def observe_xye_epoch(owner, **kwargs):
        events.append("xye-physical")
        return original_xye_epoch(owner, **kwargs)

    def publish_and_probe(owner, values):
        value = original_publish_values(owner, values)
        try:
            holder["session"].extend_live(second)
        except BaseException as error:
            early.append(error)
        else:
            early.append(None)
        return value

    monkeypatch.setattr(NexusSink, "commit_epoch", observe_nexus_epoch)
    monkeypatch.setattr(
        XyeOutputTransaction, "publish_epoch", observe_xye_epoch,
    )
    monkeypatch.setattr(TransactionalXYESink, "_publish_values", publish_and_probe)

    case = _open_case(
        tmp_path,
        "nexus-xye-epoch",
        frame_count=2,
        nexus=True,
        first_intent=first,
    )
    holder["session"] = case.session
    holder["nexus"] = case.nexus
    boundary_type = type(case.accounting.writer_boundary)
    original_promote = boundary_type.epoch_committed

    def observe_promote(owner, session, owner_token, seal, anchor):
        if owner is case.accounting.writer_boundary:
            events.append("promote")
        return original_promote(owner, session, owner_token, seal, anchor)

    monkeypatch.setattr(boundary_type, "epoch_committed", observe_promote)
    key0, token0 = _arm(case.accounting, 0)
    assert case.session.submit(case.session.scan.frames[0], attempt_token=token0)
    with pytest.raises(OSError, match="forced NeXus epoch settlement fault") as caught:
        case.session.commit_epoch()
    assert caught.value is fault
    assert nexus_calls == [case.nexus]

    frozen_accounting = case.accounting.snapshot()
    frozen_outputs = (
        _nexus_rows(case.nexus_path),
        tuple(
            (path.name, path.read_bytes())
            for path in _xye_files(case.xye_directory)
        ),
    )
    frozen_events = (tuple(events), tuple(early), tuple(nexus_calls))
    frozen_epoch = (
        case.session._dynamic_epoch_drained,
        case.session._dynamic_epoch_anchor,
        case.session._dynamic_epoch_seal,
        case.session._dynamic_epoch_nexus_promoted,
        case.session._dynamic_epoch_xye_handoff,
        case.session._dynamic_epoch_notified,
        case.xye._transition_kind,
    )
    frozen_intent = case.session._dynamic_current_intent

    def assert_frozen():
        assert case.accounting.snapshot() == frozen_accounting
        assert (
            _nexus_rows(case.nexus_path),
            tuple(
                (path.name, path.read_bytes())
                for path in _xye_files(case.xye_directory)
            ),
        ) == frozen_outputs
        assert (tuple(events), tuple(early), tuple(nexus_calls)) == frozen_events
        assert (
            case.session._dynamic_epoch_drained,
            case.session._dynamic_epoch_anchor,
            case.session._dynamic_epoch_seal,
            case.session._dynamic_epoch_nexus_promoted,
            case.session._dynamic_epoch_xye_handoff,
            case.session._dynamic_epoch_notified,
            case.xye._transition_kind,
        ) == frozen_epoch
        assert case.session._dynamic_current_intent is frozen_intent
        assert case.session._dynamic_stop_requested is False
        assert case.session._session.cancel_token.cancelled is False
        assert case.session.is_running is True

    with pytest.raises(RuntimeError, match="pending XYE settlement"):
        case.session.submit(case.session.scan.frames[1])
    assert_frozen()
    with pytest.raises(RuntimeError, match="pending XYE settlement"):
        case.session.stop()
    assert_frozen()
    with pytest.raises(RuntimeError, match="pending XYE settlement"):
        case.session.extend_live(second)
    assert_frozen()

    case.session.commit_epoch()
    assert nexus_calls == [case.nexus, case.nexus]

    assert len(early) == 1
    assert isinstance(early[0], RuntimeError)
    assert "pending XYE settlement" in str(early[0])
    assert events == ["nexus-physical", "promote", "xye-physical", "promote"]
    _assert_durable(case, key0, token0, case.nexus_target)
    _assert_durable(case, key0, token0, case.xye_target)

    foreign = DynamicAttemptToken(
        DynamicFrameIdentity("p1-a/foreign-mixed", 1), 1, 1, 2,
    )
    before_accounting = case.accounting.snapshot()
    before_files = _xye_files(case.xye_directory)
    before_rows = _nexus_rows(case.nexus_path)
    before_events = tuple(events)
    with pytest.raises(RuntimeError, match="advancing live intent"):
        case.session.submit(case.session.scan.frames[1], attempt_token=foreign)
    assert case.accounting.snapshot() == before_accounting
    assert _xye_files(case.xye_directory) == before_files
    assert _nexus_rows(case.nexus_path) == before_rows
    assert tuple(events) == before_events
    case.session.extend_live(second)
    key1, token1 = _arm(case.accounting, 1, revision=2)
    assert case.session.submit(case.session.scan.frames[1], attempt_token=token1)
    assert case.session.finish().failed is False
    _assert_durable(case, key1, token1, case.nexus_target)
    _assert_durable(case, key1, token1, case.xye_target)
    assert _nexus_rows(case.nexus_path) == (0, 1)
    assert len(_xye_files(case.xye_directory)) == 2


def test_xye_only_finish_promotes_before_terminal_custody(
    tmp_path, monkeypatch,
):
    from xrd_tools.session import (
        Light1DBufferLayout,
        Light1DCleanupHooks,
        Light1DCustodySlot,
        Light1DCustodyState,
        Light1DLayout,
        Light1DModeLayout,
        SessionResourceAuthority,
        acquire_light_1d_retention,
    )

    case = _open_case(
        tmp_path, "xye-custody", frame_count=1, nexus=False,
    )
    key, token = _arm(case.accounting, 0)
    axis = Light1DBufferLayout(1, 8, "p1-q", "<f8", shared=True)
    layout = Light1DLayout((Light1DModeLayout(
        "default", axis, Light1DBufferLayout(1, 8, "p1-i", "<f8"),
    ),), "default")
    authority = SessionResourceAuthority(capacity_bytes=4096)
    lease = acquire_light_1d_retention(
        authority,
        owner="p1-a-xye",
        generation=1,
        layout=layout,
        requested_rows=1,
        compatibility_byte_ceiling=(
            layout.shared_bytes + layout.per_row_unique_ndarray_bytes
        ),
        gui_thread_id=threading.get_ident(),
    )
    hooks = Light1DCleanupHooks()
    slot = Light1DCustodySlot(
        grant_id=lease.grant_id,
        owner=lease.owner,
        generation=lease.generation,
        cleanup_hooks=hooks,
    )
    case.accounting.bind_light_1d(
        lease, cleanup_hooks=hooks, custody_slot=slot,
    )
    original_adopt = Light1DCustodySlot.adopt
    observations = []

    def observe_adopt(owner, bound_lease, *, cleanup_hooks, terminal):
        if owner is slot:
            observations.append(case.ledger.snapshot().mode_complete)
        return original_adopt(
            owner,
            bound_lease,
            cleanup_hooks=cleanup_hooks,
            terminal=terminal,
        )

    monkeypatch.setattr(Light1DCustodySlot, "adopt", observe_adopt)
    assert case.session.submit(case.session.scan.frames[0], attempt_token=token)
    assert case.session.finish().failed is False
    _assert_durable(case, key, token, case.xye_target)
    assert observations == [frozenset({0})]
    assert slot.state is Light1DCustodyState.RETAINED
    assert case.session not in case.accounting.owner_census()
    slot.release(reason="p1-a-test-close")
    assert authority.snapshot().reserved_bytes == 0


def test_nexus_xye_finish_uses_two_promotions_and_one_terminal_settlement(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.output_transaction import XyeOutputTransaction
    from xrd_tools.reduction import NexusSink

    events = []
    original_nexus_finish = NexusSink.finish
    original_xye_publish = XyeOutputTransaction.publish

    def observe_nexus_finish(owner, result):
        events.append("nexus-physical")
        return original_nexus_finish(owner, result)

    def observe_xye_publish(owner, **kwargs):
        events.append("xye-physical")
        return original_xye_publish(owner, **kwargs)

    monkeypatch.setattr(NexusSink, "finish", observe_nexus_finish)
    monkeypatch.setattr(XyeOutputTransaction, "publish", observe_xye_publish)
    case = _open_case(
        tmp_path, "nexus-xye-finish", frame_count=1, nexus=True,
    )
    boundary_type = type(case.accounting.writer_boundary)
    original_prepare_epoch = boundary_type.prepare_epoch_commit
    original_epoch = boundary_type.epoch_committed
    original_prepare_finish = boundary_type.prepare_session_finish
    original_finish = boundary_type.session_finished

    def prepare_epoch(owner, *args, **kwargs):
        if owner is case.accounting.writer_boundary:
            events.append("prepare-nexus-promotion")
        return original_prepare_epoch(owner, *args, **kwargs)

    def promote_epoch(owner, *args, **kwargs):
        if owner is case.accounting.writer_boundary:
            events.append("promote-nexus")
        return original_epoch(owner, *args, **kwargs)

    def prepare_finish(owner, *args, **kwargs):
        if owner is case.accounting.writer_boundary:
            events.append("prepare-xye-promotion")
        return original_prepare_finish(owner, *args, **kwargs)

    def settle_finish(owner, *args, **kwargs):
        if owner is case.accounting.writer_boundary:
            events.append("terminal-settlement")
        return original_finish(owner, *args, **kwargs)

    monkeypatch.setattr(boundary_type, "prepare_epoch_commit", prepare_epoch)
    monkeypatch.setattr(boundary_type, "epoch_committed", promote_epoch)
    monkeypatch.setattr(boundary_type, "prepare_session_finish", prepare_finish)
    monkeypatch.setattr(boundary_type, "session_finished", settle_finish)
    key, token = _arm(case.accounting, 0)
    assert case.session.submit(case.session.scan.frames[0], attempt_token=token)
    assert case.session.finish().failed is False
    assert events == [
        "nexus-physical",
        "prepare-nexus-promotion",
        "promote-nexus",
        "xye-physical",
        "prepare-xye-promotion",
        "terminal-settlement",
    ]
    _assert_durable(case, key, token, case.nexus_target)
    _assert_durable(case, key, token, case.xye_target)
    assert case.accounting.snapshot().state.value == "finished"


def test_xye_stop_publishes_only_the_accepted_prefix(tmp_path, monkeypatch):
    terminal_events = None
    for nexus in (False, True):
        name = f"stop-prefix-{nexus}"
        case = _open_case(
            tmp_path, name, frame_count=2, nexus=nexus,
        )
        if terminal_events is None:
            terminal_events = _observe_terminal_callbacks(
                monkeypatch, case.accounting.writer_boundary,
            )
        key0, token0 = _arm(case.accounting, 0)
        key1, token1 = _arm(case.accounting, 1, revision=2)
        assert case.session.submit(case.session.scan.frames[0], attempt_token=token0)
        case.session.stop()
        result = case.session.finish(raise_on_failure=False)
        assert result.cancelled is True
        snapshot = case.accounting.snapshot()
        assert snapshot.state.value == "stopped"
        _assert_durable(case, key0, token0, case.xye_target)
        assert (key1, case.mode, case.xye_target) not in snapshot.durable_attempts
        assert snapshot.attempt_states[token1].value == "cancelled"
        assert len(_xye_files(case.xye_directory)) == 1
        callbacks = [
            name for name, owner in terminal_events
            if owner is case.accounting.writer_boundary
        ]
        assert callbacks == ["session_finished"]
        if nexus:
            _assert_durable(case, key0, token0, case.nexus_target)
            assert _nexus_rows(case.nexus_path) == (0,)


def test_xye_zero_row_stop_preserves_prior_bytes_without_publisher_or_cleanup(
    tmp_path, monkeypatch,
):
    import xrd_tools.io.output_transaction as output_transaction
    import xrd_tools.reduction.core as reduction_core
    from xrd_tools.io.output_transaction import OutputTransactionCoordinator

    prepared, write_calls, unlink_calls = [], [], []
    original_prepare = OutputTransactionCoordinator.prepare_xye
    original_write = reduction_core.write_xye
    original_unlink = output_transaction._unlink

    def observe_prepare(owner, directory, *, run_owner):
        transaction = original_prepare(owner, directory, run_owner=run_owner)
        prepared.append(transaction)
        return transaction

    def observe_write(*args, **kwargs):
        write_calls.append(args[0])
        return original_write(*args, **kwargs)

    def observe_unlink(path):
        unlink_calls.append(Path(path).resolve())
        return original_unlink(path)

    monkeypatch.setattr(
        OutputTransactionCoordinator, "prepare_xye", observe_prepare,
    )
    monkeypatch.setattr(reduction_core, "write_xye", observe_write)
    monkeypatch.setattr(output_transaction, "_unlink", observe_unlink)
    terminal_events = None
    for nexus in (False, True):
        name = f"zero-stop-{nexus}"
        xye_directory, _xye_target, _nexus_path, _nexus_target = _targets(
            tmp_path, name, nexus=nexus,
        )
        xye_directory.mkdir(parents=True)
        prior = xye_directory / "prior_9999.xye"
        prior.write_bytes(b"prior-xye-bytes\n")
        before = prior.read_bytes()
        case = _open_case(
            tmp_path,
            name,
            frame_count=1,
            nexus=nexus,
            stale_paths=(prior,),
        )
        if terminal_events is None:
            terminal_events = _observe_terminal_callbacks(
                monkeypatch, case.accounting.writer_boundary,
            )
        case.session.stop()
        result = case.session.finish(raise_on_failure=False)
        assert result.cancelled is True
        assert case.accounting.snapshot().state.value == "stopped"
        assert case.accounting.snapshot().durable == frozenset()
        assert prior.read_bytes() == before
        assert [
            path for path in unlink_calls
            if path.parent == xye_directory.resolve()
        ] == []
        assert write_calls == []
        transaction = prepared[-1].snapshot()
        assert transaction.complete is True
        assert transaction.retryable is False
        assert transaction.staged_indices == ()
        callbacks = [
            name for name, owner in terminal_events
            if owner is case.accounting.writer_boundary
        ]
        assert callbacks == ["session_stopped"]
        if nexus:
            assert not case.nexus_path.exists()


def test_xye_publication_failure_retries_exact_token_without_rereduction_or_restage(
    tmp_path, monkeypatch,
):
    import xrd_tools.reduction.core as reduction_core
    from xrd_tools.io.output_transaction import (
        OutputTransactionCoordinator,
        XyeOutputTransaction,
    )
    from xrd_tools.reduction import NexusSink

    prepared, stage_calls, publish_calls, retry_calls = [], [], [], []
    nexus_finishes = []
    failures = {}
    write_calls = {}
    original_prepare = OutputTransactionCoordinator.prepare_xye
    original_stage = XyeOutputTransaction.stage
    original_publish = XyeOutputTransaction.publish
    original_retry = XyeOutputTransaction.retry_publication
    original_nexus_finish = NexusSink.finish
    original_write = reduction_core.write_xye

    def observe_prepare(owner, directory, *, run_owner):
        transaction = original_prepare(owner, directory, run_owner=run_owner)
        prepared.append((run_owner, transaction))
        return transaction

    def observe_stage(owner, *args, **kwargs):
        stage_calls.append(owner)
        return original_stage(owner, *args, **kwargs)

    def observe_publish(owner, **kwargs):
        publish_calls.append(owner)
        return original_publish(owner, **kwargs)

    def observe_retry(owner, cleanup_token, **kwargs):
        retry_calls.append((owner, cleanup_token))
        return original_retry(owner, cleanup_token, **kwargs)

    def observe_nexus_finish(owner, result):
        nexus_finishes.append(owner)
        return original_nexus_finish(owner, result)

    def flaky_write(path, radial, intensity, sigma=None):
        directory = Path(path).parent.resolve()
        write_calls[directory] = write_calls.get(directory, 0) + 1
        if failures.get(directory, 0):
            failures[directory] -= 1
            raise OSError("forced XYE publication fault")
        return original_write(path, radial, intensity, sigma)

    monkeypatch.setattr(
        OutputTransactionCoordinator, "prepare_xye", observe_prepare,
    )
    monkeypatch.setattr(XyeOutputTransaction, "stage", observe_stage)
    monkeypatch.setattr(XyeOutputTransaction, "publish", observe_publish)
    monkeypatch.setattr(
        XyeOutputTransaction, "retry_publication", observe_retry,
    )
    monkeypatch.setattr(NexusSink, "finish", observe_nexus_finish)
    monkeypatch.setattr(reduction_core, "write_xye", flaky_write)

    for nexus in (False, True):
        integrator = _CountingIntegrator()
        case = _open_case(
            tmp_path,
            f"publication-retry-{nexus}",
            frame_count=1,
            nexus=nexus,
            integrator=integrator,
        )
        run_owner, transaction = prepared[-1]
        failures[case.xye_directory] = 1
        key, token = _arm(case.accounting, 0)
        assert case.session.submit(case.session.scan.frames[0], attempt_token=token)
        with pytest.raises(OSError, match="forced XYE publication fault"):
            case.session.finish()

        debt = transaction.snapshot()
        assert debt.retryable is True
        assert debt.cleanup_token is not None
        snapshot = case.accounting.snapshot()
        assert snapshot.state.value == "active"
        assert (key, case.mode, case.xye_target) not in snapshot.durable_attempts
        assert (key, case.mode, case.xye_target) not in snapshot.pending_durable
        if nexus:
            _assert_durable(case, key, token, case.nexus_target)
            assert nexus_finishes.count(case.nexus) == 1

        assert case.session.finish().failed is False
        assert retry_calls[-1] == (transaction, debt.cleanup_token)
        assert stage_calls.count(transaction) == 1
        assert publish_calls.count(transaction) == 1
        assert sum(owner is transaction for owner, _token in retry_calls) == 1
        assert write_calls[case.xye_directory] == 2
        assert integrator.calls == 1
        _assert_durable(case, key, token, case.xye_target)
        if nexus:
            assert nexus_finishes.count(case.nexus) == 1
        assert run_owner is not None


def test_xye_postpublication_receipt_failure_resumes_without_republication(
    tmp_path, monkeypatch,
):
    """Amendment 1: one exact-ledger fail-before-effect interception."""
    import xrd_tools.reduction.core as reduction_core
    from xrd_tools.io.output_transaction import XyeOutputTransaction
    from xrd_tools.session import (
        Light1DBufferLayout,
        Light1DCleanupHooks,
        Light1DCustodySlot,
        Light1DCustodyState,
        Light1DLayout,
        Light1DModeLayout,
        SessionResourceAuthority,
        StageLedger,
        acquire_light_1d_retention,
    )

    integrator = _CountingIntegrator()
    case = _open_case(
        tmp_path,
        "postpublication-receipt",
        frame_count=1,
        nexus=False,
        integrator=integrator,
    )
    key, token = _arm(case.accounting, 0)
    axis = Light1DBufferLayout(1, 8, "receipt-q", "<f8", shared=True)
    layout = Light1DLayout((Light1DModeLayout(
        "default", axis, Light1DBufferLayout(1, 8, "receipt-i", "<f8"),
    ),), "default")
    authority = SessionResourceAuthority(capacity_bytes=4096)
    lease = acquire_light_1d_retention(
        authority,
        owner="p1-a-receipt",
        generation=1,
        layout=layout,
        requested_rows=1,
        compatibility_byte_ceiling=(
            layout.shared_bytes + layout.per_row_unique_ndarray_bytes
        ),
        gui_thread_id=threading.get_ident(),
    )
    hooks = Light1DCleanupHooks()
    slot = Light1DCustodySlot(
        grant_id=lease.grant_id,
        owner=lease.owner,
        generation=lease.generation,
        cleanup_hooks=hooks,
    )
    case.accounting.bind_light_1d(
        lease, cleanup_hooks=hooks, custody_slot=slot,
    )
    fault = RuntimeError("forced receipt-promotion fault")
    write_calls, stage_calls, durable_calls, failures = [], [], [], [fault]
    publish_returns, prepare_calls, finish_seals, abort_calls = [], [], [], []
    original_write = reduction_core.write_xye
    original_stage = XyeOutputTransaction.stage
    original_publish = XyeOutputTransaction.publish
    original_durable = StageLedger.record_durable
    boundary_type = type(case.accounting.writer_boundary)
    original_prepare = boundary_type.prepare_session_finish
    original_finish = boundary_type.session_finished
    original_abort = boundary_type.epoch_aborted

    def observe_write(*args, **kwargs):
        write_calls.append(args[0])
        return original_write(*args, **kwargs)

    def observe_stage(owner, *args, **kwargs):
        stage_calls.append(owner)
        return original_stage(owner, *args, **kwargs)

    def observe_publish(owner, **kwargs):
        snapshot = original_publish(owner, **kwargs)
        publish_returns.append(snapshot)
        return snapshot

    def fail_once(owner, receipts):
        if owner is case.ledger and failures:
            raise failures.pop()
        durable_calls.append(owner)
        return original_durable(owner, receipts)

    def observe_prepare(owner, *args, **kwargs):
        if owner is case.accounting.writer_boundary:
            prepare_calls.append(owner)
        return original_prepare(owner, *args, **kwargs)

    def observe_finish(owner, session, owner_token, seal, identity):
        if owner is case.accounting.writer_boundary:
            finish_seals.append(seal)
        return original_finish(owner, session, owner_token, seal, identity)

    def observe_abort(owner, *args, **kwargs):
        if owner is case.accounting.writer_boundary:
            abort_calls.append(owner)
        return original_abort(owner, *args, **kwargs)

    monkeypatch.setattr(reduction_core, "write_xye", observe_write)
    monkeypatch.setattr(XyeOutputTransaction, "stage", observe_stage)
    monkeypatch.setattr(XyeOutputTransaction, "publish", observe_publish)
    monkeypatch.setattr(StageLedger, "record_durable", fail_once)
    monkeypatch.setattr(boundary_type, "prepare_session_finish", observe_prepare)
    monkeypatch.setattr(boundary_type, "session_finished", observe_finish)
    monkeypatch.setattr(boundary_type, "epoch_aborted", observe_abort)
    assert case.session.submit(case.session.scan.frames[0], attempt_token=token)
    with pytest.raises(RuntimeError, match="forced receipt-promotion fault") as caught:
        case.session.finish()
    assert caught.value is fault

    retained_seal = case.session._dynamic_finish_seal
    assert retained_seal is not None
    retained_ordinal = retained_seal.ordinal
    assert len(write_calls) == 1
    assert len(_xye_files(case.xye_directory)) == 1
    assert len(publish_returns) == 1
    assert publish_returns[0].complete is True
    pending = case.accounting.snapshot()
    assert (key, case.mode, case.xye_target) not in pending.durable_attempts
    assert (key, case.mode, case.xye_target) in pending.pending_durable
    assert pending.state.value != "finished"
    assert case.session in case.accounting.owner_census()
    assert lease in case.accounting.owner_census()
    assert slot in case.accounting.owner_census()
    assert slot.state is Light1DCustodyState.PENDING
    assert authority.snapshot().reserved_bytes > 0
    assert len(prepare_calls) == 1
    assert finish_seals == [retained_seal]
    assert abort_calls == []

    settled = case.session.finish()
    assert settled.failed is False
    assert settled.error is None
    assert case.session.finish() is settled
    assert durable_calls.count(case.ledger) == 1
    assert len(write_calls) == 1
    assert len(publish_returns) == 1
    assert len(stage_calls) == 1
    assert integrator.calls == 1
    assert case.session._dynamic_finish_seal is retained_seal
    assert case.session._dynamic_finish_seal.ordinal == retained_ordinal
    assert len(prepare_calls) == 1
    assert finish_seals == [retained_seal, retained_seal]
    assert abort_calls == []
    _assert_durable(case, key, token, case.xye_target)
    assert case.accounting.snapshot().state.value == "finished"
    assert case.session not in case.accounting.owner_census()
    assert slot.state is Light1DCustodyState.RETAINED
    slot.release(reason="p1-a-receipt-test-close")
    assert authority.snapshot().reserved_bytes == 0


def test_xye_identity_and_exact_graph_refusals_are_pre_effect(
    tmp_path, monkeypatch,
):
    from xdart.modules.reduction import (
        DynamicXyeReceiptBoundaryRequired,
        open_live_scan_session,
    )
    import xrd_tools.reduction as reduction
    import xrd_tools.reduction.core as reduction_core

    write_calls = []

    def forbidden_write(*args, **kwargs):
        write_calls.append(args)
        raise AssertionError("refused graph reached the XYE publisher")

    monkeypatch.setattr(reduction_core, "write_xye", forbidden_write)

    for ordinal, kind in enumerate((
        "legacy",
        "subclass",
        "proxy",
        "duplicate",
        "duplicate-identity",
        "foreign-boundary",
        "target-mismatch",
        "two-d-only",
    )):
        directory = (tmp_path / f"refusal-{kind}").resolve()
        declared = directory if kind != "target-mismatch" else (
            tmp_path / "different-target"
        ).resolve()
        xye_target = f"xye:{declared}"
        if kind == "two-d-only":
            from xrd_tools.reduction import Integration2DPlan, ReductionPlan
            from xrd_tools.session import (
                DynamicAccountingLimits,
                DynamicRunAccounting,
                ResultMode,
                StageLedger,
            )

            mode = ResultMode.two_d()
            ledger = StageLedger(
                required_modes=(mode,),
                targets_by_mode={mode: (xye_target,)},
            )
            accounting = DynamicRunAccounting(
                ledger,
                run_generation=1,
                limits=DynamicAccountingLimits(2, 3, 8),
            )
            plan = ReductionPlan(
                integration_1d=None,
                integration_2d=Integration2DPlan(npt_rad=2, npt_azim=2),
            )
        else:
            ledger, accounting, _mode = _accounting(xye_target)
            plan = _plan()
        before_dynamic = accounting.snapshot()
        before_stage = ledger.snapshot()
        before_owners = accounting.owner_census()
        exact = _transactional_sink(directory)
        boundary = exact
        if kind == "legacy":
            sink = reduction.XYESink(directory)
            boundary = sink
        elif kind == "subclass":
            sink_type = type(exact)

            class TransactionalSubclass(sink_type):
                pass

            if sink_type is reduction.XYESink:
                sink = TransactionalSubclass(directory)
            else:
                sink = TransactionalSubclass(directory, stale_paths=())
            boundary = sink
        elif kind == "proxy":
            class Proxy:
                output_sink_children = (exact,)

            sink = Proxy()
            boundary = exact
        elif kind == "duplicate":
            other = _transactional_sink(directory)
            sink = reduction.CompositeSink((exact, other))
            boundary = exact
        elif kind == "duplicate-identity":
            sink = reduction.CompositeSink((exact, exact))
            boundary = exact
        elif kind == "foreign-boundary":
            sink = exact
            boundary = _ForeignReceiptBoundary()
        else:
            sink = exact
        with pytest.raises(DynamicXyeReceiptBoundaryRequired):
            open_live_scan_session(
                _live_frames(tmp_path, 1, _CountingIntegrator()),
                plan,
                scan_name=f"refusal-{ordinal}",
                sink=sink,
                executor=1,
                accounting=accounting,
                xye_target=xye_target,
                xye_receipt_boundary=boundary,
            )
        assert accounting.snapshot() == before_dynamic
        assert ledger.snapshot() == before_stage
        assert accounting.owner_census() == before_owners
        assert not directory.exists()
    assert write_calls == []


def test_xye_abort_before_attempt_abandons_but_postattempt_requires_exact_retry(
    tmp_path, monkeypatch,
):
    import xrd_tools.reduction.core as reduction_core
    from xrd_tools.io.output_transaction import (
        OutputTransactionCoordinator,
        TransactionStateError,
        XyeOutputTransaction,
    )
    from xrd_tools.reduction import NexusSink

    prepared, transition_events = [], []
    original_prepare = OutputTransactionCoordinator.prepare_xye
    original_abandon = XyeOutputTransaction.abandon
    original_publish = XyeOutputTransaction.publish
    original_nexus_abort = NexusSink.abort

    def observe_prepare(owner, directory, *, run_owner):
        transaction = original_prepare(owner, directory, run_owner=run_owner)
        prepared.append((run_owner, transaction))
        return transaction

    def observe_abandon(owner, **kwargs):
        transition_events.append(("xye-abandon", owner))
        return original_abandon(owner, **kwargs)

    def observe_publish(owner, **kwargs):
        transition_events.append(("xye-publish", owner))
        return original_publish(owner, **kwargs)

    def observe_nexus_abort(owner, result):
        transition_events.append(("nexus-abort", owner))
        return original_nexus_abort(owner, result)

    monkeypatch.setattr(
        OutputTransactionCoordinator, "prepare_xye", observe_prepare,
    )
    monkeypatch.setattr(XyeOutputTransaction, "abandon", observe_abandon)
    monkeypatch.setattr(XyeOutputTransaction, "publish", observe_publish)
    monkeypatch.setattr(NexusSink, "abort", observe_nexus_abort)

    # Rows 11 and 12: unresolved retryable source work forces the real session
    # abort before the XYE transaction has attempted publication.
    terminal_observation_installed = False
    for nexus in (False, True):
        case = _open_case(
            tmp_path,
            f"pre-attempt-abort-{nexus}",
            frame_count=1,
            nexus=nexus,
        )
        if not terminal_observation_installed:
            _observe_terminal_callbacks(
                monkeypatch,
                case.accounting.writer_boundary,
                transition_events,
            )
            terminal_observation_installed = True
        _key, token = _arm(case.accounting, 0)
        case.accounting.record_failed(
            token, error="source remains partial", retryable=True,
        )
        _run_owner, transaction = prepared[-1]
        event_start = len(transition_events)
        result = case.session.finish(raise_on_failure=False)
        assert result.failed is True
        snapshot = transaction.snapshot()
        assert snapshot.complete is True
        assert snapshot.retryable is False
        assert snapshot.staged_indices == ()
        assert case.accounting.snapshot().state.value == "aborted"
        assert case.session not in case.accounting.owner_census()
        owners = (transaction, case.accounting.writer_boundary) + (
            (case.nexus,) if nexus else ()
        )
        observed = [
            name for name, owner in transition_events[event_start:]
            if any(owner is candidate for candidate in owners)
        ]
        assert observed == (
            ["nexus-abort", "xye-abandon", "epoch_aborted"]
            if nexus else ["xye-abandon", "epoch_aborted"]
        )
        assert "xye-publish" not in observed
        if nexus:
            assert not case.nexus_path.exists()

    # Once the publisher has been attempted, H23 itself refuses abandon and
    # the same ScanSession must discharge the exact retained retry token.
    original_write = reduction_core.write_xye
    failures = [OSError("post-attempt publication fault")]

    def fail_once(*args, **kwargs):
        if failures:
            raise failures.pop()
        return original_write(*args, **kwargs)

    monkeypatch.setattr(reduction_core, "write_xye", fail_once)
    case = _open_case(
        tmp_path, "post-attempt-abort", frame_count=1, nexus=False,
    )
    key, token = _arm(case.accounting, 0)
    run_owner, transaction = prepared[-1]
    assert case.session.submit(case.session.scan.frames[0], attempt_token=token)
    with pytest.raises(OSError, match="post-attempt publication fault"):
        case.session.finish()
    debt = transaction.snapshot()
    assert debt.retryable is True
    assert debt.cleanup_token is not None
    with pytest.raises(TransactionStateError, match="retry"):
        transaction.abandon(run_owner=run_owner)
    assert case.session in case.accounting.owner_census()
    assert case.session.finish().failed is False
    _assert_durable(case, key, token, case.xye_target)
    assert case.session not in case.accounting.owner_census()
