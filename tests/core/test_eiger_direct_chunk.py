"""Focused W1 direct-chunk ownership, decode, ordering, and cleanup oracle."""

from __future__ import annotations

import pytest


def _prepare_average_scan(module, recipe):
    runner = module.AverageScanRunner(recipe)
    runner._execute_graph = lambda _graph: runner.plan
    value = runner.start()
    if type(value) is module.AverageScanResult:
        raise ValueError(value.diagnostic or value.diagnostic_code)
    assert type(value) is module.AverageScanPlan
    runner.close()
    return value


def test_w1_policy_and_source_binding_are_exact_owner_identity_contracts() -> None:
    from dataclasses import replace
    from types import SimpleNamespace
    import numpy as np
    from xrd_tools.session.policy import SessionResourceRequirements, resolve_session_policy
    from xrd_tools.sources.eiger_direct_chunk import (
        DIRECT_CHUNK_WORKERS, EigerDirectChunkPolicy, direct_chunk_workspace_bytes,
    )
    from xrd_tools.sources.nexus import NexusStackSource
    requirements = SessionResourceRequirements(
        height=3, width=5, native_itemsize=2, modes_1d=1, npt_1d=4,
    )
    frame_bytes = requirements.native_frame_bytes
    allocation = resolve_session_policy(
        requirements, envelope_bytes=64 * 1024 ** 3,
        requests={"owner_block_bytes": 2 * frame_bytes},
        env={}).allocation
    policy = EigerDirectChunkPolicy.from_owner_grant(
        frame_bytes, allocation.owner_block_bytes,
    )
    assert DIRECT_CHUNK_WORKERS == policy.workers == 1
    assert direct_chunk_workspace_bytes(frame_bytes) == 2 * frame_bytes
    assert policy.workspace_bytes == policy.owner_grant_bytes == 2 * frame_bytes
    assert policy.enabled
    refused = EigerDirectChunkPolicy.from_owner_grant(frame_bytes, 2 * frame_bytes - 1)
    assert not refused.enabled and refused.workers == 0
    assert refused.reason == f"owner grant {2 * frame_bytes - 1}B is below {2 * frame_bytes}B"
    source = object.__new__(NexusStackSource)
    source.allocation = source._direct_chunk_policy = source._direct_chunk_fact = None
    source.bind_allocation(allocation)
    source.bind_eiger_direct_chunk(allocation)
    source.bind_eiger_direct_chunk(allocation)
    assert source._direct_chunk_policy == policy
    twin = replace(allocation)
    assert twin == allocation and twin is not allocation
    with pytest.raises(ValueError, match="bound allocation identity"):
        source.bind_eiger_direct_chunk(twin)
    with pytest.raises(ValueError, match="different allocation"):
        source.bind_allocation(twin)
    closed: list[bool] = []
    cursor = SimpleNamespace(close=lambda: closed.append(True))
    calls: list[tuple[object, int]] = []
    source._consumption_cursor = cursor
    source._iter_cursor_chunks = lambda owner, size: (
        calls.append((owner, size)) or iter(((np.arange(4).reshape(1, 2, 2), [0]),)))
    rows = list(source.iter_chunks(1))
    assert calls == [(cursor, 1)] and closed == [True]
    assert rows[0][1] == [0]
    assert rows[0][0].tolist() == [[[0, 1], [2, 3]]]


def test_external_link_direct_chunks_decode_tail_and_stay_detached_in_cursor_order(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import get_ident, enumerate as live_threads
    import h5py
    import hdf5plugin
    import numpy as np
    from xrd_tools.core.scan import SourceKind
    from xrd_tools.io.nexus import NexusImageStack
    from xrd_tools.sources import eiger_direct_chunk as direct
    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.sources.read_plan import plan_reads

    expected = np.stack([np.full((17, 17), value, dtype=np.uint16)
                         for value in (11, 22, 33, 44)])
    assert expected[0].size % 8 != 0
    paths = []
    for segment in range(2):
        path = tmp_path / f"scan_data_{segment + 1:06d}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "data", data=expected[2 * segment:2 * segment + 2],
                chunks=(1, 17, 17), **hdf5plugin.Bitshuffle(cname="lz4"))
        paths.append(path)
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        for position, path in enumerate(paths, 1):
            data[f"data_{position:06d}"] = h5py.ExternalLink(path.name, "/data")

    decoded_types: list[type] = []; raw_threads: list[int] = []
    original_decode = direct.decode_eiger_chunk
    original_read = NexusImageStack.read_eiger_direct_chunk

    def checked_decode(raw, layout):
        decoded_types.append(type(raw))
        return original_decode(raw, layout)

    def checked_read(stack, index, cap):
        raw_threads.append(get_ident())
        return original_read(stack, index, cap)

    monkeypatch.setattr(direct, "decode_eiger_chunk", checked_decode)
    monkeypatch.setattr(NexusImageStack, "read_eiger_direct_chunk", checked_read)
    owner_thread = get_ident()
    frame_bytes = expected[0].nbytes
    policy = direct.EigerDirectChunkPolicy.from_owner_grant(frame_bytes, 2 * frame_bytes)
    state = direct.EigerDirectChunkState(policy)
    plan = plan_reads(len(expected), expected.shape[1:], expected.dtype,
                      (1, 17, 17), frame_bytes, requested_block_frames=1)
    with ContainerCursor(master) as cursor:
        assert cursor.descriptor.kind is SourceKind.EIGER_MASTER
        oversize, reason = cursor._stack.read_eiger_direct_chunk(0, 0)
        assert oversize is None and "exceeds 0B cap" in reason
        frames = [block.array[0] for block in cursor.iter_eiger_direct_blocks(
            plan, policy, cancelled=lambda: False, state=state)]
    assert [frame.tobytes() for frame in frames] == [frame.tobytes() for frame in expected]
    assert decoded_types == [bytearray] * len(expected)
    assert raw_threads == [owner_thread] * (len(expected) + 1)
    assert state.selected and state.direct_frames == len(expected)
    assert state.fallback_frames == 0
    assert state.compressed_high_water_bytes <= frame_bytes
    assert state.compressed_retained_bytes == 0
    assert all(frame.flags.writeable for frame in frames)
    frames[-1][0, 0] = 65535
    assert frames[-1][0, 0] == 65535
    assert not any(thread.name.startswith("xrd-tools-eiger-decode")
                   for thread in live_threads())


def test_average_window_uses_exact_direct_grant_and_matches_conventional_frames(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py
    import hdf5plugin
    import numpy as np
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.reduction import AverageScanRecipe, ReductionPlan
    from xrd_tools.reduction import average as average_module
    from xrd_tools.sources.execution_graph import (
        open_source_execution_graph, qualify_source_execution_graph,
    )
    from xrd_tools.sources.cursor import ContainerCursor
    from threading import enumerate as live_threads

    expected = np.stack([
        np.full((17, 17), value, dtype=np.uint16) for value in (3, 7, 11)
    ])
    data_path = tmp_path / "average_data_000001.h5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset(
            "data", data=expected, chunks=(1, 17, 17),
            **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    master = tmp_path / "average_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data["data_000001"] = h5py.ExternalLink(data_path.name, "/data")

    source = SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry")
    recipe = AverageScanRecipe(source, tmp_path / "average.nxs", ReductionPlan())
    plan = _prepare_average_scan(average_module, recipe)
    assert plan.direct_eiger_eligible
    graph = qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    )
    policy = average_module._average_direct_chunk_policy(plan, graph)
    native_bytes = expected[0].nbytes
    conventional = average_module._conventional_owner_block_bytes((17, 17))
    quantum = 8 * 17 * 17
    assert plan.allocation.owner_block_bytes == (
        (conventional + native_bytes + quantum - 1) // quantum * quantum
    )
    assert plan.allocation.owner_block_bytes >= conventional + native_bytes
    assert policy.enabled and policy.workspace_bytes == 2 * native_bytes
    assert policy.owner_grant_bytes == 2 * native_bytes

    with open_source_execution_graph(graph) as conventional:
        conventional_frames = [
            conventional.read_native(index).copy() for index in range(len(expected))
        ]
    original_close = ContainerCursor.close
    close_observations = []
    def checked_close(cursor):
        close_observations.append(not any(
            thread.name.startswith("xrd-tools-eiger-decode")
            for thread in live_threads()
        ))
        return original_close(cursor)
    monkeypatch.setattr(ContainerCursor, "close", checked_close)
    with open_source_execution_graph(
        graph, direct_chunk_policy=policy,
    ) as direct:
        direct_frames = [
            direct.read_native(index).copy() for index in range(len(expected))
        ]
        fact = direct.direct_chunk_fact
    assert [value.tobytes() for value in direct_frames] == [
        value.tobytes() for value in conventional_frames
    ] == [value.tobytes() for value in expected]
    assert fact.selected and fact.direct_frames == len(expected)
    assert fact.fallback_frames == 0
    assert close_observations == [True]


def test_average_direct_iterator_cancellation_is_terminal_cancelled(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event
    import h5py
    import hdf5plugin
    import numpy as np
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.reduction import AverageScanRecipe, ReductionPlan
    from xrd_tools.reduction import average as average_module
    from xrd_tools.sources.cursor import ContainerCursor

    master = tmp_path / "cancel_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("data").create_dataset(
            "data_000001", data=np.ones((1, 17, 17), dtype="u2"),
            chunks=(1, 17, 17), **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    cancelled = Event()

    def stop_during_decode(*_args, **_kwargs):
        cancelled.set()
        if False:
            yield None

    monkeypatch.setattr(
        ContainerCursor, "iter_eiger_direct_blocks", stop_during_decode,
    )
    runner = average_module.AverageScanRunner(AverageScanRecipe(
        SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry"),
        tmp_path / "cancelled.nxs", ReductionPlan(),
    ))
    result = runner.start(cancel_token=cancelled)
    assert type(result) is average_module.AverageScanResult
    assert (result.disposition, result.diagnostic_code) == (
        "CANCELLED", "AVERAGE_CANCELLED",
    )
    assert runner.close() is result


@pytest.mark.parametrize(
    ("failure_mode", "cursor_pending"),
    (("fail-once", False), ("permanent", True)),
)
def test_average_direct_shutdown_failure_is_terminal_not_retryable(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
    failure_mode: str, cursor_pending: bool,
) -> None:
    import h5py
    import hdf5plugin
    import numpy as np
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.reduction import AverageScanRecipe, ReductionPlan
    from xrd_tools.reduction import average as average_module
    from xrd_tools.sources.cursor import ContainerCursor, ReadBlock

    master = tmp_path / f"shutdown_{failure_mode.replace('-', '_')}_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("data").create_dataset(
            "data_000001", data=np.ones((1, 17, 17), dtype="u2"),
            chunks=(1, 17, 17), **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    target = tmp_path / f"shutdown-{failure_mode}.nxs"
    iterator_close_calls = []
    cursors = []

    class FailingCloseIterator:
        def __init__(self, cursor, state):
            self.cursor = cursor
            self.index = 0
            state.select()

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= self.cursor.descriptor.frame_count:
                raise StopIteration
            index = self.index
            self.index += 1
            return ReadBlock(
                index, index + 1,
                self.cursor.read_frame(index)[np.newaxis, ...],
            )

        def close(self):
            iterator_close_calls.append(self)
            if failure_mode == "permanent" or len(iterator_close_calls) == 1:
                raise OSError(f"{failure_mode} decoder shutdown")

    def failing_iterator(cursor, _plan, _policy, *, cancelled, state):
        del cancelled
        cursors.append(cursor)
        return FailingCloseIterator(cursor, state)

    monkeypatch.setattr(
        ContainerCursor, "iter_eiger_direct_blocks", failing_iterator,
    )
    cursor_close_calls = []
    real_cursor_close = ContainerCursor.close
    if cursor_pending:
        def cursor_close(cursor):
            if cursor not in cursors:
                return real_cursor_close(cursor)
            cursor_close_calls.append(cursor)
            if len(cursor_close_calls) == 1:
                raise OSError("cursor close pending")
            return real_cursor_close(cursor)

        monkeypatch.setattr(ContainerCursor, "close", cursor_close)
    runner = average_module.AverageScanRunner(AverageScanRecipe(
        SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry"),
        target, ReductionPlan(),
    ))
    outcome = runner.start()
    assert runner.plan is not None, outcome
    assert runner.plan.direct_eiger_eligible
    if cursor_pending:
        assert type(outcome) is average_module.AverageScanPending
        assert outcome.phase is average_module.AveragePendingPhase.SOURCE_CLEANUP
        result = runner.command(average_module.AverageCommand.RETRY, outcome)
        assert len(cursor_close_calls) == 2
    else:
        result = outcome
    assert (result.disposition, result.diagnostic_code) == (
        "REFUSED", "AVERAGE_SOURCE_CLEANUP_FAILED",
    )
    assert type(result) is average_module.AverageScanResult
    assert len(iterator_close_calls) == 1 and runner.pending is None
    assert len(cursors) == 1 and cursors[0].closed
    assert runner.close() is result
    assert not target.exists()


def test_average_window_direct_layout_refusal_falls_back_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py
    import hdf5plugin
    import numpy as np
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.io.nexus import NexusImageStack
    from xrd_tools.reduction import AverageScanRecipe, ReductionPlan
    from xrd_tools.reduction import average as average_module
    from xrd_tools.sources.execution_graph import (
        open_source_execution_graph, qualify_source_execution_graph,
    )

    expected = np.stack([
        np.full((2, 2), value, dtype=np.uint16) for value in (5, 9)
    ])
    master = tmp_path / "fallback_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("data").create_dataset(
            "data_000001", data=expected, chunks=(1, 2, 2),
            **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    source = SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry")
    plan = _prepare_average_scan(average_module, AverageScanRecipe(
        source, tmp_path / "fallback.nxs", ReductionPlan(),
    ))
    graph = qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    )
    policy = average_module._average_direct_chunk_policy(plan, graph)
    assert plan.direct_eiger_eligible and policy is not None
    calls = []
    monkeypatch.setattr(
        NexusImageStack, "eiger_direct_chunk_layout",
        lambda _self: (calls.append(True) or (None, "synthetic layout refusal")),
    )
    with open_source_execution_graph(
        graph, direct_chunk_policy=policy,
    ) as window:
        frames = [window.read_native(index).copy() for index in range(len(expected))]
        fact = window.direct_chunk_fact
    assert len(calls) == 1
    assert [value.tobytes() for value in frames] == [
        value.tobytes() for value in expected
    ]
    assert not fact.selected and fact.direct_frames == 0
    assert fact.fallback_frames == len(expected)
    assert fact.reason == "synthetic layout refusal"


def test_average_provenance_records_terminal_direct_and_fallback_counts(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py
    import hdf5plugin
    import numpy as np
    from xrd_tools.core.containers import IntegrationResult1D
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.io.nexus import NexusImageStack
    from xrd_tools.reduction import (
        AverageScanRecipe, AverageScanResult, AverageScanRunner, ReductionPlan,
    )
    from xrd_tools.reduction import core as reduction_core

    expected = np.stack([
        np.full((17, 17), value, dtype=np.uint16) for value in (5, 9)
    ])
    data_path = tmp_path / "provenance_data_000001.h5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset(
            "data", data=expected, chunks=(1, 17, 17),
            **hdf5plugin.Bitshuffle(cname="lz4"),
        )
    master = tmp_path / "provenance_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("data")["data_000001"] = h5py.ExternalLink(
            data_path.name, "/data",
        )

    real_read = NexusImageStack.read_eiger_direct_chunk

    def one_runtime_fallback(stack, index, cap):
        if index == 1:
            return None, "synthetic runtime fallback"
        return real_read(stack, index, cap)

    monkeypatch.setattr(
        NexusImageStack, "read_eiger_direct_chunk", one_runtime_fallback,
    )
    monkeypatch.setattr(
        reduction_core, "integrate_1d",
        lambda image, _integrator, *, npt, **_kwargs: IntegrationResult1D(
            np.arange(npt, dtype=float),
            np.full(npt, float(np.nanmean(image))), None, "q_A^-1",
        ),
    )
    target = tmp_path / "average.nxs"
    runner = AverageScanRunner(AverageScanRecipe(
        SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry"),
        target, ReductionPlan(),
    ))
    result = runner.start()
    assert type(result) is AverageScanResult
    assert result.disposition == "COMMITTED", (
        result.diagnostic_code, result.diagnostic,
    )
    assert runner.plan.direct_eiger_eligible
    assert runner.close() is result

    persisted = read_provenance(result.target)["config"]["average_scan_v1"]
    assert persisted["direct_eiger_eligible"] is True
    fact = persisted["direct_eiger_execution"]
    assert fact == {
        "source": str(master.resolve()),
        "selected": True,
        "reason": "selected; frame fallback: synthetic runtime fallback",
        "workers": 1,
        "owner_grant_bytes": 2 * expected[0].nbytes,
        "workspace_bytes": 2 * expected[0].nbytes,
        "direct_frames": 1,
        "fallback_frames": 1,
        "compressed_high_water_bytes": fact["compressed_high_water_bytes"],
    }
    assert 0 < fact["compressed_high_water_bytes"] <= expected[0].nbytes


def test_average_plan_qualifies_layout_before_direct_grant(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py
    import numpy as np
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.io.nexus import NexusImageStack
    from xrd_tools.reduction import AverageScanRecipe, ReductionPlan
    from xrd_tools.reduction import average as average_module
    from xrd_tools.sources.execution_graph import qualify_source_execution_graph

    expected = np.stack([
        np.full((2, 2), value, dtype=np.uint16) for value in (5, 9)
    ])
    master = tmp_path / "conventional_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("data").create_dataset(
            "data_000001", data=expected, chunks=(1, 2, 2),
        )
    source = SourceSpec(master, SourceKind.EIGER_MASTER, entry="entry")
    calls = []
    real_layout = NexusImageStack.eiger_direct_chunk_layout

    def observed_layout(owner):
        calls.append(True)
        return real_layout(owner)

    monkeypatch.setattr(
        NexusImageStack, "eiger_direct_chunk_layout", observed_layout,
    )
    plan = _prepare_average_scan(average_module, AverageScanRecipe(
        source, tmp_path / "conventional.nxs", ReductionPlan(),
    ))
    graph = qualify_source_execution_graph(
        source, reader_binding="average_closed_v1",
    )
    assert calls == [True]
    assert not plan.direct_eiger_eligible
    assert plan.allocation.owner_block_bytes == (
        average_module._conventional_owner_block_bytes((2, 2))
    )
    assert average_module._average_direct_chunk_policy(plan, graph) is None


def test_missing_imagecodecs_refuses_direct_decode_and_uses_conventional_blocks(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    import h5py
    import numpy as np
    from xrd_tools.core.scan import SourceKind
    from xrd_tools.sources import eiger_direct_chunk as direct
    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.sources.probe import ProbeState
    from xrd_tools.sources.read_plan import plan_reads

    expected = np.stack([
        np.full((2, 2), value, dtype=np.uint16) for value in (7, 8, 9, 10)
    ])
    data_path = tmp_path / "scan_data_000001.h5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("data", data=expected, chunks=(1, 2, 2))
    master = tmp_path / "scan_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data["data_000001"] = h5py.ExternalLink(data_path.name, "/data")

    original_import = builtins.__import__

    def import_without_imagecodecs(name, *args, **kwargs):
        if name == "imagecodecs":
            raise ModuleNotFoundError("imagecodecs excluded by test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_imagecodecs)
    monkeypatch.setattr(
        direct,
        "ThreadPoolExecutor",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("missing codec constructed a decoder executor")
        ),
    )
    frame_bytes = expected[0].nbytes
    policy = direct.EigerDirectChunkPolicy.from_owner_grant(
        frame_bytes, 2 * frame_bytes,
    )
    assert policy.enabled
    state = direct.EigerDirectChunkState(policy)
    plan = plan_reads(
        len(expected), expected.shape[1:], expected.dtype, (1, 2, 2),
        2 * frame_bytes,
    )
    with ContainerCursor(master) as cursor:
        assert cursor.descriptor.kind is SourceKind.EIGER_MASTER
        assert cursor.descriptor.state is ProbeState.READY
        assert cursor.descriptor.finalized
        blocks = list(cursor.iter_eiger_direct_blocks(
            plan, policy, cancelled=lambda: False, state=state,
        ))

    assert [int(value) for block in blocks for value in block.array[:, 0, 0]] == [
        7, 8, 9, 10,
    ]
    assert state.selected is False
    assert state.direct_frames == 0
    assert state.fallback_frames == len(expected)
    assert state.reason == "imagecodecs unavailable: ModuleNotFoundError"


def test_direct_iterator_preserves_order_and_counts_only_observed_per_frame_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    import numpy as np
    from xrd_tools.sources import eiger_direct_chunk as direct
    from xrd_tools.sources.cursor import ContainerCursor, ReadBlock

    layout = direct.EigerDirectChunkLayout((2, 2), "|u1", 4)
    refused = direct.EigerDirectChunkPolicy.from_owner_grant(4, 7)
    refusal_state = direct.EigerDirectChunkState(refused)
    conventional = (ReadBlock(2, 4, np.asarray([[[2, 2], [2, 2]], [[3, 3], [3, 3]]])),
                    ReadBlock(4, 5, np.asarray([[[4, 4], [4, 4]]])))
    cursor = SimpleNamespace(
        _require_readable=lambda: None,
        iter_blocks=lambda _plan: iter(conventional),
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            direct, "ThreadPoolExecutor",
            lambda **_kwargs: (_ for _ in ()).throw(
                AssertionError("whole-route refusal constructed a decoder")
            ),
        )
        blocks = list(ContainerCursor.iter_eiger_direct_blocks(
            cursor, object(), refused, cancelled=lambda: False, state=refusal_state))
    assert [int(value) for block in blocks for value in block.array[:, 0, 0]] == [2, 3, 4]
    assert not refusal_state.selected and refusal_state.direct_frames == 0
    assert refusal_state.fallback_frames == 3
    assert refusal_state.reason == refused.reason

    monkeypatch.setattr(direct, "decode_eiger_chunk",
                        lambda raw, spec: np.full(spec.frame_shape, raw[0], dtype=np.uint8))
    mixed_state = direct.EigerDirectChunkState(
        direct.EigerDirectChunkPolicy.from_owner_grant(4, 8)
    )
    rows = list(direct.iter_ordered_eiger_frames(
        (0, 1, 2), layout=layout, policy=mixed_state.policy,
        read_raw=lambda index, _cap: (
            (None, "synthetic oversize") if index == 1
            else (bytearray([index]), "")
        ),
        read_fallback=lambda index: np.full((2, 2), index, dtype=np.uint8),
        cancelled=lambda: False, state=mixed_state,
    ))
    assert [index for index, _array in rows] == [0, 1, 2]
    assert [int(array[0, 0]) for _index, array in rows] == [0, 1, 2]
    assert mixed_state.selected is False
    assert mixed_state.direct_frames == 2 and mixed_state.fallback_frames == 1
    assert mixed_state.reason == "selected; frame fallback: synthetic oversize"
    assert mixed_state.compressed_high_water_bytes <= layout.frame_bytes


@pytest.mark.parametrize("mode", ("decode-error", "cancel"))
def test_direct_iterator_error_or_cancel_joins_decoder(
    mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from threading import Event, get_ident, enumerate as live_threads
    import numpy as np
    from xrd_tools.sources import eiger_direct_chunk as direct

    owner_thread = get_ident()
    raw_threads: list[int] = []
    started = Event()
    error = RuntimeError("decode failed")

    def decode(raw, layout):
        assert isinstance(raw, bytearray)
        started.set()
        if mode == "decode-error":
            raise error
        return np.full(layout.frame_shape, raw[0], dtype=np.uint8)

    def read_raw(_index, _cap):
        raw_threads.append(get_ident())
        return bytearray([9]), ""

    monkeypatch.setattr(direct, "decode_eiger_chunk", decode)
    layout = direct.EigerDirectChunkLayout((1,), "|u1", 1)
    policy = direct.EigerDirectChunkPolicy.from_owner_grant(1, 2)
    state = direct.EigerDirectChunkState(policy)
    iterator = direct.iter_ordered_eiger_frames(
        (0,), layout=layout, policy=policy, read_raw=read_raw,
        read_fallback=lambda _index: np.zeros((1,), dtype=np.uint8),
        cancelled=lambda: mode == "cancel" and started.is_set(), state=state,
    )
    if mode == "decode-error":
        with pytest.raises(RuntimeError, match="decode failed") as caught:
            list(iterator)
        assert caught.value is error
    else:
        assert list(iterator) == []
    assert started.is_set() and raw_threads == [owner_thread]
    assert not any(thread.name.startswith("xrd-tools-eiger-decode")
                   for thread in live_threads())
