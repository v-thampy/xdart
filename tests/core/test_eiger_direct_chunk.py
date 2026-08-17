"""Focused W1 direct-chunk ownership, decode, ordering, and cleanup oracle."""

from __future__ import annotations

import pytest


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
