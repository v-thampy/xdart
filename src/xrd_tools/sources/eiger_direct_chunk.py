"""Bounded, HDF5-object-free Eiger bitshuffle/LZ4 decoding."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import struct
import numpy as np


DIRECT_CHUNK_WORKERS = 1


@dataclass(frozen=True, slots=True)
class EigerDirectChunkLayout:
    frame_shape: tuple[int, ...]
    dtype_str: str
    frame_bytes: int


@dataclass(frozen=True, slots=True)
class EigerDirectChunkPolicy:
    workers: int
    frame_bytes: int
    owner_grant_bytes: int
    workspace_bytes: int
    reason: str = ""

    @property
    def enabled(self) -> bool:
        return self.workers == DIRECT_CHUNK_WORKERS and not self.reason

    @classmethod
    def from_owner_grant(cls, frame_bytes: int, owner_grant_bytes: int):
        frame_bytes, owner_grant_bytes = int(frame_bytes), int(owner_grant_bytes)
        required = direct_chunk_workspace_bytes(frame_bytes)
        reason = ("" if owner_grant_bytes >= required else
                  f"owner grant {owner_grant_bytes}B is below {required}B")
        return cls(DIRECT_CHUNK_WORKERS if not reason else 0, frame_bytes,
                   owner_grant_bytes, required, reason)


@dataclass(frozen=True, slots=True)
class EigerDirectChunkFact:
    source: str
    selected: bool
    reason: str
    workers: int
    owner_grant_bytes: int
    workspace_bytes: int
    direct_frames: int
    fallback_frames: int
    compressed_high_water_bytes: int

    def log_line(self, prefix: str) -> str:
        return (
            f"{prefix} source={self.source} selected={self.selected} "
            f"reason={self.reason!r} workers={self.workers} "
            f"owner-grant={self.owner_grant_bytes} workspace={self.workspace_bytes} "
            f"direct={self.direct_frames} fallback={self.fallback_frames} "
            f"compressed-high-water={self.compressed_high_water_bytes}"
        )


@dataclass(slots=True)
class EigerDirectChunkState:
    policy: EigerDirectChunkPolicy
    selected: bool = False
    reason: str = ""
    direct_frames: int = 0
    fallback_frames: int = 0
    compressed_retained_bytes: int = 0
    compressed_high_water_bytes: int = 0

    def refuse(self, reason: str, frame_count: int) -> None:
        self.selected = False
        self.reason = str(reason)
        self.fallback_frames += int(frame_count)

    def select(self) -> None:
        self.selected = True
        self.reason = "selected"

    def retain(self, size: int) -> None:
        self.compressed_retained_bytes += int(size)
        self.compressed_high_water_bytes = max(
            self.compressed_high_water_bytes,
            self.compressed_retained_bytes,
        )

    def release(self, size: int) -> None:
        self.compressed_retained_bytes -= int(size)

    def fact(self, source: str) -> EigerDirectChunkFact:
        return EigerDirectChunkFact(
            source, self.selected, self.reason or self.policy.reason,
            self.policy.workers, self.policy.owner_grant_bytes,
            self.policy.workspace_bytes, self.direct_frames, self.fallback_frames,
            self.compressed_high_water_bytes)


def direct_chunk_workspace_bytes(frame_bytes: int) -> int:
    """Two slots, each bounded by compressed/shuffled or shuffled/output."""
    return 2 * DIRECT_CHUNK_WORKERS * int(frame_bytes)


def decoder_unavailability() -> str:
    try:
        import imagecodecs
    except Exception as error:
        return f"imagecodecs unavailable: {type(error).__name__}"
    if not getattr(getattr(imagecodecs, "LZ4H5", None), "available", False):
        return "imagecodecs LZ4H5 codec unavailable"
    if not getattr(getattr(imagecodecs, "BITSHUFFLE", None), "available", False):
        return "imagecodecs bitshuffle codec unavailable"
    return ""


def decode_eiger_chunk(
    raw: bytearray,
    layout: EigerDirectChunkLayout,
) -> np.ndarray:
    """Decode one detached raw chunk with the validated probe semantics."""
    import imagecodecs

    if len(raw) < 12:
        raise ValueError("compressed Eiger chunk is shorter than its >QI header")
    byte_count, block_bytes = struct.unpack(">QI", raw[:12])
    dtype = np.dtype(layout.dtype_str)
    if byte_count != layout.frame_bytes:
        raise ValueError(
            f"Eiger chunk byte count {byte_count} != {layout.frame_bytes}"
        )
    if block_bytes <= 0 or block_bytes % dtype.itemsize:
        raise ValueError(f"invalid bitshuffle block byte count {block_bytes}")
    block_items = block_bytes // dtype.itemsize
    if block_items % 8:
        raise ValueError(
            f"bitshuffle block item count is not divisible by 8: {block_items}"
        )
    item_count = byte_count // dtype.itemsize
    if item_count * dtype.itemsize != byte_count:
        raise ValueError("Eiger chunk byte count is not dtype-aligned")
    tail_bytes = (item_count % 8) * dtype.itemsize
    prefix_bytes = byte_count - tail_bytes
    if prefix_bytes <= 0:
        raise ValueError("bitshuffle/LZ4 payload has no framed prefix")
    tail = bytes(raw[-tail_bytes:]) if tail_bytes else b""
    if tail_bytes:
        struct.pack_into(">Q", raw, 0, prefix_bytes)
    framed = memoryview(raw)[:-tail_bytes] if tail_bytes else memoryview(raw)
    shuffled = bytearray(prefix_bytes)
    result = imagecodecs.lz4h5_decode(framed, out=shuffled)
    if len(result) != prefix_bytes:
        raise ValueError(
            f"LZ4H5 output has {len(result)} bytes, expected {prefix_bytes}"
        )
    del result, framed, raw
    decoded = bytearray(byte_count)
    prefix = memoryview(decoded)[:prefix_bytes]
    result = imagecodecs.bitshuffle_decode(
        shuffled, itemsize=dtype.itemsize, blocksize=block_items, out=prefix,
    )
    if len(result) != prefix_bytes:
        raise ValueError(
            f"bitshuffle output has {len(result)} bytes, expected {prefix_bytes}"
        )
    del result, prefix, shuffled
    if tail:
        decoded[prefix_bytes:] = tail
    array = np.frombuffer(decoded, dtype=dtype).reshape(layout.frame_shape)
    if (array.nbytes != layout.frame_bytes or not array.flags.c_contiguous
            or not array.flags.writeable):
        raise ValueError("decoded Eiger frame violates the native-array contract")
    return array


def _decode_detached_holder(holder: list[bytearray], layout: EigerDirectChunkLayout) -> np.ndarray:
    # Pop the bytearray out of ThreadPoolExecutor's retained argument graph so
    # ``decode_eiger_chunk`` can release it before allocating final output.
    return decode_eiger_chunk(holder.pop(), layout)


def _ordered_result(
    future: Future[np.ndarray], cancelled: Callable[[], bool],
) -> np.ndarray | None:
    while not cancelled():
        try:
            value = future.result(timeout=0.05)
        except TimeoutError:
            continue
        return None if cancelled() else value
    return None


def iter_ordered_eiger_frames(
    indices: Iterable[int],
    *,
    layout: EigerDirectChunkLayout,
    policy: EigerDirectChunkPolicy,
    read_raw: Callable[[int, int], tuple[bytearray | None, str]],
    read_fallback: Callable[[int], np.ndarray],
    cancelled: Callable[[], bool],
    state: EigerDirectChunkState,
) -> Iterator[tuple[int, np.ndarray]]:
    """Read on the owner thread; decode at most two detached chunks in order."""
    pending: deque[tuple[int, int, Future[np.ndarray]]] = deque()
    executor = ThreadPoolExecutor(
        max_workers=policy.workers, thread_name_prefix="xrd-tools-eiger-decode",
    )
    primary: BaseException | None = None

    def take_head() -> tuple[int, np.ndarray] | None:
        index, size, future = pending[0]
        array = _ordered_result(future, cancelled)
        if array is None:
            return None
        pending.popleft()
        state.release(size)
        state.direct_frames += 1
        return index, array

    try:
        for index in indices:
            if cancelled():
                return
            raw, _fallback_reason = read_raw(int(index), layout.frame_bytes)
            if cancelled():
                return
            if raw is None:
                state.reason = f"selected; frame fallback: {_fallback_reason}"
                while pending:
                    row = take_head()
                    if row is None or cancelled():
                        return
                    yield row
                state.fallback_frames += 1
                fallback = np.asarray(read_fallback(int(index)))
                if cancelled():
                    return
                yield int(index), fallback
                continue
            size = len(raw)
            future = executor.submit(_decode_detached_holder, [raw], layout)
            pending.append((int(index), size, future))
            state.retain(size)
            del raw
            if len(pending) >= policy.workers:
                row = take_head()
                if row is None or cancelled():
                    return
                yield row
        while pending:
            row = take_head()
            if row is None or cancelled():
                return
            yield row
    except BaseException as error:
        primary = error
        raise
    finally:
        for _index, _size, future in pending:
            future.cancel()
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            if primary is None:
                raise
