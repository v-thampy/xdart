"""Reusable duck-contract harnesses for the two frozen seams.

``ReductionSink`` and ``FrameSource`` are duck contracts: new
implementations (Tiled sources, a zarr working store, future session
event sinks) conform by behavior, not inheritance.  This module turns
that contract into something a one-line test can verify:

    def test_my_sink_contract(tmp_path):
        check_sink_contract(lambda: MySink(tmp_path / "out.nxs"))

    def test_my_source_contract():
        check_source_contract(lambda: MySource(...), expected_indices=[0, 1])

THREAD DISCIPLINE IS PART OF THE SINK CONTRACT (streaming mode):

    begin()          caller thread, exactly once, before any write
    worker_process() pool worker threads (parallel; must not touch sink
                     state that write() owns)
    write()          the ONE writer thread — never concurrent, never the
                     caller; HDF5 single-writer discipline hangs off this
    replace()        the same writer thread (re-fed index upsert)
    finish()/abort() caller thread, after the writer is done

``ThreadSpySink`` records (hook, thread ident, frame index) per call so
tests can assert those assignments against a REAL streaming session.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from xrd_tools.core.scan import FrameSource, SourceCapabilities
from xrd_tools.reduction import (
    Frame,
    MemorySink,
    ReductionPlan,
    ReductionSession,
    Scan,
)
import xrd_tools.reduction.core as reduction_core


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------

@dataclass
class ThreadSpySink:
    """Sink that records every hook call with its thread ident.

    ``calls`` is a list of ``(hook, thread_ident, frame_index_or_None)``
    tuples, appended under a lock (worker_process arrives in parallel).
    Optionally wraps an inner sink and forwards every hook to it.
    """

    inner: Any | None = None
    calls: list[tuple[str, int, int | None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _record(self, hook: str, frame=None) -> None:
        idx = None if frame is None else int(frame.index)
        with self._lock:
            self.calls.append((hook, threading.get_ident(), idx))

    def begin(self, scan, plan):
        self._record("begin")
        if self.inner is not None:
            self.inner.begin(scan, plan)

    def write(self, frame, reduction):
        self._record("write", frame)
        if self.inner is not None:
            self.inner.write(frame, reduction)

    def replace(self, frame, reduction):
        self._record("replace", frame)
        inner_replace = getattr(self.inner, "replace", None)
        if callable(inner_replace):
            inner_replace(frame, reduction)
        elif self.inner is not None:
            self.inner.write(frame, reduction)

    def worker_process(self, frame, reduction):
        self._record("worker_process", frame)
        inner_wp = getattr(self.inner, "worker_process", None)
        if callable(inner_wp):
            inner_wp(frame, reduction)

    def finish(self, result):
        self._record("finish")
        if self.inner is not None:
            self.inner.finish(result)

    def abort(self, result):
        self._record("abort")
        inner_abort = getattr(self.inner, "abort", None)
        if callable(inner_abort):
            inner_abort(result)

    # convenience views ------------------------------------------------
    def hooks(self) -> list[str]:
        with self._lock:
            return [h for h, _, _ in self.calls]

    def threads_for(self, hook: str) -> set[int]:
        with self._lock:
            return {t for h, t, _ in self.calls if h == hook}

    def frames_for(self, hook: str) -> list[int]:
        with self._lock:
            return [i for h, _, i in self.calls if h == hook and i is not None]


@dataclass
class BoomSink:
    """Sink that raises on a configurable hook (failure-path testing)."""

    boom_on: str = "write"
    message: str = "boom"

    def _maybe_boom(self, hook: str) -> None:
        if hook == self.boom_on:
            raise RuntimeError(self.message)

    def begin(self, scan, plan):
        self._maybe_boom("begin")

    def write(self, frame, reduction):
        self._maybe_boom("write")

    def finish(self, result):
        self._maybe_boom("finish")

    def abort(self, result):
        self._maybe_boom("abort")


# ---------------------------------------------------------------------------
# session driver
# ---------------------------------------------------------------------------

def _fake_r1d(value: float):
    from xrd_tools.core.containers import IntegrationResult1D

    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([value, value + 1.0]),
        sigma=None,
        unit="q_A^-1",
    )


def drive_streaming(sink, monkeypatch, *, n_frames: int = 4,
                    refeed: int | None = None, executor: int = 2):
    """Run a real streaming session against ``sink`` with a stub integrator.

    Returns the finished session.  ``refeed`` re-submits that frame index
    after a drain, exercising the replace path.
    """
    monkeypatch.setattr(reduction_core, "integrate_1d",
                        lambda image, ai, **kw: _fake_r1d(float(np.sum(image))))
    frames = [Frame(i, image=np.full((2, 2), i, dtype=float))
              for i in range(n_frames)]
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("contract", frames, integrator=object()),
        sink=sink, execution="streaming", executor=executor,
    )
    for fr in frames:
        session.submit(fr)
    if refeed is not None:
        assert session.drain(timeout=10)
        session.submit(Frame(refeed,
                             image=np.full((2, 2), refeed, dtype=float)))
    session.finish()
    return session


# ---------------------------------------------------------------------------
# the contract checks
# ---------------------------------------------------------------------------

def check_sink_contract(sink_factory: Callable[[], Any], monkeypatch, *,
                        n_frames: int = 4) -> ThreadSpySink:
    """Drive a real streaming session through the sink and assert the
    observable contract.  Returns the spy for extra assertions.

    Asserted here:
    - begin exactly once, before the first write; finish exactly once,
      after the last write;
    - one write per distinct frame index, none concurrent (single writer
      thread);
    - if the sink defines worker_process, it ran once per frame on pool
      worker threads distinct from the writer thread;
    - begin/finish on the caller thread.
    """
    spy = ThreadSpySink(inner=sink_factory())
    caller = threading.get_ident()
    drive_streaming(spy, monkeypatch, n_frames=n_frames)

    hooks = spy.hooks()
    assert hooks.count("begin") == 1, hooks
    assert hooks.count("finish") == 1, hooks
    assert hooks[0] == "begin" and hooks[-1] == "finish", hooks
    assert sorted(spy.frames_for("write")) == list(range(n_frames))

    # thread discipline -------------------------------------------------
    assert spy.threads_for("begin") == {caller}
    assert spy.threads_for("finish") == {caller}
    writer_threads = spy.threads_for("write")
    assert len(writer_threads) == 1, (
        f"write() must stay on ONE writer thread; saw {len(writer_threads)}"
    )
    assert writer_threads != {caller}, "write() must not run on the caller"
    if callable(getattr(spy.inner, "worker_process", None)):
        wp_threads = spy.threads_for("worker_process")
        assert wp_threads, "worker_process defined but never called"
        assert not (wp_threads & writer_threads), (
            "worker_process must run on pool workers, not the writer thread"
        )
        assert sorted(spy.frames_for("worker_process")) == list(range(n_frames))
    return spy


def check_source_contract(source_factory: Callable[[], Any],
                          expected_indices: list[int], *,
                          chunk_size: int = 2,
                          require_capabilities: bool = True) -> None:
    """Assert the FrameSource duck contract on a ready-to-use source."""
    source = source_factory()

    indices = list(source.frame_indices)
    assert indices == [int(i) for i in indices], "frame_indices must be ints"
    assert indices == expected_indices

    if require_capabilities:
        caps = source.capabilities
        assert isinstance(caps, SourceCapabilities), type(caps)
        assert isinstance(source, FrameSource), (
            "runtime_checkable FrameSource isinstance failed"
        )

    loaded = {}
    for idx in expected_indices:
        img = np.asarray(source.load_frame(idx))
        assert img.ndim == 2, f"load_frame({idx}) must return a 2D image"
        loaded[idx] = img

    seen: list[int] = []
    for images, labels in source.iter_chunks(chunk_size):
        assert len(labels) <= chunk_size
        images = np.asarray(images)
        assert images.shape[0] == len(labels)
        for row, label in zip(images, labels):
            np.testing.assert_array_equal(
                np.asarray(row), loaded[int(label)],
                err_msg=f"iter_chunks row for frame {label} != load_frame",
            )
        seen.extend(int(x) for x in labels)
    assert seen == expected_indices, (
        f"iter_chunks must cover frame_indices in order; got {seen}"
    )

    # scan_manifest() contract (ADR-0006): gated on the has_scan_manifest
    # capability.  When advertised it returns (frame_index, metadata) for every
    # frame, in frame_indices order; when not, it returns None (NOT []) so a
    # caller distinguishes "can't enumerate" from "empty scan".
    manifest_fn = getattr(source, "scan_manifest", None)
    if manifest_fn is not None:
        manifest = manifest_fn()
        if getattr(source.capabilities, "has_scan_manifest", False):
            assert manifest is not None, "has_scan_manifest=True but manifest is None"
            assert [int(i) for i, _m in manifest] == expected_indices, (
                "scan_manifest() indices must equal frame_indices, in order"
            )
            for _i, meta in manifest:
                assert hasattr(meta, "keys"), "manifest metadata must be a mapping"
        else:
            assert manifest is None, (
                "scan_manifest() must return None when has_scan_manifest=False"
            )


__all__ = [
    "BoomSink",
    "MemorySink",
    "ThreadSpySink",
    "check_sink_contract",
    "check_source_contract",
    "drive_streaming",
]
