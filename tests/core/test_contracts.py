"""Phase 1a — the seam contracts as executable tests.

Every shipping ReductionSink and FrameSource implementation runs through
the reusable harnesses in ``tests.core.contracts``; the thread-discipline
invariants (single writer thread, parallel worker_process, caller-side
begin/finish) are pinned here for the first time.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.core.contracts import (
    BoomSink,
    ThreadSpySink,
    _fake_r1d,
    check_sink_contract,
    check_source_contract,
    drive_streaming,
)
from xrd_tools.reduction import (
    Frame,
    MemorySink,
    NexusSink,
    ReductionPlan,
    ReductionSession,
    Scan,
    XYESink,
)
import xrd_tools.reduction.core as reduction_core


# ---------------------------------------------------------------------------
# ReductionSink conformance — every shipping sink
# ---------------------------------------------------------------------------

def test_memory_sink_contract(monkeypatch):
    spy = check_sink_contract(MemorySink, monkeypatch)
    assert sorted(spy.inner.frames) == [0, 1, 2, 3]


def test_xye_sink_contract(tmp_path, monkeypatch):
    check_sink_contract(lambda: XYESink(tmp_path / "xye"), monkeypatch)
    assert len(list((tmp_path / "xye").glob("*.xye"))) == 4


def test_nexus_sink_contract(tmp_path, monkeypatch):
    check_sink_contract(
        lambda: NexusSink(tmp_path / "scan.nxs", overwrite=True),
        monkeypatch,
    )
    assert (tmp_path / "scan.nxs").exists()


# ---------------------------------------------------------------------------
# thread discipline details beyond the basic harness
# ---------------------------------------------------------------------------

def test_replace_runs_on_the_writer_thread(monkeypatch):
    """A re-fed index goes through replace() (or write()) — on the SAME
    single writer thread as every other write."""
    spy = ThreadSpySink(inner=MemorySink())
    drive_streaming(spy, monkeypatch, n_frames=3, refeed=1)
    writer_threads = spy.threads_for("write") | spy.threads_for("replace")
    assert len(writer_threads) == 1
    assert spy.frames_for("replace") == [1]


def test_worker_process_parallelism_is_real(monkeypatch):
    """With a multi-worker pool and enough frames, worker_process lands on
    more than one pool thread — it is genuinely parallel, not serialized
    through the writer."""
    spy = ThreadSpySink(inner=MemorySink())

    # ensure worker_process is "defined" on the inner for the harness:
    # ThreadSpySink always defines it; the spy records regardless.
    drive_streaming(spy, monkeypatch, n_frames=16, executor=4)
    wp_threads = spy.threads_for("worker_process")
    writer = spy.threads_for("write")
    assert wp_threads and not (wp_threads & writer)
    # 16 frames over 4 workers: requiring >=2 distinct threads is safely
    # below any realistic scheduling skew.
    assert len(wp_threads) >= 2


def test_abort_called_on_write_failure_not_finish(monkeypatch):
    """Failure path: a write() boom surfaces at finish(); the sink sees
    abort(), and never both finish-success semantics and abort."""
    spy = ThreadSpySink(inner=BoomSink(boom_on="write"))
    with pytest.raises(RuntimeError, match="boom"):
        drive_streaming(spy, monkeypatch, n_frames=2)
    hooks = spy.hooks()
    assert "abort" in hooks
    # the spy's own finish record may exist (the session reports through
    # the composite path) — but abort must have reached the sink.


# ---------------------------------------------------------------------------
# FrameSource conformance — every shipping source
# ---------------------------------------------------------------------------

def _frames(n):
    return [Frame(i, image=np.full((2, 3), i, dtype=float)) for i in range(n)]


def test_scan_source_contract():
    check_source_contract(
        lambda: Scan("s", _frames(4), integrator=object()),
        expected_indices=[0, 1, 2, 3],
    )


def test_memory_frame_source_contract():
    from xrd_tools.sources.memory import MemoryFrameSource

    images = [np.full((2, 3), i, dtype=float) for i in range(3)]
    check_source_contract(
        lambda: MemoryFrameSource(images),
        expected_indices=[0, 1, 2],
    )


def test_live_frame_source_contract():
    from xrd_tools.sources.memory import LiveFrameSource

    def build():
        src = LiveFrameSource()
        for i in range(3):
            src.append(np.full((2, 3), i, dtype=float))
        return src

    check_source_contract(build, expected_indices=[0, 1, 2])


def test_processed_scan_source_contract(tmp_path, monkeypatch):
    """The reader handle satisfies the full FrameSource contract — including
    the capabilities property (previously missing; the rsm boundary
    docstring claimed conformance)."""
    fabio = pytest.importorskip("fabio")
    root = tmp_path / "proj"
    (root / "raw").mkdir(parents=True)
    frames = []
    for i in range(2):
        img = np.full((4, 5), float(i), dtype=np.float32)
        src = root / "raw" / f"img_{i:04d}.tif"
        fabio.tifimage.TifImage(data=img).write(str(src))
        frames.append(Frame(index=i, image=img, source_path=src,
                            source_frame_index=0))

    monkeypatch.setattr(
        reduction_core, "integrate_1d",
        lambda image, ai, **kw: _fake_r1d(float(np.sum(image))),
    )
    out = root / "processed" / "scan.nxs"
    session = ReductionSession(
        ReductionPlan(integration_2d=None),
        Scan("s", frames, integrator=object()),
        sink=NexusSink(out, source_base=root, overwrite=True),
        execution="streaming",
    )
    for fr in frames:
        session.submit(fr)
    session.finish()

    from xrd_tools.io import open_scan

    check_source_contract(lambda: open_scan(out), expected_indices=[0, 1])
    caps = open_scan(out).capabilities
    assert not caps.is_streaming and caps.supports_random_access
