# -*- coding: utf-8 -*-
"""R2-R1 — the public ``open_source(...) -> run_reduction(...)`` durable path
consumes ONE sustained source cursor (no per-frame reopen) in standard AND GI
modes, on the real streaming engine (not the benchmark's direct-cursor path).

Fail-before: the streaming producer submitted frames with no image, so the
worker called ``frame.load_image() -> source.load_frame()`` per frame — a fresh
open per frame (5 frames -> 7 source-path opens, scaling with frame count).
Pass-after: the producer feeds decoded native arrays from the source's
``iter_chunks`` cursor via ``submit(frame, image)``; opens are bounded and do
NOT scale with the frame count, and no h5py handle crosses into a worker.
"""

from __future__ import annotations

import inspect
import os

import h5py
import numpy as np
import pytest

import xrd_tools.reduction.core as reduction_core
from xrd_tools.core.containers import IntegrationResult1D, PONI
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.reduction import (
    GIMode,
    Integration1DPlan,
    NexusSink,
    ReductionPlan,
    run_reduction,
)
from xrd_tools.sources.registry import open_source


def _fake_1d(image, ai, **kwargs):
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([float(np.sum(image)), 1.0]),
        sigma=np.array([0.1, 0.2]),
        unit="q_A^-1",
    )


def _fake_gi_1d(image, fi, **kwargs):
    return IntegrationResult1D(
        radial=np.array([0.0, 1.0]),
        intensity=np.array([float(np.sum(image)), 1.0]),
        sigma=np.array([0.1, 0.2]),
        unit="qoop_A^-1",
    )


def _write_stack(
        path, n, *, chunks=(2, 8, 8), dtype=np.uint16,
        threshold_probe=False):
    data = np.ones((n, 8, 8), dtype=dtype)
    if threshold_probe:
        data[:, 0, 0] = 3
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        e.create_group("instrument/detector").create_dataset(
            "data", data=data, chunks=chunks)
    return path


def _write_bluesky_gi(path, n):
    inc = np.linspace(0.1, 0.5, n)
    i0 = np.full(n, 10.0)
    with h5py.File(path, "w") as f:
        f.attrs["creator"] = "NXWriter"
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        inst = e.create_group("instrument")
        inst.attrs["NX_class"] = "NXinstrument"
        bl = inst.create_group("bluesky")
        bl.attrs["NX_class"] = "NXnote"
        bl.create_group("metadata").create_dataset(
            "motors", data=b"!!python/tuple\n- dummy\n")
        data = e.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "eiger_image"
        # a recognized counter column carries the per-frame incidence angle
        data.create_dataset("i1", data=inc)
        data.create_dataset("i0", data=i0)
        img = data.create_dataset(
            "eiger_image", data=np.ones((n, 8, 8), np.uint32), chunks=(2, 8, 8))
        img.attrs["signal_type"] = "detector"
        e.create_dataset("end_time", data=np.bytes_("2026-07-17"))
    return path


class _OpenCounter:
    """Counts h5py.File constructions on ONE watched source path (the write of
    the fixture happens before the watch is armed, so it is never counted)."""

    def __init__(self, monkeypatch):
        self.path = None
        self.opens = []
        counter = self
        orig = h5py.File.__init__

        def counting(inst, name, *a, **k):  # plain function -> correct self bind
            try:
                if counter.path and os.path.abspath(str(name)) == counter.path:
                    counter.opens.append(1)
            except Exception:
                pass
            return orig(inst, name, *a, **k)

        monkeypatch.setattr(h5py.File, "__init__", counting)

    def watch(self, path):
        self.path = os.path.abspath(str(path))
        self.opens.clear()


def test_public_route_standard_has_no_per_frame_reopen(tmp_path, monkeypatch):
    seen_dtypes = []

    def spy_1d(image, ai, **kwargs):
        seen_dtypes.append(np.asarray(image).dtype)
        return _fake_1d(image, ai, **kwargs)

    monkeypatch.setattr(reduction_core, "integrate_1d", spy_1d)
    counter = _OpenCounter(monkeypatch)

    def run(n):
        p = _write_stack(tmp_path / f"lab_{n:02d}_00001.nxs", n)
        counter.watch(p)
        seen_dtypes.clear()
        src = open_source(SourceSpec(p, SourceKind.NEXUS_STACK))
        src.integrator = object()
        result = run_reduction(
            ReductionPlan(integration_1d=Integration1DPlan(npt=2)),
            src,
            NexusSink(path=str(tmp_path / f"out_{n:02d}.nexus"), overwrite=True))
        return len(counter.opens), result

    o5, r5 = run(5)
    # every frame's decoded image reached a worker (integrate saw all 5)
    assert len(seen_dtypes) == 5
    o10, r10 = run(10)

    assert r5.n_processed == 5 and r10.n_processed == 10
    assert len(seen_dtypes) == 10  # all 10 frames of the second run reached a worker
    # opens are BOUNDED and do not scale with frame count -> no per-frame reopen
    assert o10 == o5, f"opens scaled with frames ({o5} -> {o10}): per-frame reopen"
    assert o5 <= 4, f"expected a small bounded open count, got {o5}"


def test_public_route_gi_has_no_per_frame_reopen(tmp_path, monkeypatch):
    monkeypatch.setattr(reduction_core, "poni_to_fiber_integrator",
                        lambda poni, **k: object())
    monkeypatch.setattr(reduction_core, "integrate_gi_1d", _fake_gi_1d)
    counter = _OpenCounter(monkeypatch)

    def run(n):
        p = _write_bluesky_gi(tmp_path / f"gi_{n:02d}_00001.nxs", n)
        counter.watch(p)
        src = open_source(SourceSpec(p, SourceKind.NEXUS_STACK))
        src.integrator = object()
        src.poni = PONI(dist=0.2, poni1=0.1, poni2=0.1, wavelength=1e-10)
        plan = ReductionPlan(
            integration_1d=Integration1DPlan(npt=2, monitor_key="i0"),
            gi=GIMode(incidence_motor="i1", mode_1d="q_oop", npt_oop=3))
        result = run_reduction(
            plan, src,
            NexusSink(path=str(tmp_path / f"giout_{n:02d}.nexus"), overwrite=True),
            gi_freeze_mode="scout_union")
        return len(counter.opens), result

    o5, r5 = run(5)
    o10, r10 = run(10)

    assert r5.n_processed == 5 and r10.n_processed == 10
    # opens bounded + do not scale (the GI freeze scout adds a fixed 2 reads,
    # the streaming read still uses one sustained iter_chunks cursor)
    assert o10 == o5, f"GI opens scaled with frames ({o5} -> {o10}): per-frame reopen"
    assert o5 <= 6, f"expected a small bounded GI open count, got {o5}"


def test_public_route_no_open_handle_leaks_after_reduction(tmp_path, monkeypatch):
    """After the durable public reduction, the source's sustained cursor is
    closed — no lingering open h5py handle on the source file."""
    monkeypatch.setattr(reduction_core, "integrate_1d", _fake_1d)
    p = _write_stack(tmp_path / "leak_00001.nxs", 4)
    src = open_source(SourceSpec(p, SourceKind.NEXUS_STACK))
    src.integrator = object()
    run_reduction(
        ReductionPlan(integration_1d=Integration1DPlan(npt=2)),
        src,
        NexusSink(path=str(tmp_path / "leakout.nexus"), overwrite=True))
    # a fresh writable open proves no reader handle is still holding the file
    with h5py.File(p, "r+") as f:
        assert "entry" in f


def test_preopened_cursor_streams_public_source_and_closes_once(tmp_path, monkeypatch):
    """A same-thread pre-opened cursor serves source metadata and all reads."""
    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.sources.nexus import NexusStackSource

    monkeypatch.setattr(reduction_core, "integrate_1d", _fake_1d)
    p = _write_stack(tmp_path / "prepared_00001.nxs", 4)
    cursor = ContainerCursor(p).open()
    source = NexusStackSource(p, cursor=cursor)
    source.integrator = object()
    result = run_reduction(
        ReductionPlan(integration_1d=Integration1DPlan(npt=2)), source,
        NexusSink(path=str(tmp_path / "prepared_out.nexus"), overwrite=True))

    assert result.n_processed == 4
    assert cursor.closed is True


def test_huge_consumer_chunk_is_capped_by_source_read_plan(tmp_path, monkeypatch):
    """A caller's progress chunk cannot bypass the source byte budget."""
    from xrd_tools.core import staging
    from xrd_tools.sources.cursor import ContainerCursor

    p = _write_stack(tmp_path / "bounded_00001.nxs", 7, chunks=(4, 8, 8))
    frame_bytes = 8 * 8 * np.dtype(np.uint16).itemsize
    monkeypatch.setattr(staging, "source_block_budget_bytes", lambda: 2 * frame_bytes)
    calls = []
    real_read_block = ContainerCursor.read_block

    def spy_read_block(self, start, stop):
        block = real_read_block(self, start, stop)
        calls.append((int(start), int(stop), block.nbytes, block.array.dtype))
        return block

    monkeypatch.setattr(ContainerCursor, "read_block", spy_read_block)
    src = open_source(SourceSpec(p, SourceKind.NEXUS_STACK))
    chunks = list(src.iter_chunks(10_000))

    assert [label for _images, labels in chunks for label in labels] == list(range(7))
    assert [(start, stop) for start, stop, _nbytes, _dtype in calls] == [
        (0, 2), (2, 4), (4, 6), (6, 7)]
    assert all(nbytes <= 2 * frame_bytes for _s, _e, nbytes, _d in calls)
    assert all(dtype == np.dtype(np.uint16) for _s, _e, _n, dtype in calls)


@pytest.mark.parametrize("source_dtype", (np.uint16, np.float64))
def test_public_submit_seam_receives_native_dtype_under_huge_chunk(
        tmp_path, monkeypatch, source_dtype):
    """Native source arrays reach ``ReductionSession.submit`` before workers.

    This guards the real producer-to-session boundary; an upcast inside
    ``_chunk_images_as_list`` would fail here even if the integrator later hid
    it by converting to float internally.
    """
    from xrd_tools.core import staging
    from xrd_tools.sources.cursor import ContainerCursor
    from xrd_tools.reduction.core import ReductionSession

    p = _write_stack(
        tmp_path / "submit_00001.nxs",
        5,
        chunks=(4, 8, 8),
        dtype=source_dtype,
        threshold_probe=True,
    )
    frame_bytes = 8 * 8 * np.dtype(source_dtype).itemsize
    monkeypatch.setattr(staging, "source_block_budget_bytes", lambda: 2 * frame_bytes)
    monkeypatch.setattr(reduction_core, "integrate_1d", _fake_1d)
    seen = []
    owner_blocks = []
    owner_snapshots = []
    real_submit = ReductionSession.submit
    real_read_block = ContainerCursor.read_block

    def spy_read_block(self, start, stop):
        block = real_read_block(self, start, stop)
        owner_blocks.append(block.array)
        owner_snapshots.append(block.array.copy())
        return block

    def spy_submit(self, frame, image=None):
        assert image is not None
        # The producer sends a view of the source owner block, not a hidden
        # float/copy allocation; copies are therefore separately zero here.
        assert any(np.shares_memory(np.asarray(image), owner)
                   for owner in owner_blocks)
        seen.append((int(frame.index), np.asarray(image).dtype))
        return real_submit(self, frame, image)

    monkeypatch.setattr(ContainerCursor, "read_block", spy_read_block)
    monkeypatch.setattr(ReductionSession, "submit", spy_submit)
    src = open_source(SourceSpec(p, SourceKind.NEXUS_STACK))
    src.integrator = object()
    result = run_reduction(
        ReductionPlan(
            integration_1d=Integration1DPlan(npt=2),
            threshold_max=2.0,
        ),
        src,
        NexusSink(path=str(tmp_path / "submit_out.nexus"), overwrite=True),
        chunk_size=10_000, inflight_max=1, executor=1)

    assert result.n_processed == 5
    assert [index for index, _dtype in seen] == list(range(5))
    assert {dtype for _index, dtype in seen} == {np.dtype(source_dtype)}
    for owner, before in zip(owner_blocks, owner_snapshots):
        np.testing.assert_array_equal(owner, before)


# ── H10-C2-B: the coordinated route binds one allocation before reading ──────

class _SpySink:
    """Records the order of the fallible boundaries C2-B must run behind."""

    def __init__(self, order):
        self.order = order

    def begin(self, scan, plan):
        self.order.append("sink.begin")

    def write(self, frame, reduction):
        self.order.append("sink.write")

    def finish(self, result):
        self.order.append("sink.finish")


def _envelope_plan():
    return ReductionPlan(integration_1d=Integration1DPlan(npt=4))


def test_c2b_coordinated_scan_session_binds_the_registry_source_before_effects(
        tmp_path, monkeypatch):
    """``ScanSession(plan, open_source(...))`` is the coordinated public path:
    it derives descriptor-backed requirements without reading pixels and binds
    the exact allocation before the sink hook or any source read."""
    from xrd_tools.session import ScanSession

    monkeypatch.setattr(reduction_core, "integrate_1d_frame", _fake_1d,
                        raising=False)
    path = _write_stack(tmp_path / "m.h5", 4)
    source = open_source(path)
    source.integrator = object()
    order = []

    real_bind = source.bind_allocation
    real_iter = source.iter_chunks

    def spy_bind(allocation):
        order.append("bind_allocation")
        return real_bind(allocation)

    def spy_iter(chunk_size):
        order.append("iter_chunks")
        return real_iter(chunk_size)

    source.bind_allocation = spy_bind
    source.iter_chunks = spy_iter

    session = ScanSession(_envelope_plan(), source, sink=_SpySink(order),
                          envelope_bytes=64 * 1024 ** 3, executor=1)
    try:
        assert order[0] == "bind_allocation", order
        assert "iter_chunks" not in order[:1]
        assert source.allocation is not None
        # the descriptor's real detector facts, never a fallback
        assert source.allocation.requirements.height == 8
        assert source.allocation.requirements.width == 8
        assert source.allocation.requirements.native_itemsize == 2
        # ... and the session's ONE policy carries that same object
        assert session.policy.allocation is source.allocation
    finally:
        session.finish(raise_on_failure=False)


def test_c2b_nexus_source_never_self_resolves_an_allocation():
    from xrd_tools.sources.nexus import NexusStackSource

    assert not hasattr(NexusStackSource, "resolve_session_policy")
    source_text = inspect.getsource(NexusStackSource)
    assert "resolve_session_policy" not in source_text


def test_c2b_a_second_conflicting_bind_is_rejected(tmp_path):
    from xrd_tools.session.policy import (
        SessionResourceRequirements,
        resolve_session_policy,
    )

    path = _write_stack(tmp_path / "m.h5", 3)
    source = open_source(path)
    req = SessionResourceRequirements(height=8, width=8, native_itemsize=2,
                                      modes_1d=1, npt_1d=4)
    first = resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                   env={}).allocation
    source.bind_allocation(first)
    source.bind_allocation(first)                    # SAME object: idempotent
    # rebinding is IDENTITY-qualified: an equal-but-distinct object rejects,
    # because the contract is one shared allocation object, not one value.
    twin = resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                  env={}).allocation
    assert twin == first and twin is not first
    with pytest.raises(ValueError):
        source.bind_allocation(twin)
    other = resolve_session_policy(req, envelope_bytes=64 * 1024 ** 3,
                                   requests={"queue_depth": 2},
                                   env={}).allocation
    with pytest.raises(ValueError):
        source.bind_allocation(other)
    assert source.allocation is first


def test_c2b_bound_source_reads_only_its_granted_owner_block(tmp_path):
    from xrd_tools.session.policy import (
        SessionResourceRequirements,
        resolve_session_policy,
    )

    path = _write_stack(tmp_path / "m.h5", 8, chunks=(2, 8, 8))
    source = open_source(path)
    req = SessionResourceRequirements(height=8, width=8, native_itemsize=2,
                                      modes_1d=1, npt_1d=4)
    frame_bytes = 8 * 8 * 2
    alloc = resolve_session_policy(
        req, envelope_bytes=64 * 1024 ** 3,
        requests={"owner_block_bytes": 2 * frame_bytes}, env={}).allocation
    source.bind_allocation(alloc)
    for block, labels in source.iter_chunks(64):
        assert block.nbytes <= alloc.owner_block_bytes
        assert len(labels) == block.shape[0]


def test_c2b_legacy_uncoordinated_run_reduction_route_needs_no_allocation(
        tmp_path, monkeypatch):
    """``run_reduction(open_source(...))`` stays the named legacy uncoordinated
    entry: unbound, it uses the compatibility source-block budget and claims no
    SessionPolicy."""
    monkeypatch.setattr(reduction_core, "integrate_1d_frame", _fake_1d,
                        raising=False)
    path = _write_stack(tmp_path / "m.h5", 4)
    source = open_source(path)
    assert source.allocation is None
    blocks = [len(labels) for _block, labels in source.iter_chunks(2)]
    assert sum(blocks) == 4
    assert source.allocation is None, "the legacy route must not manufacture one"
