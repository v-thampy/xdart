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

import os

import h5py
import numpy as np

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


def _write_stack(path, n, *, chunks=(2, 8, 8)):
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        e.create_group("instrument/detector").create_dataset(
            "data", data=np.ones((n, 8, 8), np.uint16), chunks=chunks)
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
            NexusSink(path=str(tmp_path / f"out_{n:02d}.nxs"), overwrite=True))
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
            NexusSink(path=str(tmp_path / f"giout_{n:02d}.nxs"), overwrite=True),
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
        NexusSink(path=str(tmp_path / "leakout.nxs"), overwrite=True))
    # a fresh writable open proves no reader handle is still holding the file
    with h5py.File(p, "r+") as f:
        assert "entry" in f
