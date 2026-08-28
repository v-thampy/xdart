"""P3-3B exact resource equations and retained-root bounds."""
from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np

from xrd_tools.session.policy import (
    SessionResourceRequirements,
    requirements_from,
    resolve_session_policy,
)
from xrd_tools.reduction import FrameBackgroundPlan


def _requirements(**changes):
    values = dict(height=10, width=20, native_itemsize=2, modes_1d=1,
                  npt_1d=100, background_bytes=1600,
                  resolver_background_bytes=5000,
                  worker_background_bytes=1600,
                  background_binding_bytes=64 * 1024 * 1024)
    values.update(changes)
    return SessionResourceRequirements(**values)


def test_background_resource_equations_and_mode_root_census() -> None:
    from xdart.gui.tabs.scattering.output_preflight import _background_resource_terms
    req = _requirements()
    allocation = resolve_session_policy(req, envelope_bytes=8 * 1024 ** 3,
        requests={"workers": 2, "reduction_inflight": 4}).allocation
    c = allocation.counts; p = req.native_frame_bytes; g = req.background_bytes
    a1, a2, t = req.result_1d_bytes, req.result_2d_bytes, req.thumbnail_bytes
    assert allocation.categories["source_native"] == c["owner_block_bytes"] + (c["queue_depth"] + 1 + c["reduction_inflight"]) * p + (c["reduction_inflight"] + 1) * g
    assert allocation.categories["staging"] == c["staging_items"] * (p + a1 + a2 + t)
    assert allocation.categories["publication"] == c["publication_items"] * a1 + c["publication_heavy_items"] * (p + a2) + c["thumbnail_items"] * t
    assert req.background_binding_bytes == 64 * 1024 * 1024
    none = _requirements(background_bytes=0, resolver_background_bytes=0,
                         worker_background_bytes=0, background_binding_bytes=0)
    assert all(value == 0 for value in (none.background_bytes,
        none.resolver_background_bytes, none.worker_background_bytes,
        none.background_binding_bytes))
    assert _background_resource_terms(FrameBackgroundPlan(), 200) == (0, 0, 0, 0)
    for mode, expected_r in (("Single BG File", 1600), ("BG Directory", 1600),
                             ("Series Average", 5000)):
        kwargs = ({"locator": "/bg"} if mode == "BG Directory" else
                  {"locator": "/bg_1.tif"})
        if mode == "BG Directory": kwargs["match_rule"] = "Scan Root + Frame Number"
        assert _background_resource_terms(FrameBackgroundPlan(mode=mode, **kwargs), 200) == (
            1600, expected_r, 1600, 64 * 1024 ** 2)
    legacy = _requirements(background_bytes=0, resolver_background_bytes=0,
        worker_background_bytes=0, background_binding_bytes=0)
    assert resolve_session_policy(legacy, envelope_bytes=8 * 1024 ** 3).allocation.categories == \
        resolve_session_policy(none, envelope_bytes=8 * 1024 ** 3).allocation.categories
    source = (Path(__file__).parents[2] / "src/xrd_tools/reduction/background.py").read_text()
    assert "sums[finite]" not in source and "np.add(sums, image, out=sums, where=finite)" in source


def test_submitter_plus_inflight_background_roots_without_serialization(monkeypatch) -> None:
    req = _requirements()
    allocation = resolve_session_policy(req, envelope_bytes=8 * 1024 ** 3,
        requests={"workers": 2, "reduction_inflight": 4}).allocation
    assert allocation.reduction_inflight == 4 and allocation.workers == 2
    native_without_background = (allocation.owner_block_bytes +
        (allocation.queue_depth + 1 + allocation.reduction_inflight) * req.native_frame_bytes)
    assert allocation.categories["source_native"] - native_without_background == (
        (allocation.reduction_inflight + 1) * req.background_bytes)
    roots = [np.frombuffer(np.ones(200, np.float64).tobytes(), dtype=np.float64)
             for _ in range(allocation.reduction_inflight + 1)]
    assert len({id(value.base) for value in roots}) == allocation.reduction_inflight + 1
    from xdart.gui.tabs.scattering.adapters import run_executor as module
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.reduction import FrameBackgroundResult
    second_entered = Event(); first_submitted = Event(); calls = []
    class Source:
        frame_indices = (1, 2); allocation = SimpleNamespace(queue_depth=1)
        def iter_chunks(self, size):
            yield np.ones((1, 2, 2), np.uint16), (1,)
            second_entered.set(); yield np.ones((1, 2, 2), np.uint16), (2,)
        def take_direct_chunk_fact(self): return None
    monkeypatch.setattr(module, "NexusStackSource", Source)
    raw = b'{"version":1}'; fingerprint = __import__("hashlib").sha256(raw).hexdigest()
    binding = lambda label: (label, (label, f"/raw/{label}.tif", None, 0,
        (2, 2), ()), raw, fingerprint)
    def resolve(*args, **kwargs):
        value = np.frombuffer(np.ones(4, np.float64).tobytes(), dtype=np.float64).reshape(2, 2)
        value.setflags(write=False); return FrameBackgroundResult("RESOLVED", value, raw, fingerprint)
    monkeypatch.setattr(module, "resolve_frame_background", resolve)
    class Output:
        def background_binding(self, label): return binding(label)
        def submit(self, frame, image):
            if not calls: assert second_entered.wait(2); first_submitted.set()
            calls.append(frame.index); return True
        def stop(self): return None
    frames = {label: ScanFrame(label, source_path=f"/raw/{label}.tif") for label in (1, 2)}
    run = SimpleNamespace(source=Source(), frames_by_label=frames,
        configuration=SimpleNamespace(background=FrameBackgroundPlan(
            mode="Single BG File", locator="/bg.tif"), live_mode=False),
        stop_requested=False, stop_signal=Event(), context_runtime=None,
        resource_facts=[], cleanup_failures=[], perf_enabled=False)
    module.StandardRunExecutor._submit_container_source(run, Output())
    assert second_entered.is_set() and first_submitted.is_set() and calls == [1, 2]


def test_worker_and_resolver_terms_are_not_double_charged() -> None:
    req = _requirements()
    allocation = resolve_session_policy(req, envelope_bytes=8 * 1024 ** 3,
        requests={"workers": 3, "reduction_inflight": 5}).allocation
    expected = (allocation.workers * (1024 ** 3 + 12 * req.pixels + req.worker_background_bytes)
                + allocation.reduction_inflight * (req.result_1d_bytes + req.result_2d_bytes)
                + req.resolver_background_bytes + req.background_binding_bytes)
    assert allocation.categories["worker"] == expected
    descriptor = SimpleNamespace(frame_shape=(10, 20), dtype=np.dtype("uint16"))
    plan = SimpleNamespace(integration_1d=SimpleNamespace(npt=100, error_model=None),
                           integration_2d=None, gi=None)
    rebuilt = requirements_from(descriptor, plan, background_bytes=1600,
        resolver_background_bytes=5000, worker_background_bytes=1600,
        background_binding_bytes=64 * 1024 * 1024)
    assert rebuilt.fingerprint == req.fingerprint
    validated = resolve_session_policy(rebuilt, allocation=allocation,
                                       envelope_bytes=allocation.envelope_bytes)
    assert validated.allocation is allocation
    assert _requirements(resolver_background_bytes=5001).fingerprint != req.fingerprint
    assert _requirements(background_binding_bytes=64 * 1024 * 1024 - 1).fingerprint != req.fingerprint

    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction.core import Integration1DPlan, ReductionPlan
    from xrd_tools.session import open_headless_scan_session
    plan_exact = ReductionPlan(integration_1d=Integration1DPlan(npt=100))
    scan = Scan("headless", [ScanFrame(0, image=np.ones((10, 20), np.uint16))])
    captured = []
    import xrd_tools.session.headless_scan as headless_module
    original = headless_module.ScanSession
    headless_module.ScanSession = lambda *args, **kwargs: captured.append(kwargs) or args
    try:
        from xrd_tools.session.policy import SessionPolicy
        modes = ((FrameBackgroundPlan(), (0, 0, 0, 0)),
            (FrameBackgroundPlan(mode="Single BG File", locator="/bg.tif"),
             (1600, 1600, 1600, 64 * 1024 ** 2)),
            (FrameBackgroundPlan(mode="Series Average", locator="/bg_1.tif"),
             (1600, 5000, 1600, 64 * 1024 ** 2)),
            (FrameBackgroundPlan(mode="BG Directory", locator="/bg",
                 match_rule="Scan Root + Frame Number"),
             (1600, 1600, 1600, 64 * 1024 ** 2)))
        for background, terms in modes:
            exact = _requirements(background_bytes=terms[0],
                resolver_background_bytes=terms[1], worker_background_bytes=terms[2],
                background_binding_bytes=terms[3])
            policy = resolve_session_policy(exact, envelope_bytes=8 * 1024 ** 3)
            open_headless_scan_session(scan, plan_exact, policy=policy,
                                       background_plan=background)
            assert captured[-1]["policy"].allocation is policy.allocation
            for field in ("background_bytes", "resolver_background_bytes",
                          "worker_background_bytes", "background_binding_bytes"):
                altered = replace(exact, **{field: getattr(exact, field) + 1})
                with __import__("pytest").raises(ValueError):
                    open_headless_scan_session(scan, plan_exact,
                        policy=SessionPolicy(policy.flush,
                            replace(policy.allocation, requirements=altered)),
                        background_plan=background)
            with __import__("pytest").raises(ValueError):
                open_headless_scan_session(scan, plan_exact,
                    policy=SessionPolicy(policy.flush, replace(policy.allocation,
                        requirements=replace(exact, native_itemsize=4))),
                    background_plan=background)
    finally:
        headless_module.ScanSession = original
