"""P3-3B allocation-first binding and sole insertion composition."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np

from xrd_tools.core.scan import ScanFrame
from xrd_tools.reduction import FrameBackgroundPlan, FrameBackgroundResult


def _binding(label=1):
    raw = b'{"result_sha256":"' + b"0" * 64 + b'","version":1}'
    return (label, (label, "/raw/target.tif", None, 0, (2, 2), ()), raw,
            hashlib.sha256(raw).hexdigest())


def _function_order(path: Path, name: str, needles: tuple[str, ...]) -> tuple[int, ...]:
    tree = ast.parse(path.read_text())
    node = next(item for item in ast.walk(tree)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)
    source = ast.get_source_segment(path.read_text(), node) or ""
    return tuple(source.index(needle) for needle in needles)


def _composition(tmp_path: Path, *, live=False, count=3,
                 output_mode="Overwrite", processing_mode="Int 1D (XYE)"):
    from xdart.gui.tabs.scattering.contracts import (
        AdmittedOutput, OutputDisposition, OutputFact, PlannedOutput,
        SourceExecutionStamp, SourceFileState)
    from xrd_tools.core.scan import Scan
    from xrd_tools.reduction.core import Integration1DPlan, ReductionPlan
    from xrd_tools.session.run_configuration import RunIntent
    from xrd_tools.sources.selection import image_series_spec
    paths = tuple(tmp_path / f"scan_{index}.tif" for index in range(1, count + 1))
    for path in paths: path.write_bytes(b"raw")
    spec = image_series_spec(paths[0]); states = tuple(SourceFileState.capture(path) for path in paths)
    stamp = SourceExecutionStamp(states[0], "tiff_series", count, 1, members=states)
    item = PlannedOutput(spec, paths[0], tmp_path / "out.nxs", stamp)
    decision = AdmittedOutput(item, OutputDisposition.WRITE,
        tuple(range(1, count + 1)), OutputFact(False))
    frames = [ScanFrame(index, image=np.ones((2, 2), np.uint16),
        source_path=path, source_frame_index=0) for index, path in enumerate(paths, 1)]
    configuration = RunIntent(source_spec=spec, processing_mode=processing_mode,
        output_mode=output_mode, live_mode=live, poni_file="/accepted.poni",
        save_path=str(item.target), background=FrameBackgroundPlan(
            mode="Single BG File", locator=str(tmp_path / "background.tif"))).freeze()
    return configuration, Scan("scan", frames), ReductionPlan(
        integration_1d=Integration1DPlan(npt=8)), item, decision


def _provenance(configuration):
    science = configuration.as_provenance(); science["accepted_scientific_assets"] = {
        "poni_values": None, "poni_detector_config_json": None,
        "poni_sha256": None, "mask_sha256": None}
    return {"scientific_signature": science}


def _fake_graph_io(monkeypatch, module):
    captured = SimpleNamespace(submits=[], policy=None)
    class XYE:
        canonical_target = "xye:test"
        def __init__(self, *args, **kwargs): pass
        def abort(self, result): return None
    class Session:
        is_running = True
        def submit(self, frame, image=None, **kwargs): assert frame.background is not None and frame.background_dependency_bytes is not None and frame.background_dependency_fingerprint is not None; captured.submits.append(frame.index); return True
        def stop(self): self.is_running = False
        def set_generation(self, value): captured.generation = value
        def on_frame_completed(self, callback): captured.completed = callback
        def on_checkpoint_recoverable(self, callback): captured.checkpoint = callback
    session = Session()
    monkeypatch.setattr(module, "TransactionalXYESink", XYE)
    def open_session(*args, **kwargs): captured.policy = kwargs["policy"]; return session
    monkeypatch.setattr(module, "open_headless_scan_session", open_session)
    return captured, session


def test_batch_prequalifies_array_free_map_before_effects(monkeypatch, tmp_path: Path) -> None:
    from xdart.gui.tabs.scattering.contracts import AdmittedOutput
    from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor
    configuration, scan, plan, item, decision = _composition(tmp_path)
    calls = []
    def resolve(bg_plan, fact, *, cancelled=None):
        calls.append((fact[0], cancelled)); raw = _binding(fact[0])[2]
        value = np.frombuffer(np.ones((2, 2), np.float64).tobytes(), dtype=np.float64).reshape(2, 2); value.setflags(write=False)
        return FrameBackgroundResult("RESOLVED", value, raw, hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(run_executor, "resolve_frame_background", resolve)
    adapter = dynamic_output.DynamicOutputAdapter(configuration); stop = Event(); allocations = []
    preparation = adapter.prepare_admission(scan, plan, item, decision, stop,
        qualify=lambda policy, prior: (allocations.append(policy.allocation) or run_executor._qualify_background_bindings(configuration, scan, item, decision, policy, prior, stop)))
    assert type(preparation.effective) is AdmittedOutput
    assert tuple(value[0] for value in preparation.effective.background_bindings) == (1, 2, 3)
    assert len(allocations) == 1 and preparation.resources[0].allocation is allocations[0]
    assert all(frame.background is frame.background_dependency_bytes is None for frame in scan.frames)
    assert [label for label, signal in calls] == [1, 2, 3] and all(signal is stop for _, signal in calls)
    assert not item.target.exists() and adapter.session is None
    assert "background_bindings" in AdmittedOutput.__dataclass_fields__
    dynamic = Path(__file__).parents[3] / "src/xdart/gui/tabs/scattering/adapters/dynamic_output.py"
    source = dynamic.read_text()
    assert "def prepare_admission(" in source and "class _PreparedAdmission" in source
    executor = Path(__file__).parents[3] / "src/xdart/gui/tabs/scattering/adapters/run_executor.py"
    order = _function_order(executor, "_construct", (
        "prepare_admission(", "target_state_matches(", "add_artifact(", "activate("))
    assert order == tuple(sorted(order))
    route_source = executor.read_text()
    assert route_source.count("run.display.configure(") == 1
    assert route_source.index("prepare_admission(") < route_source.index("run.display.configure(")


def test_exact_append_noop_defers_policy_to_locked_skip(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.io import (
        AppendDisposition,
        AppendPreflightState,
    )

    configuration, scan, plan, item, decision = _composition(
        tmp_path,
        count=1,
        output_mode="Append",
        processing_mode="Int 1D",
    )
    configuration = replace(
        configuration,
        background=FrameBackgroundPlan(),
    )
    decision = replace(decision, labels=())
    calls = SimpleNamespace(layout=0, qualify=0, sink=0, session=0)

    def unexpected_layout(*args, **kwargs):
        calls.layout += 1
        raise AssertionError("exact Append no-op sized source resources")

    def unexpected_qualify(*args, **kwargs):
        calls.qualify += 1
        raise AssertionError("exact Append no-op qualified Background")

    monkeypatch.setattr(
        dynamic_output,
        "_light_policy_layout",
        unexpected_layout,
    )

    class Preflight:
        def __init__(self, disposition):
            self.disposition = disposition
            self.state = AppendPreflightState.RESERVED

        @property
        def snapshot(self):
            return SimpleNamespace(
                disposition=self.disposition,
                skip_labels=(1,) if self.disposition is AppendDisposition.SKIP else (),
                write_labels=() if self.disposition is AppendDisposition.SKIP else (1,),
                state=self.state,
            )

        def complete_noop(self):
            assert self.disposition is AppendDisposition.SKIP
            self.state = AppendPreflightState.NOOP
            return self.snapshot

        def abort(self):
            self.state = AppendPreflightState.ABORTED
            return self.snapshot

        def retry_cleanup(self):
            return self.snapshot

    dispositions = iter((AppendDisposition.SKIP, AppendDisposition.WRITE))
    preflights = []

    def preflight(*args, **kwargs):
        value = Preflight(next(dispositions))
        preflights.append(value)
        return value

    monkeypatch.setattr(dynamic_output, "prepare_append_preflight", preflight)
    monkeypatch.setattr(
        dynamic_output,
        "NexusSink",
        lambda *args, **kwargs: (
            setattr(calls, "sink", calls.sink + 1),
            (_ for _ in ()).throw(AssertionError("no-op opened a Nexus sink")),
        )[1],
    )
    monkeypatch.setattr(
        dynamic_output,
        "open_headless_scan_session",
        lambda *args, **kwargs: (
            setattr(calls, "session", calls.session + 1),
            (_ for _ in ()).throw(AssertionError("no-op opened a session")),
        )[1],
    )

    adapter = dynamic_output.DynamicOutputAdapter(configuration)
    stop = Event()
    preparation = adapter.prepare_admission(
        scan,
        plan,
        item,
        decision,
        stop,
        qualify=unexpected_qualify,
    )
    assert preparation.dormant_noop and preparation.resources is None
    assert preparation.effective.background_bindings == ()
    assert calls.layout == calls.qualify == 0
    active, created = adapter.activate(
        preparation,
        record_store=object(),
        run_provenance=_provenance(configuration),
    )
    assert active is None and not created
    assert adapter.persisted_prefix_labels == (1,)
    assert preflights[0].state is AppendPreflightState.NOOP
    with __import__("pytest").raises(RuntimeError, match="active graph"):
        adapter.background_admission_context()
    with __import__("pytest").raises(RuntimeError, match="active graph"):
        adapter.admit_background_binding(_binding())

    second = adapter.prepare_admission(
        scan,
        plan,
        item,
        decision,
        stop,
        qualify=unexpected_qualify,
    )
    assert second.dormant_noop and second.resources is None
    with __import__("pytest").raises(RuntimeError, match="locked preflight"):
        adapter.activate(
            second,
            record_store=object(),
            run_provenance=_provenance(configuration),
        )
    assert preflights[1].state is AppendPreflightState.ABORTED
    assert adapter._pending_preflights == []
    assert calls.layout == calls.qualify == calls.sink == calls.session == 0


def test_all_routes_use_one_pre_submit_resolver_outside_command_lock(monkeypatch, tmp_path: Path) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor as module
    event = Event(); seen = []
    def resolve(plan, fact, *, cancelled=None):
        seen.append(cancelled)
        value = np.frombuffer(np.ones((2, 2), dtype=np.float64).tobytes(),
                              dtype=np.float64).reshape(2, 2)
        value.setflags(write=False)
        binding = _binding()
        return FrameBackgroundResult("RESOLVED", value, binding[2], binding[3])
    monkeypatch.setattr(module, "resolve_frame_background", resolve)
    frame = ScanFrame(1, image=np.ones((2, 2)), source_path="/raw/target.tif")
    assert module._resolve_background_before_submit(
        frame, FrameBackgroundPlan(mode="Single BG File", locator="/bg.tif"),
        _binding(), cancelled=event)
    assert seen == [event] and frame.background is not None
    assert frame.background_dependency_bytes == _binding()[2]
    assert frame.background_dependency_fingerprint == _binding()[3]
    configuration, scan, plan, item, decision = _composition(tmp_path, live=True, count=1)
    captured, session = _fake_graph_io(monkeypatch, dynamic_output); allocations = []; display_allocations = []
    real_layout = dynamic_output._light_policy_layout
    def no_light(*args, **kwargs):
        result = real_layout(*args, **kwargs); return result[0], None, 0, 0, result[4]
    monkeypatch.setattr(dynamic_output, "_light_policy_layout", no_light)
    adapter = dynamic_output.DynamicOutputAdapter(configuration); stop = Event()
    preparation = adapter.prepare_admission(scan, plan, item, decision, stop,
        qualify=lambda policy, prior: prior)
    class Display:
        def bind_heavy_allocation(self, allocation): display_allocations.append(allocation); return 0
        def bind_light_subscription(self, *values): captured.subscription = values
    source_owner = SimpleNamespace(bind_allocation=lambda allocation: allocations.append(allocation))
    callback = lambda *args: None; owner = object()
    activation = dict(record_store=object(), run_provenance=_provenance(configuration),
        publication_store=object(), display_owner=owner, display_state=Display(),
        source_owner=source_owner, gui_thread_id=1, light_cancel=callback,
        light_drain=callback, light_verify=callback, on_frame_completed=callback,
        on_checkpoint_recoverable=callback)
    with __import__("pytest").raises(RuntimeError, match="identity"): adapter.activate(replace(preparation), **activation)
    active, created = adapter.activate(preparation, **activation)
    allocation = preparation.resources[0].allocation
    assert active is session and created and captured.policy is preparation.resources[0]
    assert allocations == [allocation] and display_allocations == [allocation] and captured.generation == configuration.generation
    assert adapter._current["background_bindings"] is preparation.effective.background_bindings
    with __import__("pytest").raises(RuntimeError, match="consumed"): adapter.activate(preparation, **activation)
    class Source:
        frame_indices = (1,); allocation = SimpleNamespace(queue_depth=1)
        def iter_chunks(self, size): yield np.ones((1, 2, 2), np.uint16), (1,)
        def take_direct_chunk_fact(self): return None
    monkeypatch.setattr(module, "NexusStackSource", Source)
    run = SimpleNamespace(configuration=configuration, stop_signal=stop,
        source=Source(), frames_by_label={1: scan.frames[0]}, stop_requested=False,
        context_runtime=None, resource_facts=[], cleanup_failures=[], perf_enabled=False)
    module.StandardRunExecutor._submit_container_source(run, adapter)
    assert seen[-1] is stop
    binding = adapter.background_binding(1)
    assert scan.frames[0].background_dependency_bytes is binding[2] and scan.frames[0].background_dependency_fingerprint is binding[3]
    assert captured.submits == [1]
    stale = ScanFrame(9, image=np.ones((2, 2)), background=np.ones((2, 2)),
        background_dependency_bytes=binding[2], background_dependency_fingerprint=binding[3])
    none_run = SimpleNamespace(configuration=SimpleNamespace(background=FrameBackgroundPlan(), live_mode=False), stop_signal=stop)
    assert module._background_ready(none_run, adapter, stale)
    assert stale.background is stale.background_dependency_bytes is stale.background_dependency_fingerprint is None
    executor = Path(module.__file__); order = _function_order(executor, "_construct", ("prepare_admission(", "target_state_matches(", "add_artifact(", "activate("))
    assert order == tuple(sorted(order))
    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    assert sum(isinstance(node, ast.FunctionDef)
               and node.name == "_resolve_background_before_submit" for node in tree.body) == 1
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_resolve_background_before_submit")
    helper_text = ast.get_source_segment(source, helper) or ""
    assert "_command_lock" not in helper_text
    output = SimpleNamespace(background_binding=lambda label: _binding(label))
    run = SimpleNamespace(configuration=SimpleNamespace(
        background=FrameBackgroundPlan(mode="Single BG File", locator="/bg.tif"),
        live_mode=False), stop_signal=event)
    second = ScanFrame(1, image=np.ones((2, 2)), source_path="/raw/target.tif")
    assert module._background_ready(run, output, second) and seen[-1] is event
    guards = tuple(node for node in ast.walk(tree) if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not) and isinstance(node.test.operand, ast.Call) and isinstance(node.test.operand.func, ast.Name) and node.test.operand.func.id == "_background_ready")
    assert len(guards) == 3 and all(any(isinstance(child, (ast.Break, ast.Return)) for child in ast.walk(node)) for node in guards)
    submits = tuple(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "output" and node.func.attr == "submit" and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id == "frame")
    assert len(submits) == 3 and all(0 < submit.lineno - guard.lineno <= 7 for guard, submit in zip(sorted(guards, key=lambda node: node.lineno), sorted(submits, key=lambda node: node.lineno)))
    run_class = next(node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "_StandardRun")
    assert "_preparations" not in (ast.get_source_segment(source, run_class) or "")


def test_live_retries_absent_torn_then_submits_and_writes_once(monkeypatch, tmp_path: Path) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor as module
    configuration, scan, plan, item, decision = _composition(tmp_path, live=True, count=1)
    captured, session = _fake_graph_io(monkeypatch, dynamic_output); adapter = dynamic_output.DynamicOutputAdapter(configuration); stop = Event()
    preparation = adapter.prepare_admission(scan, plan, item, decision, stop, qualify=lambda policy, prior: prior)
    active, created = adapter.activate(preparation, record_store=object(), run_provenance=_provenance(configuration))
    assert active is session and created and captured.policy is preparation.resources[0] and adapter.background_binding(1) is None
    dispositions = iter(("RETRYABLE", "RESOLVED")); calls = []; raw = _binding()[2]
    value = np.frombuffer(np.ones((2, 2), dtype=np.float64).tobytes(), dtype=np.float64).reshape(2, 2); value.setflags(write=False)
    def resolve(plan, fact, *, cancelled=None):
        disposition = next(dispositions); calls.append((disposition, cancelled))
        return FrameBackgroundResult(disposition, value if disposition == "RESOLVED" else None, raw if disposition == "RESOLVED" else None, hashlib.sha256(raw).hexdigest() if disposition == "RESOLVED" else None)
    monkeypatch.setattr(module, "resolve_frame_background", resolve); monkeypatch.setattr(module, "_LIVE_DIRECTORY_POLL_S", 0)
    run = SimpleNamespace(configuration=configuration, stop_signal=stop); frame = scan.frames[0]
    assert module._background_ready(run, adapter, frame)
    binding = adapter.background_binding(1)
    assert [item[0] for item in calls] == ["RETRYABLE", "RESOLVED"] and all(item[1] is stop for item in calls)
    assert frame.background_dependency_bytes is binding[2] and frame.background_dependency_fingerprint is binding[3]
    assert adapter.submit(frame) and captured.submits == [1]
    monkeypatch.setattr(module, "resolve_frame_background", lambda *a, **k: FrameBackgroundResult("CANCELLED", None, None, None))
    with __import__("pytest").raises(RuntimeError, match="admission cancelled"): module._background_ready(run, adapter, frame)
    assert captured.submits == [1]


def test_reduction_subtracts_without_mutation_or_alternate_owner() -> None:
    from xrd_tools.reduction.core import _subtract_background
    image = np.arange(4, dtype=np.uint16).reshape(2, 2)
    background = np.ones((2, 2), dtype=np.float64)
    before_image, before_background = image.copy(), background.copy()
    result = _subtract_background(image, background)
    np.testing.assert_array_equal(result, [[-1.0, 0.0], [1.0, 2.0]])
    np.testing.assert_array_equal(image, before_image)
    np.testing.assert_array_equal(background, before_background)
    root = Path(__file__).parents[3]
    owners = tuple(root / path for path in (
        "src/xrd_tools/reduction/background.py", "src/xrd_tools/reduction/__init__.py",
        "src/xrd_tools/core/scan.py", "src/xrd_tools/session/run_configuration.py",
        "src/xrd_tools/session/policy.py", "src/xrd_tools/session/readiness.py",
        "src/xdart/gui/tabs/scattering/controls_inventory.py",
        "src/xdart/gui/tabs/scattering/controls_editing.py",
        "src/xdart/gui/tabs/scattering/controls_projection.py",
        "src/xdart/gui/pages/scattering_workspace.py",
        "src/xdart/gui/tabs/scattering/contracts.py",
        "src/xdart/gui/tabs/scattering/output_preflight.py",
        "src/xdart/gui/tabs/scattering/adapters/run_executor.py",
        "src/xdart/gui/tabs/scattering/adapters/dynamic_output.py",
        "src/xrd_tools/reduction/core.py", "src/xrd_tools/io/record_writer.py",
        "src/xrd_tools/io/nexus_record.py",
        "src/xrd_tools/session/headless_scan.py"))
    definitions = sum(path.read_text().count("def _resolve_background_before_submit") for path in owners)
    assert definitions == 1
    assert sum(path.read_text().count("class NexusRecordWriter") for path in owners) == 1
    assert sum(path.read_text().count("def write_background_dependency") for path in owners) == 1
    assert sum(path.read_text().count("def _subtract_background") for path in owners) == 1
    assert all(path.is_file() for path in owners)
