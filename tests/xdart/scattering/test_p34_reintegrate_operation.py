"""P3-4A headless reintegration composition and owner oracle."""

from __future__ import annotations

import copy, inspect, json, os, statistics, subprocess, sys, threading, time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.core.test_vnext_p34_existing_replacement import _seed_existing, _stub_integrators
from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
@pytest.fixture
def qapp():
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
def _wait(call, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = call()
        if value is not None: return value
        time.sleep(.005)
    raise AssertionError("timed out waiting for bounded operation")
def _loaded_page(tmp_path, monkeypatch, qapp, seed=None):
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    seeded = seed or _seed_existing(tmp_path)
    page, store = _page(tmp_path, monkeypatch); request = page._context_controller.begin_browse(str(seeded.target.resolve()))
    outcome = _wait(page._context_controller.poll_browse); assert outcome.request is request and outcome.status is BrowseLoadStatus.READY
    context = page._context_controller.browse_context; assert context is not None and context.loaded
    return page, store, seeded, context
def _persisted(selected, workers=1):
    selected=copy.deepcopy(selected); selected["gi_mode"]=selected["gi_mode"] or "q_total"; return {"api_version": 1, "selected_plan": selected,
            "requested_shared_science": {"version": 1, "kind": "persisted_target"},
            "resource_policy": {"version": 1, "kind": "resolve", "envelope_bytes": 8 << 30,
                                "requests": {"workers": workers, "reduction_inflight": 1}}}
def _join(slot, identity):
    worker = slot._worker; assert worker is not None; worker.join(20); assert not worker.is_alive()
    update = slot.poll(identity); assert update is not None and update.terminal is not None
    return update

def test_ordinary_output_routes_group_target_through_one_borrowed_lock(tmp_path):
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    target = tmp_path / "canonical.nxs"; target.write_bytes(b"x")
    alias = tmp_path / "alias.nxs"; alias.symlink_to(target)
    assert dynamic_output._target_key(alias) == dynamic_output._target_key(target)
    adapter = dynamic_output.DynamicOutputAdapter(SimpleNamespace())
    def prepared(shown):
        item = SimpleNamespace(target=shown, group=SimpleNamespace(target=target))
        value = dynamic_output._PreparedAdmission(adapter, object(), object(), item, object(), object(), threading.Event(), (None,) * 6, (None,) * 5, None)
        object.__setattr__(value, "identity", value); return value
    with pytest.raises(TypeError, match="exact run provenance"):
        adapter._activate_owned(prepared(alias), record_store=None, run_provenance=object(), cancelled=lambda: False)
    with pytest.raises(ValueError, match="canonical group target"):
        adapter._activate_owned(prepared(tmp_path / "other.nxs"), record_store=None, run_provenance=object(), cancelled=lambda: False)
    source = inspect.getsource(dynamic_output.DynamicOutputAdapter._activate_owned)
    assert "_target_key(item.target) != _target_key(item.group.target)" in source
    assert "target = Path(item.group.target)" in source
    assert "file_lock=self._command_lock" in source
    assert source.count("file_lock=self._command_lock") == 3
    assert adapter._command_lock is adapter._command_lock


def test_reintegrate_values_progress_cancel_recipe_and_owner_census(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    probe_code = "import sys;sys.path.insert(0," + repr(str(root / "src")) + ");from xrd_tools.reduction import ReintegratePlan,ReintegrateProgress,ReintegrateResult,ReintegrateRunner,run_reintegrate;bad=sorted(name for name in sys.modules if name.split('.')[0] in {'xdart','qtpy','PyQt5','PySide6','pyFAI'});assert not bad,bad"
    probe = subprocess.run([sys.executable, "-I", "-c", probe_code], capture_output=True, text=True, check=False)
    assert probe.returncode == 0, probe.stderr
    from xrd_tools.reduction import (
        ReintegratePlan,
        ReintegrateProgress,
        ReintegrateResult,
        ReintegrateRunner,
        run_reintegrate,
    )
    from xrd_tools.reduction import reintegrate as module
    from xrd_tools.io.output_transaction import StreamTerminal

    assert all(value is not None for value in (
        ReintegratePlan, ReintegrateProgress, ReintegrateResult,
        ReintegrateRunner, run_reintegrate,
    ))
    assert tuple(field.name for field in fields(ReintegratePlan)) == (
        "api_version", "target", "entry", "expected_target_snapshot", "dimension",
        "labels", "detector_shape", "native_dtype", "selected_plan",
        "requested_shared_science", "gi_bootstrap_incidence", "session_policy",
        "rollback_policy", "science_identity", "operation_identity",
    )
    assert set(field.name for field in fields(ReintegrateProgress)) == {
        "operation_identity", "stage", "completed", "total", "revision",
    }
    assert tuple(field.name for field in fields(ReintegrateResult)) == (
        "disposition", "input_labels", "committed_labels",
        "publication_dropped_labels", "diagnostics", "science_identity",
        "operation_identity", "audit_identity", "commit_identity",
    )
    assert tuple(inspect.signature(ReintegratePlan.from_artifact).parameters) == (
        "target", "entry", "dimension", "preparation", "expected_target_snapshot",
        "expected_labels", "cancel_token",
    )
    assert tuple(inspect.signature(ReintegratePlan.from_recipe).parameters) == ("recipe",)
    assert tuple(inspect.signature(ReintegrateRunner).parameters) == ("plan", "cancel_token", "progress_cb")
    assert tuple(inspect.signature(run_reintegrate).parameters) == ("plan", "cancel_token", "progress_cb")
    progress = module._progress("a" * 64, "read", 1, 3, 2)
    assert (progress.completed, progress.total, progress.revision) == (1, 3, 2)
    assert ReintegrateProgress.__dataclass_params__.frozen
    assert ReintegrateResult.__dataclass_params__.frozen
    assert ReintegratePlan.__dataclass_params__.frozen
    for cls in (ReintegratePlan, ReintegrateProgress, ReintegrateResult):
        with pytest.raises(TypeError, match="factory-constructed"): cls()
    failures = module._ExecutionRuntime(SimpleNamespace(operation_identity="a" * 64), None, lambda _value: (_ for _ in ()).throw(RuntimeError("é" * 2000)))
    for revision in range(20): failures._report("read", 0, 1)
    assert failures.revision == 20 and len(failures.diagnostics) == 16
    assert all(len(value.encode("utf-8")) <= 1024 for value in failures.diagnostics)

    plan = object.__new__(ReintegratePlan)
    object.__setattr__(plan, "labels", (2, 5)); object.__setattr__(plan, "science_identity", "b" * 64); object.__setattr__(plan, "operation_identity", "c" * 64)
    terminal = StreamTerminal("/detached", 1, "d" * 64, 1)
    class Pending:
        primary = None
        def __init__(self): self.custody = True; self.retries = 0
        def run(self): return module._RuntimeOutcome("SETTLEMENT_PENDING", (), (), (), None, None)
        def finish_current(self): self.retries += 1; self.custody = False; return module._RuntimeOutcome("COMMITTED", (2, 5), (), (), "e" * 64, terminal)
        def _has_custody(self): return self.custody
        def close(self): assert not self.custody
    created = []
    monkeypatch.setattr(module, "_open_runtime", lambda *_a, **_k: created.append(Pending()) or created[-1])
    runner = ReintegrateRunner(plan); pending = runner.run()
    assert pending.disposition == "SETTLEMENT_PENDING" and pending.commit_identity is None
    with pytest.raises(RuntimeError, match="custody remains pending"): runner.close()
    committed = runner.finish_current(); runner.close()
    assert committed.disposition == "COMMITTED" and committed.commit_identity is terminal and created[0].retries == 1
    convenient = run_reintegrate(plan)
    assert convenient.disposition == "COMMITTED" and convenient.commit_identity is terminal and created[1].retries == 1
    from xrd_tools.io import record_writer
    from xrd_tools.reduction import Frame
    trace = []
    class TraceLock:
        def __enter__(self): trace.append("lock-enter"); return self
        def __exit__(self, *_args): trace.append("lock-exit")
    writer = object.__new__(record_writer.NexusRecordWriter)
    writer.phase = record_writer.WriterPhase.ACTIVE; writer.file_lock = TraceLock(); writer._in_boundary = False; writer._h5 = object(); writer.entry = "entry"; writer._row_cursors = {}; writer._replacement_read_context = None
    fact = {"label": 7, "path": "/raw", "frame_index": 0, "snapshot": {"mtime_ns": 1}, "metadata": {}, "geometry": {}, "background_dependency": None}
    monkeypatch.setattr(record_writer, "_decode_replacement_fact", lambda *_a, **_k: trace.append("detach") or fact)
    marker = object(); monkeypatch.setattr(module, "_load_fact", lambda *_a, **_k: trace.append("raw") or (Path("/raw"), marker, None, None))
    shared = {"background": {"version": 1, "mode": "None"}, "gi": {"enabled": False, "resolved_motor": "Manual"}, "geometry": None}
    source_plan = SimpleNamespace(requested_shared_science=shared, labels=(7,), detector_shape=(2, 2), native_dtype="<u2", resource_allocation=None)
    frame = Frame(7); source = module._ReintegrateFrameSource(source_plan); source._frames = {7: frame}; source.bind_fact_reader(writer._detach_replacement_fact)
    assert source.prepare(frame)[0] is marker and trace == ["lock-enter", "detach", "lock-exit", "raw"]
    runtime_source = inspect.getsource(module._ExecutionRuntime.run)
    assert runtime_source.index("self.source.prepare(frame)") < runtime_source.index("self.session._session.drain()") and "with " not in runtime_source
    source.clear_jit()

    owners = {
        "src/xrd_tools/io/output_transaction.py",
        "src/xrd_tools/io/append.py",
        "src/xrd_tools/io/record_writer.py",
        "src/xrd_tools/reduction/core.py",
        "src/xrd_tools/reduction/reintegrate.py",
        "src/xrd_tools/reduction/__init__.py",
        "src/xdart/gui/tabs/scattering/adapters/dynamic_output.py",
        "src/xrd_tools/session/scan_session.py",
    }
    assert len(owners) == 8
    assert all((root / path).exists() for path in owners)
    source = (root / "src/xrd_tools/reduction/reintegrate.py").read_text()
    assert "import xdart" not in source and "from xdart" not in source
    assert "PyQt" not in source and "PySide" not in source and "qtpy" not in source

def test_parent_red_stable_loaded_browse_enables_reintegrate_1d_start(tmp_path, monkeypatch, qapp):
    from xrd_tools.session.readiness import ControlAction, SectionId
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    page, store, _seed, _context = _loaded_page(tmp_path, monkeypatch, qapp)
    projected = page._project_controls(store.snapshot()).profile; actions = {a.action: a for a in projected.actions_for(SectionId.PROCESSING)}; experiment = {a.action: a for a in projected.actions_for(SectionId.EXPERIMENT)}
    assert actions[ControlAction.REINTEGRATE_1D].enabled and actions[ControlAction.REINTEGRATE_1D].label == "Reintegrate 1-D"
    assert not actions[ControlAction.REINTEGRATE_2D].enabled and experiment[ControlAction.CALIBRATE].label == "Calibrate" and experiment[ControlAction.MAKE_MASK].label == "Make Mask"
    page._lifecycle._phase = RunPhase.FAILED; page._lifecycle._owners_closed = False; actions = {a.action: a for a in page._project_controls(store.snapshot()).profile.actions_for(SectionId.PROCESSING)}
    assert not actions[ControlAction.REINTEGRATE_1D].enabled
    page._lifecycle._owners_closed = True; actions = {a.action: a for a in page._project_controls(store.snapshot()).profile.actions_for(SectionId.PROCESSING)}
    assert actions[ControlAction.REINTEGRATE_1D].enabled; page.close_workspace()
def test_browse_snapshot_brackets_complete_load_and_refuses_drift(tmp_path, monkeypatch):
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest, BrowseLoadStatus
    from xrd_tools.io.output_transaction import capture_target_snapshot
    target, _raw = _write_processed(tmp_path, labels=(2, 5, 9)); calls = []; real = capture_target_snapshot
    monkeypatch.setattr(module, "capture_target_snapshot", lambda path: calls.append(threading.current_thread().name) or real(path), raising=False)
    loader = module.BrowseLoader(); request = BrowseLoadRequest("snapshot", 1, str(target.resolve())); outcome = _wait(lambda: loader.poll(loader.begin(request)))
    context = loader.consume(outcome); assert outcome.status is BrowseLoadStatus.READY and context.loaded_labels == (2, 5, 9) and len(calls) == 2 and calls == ["scattering-browse"] * 2
    with pytest.raises(Exception): context.target_snapshot = None
    assert loader.release_context(context).cleanup_status.value == "cleaned"
    snap = real(target); values = iter((snap, replace(snap, digest="0" * 64))); monkeypatch.setattr(module, "capture_target_snapshot", lambda _path: next(values))
    bad = module.BrowseLoader(); bad_request = BrowseLoadRequest("drift", 1, str(target.resolve())); bad.begin(bad_request); refused = _wait(lambda: bad.poll(bad_request)); assert refused.status is BrowseLoadStatus.FAILED and bad.context_for_outcome(refused) is None
def test_private_request_prepares_plan_only_on_existing_operation_worker(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import external_operation as module
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationTerminalStatus
    from xrd_tools.reduction import ReintegrateResult, reintegrate as core
    entered, release, seen = threading.Event(), threading.Event(), []; plan = SimpleNamespace(operation_identity="a" * 64)
    result = core._value(ReintegrateResult, "COMMITTED", (2,), (2,), (), (), "b"*64, "a"*64, "c"*64, None)
    def build(target, **kw): entered.set(); release.wait(2); seen.append((threading.current_thread().name, target, kw)); return plan
    def run(got, **kw): seen.append((got, kw)); return result
    monkeypatch.setattr(module.ReintegratePlan, "from_artifact", staticmethod(build)); monkeypatch.setattr(module, "run_reintegrate", run)
    values = _persisted({"version": 1, "dimension": "1d", "bai_args": {}, "gi_mode": "q_total"}); slot = module.OperationSlot(); identity = slot.begin_reintegrate(target="/detached", entry="entry", expected_target_snapshot=core.TargetSnapshot(True,1,2,3,4,"d"*64), expected_labels=(2,), dimension="1d", preparation_values=values, stamp=OperationContextStamp(0))
    assert entered.wait(2); values["selected_plan"]["bai_args"]["npt"] = 99; release.set(); update = _join(slot, identity)
    assert update.terminal.status is OperationTerminalStatus.RETURNED and seen[0][0].startswith("scattering-operation-") and "npt" not in seen[0][2]["preparation"]["selected_plan"]["bai_args"] and seen[1][0] is plan and seen[0][2]["cancel_token"] is seen[1][1]["cancel_token"]
def test_expected_labels_are_rederived_and_exactly_compared(tmp_path):
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegratePlan
    seeded = _seed_existing(tmp_path); snap = capture_target_snapshot(seeded.target); prep = _persisted(seeded.preparation["selected_plan"])
    for labels in ((5, 2, 9), (2, 2, 9), (2, 5), (2, 5, 9, 10)):
        with pytest.raises(ValueError): ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=prep, expected_target_snapshot=snap, expected_labels=labels)
    assert ReintegratePlan.from_artifact(seeded.target, entry="entry", dimension="1d", preparation=prep, expected_target_snapshot=snap, expected_labels=seeded.labels).labels == seeded.labels
def test_direct_and_gui_scheduled_1d_match_after_reopen(tmp_path, monkeypatch):
    import h5py, numpy as np
    from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationTerminalStatus
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import ReintegratePlan, run_reintegrate
    first = _seed_existing(tmp_path, name="direct"); second = _seed_existing(tmp_path, name="gui"); _stub_integrators(monkeypatch)
    direct = run_reintegrate(ReintegratePlan.from_artifact(first.target, entry="entry", dimension="1d", preparation=first.preparation)); slot = OperationSlot(); identity = slot.begin_reintegrate(target=str(second.target.resolve()), entry="entry", expected_target_snapshot=capture_target_snapshot(second.target), expected_labels=second.labels, dimension="1d", preparation_values=_persisted(second.preparation["selected_plan"]), stamp=OperationContextStamp(0)); gui = _join(slot, identity)
    assert gui.terminal.status is OperationTerminalStatus.RETURNED and direct.disposition == gui.terminal.payload.disposition == "COMMITTED"
    with h5py.File(first.target) as a, h5py.File(second.target) as b: assert np.array_equal(a["entry/integrated_1d/intensity"], b["entry/integrated_1d/intensity"]) and np.array_equal(a["entry/integrated_2d/intensity"], b["entry/integrated_2d/intensity"])
def test_browse_invalidation_terminal_reload_and_foreign_stale_refusal(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.operation_values import OperationIdentity, OperationTerminal, OperationTerminalStatus, OperationUpdate; from xrd_tools.reduction import ReintegrateResult, reintegrate as core; page, _store, _seed, context = _loaded_page(tmp_path, monkeypatch, qapp); controller = page._context_controller
    captured = controller.capture_reintegrate_browse(); assert captured[0] is context and controller.invalidate_reintegrate_browse(*captured)
    request, target = context.load_request, context.requested_path; foreign = replace(request, token=request.token + "-foreign"); assert foreign is not request and controller.reload_reintegrate_browse(foreign, target) is None and controller.browse_context is context and context.invalidated and not context.released
    rid = OperationIdentity(77); result = core._value(ReintegrateResult, "COMMITTED", (1,), (1,), (), (), "b"*64, "a"*64, "c"*64, None); page._reintegrate_identity, page._reintegrate_request, page._reintegrate_target = rid, request, target; assert page._consume_reintegrate_update(OperationUpdate(rid, terminal=OperationTerminal(rid, OperationTerminalStatus.RETURNED, payload=result), stale=True))
    assert result.disposition == "COMMITTED" and page._reintegrate_identity is None and controller._browse_request is not None and controller._browse_request is not request and context.released; page.close_workspace()
def test_same_event_cancels_prepare_and_run_without_false_terminal(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import external_operation as module
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationTerminalStatus
    from xrd_tools.reduction import reintegrate as core; from xrd_tools.reduction.reintegrate import ReintegrateCancelled
    seen = []; entered = threading.Event()
    def cancelled(_target, **kw): seen.append(kw["cancel_token"]); entered.set(); kw["cancel_token"].wait(2); raise ReintegrateCancelled()
    monkeypatch.setattr(module.ReintegratePlan, "from_artifact", staticmethod(cancelled)); slot = module.OperationSlot(); identity = slot.begin_reintegrate(target="/cancel", entry="entry", expected_target_snapshot=core.TargetSnapshot(True,1,2,3,4,"d"*64), expected_labels=(1,), dimension="1d", preparation_values=_persisted({"version":1,"dimension":"1d","bai_args":{},"gi_mode":"q_total"}), stamp=OperationContextStamp(0))
    assert entered.wait(2) and slot.cancel(identity); update = _join(slot, identity)
    assert update.terminal.status is OperationTerminalStatus.CANCELLED and seen[0].is_set()
def test_reintegrate_progress_result_close_and_control_projection(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import external_operation as module
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationTerminalStatus
    from xrd_tools.reduction import ReintegrateResult, reintegrate as core
    plan = SimpleNamespace(operation_identity="a"*64); result = core._value(ReintegrateResult,"COMMITTED",(1,),(1,),(),(),"b"*64,"a"*64,"c"*64,None)
    monkeypatch.setattr(module.ReintegratePlan,"from_artifact",staticmethod(lambda *_a,**_k: plan))
    def run(_plan, **kw): kw["progress_cb"](core._progress("a"*64,"read",0,1,1)); kw["progress_cb"](core._progress("a"*64,"write",1,1,2)); kw["progress_cb"](core._progress("x"*64,"settle",1,1,3)); return result
    monkeypatch.setattr(module,"run_reintegrate",run); slot=module.OperationSlot(); identity=slot.begin_reintegrate(target="/progress",entry="entry",expected_target_snapshot=core.TargetSnapshot(True,1,2,3,4,"d"*64),expected_labels=(1,),dimension="1d",preparation_values=_persisted({"version":1,"dimension":"1d","bai_args":{},"gi_mode":"q_total"}),stamp=OperationContextStamp(0)); update=_join(slot,identity)
    assert update.terminal.status is OperationTerminalStatus.RETURNED and update.progress.identity is identity and (update.progress.stage,update.progress.revision)==("write",4) and all(getattr(slot,n) is None for n in ("_frozen","_worker","_cancel_event"))

def test_reintegrate_gui_owner_import_writer_and_snapshot_frequency_census():
    root=Path(__file__).resolve().parents[3]; names=("src/xrd_tools/reduction/reintegrate.py","src/xdart/modules/display_context.py","src/xdart/gui/tabs/scattering/adapters/browse_loader.py","src/xdart/gui/tabs/scattering/context_controller.py","src/xdart/gui/tabs/scattering/adapters/external_operation.py","src/xdart/gui/tabs/scattering/page.py","src/xdart/gui/tabs/scattering/controls_projection.py")
    sources={name:(root/name).read_text() for name in names}; gui_text="\n".join(sources[name] for name in names[1:]); external_source=sources[names[4]]; page_source=sources[names[5]]
    assert len(sources)==7 and sources[names[2]].count("capture_target_snapshot(")==2 and gui_text.count("Thread(")==2 and gui_text.count("OperationSlot()")==1
    assert not any(value in gui_text for value in ("h5py","NexusSink","NexusRecordWriter","_core_plan","_integration_1d_args","_integration_2d_args","resolve_session_policy")) and "ReintegrateRunner" not in external_source and external_source.count("ReintegratePlan.from_artifact(")==1 and external_source.count("run_reintegrate(")==1 and page_source.count("jsonable_run_value(")==1

def test_reintegrate_request_accepts_exact_dimensions_but_p34b_gui_is_1d_only(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationIdentity
    from xrd_tools.session.readiness import ControlAction, SectionId; from xrd_tools.reduction import reintegrate as core
    slot=OperationSlot(); seen=[]; monkeypatch.setattr(slot,"_begin",lambda value,stamp,body: seen.append(value) or object())
    kwargs=dict(target="/target",entry="entry",expected_target_snapshot=core.TargetSnapshot(True,1,2,3,4,"d"*64),expected_labels=(1,),preparation_values=_persisted({"version":1,"dimension":"1d","bai_args":{},"gi_mode":"q_total"}),stamp=OperationContextStamp(0))
    assert slot.begin_reintegrate(dimension="1d",**kwargs) is not None and slot.begin_reintegrate(dimension="2d",**kwargs) is not None and slot.begin_reintegrate(dimension="3d",**kwargs) is None and tuple(f.name for f in fields(type(seen[0]))) == ("target","entry","expected_target_snapshot","expected_labels","dimension","preparation_json")
    page,store,_seed,_context=_loaded_page(tmp_path,monkeypatch,qapp); rid=OperationIdentity(77); page._operation_slot._identity=rid; page._reintegrate_identity=rid; captured=page._context_controller.capture_reintegrate_browse(); assert captured is not None and page._context_controller.invalidate_reintegrate_browse(*captured); actions={a.action:a for a in page._project_controls(store.snapshot()).profile.actions_for(SectionId.PROCESSING)}; assert actions[ControlAction.REINTEGRATE_1D].enabled and actions[ControlAction.REINTEGRATE_1D].label=="Cancel Reintegrate 1-D" and not actions[ControlAction.REINTEGRATE_2D].enabled and 'if command.value == "reintegrate_2d"' not in inspect.getsource(page._handle_shell_command); page._operation_slot._identity=None; page._reintegrate_identity=None; page.close_workspace()

def test_persisted_target_resolves_after_authenticated_inventory_and_matches_explicit_recipe(tmp_path, monkeypatch):
    from xrd_tools.io.output_transaction import capture_target_snapshot; from xrd_tools.reduction import ReintegratePlan, reintegrate as module
    seeded=_seed_existing(tmp_path); snapshot=capture_target_snapshot(seeded.target); trace=[]; real_inspect=module._inspect_artifact; real_capture=module.capture_target_snapshot
    monkeypatch.setattr(module,"capture_target_snapshot",lambda *a,**k: trace.append(("capture",real_capture(*a,**k))) or trace[-1][1]); monkeypatch.setattr(module,"_inspect_artifact",lambda *a,**k: trace.append(("inspect",real_inspect(*a,**k))) or trace[-1][1])
    selected={"version":1,"dimension":"1d","bai_args":{},"gi_mode":"q_total"}; request=_persisted(selected,workers=2); request["resource_policy"]["requests"]={"workers":2}; pytest.raises(ValueError,ReintegratePlan.from_artifact,seeded.target,entry="entry",dimension="1d",preparation=request,expected_target_snapshot=None,expected_labels=seeded.labels); pytest.raises(ValueError,ReintegratePlan.from_artifact,seeded.target,entry="entry",dimension="1d",preparation=request,expected_target_snapshot=snapshot,expected_labels=None); assert trace==[]
    persisted=ReintegratePlan.from_artifact(seeded.target,entry="entry",dimension="1d",preparation=request,expected_target_snapshot=snapshot,expected_labels=seeded.labels); expected={"version":1,"dimension":"1d","bai_args":{"npt":1000,"unit":"q_A^-1","method":"csr","radial_range":None,"azimuth_range":None},"gi_mode":None}; shared=module._plain(persisted.requested_shared_science)
    explicit={"api_version":1,"selected_plan":expected,"requested_shared_science":shared,"resource_policy":request["resource_policy"]}; direct=ReintegratePlan.from_artifact(seeded.target,entry="entry",dimension="1d",preparation=explicit,expected_target_snapshot=snapshot,expected_labels=seeded.labels)
    assert [kind for kind,_value in trace]==["capture","inspect","capture"]*2 and trace[0][1]==snapshot and module._plain(persisted.selected_plan)==expected and persisted.resource_allocation.counts["workers"]==2 and persisted.as_recipe()==direct.as_recipe() and persisted.operation_identity==direct.operation_identity and '"kind":"persisted_target"' not in json.dumps(persisted.as_recipe(),separators=(",",":")) and set(persisted.requested_shared_science)=={"version","gi","threshold","poni_values","accepted_scientific_assets","geometry","background"}

def test_gui_persisted_science_disclosure_and_no_shared_control_override(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.operation_values import OperationIdentity
    page,store,seeded,_context=_loaded_page(tmp_path,monkeypatch,qapp); plain=page._reintegrate_1d_preparation(store.snapshot().thaw()); assert plain["selected_plan"]["bai_args"]=={}; candidate=store.snapshot().thaw(); candidate.bai_1d_args={"radial_range":(.1,1.)}; candidate.max_cores=2; store.commit(candidate,expected_revision=store.revision); captured={}
    monkeypatch.setattr(page._operation_slot,"begin_reintegrate",lambda **kw: captured.update(kw) or OperationIdentity(91)); page._reintegrate_1d_action()
    prep=captured["preparation_values"]; assert prep["requested_shared_science"]=={"version":1,"kind":"persisted_target"} and prep["selected_plan"]["bai_args"]=={"radial_range":[.1,1.]} and prep["selected_plan"]["gi_mode"]==candidate.gi.mode_1d and prep["resource_policy"]["requests"]=={"workers":2}
    shown=json.dumps(prep); assert seeded.preparation["requested_shared_science"]["accepted_scientific_assets"]["poni_sha256"] not in shown and "loaded artifact" in page._notice_text and captured["dimension"]=="1d"; page._reintegrate_identity=None; page.close_workspace()

def test_p34b_production_browse_reload_performance_probe(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest, BrowseLoadStatus
    from xrd_tools.io.output_transaction import capture_target_snapshot
    target=Path(os.environ["P3_4B_BROWSE_ARTIFACT"]).resolve(); loads=int(os.environ["P3_4B_BROWSE_LOADS"]); warm=int(os.environ["P3_4B_BROWSE_WARMUPS"]); calls=[]; real=capture_target_snapshot
    monkeypatch.setattr(module,"capture_target_snapshot",lambda path: calls.append(threading.current_thread().name) or real(path),raising=False); walls=[]
    for index in range(loads):
        loader=module.BrowseLoader(max_items=512); request=BrowseLoadRequest(f"perf-{index}",1,str(target)); started=time.perf_counter(); outcome=_wait(lambda: loader.poll(loader.begin(request)),60); context=loader.consume(outcome); walls.append(time.perf_counter()-started)
        assert outcome.status is BrowseLoadStatus.READY and tuple(getattr(context,"loaded_labels",context.frame_ids))==tuple(range(651)); assert loader.release_context(context).cleanup_status.value=="cleaned" and loader._active is loader._queued is None
    started=time.perf_counter(); left,right=real(target),real(target); direct=time.perf_counter()-started; measured=walls[warm:]; payload={"walls":walls,"measured":measured,"median":statistics.median(measured),"early":statistics.median(measured[:3]),"late":statistics.median(measured[-3:]),"snapshot_pair_wall":direct,"digest":left.digest,"loader_snapshot_calls":len(calls),"snapshot_threads":calls}
    assert left==right; parent=os.environ.get("P3_4B_BROWSE_PARENT_RESULT")
    if parent:
        baseline=json.loads(Path(parent).read_text()); projected=baseline["median"]+baseline["snapshot_pair_wall"]; assert len(calls)==14 and calls==["scattering-browse"]*14 and payload["median"]<=max(1.2*projected,projected+.5) and payload["late"]<=max(1.2*payload["early"],payload["early"]+.5)
    Path(os.environ["P3_4B_BROWSE_RESULT"]).write_text(json.dumps(payload,sort_keys=True,separators=(",",":")))
