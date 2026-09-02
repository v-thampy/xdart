"""P3-4A headless reintegration composition and owner oracle."""

from __future__ import annotations

import copy, hashlib, inspect, json, os, statistics, subprocess, sys, threading, time
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.core.test_vnext_p34_existing_replacement import _seed_existing, _stub_integrators
from tests.xdart.scattering.test_e4_preview_transport import _write_processed
from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
from xdart.gui.tabs.scattering.processed_browser import (
    TerminalBrowseHandoff,
    TerminalPaintMode,
)
from xdart.gui.tabs.scattering.workspace_operations import (
    ReintegrateOperationState,
)


def _reintegrate_slot(page):
    return page._workspace_operations._slot


def _set_reintegrate_state(page, identity, capture, dimension="1d"):
    page._workspace_operations._reintegrate = ReintegrateOperationState(
        identity, capture, dimension
    )
    return page._workspace_operations


def _begin_terminal_handoff(page, expected: TerminalBrowseHandoff):
    handoff = page._processed_browser.begin_terminal_handoff(
        expected.request,
        expected.run_identity,
        expected.source_artifact,
        expected.current_label,
        expected.selected_labels,
        expected.commit_identity,
        timing_start=None,
    )
    assert handoff == expected
    assert page._processed_browser.terminal_handoff is handoff
    return handoff


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
def _loaded_page(
    tmp_path, monkeypatch, qapp, seed=None, *, terminal_browse=False,
):
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    seeded = seed or _seed_existing(tmp_path)
    page, store = _page(tmp_path, monkeypatch)
    if terminal_browse:
        from tests.xdart.scattering.test_e3_context_contract import _acquisition
        identity, acquisition = _acquisition()
        page._context_controller._runtime.adopt_acquisition(
            identity, acquisition,
        )
    request = page._context_controller.begin_browse(
        str(seeded.target.resolve()),
        source_root=str(seeded.target.parent.resolve()),
        terminal_commit_identity=(
            seeded.terminal.commit_identity if terminal_browse else None
        ),
    )
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
def _resolved_2d(workers=1): return {"api_version":1,"selected_plan":{"version":1,"dimension":"2d","bai_args":{},"gi_mode":"qip_qoop"},"requested_shared_science":{"version":1,"kind":"persisted_target"},"resource_policy":{"version":1,"kind":"resolve","envelope_bytes":None,"requests":{"workers":workers}}}
def _tree_manifest(path, root):
    from tests.core.h5sig import h5_content_signature; signature=h5_content_signature(path)
    with __import__("h5py").File(path,"r") as handle: return {name:(type(link).__name__,getattr(link,"filename",None),getattr(link,"path",None),value) for name,value in signature.items() if (name==root or name.startswith(root+"/")) for link in (handle.get(name,getlink=True),)}
def _audit(path):
    with __import__("h5py").File(path,"r") as handle: return json.loads(handle["entry/reduction/config/dimension_replacement_2d"].asstr()[()])


def test_reintegration_gi_identity_decodes_exact_current_and_historical_outer_maps():
    from xrd_tools.corrections.grazing import (
        GI_EXIT_ANGLE_CONVENTION,
        LEGACY_GI_EXIT_ANGLE_CONVENTION,
    )
    from xrd_tools.reduction import reintegrate as module

    legacy_gi = {
        "enabled": True,
        "incidence_motor": "Manual",
        "resolved_motor": "Manual",
        "th_val": 0.3,
        "sample_orientation": 4,
        "tilt_angle": 0.0,
        "mode_1d": "q_oop",
        "mode_2d": "qip_qoop",
    }
    assets = {
        "poni_values": None,
        "poni_detector_config_json": None,
        "poni_sha256": None,
        "mask_sha256": None,
    }

    def persisted(gi):
        outer = {
            "schema_version": 1,
            "generation": 1,
            "fingerprint": "f" * 64,
            "source": None,
            "processing_mode": "Int 2D",
            "output_mode": "Overwrite",
            "live_mode": False,
            "batch_mode": False,
            "max_cores": 1,
            "gi": dict(gi),
            "threshold": {
                "apply_threshold": False,
                "threshold_min": None,
                "threshold_max": None,
                "mask_saturation": False,
            },
            "poni_file": "",
            "poni_values": None,
            "mask_file": "",
            "project_root": "",
            "save_path": "",
            "bai_1d_args": {},
            "bai_2d_args": {},
            "run_options": {},
        }
        outer["scientific_signature"] = {
            **copy.deepcopy(outer),
            "accepted_scientific_assets": dict(assets),
        }
        return outer

    historical_run = persisted(legacy_gi)
    historical = module._validated_shared_science(historical_run)
    assert historical["gi"]["gi_exit_angle_convention"] == (
        LEGACY_GI_EXIT_ANGLE_CONVENTION
    )
    assert historical_run["gi"] == legacy_gi
    assert len(historical_run["gi"]) == 8

    current_run = persisted({
        **legacy_gi,
        "gi_exit_angle_convention": GI_EXIT_ANGLE_CONVENTION,
    })
    current = module._validated_shared_science(current_run)
    assert current["gi"]["gi_exit_angle_convention"] == (
        GI_EXIT_ANGLE_CONVENTION
    )
    assert len(current_run["gi"]) == 9

    selected = {
        "version": 1,
        "dimension": "1d",
        "bai_args": {
            "npt": 1000,
            "unit": "qoop_A^-1",
            "method": "csr",
            "radial_range": None,
            "azimuth_range": None,
            "gi_method_1d": "cython",
        },
        "gi_mode": "q_oop",
    }
    module._validate_science(selected, historical, "1d")
    module._validate_science(selected, current, "1d")
    assert module._digest(historical) != module._digest(current)

    with pytest.raises(ValueError, match="persisted GI is malformed"):
        module._validated_shared_science(persisted({
            **legacy_gi,
            "third_schema": True,
        }))

def test_ordinary_output_routes_group_target_through_one_borrowed_lock(tmp_path):
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    target = tmp_path / "canonical.nxs"; target.write_bytes(b"x")
    alias = tmp_path / "alias.nxs"; alias.symlink_to(target)
    assert dynamic_output._target_key(alias) == dynamic_output._target_key(target)
    adapter = dynamic_output.DynamicOutputAdapter(SimpleNamespace())
    def prepared(shown):
        item = SimpleNamespace(target=shown, group=SimpleNamespace(target=target))
        value = dynamic_output._PreparedAdmission(adapter, object(), object(), item, object(), object(), threading.Event(), (None,) * 5, (None,) * 5, None)
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
        "api_version", "target", "entry", "source_root", "expected_target_snapshot", "dimension",
        "labels", "detector_shape", "native_dtype", "selected_plan",
        "requested_shared_science", "gi_bootstrap_incidence",
        "retained_mask_bytes", "mask_decode_bytes", "session_policy",
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
        "target", "entry", "dimension", "preparation", "source_root", "expected_target_snapshot",
        "expected_terminal_identity", "expected_labels", "cancel_token",
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
    terminal = StreamTerminal(
        "/detached", 1, "d" * 64, 1, 1, 1, 1, 1,
    )
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
    execution = {}
    fact = {
        "label": 7, "path": "/raw", "frame_index": 0,
        "snapshot": {"mtime_ns": 1}, "metadata": {}, "geometry": {},
        "background_dependency": None, "source_execution": execution,
        "append_lineage": None, "source_base": "/",
    }
    monkeypatch.setattr(record_writer, "_decode_replacement_fact", lambda *_a, **_k: trace.append("detach") or fact)
    marker = object(); monkeypatch.setattr(module, "_load_fact", lambda *_a, **_k: trace.append("raw") or (Path("/raw"), marker, None, None))
    shared = {"background": {"version": 1, "mode": "None"}, "gi": {"enabled": False, "resolved_motor": "Manual"}, "geometry": None}
    allocation = SimpleNamespace(reduction_inflight=1)
    source_plan = SimpleNamespace(requested_shared_science=shared, labels=(7,), detector_shape=(2, 2), native_dtype="<u2", resource_allocation=allocation)
    topology = SimpleNamespace(
        execution=execution, lineage=None, source_base="/",
    )
    frame = Frame(7); source = module._ReintegrateFrameSource(source_plan, topology=topology); source._frames = {7: frame}; source.bind_allocation(allocation); source.bind_fact_reader(writer._detach_replacement_fact)
    assert source.prepare(frame)[0] is marker and trace == ["lock-enter", "detach", "lock-exit", "raw"]
    runtime_source = inspect.getsource(module._ExecutionRuntime.run)
    assert (
        runtime_source.index("self.source.prepare(frame)")
        < runtime_source.index("_drain_reintegration_engine(")
        and "with " not in runtime_source
    )
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


def test_reintegrate_rolls_exact_settlement_before_one_terminal_drain(
    tmp_path, monkeypatch,
):
    import h5py
    import numpy as np

    from tests.core.test_vnext_p34_existing_replacement import _r1
    from xrd_tools.io.record_writer import NexusRecordWriter
    from xrd_tools.reduction import core
    from xrd_tools.reduction import reintegrate as module

    labels = tuple(range(12))
    seeded = _seed_existing(tmp_path, labels=labels, name="chunked")
    preparation = copy.deepcopy(seeded.preparation)
    preparation["resource_policy"]["requests"] = {
        "workers": 4,
        "reduction_inflight": 8,
    }
    plan = module.ReintegratePlan.from_artifact(
        seeded.target,
        entry="entry",
        dimension="1d",
        preparation=preparation,
    )
    assert (
        plan.resource_allocation.workers,
        plan.resource_allocation.reduction_inflight,
    ) == (4, 8)

    sources = []
    prepare_snapshots = []
    drain_snapshots = []
    cleared = []
    receipt_sizes = []
    receipt_threads = []
    batch_sizes = []
    release = threading.Event()
    owner_ident = threading.get_ident()
    source_init = module._ReintegrateFrameSource.__init__
    source_prepare = module._ReintegrateFrameSource.prepare
    source_clear = module._ReintegrateFrameSource._clear_label_locked
    source_enqueue = module._ReintegrateFrameSource.enqueue_settled_batch
    engine_drain = core.ReductionSession.drain
    writer_batch = NexusRecordWriter.write_batch

    def initialize(owner, *args, **kwargs):
        source_init(owner, *args, **kwargs)
        sources.append(owner)

    def prepare(owner, frame):
        value = source_prepare(owner, frame)
        active = owner.jit_labels
        prepare_snapshots.append(active)
        if len(active) == 8:
            release.set()
        return value

    def clear(owner, label):
        active = owner.jit_labels
        if int(label) in active:
            cleared.append((int(label), active, threading.get_ident()))
        return source_clear(owner, label)

    def enqueue(owner, receipt):
        receipt_sizes.append(len(receipt.attempts))
        receipt_threads.append(threading.get_ident())
        return source_enqueue(owner, receipt)

    def drain(session, *args, **kwargs):
        assert release.is_set()
        assert len(prepare_snapshots) == len(labels)
        active = session.source.jit_labels
        value = engine_drain(session, *args, **kwargs)
        drain_snapshots.append((active, session.source.jit_labels))
        return value

    def write_batch(owner, records):
        records = tuple(records)
        batch_sizes.append(len(records))
        return writer_batch(owner, records)

    monkeypatch.setattr(module._ReintegrateFrameSource, "__init__", initialize)
    monkeypatch.setattr(module._ReintegrateFrameSource, "prepare", prepare)
    monkeypatch.setattr(module._ReintegrateFrameSource, "_clear_label_locked", clear)
    monkeypatch.setattr(
        module._ReintegrateFrameSource, "enqueue_settled_batch", enqueue,
    )
    monkeypatch.setattr(core.ReductionSession, "drain", drain)
    monkeypatch.setattr(NexusRecordWriter, "write_batch", write_batch)
    _stub_integrators(monkeypatch)
    integrate_1d = core.integrate_1d

    def gated_integrate(*args, **kwargs):
        assert release.wait(10)
        return integrate_1d(*args, **kwargs)

    monkeypatch.setattr(core, "integrate_1d", gated_integrate)

    result = module.run_reintegrate(plan)
    assert result.disposition == "COMMITTED"
    assert result.committed_labels == labels
    assert len(prepare_snapshots) == len(labels)
    assert max(map(len, prepare_snapshots)) == 8
    assert all(len(active) <= 8 for active in prepare_snapshots)
    assert len(drain_snapshots) == 1
    assert tuple(label for label, _active, _thread in cleared) == labels
    assert all(label in active for label, active, _thread in cleared)
    assert {thread for _label, _active, thread in cleared} == {owner_ident}
    assert batch_sizes[:2] == [4, 4]
    assert sum(batch_sizes) == len(labels)
    assert all(1 <= size <= 4 for size in batch_sizes)
    assert receipt_sizes == batch_sizes
    assert receipt_threads and set(receipt_threads) != {owner_ident}
    assert len(sources) == 1
    assert sources[0]._jit_high_water == 8
    assert sources[0]._receipt_high_water <= 8
    assert sources[0]._receipt_attempt_high_water <= 8
    assert sources[0].jit_roots == ()
    assert sources[0]._frames == {}
    with h5py.File(seeded.target, "r") as handle:
        group = handle["entry/integrated_1d"]
        np.testing.assert_array_equal(group["frame_index"][()], labels)
        np.testing.assert_allclose(
            group["intensity"][()],
            np.stack([_r1(label + 100).intensity for label in labels]),
        )


def test_reintegrate_settlement_receipt_is_atomic_and_attempt_exact():
    from xrd_tools.reduction import reintegrate as module
    from xrd_tools.reduction.core import Frame
    from xrd_tools.session import (
        DynamicAttemptToken,
        DynamicBatchSettlementReceipt,
        DynamicFrameIdentity,
    )

    shared = {
        "background": {"version": 1, "mode": "None"},
        "gi": {"enabled": False, "resolved_motor": "Manual"},
        "geometry": None,
    }
    allocation = SimpleNamespace(reduction_inflight=2)
    plan = SimpleNamespace(
        requested_shared_science=shared,
        selected_plan=None,
        labels=(1, 2),
        resource_allocation=allocation,
        operation_identity="operation",
    )
    source = module._ReintegrateFrameSource(plan)
    source.bind_allocation(allocation)
    frames = {1: Frame(1, metadata={"kept": 1}),
              2: Frame(2, metadata={"kept": 2})}
    first = DynamicAttemptToken(
        DynamicFrameIdentity("operation", 1), 1, 1, 11,
    )
    second = DynamicAttemptToken(
        DynamicFrameIdentity("operation", 2), 1, 1, 12,
    )
    foreign_second = DynamicAttemptToken(
        DynamicFrameIdentity("operation", 2), 1, 2, 12,
    )
    source._frames = frames
    with source._jit_condition:
        source._jit.update({1: {"root": object()}, 2: {"root": object()}})
        source._attempts.update({1: first, 2: second})
    source.enqueue_settled_batch(DynamicBatchSettlementReceipt(
        (first, foreign_second),
    ))

    with pytest.raises(RuntimeError, match="stale or foreign"):
        source.consume_settled()
    assert source.jit_labels == (1, 2)
    assert source.pending_settlement_count == 1
    assert frames[1].metadata == {"kept": 1}
    assert frames[2].metadata == {"kept": 2}
    source.clear_jit()
    assert source.jit_labels == ()
    assert source.pending_settlement_count == 0


@pytest.mark.parametrize(
    ("inflight", "batch"), ((1, 1), (2, 2), (3, 3), (4, 4), (8, 4)),
)
def test_reintegrate_writer_batch_tracks_low_inflight_grants(inflight, batch):
    from xrd_tools.reduction import reintegrate as module

    allocation = SimpleNamespace(reduction_inflight=inflight)
    assert module._replacement_writer_batch_size(allocation) == batch


def test_reintegrate_capacity_wait_surfaces_sticky_failure_and_cancel():
    from xrd_tools.reduction import reintegrate as module

    shared = {
        "background": {"version": 1, "mode": "None"},
        "gi": {"enabled": False, "resolved_motor": "Manual"},
        "geometry": None,
    }

    def source_for(token=None):
        allocation = SimpleNamespace(reduction_inflight=1)
        plan = SimpleNamespace(
            requested_shared_science=shared,
            selected_plan=None,
            labels=(1,),
            resource_allocation=allocation,
            operation_identity="operation",
        )
        source = module._ReintegrateFrameSource(plan, token=token)
        source.bind_allocation(allocation)
        with source._jit_condition:
            source._jit[1] = {"root": object()}
        return source

    failure = RuntimeError("sticky writer failure")
    failed = source_for()
    failed.bind_failure_probe(lambda: failure)
    with pytest.raises(RuntimeError, match="sticky writer failure") as caught:
        failed.wait_for_capacity()
    assert caught.value is failure

    dead_writer = RuntimeError("writer died without recording a failure")
    dead_engine = SimpleNamespace(
        _current_failure=lambda: None,
        _stream_started=True,
        _writer_thread=SimpleNamespace(is_alive=lambda: False),
        drain=lambda **_kwargs: pytest.fail("dead writer must precede drain"),
    )
    dead = source_for()
    dead.bind_failure_probe(
        lambda: module._replacement_engine_failure(dead_engine, dead_writer)
    )
    with pytest.raises(RuntimeError, match="writer died") as caught:
        dead.wait_for_capacity()
    assert caught.value is dead_writer
    with pytest.raises(RuntimeError, match="writer died") as caught:
        module._drain_reintegration_engine(dead_engine, dead, 0.1)
    assert caught.value is dead_writer

    drain_calls = []
    alive_engine = SimpleNamespace(
        drain=lambda **kwargs: drain_calls.append(kwargs) or False,
    )
    bounded = source_for()
    bounded.bind_failure_probe(lambda: None)
    with pytest.raises(TimeoutError, match="bounded timeout"):
        module._drain_reintegration_engine(alive_engine, bounded, 0.02)
    assert drain_calls

    cancelled = threading.Event()
    waiting = source_for(cancelled)
    timer = threading.Timer(0.02, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(module.ReintegrateCancelled):
            waiting.wait_for_capacity()
    finally:
        timer.join()
    assert time.monotonic() - started < 0.5


def test_reintegrate_cancelled_stalled_worker_returns_pending_bounded(
    tmp_path, monkeypatch,
):
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import core
    from xrd_tools.reduction import reintegrate as module

    labels = tuple(range(12))
    seeded = _seed_existing(tmp_path, labels=labels, name="stalled-worker")
    preparation = copy.deepcopy(seeded.preparation)
    preparation["resource_policy"]["requests"] = {
        "workers": 4,
        "reduction_inflight": 8,
    }
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d", preparation=preparation,
    )
    expected = capture_target_snapshot(seeded.target)
    _stub_integrators(monkeypatch)
    integrate_1d = core.integrate_1d
    release = threading.Event()
    entered = threading.Event()

    def stalled(*args, **kwargs):
        entered.set()
        assert release.wait(5.0)
        return integrate_1d(*args, **kwargs)

    monkeypatch.setattr(core, "integrate_1d", stalled)
    monkeypatch.setattr(module, "_TERMINAL_DRAIN_TIMEOUT_SECONDS", 0.1)
    cancelled = threading.Event()
    def cancel_stalled():
        assert entered.wait(2.0)
        time.sleep(0.02)
        cancelled.set()
    timer = threading.Thread(target=cancel_stalled)
    runner = module.ReintegrateRunner(plan, cancel_token=cancelled)
    timer.start()
    started = time.monotonic()
    try:
        pending = runner.run()
        assert pending.disposition == "SETTLEMENT_PENDING"
        assert time.monotonic() - started < 1.0
        assert runner._runtime._has_custody()
    finally:
        release.set()
        timer.join()

    engine = runner._runtime.session._session
    deadline = time.monotonic() + 2.0
    while engine._writer_thread is not None and engine._writer_thread.is_alive():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert engine.sink_terminal_safe
    assert runner._runtime.session._session is engine
    assert runner._runtime.session._dynamic_frozen_result is not None
    with pytest.raises(TimeoutError, match="writer thread did not exit"):
        runner.finish_current()
    assert capture_target_snapshot(seeded.target) == expected
    runner.close()


def test_reintegrate_successful_drain_renews_join_budget_after_slow_validation(
    tmp_path, monkeypatch,
):
    from xrd_tools.reduction import reintegrate as module
    from xrd_tools.session.scan_session import ScanSession

    seeded = _seed_existing(tmp_path, labels=(0, 1, 2), name="join-budget")
    plan = module.ReintegratePlan.from_artifact(
        seeded.target, entry="entry", dimension="1d",
        preparation=seeded.preparation,
    )
    _stub_integrators(monkeypatch)
    monkeypatch.setattr(module, "_TERMINAL_DRAIN_TIMEOUT_SECONDS", 0.05)
    validate = module._ReintegrateFrameSource.validate_terminal_topology
    finish = ScanSession.finish
    observed = []

    def slow_validate(owner):
        value = validate(owner)
        time.sleep(0.08)
        return value

    def capture_finish(owner, *args, **kwargs):
        observed.append(kwargs.get("join_timeout"))
        return finish(owner, *args, **kwargs)

    monkeypatch.setattr(
        module._ReintegrateFrameSource,
        "validate_terminal_topology",
        slow_validate,
    )
    monkeypatch.setattr(ScanSession, "finish", capture_finish)

    result = module.run_reintegrate(plan)
    assert result.disposition == "COMMITTED"
    assert len(observed) == 1
    assert type(observed[0]) is float and observed[0] > 0.02


def test_direct_hdf_window_is_exact_two_handle_mask_bounded_and_retryable(
    tmp_path, monkeypatch,
):
    import h5py
    import hdf5plugin
    import numpy as np

    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState,
        SourceExecutionStamp,
        SourceFileState,
    )
    from xrd_tools.io.nexus import NexusImageStack
    from xrd_tools.io.output_transaction import capture_target_snapshot
    from xrd_tools.reduction import reintegrate as module

    shape = (17, 17)
    members = []
    expected = []
    cursor = 0
    for member_index in range(2):
        path = tmp_path / f"member-{member_index}.h5"
        frames = np.stack([
            np.full(shape, member_index * 10 + offset, dtype=np.uint32)
            for offset in range(2)
        ])
        with h5py.File(path, "w") as handle:
            handle.create_group("entry/data").create_dataset(
                "data", data=frames, chunks=(1, *shape),
                **hdf5plugin.Bitshuffle(cname="lz4"),
            )
        members.append(ExternalSourceState(
            SourceFileState.capture(path), "/entry/data/data",
            cursor, cursor + len(frames), member_index,
        ))
        expected.extend(frames)
        cursor += len(frames)
    master = tmp_path / "master.h5"
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry/data")
        for index, member in enumerate(members, 1):
            data[f"data_{index:06d}"] = h5py.ExternalLink(
                Path(member.file.path).name, member.dataset,
            )
    master_state = SourceFileState.capture(master)
    execution = SourceExecutionStamp(
        master_state, "nexus_hdf5", cursor, 0,
        external_members=tuple(members),
    ).as_dict()
    fact = {
        "label": 0,
        "path": master_state.path,
        "frame_index": 0,
        "source_base": "",
        "snapshot": {
            "adapter_id": "nexus_hdf5",
            "size": master_state.size,
            "mtime_ns": master_state.mtime_ns,
            "frame_count": cursor,
            "dataset_path": "/entry/data/data_000001",
            "self_contained": False,
        },
        "source_execution": execution,
        "append_lineage": None,
        "metadata": {},
        "geometry": {},
        "background_dependency": None,
    }
    topology = module._admit_source_topology(
        fact, full_inventory=True, selected_labels=tuple(range(cursor)),
    )
    target = tmp_path / "processed.nexus"
    target.write_bytes(b"distinct replacement target")
    target_snapshot = capture_target_snapshot(target)
    frame_bytes = int(np.prod(shape)) * np.dtype("<u4").itemsize
    allocation = SimpleNamespace(owner_block_bytes=3 * frame_bytes)
    file_count = lambda: h5py.h5f.get_obj_count(
        h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE)
    baseline = file_count()

    owner = module._ReintegrateDirectHdfWindow(
        topology, allocation, shape, "<u4", target_snapshot, 0,
    )
    owner.open()
    assert file_count() == baseline + 1
    for label in (0, 1, 2, 3):
        route = topology.frame_routes[label].hdf
        np.testing.assert_array_equal(owner.read(route, label), expected[label])
        assert file_count() <= baseline + 2
    errors = []
    def cross_thread():
        try:
            owner.validate()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=cross_thread)
    thread.start(); thread.join()
    assert len(errors) == 1
    assert "owner-thread-only" in str(errors[0])

    real_close = NexusImageStack.close
    close_calls = []
    active_view = owner._active[2]
    def flaky_close(view):
        if view is active_view and not close_calls:
            close_calls.append("failed")
            raise OSError("injected direct-reader close failure")
        return real_close(view)
    monkeypatch.setattr(NexusImageStack, "close", flaky_close)
    with pytest.raises(OSError, match="injected direct-reader close failure"):
        owner.close(validate=True)
    assert owner._active is not None and owner._master is not None
    owner.close(validate=True)
    assert owner._closed and owner._active is owner._master is None
    assert file_count() == baseline

    constrained = module._ReintegrateDirectHdfWindow(
        topology, SimpleNamespace(owner_block_bytes=2 * frame_bytes),
        shape, "<u4", target_snapshot, 1,
    )
    constrained.open()
    np.testing.assert_array_equal(
        constrained.read(topology.frame_routes[0].hdf, 0), expected[0],
    )
    assert constrained._active[-1] is None
    constrained.close(validate=True)
    assert file_count() == baseline

    conflicting = replace(
        topology,
        frame_routes=__import__("types").MappingProxyType({
            0: topology.frame_routes[0],
            2: topology.frame_routes[2],
            4: topology.frame_routes[0]._replace(
                hdf=topology.frame_routes[0].hdf._replace(start=2, stop=4),
            ),
        }),
    )
    conflict_owner = module._ReintegrateDirectHdfWindow(
        conflicting, allocation, shape, "<u4", target_snapshot, 0,
    )
    with pytest.raises(
        ValueError,
        match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED",
    ):
        conflict_owner.open()
    assert file_count() == baseline
    conflict_owner.close(validate=False)

    provisional_owner = module._ReintegrateDirectHdfWindow(
        topology, SimpleNamespace(owner_block_bytes=frame_bytes),
        shape, "<u4", target_snapshot, 1,
    )
    provisional_owner.open()
    provisional_failures = []
    def twice_failing_close(view):
        if len(provisional_failures) < 2:
            provisional_failures.append(view)
            raise OSError("injected provisional close failure")
        return real_close(view)
    monkeypatch.setattr(NexusImageStack, "close", twice_failing_close)
    with pytest.raises(OSError, match="injected provisional close failure"):
        provisional_owner.read(topology.frame_routes[0].hdf, 0)
    assert provisional_owner._provisional is not None
    assert provisional_owner._master is not None
    monkeypatch.setattr(NexusImageStack, "close", real_close)
    provisional_owner.close(validate=False)
    assert provisional_owner._closed and file_count() == baseline

    same_inode = module._ReintegrateDirectHdfWindow(
        topology, allocation, shape, "<u4",
        capture_target_snapshot(master), 0,
    )
    with pytest.raises(
        ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED",
    ):
        same_inode.open()
    assert file_count() == baseline
    same_inode.close(validate=False)

    raw_refusal = module._ReintegrateDirectHdfWindow(
        topology, allocation, shape, "<u4", target_snapshot, 0,
    )
    raw_refusal.open()
    real_direct_read = NexusImageStack.read_eiger_direct_chunk
    monkeypatch.setattr(
        NexusImageStack, "read_eiger_direct_chunk",
        lambda *_args, **_kwargs: (None, "injected pre-read refusal"),
    )
    np.testing.assert_array_equal(
        raw_refusal.read(topology.frame_routes[0].hdf, 0), expected[0],
    )
    assert file_count() <= baseline + 2
    raw_refusal.close(validate=True)
    assert file_count() == baseline
    monkeypatch.setattr(
        NexusImageStack, "read_eiger_direct_chunk", real_direct_read,
    )

    token = threading.Event()
    cancelled = module._ReintegrateDirectHdfWindow(
        topology, allocation, shape, "<u4", target_snapshot, 0, token,
    )
    cancelled.open()
    def cancel_after_read(view, *args, **kwargs):
        value = real_direct_read(view, *args, **kwargs)
        token.set()
        return value
    monkeypatch.setattr(
        NexusImageStack, "read_eiger_direct_chunk", cancel_after_read,
    )
    with pytest.raises(module.ReintegrateCancelled):
        cancelled.read(topology.frame_routes[0].hdf, 0)
    cancelled.close(validate=False)
    assert file_count() == baseline
    monkeypatch.setattr(
        NexusImageStack, "read_eiger_direct_chunk", real_direct_read,
    )

    drifted = module._ReintegrateDirectHdfWindow(
        topology, allocation, shape, "<u4", target_snapshot, 0,
    )
    drifted.open()
    member_path = Path(members[0].file.path)
    def drift_after_read(view, *args, **kwargs):
        value = real_direct_read(view, *args, **kwargs)
        observed = member_path.stat()
        os.utime(member_path, ns=(
            observed.st_atime_ns, observed.st_mtime_ns + 1_000_000,
        ))
        return value
    monkeypatch.setattr(
        NexusImageStack, "read_eiger_direct_chunk", drift_after_read,
    )
    with pytest.raises(
        ValueError, match="REPLACEMENT_SOURCE_REVISION_CHANGED",
    ):
        drifted.read(topology.frame_routes[0].hdf, 0)
    drifted.close(validate=False)
    assert file_count() == baseline


def test_direct_hdf_construction_preserves_primary_and_retry_custody(
    monkeypatch,
):
    from xrd_tools.reduction import reintegrate as module

    allocation = object()
    plan = SimpleNamespace(
        resource_allocation=allocation,
        detector_shape=(2, 2),
        native_dtype="<u2",
        expected_target_snapshot=object(),
        retained_mask_bytes=0,
    )
    source = object.__new__(module._ReintegrateFrameSource)
    source.plan = plan
    source.token = None
    source.bound_allocation = allocation
    source._topology = object()
    source._direct_hdf = None
    source._owner_ident = threading.get_ident()
    owners = []
    class InjectedOwner:
        def __init__(self, *args, **kwargs):
            self.close_calls = 0
            owners.append(self)
        def open(self):
            raise ValueError("injected direct admission primary")
        def close(self, *, validate=False):
            self.close_calls += 1
            if self.close_calls == 1:
                raise OSError("injected direct cleanup failure")
    monkeypatch.setattr(
        module, "_ReintegrateDirectHdfWindow", InjectedOwner,
    )
    with pytest.raises(
        ValueError, match="injected direct admission primary",
    ) as caught:
        source.open_direct_hdf()
    assert "direct HDF construction cleanup" in "\n".join(
        getattr(caught.value, "__notes__", ()))
    assert source._direct_hdf is owners[0]
    source.close_direct_hdf(validate=False)
    assert source._direct_hdf is None and owners[0].close_calls == 2


def test_parent_red_stable_loaded_browse_enables_reintegrate_1d_start(tmp_path, monkeypatch, qapp):
    from xrd_tools.session.readiness import ControlAction, SectionId
    from xdart.gui.tabs.scattering.state_machine import RunPhase
    page, store, _seed, _context = _loaded_page(tmp_path, monkeypatch, qapp)
    projected = page._project_controls(store.snapshot()); actions = {a.action: a for a in projected.actions_for(SectionId.PROCESSING)}; experiment = {a.action: a for a in projected.actions_for(SectionId.EXPERIMENT)}
    assert actions[ControlAction.REINTEGRATE_1D].enabled and actions[ControlAction.REINTEGRATE_1D].label == "Reintegrate 1-D"
    assert actions[ControlAction.REINTEGRATE_2D].enabled and actions[ControlAction.REINTEGRATE_2D].label == "Reintegrate 2-D" and experiment[ControlAction.CALIBRATE].label == "Calibrate" and experiment[ControlAction.MAKE_MASK].label == "Make Mask"
    page._lifecycle._phase = RunPhase.FAILED; page._lifecycle._owners_closed = False; actions = {a.action: a for a in page._project_controls(store.snapshot()).actions_for(SectionId.PROCESSING)}
    assert not actions[ControlAction.REINTEGRATE_1D].enabled and not actions[ControlAction.REINTEGRATE_2D].enabled
    page._lifecycle._owners_closed = True; actions = {a.action: a for a in page._project_controls(store.snapshot()).actions_for(SectionId.PROCESSING)}
    assert actions[ControlAction.REINTEGRATE_1D].enabled and actions[ControlAction.REINTEGRATE_2D].enabled; page.close_workspace()


@pytest.mark.parametrize(
    ("relative_target", "selection_case", "plot_mode"),
    (
        (False, "all", "Overlay"),
        (False, "exclude-latest", "Waterfall"),
        (False, "manual-then-auto-last", "Overlay"),
        (True, "manual-history", None),
    ),
    ids=(
        "absolute-auto-last-overlay",
        "absolute-auto-last-waterfall-exclusion",
        "absolute-manual-then-auto-last-overlay",
        "relative-manual-history",
    ),
)
def test_finished_run_auto_browse_enables_reintegration_without_second_click(
    tmp_path, monkeypatch, qapp, relative_target, selection_case, plot_mode,
):
    """The published terminal artifact becomes the same authenticated Browse."""

    from tests.xdart.scattering.test_e1b2_page_command_boundaries import (
        _Executor,
        _active_page,
        _dispose,
    )
    from tests.xdart.scattering.test_e3_context_contract import _acquisition
    from xdart.gui.tabs.scattering.display_values import (
        StandardEventKind,
        StandardRunEvent,
    )
    from xdart.gui.tabs.scattering.events import CleanupStatus
    from xrd_tools.session.readiness import ControlAction, SectionId
    from xrd_tools.session.run_configuration import RunIntent

    seeded = _seed_existing(tmp_path)
    resolved_target = str(seeded.target.resolve())
    target = (
        os.path.relpath(resolved_target, Path.cwd())
        if relative_target
        else resolved_target
    )
    executor = _Executor()
    page, lifecycle, identity = _active_page(executor)
    try:
        if plot_mode is not None:
            page._preferences = replace(
                page._preferences, plot_mode=plot_mode,
            )
        configuration = RunIntent(output_mode="Overwrite").freeze()
        assert configuration.identity == (
            identity.generation, identity.fingerprint,
        )
        _, acquisition = _acquisition(
            configuration=configuration,
            identity=identity,
        )
        executor.acquisition_context = lambda candidate: (
            acquisition if candidate is identity else None
        )
        page._context_controller.adopt_acquisition(identity)
        acquisition.publication_store.catalog.resize(16)

        deltas = tuple(
            acquisition.publication_store.append_navigation(
                "terminal.run", target, label,
            )
            for label in seeded.labels
        )
        executor.events.extend(
            StandardRunEvent(
                identity,
                StandardEventKind.FRAME_READY,
                completed=index,
                total=len(deltas),
                artifact=target,
                frame_key=delta.appended,
                navigation_delta=delta,
                artifact_completed=index,
                artifact_total=len(deltas),
            )
            for index, delta in enumerate(deltas, start=1)
        )
        page._drain_executor()
        navigation = page._context_controller.navigation
        assert navigation.current.local_frame_label == seeded.labels[-1]
        assert navigation.current.artifact == target
        if selection_case != "manual-history":
            assert tuple(
                frame.local_frame_label
                for frame in navigation.selected
                if frame.artifact == target
            ) == seeded.labels

        from xdart.gui.tabs.scattering.shell_values import (
            ShellCommand,
            ShellCommandKind,
        )

        if selection_case == "exclude-latest":
            latest = deltas[-1].appended
            included = tuple(delta.appended for delta in deltas[:-1])
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SELECT_FRAME,
                frame=latest,
                frames=included,
            ))
            assert page._processed_browser.auto_last
            assert page._context_controller.navigation.current is latest
            assert page._context_controller.navigation.selected == included

        real_begin = page._context_controller.begin_browse
        browse_calls = []
        def begin_browse(artifact, **kwargs):
            browse_calls.append((artifact, kwargs))
            return real_begin(artifact, **kwargs)

        monkeypatch.setattr(
            page._context_controller,
            "begin_browse",
            begin_browse,
        )

        executor.events.append(StandardRunEvent(
            identity,
            StandardEventKind.FINISHED,
            completed=len(deltas),
            total=len(deltas),
            artifact=target,
            cleanup_status=CleanupStatus.CLEANED,
            artifact_completed=len(deltas),
            artifact_total=len(deltas),
        ))
        page._drain_executor()
        assert lifecycle.phase.value == "idle"
        assert browse_calls == [(target, {"source_root": None})]
        handoff = page._processed_browser.terminal_handoff
        assert handoff is not None
        assert handoff.request.source_path == resolved_target
        assert handoff.source_artifact == target
        assert page._context_controller.browse_pending

        historical = deltas[1].appended
        if selection_case in {"manual-history", "manual-then-auto-last"}:
            # A deliberate historical-frame choice during the asynchronous
            # terminal load must survive adoption into the persisted Browse
            # context unless Auto Last is explicitly re-enabled.
            page._handle_shell_command(ShellCommand(
                ShellCommandKind.SELECT_FRAME,
                frame=historical,
                frames=(historical,),
            ))
            assert not page._processed_browser.auto_last
            assert page._processed_browser.terminal_handoff.current_label == (
                historical.local_frame_label
            )
            assert page._processed_browser.terminal_handoff.selected_labels == (
                historical.local_frame_label,
            )
            if selection_case == "manual-then-auto-last":
                page._handle_shell_command(ShellCommand(
                    ShellCommandKind.SET_AUTO_LAST,
                    True,
                ))
                assert page._processed_browser.auto_last
                # Acquisition can follow latest immediately, but the terminal
                # remap snapshot intentionally remains the last explicit
                # choice. Settlement must apply Auto Last to the authenticated
                # Browse identities while retaining that exact membership.
                assert (
                    page._context_controller.navigation.current
                    is deltas[-1].appended
                )
                assert page._context_controller.navigation.selected == (
                    historical,
                )
                assert page._processed_browser.terminal_handoff.current_label == (
                    historical.local_frame_label
                )

        def settled():
            page._drain_executor()
            return page._context_controller.capture_loaded_browse(
                handoff.request
            )

        captured = _wait(settled)
        assert captured.target == resolved_target
        assert captured.labels == seeded.labels
        assert captured.target_snapshot.exists
        assert page._processed_browser.terminal_handoff is None
        assert not page._context_controller.browse_pending
        navigation = page._context_controller.navigation
        expected_current = (
            historical.local_frame_label
            if selection_case == "manual-history"
            else seeded.labels[-1]
        )
        expected_selected = (
            (historical.local_frame_label,)
            if selection_case in {"manual-history", "manual-then-auto-last"}
            else seeded.labels[:-1]
            if selection_case == "exclude-latest"
            else seeded.labels
        )
        assert navigation.current.local_frame_label == expected_current
        assert tuple(
            frame.local_frame_label for frame in navigation.selected
        ) == expected_selected
        if selection_case != "manual-history":
            def overlay_ready():
                page._drain_executor()
                scientific = page._last_scientific_projection
                if (
                    scientific is None
                    or scientific.heavy is None
                    or len(scientific.traces) != len(expected_selected)
                ):
                    return None
                return scientific

            scientific = _wait(overlay_ready)
            assert scientific.heavy.frame.local_frame_label == seeded.labels[-1]
            assert tuple(
                trace.frame.local_frame_label for trace in scientific.traces
            ) == expected_selected
            assert tuple(
                frame.local_frame_label
                for frame in page._shell.scientific.trace_history_keys
            ) == expected_selected
        actions = {
            action.action: action
            for action in page._project_controls(
                page._intents.snapshot()
            ).actions_for(SectionId.PROCESSING)
        }
        assert actions[ControlAction.REINTEGRATE_1D].enabled
        assert actions[ControlAction.REINTEGRATE_2D].enabled
        from xdart.gui.widgets.controls_panel import ActionButton

        qapp.processEvents()
        mounted = {
            button.spec.action: button
            for button in page._shell.controls.findChildren(ActionButton)
            if button.isEnabled()
        }
        assert mounted[ControlAction.REINTEGRATE_1D].isEnabled()
        assert mounted[ControlAction.REINTEGRATE_2D].isEnabled()
        assert "stable processed Browse artifact" not in (
            mounted[ControlAction.REINTEGRATE_1D].toolTip()
        )
    finally:
        _dispose(page, qapp)


def test_terminal_browse_rebind_reuses_exact_single_latest_across_65_frames(
    tmp_path, monkeypatch, qapp,
):
    from tests.xdart.scattering.test_e3_context_contract import (
        _acquisition, _display, _view,
    )
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus
    from xdart.gui.tabs.scattering.display_values import StandardDisplayPayload
    from xdart.gui.tabs.scattering.shell_values import ShellCommandKind
    from xdart.modules.frame_publication import FramePublication
    from xrd_tools.core import FrameRecord
    from xrd_tools.io.output_transaction import StreamTerminal

    seed = _seed_existing(tmp_path, labels=tuple(range(1, 66)))
    page, _store = _page(tmp_path, monkeypatch)
    try:
        target = str(seed.target.resolve())
        identity, template = _acquisition()
        display = _display(
            identity,
            artifact=target,
            scan_key="existing",
            value=1.0,
        )
        display.catalog.resize(65)
        owner = display.artifacts[target]
        for label in seed.labels[1:]:
            frame_view = _view(label, float(label))
            record = FrameRecord.from_view(frame_view)
            publication = FramePublication(
                frame_view,
                record=record,
                source_identity=f"{frame_view.source_path}#{label}",
                scan_key="existing",
            )
            frame = display.append_navigation(
                "existing", target, label,
            ).appended
            display.retain_frame(
                owner,
                frame,
                record,
                publication,
                source_identity=publication.source_identity,
                frame_mask_qualified=False,
            )
            if label == seed.labels[-1]:
                display.put_payload(StandardDisplayPayload(
                    0,
                    frame,
                    f"Standard · existing · frame {label}",
                    frame_view,
                ))
        acquisition = replace(
            template,
            run_scan_key="existing",
            source_path="/data/scan_1.tif",
            frame_ids=display.catalog,
            frames=display.artifacts,
            publication_store=display,
            record_store=None,
        )
        acquisition.adopt_record_store(display)

        controller = page._context_controller
        controller._runtime.adopt_acquisition(identity, acquisition)
        original = controller.navigation
        assert len(original.frames) == 65
        latest = original.frames[-1]
        assert controller.select_navigation(latest, (latest,))
        page._preferences = replace(page._preferences, plot_mode="Single")
        view = page._shell.scientific

        def complete_single():
            page._drain_executor()
            page._refresh_shell()
            projection = page._last_scientific_projection
            return (
                projection
                if projection is not None
                and view.navigation_frame_keys == original.frames
                and view.navigation_current_key is latest
                and view.navigation_selected_keys == (latest,)
                and view.trace_history_keys == (latest,)
                and len(projection.traces) == 1
                and projection.heavy is not None
                and projection.heavy.frame is latest
                else None
            )

        before = _wait(complete_single)
        # This oracle drives the private terminal-paint transaction directly.
        # Leave no page-owned drain timer able to start a competing ordinary
        # paint for the same presentation while that transaction is staged.
        page._run_timer.stop()
        before_history = view.trace_history_projections
        before_items = tuple(view.curve.listDataItems())
        assert len(before_history) == len(before_items) == 1
        before_axis = before.traces[0].axis.values
        before_intensity = before.traces[0].intensity
        before_raw = view.raw.canvas.displayed_image
        before_cake = view.cake.canvas.displayed_image
        before_raw_widget = view.raw
        before_cake_widget = view.cake
        before_bottom = view.bottom_stack.currentWidget()

        request = controller.begin_browse(
            target,
            source_root=str(seed.target.parent.resolve()),
            terminal_commit_identity=seed.terminal.commit_identity,
        )
        outcome = _wait(controller.poll_browse)
        assert outcome.request is request
        assert outcome.status is BrowseLoadStatus.READY
        context = controller.browse_context
        assert context is not None and context.loaded
        commit_identity = request.terminal_commit_identity
        assert type(commit_identity) is StreamTerminal
        expected_handoff = TerminalBrowseHandoff(
            request,
            latest.run_identity,
            latest.artifact,
            latest.local_frame_label,
            (latest.local_frame_label,),
            commit_identity,
        )
        handoff = _begin_terminal_handoff(page, expected_handoff)
        capture = controller.capture_loaded_browse(request)
        assert capture is not None and capture.context is context
        assert page._settle_terminal_browse(outcome)
        rebound = controller.navigation
        clones = rebound.frames
        assert len(clones) == len(original.frames) == 65
        assert tuple(frame.local_frame_label for frame in clones) == seed.labels
        assert all(
            old is not new
            for old, new in zip(original.frames, clones, strict=True)
        )
        assert rebound.current is clones[-1]
        assert rebound.selected == (clones[-1],)
        assert page._terminal_scientific_matches(handoff, rebound)
        presentation = page._processed_browser.terminal_presentation
        assert presentation is not None
        preference_before = page._preferences

        controller._runtime._committed_trace_scope = ("stale",)
        controller._runtime._committed_trace_selection = (clones[-1],)
        controller._runtime._pending_trace_projection = object()
        monkeypatch.setattr(
            controller,
            "project_navigation",
            lambda **_kwargs: pytest.fail(
                "terminal Single rebind rebuilt numeric projections"
            ),
        )
        monkeypatch.setattr(
            view,
            "_render_traces",
            lambda *_args, **_kwargs: pytest.fail(
                "terminal Single rebind repainted traces"
            ),
        )
        monkeypatch.setattr(
            before_items[0],
            "setData",
            lambda *_args, **_kwargs: pytest.fail(
                "terminal Single rebind replaced trace arrays"
            ),
        )
        for renderer, description in (
            (view.raw, "raw"),
            (view.cake, "cake"),
            (view.waterfall, "waterfall"),
        ):
            monkeypatch.setattr(
                renderer,
                "render",
                lambda *_args, _description=description, **_kwargs: pytest.fail(
                    f"terminal Single rebind repainted {_description}"
                ),
            )

        paint = page._begin_processed_terminal_paint(
            presentation, reuse_science=True,
        )
        assert paint is not None
        assert paint.mode is TerminalPaintMode.REBIND
        revision = page._shell_revision
        page._refresh_shell(
            preserve_scientific=True,
            skip_scientific_projection=True,
            rebind_scientific_navigation=True,
            terminal_paint=paint,
        )
        page._complete_processed_terminal_paint(
            paint, applied=page._shell_revision > revision,
        )

        after = page._last_scientific_projection
        assert after is not None
        assert view.navigation_frame_keys == clones
        assert view.navigation_current_key is clones[-1]
        assert view.navigation_selected_keys == (clones[-1],)
        assert view.trace_history_keys == (clones[-1],)
        assert len(view.trace_history_projections) == 1
        assert len(after.traces) == 1
        assert after.traces[0].frame is clones[-1]
        assert after.traces[0].axis.values is before_axis
        assert after.traces[0].intensity is before_intensity
        assert after.heavy is not None and after.heavy.frame is clones[-1]
        assert after.heavy_available == frozenset((clones[-1],))
        assert view.heavy_available_keys == after.heavy_available
        assert all(
            any(frame is owned for owned in clones)
            for frame in view.heavy_available_keys
        )
        assert tuple(view.curve.listDataItems()) == before_items
        assert view.raw is before_raw_widget
        assert view.cake is before_cake_widget
        assert view.raw.canvas.displayed_image is before_raw
        assert view.cake.canvas.displayed_image is before_cake
        assert view.bottom_stack.currentWidget() is before_bottom
        assert replace(
            page._preferences,
            detector_available=preference_before.detector_available,
            detector_pending=preference_before.detector_pending,
            detector_diagnostic=preference_before.detector_diagnostic,
        ) == preference_before
        assert controller._runtime._committed_trace_scope is None
        assert controller._runtime._committed_trace_selection == ()
        assert controller._runtime._pending_trace_projection is None
        assert not page._scientific_repaint_pending
        commands = []
        view.commandRequested.disconnect()
        view.commandRequested.connect(commands.append)
        view._frame_selected(view.frame_selector.currentIndex())
        assert len(commands) == 1
        assert commands[0].kind is ShellCommandKind.SELECT_FRAME
        assert commands[0].frame is clones[-1]
        assert commands[0].frames == (clones[-1],)
    finally:
        page.close_workspace()


def test_terminal_browse_single_rebind_refuses_multi_selected_or_foreign(
    tmp_path, monkeypatch, qapp,
):
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadOutcome, BrowseLoadStatus,
    )
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
    from xrd_tools.io.output_transaction import StreamTerminal

    page, _store, _seeded, context = _loaded_page(
        tmp_path, monkeypatch, qapp, terminal_browse=True,
    )
    try:
        controller = page._context_controller
        original = controller.navigation
        latest = original.frames[-1]
        assert controller.select_navigation(latest, (latest,))
        page._preferences = replace(page._preferences, plot_mode="Single")
        view = page._shell.scientific

        def complete_single():
            page._refresh_shell()
            projection = page._last_scientific_projection
            return (
                projection
                if projection is not None
                and view.trace_history_keys == (latest,)
                and projection.heavy is not None
                else None
            )

        before = _wait(complete_single)
        clones = tuple(
            DisplayFrameKey(
                frame.run_identity,
                frame.source_scan,
                frame.artifact,
                frame.local_frame_label,
                frame.work_ordinal,
            )
            for frame in original.frames
        )
        valid = FrameNavigationProjection(
            clones, clones[-1], (clones[-1],),
        )
        request = context.load_request
        commit_identity = request.terminal_commit_identity
        assert type(commit_identity) is StreamTerminal
        expected_handoff = TerminalBrowseHandoff(
            request,
            latest.run_identity,
            latest.artifact,
            latest.local_frame_label,
            (latest.local_frame_label,),
            commit_identity,
        )
        handoff = _begin_terminal_handoff(page, expected_handoff)
        assert page._terminal_scientific_matches(handoff, valid)

        foreign_identity = RunIdentity(
            latest.run_identity.generation,
            latest.run_identity.fingerprint,
        )
        foreign = tuple(
            DisplayFrameKey(
                foreign_identity,
                frame.source_scan,
                frame.artifact,
                frame.local_frame_label,
                frame.work_ordinal,
            )
            for frame in clones
        )
        assert not page._terminal_scientific_matches(
            handoff,
            FrameNavigationProjection(foreign, foreign[-1], (foreign[-1],)),
        )

        page._last_scientific_projection = replace(
            before,
            heavy=replace(before.heavy, frame=original.frames[0]),
        )
        assert not page._terminal_scientific_matches(handoff, valid)
        page._last_scientific_projection = replace(
            before,
            traces=(replace(before.traces[0], frame=original.frames[0]),),
        )
        assert not page._terminal_scientific_matches(handoff, valid)
        page._last_scientific_projection = before

        multiple = FrameNavigationProjection(
            clones, clones[-1], (clones[-2], clones[-1]),
        )
        assert page._processed_browser.update_terminal_selection(
            request,
            run_identity=handoff.run_identity,
            source_artifact=handoff.source_artifact,
            current_label=handoff.current_label,
            selected_labels=(
                clones[-2].local_frame_label,
                clones[-1].local_frame_label,
            ),
        )
        controller._runtime._set_browse_navigation(multiple)
        capture = controller.capture_loaded_browse(request)
        assert capture is not None and capture.context is context
        assert not page._settle_terminal_browse(BrowseLoadOutcome(
            request, BrowseLoadStatus.READY,
        ))
        presentation = page._processed_browser.terminal_presentation
        assert presentation is not None
        paint = page._begin_processed_terminal_paint(
            presentation, reuse_science=True,
        )
        assert paint is not None
        assert paint.mode is TerminalPaintMode.REPAINT
        projected = []
        original_project = controller.project_browse_1d_cache

        def project_browse_1d_cache(**kwargs):
            result = original_project(**kwargs)
            projected.append(result)
            return result

        monkeypatch.setattr(
            controller, "project_browse_1d_cache", project_browse_1d_cache,
        )
        revision = page._shell_revision
        page._refresh_shell(terminal_paint=paint)
        page._complete_processed_terminal_paint(
            paint, applied=page._shell_revision > revision,
        )
        assert projected
    finally:
        page.close_workspace()


def test_terminal_waterfall_match_requires_exact_painted_source_receipt(
    tmp_path, monkeypatch, qapp,
):
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
    from xrd_tools.io.output_transaction import StreamTerminal

    page, _store, _seeded, context = _loaded_page(
        tmp_path, monkeypatch, qapp, terminal_browse=True,
    )
    try:
        controller = page._context_controller
        latest = controller.navigation.frames[-1]
        assert controller.select_navigation(latest, (latest,))
        page._preferences = replace(page._preferences, plot_mode="Single")
        view = page._shell.scientific

        def complete_template():
            page._drain_executor()
            page._refresh_shell()
            projection = page._last_scientific_projection
            return (
                projection
                if (
                    projection is not None
                    and projection.traces
                    and projection.heavy is not None
                )
                else None
            )

        template = _wait(complete_template)
        run_identity = latest.run_identity
        frames = tuple(
            DisplayFrameKey(
                run_identity,
                latest.source_scan,
                latest.artifact,
                1_000 + index,
                1_000 + index,
            )
            for index in range(1, 22)
        )
        navigation = FrameNavigationProjection(
            frames, frames[-1], frames,
        )
        options = replace(
            page._preferences.plot_options,
            waterfall_start=2,
            waterfall_stop=20,
            waterfall_step=3,
        )
        scientific = replace(
            template,
            traces=tuple(
                replace(template.traces[0], frame=frame)
                for frame in frames
            ),
            heavy=replace(template.heavy, frame=frames[-1]),
            heavy_available=frozenset(frames),
            plot_mode="Waterfall",
            plot_options=options,
            slice_pins=(),
            pinned_traces=(),
            retain_display=False,
            live_update=False,
            browse_trace_snapshot=None,
        )
        page._preferences = replace(
            page._preferences,
            plot_mode="Waterfall",
            plot_options=options,
        )
        page._last_scientific_projection = scientific
        view.reconcile(
            scientific,
            navigation,
            completed=len(frames),
            total=len(frames),
            detail="Ready",
        )
        assert view.bottom_waterfall_active
        assert len(view.trace_history_keys) == len(frames)
        assert all(
            painted is expected
            for painted, expected in zip(
                view.trace_history_keys, frames, strict=True,
            )
        )
        full_waterfall_source = view._waterfall_source_keys
        assert len(full_waterfall_source) == 7

        clones = tuple(
            DisplayFrameKey(
                frame.run_identity,
                frame.source_scan,
                frame.artifact,
                frame.local_frame_label,
                frame.work_ordinal,
            )
            for frame in frames
        )
        rebound = FrameNavigationProjection(
            clones, clones[-1], clones,
        )
        request = context.load_request
        commit_identity = request.terminal_commit_identity
        assert type(commit_identity) is StreamTerminal
        handoff = _begin_terminal_handoff(
            page,
            TerminalBrowseHandoff(
                request,
                run_identity,
                latest.artifact,
                clones[-1].local_frame_label,
                tuple(frame.local_frame_label for frame in clones),
                commit_identity,
            ),
        )
        assert page._terminal_scientific_matches(handoff, rebound)
        page._last_scientific_projection = replace(
            scientific,
            plot_options=replace(options, waterfall_step=1),
        )
        assert not page._terminal_scientific_matches(handoff, rebound)
        page._last_scientific_projection = scientific
        view._waterfall_source_keys = full_waterfall_source[:-1]
        assert not page._terminal_scientific_matches(handoff, rebound)
        view._waterfall_source_keys = full_waterfall_source
        assert page._terminal_scientific_matches(handoff, rebound)

        foreign_identity = RunIdentity(
            run_identity.generation,
            run_identity.fingerprint,
        )
        assert foreign_identity == run_identity
        assert foreign_identity is not run_identity
        foreign = tuple(
            DisplayFrameKey(
                foreign_identity,
                frame.source_scan,
                frame.artifact,
                frame.local_frame_label,
                frame.work_ordinal,
            )
            for frame in clones
        )
        assert not page._terminal_scientific_matches(
            handoff,
            FrameNavigationProjection(foreign, foreign[-1], foreign),
        )
        assert view.rebind_navigation(
            rebound,
            heavy_available=frozenset(clones),
        )
        expected_rebound_source = clones[1:20:3]
        assert len(view.waterfall_source_frame_keys) == len(
            expected_rebound_source
        )
        assert all(
            painted is expected
            for painted, expected in zip(
                view.waterfall_source_frame_keys,
                expected_rebound_source,
                strict=True,
            )
        )
    finally:
        page.close_workspace()


def test_terminal_browse_rebind_reuses_complete_identity_distinct_waterfall(
    tmp_path, monkeypatch, qapp,
):
    import numpy as np

    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadOutcome, BrowseLoadStatus,
    )
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
    from xrd_tools.io.output_transaction import StreamTerminal
    from pyqtgraph.Qt import QtCore

    page, _store, _seeded, context = _loaded_page(
        tmp_path, monkeypatch, qapp, terminal_browse=True,
    )
    try:
        controller = page._context_controller
        navigation = controller.navigation
        run_identity = navigation.current.run_identity
        assert controller.select_navigation(
            navigation.frames[-1], navigation.frames,
        )
        page._preferences = replace(
            page._preferences, plot_mode="Waterfall",
        )
        view = page._shell.scientific

        def complete_waterfall():
            page._drain_executor()
            page._refresh_shell()
            if page._shell.scientific is not view:
                raise AssertionError(
                    f"initial cache paint replaced its view: {page._notice_text}"
                )
            return (
                view
                if view.trace_history_keys
                == controller.navigation.selected
                else None
            )

        view = _wait(complete_waterfall)
        before = page._last_scientific_projection
        assert before is not None
        before_history = view.trace_history_projections
        assert before_history
        before_axis_arrays = tuple(trace.axis.values for trace in before.traces)
        before_intensity_arrays = tuple(trace.intensity for trace in before.traces)
        before_axis_values = tuple(array.copy() for array in before_axis_arrays)
        before_intensity_values = tuple(
            array.copy() for array in before_intensity_arrays
        )
        preference_before = page._preferences
        bottom_widget = view.bottom_stack.currentWidget()
        waterfall_plot = view.waterfall.plot
        waterfall_image = view.waterfall.image
        combo_blocker = QtCore.QSignalBlocker(view.plot_axis)
        if view.plot_axis.findData("q_ip") < 0:
            view.plot_axis.addItem("Qᵢₚ (Å⁻¹)", "q_ip")
        view.plot_axis.setCurrentIndex(view.plot_axis.findData("q_ip"))
        del combo_blocker
        assert view.plot_axis.currentData() == "q_ip"

        clones = tuple(
            DisplayFrameKey(
                frame.run_identity,
                frame.source_scan,
                frame.artifact,
                frame.local_frame_label,
                frame.work_ordinal,
            )
            for frame in controller.navigation.frames
        )
        rebound = FrameNavigationProjection(clones, clones[-1], clones)
        request = context.load_request
        commit_identity = request.terminal_commit_identity
        assert type(commit_identity) is StreamTerminal
        expected_handoff = TerminalBrowseHandoff(
            request, clones[-1].run_identity, clones[-1].artifact,
            clones[-1].local_frame_label,
            tuple(frame.local_frame_label for frame in clones),
            commit_identity,
        )
        handoff = _begin_terminal_handoff(page, expected_handoff)
        foreign_identity = RunIdentity(
            run_identity.generation,
            run_identity.fingerprint,
        )
        assert foreign_identity == run_identity
        assert foreign_identity is not run_identity
        foreign_clones = tuple(
            DisplayFrameKey(
                foreign_identity,
                frame.source_scan,
                frame.artifact,
                frame.local_frame_label,
                frame.work_ordinal,
            )
            for frame in clones
        )
        foreign_navigation = FrameNavigationProjection(
            foreign_clones, foreign_clones[-1], foreign_clones,
        )
        assert not page._terminal_scientific_matches(
            handoff, foreign_navigation,
        )
        assert not page._terminal_scientific_rebind_authorized(
            foreign_navigation,
            (run_identity, handoff.source_artifact, request.source_path),
        )
        assert page._terminal_scientific_matches(handoff, rebound)

        controller._runtime._set_browse_navigation(rebound)
        capture = controller.capture_loaded_browse(request)
        assert capture is not None and capture.context is context
        assert page._settle_terminal_browse(BrowseLoadOutcome(
            request, BrowseLoadStatus.READY,
        ))
        assert page._processed_browser.terminal_handoff is None
        presentation = page._processed_browser.terminal_presentation
        assert presentation is not None
        # Cache-backed Browse receipts deliberately cannot zero-copy rebind
        # acquisition arrays.  The first terminal rebind attempt must retain
        # the coherent old paint and schedule the copied-cache fallback.
        paint = page._begin_processed_terminal_paint(
            presentation, reuse_science=True,
        )
        assert paint is not None
        assert paint.mode is TerminalPaintMode.REBIND
        revision = page._shell_revision
        page._refresh_shell(
            preserve_scientific=True,
            skip_scientific_projection=True,
            rebind_scientific_navigation=True,
            terminal_paint=paint,
        )
        page._complete_processed_terminal_paint(
            paint, applied=page._shell_revision > revision,
        )
        assert page._scientific_repaint_pending

        def complete_copied_fallback():
            page._drain_executor()
            page._refresh_shell()
            current_view = page._shell.scientific
            if current_view is not view:
                raise AssertionError(
                    f"cache fallback replaced its view: {page._notice_text}"
                )
            projection = page._last_scientific_projection
            snapshot = (
                None if projection is None
                else projection.browse_trace_snapshot
            )
            return (
                projection
                if snapshot is not None
                and snapshot.logical_frames == clones
                and view.trace_history_keys == clones
                and not page._scientific_repaint_pending
                else None
            )

        after = _wait(complete_copied_fallback)

        assert view.navigation_frame_keys == clones
        assert view.navigation_current_key is clones[-1]
        assert view.navigation_selected_keys == clones
        assert view.trace_history_keys == clones
        assert after.browse_trace_snapshot is not None
        assert after.browse_trace_snapshot.logical_positions == tuple(
            range(1, len(clones) + 1)
        )
        assert view.trace_row_count == len(clones)
        assert all(
            trace.axis.values is not old
            and not trace.axis.values.flags.writeable
            and np.array_equal(trace.axis.values, values, equal_nan=True)
            for trace, old, values in zip(
                after.traces,
                before_axis_arrays,
                before_axis_values,
                strict=True,
            )
        )
        after_history = view.trace_history_projections
        assert all(
            old.intensity is not new.intensity
            and np.array_equal(
                new.intensity, values, equal_nan=True,
            )
            for old, new, values in zip(
                before_history,
                after_history,
                before_intensity_values,
                strict=True,
            )
        )
        assert all(
            trace.intensity is not old
            and not trace.intensity.flags.writeable
            and np.array_equal(trace.intensity, values, equal_nan=True)
            for trace, old, values in zip(
                after.traces,
                before_intensity_arrays,
                before_intensity_values,
                strict=True,
            )
        )
        assert page._preferences is preference_before
        assert view.bottom_stack.currentWidget() is bottom_widget
        assert view.waterfall.plot is waterfall_plot
        assert view.waterfall.image is waterfall_image
        assert view.plot_axis.currentData() == "Q"
        active_plot = (
            view.waterfall.plot if bottom_widget is view.waterfall else view.curve
        )
        bottom_axis = active_plot.getAxis("bottom")
        assert bottom_axis.labelText == "Q"
        assert bottom_axis.labelUnits == "Å⁻¹"
        # Cache-backed repaint owns its logical selection in the immutable
        # BrowseTraceSnapshot; it must not populate the legacy projection-lane
        # commit cache as a side effect.
        assert controller._runtime._committed_trace_selection == ()
        assert not page._scientific_repaint_pending
        assert context.loaded_labels == tuple(
            frame.local_frame_label for frame in clones
        )
    finally:
        page.close_workspace()


def test_terminal_browse_same_labels_changed_content_forces_normal_repaint(
    tmp_path, monkeypatch, qapp,
):
    """A path/label match cannot reuse arrays from a different file seal."""

    import numpy as np

    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadOutcome, BrowseLoadStatus,
    )
    from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
    from xrd_tools.io.output_transaction import StreamTerminal

    page, _store, _seeded, context = _loaded_page(
        tmp_path, monkeypatch, qapp, terminal_browse=True,
    )
    try:
        controller = page._context_controller
        navigation = controller.navigation
        assert controller.select_navigation(
            navigation.frames[-1], navigation.frames,
        )
        page._preferences = replace(
            page._preferences, plot_mode="Waterfall",
        )
        view = page._shell.scientific

        def complete_waterfall():
            page._drain_executor()
            page._refresh_shell()
            return (
                page._last_scientific_projection
                if view.trace_history_keys
                == controller.navigation.selected
                else None
            )

        before = _wait(complete_waterfall)
        clones = tuple(
            DisplayFrameKey(
                frame.run_identity,
                frame.source_scan,
                frame.artifact,
                frame.local_frame_label,
                frame.work_ordinal,
            )
            for frame in navigation.frames
        )
        rebound = FrameNavigationProjection(clones, clones[-1], clones)
        request = context.load_request
        snapshot = context.target_snapshot
        accepted_commit = request.terminal_commit_identity
        assert type(accepted_commit) is StreamTerminal
        changed_digest = (
            "0" * 64
            if snapshot.digest != "0" * 64
            else "1" * 64
        )
        expected_handoff = TerminalBrowseHandoff(
            request,
            clones[-1].run_identity,
            clones[-1].artifact,
            clones[-1].local_frame_label,
            tuple(frame.local_frame_label for frame in clones),
            StreamTerminal(
                request.source_path,
                snapshot.size,
                changed_digest,
                accepted_commit.ordinal,
                accepted_commit.device,
                accepted_commit.inode,
                accepted_commit.mtime_ns,
                accepted_commit.ctime_ns,
            ),
        )
        handoff = _begin_terminal_handoff(page, expected_handoff)
        assert page._terminal_scientific_matches(handoff, rebound)

        stale_arrays = tuple(
            np.full_like(trace.intensity, -123.0)
            for trace in before.traces
        )
        page._last_scientific_projection = replace(
            before,
            traces=tuple(
                replace(trace, intensity=intensity)
                for trace, intensity in zip(
                    before.traces, stale_arrays, strict=True,
                )
            ),
        )
        controller._runtime._set_browse_navigation(rebound)
        capture = controller.capture_loaded_browse(request)
        assert capture is not None and capture.context is context
        assert not page._settle_terminal_browse(BrowseLoadOutcome(
            request, BrowseLoadStatus.READY,
        ))
        presentation = page._processed_browser.terminal_presentation
        assert presentation is not None
        paint = page._begin_processed_terminal_paint(
            presentation, reuse_science=False,
        )
        assert paint is not None
        assert paint.mode is TerminalPaintMode.REPAINT

        projection_calls = []
        original_project = controller.project_browse_1d_cache

        def project_browse_1d_cache(**kwargs):
            value = original_project(**kwargs)
            projection_calls.append(value)
            return value

        monkeypatch.setattr(
            controller, "project_browse_1d_cache", project_browse_1d_cache,
        )
        revision = page._shell_revision
        page._refresh_shell(terminal_paint=paint)
        page._complete_processed_terminal_paint(
            paint, applied=page._shell_revision > revision,
        )
        assert projection_calls
        assert all(
            not np.all(trace.intensity == -123.0)
            for trace in page._last_scientific_projection.traces
        )
    finally:
        page.close_workspace()


def test_xye_loaded_browse_disables_and_refuses_reintegrate(
    tmp_path, monkeypatch, qapp,
):
    """XYE output cannot mutate a retained NeXus Browse context."""

    from xrd_tools.session.readiness import ControlAction, SectionId

    page, store, _seeded, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    try:
        assert page._capture_current_loaded_browse() is not None
        snapshot = store.snapshot()
        candidate = snapshot.thaw()
        candidate.processing_mode = "Int 1D (XYE)"
        accepted = store.commit(
            candidate, expected_revision=snapshot.revision,
        )
        page._reconcile_snapshot(snapshot, accepted.snapshot)

        actions = {
            action.action: action
            for action in page._project_controls(
                store.snapshot()
            ).actions_for(SectionId.PROCESSING)
        }
        assert not actions[ControlAction.REINTEGRATE_1D].enabled
        assert not actions[ControlAction.REINTEGRATE_2D].enabled

        dispatches = []
        monkeypatch.setattr(
            page._workspace_operations._slot,
            "begin_reintegrate",
            lambda **kwargs: dispatches.append(kwargs),
        )
        page._reintegrate_action("1d")
        assert dispatches == []
        assert "XYE-only output" in page._notice_text
        assert page._capture_current_loaded_browse() is not None
    finally:
        page.close_workspace()


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


def test_terminal_browse_reuses_writer_seal_without_full_artifact_hash(
    tmp_path, monkeypatch,
):
    """Terminal Browse uses two object fences and no redundant SHA pass."""

    import shutil

    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest, BrowseLoadStatus,
    )
    from xrd_tools.io import output_transaction as transaction

    seeded = _seed_existing(tmp_path)
    target = seeded.target.resolve()
    seal = seeded.terminal.commit_identity
    real_capture = module.capture_target_snapshot
    real_revalidate = module.revalidate_stream_terminal
    real_sha = transaction._sha256_handle
    full_captures = []
    revalidations = []
    hashed_bytes = []

    def capture(path):
        full_captures.append(threading.current_thread().name)
        return real_capture(path)

    def revalidate(path, terminal):
        revalidations.append(threading.current_thread().name)
        return real_revalidate(path, terminal)

    def sha256_handle(handle):
        hashed_bytes.append(os.fstat(handle.fileno()).st_size)
        return real_sha(handle)

    monkeypatch.setattr(module, "capture_target_snapshot", capture)
    monkeypatch.setattr(module, "revalidate_stream_terminal", revalidate)
    monkeypatch.setattr(transaction, "_sha256_handle", sha256_handle)

    terminal_loader = module.BrowseLoader()
    terminal_request = BrowseLoadRequest(
        "terminal-seal", 1, str(target), seal,
    )
    terminal_loader.begin(terminal_request)
    terminal_outcome = _wait(
        lambda: terminal_loader.poll(terminal_request),
    )
    terminal_context = terminal_loader.consume(terminal_outcome)
    assert terminal_outcome.status is BrowseLoadStatus.READY
    assert terminal_context.loaded_labels == seeded.labels
    assert (
        terminal_context.target_snapshot.size,
        terminal_context.target_snapshot.digest,
    ) == (seal.size, seal.digest)
    assert full_captures == hashed_bytes == []
    assert revalidations == ["scattering-browse"] * 2
    assert terminal_loader.release_context(
        terminal_context
    ).cleanup_status.value == "cleaned"

    legacy = type(seal)(seal.target, seal.size, seal.digest, seal.ordinal)
    full_captures.clear(); revalidations.clear(); hashed_bytes.clear()
    legacy_loader = module.BrowseLoader()
    legacy_request = BrowseLoadRequest(
        "legacy-two-fence", 1, str(target), legacy,
    )
    legacy_loader.begin(legacy_request)
    legacy_outcome = _wait(lambda: legacy_loader.poll(legacy_request))
    legacy_context = legacy_loader.consume(legacy_outcome)
    assert legacy_outcome.status is BrowseLoadStatus.READY
    assert full_captures == ["scattering-browse"] * 2
    assert revalidations == []
    assert hashed_bytes == [target.stat().st_size] * 2
    assert legacy_loader.release_context(
        legacy_context
    ).cleanup_status.value == "cleaned"

    full_captures.clear(); revalidations.clear(); hashed_bytes.clear()
    generic_loader = module.BrowseLoader()
    generic_request = BrowseLoadRequest(
        "generic-two-fence", 1, str(target),
    )
    generic_loader.begin(generic_request)
    generic_outcome = _wait(lambda: generic_loader.poll(generic_request))
    generic_context = generic_loader.consume(generic_outcome)
    assert generic_outcome.status is BrowseLoadStatus.READY
    assert full_captures == ["scattering-browse"] * 2
    assert revalidations == []
    assert hashed_bytes == [target.stat().st_size] * 2
    assert generic_loader.release_context(
        generic_context
    ).cleanup_status.value == "cleaned"

    original_inode = target.stat().st_ino
    replacement = target.with_name("byte-identical-replacement.nxs")
    shutil.copy2(target, replacement)
    os.replace(replacement, target)
    assert target.stat().st_ino != original_inode
    full_captures.clear(); revalidations.clear(); hashed_bytes.clear()
    opened = []

    def forbidden_open(_path):
        opened.append(True)
        raise AssertionError("replaced target reached Browse decoding")

    replaced_loader = module.BrowseLoader(open_scan=forbidden_open)
    replaced_request = BrowseLoadRequest(
        "terminal-replaced", 1, str(target), seal,
    )
    replaced_loader.begin(replaced_request)
    replaced_outcome = _wait(
        lambda: replaced_loader.poll(replaced_request),
    )
    assert replaced_outcome.status is BrowseLoadStatus.FAILED
    assert replaced_loader.context_for_outcome(replaced_outcome) is None
    assert opened == full_captures == hashed_bytes == []
    assert revalidations == ["scattering-browse"]


def test_terminal_browse_uses_one_targeted_presentation_read(
    tmp_path, monkeypatch,
):
    """Terminal cleanup never performs the old full provenance/metadata pass."""

    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest, BrowseLoadStatus,
    )
    from xrd_tools.core import provenance as provenance_module
    from xrd_tools.io.browse_presentation import read_browse_presentation

    seeded = _seed_existing(tmp_path)
    target = seeded.target.resolve()
    expected_presentation, expected_mask = read_browse_presentation(target)
    calls = []

    def targeted(path):
        calls.append((str(path), threading.current_thread().name))
        return read_browse_presentation(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("general provenance/metadata traversal is forbidden")

    monkeypatch.setattr(module, "read_browse_presentation", targeted)
    monkeypatch.setattr(provenance_module, "read_provenance", forbidden)
    monkeypatch.setattr(
        module.ProcessedScan,
        "metadata",
        property(lambda _scan: forbidden()),
    )
    loader = module.BrowseLoader()
    request = BrowseLoadRequest(
        "targeted-presentation",
        1,
        str(target),
        seeded.terminal.commit_identity,
    )
    loader.begin(request)
    outcome = _wait(lambda: loader.poll(request))
    context = loader.consume(outcome)
    assert outcome.status is BrowseLoadStatus.READY
    assert calls == [(str(target), "scattering-browse")]
    assert json.loads(context.calibration_identity) == expected_presentation
    assert expected_mask is None
    assert context.mask_identity == "False"
    assert context.loaded_labels == seeded.labels
    assert loader.release_context(context).cleanup_status.value == "cleaned"


def test_terminal_browse_path_binding_does_not_resolve_on_gui_thread(
    tmp_path, monkeypatch, qapp,
):
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus

    seeded = _seed_existing(tmp_path)
    target = str(seeded.target.resolve())
    page, _store = _page(tmp_path, monkeypatch)
    main_ident = threading.get_ident()
    resolve_threads = []
    real_resolve = Path.resolve

    def tracked_resolve(path, *args, **kwargs):
        resolve_threads.append(threading.get_ident())
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", tracked_resolve)
    try:
        request = page._context_controller.begin_browse(
            target,
            terminal_commit_identity=seeded.terminal.commit_identity,
        )
        outcome = _wait(page._context_controller.poll_browse)
        assert outcome.request is request
        assert outcome.status is BrowseLoadStatus.READY
        assert main_ident not in resolve_threads
    finally:
        page.close_workspace()


def test_unsealed_browse_canonicalization_runs_only_on_browse_worker(
    tmp_path, monkeypatch, qapp,
):
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadStatus

    seeded = _seed_existing(tmp_path)
    target = str(seeded.target.absolute())
    page, _store = _page(tmp_path, monkeypatch)
    main_ident = threading.get_ident()
    resolve_threads = []
    real_resolve = Path.resolve

    def tracked_resolve(path, *args, **kwargs):
        resolve_threads.append(threading.get_ident())
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", tracked_resolve)
    try:
        request = page._context_controller.begin_browse(target)
        assert resolve_threads == []
        outcome = _wait(page._context_controller.poll_browse)
        assert outcome.request is request
        assert outcome.status is BrowseLoadStatus.READY
        assert resolve_threads
        assert main_ident not in resolve_threads
    finally:
        page.close_workspace()


def test_terminal_browse_semantic_seal_reintegrates_from_fresh_full_snapshot(
    tmp_path, monkeypatch, qapp,
):
    from xdart.gui.tabs.scattering.adapters import external_operation
    from xdart.gui.tabs.scattering.operation_values import (
        OperationTerminalStatus,
    )

    page, _store, seeded, context = _loaded_page(
        tmp_path, monkeypatch, qapp, terminal_browse=True,
    )
    request = context.load_request
    seal = request.terminal_commit_identity
    full_digest = hashlib.sha256(seeded.target.read_bytes()).hexdigest()
    assert seal is seeded.terminal.commit_identity
    assert seal.digest != full_digest
    built = []
    real_build = external_operation.ReintegratePlan.from_artifact

    def build(target, **kwargs):
        plan = real_build(target, **kwargs)
        built.append((kwargs, plan))
        return plan

    monkeypatch.setattr(
        external_operation.ReintegratePlan,
        "from_artifact",
        staticmethod(build),
    )
    _stub_integrators(monkeypatch)
    try:
        page._reintegrate_action("1d")
        update = _join(
            _reintegrate_slot(page),
            page._workspace_operations.reintegrate_identity,
        )
        result = update.terminal.payload
        assert update.terminal.status is OperationTerminalStatus.RETURNED
        assert result.disposition == "COMMITTED"
        assert built[0][0]["expected_terminal_identity"] is seal
        assert built[0][1].expected_target_snapshot.digest == full_digest
        assert page._consume_reintegrate_update(update)
        reload_request = page._context_controller._browse_request
        assert reload_request is not None
        assert reload_request.terminal_commit_identity is result.commit_identity
    finally:
        page.close_workspace()


def test_private_request_prepares_plan_only_on_existing_operation_worker(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import external_operation as module
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationTerminalStatus
    from xrd_tools.reduction import ReintegrateResult, reintegrate as core
    entered, release, seen = threading.Event(), threading.Event(), []; plan = SimpleNamespace(operation_identity="a" * 64)
    result = core._value(ReintegrateResult, "COMMITTED", (2,), (2,), (), (), "b"*64, "a"*64, "c"*64, None)
    def build(target, **kw): entered.set(); release.wait(2); seen.append((threading.current_thread().name, target, kw)); return plan
    def run(got, **kw): seen.append((got, kw)); return result
    monkeypatch.setattr(module.ReintegratePlan, "from_artifact", staticmethod(build)); monkeypatch.setattr(module, "run_reintegrate", run)
    values = _persisted({"version": 1, "dimension": "1d", "bai_args": {}, "gi_mode": "q_total"}); slot = module.OperationSlot(); identity = slot.begin_reintegrate(target="/detached", entry="entry", source_root="/", expected_target_snapshot=core.TargetSnapshot(True,1,2,3,4,"d"*64), expected_labels=(2,), dimension="1d", preparation_values=values, stamp=OperationContextStamp(0))
    assert entered.wait(2); values["selected_plan"]["bai_args"]["npt"] = 99; release.set(); update = _join(slot, identity)
    assert update.terminal.status is OperationTerminalStatus.RETURNED and seen[0][0].startswith("scattering-operation-") and seen[0][2]["source_root"] == "/" and "npt" not in seen[0][2]["preparation"]["selected_plan"]["bai_args"] and seen[1][0] is plan and seen[0][2]["cancel_token"] is seen[1][1]["cancel_token"]


def test_persisted_v3_gi_calibration_reconstructs_parallax_for_reintegration():
    from xrd_tools.corrections.grazing import GI_EXIT_ANGLE_CONVENTION
    from xrd_tools.reduction import reintegrate as module
    from xrd_tools.session.run_configuration import GIIntent, RunIntent

    config = {
        "pixel1": 1.0e-4,
        "pixel2": 1.0e-4,
        "max_shape": [12, 14],
        "orientation": 3,
        "sensor": {"material": "Ge", "thickness": 0.00075},
    }
    projection = {
        "dist": 0.2,
        "poni1": 0.0006,
        "poni2": 0.0007,
        "rot1": 0.0,
        "rot2": 0.0,
        "rot3": 0.0,
        "wavelength": 1.0e-10,
        "detector": "Detector",
        "detector_config": config,
        "parallax": True,
    }
    assets = {
        "poni_values": projection,
        "poni_detector_config_json": json.dumps(
            config, sort_keys=True, separators=(",", ":"),
        ),
        "poni_sha256": "a" * 64,
        "mask_sha256": None,
    }
    frozen = RunIntent(
        poni_values=projection,
        gi=GIIntent(enabled=True, incidence_motor="Manual", th_val=0.2),
    ).freeze()
    outer = frozen.as_provenance()
    signed = copy.deepcopy(outer)
    signed["accepted_scientific_assets"] = copy.deepcopy(assets)
    persisted = {**copy.deepcopy(outer), "scientific_signature": signed}

    shared = module._validated_shared_science(persisted)
    calibration, integrator, fiber = module._calibration(shared)

    assert shared["gi"]["gi_exit_angle_convention"] == (
        GI_EXIT_ANGLE_CONVENTION
    )
    assert calibration.parallax is True
    assert calibration.detector_config["sensor"] == {
        "material": "Ge",
        "thickness": pytest.approx(0.00075),
    }
    assert integrator.parallax is not None
    assert integrator.array_from_unit(unit="q_A^-1").shape == (12, 14)
    assert fiber.parallax is not None
    assert fiber.detector.get_config()["sensor"] == {
        "material": "Ge",
        "thickness": pytest.approx(0.00075),
    }
    assert fiber.wavelength == projection["wavelength"]
    from xrd_tools.integrate.gid import _fiber_from_integrator
    worker_fiber = _fiber_from_integrator(
        fiber,
        incident_angle=0.2,
        tilt_angle=0.0,
        sample_orientation=4,
    )
    assert worker_fiber is not fiber
    assert worker_fiber.parallax is not None
    assert worker_fiber.detector.get_config() == fiber.detector.get_config()
    assert worker_fiber.wavelength == fiber.wavelength
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
    direct = run_reintegrate(ReintegratePlan.from_artifact(first.target, entry="entry", dimension="1d", preparation=first.preparation)); slot = OperationSlot(); identity = slot.begin_reintegrate(target=str(second.target.resolve()), entry="entry", source_root=str(second.target.parent.resolve()), expected_target_snapshot=capture_target_snapshot(second.target), expected_labels=second.labels, dimension="1d", preparation_values=_persisted(second.preparation["selected_plan"]), stamp=OperationContextStamp(0)); gui = _join(slot, identity)
    assert gui.terminal.status is OperationTerminalStatus.RETURNED and direct.disposition == gui.terminal.payload.disposition == "COMMITTED"
    with h5py.File(first.target) as a, h5py.File(second.target) as b: assert np.array_equal(a["entry/integrated_1d/intensity"], b["entry/integrated_1d/intensity"]) and np.array_equal(a["entry/integrated_2d/intensity"], b["entry/integrated_2d/intensity"])
def test_browse_invalidation_terminal_reload_and_foreign_stale_refusal(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.operation_values import OperationIdentity, OperationTerminal, OperationTerminalStatus, OperationUpdate; from xrd_tools.reduction import ReintegrateResult, reintegrate as core; page, _store, _seed, context = _loaded_page(tmp_path, monkeypatch, qapp); controller = page._context_controller
    seeded = _seed
    owner = controller._browse_hydration_owner; assert owner is not None
    with owner._one_d_lane._lock: owner._one_d_lane._ready_pending = True
    request, target = context.load_request, context.requested_path; foreign = replace(request, token=request.token + "-foreign"); assert foreign is not request and controller.capture_loaded_browse(foreign) is None
    captured = controller.capture_loaded_browse(request); assert captured.context is context and controller.invalidate_reintegrate_browse(captured)
    assert controller.reload_reintegrate_browse(foreign, target) is None and controller.browse_context is context and context.invalidated and not context.released
    seal = seeded.terminal.commit_identity
    rid = OperationIdentity(77); result = core._value(ReintegrateResult, "COMMITTED", seeded.labels, seeded.labels, (), (), "b"*64, "a"*64, "c"*64, seal); _set_reintegrate_state(page, rid, captured); assert page._consume_reintegrate_update(OperationUpdate(rid, terminal=OperationTerminal(rid, OperationTerminalStatus.RETURNED, payload=result), stale=True))
    reload_request = controller._browse_request
    assert result.disposition == "COMMITTED" and page._workspace_operations.reintegrate_state is None and reload_request is not None and reload_request is not request and reload_request.terminal_commit_identity is seal and context.released and owner._one_d_lane._closed and not controller.owns_browse_request(request) and page._processed_browser.pending_reintegrate_reload is None; page.close_workspace()
def test_same_event_cancels_prepare_and_run_without_false_terminal(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import external_operation as module
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationTerminalStatus
    from xrd_tools.reduction import reintegrate as core; from xrd_tools.reduction.reintegrate import ReintegrateCancelled
    seen = []; entered = threading.Event()
    def cancelled(_target, **kw): seen.append(kw["cancel_token"]); entered.set(); kw["cancel_token"].wait(2); raise ReintegrateCancelled()
    monkeypatch.setattr(module.ReintegratePlan, "from_artifact", staticmethod(cancelled)); slot = module.OperationSlot(); identity = slot.begin_reintegrate(target="/cancel", entry="entry", source_root="/", expected_target_snapshot=core.TargetSnapshot(True,1,2,3,4,"d"*64), expected_labels=(1,), dimension="1d", preparation_values=_persisted({"version":1,"dimension":"1d","bai_args":{},"gi_mode":"q_total"}), stamp=OperationContextStamp(0))
    assert entered.wait(2) and slot.cancel(identity); update = _join(slot, identity)
    assert update.terminal.status is OperationTerminalStatus.CANCELLED and seen[0].is_set()
def test_reintegrate_progress_result_close_and_control_projection(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import external_operation as module
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationTerminalStatus
    from xrd_tools.reduction import ReintegrateResult, reintegrate as core
    plan = SimpleNamespace(operation_identity="a"*64); result = core._value(ReintegrateResult,"COMMITTED",(1,),(1,),(),(),"b"*64,"a"*64,"c"*64,None)
    monkeypatch.setattr(module.ReintegratePlan,"from_artifact",staticmethod(lambda *_a,**_k: plan))
    def run(_plan, **kw): kw["progress_cb"](core._progress("a"*64,"read",0,1,1)); kw["progress_cb"](core._progress("a"*64,"write",1,1,2)); kw["progress_cb"](core._progress("x"*64,"settle",1,1,3)); return result
    monkeypatch.setattr(module,"run_reintegrate",run); slot=module.OperationSlot(); identity=slot.begin_reintegrate(target="/progress",entry="entry",source_root="/",expected_target_snapshot=core.TargetSnapshot(True,1,2,3,4,"d"*64),expected_labels=(1,),dimension="1d",preparation_values=_persisted({"version":1,"dimension":"1d","bai_args":{},"gi_mode":"q_total"}),stamp=OperationContextStamp(0)); update=_join(slot,identity)
    assert update.terminal.status is OperationTerminalStatus.RETURNED and update.progress.identity is identity and (update.progress.stage,update.progress.revision)==("write",4) and all(getattr(slot,n) is None for n in ("_frozen","_worker","_cancel_event"))


def test_reintegrate_progress_and_cancel_touch_only_current_scalar_footer(
    tmp_path, monkeypatch, qapp,
):
    from xdart.gui.tabs.scattering.operation_values import (
        OperationIdentity,
        OperationProgress,
        OperationUpdate,
    )
    page, _store, _seed, _context = _loaded_page(
        tmp_path, monkeypatch, qapp,
    )
    operations = page._workspace_operations
    slot = _reintegrate_slot(page)
    identity = OperationIdentity(78)
    capture = page._capture_current_loaded_browse()
    assert capture is not None
    slot._identity = identity
    _set_reintegrate_state(page, identity, capture, "1d")
    view = page._shell.scientific
    original_progress = view.progress.text()
    original_navigation = page._context_controller.navigation
    original_shell_revision = page._shell_revision
    original_selector = (
        view.frame_selector.count(), view.frame_selector.currentIndex()
    )
    original_plots = (
        view.raw._render_contract,
        view.cake._render_contract,
        tuple(view.curve.listDataItems()),
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("Reintegrate scalar progress repainted scientific state")

    monkeypatch.setattr(page, "_refresh_shell", forbidden)
    monkeypatch.setattr(view, "reconcile", forbidden)
    first = OperationProgress(identity, "integrate", 17, 651, 1)
    pending = [OperationUpdate(identity, progress=first)]
    polled = []
    monkeypatch.setattr(
        operations,
        "poll",
        lambda current: (
            polled.append(current), pending.pop(0) if pending else None
        )[1],
    )
    page._drain_executor()
    assert polled == [identity]
    assert view.status.text() == "Reintegrate 1-D: integrate 17/651…"
    assert view.progress.text() == original_progress
    assert page._shell_revision == original_shell_revision
    assert page._context_controller.navigation is original_navigation
    assert (
        view.frame_selector.count(), view.frame_selector.currentIndex()
    ) == original_selector
    assert (
        view.raw._render_contract,
        view.cake._render_contract,
        tuple(view.curve.listDataItems()),
    ) == original_plots

    foreign_identity = OperationIdentity(identity.serial)
    for update in (
        OperationUpdate(
            identity,
            progress=OperationProgress(identity, "integrate", 16, 651, 2),
        ),
        OperationUpdate(
            identity,
            progress=OperationProgress(identity, "integrate", 18, 651, 3),
            stale=True,
        ),
        OperationUpdate(
            foreign_identity,
            progress=OperationProgress(
                foreign_identity, "integrate", 19, 651, 4,
            ),
        ),
    ):
        page._consume_reintegrate_update(update)
        assert view.status.text() == "Reintegrate 1-D: integrate 17/651…"

    page._reintegrate_action("2d")
    assert view.status.text() == "Reintegrate 1-D: integrate 17/651…"
    cancelled = []
    monkeypatch.setattr(
        slot,
        "cancel",
        lambda current: cancelled.append(current) or current is identity,
    )
    page._reintegrate_action("1d")
    assert cancelled == [identity]
    assert operations.reintegrate_cancel_accepted
    assert view.status.text() == "Cancelling Reintegrate 1-D…"
    page._reintegrate_action("1d")
    page._reintegrate_action("2d")
    page._consume_reintegrate_update(OperationUpdate(
        identity,
        progress=OperationProgress(identity, "write", 651, 651, 5),
    ))
    assert cancelled == [identity]
    assert view.status.text() == "Cancelling Reintegrate 1-D…"
    assert page._run_timer.interval() == 125
    assert page._shell_revision == original_shell_revision
    operations._reintegrate = None
    slot._identity = None
    page.close_workspace()


def test_reintegrate_gui_owner_import_writer_and_snapshot_frequency_census():
    root=Path(__file__).resolve().parents[3]; names=("src/xrd_tools/reduction/reintegrate.py","src/xdart/modules/display_context.py","src/xdart/gui/tabs/scattering/adapters/browse_loader.py","src/xdart/gui/tabs/scattering/context_controller.py","src/xdart/gui/tabs/scattering/adapters/external_operation.py","src/xdart/gui/tabs/scattering/page.py","src/xdart/gui/tabs/scattering/controls_projection.py","src/xdart/gui/tabs/scattering/workspace_operations.py")
    sources={name:(root/name).read_text() for name in names}; gui_text="\n".join(sources[name] for name in names[1:]); external_source=sources[names[4]]; page_source=sources[names[5]]
    assert len(sources)==8 and sources[names[2]].count("capture_target_snapshot(")==2 and gui_text.count("Thread(")==2 and gui_text.count("OperationSlot()")==2
    assert not any(value in gui_text for value in ("h5py","NexusSink","NexusRecordWriter","_core_plan","_integration_1d_args","_integration_2d_args","resolve_session_policy")) and "ReintegrateRunner" not in external_source and external_source.count("ReintegratePlan.from_artifact(")==1 and external_source.count("run_reintegrate(")==1 and page_source.count("jsonable_run_value(")==1

def test_reintegrate_request_accepts_exact_dimensions_but_p34b_gui_is_1d_only(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.adapters.external_operation import OperationSlot
    from xdart.gui.tabs.scattering.operation_values import OperationContextStamp, OperationIdentity, OperationTerminal, OperationTerminalStatus, OperationUpdate
    from xdart.gui.tabs.scattering.shell_values import ShellCommand, ShellCommandKind
    from xrd_tools.session.readiness import ControlAction, SectionId; from xrd_tools.reduction import reintegrate as core
    slot=OperationSlot(); seen=[]; monkeypatch.setattr(slot,"_begin",lambda value,stamp,body: seen.append(value) or object())
    kwargs=dict(target="/target",entry="entry",source_root="/",expected_target_snapshot=core.TargetSnapshot(True,1,2,3,4,"d"*64),expected_labels=(1,),preparation_values=_persisted({"version":1,"dimension":"1d","bai_args":{},"gi_mode":"q_total"}),stamp=OperationContextStamp(0))
    assert slot.begin_reintegrate(dimension="1d",**kwargs) is not None and slot.begin_reintegrate(dimension="2d",**kwargs) is not None and slot.begin_reintegrate(dimension="3d",**kwargs) is None and tuple(f.name for f in fields(type(seen[0]))) == ("target","entry","source_root","expected_target_snapshot","expected_terminal_identity","expected_labels","dimension","preparation_json")
    page,store,_seed,_context=_loaded_page(tmp_path,monkeypatch,qapp); calls=[]; monkeypatch.setattr(page,"_reintegrate_action",calls.append); page._handle_shell_command(ShellCommand(ShellCommandKind.CONTROL_ACTION,"reintegrate_1d")); page._handle_shell_command(ShellCommand(ShellCommandKind.CONTROL_ACTION,"reintegrate_2d")); assert calls==["1d","2d"]
    rid=OperationIdentity(77); _reintegrate_slot(page)._identity=rid; captured=page._capture_current_loaded_browse(); assert captured is not None; operations=_set_reintegrate_state(page,rid,captured); assert page._context_controller.invalidate_reintegrate_browse(captured)
    for dimension,matching,other in (("1d",ControlAction.REINTEGRATE_1D,ControlAction.REINTEGRATE_2D),("2d",ControlAction.REINTEGRATE_2D,ControlAction.REINTEGRATE_1D)):
        operations._reintegrate=replace(operations.reintegrate_state,dimension=dimension); actions={a.action:a for a in page._project_controls(store.snapshot()).actions_for(SectionId.PROCESSING)}; assert actions[matching].enabled and actions[matching].label==f"Cancel Reintegrate {dimension[0]}-D" and not actions[other].enabled
    page._handle_shell_command(ShellCommand(ShellCommandKind.CONTROL_ACTION,"reintegrate_1d")); assert calls==["1d","2d"]; page._handle_shell_command(ShellCommand(ShellCommandKind.CONTROL_ACTION,"reintegrate_2d")); assert calls==["1d","2d","2d"]
    assert page._consume_reintegrate_update(OperationUpdate(rid,terminal=OperationTerminal(rid,OperationTerminalStatus.CANCELLED))) and operations.reintegrate_state is None; _reintegrate_slot(page)._identity=None; page.close_workspace()

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
    page,store,seeded,_context=_loaded_page(tmp_path,monkeypatch,qapp); plain=page._reintegrate_preparation(store.snapshot().thaw(),"1d"); assert plain["selected_plan"]["bai_args"]=={}; candidate=store.snapshot().thaw(); candidate.bai_1d_args={"numpoints":10,"radial_range":(.1,1.)}; candidate.max_cores=2; store.commit(candidate,expected_revision=store.revision); captured={}
    monkeypatch.setattr(_reintegrate_slot(page),"begin_reintegrate",lambda **kw: captured.update(kw) or OperationIdentity(91)); page._reintegrate_action("1d")
    prep=captured["preparation_values"]; assert prep["requested_shared_science"]=={"version":1,"kind":"persisted_target"} and prep["selected_plan"]["bai_args"]=={"numpoints":10,"radial_range":[.1,1.]} and prep["selected_plan"]["gi_mode"]==candidate.gi.mode_1d and prep["resource_policy"]["requests"]=={"workers":2}
    shown=json.dumps(prep); assert seeded.preparation["requested_shared_science"]["accepted_scientific_assets"]["poni_sha256"] not in shown and "loaded artifact" in page._notice_text and captured["dimension"]=="1d"; page._workspace_operations._reintegrate=None; page.close_workspace()

def test_parent_red_stable_loaded_browse_enables_reintegrate_2d_start(tmp_path, monkeypatch, qapp):
    from xrd_tools.session.readiness import ControlAction, SectionId; page,store,_seed,_context=_loaded_page(tmp_path,monkeypatch,qapp); action={a.action:a for a in page._project_controls(store.snapshot()).actions_for(SectionId.PROCESSING)}[ControlAction.REINTEGRATE_2D]; assert action.enabled and action.label=="Reintegrate 2-D"; page.close_workspace()
def test_reintegrate_2d_uses_exact_existing_worker_builder_and_runner(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.adapters import external_operation as module; from xdart.gui.tabs.scattering.operation_values import OperationTerminalStatus; from xrd_tools.reduction import ReintegrateResult, reintegrate as core
    page,_store,_seed,_context=_loaded_page(tmp_path,monkeypatch,qapp); seen=[]; plan=SimpleNamespace(operation_identity="a"*64); result=core._value(ReintegrateResult,"COMMITTED",(2,),(2,),(),(),"b"*64,"a"*64,"c"*64,None)
    build=lambda target,**kw:(seen.append(("build",threading.current_thread().name,target,kw)),plan)[1]; run=lambda got,**kw:(seen.append(("run",got,kw)),kw["progress_cb"](core._progress(plan.operation_identity,"write",1,1,1)),result)[2]
    monkeypatch.setattr(module.ReintegratePlan,"from_artifact",staticmethod(build)); monkeypatch.setattr(module,"run_reintegrate",run); page._reintegrate_action("2d"); identity=page._workspace_operations.reintegrate_identity; assert identity is not None and page._workspace_operations.reintegrate_dimension=="2d"; update=_join(_reintegrate_slot(page),identity)
    assert update.terminal.status is OperationTerminalStatus.RETURNED and [row[0] for row in seen]==["build","run"] and seen[0][1].startswith("scattering-operation-") and seen[0][3]["dimension"]=="2d" and seen[1][1] is plan and seen[0][3]["cancel_token"] is seen[1][2]["cancel_token"] and update.progress.identity is identity; assert page._consume_reintegrate_update(update) and page._workspace_operations.reintegrate_dimension is None; page.close_workspace()


def test_default_browse_retains_651_scalar_catalog_without_eager_rows(tmp_path):
    import h5py
    import numpy as np

    from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
    from xdart.gui.tabs.scattering.browse_values import (
        BrowseLoadRequest, BrowseLoadStatus,
    )

    labels = tuple(range(651))
    target, _raw = _write_processed(
        tmp_path, labels=labels, thumbnails=False, two_d=False,
    )
    with h5py.File(target, "r+") as handle:
        scan_data = handle["entry"].create_group("scan_data")
        scan_data.create_dataset(
            "frame_index", data=np.asarray(labels, dtype=np.int64),
        )
        scan_data.create_dataset(
            "monitor", data=np.asarray(labels, dtype=np.float32),
        )

    loader = BrowseLoader()
    request = BrowseLoadRequest("retain-651", 1, str(target))
    loader.begin(request)
    outcome = _wait(lambda: loader.poll(request))
    assert outcome.status is BrowseLoadStatus.READY
    context = loader.consume(outcome)
    assert context.loaded_labels == labels
    assert len(context.record_store) == 0
    assert len(context.publication_store) == 0
    assert context.browse_1d_cache.resident_keys == ()
    for label in (0, 325, 650):
        row = context.scalar_catalog.row(label)
        assert row is not None
        assert row.active_mode_1d in row.modes_1d
        assert row.metadata_raw["monitor"] == pytest.approx(label)
    assert loader.release_context(context).cleanup_status.value == "cleaned"


def test_direct_and_gui_scheduled_2d_match_after_reopen_and_preserve_1d(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.adapters import external_operation; from xdart.gui.tabs.scattering.operation_values import OperationTerminalStatus; from xrd_tools.io.output_transaction import StreamTerminal, capture_target_snapshot; from xrd_tools.reduction import ReintegratePlan, run_reintegrate, reintegrate as module
    seed=_seed_existing(tmp_path,labels=(0,1,2),append=True); direct_path=seed.target.parent/"direct.nexus"; gui_path=seed.target.parent/"gui.nexus"; direct_path.write_bytes(seed.target.read_bytes()); gui_path.write_bytes(seed.target.read_bytes()); before=(_tree_manifest(direct_path,"entry/integrated_1d"),_tree_manifest(gui_path,"entry/integrated_1d")); request=_resolved_2d()
    direct_plan=ReintegratePlan.from_artifact(direct_path,entry="entry",dimension="2d",preparation=copy.deepcopy(request),expected_target_snapshot=capture_target_snapshot(direct_path),expected_labels=seed.labels); plans=[]; real_build=ReintegratePlan.from_artifact
    def build(target,**kw): plans.append((copy.deepcopy(kw["preparation"]),real_build(target,**kw))); return plans[-1][1]
    bound=[]; real_bind=module._ReintegrateFrameSource.bind_allocation
    def bind(owner,allocation): bound.append((owner.plan.resource_allocation,allocation)); return real_bind(owner,allocation)
    monkeypatch.setattr(external_operation.ReintegratePlan,"from_artifact",staticmethod(build)); monkeypatch.setattr(module._ReintegrateFrameSource,"bind_allocation",bind); _stub_integrators(monkeypatch); direct=run_reintegrate(direct_plan); gui_seed=copy.copy(seed); gui_seed.target=gui_path; page,_store,_seed,_context=_loaded_page(tmp_path,monkeypatch,qapp,seed=gui_seed); page._reintegrate_action("2d"); identity=page._workspace_operations.reintegrate_identity; update=_join(_reintegrate_slot(page),identity); gui=update.terminal.payload; assert page._consume_reintegrate_update(update)
    gui_plan=plans[0][1]; da,ga=direct_plan.resource_allocation,gui_plan.resource_allocation; canonical=lambda value:json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode(); assert canonical(plans[0][0])==canonical(request) and da is not ga and da==ga and da.origin==ga.origin=="automatic" and da.requirements is not ga.requirements and da.requirements==ga.requirements and len(bound)==2 and all(left is right for left,right in bound) and {id(left) for left,_right in bound}=={id(da),id(ga)}
    assert update.terminal.status is OperationTerminalStatus.RETURNED and (direct.input_labels,direct.committed_labels,direct.publication_dropped_labels)==(gui.input_labels,gui.committed_labels,gui.publication_dropped_labels)==(seed.labels,seed.labels,()) and module._plain(direct_plan.selected_plan)==module._plain(gui_plan.selected_plan) and module._plain(direct_plan.requested_shared_science)==module._plain(gui_plan.requested_shared_science) and direct_plan.rollback_policy==gui_plan.rollback_policy=="ROLLBACK_ON_STOP" and direct_plan.session_policy.flush==gui_plan.session_policy.flush and direct.science_identity==gui.science_identity and direct.operation_identity!=gui.operation_identity
    assert _tree_manifest(direct_path,"entry/integrated_2d")==_tree_manifest(gui_path,"entry/integrated_2d") and _tree_manifest(direct_path,"entry/integrated_1d")==before[0]==before[1]==_tree_manifest(gui_path,"entry/integrated_1d"); audits=(_audit(direct_path),_audit(gui_path)); results=(direct,gui); plans_only=(direct_plan,gui_plan)
    for audit,result,plan in zip(audits,results,plans_only): assert audit["operation_identity"]==plan.operation_identity==result.operation_identity and result.audit_identity==hashlib.sha256(json.dumps(audit,sort_keys=True,separators=(",",":")).encode()).hexdigest() and audit["append_lineage_action"]=="preserved_append_disabled" and len(audit["append_lineage_sha256"])==64 and audit["selected_gi_mode"] is None and type(result.commit_identity) is StreamTerminal
    independent=[dict(value) for value in audits]; [value.pop("operation_identity") for value in independent]; assert independent[0]==independent[1] and direct.audit_identity!=gui.audit_identity and direct.commit_identity!=gui.commit_identity and direct.commit_identity.target!=gui.commit_identity.target; page.close_workspace()
def test_reintegrate_dimensions_share_one_slot_cancel_terminal_and_reload(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.adapters import external_operation as module; from xdart.gui.tabs.scattering.operation_values import OperationIdentity, OperationTerminalStatus; from xrd_tools.reduction.reintegrate import ReintegrateCancelled; from xrd_tools.session.readiness import ControlAction, SectionId
    page,store,_seed,_context=_loaded_page(tmp_path,monkeypatch,qapp); entered=threading.Event(); tokens=[]
    def cancelled(_target,**kw): tokens.append(kw["cancel_token"]); entered.set(); kw["cancel_token"].wait(2); raise ReintegrateCancelled()
    monkeypatch.setattr(module.ReintegratePlan,"from_artifact",staticmethod(cancelled)); page._reintegrate_action("2d"); operations=page._workspace_operations; identity=operations.reintegrate_identity; capture=operations.reintegrate_capture; request=capture.request; target=capture.target; reloads=[]; real_reload=page._context_controller.reload_reintegrate_browse; monkeypatch.setattr(page._context_controller,"reload_reintegrate_browse",lambda got_request,got_target:(reloads.append((got_request,got_target)),real_reload(got_request,got_target))[1]); assert entered.wait(2) and operations.owned and operations.reintegrate_dimension=="2d"
    actions={a.action:a for a in page._project_controls(store.snapshot()).actions_for(SectionId.PROCESSING)}; assert actions[ControlAction.REINTEGRATE_2D].enabled and actions[ControlAction.REINTEGRATE_2D].label=="Cancel Reintegrate 2-D" and not actions[ControlAction.REINTEGRATE_1D].enabled; page._reintegrate_action("1d"); assert not tokens[0].is_set(); page._reintegrate_action("2d"); assert tokens[0].is_set()
    update=_join(_reintegrate_slot(page),identity); assert update.terminal.status is OperationTerminalStatus.CANCELLED and page._consume_reintegrate_update(update) and reloads==[(request,target)] and operations.reintegrate_state is None and page._context_controller._browse_request is not request
    outcome=_wait(page._context_controller.poll_browse); assert outcome is not None; reloaded_capture=page._capture_current_loaded_browse(); assert reloaded_capture is not None; monkeypatch.setattr(_reintegrate_slot(page),"begin_reintegrate",lambda **_kw:None); page._reintegrate_action("2d"); assert operations.reintegrate_state is None
    foreign=OperationIdentity(101); _reintegrate_slot(page)._identity=foreign; actions={a.action:a for a in page._project_controls(store.snapshot()).actions_for(SectionId.PROCESSING)}; assert not actions[ControlAction.REINTEGRATE_1D].enabled and not actions[ControlAction.REINTEGRATE_2D].enabled; _reintegrate_slot(page)._identity=None; _set_reintegrate_state(page,foreign,reloaded_capture); page.close_workspace(); assert operations.reintegrate_state is None

def test_active_reintegrate_progress_and_cancel_preserve_invalidated_browse_science(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering.adapters import external_operation as module
    from xdart.gui.tabs.scattering.operation_values import OperationTerminalStatus
    from xrd_tools.reduction.reintegrate import ReintegrateCancelled
    from xrd_tools.session.readiness import ControlAction, SectionId
    page,store,_seed,context=_loaded_page(tmp_path,monkeypatch,qapp); entered=threading.Event(); tokens=[]; view=page._shell.scientific
    def painted():
        page._drain_executor(); page._refresh_shell(); projection=page._last_scientific_projection
        return projection if projection is not None and view.trace_history_keys else None
    before=_wait(painted); before_history=view.trace_history_projections
    def cancelled(_target,**kw): tokens.append(kw["cancel_token"]); entered.set(); kw["cancel_token"].wait(2); raise ReintegrateCancelled()
    monkeypatch.setattr(module.ReintegratePlan,"from_artifact",staticmethod(cancelled)); page._reintegrate_action("2d"); identity=page._workspace_operations.reintegrate_identity; request=page._workspace_operations.reintegrate_capture.request
    assert entered.wait(2) and context.invalidated and page._last_scientific_projection is before and view.trace_history_projections==before_history and "authenticated loaded artifact" in page._notice_text
    real_project=page._context_controller.project_browse_1d_cache; invalidated_calls=[]
    def qualified_project(*args,**kwargs):
        browse=page._context_controller.browse_context
        if browse is context and context.invalidated:
            invalidated_calls.append(True); raise AssertionError("invalidated Browse must not be projected")
        return real_project(*args,**kwargs)
    monkeypatch.setattr(page._context_controller,"project_browse_1d_cache",qualified_project); page._refresh_shell(); assert invalidated_calls==[]
    page._drain_executor(); assert page._last_scientific_projection is before and view.trace_history_projections==before_history and page._notice_text.startswith("Reintegrate 2-D: prepare")
    actions={a.action:a for a in page._project_controls(store.snapshot()).actions_for(SectionId.PROCESSING)}; assert actions[ControlAction.REINTEGRATE_2D].label=="Cancel Reintegrate 2-D" and not actions[ControlAction.REINTEGRATE_1D].enabled
    progress_notice=page._notice_text; page._reintegrate_action("1d"); assert page._notice_text==progress_notice and not tokens[0].is_set() and page._last_scientific_projection is before and view.trace_history_projections==before_history
    page._reintegrate_action("2d"); assert tokens[0].is_set() and page._notice_text=="Cancelling Reintegrate 2-D…" and page._last_scientific_projection is before and view.trace_history_projections==before_history
    update=_join(_reintegrate_slot(page),identity); assert update.terminal.status is OperationTerminalStatus.CANCELLED and page._consume_reintegrate_update(update) and page._context_controller._browse_request is not request
    def reloaded():
        page._drain_executor(); captured=page._capture_current_loaded_browse()
        if captured is None:
            assert page._last_scientific_projection is before and view.trace_history_projections==before_history
        return captured
    assert _wait(reloaded) is not None; page.close_workspace()
def test_reintegrate_2d_uses_only_accepted_runtime_route_and_keeps_gui_result_free(tmp_path, monkeypatch, qapp):
    from xdart.gui.tabs.scattering import page as page_module; from xdart.gui.tabs.scattering.adapters import external_operation; from xrd_tools.io.output_transaction import StreamTerminal; from xrd_tools.reduction import ReintegrateResult; from xrd_tools.session.policy import SessionResourceAllocation
    seed=_seed_existing(tmp_path,name="g05"); page,_store,_seed,_context=_loaded_page(tmp_path,monkeypatch,qapp,seed=seed); before={path.name for path in seed.target.parent.iterdir()}; calls=[]
    def forbidden(*_a,**_k): raise AssertionError("forbidden GUI persistence route")
    for name in ("swap_reintegrated_groups","finalize_reintegrated_groups","write_integrated_stack","NexusSink","NexusRecordWriter","REINTEGRATE_SHADOW_SUFFIX"): monkeypatch.setattr(page_module,name,forbidden,raising=False)
    real_build,real_run=external_operation.ReintegratePlan.from_artifact,external_operation.run_reintegrate
    def build(*args,**kwargs): calls.append("build"); return real_build(*args,**kwargs)
    def run(*args,**kwargs): calls.append("run"); return real_run(*args,**kwargs)
    monkeypatch.setattr(external_operation.ReintegratePlan,"from_artifact",staticmethod(build)); monkeypatch.setattr(external_operation,"run_reintegrate",run); _stub_integrators(monkeypatch); page._reintegrate_action("2d"); update=_join(_reintegrate_slot(page),page._workspace_operations.reintegrate_identity); result=update.terminal.payload; assert page._consume_reintegrate_update(update)
    assert calls==["build","run"] and type(result) is ReintegrateResult and type(result.commit_identity) is StreamTerminal and before=={path.name for path in seed.target.parent.iterdir()} and not any(type(value) in {ReintegrateResult,SessionResourceAllocation} for value in vars(page).values()) and not any("__reint" in path.name for path in seed.target.parent.iterdir()); page.close_workspace()
def test_p34b_production_browse_reload_performance_probe(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import browse_loader as module
    from xdart.gui.tabs.scattering.browse_values import BrowseLoadRequest, BrowseLoadStatus
    from xrd_tools.io.output_transaction import capture_target_snapshot
    artifact=os.environ.get("P3_4B_BROWSE_ARTIFACT")
    if not artifact: pytest.skip("requires P3_4B_BROWSE_ARTIFACT")
    target=Path(artifact).resolve(); loads=int(os.environ["P3_4B_BROWSE_LOADS"]); warm=int(os.environ["P3_4B_BROWSE_WARMUPS"]); calls=[]; real=capture_target_snapshot
    monkeypatch.setattr(module,"capture_target_snapshot",lambda path: calls.append(threading.current_thread().name) or real(path),raising=False); walls=[]
    for index in range(loads):
        loader=module.BrowseLoader(max_items=512); request=BrowseLoadRequest(f"perf-{index}",1,str(target)); started=time.perf_counter(); outcome=_wait(lambda: loader.poll(loader.begin(request)),60); context=loader.consume(outcome); walls.append(time.perf_counter()-started)
        assert outcome.status is BrowseLoadStatus.READY and tuple(getattr(context,"loaded_labels",context.frame_ids))==tuple(range(651)); assert loader.release_context(context).cleanup_status.value=="cleaned" and loader._active is loader._queued is None
    started=time.perf_counter(); left,right=real(target),real(target); direct=time.perf_counter()-started; measured=walls[warm:]; payload={"walls":walls,"measured":measured,"median":statistics.median(measured),"early":statistics.median(measured[:3]),"late":statistics.median(measured[-3:]),"snapshot_pair_wall":direct,"digest":left.digest,"loader_snapshot_calls":len(calls),"snapshot_threads":calls}
    assert left==right; parent=os.environ.get("P3_4B_BROWSE_PARENT_RESULT")
    if parent:
        baseline=json.loads(Path(parent).read_text()); projected=baseline["median"]+baseline["snapshot_pair_wall"]; assert len(calls)==14 and calls==["scattering-browse"]*14 and payload["median"]<=max(1.2*projected,projected+.5) and payload["late"]<=max(1.2*payload["early"],payload["early"]+.5)
    Path(os.environ["P3_4B_BROWSE_RESULT"]).write_text(json.dumps(payload,sort_keys=True,separators=(",",":")))
