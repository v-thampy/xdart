"""P3-4A headless reintegration composition and owner oracle."""

from __future__ import annotations

import inspect
import subprocess
import sys
import threading
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest


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
