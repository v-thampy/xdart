"""P-1K cancellation and raw HDF dependency-boundary oracles."""

from __future__ import annotations

import ast
from contextlib import contextmanager
import inspect
import os
from pathlib import Path
from threading import Event

import h5py
import numpy as np
import pytest

from tests.xdart.scattering.test_e6_live_directory_wait import (
    _admitted_live_group,
)
from tests.xdart.scattering.test_e6_prestamp_inventory_failure import (
    _RejectMap,
    _retarget,
    _write_hardlink_dependency_graph,
)
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.contracts import SourceFileState
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    RunIdentity,
)
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources import execution_graph as source_graph
from xrd_tools.sources.selection import DirectorySourceSpec


_OPEN_SENTINEL = "P-1K stable open sentinel"
_CLOSE_SENTINEL = "P-1K stable close sentinel"
_CAPTURE_SENTINEL = "dependency capture is unverifiable"


def _adaptive_stable_open(path: Path, states: dict, cancelled):
    """Exercise the semantic boundary on both sides of the API correction."""

    helper = source_graph._open_stable_hdf5_dependency
    kwargs = (
        {"cancelled": cancelled}
        if "cancelled" in inspect.signature(helper).parameters
        else {}
    )
    return helper(path, states, **kwargs)


def _seed_topology(path: Path, *, owner: str | None = None):
    topology = source_graph._topology_from_captured_state(
        SourceFileState.capture(path),
        candidate_owner_id=owner,
    )
    return {source_graph._source_state_key(path): topology}


def test_stable_hdf_open_requires_and_forwards_exact_callback() -> None:
    """K-M09: stable-open boundaries require the exact Stop callback."""

    helper = source_graph._open_stable_hdf5_dependency
    parameter = inspect.signature(helper).parameters.get("cancelled")
    assert parameter is not None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty

    tree = ast.parse(inspect.getsource(source_graph))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_open_stable_hdf5_dependency"
    ]
    assert len(calls) == 3
    for call in calls:
        callbacks = [
            keyword.value
            for keyword in call.keywords
            if keyword.arg == "cancelled"
        ]
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], ast.Name)
        assert callbacks[0].id == "cancelled"


def _trace_after(event: Event, name: str, real, trace: list[str]):
    def wrapped(*args, **kwargs):
        if event.is_set():
            trace.append(name)
        return real(*args, **kwargs)

    return wrapped


def test_stop_after_hdf_open_failure_precedes_topology_revalidation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K01: Stop after open failure forbids every validation read."""

    source = tmp_path / "dependency.h5"
    source.write_bytes(b"stable source bytes")
    states = _seed_topology(source, owner="nexus")
    stopped = Event()
    later_io: list[str] = []

    def failing_file(*_args, **_kwargs):
        stopped.set()
        raise OSError(_OPEN_SENTINEL)

    monkeypatch.setattr(h5py, "File", failing_file)
    for name in (
        "_resolve_source_alias",
        "_capture_canonical_source_target",
        "_candidate_owner_id",
    ):
        real = getattr(source_graph, name)
        monkeypatch.setattr(
            source_graph,
            name,
            _trace_after(stopped, name, real, later_io),
        )

    with pytest.raises(RuntimeError, match="admission cancelled"):
        with _adaptive_stable_open(source, states, stopped.is_set):
            raise AssertionError("failed HDF open unexpectedly yielded")
    assert later_io == []


def test_stable_hdf_open_failure_without_stop_remains_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K01 control: unchanged open failures remain terminal."""

    source = tmp_path / "dependency.h5"
    source.write_bytes(b"stable source bytes")
    monkeypatch.setattr(
        h5py,
        "File",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError(_OPEN_SENTINEL)
        ),
    )
    with pytest.raises(ValueError, match=_OPEN_SENTINEL) as error:
        with _adaptive_stable_open(source, {}, lambda: False):
            raise AssertionError("failed HDF open unexpectedly yielded")
    assert not isinstance(error.value, output_preflight.SourceRevisionChanged)


class _FakeHandle:
    def __init__(self, *, on_close=lambda: None, close_error: bool = False):
        self._on_close = on_close
        self._close_error = close_error

    def __enter__(self):
        return self

    def close(self):
        self._on_close()
        if self._close_error:
            raise OSError(_CLOSE_SENTINEL)

    def __exit__(self, _kind, _error, _traceback):
        self.close()
        return False


def test_stop_from_hdf_close_forbids_final_recapture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K02: close observes Stop before any final source recapture."""

    source = tmp_path / "dependency.h5"
    source.write_bytes(b"stable source bytes")
    stopped = Event()
    post_close_captures: list[Path] = []
    real_capture = source_graph._capture_source_topology

    def traced_capture(path, *args, **kwargs):
        if stopped.is_set():
            post_close_captures.append(Path(path))
        return real_capture(path, *args, **kwargs)

    monkeypatch.setattr(
        h5py,
        "File",
        lambda *_args, **_kwargs: _FakeHandle(
            on_close=stopped.set,
            close_error=True,
        ),
    )
    monkeypatch.setattr(
        source_graph,
        "_capture_source_topology",
        traced_capture,
    )
    with pytest.raises(RuntimeError, match="admission cancelled"):
        with _adaptive_stable_open(source, {}, stopped.is_set):
            pass
    assert post_close_captures == []


def test_unchanged_hdf_close_failure_without_stop_remains_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K02 control: unchanged close errors retain their original type."""

    source = tmp_path / "dependency.h5"
    source.write_bytes(b"stable source bytes")
    monkeypatch.setattr(
        h5py,
        "File",
        lambda *_args, **_kwargs: _FakeHandle(close_error=True),
    )
    with pytest.raises(OSError, match=_CLOSE_SENTINEL):
        with _adaptive_stable_open(source, {}, lambda: False):
            pass


def test_hdf_close_mutation_without_stop_is_revision_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K02 control: a post-close revision change remains typed drift."""

    source = tmp_path / "dependency.h5"
    source.write_bytes(b"stable source bytes")

    def mutate() -> None:
        source.write_bytes(source.read_bytes() + b" changed")

    monkeypatch.setattr(
        h5py,
        "File",
        lambda *_args, **_kwargs: _FakeHandle(on_close=mutate),
    )
    with pytest.raises(output_preflight.SourceRevisionChanged):
        with _adaptive_stable_open(source, {}, lambda: False):
            pass


def test_proved_post_close_drift_precedes_late_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K02 control: Stop cannot mask drift proved after a false boundary."""

    source = tmp_path / "dependency.h5"
    source.write_bytes(b"stable source bytes")
    closed = Event()
    stopped = Event()
    real_capture = source_graph._capture_source_topology

    def drifting_capture(path, *args, **kwargs):
        if closed.is_set():
            source.write_bytes(source.read_bytes() + b" changed")
        topology = real_capture(path, *args, **kwargs)
        if not closed.is_set():
            return topology
        stopped.set()
        return topology

    monkeypatch.setattr(
        h5py,
        "File",
        lambda *_args, **_kwargs: _FakeHandle(on_close=closed.set),
    )
    monkeypatch.setattr(
        source_graph,
        "_capture_source_topology",
        drifting_capture,
    )
    with pytest.raises(output_preflight.SourceRevisionChanged):
        with _adaptive_stable_open(source, {}, stopped.is_set):
            pass
    assert stopped.is_set()


def _write_trial_graph(tmp_path: Path, layout: str) -> tuple[Path, str]:
    raw = tmp_path / "raw"
    landing = tmp_path / "landing"
    raw.mkdir()
    landing.mkdir()
    if layout == "apstools":
        master = raw / "scan_0001.nxs"
        sidecar = landing / "pixels.h5"
        with h5py.File(sidecar, "w") as handle:
            data = handle.create_group("entry").create_group("data")
            data.attrs["NX_class"] = "NXdata"
            pixels = data.create_dataset(
                "pixels",
                data=np.ones((3, 4, 5), dtype=np.uint16),
            )
            pixels.attrs["signal_type"] = "detector"
        with h5py.File(master, "w") as handle:
            handle.attrs["creator"] = "NXWriter"
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            entry.create_dataset("end_time", data="complete")
            entry.create_group("instrument").create_group("bluesky")
            handle["selected_nxdata"] = h5py.ExternalLink(
                str(sidecar),
                "/entry/data",
            )
            entry["data"] = h5py.SoftLink("/selected_nxdata")
        return raw, ".nxs"

    master = raw / "scan_master.h5"
    entry_file = landing / "entry.h5"
    with h5py.File(entry_file, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
    with h5py.File(master, "w") as handle:
        handle["entry"] = h5py.ExternalLink(str(entry_file), "/entry")
    return raw, ".h5"


@pytest.mark.parametrize("layout", ("apstools", "external_root"))
def test_trial_verification_uses_exact_callback_before_filesystem_work(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K03: real trial copies cancel before verification or promotion."""

    raw, suffix = _write_trial_graph(tmp_path, layout)
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=(suffix,), metadata_format=None),
        request_value=13030 + (0 if layout == "apstools" else 1),
    )
    stopped = Event()
    later_io: list[str] = []
    trial_callbacks: list[object] = []
    verifier_returned = [False]
    open_count = 0
    real_open = source_graph._open_stable_hdf5_dependency
    real_verify = source_graph._verify_source_states

    def cancelled() -> bool:
        return stopped.is_set()

    @contextmanager
    def traced_open(path, states, **kwargs):
        nonlocal open_count
        open_count += 1
        selected = open_count
        accepted = (
            kwargs
            if "cancelled" in inspect.signature(real_open).parameters
            else {}
        )
        with real_open(path, states, **accepted) as handle:
            yield handle
        if selected == 2:
            stopped.set()

    def traced_verify(states, **kwargs):
        active = stopped.is_set()
        if active:
            trial_callbacks.append(kwargs.get("cancelled"))
        result = real_verify(states, **kwargs)
        if active:
            verifier_returned[0] = True
        return result

    monkeypatch.setattr(
        source_graph,
        "_open_stable_hdf5_dependency",
        traced_open,
    )
    monkeypatch.setattr(
        source_graph,
        "_verify_source_states",
        traced_verify,
    )
    for name in (
        "_resolve_source_alias",
        "_capture_canonical_source_target",
        "_candidate_owner_id",
        "_capture_source_topology",
    ):
        real = getattr(source_graph, name)
        monkeypatch.setattr(
            source_graph,
            name,
            _trace_after(stopped, name, real, later_io),
        )
    try:
        with pytest.raises(RuntimeError, match="admission cancelled"):
            output_preflight.materialize_live_directory_group(
                receipt,
                intent.freeze(),
                session,
                group,
                cancelled=cancelled,
            )
        assert open_count >= 2
        assert trial_callbacks and trial_callbacks[0] is cancelled
        assert verifier_returned == [False]
        assert later_io == []
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def _assert_live_pending_fence(
    executor,
    intent,
    receipt,
    session,
    group,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden: list[str] = []
    attempts = []
    errors: list[BaseException] = []
    outcome = Event()
    execution_gate = Event()
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)
    real_execute_live = executor._execute_live_directory
    real_materialize = executor_module.materialize_live_directory_group

    def forbidden_call(name: str):
        def fail(*_args, **_kwargs):
            forbidden.append(name)
            raise AssertionError(f"forbidden post-validation call: {name}")

        return fail

    def gated_execute_live(*args, **kwargs):
        if not execution_gate.wait(5.0):
            raise AssertionError("Live P-1K gate was not released")
        return real_execute_live(*args, **kwargs)

    def capture_outcome(*args, **kwargs):
        try:
            attempt = real_materialize(*args, **kwargs)
            attempts.append(attempt)
            if attempt.state is ProbeState.READY:
                forbidden.append("ready_publication")
                raise AssertionError("pre-stamp drift published READY")
            if attempt.revision_changed:
                executor.stop(identity)
        except BaseException as error:
            errors.append(error)
            outcome.set()
            raise
        outcome.set()
        return attempt

    monkeypatch.setattr(
        output_preflight,
        "inspect_output",
        forbidden_call("inspect_output"),
    )
    monkeypatch.setattr(
        executor_module,
        "materialize_live_directory_group",
        capture_outcome,
    )
    monkeypatch.setattr(executor, "_execute_live_directory", gated_execute_live)
    monkeypatch.setattr(executor, "_construct", forbidden_call("construct"))
    monkeypatch.setattr(
        executor_module.TargetLease,
        "acquire",
        classmethod(lambda _cls, _paths: forbidden_call("target_lease")()),
    )
    try:
        accepted = executor.start(
            configuration,
            receipt.source_capture,
            identity,
            receipt,
        )
        assert type(accepted) is ExecutorAccepted
        run = executor._exact_run(identity)
        assert run is not None
        run.processed_live_revisions = _RejectMap(forbidden)
        run.deferred_live_revisions = _RejectMap(forbidden)
        execution_gate.set()
        assert outcome.wait(10.0), "Live P-1K materialization did not settle"

        assert errors == []
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt.state is ProbeState.IN_PROGRESS
        assert attempt.revision_changed is True
        assert attempt.decision is None
        assert forbidden == []
        assert run.processed_live_revisions == {}
        assert run.deferred_live_revisions == {}
        assert not any(
            event.kind in {
                StandardEventKind.CONTEXT_READY,
                StandardEventKind.FRAME_READY,
                StandardEventKind.DISPLAY_READY,
            }
            for event in executor.drain_events()
        )
    finally:
        execution_gate.set()
        executor.stop(identity)
        closed = executor.close(identity)
    assert closed.cleanup_status is CleanupStatus.CLEANED


def test_each_raw_external_link_alias_is_captured_before_canonical_dedup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K04: alias B remains protected even when A names the same object."""

    raw = tmp_path / "raw"
    landing = tmp_path / "landing"
    raw.mkdir()
    landing.mkdir()
    master = raw / "scan_master.h5"
    target = landing / "target-a.h5"
    alternate = landing / "target-b.h5"
    alias_a = landing / "alias-a.h5"
    alias_b = landing / "alias-b.h5"
    with h5py.File(target, "w") as handle:
        handle.create_dataset(
            "/entry/data/data",
            data=np.ones((1, 4, 5), dtype=np.uint16),
        )
    os.link(target, alternate)
    alias_a.symlink_to(target)
    alias_b.symlink_to(target)
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data["data_000001"] = h5py.ExternalLink(
            str(alias_a), "/entry/data/data"
        )
        data["data_000002"] = h5py.ExternalLink(
            str(alias_b), "/entry/data/data"
        )
    executor, intent, receipt, _operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=(".h5",), metadata_format=None),
        request_value=13040,
    )
    real_trace = source_graph._trace_hdf5_object_dependencies
    real_capture = source_graph._capture_source_topology
    captured: list[str] = []
    retargeted = [False]

    def racing_trace(file_path, object_path, **kwargs):
        result = real_trace(file_path, object_path, **kwargs)
        if (
            source_graph._source_state_key(file_path)
            == source_graph._source_state_key(alias_b)
            and not retargeted[0]
        ):
            retargeted[0] = True
            _retarget(alias_b, alternate)
        return result

    def traced_capture(path, *args, **kwargs):
        captured.append(source_graph._source_state_key(path))
        return real_capture(path, *args, **kwargs)

    monkeypatch.setattr(
        source_graph,
        "_trace_hdf5_object_dependencies",
        racing_trace,
    )
    monkeypatch.setattr(
        source_graph,
        "_capture_source_topology",
        traced_capture,
    )
    _assert_live_pending_fence(
        executor, intent, receipt, session, group, monkeypatch
    )
    assert retargeted == [True]
    assert source_graph._source_state_key(alias_a) in captured
    assert source_graph._source_state_key(alias_b) in captured


def test_stop_before_external_storage_capture_has_no_later_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K05: Stop precedes external-storage capture and all later work."""

    raw, _master, _alias, _target, _replacement = (
        _write_hardlink_dependency_graph(tmp_path, "external_storage")
    )
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw, suffixes=("_master.h5",), metadata_format=None
        ),
        request_value=13050,
    )
    stopped = Event()
    later: list[str] = []
    real_extend = source_graph._extend_hdf5_dataset_dependency_paths

    def stopping_extend(dataset, **kwargs):
        stopped.set()
        return real_extend(dataset, **kwargs)

    monkeypatch.setattr(
        source_graph,
        "_extend_hdf5_dataset_dependency_paths",
        stopping_extend,
    )
    for name in (
        "_capture_source_topology",
        "_resolve_source_alias",
        "_capture_canonical_source_target",
        "_candidate_owner_id",
    ):
        real = getattr(source_graph, name)
        monkeypatch.setattr(
            source_graph,
            name,
            _trace_after(stopped, name, real, later),
        )
    real_state_capture = SourceFileState.capture

    def traced_state_capture(path):
        if stopped.is_set():
            later.append("SourceFileState.capture")
        return real_state_capture(path)

    real_file = h5py.File

    def traced_file(*args, **kwargs):
        if stopped.is_set():
            later.append("h5py.File")
        return real_file(*args, **kwargs)

    monkeypatch.setattr(
        SourceFileState,
        "capture",
        staticmethod(traced_state_capture),
    )
    monkeypatch.setattr(h5py, "File", traced_file)
    monkeypatch.setattr(
        output_preflight,
        "inspect_output",
        lambda *_args, **_kwargs: later.append("inspect_output"),
    )
    monkeypatch.setattr(
        executor_module.TargetLease,
        "acquire",
        classmethod(lambda _cls, _paths: later.append("target_lease")),
    )
    try:
        with pytest.raises(RuntimeError, match="admission cancelled"):
            output_preflight.materialize_live_directory_group(
                receipt,
                intent.freeze(),
                session,
                group,
                cancelled=stopped.is_set,
            )
        assert later == []
        assert operation.target_lease is None
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_required_external_storage_capture_oserror_is_pending_without_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """K06: unverifiable required storage capture is typed pending."""

    raw, _master, alias, _target, _replacement = (
        _write_hardlink_dependency_graph(tmp_path, "external_storage")
    )
    executor, intent, receipt, _operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw, suffixes=("_master.h5",), metadata_format=None
        ),
        request_value=13060,
    )
    real_capture = source_graph._capture_source_topology

    def failing_capture(path, *args, **kwargs):
        if (
            source_graph._source_state_key(path)
            == source_graph._source_state_key(alias)
        ):
            raise OSError(_CAPTURE_SENTINEL)
        return real_capture(path, *args, **kwargs)

    monkeypatch.setattr(
        source_graph,
        "_capture_source_topology",
        failing_capture,
    )
    _assert_live_pending_fence(
        executor, intent, receipt, session, group, monkeypatch
    )
