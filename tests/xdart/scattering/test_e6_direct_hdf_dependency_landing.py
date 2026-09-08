"""P-1M direct-container required-dependency landing oracles."""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from threading import Event
import time

import h5py
import numpy as np
import pytest

from tests.xdart.scattering._e2sd_support import write_poni
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import (
    AdmissionFailure,
    SourceCapture,
    SourceFileState,
    StartCapture,
)
from xdart.gui.tabs.scattering.events import CleanupStatus, RequestId
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources import execution_graph as source_graph
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.selection import image_series_spec


_HDF_LAYOUTS = ("external_link", "soft_external", "vds")
_ALL_LAYOUTS = (*_HDF_LAYOUTS, "external_storage")
_CAPTURE_SENTINEL = "P-1M required dependency capture"


def _key(path: Path | str) -> str:
    return source_graph._source_state_key(path)


def _write_graph(
    tmp_path: Path,
    layout: str,
) -> tuple[Path, Path, Path]:
    raw = tmp_path / "raw"
    dependencies = tmp_path / "dependencies"
    raw.mkdir()
    dependencies.mkdir()
    master = raw / "scan_master.h5"
    dependency = dependencies / (
        "detector.raw" if layout == "external_storage" else "detector.h5"
    )
    shape = (2, 4, 5)

    if layout == "external_storage":
        np.arange(np.prod(shape), dtype=np.uint16).tofile(dependency)
        with h5py.File(master, "w") as handle:
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            data = entry.create_group("data")
            data.attrs["NX_class"] = "NXdata"
            data.create_dataset(
                "data",
                shape=shape,
                dtype=np.uint16,
                external=[(str(dependency), 0, dependency.stat().st_size)],
            )
        return raw, master, dependency

    with h5py.File(dependency, "w") as handle:
        handle.create_dataset(
            "/entry/data/data",
            data=np.ones(shape, dtype=np.uint16),
        )
    with h5py.File(master, "w", libver="latest") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        if layout == "external_link":
            data["data_000001"] = h5py.ExternalLink(
                str(dependency),
                "/entry/data/data",
            )
        elif layout == "soft_external":
            handle["selected_detector"] = h5py.ExternalLink(
                str(dependency),
                "/entry/data/data",
            )
            data["data"] = h5py.SoftLink("/selected_detector")
        elif layout == "vds":
            virtual = h5py.VirtualLayout(shape=shape, dtype=np.uint16)
            virtual[:] = h5py.VirtualSource(
                str(dependency),
                "/entry/data/data",
                shape=shape,
            )
            data.create_virtual_dataset("data", virtual)
        else:  # pragma: no cover - finite parameter owns the variants
            raise AssertionError(layout)
    return raw, master, dependency


def _direct_start(
    tmp_path: Path,
    master: Path,
    *,
    request_value: int,
) -> StartCapture:
    poni = tmp_path / f"cal-{request_value}.poni"
    write_poni(poni)
    source = image_series_spec(master)
    request = RequestId(request_value)
    capture = FilesystemSourceAdapter().capture(source, request)
    snapshot = RunIntentStore(
        RunIntent(
            source_spec=source,
            poni_file=str(poni),
            save_path=str(tmp_path / f"processed-{request_value}.nxs"),
            output_mode="Overwrite",
        )
    ).snapshot()
    return StartCapture(request, 1, snapshot, capture)


class _ProbeOwner:
    def __init__(self, owner, action) -> None:
        self._owner = owner
        self._action = action
        self._fired = False

    def __getattr__(self, name):
        return getattr(self._owner, name)

    def probe(self, path):
        result = self._owner.probe(path)
        if (
            not self._fired
            and result.state is ProbeState.READY
            and result.descriptor is not None
        ):
            self._fired = True
            self._action()
        return result


def _after_ready_probe(
    monkeypatch: pytest.MonkeyPatch,
    master: Path,
    action,
) -> _ProbeOwner:
    real_owner = output_preflight.candidate_owner
    selected = real_owner(master)
    assert selected is not None
    proxy = _ProbeOwner(selected, action)

    def owner(path):
        value = real_owner(path)
        return proxy if value is not None and _key(path) == _key(master) else value

    monkeypatch.setattr(output_preflight, "candidate_owner", owner)
    return proxy


def _assert_direct_failure(
    monkeypatch: pytest.MonkeyPatch,
    executor: StandardRunExecutor,
    start: StartCapture,
    expected: type[BaseException],
    *,
    exact_args: tuple[object, ...] | None = None,
) -> BaseException:
    """Use the real direct admission worker and forbid every later owner."""

    errors: list[BaseException] = []
    forbidden: list[str] = []
    settled = Event()
    real_build = executor_module.build_admission_receipt

    def build(*args, **kwargs):
        try:
            return real_build(*args, **kwargs)
        except BaseException as error:
            errors.append(error)
            settled.set()
            raise

    def forbidden_call(name: str):
        def fail(*_args, **_kwargs):
            forbidden.append(name)
            raise AssertionError(f"forbidden post-dependency call: {name}")

        return fail

    monkeypatch.setattr(executor_module, "build_admission_receipt", build)
    monkeypatch.setattr(
        output_preflight,
        "_validate_targets",
        forbidden_call("validate_targets"),
    )
    monkeypatch.setattr(
        output_preflight,
        "inspect_output",
        forbidden_call("inspect_output"),
    )
    monkeypatch.setattr(
        executor_module.TargetLease,
        "acquire",
        classmethod(lambda _cls, _paths: forbidden_call("target_lease")()),
    )
    monkeypatch.setattr(
        executor,
        "_construct",
        forbidden_call("construct"),
    )

    token = executor.begin_admission(start)
    deadline = time.monotonic() + 10.0
    result = None
    while time.monotonic() < deadline:
        result = executor.poll_admission(token)
        if result is not None or settled.is_set():
            break
        time.sleep(0.01)
    assert settled.wait(1.0), "direct dependency admission did not settle"
    assert len(errors) == 1
    error = errors[0]
    assert type(error) is expected
    if exact_args is not None:
        assert error.args == exact_args
    if result is not None:
        assert type(result) is AdmissionFailure
    operation = executor._admission
    if operation is not None:
        assert operation.directory_session is None
        assert operation.target_lease is None
    assert executor._active is None
    assert executor.drain_events() == ()
    assert forbidden == []
    released = executor.release_admission(token)
    assert released.cleanup_status is CleanupStatus.CLEANED
    return error


def _replace_dependency(dependency: Path) -> None:
    replacement = dependency.with_name(f"replacement-{dependency.name}")
    with h5py.File(replacement, "w") as handle:
        handle.create_dataset(
            "/entry/data/data",
            data=np.full((2, 4, 5), 2, dtype=np.uint16),
        )
    os.replace(replacement, dependency)


@pytest.mark.parametrize("layout", _HDF_LAYOUTS)
def test_post_probe_required_dependency_disappearance_is_revision(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MD01/MD08: post-READY disappearance is typed before side effects."""

    _raw, master, dependency = _write_graph(tmp_path, layout)
    start = _direct_start(tmp_path, master, request_value=14010)
    _after_ready_probe(monkeypatch, master, dependency.unlink)
    _assert_direct_failure(
        monkeypatch,
        StandardRunExecutor(join_timeout=2.0),
        start,
        output_preflight.SourceRevisionChanged,
    )


@pytest.mark.parametrize("layout", _ALL_LAYOUTS)
def test_stop_before_required_dependency_capture_has_no_later_io(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MD02/MD08: early Stop precedes capture and every later dependency I/O."""

    _raw, master, dependency = _write_graph(tmp_path, layout)
    start = _direct_start(tmp_path, master, request_value=14020)
    executor = StandardRunExecutor(join_timeout=2.0)
    fired = [False]
    later: list[str] = []

    def stop() -> None:
        if fired[0]:
            return
        fired[0] = True
        operation = executor._admission
        assert operation is not None
        operation.cancelled.set()

    real_trace = source_graph._trace_hdf5_object_dependencies

    def traced(file_path, object_path, **kwargs):
        if _key(file_path) == _key(dependency):
            stop()
        return real_trace(file_path, object_path, **kwargs)

    real_extend = source_graph._extend_hdf5_dataset_dependency_paths

    def extended(dataset, **kwargs):
        if layout == "external_storage":
            stop()
        return real_extend(dataset, **kwargs)

    real_link = source_graph._hdf5_link_file

    def linked(parent, filename):
        value = real_link(parent, filename)
        if layout == "external_link" and _key(value) == _key(dependency):
            stop()
        return value

    real_capture = SourceFileState.capture

    def captured(path):
        operation = executor._admission
        if (
            operation is not None
            and operation.cancelled.is_set()
            and _key(path) == _key(dependency)
        ):
            later.append("SourceFileState.capture")
        return real_capture(path)

    real_file = h5py.File

    def opened(*args, **kwargs):
        operation = executor._admission
        if operation is not None and operation.cancelled.is_set():
            later.append("h5py.File")
        return real_file(*args, **kwargs)

    monkeypatch.setattr(
        source_graph,
        "_trace_hdf5_object_dependencies",
        traced,
    )
    monkeypatch.setattr(
        source_graph,
        "_extend_hdf5_dataset_dependency_paths",
        extended,
    )
    monkeypatch.setattr(source_graph, "_hdf5_link_file", linked)
    monkeypatch.setattr(SourceFileState, "capture", staticmethod(captured))
    monkeypatch.setattr(h5py, "File", opened)

    _assert_direct_failure(
        monkeypatch,
        executor,
        start,
        RuntimeError,
        exact_args=("admission cancelled",),
    )
    assert fired == [True]
    assert later == []


@pytest.mark.parametrize("layout", _HDF_LAYOUTS)
def test_dependency_removed_after_capture_is_revision(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MD03/MD08: capture-then-disappearance is proved drift."""

    _raw, master, dependency = _write_graph(tmp_path, layout)
    start = _direct_start(tmp_path, master, request_value=14030)
    ready = [False]
    fired = [False]
    _after_ready_probe(monkeypatch, master, lambda: ready.__setitem__(0, True))
    real_capture = SourceFileState.capture

    def captured(path):
        state = real_capture(path)
        if ready[0] and not fired[0] and _key(path) == _key(dependency):
            fired[0] = True
            dependency.unlink()
        return state

    monkeypatch.setattr(SourceFileState, "capture", staticmethod(captured))
    _assert_direct_failure(
        monkeypatch,
        StandardRunExecutor(join_timeout=2.0),
        start,
        output_preflight.SourceRevisionChanged,
    )
    assert fired == [True]


@pytest.mark.parametrize("layout", _HDF_LAYOUTS)
def test_proved_dependency_drift_precedes_late_stop(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MD04/MD08: Stop cannot mask an already-proved dependency revision."""

    _raw, master, dependency = _write_graph(tmp_path, layout)
    start = _direct_start(tmp_path, master, request_value=14040)
    executor = StandardRunExecutor(join_timeout=2.0)
    ready = [False]
    replaced = [False]
    stopped = [False]
    _after_ready_probe(monkeypatch, master, lambda: ready.__setitem__(0, True))
    real_capture = SourceFileState.capture

    def captured(path):
        state = real_capture(path)
        if ready[0] and not replaced[0] and _key(path) == _key(dependency):
            replaced[0] = True
            _replace_dependency(dependency)
        return state

    real_same = source_graph._same_followed_source_revision

    def compared(left, right):
        result = real_same(left, right)
        if (
            not result
            and not stopped[0]
            and _key(left.path) == _key(dependency)
        ):
            stopped[0] = True
            operation = executor._admission
            assert operation is not None
            operation.cancelled.set()
        return result

    monkeypatch.setattr(SourceFileState, "capture", staticmethod(captured))
    monkeypatch.setattr(
        source_graph,
        "_same_followed_source_revision",
        compared,
    )
    _assert_direct_failure(
        monkeypatch,
        executor,
        start,
        output_preflight.SourceRevisionChanged,
    )
    assert replaced == [True]
    assert stopped == [True]


@pytest.mark.parametrize("layout", _HDF_LAYOUTS)
def test_generic_required_dependency_capture_error_is_revision(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MD05/MD08: generic required-capture OSError is typed pending."""

    _raw, master, dependency = _write_graph(tmp_path, layout)
    start = _direct_start(tmp_path, master, request_value=14050)
    ready = [False]
    _after_ready_probe(monkeypatch, master, lambda: ready.__setitem__(0, True))
    real_capture = SourceFileState.capture

    def captured(path):
        if ready[0] and _key(path) == _key(dependency):
            raise OSError(_CAPTURE_SENTINEL)
        return real_capture(path)

    monkeypatch.setattr(SourceFileState, "capture", staticmethod(captured))
    _assert_direct_failure(
        monkeypatch,
        StandardRunExecutor(join_timeout=2.0),
        start,
        output_preflight.SourceRevisionChanged,
    )


def test_direct_external_storage_generic_capture_is_revision_before_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """N01: direct external-raw capture failure is typed before side effects."""

    _raw, master, dependency = _write_graph(tmp_path, "external_storage")
    start = _direct_start(tmp_path, master, request_value=14051)
    ready = [False]
    fired = [0]
    capture_error = OSError(_CAPTURE_SENTINEL)
    _after_ready_probe(monkeypatch, master, lambda: ready.__setitem__(0, True))
    real_capture = SourceFileState.capture

    def captured(path):
        if ready[0] and _key(path) == _key(dependency):
            fired[0] += 1
            raise capture_error
        return real_capture(path)

    monkeypatch.setattr(SourceFileState, "capture", staticmethod(captured))
    error = _assert_direct_failure(
        monkeypatch,
        StandardRunExecutor(join_timeout=2.0),
        start,
        output_preflight.SourceRevisionChanged,
        exact_args=(
            f"external HDF5 storage capture is unverifiable: {dependency}",
        ),
    )
    assert error.__cause__ is capture_error
    assert fired == [1]


def test_external_storage_capture_classifier_has_one_ast_owner() -> None:
    """N02: external-storage pending classification has one structural owner."""

    landing_prefix = "external HDF5 storage is still landing: "
    capture_prefix = "external HDF5 storage capture is unverifiable: "
    tree = ast.parse(inspect.getsource(output_preflight))
    parents: dict[ast.AST, ast.AST] = {}
    owners: dict[ast.AST, ast.AST | None] = {}

    def index(
        node: ast.AST,
        *,
        parent: ast.AST | None = None,
        owner: ast.AST | None = None,
    ) -> None:
        if parent is not None:
            parents[node] = parent
        current_owner = node if isinstance(node, ast.FunctionDef) else owner
        owners[node] = current_owner
        for child in ast.iter_child_nodes(node):
            index(child, parent=node, owner=current_owner)

    def descendants(node: ast.AST, ancestor: ast.AST) -> bool:
        current = node
        while current in parents:
            current = parents[current]
            if current is ancestor:
                return True
        return False

    def called_name(call: ast.Call) -> str | None:
        return call.func.id if isinstance(call.func, ast.Name) else None

    def caught_name(handler: ast.ExceptHandler) -> str | None:
        return handler.type.id if isinstance(handler.type, ast.Name) else None

    def source_revision_raise(node: ast.AST, prefix: str) -> bool:
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            return False
        if called_name(node.exc) != "SourceRevisionChanged":
            return False
        return any(
            isinstance(value, ast.Constant) and value.value == prefix
            for value in ast.walk(node.exc)
        )

    index(tree)
    top_level = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    extend_defs = [
        node
        for node in top_level
        if node.name == "_extend_hdf5_dataset_dependency_paths"
    ]
    assert len(extend_defs) == 1
    extend = extend_defs[0]

    remember_calls = [
        node
        for node in ast.walk(extend)
        if isinstance(node, ast.Call)
        and called_name(node) == "_remember_source_state"
        and owners[node] is extend
    ]
    assert len(remember_calls) == 1
    remember_call = remember_calls[0]
    try_ancestors: list[ast.Try] = []
    current = remember_call
    while current in parents and parents[current] is not extend:
        current = parents[current]
        if isinstance(current, ast.Try):
            try_ancestors.append(current)
    assert len(try_ancestors) == 1
    protected_try = try_ancestors[0]
    assert any(
        remember_call is node or descendants(remember_call, node)
        for node in protected_try.body
    )

    missing_handlers = [
        handler
        for handler in protected_try.handlers
        if caught_name(handler) == "FileNotFoundError"
    ]
    os_handlers = [
        handler
        for handler in protected_try.handlers
        if caught_name(handler) == "OSError"
    ]
    assert len(missing_handlers) == 1
    assert len(os_handlers) == 1
    missing_raises = [
        node
        for node in ast.walk(missing_handlers[0])
        if source_revision_raise(node, landing_prefix)
    ]
    assert len(missing_raises) == 1

    os_handler = os_handlers[0]
    assert len(os_handler.body) == 2
    required_branch, non_required_raise = os_handler.body
    assert isinstance(required_branch, ast.If)
    assert isinstance(required_branch.test, ast.Name)
    assert required_branch.test.id == "required"
    assert required_branch.orelse == []
    capture_raises = [
        node
        for node in ast.walk(required_branch)
        if source_revision_raise(node, capture_prefix)
    ]
    assert len(capture_raises) == 1
    assert isinstance(non_required_raise, ast.Raise)
    assert non_required_raise.exc is None

    landing_raises = [
        node
        for node in ast.walk(tree)
        if source_revision_raise(node, landing_prefix)
    ]
    capture_raises = [
        node
        for node in ast.walk(tree)
        if source_revision_raise(node, capture_prefix)
    ]
    assert len(landing_raises) == 1
    assert len(capture_raises) == 1
    assert owners[landing_raises[0]] is extend
    assert owners[capture_raises[0]] is extend

@pytest.mark.parametrize("layout", _HDF_LAYOUTS)
def test_unchanged_missing_dependency_object_is_stable_value_error(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MD06/MD08: unchanged malformed object paths stay terminal."""

    _raw, master, dependency = _write_graph(tmp_path, layout)
    start = _direct_start(tmp_path, master, request_value=14060)
    ready = [False]
    _after_ready_probe(monkeypatch, master, lambda: ready.__setitem__(0, True))
    real_get = h5py.Group.get

    def missing(group, name, *args, **kwargs):
        getlink = bool(kwargs.get("getlink", False))
        if ready[0] and not getlink:
            owner = Path(os.fsdecode(group.file.filename))
            if (
                (_key(owner) == _key(dependency) and str(name) == "data")
                or (
                    layout == "external_link"
                    and _key(owner) == _key(master)
                    and str(name) == "data_000001"
                )
            ):
                return None
        return real_get(group, name, *args, **kwargs)

    monkeypatch.setattr(h5py.Group, "get", missing)
    error = _assert_direct_failure(
        monkeypatch,
        StandardRunExecutor(join_timeout=2.0),
        start,
        ValueError,
    )
    assert not isinstance(error, output_preflight.SourceRevisionChanged)
