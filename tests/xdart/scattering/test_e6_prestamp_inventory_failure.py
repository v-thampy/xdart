"""P-1J pre-stamp source-topology and inventory-failure oracles."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event
import time

import h5py
import numpy as np
import pytest

from tests.xdart.scattering.test_e6_live_directory_wait import (
    _admitted_live_group,
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
from xrd_tools.sources.selection import DirectorySourceSpec


_SCHEMA_SENTINEL = "P-1J stable inventory schema sentinel"


def _write_hardlink_dependency_graph(
    tmp_path: Path,
    dependency_kind: str,
) -> tuple[Path, Path, Path, Path, Path]:
    """Create one real HDF graph with an aliased hard-link target pair."""

    raw = tmp_path / "raw"
    dependencies = tmp_path / "dependencies"
    raw.mkdir()
    dependencies.mkdir()
    master = raw / f"{dependency_kind}_master.h5"

    if dependency_kind == "external_storage":
        target = dependencies / "target-a.raw"
        replacement = dependencies / "target-b.raw"
        alias = dependencies / "detector-alias.raw"
        np.arange(16, dtype=np.uint16).tofile(target)
        os.link(target, replacement)
        alias.symlink_to(target)
        with h5py.File(master, "w") as handle:
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            data = entry.create_group("data")
            data.attrs["NX_class"] = "NXdata"
            data.create_dataset(
                "data",
                shape=(1, 4, 4),
                dtype=np.uint16,
                external=[(str(alias), 0, target.stat().st_size)],
            )
        return raw, master, alias, target, replacement

    target = dependencies / "target-a.h5"
    replacement = dependencies / "target-b.h5"
    alias = dependencies / "detector-alias.h5"
    with h5py.File(target, "w") as handle:
        handle.create_dataset(
            "/entry/data/data",
            data=np.arange(16, dtype=np.uint16).reshape(1, 4, 4),
        )
    os.link(target, replacement)
    alias.symlink_to(target)
    with h5py.File(master, "w", libver="latest") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        if dependency_kind == "external_link":
            data["data_000001"] = h5py.ExternalLink(
                str(alias),
                "/entry/data/data",
            )
        elif dependency_kind == "vds":
            layout = h5py.VirtualLayout(shape=(1, 4, 4), dtype=np.uint16)
            layout[:] = h5py.VirtualSource(
                str(alias),
                "/entry/data/data",
                shape=(1, 4, 4),
            )
            data.create_virtual_dataset("data", layout)
        else:  # pragma: no cover - finite parameter owns the variants
            raise AssertionError(dependency_kind)
    return raw, master, alias, target, replacement


def _retarget(alias: Path, target: Path) -> None:
    replacement = alias.with_name(f".{alias.name}.replacement")
    replacement.symlink_to(target)
    replacement.replace(alias)


def _install_post_discovery_failure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    dependency_kind: str,
    alias: Path,
    action,
    before_raise=lambda: None,
    exception_factory=lambda: ValueError(_SCHEMA_SENTINEL),
) -> list[bool]:
    """Raise once after the real format-specific discovery helper returns."""

    fired = [False]
    alias_key = output_preflight._source_state_key(alias)

    def trigger() -> None:
        if fired[0]:
            return
        fired[0] = True
        action()
        before_raise()
        raise exception_factory()

    if dependency_kind == "external_storage":
        real_extend = output_preflight._extend_hdf5_dataset_dependency_paths

        def failing_extend(dataset, *, paths, **kwargs):
            result = real_extend(dataset, paths=paths, **kwargs)
            if any(
                output_preflight._source_state_key(path) == alias_key
                for path in paths
            ):
                trigger()
            return result

        monkeypatch.setattr(
            output_preflight,
            "_extend_hdf5_dataset_dependency_paths",
            failing_extend,
        )
    else:
        real_trace = output_preflight._trace_hdf5_object_dependencies

        def failing_trace(file_path, object_path, **kwargs):
            result = real_trace(file_path, object_path, **kwargs)
            if output_preflight._source_state_key(file_path) == alias_key:
                trigger()
            return result

        monkeypatch.setattr(
            output_preflight,
            "_trace_hdf5_object_dependencies",
            failing_trace,
        )
    return fired


class _RejectMap(dict):
    def __init__(self, forbidden: list[str]) -> None:
        super().__init__()
        self._forbidden = forbidden

    def __setitem__(self, _key, _value) -> None:
        self._forbidden.append("live_map_insertion")
        raise AssertionError("pre-stamp drift reached a Live revision map")


@pytest.mark.parametrize(
    "dependency_kind",
    ("external_link", "vds", "external_storage"),
)
def test_prestamp_inventory_schema_failure_with_hardlink_retarget_is_pending(
    dependency_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """J01: topology drift wins before every Live output side effect."""

    raw, _master, alias, _target, replacement = (
        _write_hardlink_dependency_graph(tmp_path, dependency_kind)
    )
    executor, intent, receipt, _operation, _session, _group = (
        _admitted_live_group(
            tmp_path,
            DirectorySourceSpec(
                raw,
                suffixes=("_master.h5",),
                metadata_format=None,
            ),
            request_value={
                "external_link": 12001,
                "vds": 12002,
                "external_storage": 12003,
            }[dependency_kind],
        )
    )
    fired = _install_post_discovery_failure(
        monkeypatch,
        dependency_kind=dependency_kind,
        alias=alias,
        action=lambda: _retarget(alias, replacement),
    )
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
            raise AssertionError("Live P-1J gate was not released")
        return real_execute_live(*args, **kwargs)

    def capture_outcome(*args, **kwargs):
        try:
            attempt = real_materialize(*args, **kwargs)
        except BaseException as error:
            errors.append(error)
            outcome.set()
            raise
        attempts.append(attempt)
        if attempt.state is ProbeState.READY:
            forbidden.append("ready_publication")
            outcome.set()
            raise AssertionError("pre-stamp drift published READY")
        if attempt.revision_changed:
            executor.stop(identity)
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
        assert outcome.wait(10.0), (
            "Live P-1J materialization returned neither a result nor an "
            "exception"
        )

        assert errors == [], f"unexpected terminal inventory error: {errors!r}"
        assert fired == [True]
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


@pytest.mark.parametrize(
    ("dependency_kind", "refresh"),
    (
        ("external_link", False),
        ("vds", False),
        ("external_storage", False),
        ("external_link", True),
    ),
)
def test_prestamp_unchanged_inventory_schema_failure_remains_terminal(
    dependency_kind: str,
    refresh: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """J02: unchanged topology and same-target refresh stay terminal."""

    raw, _master, alias, target, _replacement = (
        _write_hardlink_dependency_graph(tmp_path, dependency_kind)
    )
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw,
            suffixes=("_master.h5",),
            metadata_format=None,
        ),
        request_value=12010 + len(dependency_kind) + int(refresh),
    )
    fired = _install_post_discovery_failure(
        monkeypatch,
        dependency_kind=dependency_kind,
        alias=alias,
        action=(lambda: _retarget(alias, target)) if refresh else (lambda: None),
    )
    forbidden: list[str] = []

    def fail(name: str):
        def forbidden_call(*_args, **_kwargs):
            forbidden.append(name)
            raise AssertionError(f"stable schema error reached {name}")
        return forbidden_call

    monkeypatch.setattr(output_preflight, "inspect_output", fail("inspect_output"))
    monkeypatch.setattr(
        executor_module.TargetLease,
        "acquire",
        classmethod(lambda _cls, _paths: fail("target_lease")()),
    )
    try:
        with pytest.raises(ValueError, match=_SCHEMA_SENTINEL):
            output_preflight.materialize_live_directory_group(
                receipt,
                intent.freeze(),
                session,
                group,
                cancelled=lambda: False,
            )
        assert fired == [True]
        assert forbidden == []
        assert operation.target_lease is None
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


@pytest.mark.parametrize("cancel_at", (None, 4, 5, 6, 7))
def test_prestamp_inventory_failure_two_sweep_order_and_cancellation(
    cancel_at: int | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """J03: failure classification has exact two-sweep Stop ordering."""

    raw, master, alias, _target, _replacement = (
        _write_hardlink_dependency_graph(tmp_path, "external_link")
    )
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw,
            suffixes=("_master.h5",),
            metadata_format=None,
        ),
        request_value=12020 + (0 if cancel_at is None else cancel_at),
    )
    traces: list[str] = []
    armed = False
    cancel_calls = 0

    real_resolve = output_preflight._resolve_source_alias
    real_capture = output_preflight._capture_canonical_source_target
    real_owner = output_preflight._candidate_owner_id

    def before_raise() -> None:
        nonlocal armed, cancel_calls
        traces.clear()
        cancel_calls = 0
        armed = True

    def cancelled() -> bool:
        nonlocal cancel_calls
        if not armed:
            return False
        cancel_calls += 1
        traces.append("cancel")
        return cancel_at == cancel_calls

    def traced_resolve(path: str) -> str:
        if armed:
            traces.append(f"resolve:{Path(path).name}")
        return real_resolve(path)

    def traced_capture(path: str) -> SourceFileState:
        if armed:
            traces.append(f"capture:{Path(path).name}")
        return real_capture(path)

    def traced_owner(path: str) -> str | None:
        if armed:
            traces.append(f"owner:{Path(path).name}")
        return real_owner(path)

    monkeypatch.setattr(output_preflight, "_resolve_source_alias", traced_resolve)
    monkeypatch.setattr(
        output_preflight,
        "_capture_canonical_source_target",
        traced_capture,
    )
    monkeypatch.setattr(output_preflight, "_candidate_owner_id", traced_owner)
    _install_post_discovery_failure(
        monkeypatch,
        dependency_kind="external_link",
        alias=alias,
        action=lambda: None,
        before_raise=before_raise,
    )
    try:
        if cancel_at is None:
            with pytest.raises(ValueError, match=_SCHEMA_SENTINEL):
                output_preflight.materialize_live_directory_group(
                    receipt,
                    intent.freeze(),
                    session,
                    group,
                    cancelled=cancelled,
                )
            assert traces == [
                "cancel",
                "cancel",
                "cancel",
                "cancel",
                f"resolve:{master.name}",
                f"capture:{master.name}",
                f"owner:{master.name}",
                "cancel",
                f"resolve:{alias.name}",
                f"capture:{alias.resolve().name}",
                "cancel",
                f"resolve:{master.name}",
                f"capture:{master.name}",
                "cancel",
                f"resolve:{alias.name}",
                f"capture:{alias.resolve().name}",
                "cancel",
            ]
        else:
            with pytest.raises(RuntimeError, match="admission cancelled"):
                output_preflight.materialize_live_directory_group(
                    receipt,
                    intent.freeze(),
                    session,
                    group,
                    cancelled=cancelled,
                )
            expected = {
                4: ["cancel", "cancel", "cancel", "cancel"],
                5: [
                    "cancel",
                    "cancel",
                    "cancel",
                    "cancel",
                    f"resolve:{master.name}",
                    f"capture:{master.name}",
                    f"owner:{master.name}",
                    "cancel",
                ],
                6: [
                    "cancel",
                    "cancel",
                    "cancel",
                    "cancel",
                    f"resolve:{master.name}",
                    f"capture:{master.name}",
                    f"owner:{master.name}",
                    "cancel",
                    f"resolve:{alias.name}",
                    f"capture:{alias.resolve().name}",
                    "cancel",
                ],
                7: [
                    "cancel",
                    "cancel",
                    "cancel",
                    "cancel",
                    f"resolve:{master.name}",
                    f"capture:{master.name}",
                    f"owner:{master.name}",
                    "cancel",
                    f"resolve:{alias.name}",
                    f"capture:{alias.resolve().name}",
                    "cancel",
                    f"resolve:{master.name}",
                    f"capture:{master.name}",
                    "cancel",
                ],
            }
            assert traces == expected[cancel_at]
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_prestamp_exact_cancellation_skips_failure_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """J03 control: an exact discovery cancellation remains immediate."""

    raw, _master, alias, _target, _replacement = (
        _write_hardlink_dependency_graph(tmp_path, "external_link")
    )
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw,
            suffixes=("_master.h5",),
            metadata_format=None,
        ),
        request_value=12030,
    )
    _install_post_discovery_failure(
        monkeypatch,
        dependency_kind="external_link",
        alias=alias,
        action=lambda: None,
        exception_factory=lambda: RuntimeError("admission cancelled"),
    )

    def forbidden_classifier(*_args, **_kwargs):
        raise AssertionError("exact cancellation reached failure classification")

    monkeypatch.setattr(
        output_preflight,
        "_classify_inventory_failure",
        forbidden_classifier,
    )
    try:
        with pytest.raises(RuntimeError, match="admission cancelled"):
            output_preflight.materialize_live_directory_group(
                receipt,
                intent.freeze(),
                session,
                group,
                cancelled=lambda: False,
            )
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED
