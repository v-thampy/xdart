"""Portable P-1I source-alias identity and provenance oracles."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace

import fabio
import h5py
import numpy as np
import pytest

from xdart.gui.tabs.scattering import contracts as contracts_module
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.contracts import (
    AdmittedMetadataSource,
    ExternalSourceState,
    PlannedOutput,
    SourceExecutionStamp,
    SourceFileState,
)
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    RunIdentity,
)
from xdart.gui.tabs.scattering.output_preflight import SourceRevisionChanged
from xrd_tools.core.scan import ScanFrame, SourceKind, SourceSpec
from xrd_tools.io.output_safety import OutputCollisionError
from xrd_tools.reduction.core import FrameReduction, NexusSink
from xrd_tools.sources.descriptor import ContainerDescriptor
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.selection import DirectorySourceSpec

from tests.xdart.scattering.test_e6_live_directory_wait import (
    _add_display_artifact,
    _admitted_live_group,
)


_LEGACY_EXECUTION_KEYS = (
    "path",
    "size",
    "mtime_ns",
    "ctime_ns",
    "device",
    "inode",
    "adapter_id",
    "frame_count",
    "first_label",
    "member_stamps",
    "external_members",
    "dependency_files",
    "admitted_motor_values",
    "metadata_sources",
)


def _write_tiff(path: Path, value: int = 1) -> None:
    fabio.tifimage.TifImage(
        data=np.full((4, 4), value, dtype=np.uint16)
    ).write(str(path))


def _image_item(
    source: Path,
    stamp: SourceExecutionStamp,
    target: Path,
    *,
    descriptor: ContainerDescriptor | None = None,
) -> PlannedOutput:
    return PlannedOutput(
        SourceSpec(source, SourceKind.IMAGE_FILE),
        source,
        target,
        stamp,
        descriptor=descriptor,
    )


def _identity_mapping(stamp: SourceExecutionStamp) -> dict[str, object]:
    identity = stamp.execution_identity_v1
    return identity.as_dict()


def test_aliased_primary_and_target_member_keep_exact_legacy_raw_serialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O01: aliased ``file`` is never lost or canonicalized in legacy data."""

    target = tmp_path / "target_0001.tif"
    replacement = tmp_path / "replacement_0001.tif"
    alias = tmp_path / "selected_0001.tif"
    _write_tiff(target, 1)
    _write_tiff(replacement, 2)
    alias.symlink_to(target)
    raw_alias = SourceFileState.capture(alias)
    raw_target = SourceFileState.capture(target)
    stamp = SourceExecutionStamp(
        raw_alias,
        "tiff_series",
        1,
        1,
        members=(raw_target,),
    )

    before = json.dumps(stamp.as_dict(), separators=(",", ":"))
    alias.unlink()
    alias.symlink_to(replacement)

    def forbidden_capture(_path: Path) -> SourceFileState:
        raise AssertionError("legacy serialization touched the filesystem")

    def forbidden_path_io(*_args, **_kwargs):
        raise AssertionError("constructor/serialization touched the filesystem")

    with monkeypatch.context() as blocked:
        blocked.setattr(
            SourceFileState,
            "capture",
            staticmethod(forbidden_capture),
        )
        blocked.setattr(os.path, "realpath", forbidden_path_io)
        blocked.setattr(Path, "resolve", forbidden_path_io)
        blocked.setattr(Path, "stat", forbidden_path_io)
        persisted = stamp.as_dict()
        synthetic_state = SourceFileState(
            str(tmp_path / "synthetic_0001.tif"),
            1,
            2,
            3,
            4,
            5,
        )
        synthetic_stamp = SourceExecutionStamp(
            synthetic_state,
            "image_file",
            1,
            1,
        )
        assert synthetic_stamp.as_dict()["path"] == synthetic_state.path
    after = json.dumps(persisted, separators=(",", ":"))

    assert tuple(persisted) == _LEGACY_EXECUTION_KEYS
    assert persisted["path"] == str(alias.absolute())
    assert persisted["member_stamps"][0]["path"] == str(target.absolute())
    assert before == after


def test_several_raw_aliases_share_one_target_without_losing_roles_or_identity(
    tmp_path: Path,
) -> None:
    """O02: raw bindings survive canonical de-duplication and drive identity."""

    target = tmp_path / "target_0001.tif"
    hardlink = tmp_path / "hardlink_0001.tif"
    primary_alias = tmp_path / "primary_0001.tif"
    secondary_alias = tmp_path / "secondary_0001.tif"
    _write_tiff(target, 1)
    os.link(target, hardlink)
    primary_alias.symlink_to(target)
    secondary_alias.symlink_to(target)

    primary = SourceFileState.capture(primary_alias)
    member = SourceFileState.capture(target)
    secondary = SourceFileState.capture(secondary_alias)
    stamp = SourceExecutionStamp(
        primary,
        "tiff_series",
        1,
        1,
        members=(member,),
        external_members=(
            ExternalSourceState(secondary, "/entry/data", 0, 1, 0),
        ),
        dependency_files=(secondary,),
        metadata_sources=(AdmittedMetadataSource(member.path, member),),
    )

    identity = _identity_mapping(stamp)
    aliases = identity["aliases"]
    targets = identity["targets"]
    assert [value["raw_path"] for value in aliases] == [
        str(primary_alias.absolute()),
        str(target.absolute()),
        str(secondary_alias.absolute()),
    ]
    assert [tuple(value["roles"]) for value in aliases] == [
        ("source_file",),
        ("source_member", "image_metadata"),
        ("external_member", "detector_dependency"),
    ]
    assert len(targets) == 1
    assert tuple(targets[0]["roles"]) == (
        "source_file",
        "source_member",
        "external_member",
        "detector_dependency",
        "image_metadata",
    )

    # Retargeting to a different pathname of the same inode preserves every
    # legacy SourceFileState value while changing the resolved alias topology.
    secondary_alias.unlink()
    secondary_alias.symlink_to(hardlink)
    revised_secondary = SourceFileState.capture(secondary_alias)
    assert revised_secondary == secondary
    revised = replace(
        stamp,
        external_members=(
            ExternalSourceState(
                revised_secondary,
                "/entry/data",
                0,
                1,
                0,
            ),
        ),
        dependency_files=(revised_secondary,),
    )
    assert revised == stamp
    assert revised.execution_identity_v1 != stamp.execution_identity_v1


def test_conflicting_states_for_one_canonical_target_are_refused_at_freeze(
    tmp_path: Path,
) -> None:
    """O03: one resolved target cannot be first/last-wins rebaselined."""

    selected = tmp_path / "scan_0001.tif"
    alias = tmp_path / "scan_alias_0001.tif"
    _write_tiff(selected)
    alias.symlink_to(selected)
    accepted = SourceFileState.capture(selected)
    conflicting = SourceFileState.capture(alias)
    object.__setattr__(conflicting, "size", accepted.size + 1)

    with pytest.raises(ValueError, match="conflicting .*source .*state"):
        SourceExecutionStamp(
            accepted,
            "image_file",
            1,
            1,
            dependency_files=(conflicting,),
        )


def test_replacing_symlink_with_same_target_and_owner_is_not_drift(
    tmp_path: Path,
) -> None:
    """O04: link-entry refresh is not an lstat-based false revision."""

    target = tmp_path / "target_0001.tif"
    alias = tmp_path / "selected_0001.tif"
    refreshed = tmp_path / "refreshed_0001.tif"
    _write_tiff(target)
    alias.symlink_to(target)
    stamp = SourceExecutionStamp(
        SourceFileState.capture(alias),
        "image_file",
        1,
        1,
    )
    item = _image_item(alias, stamp, tmp_path / "processed.nxs")
    identity = (
        stamp.execution_identity_v1
        if hasattr(stamp, "execution_identity_v1")
        else stamp.strong_source_states
    )

    refreshed.symlink_to(target)
    refreshed.replace(alias)

    output_preflight.validate_planned_source(item)
    current_identity = (
        stamp.execution_identity_v1
        if hasattr(stamp, "execution_identity_v1")
        else stamp.strong_source_states
    )
    assert current_identity == identity


@pytest.mark.parametrize(
    "change",
    ("raw_unlinked", "broken_link", "different_target"),
)
def test_alias_validator_rejects_missing_broken_and_retargeted_aliases(
    change: str,
    tmp_path: Path,
) -> None:
    """O05: every admitted raw locator remains bound to one pathname."""

    target = tmp_path / "target_0001.tif"
    hardlink = tmp_path / "hardlink_0001.tif"
    alias = tmp_path / "selected_0001.tif"
    _write_tiff(target)
    os.link(target, hardlink)
    alias.symlink_to(target)
    stamp = SourceExecutionStamp(
        SourceFileState.capture(alias),
        "image_file",
        1,
        1,
    )
    item = _image_item(alias, stamp, tmp_path / "processed.nxs")

    alias.unlink()
    if change == "broken_link":
        alias.symlink_to(tmp_path / "missing.tif")
    elif change == "different_target":
        # A different resolved pathname is drift even when samefile/stat says
        # that the hard-link target is physically equivalent.
        alias.symlink_to(hardlink)

    with pytest.raises(SourceRevisionChanged):
        output_preflight.validate_planned_source(item)


@pytest.mark.parametrize(
    "changed_field",
    ("size", "mtime_ns", "ctime_ns", "device", "inode", "owner", "missing_owner", None),
)
def test_alias_validator_rejects_target_revision_and_candidate_owner_change(
    changed_field: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O06: all revision fields and raw-path ownership are fail-closed."""

    selected = tmp_path / "selected_0001.tif"
    _write_tiff(selected)
    accepted = SourceFileState.capture(selected)
    stamp = SourceExecutionStamp(accepted, "image_file", 1, 1)
    item = _image_item(selected, stamp, tmp_path / "processed.nxs")

    if changed_field in {"size", "mtime_ns", "ctime_ns", "device", "inode"}:
        changed = replace(
            accepted,
            **{changed_field: getattr(accepted, changed_field) + 1},
        )
        monkeypatch.setattr(
            SourceFileState,
            "capture",
            staticmethod(lambda _path: changed),
        )
    elif changed_field in {"owner", "missing_owner"}:
        owner = None if changed_field == "missing_owner" else SimpleNamespace(id="nexus")
        monkeypatch.setattr(output_preflight, "candidate_owner", lambda _path: owner)

    if changed_field is None:
        output_preflight.validate_planned_source(item)
    else:
        with pytest.raises(SourceRevisionChanged):
            output_preflight.validate_planned_source(item)


def test_sidecar_alias_retarget_is_pending_before_every_output_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O07: the frozen retarget race stops before a READY decision."""

    raw = tmp_path / "raw"
    raw.mkdir()
    first = raw / "scan_0001.tif"
    second = raw / "scan_0002.tif"
    old_sidecar = raw / "old-sidecar.txt"
    new_sidecar = raw / "new-sidecar.txt"
    first_link = first.with_suffix(".txt")
    for index, image in enumerate((first, second), start=1):
        _write_tiff(image, index)
    old_sidecar.write_text(
        "th=0.1\nsequence=1\nexposure=0.5\n",
        encoding="utf-8",
    )
    new_sidecar.write_text(
        "th=0.1\nsequence=1\nexposure=0.5\n",
        encoding="utf-8",
    )
    first_link.symlink_to(old_sidecar)
    second.with_suffix(".txt").write_text(
        "th=0.2\nsequence=2\n",
        encoding="utf-8",
    )
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="auto"),
        request_value=1107,
    )
    real_read = output_preflight.read_image_motor_metadata
    retargeted = False
    forbidden: list[str] = []

    def retarget_earlier_alias(path, *args, **kwargs):
        nonlocal retargeted
        if Path(path) == second and not retargeted:
            first_link.unlink()
            first_link.symlink_to(new_sidecar)
            retargeted = True
        return real_read(path, *args, **kwargs)

    def forbidden_call(name: str):
        def fail(*_args, **_kwargs):
            forbidden.append(name)
            raise AssertionError(f"forbidden post-validation call: {name}")
        return fail

    monkeypatch.setattr(
        output_preflight,
        "read_image_motor_metadata",
        retarget_earlier_alias,
    )
    real_execute_live = executor._execute_live_directory
    execution_gate = Event()

    def gated_execute_live(*args, **kwargs):
        if not execution_gate.wait(5.0):
            raise AssertionError("Live side-effect gate was not released")
        return real_execute_live(*args, **kwargs)

    real_materialize = executor_module.materialize_live_directory_group
    attempts = []
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)

    def capture_pending_attempt(*args, **kwargs):
        attempt = real_materialize(*args, **kwargs)
        attempts.append(attempt)
        if attempt.state is ProbeState.READY:
            forbidden.append("ready_publication")
            raise AssertionError("alias drift published READY")
        if attempt.revision_changed:
            executor.stop(identity)
        return attempt

    class RejectMap(dict):
        def __setitem__(self, _key, _value):
            forbidden.append("live_map_insertion")
            raise AssertionError("alias drift reached a Live revision map")

    monkeypatch.setattr(
        output_preflight,
        "inspect_output",
        forbidden_call("inspect_output"),
    )
    monkeypatch.setattr(
        executor_module,
        "materialize_live_directory_group",
        capture_pending_attempt,
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
        run.processed_live_revisions = RejectMap()
        run.deferred_live_revisions = RejectMap()
        execution_gate.set()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not attempts and not forbidden:
            time.sleep(0.01)

        assert retargeted is True
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


def test_live_pending_identity_revalidates_before_equal_candidate_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O02 integration: a cheap-equal candidate cannot hide alias topology."""

    raw = tmp_path / "raw"
    raw.mkdir()
    target = tmp_path / "target-a.tif"
    hardlink = tmp_path / "target-b.tif"
    selected = raw / "scan_0001.tif"
    _write_tiff(target)
    os.link(target, hardlink)
    selected.symlink_to(target)
    executor, intent, receipt, _operation, _session, _group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format=None),
        request_value=1122,
    )
    attempts = []
    processed = []
    real_materialize = executor_module.materialize_live_directory_group

    def capture_attempt(*args, **kwargs):
        attempt = real_materialize(*args, **kwargs)
        if attempt.decision is not None:
            attempts.append(attempt)
        return attempt

    def construct(run, *, item, labels, decision):
        _add_display_artifact(run, item)
        run.artifact = item.target
        run.current_total = item.source_stamp.frame_count
        run.current_completed = run.current_total - len(labels)
        run.current_published = run.current_completed
        processed.append(item.source_path)
        return run

    def execute_current(run, *, construct=False):
        assert construct is False
        added = run.current_total - run.current_completed
        run.completed += added
        run.current_completed += added
        run.current_published += added
        if run.artifact not in run.artifacts:
            run.artifacts.append(run.artifact)
        return False

    monkeypatch.setattr(
        executor_module,
        "materialize_live_directory_group",
        capture_attempt,
    )
    monkeypatch.setattr(executor, "_construct", construct)
    monkeypatch.setattr(executor, "_execute_current", execute_current)
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)
    assert type(executor.start(
        configuration,
        receipt.source_capture,
        identity,
        receipt,
    )) is ExecutorAccepted
    try:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not executor.processed_live_revisions(identity):
            time.sleep(0.01)
        original = executor.processed_live_revisions(identity)
        assert len(original) == 1

        replacement = raw / "replacement.tif"
        replacement.symlink_to(hardlink)
        replacement.replace(selected)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and not executor.deferred_live_revisions(identity):
            time.sleep(0.01)
        deferred = executor.deferred_live_revisions(identity)
        assert len(deferred) == 1
        assert len(attempts) >= 2
        first, revised = original[0], deferred[0]
        assert first.group.revision == revised.group.revision
        assert first.decision is not None and revised.decision is not None
        assert first.decision.item.source_stamp == revised.decision.item.source_stamp
        assert (
            first.decision.item.source_stamp.execution_identity_v1
            != revised.decision.item.source_stamp.execution_identity_v1
        )
        assert processed == [selected]
    finally:
        executor.stop(identity)
        closed = executor.close(identity)
    assert closed.cleanup_status is CleanupStatus.CLEANED


@pytest.mark.parametrize("dependency_kind", ("external_link", "vds", "external_storage"))
def test_hdf_dependency_raw_alias_retarget_is_caught_before_decision(
    dependency_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O08: HDF5 dependencies preserve lexical aliases through final sweep."""

    raw = tmp_path / "raw"
    raw.mkdir()
    dependency_root = tmp_path / "dependencies"
    dependency_root.mkdir()
    master, alias, replacement, _witness = _write_hdf_alias_graph(
        raw,
        dependency_root,
        dependency_kind,
    )
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw,
            suffixes=("_master.h5",),
            metadata_format=None,
        ),
        request_value={
            "external_link": 1108,
            "vds": 1118,
            "external_storage": 1128,
        }[dependency_kind],
    )
    retargeted = False
    captured_stamps: list[SourceExecutionStamp] = []
    forbidden: list[str] = []
    real_owner = output_preflight._selected_dependency_files

    def arm_dependency_files(*args, **kwargs):
        nonlocal retargeted
        result = real_owner(*args, **kwargs)
        # Every dependency form is now fully captured and closure-validated,
        # but the complete stamp has not yet been constructed.  Retarget at
        # this exact post-closure/pre-stamp boundary so only capture-time
        # topology plus the final central sweep can reject the rebaseline.
        if not retargeted:
            alias.unlink()
            alias.symlink_to(replacement)
            retargeted = True
        return result

    monkeypatch.setattr(
        output_preflight,
        "_selected_dependency_files",
        arm_dependency_files,
    )

    if hasattr(output_preflight, "validate_source_aliases"):
        real_validator = output_preflight.validate_source_aliases

        def capture_validator(stamp, *args, **kwargs):
            captured_stamps.append(stamp)
            return real_validator(stamp, *args, **kwargs)

        monkeypatch.setattr(
            output_preflight,
            "validate_source_aliases",
            capture_validator,
        )

    def forbid_inspect(item, *_args, **_kwargs):
        captured_stamps.append(item.source_stamp)
        forbidden.append("inspect_output")
        raise AssertionError("HDF alias retarget reached output decision")

    monkeypatch.setattr(output_preflight, "inspect_output", forbid_inspect)
    try:
        attempt = output_preflight.materialize_live_directory_group(
            receipt,
            intent.freeze(),
            session,
            group,
            cancelled=lambda: False,
        )
        assert retargeted is True
        assert attempt.state is ProbeState.IN_PROGRESS
        assert attempt.revision_changed is True
        assert attempt.decision is None
        assert forbidden == []
        assert captured_stamps
        stamp = captured_stamps[-1]
        raw_alias = str(alias.absolute())
        legacy = stamp.as_dict()
        legacy_dependencies = [
            value["file"]["path"]
            for value in legacy["external_members"]
        ] + [value["path"] for value in legacy["dependency_files"]]
        assert raw_alias in legacy_dependencies
        identity = _identity_mapping(stamp)
        assert raw_alias in {
            value["raw_path"] for value in identity["aliases"]
        }
        assert operation.target_lease is None
        assert master.exists()
    finally:
        executor.cancel_admission(operation.token)


def _write_hdf_alias_graph(
    raw: Path,
    dependency_root: Path,
    dependency_kind: str,
) -> tuple[Path, Path, Path, Path]:
    """Write one two-dependency graph whose first locator is a symlink."""

    master = raw / f"{dependency_kind}_master.h5"
    alias = dependency_root / "first-alias.h5"
    if dependency_kind == "external_storage":
        first = dependency_root / "first.raw"
        replacement = dependency_root / "replacement.raw"
        witness = dependency_root / "witness.raw"
        np.full((1, 4, 4), 1, dtype=np.uint16).tofile(first)
        np.full((1, 4, 4), 2, dtype=np.uint16).tofile(replacement)
        np.full((1, 4, 4), 3, dtype=np.uint16).tofile(witness)
        alias = dependency_root / "first-alias.raw"
        alias.symlink_to(first)
        frame_bytes = int(np.zeros((1, 4, 4), dtype=np.uint16).nbytes)
        with h5py.File(master, "w") as handle:
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            data = entry.create_group("data")
            data.attrs["NX_class"] = "NXdata"
            data.create_dataset(
                "data",
                shape=(2, 4, 4),
                dtype=np.uint16,
                external=[
                    (str(alias), 0, frame_bytes),
                    (str(witness), 0, frame_bytes),
                ],
            )
        return master, alias, replacement, witness

    first = dependency_root / "first.h5"
    replacement = dependency_root / "replacement.h5"
    witness = dependency_root / "witness.h5"
    for index, path in enumerate((first, replacement, witness), start=1):
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "/entry/data/data",
                data=np.full((1, 4, 4), index, dtype=np.uint16),
            )
    alias.symlink_to(first)
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
            data["data_000002"] = h5py.ExternalLink(
                str(witness),
                "/entry/data/data",
            )
        elif dependency_kind == "vds":
            layout = h5py.VirtualLayout(shape=(2, 4, 4), dtype=np.uint16)
            layout[0:1] = h5py.VirtualSource(
                str(alias),
                "/entry/data/data",
                shape=(1, 4, 4),
            )
            layout[1:2] = h5py.VirtualSource(
                str(witness),
                "/entry/data/data",
                shape=(1, 4, 4),
            )
            data.create_virtual_dataset("data", layout)
        else:  # pragma: no cover - finite parameter owns the variants
            raise AssertionError(dependency_kind)
    return master, alias, replacement, witness


def test_hardlink_locators_remain_distinct_and_each_blocks_output_collision(
    tmp_path: Path,
) -> None:
    """O09: hard-link names never collapse, and samefile blocks publication."""

    first = tmp_path / "first_0001.tif"
    second = tmp_path / "second_0001.tif"
    third = tmp_path / "third_0001.tif"
    _write_tiff(first)
    try:
        os.link(first, second)
        os.link(first, third)
    except OSError as error:
        pytest.skip(f"hard links unavailable: {error}")
    stamp = SourceExecutionStamp(
        SourceFileState.capture(first),
        "image_file",
        1,
        1,
        dependency_files=(SourceFileState.capture(second),),
    )
    item = _image_item(first, stamp, tmp_path / "unrelated.nxs")
    targets = (
        stamp.canonical_targets
        if hasattr(stamp, "canonical_targets")
        else stamp.strong_source_states
    )
    assert len(targets) == 2
    source = item.source_spec
    configuration = SimpleNamespace(poni_file="", mask_file="")
    output_preflight._validate_targets(configuration, source, (item,))

    for collision in (first, second, third):
        with pytest.raises(OutputCollisionError):
            output_preflight._validate_targets(
                configuration,
                source,
                (replace(item, target=collision),),
            )


def test_nullable_metadata_absence_has_no_binding_and_appearance_is_pending(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O10: semantic absence stays None and cannot silently become a file."""

    raw = tmp_path / "raw"
    raw.mkdir()
    image = raw / "scan_0001.tif"
    _write_tiff(image)
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format="auto"),
        request_value=1110,
    )
    real_knowledge = output_preflight._tiff_motor_knowledge
    appeared = False
    forbidden: list[str] = []

    def appear_after_absent_freeze(*args, **kwargs):
        nonlocal appeared
        observed = real_knowledge(*args, **kwargs)
        if not appeared:
            image.with_suffix(".txt").write_text(
                "th=0.1\nsequence=1\nexposure=0.5\n",
                encoding="utf-8",
            )
            appeared = True
        return observed

    def forbid_inspect(*_args, **_kwargs):
        forbidden.append("inspect_output")
        raise AssertionError("nullable metadata appearance reached decision")

    monkeypatch.setattr(output_preflight, "_tiff_motor_knowledge", appear_after_absent_freeze)
    monkeypatch.setattr(output_preflight, "inspect_output", forbid_inspect)
    try:
        attempt = output_preflight.materialize_live_directory_group(
            receipt,
            intent.freeze(),
            session,
            group,
            cancelled=lambda: False,
        )
        assert appeared is True
        assert attempt.state is ProbeState.IN_PROGRESS
        assert attempt.revision_changed is True
        assert attempt.decision is None
        assert forbidden == []
        assert operation.target_lease is None
    finally:
        executor.cancel_admission(operation.token)

    image.with_suffix(".txt").unlink()
    state = SourceFileState.capture(image)
    stamp = SourceExecutionStamp(
        state,
        "tiff_series",
        1,
        1,
        members=(state,),
        metadata_sources=(AdmittedMetadataSource(state.path, None),),
    )
    legacy = stamp.as_dict()
    assert legacy["metadata_sources"] == [
        {"source_path": str(image.absolute()), "metadata_file": None}
    ]
    identity = _identity_mapping(stamp)
    assert all(
        "image_metadata" not in value["roles"]
        for value in (*identity["aliases"], *identity["targets"])
    )


@pytest.mark.parametrize("cancel_at", (None, 1, 2, 3, 4))
def test_two_sweep_validation_order_and_cancellation_are_exact(
    cancel_at: int | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O11: one ordered two-sweep validator owns every Stop boundary."""

    first = tmp_path / "first_0001.tif"
    second = tmp_path / "second.dat"
    _write_tiff(first)
    second.write_bytes(b"dependency")
    stamp = SourceExecutionStamp(
        SourceFileState.capture(first),
        "image_file",
        1,
        1,
        dependency_files=(SourceFileState.capture(second),),
    )
    traces: list[str] = []
    states = {
        str(first.resolve()): SourceFileState.capture(first),
        str(second.resolve()): SourceFileState.capture(second),
    }
    owners = {str(first.absolute()): "image_file"}

    def resolve_alias(path: str) -> str:
        traces.append(f"resolve:{Path(path).name}")
        return str(Path(path).resolve(strict=True))

    def capture_target(path: str) -> SourceFileState:
        traces.append(f"capture:{Path(path).name}")
        return states[path]

    def owner_id(path: str) -> str | None:
        traces.append(f"owner:{Path(path).name}")
        return owners.get(path)

    cancel_calls = 0

    def cancelled() -> bool:
        nonlocal cancel_calls
        cancel_calls += 1
        traces.append("cancel")
        return cancel_at == cancel_calls

    monkeypatch.setattr(output_preflight, "_resolve_source_alias", resolve_alias, raising=False)
    monkeypatch.setattr(output_preflight, "_capture_canonical_source_target", capture_target, raising=False)
    monkeypatch.setattr(output_preflight, "_candidate_owner_id", owner_id, raising=False)

    if cancel_at is None:
        validated = output_preflight.validate_source_aliases(
            stamp,
            cancelled=cancelled,
        )
        assert validated.targets == stamp.canonical_targets
        assert traces == [
            "cancel", "resolve:first_0001.tif", "capture:first_0001.tif", "owner:first_0001.tif",
            "cancel", "resolve:second.dat", "capture:second.dat",
            "cancel", "resolve:first_0001.tif", "capture:first_0001.tif",
            "cancel", "resolve:second.dat", "capture:second.dat",
        ]
        traces.clear()
        item = _image_item(first, stamp, tmp_path / "processed.nxs")
        collision_calls = []

        def forbidden_collision(*_args, **_kwargs):
            collision_calls.append("collision")
            raise AssertionError("cancelled validation reached collision I/O")

        monkeypatch.setattr(
            output_preflight,
            "check_output_not_source",
            forbidden_collision,
        )
        with pytest.raises(RuntimeError, match="admission cancelled"):
            output_preflight._validate_targets(
                SimpleNamespace(poni_file="", mask_file=""),
                item.source_spec,
                (item,),
                cancelled=lambda: True,
            )
        assert traces == []
        assert collision_calls == []
    else:
        with pytest.raises(RuntimeError, match="admission cancelled"):
            output_preflight.validate_source_aliases(
                stamp,
                cancelled=cancelled,
            )
        expected_prefixes = {
            1: ["cancel"],
            2: [
                "cancel", "resolve:first_0001.tif", "capture:first_0001.tif", "owner:first_0001.tif", "cancel",
            ],
            3: [
                "cancel", "resolve:first_0001.tif", "capture:first_0001.tif", "owner:first_0001.tif",
                "cancel", "resolve:second.dat", "capture:second.dat", "cancel",
            ],
            4: [
                "cancel", "resolve:first_0001.tif", "capture:first_0001.tif", "owner:first_0001.tif",
                "cancel", "resolve:second.dat", "capture:second.dat",
                "cancel", "resolve:first_0001.tif", "capture:first_0001.tif", "cancel",
            ],
        }
        assert traces == expected_prefixes[cancel_at]


def test_preopen_validation_consumes_run_stop_before_reader_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O11 integration: the executor threads Stop into pre-open validation."""

    source = tmp_path / "selected_0001.tif"
    _write_tiff(source)
    state = SourceFileState.capture(source)
    stamp = SourceExecutionStamp(state, "image_file", 1, 1)
    item = _image_item(source, stamp, tmp_path / "processed.nxs")
    configuration = SimpleNamespace(
        output_mode="Overwrite",
        poni_file="accepted.poni",
        save_path=str(tmp_path / "processed.nxs"),
        thaw_source_spec=lambda: item.source_spec,
    )
    run = SimpleNamespace(
        configuration=configuration,
        capture=object(),
        artifact=tmp_path / "old.nxs",
        stop_requested=True,
        source=None,
        scan=None,
    )
    callbacks = []

    def cancelled_validation(_item, *, cancelled=None):
        callbacks.append(cancelled)
        if cancelled is None:
            raise AssertionError("pre-open validation lost the Stop callable")
        if cancelled():
            raise RuntimeError("admission cancelled")

    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("cancelled pre-open validation reached a reader")

    monkeypatch.setattr(
        executor_module,
        "validate_planned_source",
        cancelled_validation,
    )
    monkeypatch.setattr(executor_module, "open_source", forbidden_open)
    executor = StandardRunExecutor(join_timeout=1.0)
    with pytest.raises(RuntimeError, match="admission cancelled"):
        executor._construct(run, item=item, labels=(1,), decision=None)
    assert len(callbacks) == 1
    assert callable(callbacks[0])


@pytest.mark.parametrize("failure_kind", ("stable_schema", "stable_configuration"))
def test_stable_semantic_and_schema_failures_are_terminal_not_pending(
    failure_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O12: unchanged malformed input is terminal, never infinite pending."""

    if failure_kind == "stable_schema":
        source = tmp_path / "stable-malformed.h5"
        with h5py.File(source, "w") as handle:
            handle.create_group("entry/data")
        with pytest.raises(ValueError, match="unavailable") as error:
            output_preflight._trace_hdf5_object_dependencies(
                source,
                "/entry/data/missing",
                paths=[],
                seen=set(),
                cancelled=lambda: False,
                required=True,
                states={},
            )
        assert not isinstance(error.value, SourceRevisionChanged)
        return

    raw = tmp_path / "raw"
    raw.mkdir()
    image = raw / "scan_0001.tif"
    _write_tiff(image)
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=(".tif",), metadata_format=None),
        request_value=1112,
    )

    def stable_failure(*_args, **_kwargs):
        raise ValueError("stable output configuration invalid")

    monkeypatch.setattr(output_preflight, "_validate_deferred_targets", stable_failure)
    try:
        with pytest.raises(ValueError, match="stable output configuration invalid") as error:
            output_preflight.materialize_live_directory_group(
                receipt,
                intent.freeze(),
                session,
                group,
                cancelled=lambda: False,
            )
        assert not isinstance(error.value, SourceRevisionChanged)
    finally:
        executor.cancel_admission(operation.token)


def test_raw_frame_path_keeps_legacy_snapshot_and_provenance_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """O13: raw frame paths remain the legacy lookup and persisted shape."""

    target = tmp_path / "detector-target.nxs"
    alias = tmp_path / "selected.nxs"
    target.write_bytes(b"container")
    alias.symlink_to(target)
    state = SourceFileState.capture(alias)
    stamp = SourceExecutionStamp(state, "nexus", 3, 0)
    descriptor = ContainerDescriptor(
        alias,
        adapter_id="nexus",
        kind=SourceKind.NEXUS_STACK,
        dataset_path="/entry/data/data",
        frame_count=3,
        self_contained=True,
    )
    item = PlannedOutput(
        SourceSpec(alias, SourceKind.NEXUS_STACK),
        alias,
        tmp_path / "processed.nxs",
        stamp,
        descriptor=descriptor,
    )

    def forbidden_capture(_path: Path) -> SourceFileState:
        raise AssertionError("provenance serialization touched the filesystem")

    def forbidden_path_io(*_args, **_kwargs):
        raise AssertionError("provenance serialization touched the filesystem")

    with monkeypatch.context() as blocked:
        blocked.setattr(
            SourceFileState,
            "capture",
            staticmethod(forbidden_capture),
        )
        blocked.setattr(os.path, "realpath", forbidden_path_io)
        blocked.setattr(Path, "resolve", forbidden_path_io)
        blocked.setattr(Path, "stat", forbidden_path_io)
        legacy = stamp.as_dict()
        snapshots = output_preflight.source_snapshots(item)
    raw_key = str(alias.absolute())
    assert tuple(legacy) == _LEGACY_EXECUTION_KEYS
    assert legacy["path"] == raw_key
    assert set(snapshots) == {raw_key}
    assert snapshots[raw_key] == {
        "adapter_id": "nexus",
        **state.as_dict(),
        "frame_count": 3,
        "dataset_path": "/entry/data/data",
        "self_contained": True,
    }
    alias_identity = _identity_mapping(stamp)
    assert alias_identity["schema_version"] == 1
    assert "execution_identity_v1" not in legacy

    sink = NexusSink(
        tmp_path / "unused.nxs",
        source_execution_provenance=legacy,
        source_snapshots_provenance=snapshots,
    )
    captured: dict[str, object] = {}
    import xrd_tools.io.nexus_record as nexus_record

    monkeypatch.setattr(nexus_record, "ensure_frames_container", lambda _entry: object())

    def capture_record(_container, _name, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(nexus_record, "write_frame_record", capture_record)
    sink._h5 = {"entry": object()}
    frame = ScanFrame(0, source_path=alias, source_frame_index=0)
    sink._write_frame_record(
        frame,
        FrameReduction(0),
        prepared=(None, False),
    )
    assert captured["source_snapshot"] == snapshots[raw_key]


def _p36_tiff_graph(root: Path, *, conflict: bool = False):
    from xrd_tools.core.scan import SourceKind, SourceSpec
    from xrd_tools.sources.execution_graph import AdmittedMetadataSource, SourceFileState, freeze_source_execution_graph
    root.mkdir(parents=True, exist_ok=True)
    image = root / "scan_0001.tif"; sidecar = root / "scan_0001.txt"
    fabio.tifimage.TifImage(data=np.arange(4, dtype="u2").reshape(2, 2)).write(str(image))
    sidecar.write_text("# Counters\nI0 = 2\n# Motors\n\n"
                       "User: p36, time: Mon Jan 15 10:30:00 2024  # Temp\n")
    image_state = SourceFileState.capture(image); metadata_state = SourceFileState.capture(sidecar)
    if conflict: metadata_state = replace(metadata_state, path=image_state.path)
    source = SourceSpec(root, SourceKind.TIFF_SERIES, options={
        "selected_file": str(image), "files": (str(image),), "pattern": "scan_*.tif",
        "scan_name": "scan", "metadata_format": "txt"})
    return freeze_source_execution_graph(
        source, source, source_path=image, group_key="scan",
        file=image_state, adapter_id="tiff_series", frame_count=1, first_label=1,
        detector_shape=None, native_dtype=None,
        members=(image_state,),
        metadata_sources=(AdmittedMetadataSource(image_state.path, metadata_state),),
    )


def test_shared_source_graph_direct_and_ordinary_projections_are_byte_exact(
    tmp_path, monkeypatch,
):
    from xdart.gui.tabs.scattering import contracts, output_preflight
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.sources import execution_graph as graph_api
    graph = _p36_tiff_graph(tmp_path)
    item = PlannedOutput(graph.execution_source, Path(graph.source_path),
        tmp_path / "out.nxs", graph.stamp, descriptor=graph.descriptor, motor_names=graph.motor_names)
    assert contracts.SourceExecutionStamp is graph_api.SourceExecutionStamp
    assert contracts.SourceFileState is graph_api.SourceFileState
    assert graph_api.source_execution_projection(graph) == graph.stamp.as_dict()
    assert graph_api.source_execution_identity_v1_projection(graph) == graph.stamp.execution_identity_v1.as_dict()
    payload = graph_api.source_graph_payload(graph)
    assert payload["source_execution"] == graph.stamp.as_dict()
    assert payload["reader_binding"] is None
    assert graph_api.source_snapshots_projection(graph, writer=False) == output_preflight.source_snapshots(item)
    assert graph_api.source_snapshots_projection(graph, writer=True) == dynamic_output._writer_source_snapshots(item)
    assert graph_api.stable_lineage_projection(graph, target=item.target) == dynamic_output._stable_lineage(item)
    assert graph_api.append_source_from_execution_graph(graph, generation=3) == dynamic_output._append_source(graph.stamp, item, generation=3)
    qualified, frozen = [], []
    real_qualify = output_preflight.qualify_source_execution_graph
    real_freeze = output_preflight.freeze_source_execution_graph
    def qualify(source, **kwargs):
        qualified.append(source.kind); return real_qualify(source, **kwargs)
    def freeze(*args, **kwargs):
        frozen.append(kwargs["adapter_id"]); return real_freeze(*args, **kwargs)
    monkeypatch.setattr(output_preflight, "qualify_source_execution_graph", qualify)
    monkeypatch.setattr(output_preflight, "freeze_source_execution_graph", freeze)
    monkeypatch.setattr(output_preflight, "SourceExecutionStamp", lambda *_a, **_k: pytest.fail("GUI constructed source stamp"))
    monkeypatch.setattr(output_preflight, "PreparedSourceExecutionGraph", lambda *_a, **_k: pytest.fail("GUI constructed source graph"))
    config = output_preflight.OutputCandidate(graph.execution_source, "", "", str(tmp_path / "outputs"), "{}", "p36")
    direct = output_preflight._series_item(config, graph.execution_source)
    container = tmp_path / "ordinary.nxs"
    with h5py.File(container, "w") as handle:
        entry = handle.create_group("entry"); entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data"); data.attrs["NX_class"] = "NXdata"
        image = data.create_dataset("image", data=np.ones((1, 2, 2), dtype="u2"))
        image.attrs["signal_type"] = "detector"
    container_source = SourceSpec(container, SourceKind.NEXUS_STACK)
    container_item = output_preflight._container_item(replace(config, source=container_source), container_source)
    from xrd_tools.sources.adapters import candidate_owner
    from xrd_tools.sources.discover import Candidate
    from xrd_tools.sources.run_plan import RunCandidatePlan
    from xrd_tools.sources.selection import DirectorySourceSpec
    state = Path(graph.source_path).stat(); owner = candidate_owner(Path(graph.source_path))
    plan = RunCandidatePlan(0, (Candidate(Path(graph.source_path), owner.id,
        state.st_size, state.st_mtime_ns),), tmp_path, False, None)
    directory = output_preflight._directory_items(output_preflight.OutputCandidate(
        DirectorySourceSpec(tmp_path, suffixes=(".tif",), metadata_format="txt"),
        "", "", str(tmp_path / "directory"), "{}", "p36",
    ), plan)
    assert qualified == [SourceKind.TIFF_SERIES, SourceKind.NEXUS_STACK]
    assert len(directory) == 1 and frozen == ["tiff_series"]
    output_preflight.validate_planned_source(directory[0])
    assert frozen == ["tiff_series", "tiff_series"] and direct.source_stamp.adapter_id == "tiff_series" and container_item.source_stamp.adapter_id != "tiff_series"


def test_duplicate_snapshot_conflict_refuses_without_changing_valid_append_hashes(
    tmp_path,
):
    from xrd_tools.sources.execution_graph import (
        append_source_from_execution_graph,
        source_snapshots_projection,
    )
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xdart.gui.tabs.scattering.output_preflight import source_snapshots

    valid = _p36_tiff_graph(tmp_path / "valid")
    item = PlannedOutput(valid.execution_source, Path(valid.source_path),
                         tmp_path / "valid.nxs", valid.stamp)
    before = append_source_from_execution_graph(valid, generation=1)
    before_snapshots = source_snapshots(item)
    before_writer = dynamic_output._writer_source_snapshots(item)
    before_lineage = dynamic_output._stable_lineage(item)
    assert source_snapshots_projection(valid, writer=True) == before_writer
    with pytest.raises(ValueError, match="conflict|state"):
        _p36_tiff_graph(tmp_path / "conflict", conflict=True)
    after = append_source_from_execution_graph(valid, generation=1)
    assert before == after and before.digest == after.digest
    assert source_snapshots(item) == before_snapshots
    assert dynamic_output._writer_source_snapshots(item) == before_writer
    assert dynamic_output._stable_lineage(item) == before_lineage

    master_path = (tmp_path / "eiger_master.h5").absolute()
    member_path = (tmp_path / "eiger_data_000001.h5").absolute()
    master_state = SourceFileState(str(master_path), 1, 2, 3, 4, 5)
    member_state = SourceFileState(str(member_path), 6, 7, 8, 9, 10)
    external = ExternalSourceState(member_state, "/entry/data/data", 0, 2, 0)
    descriptor = ContainerDescriptor(
        master_path, kind=SourceKind.EIGER_MASTER,
        dataset_path="/entry/data/data_000001",
        segment_paths=("/entry/data/data_000001",), frame_count=2,
        frame_shape=(2, 2), dtype=np.dtype("<u2"), self_contained=False,
    )
    from xrd_tools.sources.execution_graph import freeze_source_execution_graph
    source = SourceSpec(master_path, SourceKind.EIGER_MASTER)
    eiger = freeze_source_execution_graph(
        source, source, source_path=master_path, group_key="eiger",
        file=master_state, adapter_id="nexus_hdf5", frame_count=2,
        first_label=0, detector_shape=(2, 2), native_dtype="<u2",
        external_members=(external,), descriptor=descriptor,
    )
    assert append_source_from_execution_graph(
        eiger, generation=1,
    ).dataset_paths == ("/entry/data/data_000001",)


def test_average_lineage_and_provenance_match_exact_ordinary_projections(
    tmp_path, monkeypatch,
):
    from xrd_tools.core.containers import IntegrationResult1D
    from xrd_tools.core.provenance import read_provenance
    from xrd_tools.io import AppendIntent, science_fingerprint
    from xrd_tools.reduction import AverageScanRecipe, Integration1DPlan, ReductionPlan
    from xrd_tools.reduction import average as average_module
    from xrd_tools.reduction import core as reduction_core
    from xrd_tools.sources.execution_graph import (
        append_source_from_execution_graph,
        source_execution_projection,
        source_snapshots_projection,
        stable_lineage_projection,
    )
    from xdart.gui.tabs.scattering import output_preflight
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    graph = _p36_tiff_graph(tmp_path / "gráph")
    target = tmp_path / "average.nxs"
    item = PlannedOutput(graph.execution_source, Path(graph.source_path),
                         target, graph.stamp)
    append = append_source_from_execution_graph(graph, generation=1)
    assert append.extent == 1 and append.generation == 1
    assert append.image_members[0].source_start == 0
    assert append.image_members[0].source_stop == 1
    assert append.image_members[0].ordinal == 0
    assert source_execution_projection(graph) == graph.stamp.as_dict()
    snapshots = source_snapshots_projection(graph, writer=True)
    assert set(snapshots) == {
        graph.stamp.members[0].path,
        graph.stamp.metadata_sources[0].metadata_file.path,
    }
    lineage = stable_lineage_projection(graph, target=target)
    assert lineage[0:2] == ("tiff_series", "scan")
    assert source_execution_projection(graph) == item.source_stamp.as_dict()
    assert source_snapshots_projection(graph, writer=False) == output_preflight.source_snapshots(item)
    assert snapshots == dynamic_output._writer_source_snapshots(item)
    assert lineage == dynamic_output._stable_lineage(item)
    assert append == dynamic_output._append_source(item.source_stamp, item, generation=1)
    canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str).encode("utf-8")
    assert canonical(source_execution_projection(graph)) == canonical(item.source_stamp.as_dict())
    assert "gráph".encode() in canonical(source_execution_projection(graph))
    prepared = __import__("xrd_tools.sources.execution_graph", fromlist=["x"]).qualify_source_execution_graph(
        graph.execution_source, reader_binding="average_closed_v1",
    )
    captured = {}; real_sink = average_module.NexusSink
    def sink(*args, **kwargs):
        captured.update(kwargs); return real_sink(*args, **kwargs)
    monkeypatch.setattr(average_module, "NexusSink", sink)
    monkeypatch.setattr(reduction_core, "integrate_1d", lambda _image, _ai, *, npt, **_k:
        IntegrationResult1D(np.arange(npt, dtype=float), np.ones(npt), None, "q_A^-1"))
    runner = average_module.AverageScanRunner(AverageScanRecipe(
        graph.execution_source, target,
        ReductionPlan(integration_1d=Integration1DPlan(npt=3)),
        numeric_metadata_keys=("I0",),
    ))
    result = runner.start()
    assert type(result) is average_module.AverageScanResult
    assert runner.close() is result
    assert result.disposition == "COMMITTED"
    expected_execution = source_execution_projection(prepared)
    expected_snapshots = source_snapshots_projection(prepared, writer=True)
    expected_append = append_source_from_execution_graph(prepared, generation=1)
    ordinary_intent = AppendIntent("entry", "",
        science_fingerprint(stable_lineage_projection(prepared, target=target)),
        result.science_identity, ("1d:default",), expected_append, (1,))
    assert captured["source_execution_provenance"] == expected_execution
    assert captured["source_snapshots_provenance"] == expected_snapshots
    assert captured["same_run_intent"] == ordinary_intent
    assert read_provenance(target)["config"]["source_execution"] == expected_execution
