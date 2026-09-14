"""Focused source-side lifecycle for an initially empty Live directory."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import time

import fabio
import h5py
import numpy as np
import pytest

from tests.xdart.scattering._e2sd_support import write_poni
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    SourceFileState,
    SourceCapture,
    StartCapture,
)
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    RequestId,
    RunIdentity,
)
from xdart.gui.tabs.scattering import output_preflight
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources import execution_graph as source_graph
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.selection import DirectorySourceSpec


def _write_small_poni(path: Path) -> None:
    """Real calibration matching the four-by-four detector fixtures."""
    write_poni(path)
    path.write_text(path.read_text().replace(
        "Detector: Pilatus100k\nDetector_config: {}",
        'Detector: Detector\nDetector_config: {"pixel1": 0.0001, '
        '"pixel2": 0.0001, "max_shape": [4, 4], "orientation": 3}',
    ))


def _add_display_artifact(run, item) -> None:
    if not run.display.configured:
        run.display.configure(
            partition_count=1,
            npt=0,
            frame_bytes=None,
        )
    run.display.add_artifact(
        item.target,
        item.source_path.stem,
        mask=None,
        mask_saturation=True,
        measurement_mode="Standard",
    )


def test_live_tiff_metadata_changes_after_capture_before_ready_is_typed_pending(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    first = raw / "scan_0001.tif"
    second = raw / "scan_0002.tif"
    for index, image in enumerate((first, second), start=1):
        fabio.tifimage.TifImage(
            data=np.full((4, 4), index, dtype=np.uint16)
        ).write(str(image))
        image.with_suffix(".txt").write_text(
            f"th={index / 10}\nsequence={index}\nexposure={index}\n",
            encoding="utf-8",
        )
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw,
            suffixes=(".tif",),
            metadata_format="auto",
        ),
        request_value=1002,
        gi=GIIntent(enabled=True, incidence_motor="th"),
    )
    from xrd_tools.io import metadata as metadata_io
    real_read = metadata_io.read_image_metadata_observed
    changed = False

    def change_earlier_sidecar_while_later_is_inspected(path, *args, **kwargs):
        nonlocal changed
        if Path(path).resolve() == second.resolve() and not changed:
            first.with_suffix(".txt").write_text(
                "th=9.25\nsequence=1\nexposure=1\nrevision=changed\n",
                encoding="utf-8",
            )
            changed = True
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(
        metadata_io,
        "read_image_metadata_observed",
        change_earlier_sidecar_while_later_is_inspected,
    )
    identity = RunIdentity.from_configuration(intent.freeze())
    try:
        attempt = output_preflight.materialize_live_directory_group(
            receipt,
            intent.freeze(),
            session,
            group,
            cancelled=lambda: False,
        )

        assert changed is True
        assert attempt.state is ProbeState.IN_PROGRESS
        assert attempt.revision_changed is True
        assert attempt.decision is None
        assert executor.processed_live_revisions(identity) == ()
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_live_stable_tiff_metadata_remains_exact_through_pending_projection(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    first = raw / "scan_0001.tif"
    second = raw / "scan_0002.tif"
    for index, image in enumerate((first, second), start=1):
        fabio.tifimage.TifImage(
            data=np.full((4, 4), index, dtype=np.uint16)
        ).write(str(image))
        image.with_suffix(".txt").write_text(
            f"th={index / 10}\nsequence={index}\nexposure={index}\n",
            encoding="utf-8",
        )
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw,
            suffixes=(".tif",),
            metadata_format="auto",
        ),
        request_value=1003,
        gi=GIIntent(enabled=True, incidence_motor="th"),
    )
    identity = RunIdentity.from_configuration(intent.freeze())
    try:
        attempt = output_preflight.materialize_live_directory_group(
            receipt,
            intent.freeze(),
            session,
            group,
            cancelled=lambda: False,
        )

        assert attempt.state is ProbeState.READY
        assert attempt.revision_changed is False
        assert attempt.decision is not None
        item = attempt.decision.item
        stamp = item.source_stamp
        assert tuple(
            (value.source_path, value.motor, value.value)
            for value in stamp.admitted_motor_values
        ) == (
            (str(first.resolve()), "th", pytest.approx(0.1)),
            (str(second.resolve()), "th", pytest.approx(0.2)),
        )
        metadata_states = tuple(
            value.metadata_file for value in stamp.metadata_sources
        )
        assert all(value is not None for value in metadata_states)
        assert all(value.matches_disk() for value in metadata_states if value)
        assert tuple(
            (Path(value.raw_path), value.roles)
            for value in stamp.source_aliases
            if "image_metadata" in value.roles
        ) == tuple(
            (image.with_suffix(".txt").resolve(), ("image_metadata",))
            for image in (first, second)
        )
        snapshots = output_preflight.source_snapshots(item)
        for image in (first, second):
            sidecar = str(image.with_suffix(".txt").resolve())
            assert snapshots[sidecar]["source_role"] == "image_metadata"
            assert snapshots[sidecar]["path"] == sidecar
        assert attempt.group is group
        assert attempt.decision.item.source_stamp is stamp
        assert executor.processed_live_revisions(identity) == ()
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_live_display_admits_one_additional_partition_monotonically(
    tmp_path: Path,
) -> None:
    identity = RunIdentity(1, "live-display")
    display = RunDisplayState(identity, max_payload_items=2)
    display.configure(partition_count=1, npt=8, frame_bytes=32)
    common = {
        "mask": None,
        "mask_saturation": True,
        "measurement_mode": "Standard",
    }
    first = display.add_artifact(
        tmp_path / "first.nexus",
        "first",
        **common,
    )
    residency = display._residency

    display.admit_additional_partition()
    second = display.add_artifact(
        tmp_path / "second.nexus",
        "second",
        **common,
    )

    assert first is display.artifacts[str(tmp_path / "first.nexus")]
    assert second is display.artifacts[str(tmp_path / "second.nexus")]
    assert display._residency is residency


def test_live_admission_owns_a_nonexpiring_provisional_session(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(
        raw,
        suffixes=(".nxs",),
        metadata_format=None,
    )
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        live_mode=True,
        max_cores=1,
    )
    snapshot = RunIntentStore(intent).snapshot()
    request = RequestId(990)
    capture = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source),
    )
    executor = StandardRunExecutor(join_timeout=2.0)

    _wait_for_admission(executor, capture)
    operation = executor._admission
    assert operation is not None
    try:
        assert operation.directory_session is not None
        assert math.isinf(operation.directory_session._retry_deadline)
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


@pytest.mark.parametrize("image_kind", ("tiff", "raw"))
def test_live_probe_to_strong_freeze_never_mixes_candidate_revisions(
    image_kind: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    if image_kind == "tiff":
        image = raw / "race_0001.tif"
        suffix = ".tif"

        def rewrite(value: int) -> None:
            fabio.tifimage.TifImage(
                data=np.full((4, 4), value, dtype=np.uint16)
            ).write(str(image))
    else:
        image = raw / "race_0001.raw"
        suffix = ".raw"

        def rewrite(value: int) -> None:
            np.full((195, 487), value, dtype=np.int32).tofile(image)

    rewrite(1)
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(
            raw,
            suffixes=(suffix,),
            metadata_format=None,
        ),
        request_value=996 if image_kind == "tiff" else 997,
    )
    original = SourceFileState.capture
    changed = False

    def capture_after_probe(path: Path) -> SourceFileState:
        nonlocal changed
        if Path(path).resolve() == image.resolve() and not changed:
            rewrite(2)
            changed = True
        return original(Path(path))

    monkeypatch.setattr(
        SourceFileState,
        "capture",
        staticmethod(capture_after_probe),
    )
    try:
        attempt = output_preflight.materialize_live_directory_group(
            receipt,
            intent.freeze(),
            session,
            group,
            cancelled=lambda: False,
        )

        assert changed is True
        assert attempt.state is ProbeState.IN_PROGRESS
        assert attempt.revision_changed is True
        assert attempt.decision is None
        assert executor.processed_live_revisions(
            RunIdentity.from_configuration(intent.freeze())
        ) == ()
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_live_hdf_initial_capture_disappearance_is_typed_pending(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from xrd_tools.io.output_transaction import get_output_transaction_coordinator

    raw = tmp_path / "raw"
    raw.mkdir()
    master = raw / "capture-race.h5"
    _write_self_contained_hdf(master)
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=(".h5",), metadata_format=None),
        request_value=998,
    )
    capture = output_preflight._capture_exact_candidate_state
    coordinator = get_output_transaction_coordinator()
    prior_leases = dict(coordinator._leases)
    changed = False

    def disappear_before_capture(candidate):
        nonlocal changed
        if candidate.path == master and not changed:
            master.unlink()
            changed = True
        return capture(candidate)

    monkeypatch.setattr(
        output_preflight,
        "_capture_exact_candidate_state",
        disappear_before_capture,
    )
    try:
        attempt = output_preflight.materialize_live_directory_group(
            receipt,
            intent.freeze(),
            session,
            group,
            cancelled=lambda: False,
        )
        assert changed is True
        assert attempt.state is ProbeState.IN_PROGRESS
        assert attempt.revision_changed is True
        assert attempt.decision is None
        assert coordinator._leases == prior_leases
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


@pytest.mark.parametrize("open_target", ("master", "member"))
def test_live_hdf_change_during_stable_open_is_typed_pending(
    open_target: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    master = raw / "scan_master.h5"
    member = raw / "scan_data_000001.h5"
    if open_target == "member":
        _write_external_hdf(master, member)
        selected = member
    else:
        _write_self_contained_hdf(master)
        selected = master
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=("_master.h5",), metadata_format=None),
        request_value=999 if open_target == "master" else 1000,
    )
    original_capture = SourceFileState.capture
    real_h5_file_init = h5py.File.__init__
    armed = False
    changed = False

    def capture_before_open(path: Path) -> SourceFileState:
        nonlocal armed
        state = original_capture(Path(path))
        if Path(path).resolve() == selected.resolve():
            armed = True
        return state

    def change_then_fail(handle, path, *args, **kwargs):
        nonlocal changed
        if (
            armed
            and isinstance(path, (str, Path))
            and not changed
            and Path(path).resolve() == selected.resolve()
            and (not args or args[0] == "r")
            and kwargs.get("mode", "r") == "r"
        ):
            selected.unlink()
            changed = True
            raise OSError("source disappeared during HDF5 open")
        return real_h5_file_init(handle, path, *args, **kwargs)

    monkeypatch.setattr(
        SourceFileState,
        "capture",
        staticmethod(capture_before_open),
    )
    monkeypatch.setattr(h5py.File, "__init__", change_then_fail)
    try:
        sibling_candidates = tuple(
            candidate
            for candidate in group.plan.candidates
            if candidate.path != master
        )
        before = dict(session._results)
        attempt = output_preflight.materialize_live_directory_group(
            receipt,
            intent.freeze(),
            session,
            group,
            cancelled=lambda: False,
        )

        assert changed is True
        assert attempt.state is ProbeState.IN_PROGRESS
        assert attempt.revision_changed is True
        assert attempt.decision is None
        for sibling in sibling_candidates:
            assert session._results.get(sibling.path) == before.get(sibling.path)
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_live_stable_hdf_open_error_remains_hard(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    master = raw / "stable_master.h5"
    _write_self_contained_hdf(master)
    executor, intent, receipt, operation, session, group = _admitted_live_group(
        tmp_path,
        DirectorySourceSpec(raw, suffixes=("_master.h5",), metadata_format=None),
        request_value=1001,
    )
    from contextlib import contextmanager

    real_open = source_graph._open_stable_hdf5_dependency
    real_h5_file_init = h5py.File.__init__
    armed = False

    @contextmanager
    def armed_stable_open(path, *args, **kwargs):
        nonlocal armed
        if Path(path).resolve() == master.resolve():
            armed = True
        with real_open(path, *args, **kwargs) as handle:
            yield handle

    def fail_stably(handle, path, *args, **kwargs):
        if (
            armed
            and isinstance(path, (str, Path))
            and Path(path).resolve() == master.resolve()
            and (not args or args[0] == "r")
            and kwargs.get("mode", "r") == "r"
        ):
            raise OSError("stable HDF5 open refusal")
        return real_h5_file_init(handle, path, *args, **kwargs)

    monkeypatch.setattr(
        source_graph,
        "_open_stable_hdf5_dependency",
        armed_stable_open,
    )
    monkeypatch.setattr(h5py.File, "__init__", fail_stably)
    try:
        with pytest.raises(ValueError, match="stable HDF5 open refusal") as error:
            output_preflight.materialize_live_directory_group(
                receipt,
                intent.freeze(),
                session,
                group,
                cancelled=lambda: False,
            )
        assert not isinstance(
            error.value,
            output_preflight.SourceRevisionChanged,
        )
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_live_construct_source_revision_drift_reprobes_and_retries(
    monkeypatch,
    request,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    late = raw / "racy_0001.tif"
    fabio.tifimage.TifImage(
        data=np.ones((4, 4), dtype=np.uint16)
    ).write(str(late))
    poni = tmp_path / "cal.poni"
    _write_small_poni(poni)
    source = DirectorySourceSpec(
        raw,
        suffixes=(".tif",),
        metadata_format=None,
    )
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        live_mode=True,
        max_cores=1,
        bai_1d_args={"npt": 8},
        bai_2d_args={"npt_rad": 8, "npt_azim": 4},
    )
    snapshot = RunIntentStore(intent).snapshot()
    request_id = RequestId(992)
    source_capture = SourceCapture(request_id, 1, source)
    capture = StartCapture(request_id, 1, snapshot, source_capture)
    executor = StandardRunExecutor(join_timeout=2.0)
    construct_calls = 0
    processed = []

    real_construct = executor._construct

    def construct(run, *, item, labels, decision):
        nonlocal construct_calls
        construct_calls += 1
        if construct_calls == 1:
            raise output_preflight.SourceRevisionChanged(
                "source changed immediately before open"
            )
        result = real_construct(run, item=item, labels=labels, decision=decision)
        processed.append(item.source_path)
        return result

    monkeypatch.setattr(executor, "_construct", construct)
    receipt = _wait_for_admission(executor, capture)
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)
    assert type(executor.start(
        configuration,
        source_capture,
        identity,
        receipt,
    )) is ExecutorAccepted
    request.addfinalizer(lambda: executor.stop(identity))

    events = []
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not processed:
        events.extend(executor.drain_events())
        time.sleep(0.01)
    assert construct_calls == 2
    assert processed == [late]
    assert not any(
        event.kind is StandardEventKind.FAILED for event in events
    )
    run = executor._exact_run(identity)
    assert run is not None
    assert run.resources is not None
    executor.stop(identity)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        events.extend(executor.drain_events())
        if any(
            event.kind is StandardEventKind.STOPPED for event in events
        ):
            break
        time.sleep(0.01)
    assert any(
        event.kind is StandardEventKind.STOPPED for event in events
    )


def test_live_stable_construct_value_error_is_terminal_not_retried(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    image = raw / "stable_bad_0001.tif"
    fabio.tifimage.TifImage(
        data=np.ones((4, 4), dtype=np.uint16)
    ).write(str(image))
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(
        raw,
        suffixes=(".tif",),
        metadata_format=None,
    )
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        live_mode=True,
        max_cores=1,
    )
    snapshot = RunIntentStore(intent).snapshot()
    request_id = RequestId(995)
    source_capture = SourceCapture(request_id, 1, source)
    capture = StartCapture(request_id, 1, snapshot, source_capture)
    executor = StandardRunExecutor(join_timeout=2.0)
    construct_calls = 0

    def construct(_run, *, item, labels, decision):
        nonlocal construct_calls
        construct_calls += 1
        raise ValueError("stable construction invalid")

    monkeypatch.setattr(executor, "_construct", construct)
    receipt = _wait_for_admission(executor, capture)
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)
    assert type(executor.start(
        configuration,
        source_capture,
        identity,
        receipt,
    )) is ExecutorAccepted

    events = []
    deadline = time.monotonic() + 10.0
    terminal = None
    while time.monotonic() < deadline:
        events.extend(executor.drain_events())
        terminal = next(
            (
                event
                for event in reversed(events)
                if event.kind is StandardEventKind.FAILED
            ),
            None,
        )
        if terminal is not None:
            break
        time.sleep(0.01)
    assert construct_calls == 1
    assert terminal is not None
    assert terminal.primary is not None
    assert terminal.primary.message == "stable construction invalid"
    assert terminal.cleanup_status is CleanupStatus.CLEANED


def test_live_external_member_growth_during_jit_is_typed_pending(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    master = raw / "scan_master.h5"
    member = tmp_path / "scan_data_000001.h5"
    with h5py.File(member, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.ones((2, 4, 4), dtype=np.uint16),
            maxshape=(None, 4, 4),
        )
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data["data_000001"] = h5py.ExternalLink(
            str(member),
            "/entry/data/data",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(
        raw,
        suffixes=("_master.h5",),
        metadata_format=None,
    )
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        live_mode=True,
        max_cores=1,
    )
    snapshot = RunIntentStore(intent).snapshot()
    request_id = RequestId(993)
    capture = StartCapture(
        request_id,
        1,
        snapshot,
        SourceCapture(request_id, 1, source),
    )
    executor = StandardRunExecutor(join_timeout=2.0)
    receipt = _wait_for_admission(executor, capture)
    operation = executor._admission
    assert operation is not None
    session = operation.directory_session
    assert session is not None
    changed = False

    def grow_after_ready(path):
        nonlocal changed
        adapter = original_get_adapter(group.plan.candidates[0].adapter_id)
        assert adapter is not None
        observed = adapter.probe(path)
        if not changed and observed.state is ProbeState.READY:
            with h5py.File(member, "a") as handle:
                dataset = handle["entry/data/data"]
                dataset.resize((3, 4, 4))
                dataset[2] = 2
            changed = True
        return observed
    try:
        observation = session.observe(refresh=True)
        groups = output_preflight.live_directory_groups(
            receipt,
            observation,
        )
        assert len(groups) == 1
        group = groups[0]
        original_get_adapter = output_preflight.get_adapter

        def changing_adapter(adapter_id):
            adapter = original_get_adapter(adapter_id)
            if adapter_id != group.plan.candidates[0].adapter_id:
                return adapter
            assert adapter is not None
            return replace(adapter, probe=grow_after_ready)

        monkeypatch.setattr(
            output_preflight,
            "get_adapter",
            changing_adapter,
        )
        attempt = output_preflight.materialize_live_directory_group(
            receipt,
            intent.freeze(),
            session,
            group,
            cancelled=lambda: False,
        )
        assert changed is True
        assert attempt.state is ProbeState.IN_PROGRESS
        assert attempt.decision is None
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_live_stable_configuration_value_error_remains_terminal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    image = raw / "scan_0001.tif"
    fabio.tifimage.TifImage(
        data=np.ones((4, 4), dtype=np.uint16)
    ).write(str(image))
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(
        raw,
        suffixes=(".tif",),
        metadata_format=None,
    )
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        live_mode=True,
    )
    snapshot = RunIntentStore(intent).snapshot()
    request_id = RequestId(994)
    capture = StartCapture(
        request_id,
        1,
        snapshot,
        SourceCapture(request_id, 1, source),
    )
    executor = StandardRunExecutor(join_timeout=2.0)
    receipt = _wait_for_admission(executor, capture)
    operation = executor._admission
    assert operation is not None
    session = operation.directory_session
    assert session is not None

    def stable_failure(*_args, **_kwargs):
        raise ValueError("stable output configuration invalid")

    monkeypatch.setattr(
        output_preflight,
        "_validate_deferred_targets",
        stable_failure,
    )
    try:
        observation = session.observe(refresh=True)
        group = output_preflight.live_directory_groups(
            receipt,
            observation,
        )[0]
        with pytest.raises(
            ValueError,
            match="stable output configuration invalid",
        ) as error:
            output_preflight.materialize_live_directory_group(
                receipt,
                intent.freeze(),
                session,
                group,
                cancelled=lambda: False,
            )
        source_drift = getattr(
            output_preflight,
            "SourceRevisionChanged",
            (),
        )
        assert not isinstance(error.value, source_drift)
    finally:
        released = executor.cancel_admission(operation.token)
    assert released.cleanup_status is CleanupStatus.CLEANED


def test_stable_malformed_hdf5_path_is_not_source_revision_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "stable-malformed.h5"
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_group("data")
    states = {}

    with pytest.raises(
        ValueError,
        match="selected HDF5 dependency path is unavailable",
    ) as error:
        source_graph._trace_hdf5_object_dependencies(
            source,
            "/entry/data/missing",
            paths=[],
            seen=set(),
            cancelled=lambda: False,
            required=True,
            states=states,
        )

    assert not isinstance(
        error.value,
        output_preflight.SourceRevisionChanged,
    )
    assert states
    assert all(state.matches_disk() for state in states.values())


def _write_self_contained_hdf(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.create_dataset(
            "data",
            data=np.ones((2, 4, 4), dtype=np.uint16),
        )


def _write_external_hdf(master: Path, member: Path) -> None:
    with h5py.File(member, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.ones((2, 4, 4), dtype=np.uint16),
        )
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data["data_000001"] = h5py.ExternalLink(
            member.name,
            "/entry/data/data",
        )


def _admitted_live_group(
    tmp_path: Path,
    source: DirectorySourceSpec,
    *,
    request_value: int,
    gi: GIIntent | None = None,
):
    poni = tmp_path / f"cal-{request_value}.poni"
    write_poni(poni)
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / f"processed-{request_value}"),
        output_mode="Overwrite",
        live_mode=True,
        max_cores=1,
        gi=GIIntent() if gi is None else gi,
    )
    snapshot = RunIntentStore(intent).snapshot()
    request_id = RequestId(request_value)
    capture = StartCapture(
        request_id,
        1,
        snapshot,
        SourceCapture(request_id, 1, source),
    )
    executor = StandardRunExecutor(join_timeout=2.0)
    receipt = _wait_for_admission(executor, capture)
    operation = executor._admission
    assert operation is not None
    session = operation.directory_session
    assert session is not None
    observation = session.observe(refresh=True)
    groups = output_preflight.live_directory_groups(receipt, observation)
    assert len(groups) == 1
    return executor, intent, receipt, operation, session, groups[0]


def _wait_for_admission(
    executor: StandardRunExecutor,
    capture: StartCapture,
    *,
    timeout: float = 10.0,
) -> AdmissionReceipt:
    token = executor.begin_admission(capture)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = executor.poll_admission(token)
        if type(result) is AdmissionReceipt:
            return result
        if type(result) is AdmissionFailure:
            raise AssertionError(result.reason)
        time.sleep(0.01)
    raise AssertionError("empty Live directory admission did not settle")


@pytest.mark.parametrize("arrival", (
    "distinct-group", "changed-prior-frame", "same-group-suffix",
))
def test_empty_directory_live_waits_for_one_stable_late_group_until_stop(
    monkeypatch,
    request,
    arrival,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    poni = tmp_path / "cal.poni"
    _write_small_poni(poni)
    source = DirectorySourceSpec(
        raw,
        suffixes=(".tif",),
        metadata_format=None,
    )
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        live_mode=True,
        max_cores=1,
        bai_1d_args={"npt": 8},
        bai_2d_args={"npt_rad": 8, "npt_azim": 4},
    )
    snapshot = RunIntentStore(intent).snapshot()
    request_id = RequestId(991)
    source_capture = SourceCapture(request_id, 1, source)
    capture = StartCapture(request_id, 1, snapshot, source_capture)
    executor = StandardRunExecutor(join_timeout=2.0)
    processed: list[tuple[Path, tuple[int, ...]]] = []
    ready_attempts = []
    materialize = executor_module.materialize_live_directory_group

    def capture_attempt(*args, **kwargs):
        attempt = materialize(*args, **kwargs)
        if attempt.decision is not None:
            ready_attempts.append(attempt)
        return attempt

    real_construct = executor._construct

    def construct(run, *, item, labels, decision):
        result = real_construct(run, item=item, labels=labels, decision=decision)
        processed.append((item.source_path, labels))
        return result

    monkeypatch.setattr(
        executor_module,
        "materialize_live_directory_group",
        capture_attempt,
    )
    monkeypatch.setattr(executor, "_construct", construct)

    receipt = _wait_for_admission(executor, capture)
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)
    assert type(executor.start(
        configuration,
        source_capture,
        identity,
        receipt,
    )) is ExecutorAccepted
    request.addfinalizer(lambda: executor.stop(identity))

    events = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        events.extend(executor.drain_events())
        if any(
            event.kind is StandardEventKind.DISCOVERY
            and event.files_discovered == 0
            for event in events
        ):
            break
        time.sleep(0.01)
    assert processed == []
    assert not any(
        event.kind in {
            StandardEventKind.FINISHED,
            StandardEventKind.FAILED,
            StandardEventKind.STOPPED,
        }
        for event in events
    )

    late = raw / "late_0001.tif"
    late.write_bytes(b"partially written TIFF")
    time.sleep(0.2)
    events.extend(executor.drain_events())
    assert processed == []
    assert not any(
        event.kind in {
            StandardEventKind.FINISHED,
            StandardEventKind.FAILED,
            StandardEventKind.STOPPED,
        }
        for event in events
    )

    fabio.tifimage.TifImage(
        data=np.ones((4, 4), dtype=np.uint16)
    ).write(str(late))
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and len(processed) < 1:
        events.extend(executor.drain_events())
        time.sleep(0.01)
    assert processed == [(late, (1,))]

    deadline = time.monotonic() + 10.0
    while not executor.processed_live_revisions(identity) and time.monotonic() < deadline:
        events.extend(executor.drain_events())
        time.sleep(0.01)
    original, = executor.processed_live_revisions(identity)
    assert original is ready_attempts[0]
    assert original.decision is not None

    if arrival == "changed-prior-frame":
        from xrd_tools.io.frame_view import FrameViewReader

        target = original.decision.item.target
        with FrameViewReader(target, resolve_source=False) as reader:
            prior_view = reader.read(1)
        time.sleep(0.01)
        fabio.tifimage.TifImage(
            data=np.full((4, 4), 2, dtype=np.uint16)
        ).write(str(late))
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            events.extend(executor.drain_events())
            failed = [event for event in events
                      if event.kind is StandardEventKind.FAILED]
            if failed:
                break
            time.sleep(0.01)
        assert len(failed) == 1
        assert failed[0].primary.type_qualname == "AppendRefused"
        assert failed[0].primary.message == (
            "source does not extend exact image-member history"
        )
        assert processed == [(late, (1,))]
        assert executor.processed_live_revisions(identity) == (original,)
        assert ready_attempts[-1].decision.item.source_stamp != (
            original.decision.item.source_stamp
        )
        with FrameViewReader(target, resolve_source=False) as reader:
            preserved = reader.read(1)
        np.testing.assert_array_equal(preserved.intensity_1d, prior_view.intensity_1d)
        np.testing.assert_array_equal(preserved.intensity_2d, prior_view.intensity_2d)
        return

    # A suffix reuses the same output owner; a distinct group retires it before
    # constructing its own owner. Both arrivals remain processable in Live.
    same_group = arrival == "same-group-suffix"
    second = raw / ("late_0002.tif" if same_group else "second_0001.tif")
    fabio.tifimage.TifImage(
        data=np.full((4, 4), 3, dtype=np.uint16)
    ).write(str(second))
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and len(processed) < 2:
        events.extend(executor.drain_events())
        time.sleep(0.01)
    assert processed == [(late, (1,)),
                         (late, (1, 2)) if same_group else (second, (1,))]
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        revisions = executor.processed_live_revisions(identity)
        if sum(attempt.decision.item.source_stamp.frame_count for attempt in revisions) == 2:
            break
        events.extend(executor.drain_events())
        time.sleep(0.01)
    assert sum(attempt.decision.item.source_stamp.frame_count for attempt in revisions) == 2

    # The Live owner remains armed after the first stable arrival.  It reaches
    # a terminal state only when the operator asks it to stop.
    assert not any(
        event.kind in {
            StandardEventKind.FINISHED,
            StandardEventKind.FAILED,
            StandardEventKind.STOPPED,
        }
        for event in events
    )

    executor.stop(identity)
    deadline = time.monotonic() + 5.0
    terminal = None
    while time.monotonic() < deadline:
        events.extend(executor.drain_events())
        terminal = next(
            (
                event
                for event in reversed(events)
                if event.kind in {
                    StandardEventKind.FINISHED,
                    StandardEventKind.FAILED,
                    StandardEventKind.STOPPED,
                }
            ),
            None,
        )
        if terminal is not None:
            break
        time.sleep(0.01)
    assert terminal is not None
    assert terminal.kind is StandardEventKind.STOPPED
    assert terminal.cleanup_status is CleanupStatus.CLEANED
    run = executor._exact_run(identity)
    assert run is not None
    assert run.resources is None
    assert run.worker is not None and not run.worker.is_alive()
    if same_group:
        from xrd_tools.io.frame_view import FrameViewReader
        assert len(executor.frame_catalog(identity).entries) == 2
        with FrameViewReader(original.decision.item.target, resolve_source=False) as reader:
            assert reader.labels() == (1, 2)
