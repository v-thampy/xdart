"""Finite P1-B comprehensive Live and Stop/durability acceptance oracle."""

from __future__ import annotations

from pathlib import Path
import re
from threading import Event, Lock, Thread
import time

import fabio
import h5py
import numpy as np
import pytest

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_output_graph import (
    _TERMINAL,
    _bridge_legacy_expected_target_state,
    _drain_until,
    _start,
    _written,
)
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.selection import DirectorySourceSpec
import xrd_tools.sources.registry  # noqa: F401  # register built-in readers


def _write_tiff(path: Path, value: int) -> None:
    fabio.tifimage.TifImage(
        data=np.full((195, 487), value, dtype=np.uint16)
    ).write(str(path))


def _write_flat(path: Path, kind: str, value: int) -> None:
    image = np.full((195, 487), value, dtype=np.uint16)
    owner = {
        "tif": fabio.tifimage.TifImage,
        "cbf": fabio.cbfimage.CbfImage,
        "edf": fabio.edfimage.EdfImage,
    }[kind]
    owner(data=image).write(str(path))


def _write_raw(path: Path, value: int) -> None:
    np.full((195, 487), value, dtype=np.int32).tofile(path)


def _write_stack(path: Path, frames: int) -> None:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/detector")
        detector.create_dataset(
            "data",
            data=np.full((frames, 195, 487), 1, dtype=np.uint16),
            chunks=(1, 195, 487),
            maxshape=(None, 195, 487),
        )


def _grow_stack(path: Path, frames: int, value: int = 2) -> None:
    with h5py.File(path, "r+") as handle:
        data = handle["entry/instrument/detector/data"]
        prior = int(data.shape[0])
        data.resize((frames, *data.shape[1:]))
        if frames > prior:
            data[prior:frames] = value
        handle.flush()


def _write_eiger(master: Path, member: Path, frames: int) -> None:
    with h5py.File(member, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.ones((frames, 195, 487), dtype=np.uint16),
            chunks=(1, 195, 487),
            maxshape=(None, 195, 487),
        )
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data["data_000001"] = h5py.ExternalLink(
            str(member), "/entry/data/data",
        )


def _grow_eiger_member(member: Path, frames: int) -> None:
    with h5py.File(member, "r+") as handle:
        data = handle["entry/data/data"]
        prior = int(data.shape[0])
        data.resize((frames, *data.shape[1:]))
        data[prior:frames] = 2
        handle.flush()


def _live_intent(
    root: Path,
    output_root: Path,
    poni: Path,
    *,
    suffixes: tuple[str, ...],
    processing_mode: str = "Int 1D",
) -> RunIntent:
    return RunIntent(
        source_spec=DirectorySourceSpec(
            root,
            recursive=False,
            suffixes=suffixes,
            metadata_format=None,
        ),
        poni_file=str(poni),
        project_root=str(root),
        save_path=str(output_root),
        output_mode="Overwrite",
        processing_mode=processing_mode,
        live_mode=True,
        max_cores=1,
        bai_1d_args={"npt": 8, "method": "numpy"},
        bai_2d_args={"npt_rad": 8, "npt_azim": 4, "method": "numpy"},
    )


def _batch_intent(
    image: Path,
    target: Path,
    poni: Path,
) -> RunIntent:
    from xrd_tools.sources.selection import image_series_spec

    return RunIntent(
        source_spec=image_series_spec(image, metadata_format=None),
        poni_file=str(poni),
        project_root=str(image.parent),
        save_path=str(target),
        output_mode="Overwrite",
        processing_mode="Int 1D",
        max_cores=1,
        bai_1d_args={"npt": 8, "method": "numpy"},
        bai_2d_args={"npt_rad": 8, "npt_azim": 4, "method": "numpy"},
    )


def _frame_events(events) -> tuple[object, ...]:
    return tuple(
        event
        for event in events
        if event.kind is StandardEventKind.FRAME_READY
    )


def _output_rows(target: Path) -> tuple[int, ...]:
    if not target.is_file():
        return ()
    with h5py.File(target, "r") as handle:
        return tuple(
            int(value)
            for value in handle["entry/integrated_1d/frame_index"][()]
        )


def _file_facts(paths) -> dict[str, tuple[bytes, int, int, int]]:
    return {
        str(path): (
            path.read_bytes(),
            path.stat().st_dev,
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        )
        for path in sorted(paths)
    }


_XYE_LABEL = re.compile(r"_(-?\d+)\.xye$")


def _xye_labels(root: Path) -> tuple[int, ...]:
    paths = tuple(sorted(root.rglob("iq_*.xye")))
    matches = tuple(_XYE_LABEL.search(path.name) for path in paths)
    assert all(match is not None for match in matches), paths
    return tuple(int(match.group(1)) for match in matches if match is not None)


def _wait_physical_results(
    executor: StandardRunExecutor,
    identity,
    events,
    *,
    expected_rows: tuple[int, ...],
    expected_files: tuple[int, int, int, int],
    output_root: Path,
    timeout: float = 30.0,
) -> tuple[object, ...]:
    """Wait for both physical sinks, then verify projected receipt custody."""

    values = list(events)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        values.extend(executor.drain_events())
        frames = _frame_events(values)
        run = executor._exact_run(identity)
        targets = tuple(dict.fromkeys(
            Path(event.artifact)
            for event in frames
            if event.artifact
        ))
        last_frame_position = max(
            (
                index
                for index, event in enumerate(values)
                if event.kind is StandardEventKind.FRAME_READY
            )
            ,
            default=-1,
        )
        settled = any(
            index > last_frame_position
            and event.kind is StandardEventKind.DISCOVERY
            and (
                event.files_processed,
                event.files_skipped,
                event.files_pending,
                event.files_discovered,
            ) == expected_files
            for index, event in enumerate(values)
        )
        physical_rows = None
        if settled and len(targets) == 1:
            try:
                physical_rows = _output_rows(targets[0])
            except (KeyError, OSError, ValueError):
                pass
        physical = (
            settled
            and len(frames) >= len(expected_rows)
            and len(targets) == 1
            and physical_rows == expected_rows
            and _xye_labels(output_root) == expected_rows
        )
        dynamic = (
            None
            if run is None or run.session is None
            else getattr(run.session, "_dynamic_accounting", None)
        )
        dynamic_settled = True
        if dynamic is not None:
            dynamic_settled = (
                len(_fully_durable_keys(dynamic, dynamic.snapshot()))
                == len(expected_rows)
            )
        receipt_truth = run is not None and all(
            event.frame_key is not None
            and run.display.artifacts[
                event.frame_key.artifact
            ].records.is_persisted(event.frame_key.local_frame_label)
            for event in frames[:len(expected_rows)]
        )
        if physical and settled and dynamic_settled:
            assert receipt_truth
            return tuple(values)
        time.sleep(0.01)
    raise AssertionError(
        "P1-B NeXus/XYE prefix did not become physically durable with receipts"
    )


def _fully_durable_keys(accounting, snapshot) -> frozenset:
    required = accounting.ledger.targets_by_mode
    return frozenset(
        key
        for key in snapshot.discovered
        if all(
            (key, mode, target) in snapshot.durable
            for mode, targets in required.items()
            for target in targets
        )
    )


def _finish_live(
    executor: StandardRunExecutor,
    identity,
    *,
    prior=(),
) -> tuple[object, ...]:
    events = list(prior)
    if not any(event.kind in _TERMINAL for event in events):
        executor.stop(identity)
        events.extend(_drain_until(
            executor,
            lambda values: any(event.kind in _TERMINAL for event in values),
            timeout=30.0,
        ))
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.cleanup_status is CleanupStatus.CLEANED
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED
    return tuple(events)


def _wait_revision_outcome(
    executor: StandardRunExecutor,
    prior,
    *,
    expected_frames: int,
    timeout: float = 20.0,
) -> tuple[object, ...]:
    return _drain_until(
        executor,
        lambda values: (
            len(_frame_events((*prior, *values))) >= expected_frames
            or any(
                "defer" in event.detail.casefold()
                for event in values
                if event.kind is StandardEventKind.DISCOVERY
            )
            or any(event.kind in _TERMINAL for event in values)
        ),
        timeout=timeout,
    )


def test_p1b_b05_empty_live_waits_for_stop_or_arrival(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B05: one empty watcher stays armed until Stop or first READY."""

    from xrd_tools.reduction import NexusSink

    _bridge_legacy_expected_target_state(monkeypatch)
    graph_effects = []
    real_nexus_init = NexusSink.__init__

    def observed_nexus_init(owner, *args, **kwargs):
        real_nexus_init(owner, *args, **kwargs)
        graph_effects.append(owner)

    monkeypatch.setattr(NexusSink, "__init__", observed_nexus_init)
    poni = tmp_path / "cal.poni"
    write_poni(poni)

    empty = tmp_path / "empty"
    empty.mkdir()
    empty_output = tmp_path / "empty-output"
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        _live_intent(empty, empty_output, poni, suffixes=(".tif",)),
        request_value=1501,
    )
    events = _drain_until(
        executor,
        lambda values: any(
            event.kind is StandardEventKind.DISCOVERY
            and event.files_discovered == 0
            for event in values
        ),
    )
    assert _frame_events(events) == ()
    assert not any(event.kind in _TERMINAL for event in events)
    run = executor._exact_run(identity)
    assert run is not None and run.resources is not None
    assert run.resources.directory_session is not None
    assert run.session is None and run.sink is None
    assert run.scan is None and run.records is None
    assert run.display.artifacts == {}
    assert graph_effects == []
    stopped = _finish_live(executor, identity, prior=events)
    assert next(event for event in stopped if event.kind in _TERMINAL).kind \
        is StandardEventKind.STOPPED
    assert tuple(empty_output.rglob("*.nexus")) == ()
    assert tuple(empty_output.rglob("*.xye")) == ()

    arrival = tmp_path / "arrival"
    arrival.mkdir()
    arrival_output = tmp_path / "arrival-output"
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        _live_intent(arrival, arrival_output, poni, suffixes=(".tif",)),
        request_value=1502,
    )
    before = _drain_until(
        executor,
        lambda values: any(
            event.kind is StandardEventKind.DISCOVERY
            and event.files_discovered == 0
            for event in values
        ),
    )
    late = arrival / "late_0001.tif"
    _write_tiff(late, 1)
    after = _drain_until(
        executor,
        lambda values: bool(_frame_events(values)),
        timeout=60.0,
    )
    combined = _wait_physical_results(
        executor,
        identity,
        (*before, *after),
        expected_rows=(1,),
        expected_files=(1, 0, 0, 1),
        output_root=arrival_output,
        timeout=60.0,
    )
    duplicate_polls = executor.drain_events()
    combined = (*combined, *duplicate_polls)
    assert len(_frame_events(combined)) == 1
    assert len(executor.frame_catalog(identity).entries) == 1
    assert len(graph_effects) == 1
    stopped = _finish_live(executor, identity, prior=combined)
    assert next(event for event in stopped if event.kind in _TERMINAL).kind \
        is StandardEventKind.STOPPED
    assert len(tuple(arrival_output.rglob("*.nexus"))) == 1


def test_p1b_b06_growing_nexus_and_h5_extend_one_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B06: both ordinary container suffixes append only their new frame."""

    _bridge_legacy_expected_target_state(monkeypatch)
    results = {}
    for offset, suffix in enumerate((".nxs", ".h5"), start=1):
        root = tmp_path / f"case-{suffix[1:]}"
        root.mkdir()
        source = root / f"growing{suffix}"
        _write_stack(source, 1)
        poni = root / "cal.poni"
        write_poni(poni)
        output = root / "processed"
        executor = StandardRunExecutor(join_timeout=2.0)
        identity = _start(
            executor,
            _live_intent(root, output, poni, suffixes=(suffix,)),
            request_value=1600 + offset,
        )
        first = _drain_until(
            executor,
            lambda values: len(_frame_events(values)) >= 1,
            timeout=60.0,
        )
        first = _wait_physical_results(
            executor,
            identity,
            first,
            expected_rows=(0,),
            expected_files=(1, 0, 0, 1),
            output_root=output,
            timeout=60.0,
        )
        first_run = executor._exact_run(identity)
        assert first_run is not None and first_run.session is not None
        session_identity = first_run.session
        _grow_stack(source, 2)
        second = _wait_revision_outcome(
            executor, first, expected_frames=2, timeout=30.0,
        )
        all_events = (*first, *second)
        if len(_frame_events(all_events)) >= 2:
            all_events = _wait_physical_results(
                executor,
                identity,
                all_events,
                expected_rows=(0, 1),
                expected_files=(1, 0, 0, 1),
                output_root=output,
                timeout=60.0,
            )
        frames = _frame_events(all_events)
        assert tuple(event.completed for event in frames) == (1, 2), suffix
        assert all(
            0 <= event.completed <= event.total for event in frames
        ), suffix
        target = Path(frames[0].artifact)
        current_run = executor._exact_run(identity)
        assert current_run is not None
        assert current_run.session is session_identity
        dynamic = getattr(session_identity, "_dynamic_accounting", None)
        if dynamic is not None:
            assert len(dynamic.snapshot().attempts) == 2
        results[suffix] = (
            len(frames),
            len({event.artifact for event in frames}),
            _output_rows(target),
        )
        _finish_live(executor, identity, prior=all_events)

    for suffix, result in results.items():
        assert result == (2, 1, (0, 1)), suffix


def test_p1b_b07_eiger_dependency_revision_extends_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B07: a member-only Eiger revision extends the master lineage once."""

    _bridge_legacy_expected_target_state(monkeypatch)
    root = tmp_path / "eiger"
    root.mkdir()
    master = root / "scan_master.h5"
    member = tmp_path / "scan_data_000001.h5"
    _write_eiger(master, member, 1)
    master_stamp = master.stat().st_mtime_ns
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    output = tmp_path / "processed"
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        _live_intent(root, output, poni, suffixes=("_master.h5",)),
        request_value=1701,
    )
    first = _drain_until(
        executor,
        lambda values: len(_frame_events(values)) >= 1,
        timeout=60.0,
    )
    first = _wait_physical_results(
        executor,
        identity,
        first,
        expected_rows=(0,),
        expected_files=(1, 0, 0, 1),
        output_root=output,
        timeout=60.0,
    )
    first_run = executor._exact_run(identity)
    assert first_run is not None and first_run.session is not None
    session_identity = first_run.session
    _grow_eiger_member(member, 2)
    assert master.stat().st_mtime_ns == master_stamp
    second = _wait_revision_outcome(
        executor, first, expected_frames=2, timeout=30.0,
    )
    all_events = (*first, *second)
    if len(_frame_events(all_events)) >= 2:
        all_events = _wait_physical_results(
            executor,
            identity,
            all_events,
            expected_rows=(0, 1),
            expected_files=(1, 0, 0, 1),
            output_root=output,
            timeout=60.0,
        )
    frames = _frame_events(all_events)
    target = Path(frames[0].artifact)
    current_run = executor._exact_run(identity)
    assert current_run is not None
    assert current_run.session is session_identity
    dynamic = getattr(session_identity, "_dynamic_accounting", None)
    if dynamic is not None:
        assert len(dynamic.snapshot().attempts) == 2
    _finish_live(executor, identity, prior=all_events)
    assert len(frames) == 2
    assert len({event.artifact for event in frames}) == 1
    assert _output_rows(target) == (0, 1)


def test_p1b_b08_partial_image_files_stabilize_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B08: TIFF/CBF/EDF partial bytes become one exact processed frame."""

    from xrd_tools.sources.adapters import candidate_owner

    _bridge_legacy_expected_target_state(monkeypatch)
    cbf_payload_path = tmp_path / "valid-cbf-payload.cbf"
    _write_flat(cbf_payload_path, "cbf", 2)
    cbf_payload = cbf_payload_path.read_bytes()
    cbf_starter = cbf_payload.index(b"\x0c\x1a\x04\xd5")
    cbf_size = re.search(
        rb"(?m)^X-Binary-Size:[ \t]*([0-9]+)[ \t]*\r?$",
        cbf_payload[:cbf_starter],
    )
    assert cbf_size is not None
    declared_cbf_size = int(cbf_size.group(1))
    cbf_required_end = cbf_starter + 4 + declared_cbf_size
    cbf_staged = cbf_payload[:cbf_required_end - 1]
    assert len(cbf_staged) == cbf_required_end - 1
    cbf_source = tmp_path / "cbf" / "partial_0002.cbf"
    cbf_incomplete_fabio_sizes = []
    cbf_complete_fabio_sizes = []
    real_fabio_open = fabio.open

    def guarded_fabio_open(filename, *args, **kwargs):
        candidate = Path(filename)
        if candidate == cbf_source:
            size = candidate.stat().st_size
            if size < cbf_required_end:
                cbf_incomplete_fabio_sizes.append(size)
                raise AssertionError("Fabio entered an incomplete CBF revision")
            cbf_complete_fabio_sizes.append(size)
        return real_fabio_open(filename, *args, **kwargs)

    monkeypatch.setattr(fabio, "open", guarded_fabio_open)
    outcomes = {}
    for offset, kind in enumerate(("tif", "cbf", "edf"), start=1):
        root = tmp_path / kind
        root.mkdir()
        source = root / f"partial_{offset:04d}.{kind}"
        source.write_bytes(b"incomplete image payload")
        initial_stat = source.stat()
        initial_stamp = (initial_stat.st_size, initial_stat.st_mtime_ns)
        owner = candidate_owner(source)
        assert owner is not None
        assert owner.probe(source).state is ProbeState.IN_PROGRESS, kind
        poni = root / "cal.poni"
        write_poni(poni)
        output = root / "processed"
        executor = StandardRunExecutor(join_timeout=2.0)
        identity = _start(
            executor,
            _live_intent(root, output, poni, suffixes=(f".{kind}",)),
            request_value=1800 + offset,
        )
        pending = _drain_until(
            executor,
            lambda values: any(
                event.kind is StandardEventKind.DISCOVERY
                and event.files_discovered == 1
                and event.files_pending == 1
                for event in values
            ),
        )
        if kind == "cbf":
            source.write_bytes(cbf_staged)
            staged_stamp = (source.stat().st_size, source.stat().st_mtime_ns)
            assert staged_stamp != initial_stamp
            assert owner.probe(source).state is ProbeState.IN_PROGRESS
            assert cbf_incomplete_fabio_sizes == []
            source.write_bytes(cbf_payload)
        else:
            _write_flat(source, kind, offset)
        ready = _drain_until(
            executor,
            lambda values: bool(_frame_events(values)),
            timeout=60.0,
        )
        events = _wait_physical_results(
            executor,
            identity,
            (*pending, *ready),
            expected_rows=(1,),
            expected_files=(1, 0, 0, 1),
            output_root=output,
            timeout=60.0,
        )
        duplicate = executor.drain_events()
        events = (*events, *duplicate)
        frames = _frame_events(events)
        target = Path(frames[0].artifact)
        catalog = executor.frame_catalog(identity)
        cbf_receipt = None
        if kind == "cbf":
            run = executor._exact_run(identity)
            key = frames[0].frame_key
            assert run is not None and key is not None
            cbf_receipt = (
                run.display.artifacts[key.artifact].records,
                key.local_frame_label,
            )
        outcomes[kind] = (
            len(frames),
            0 if catalog is None else len(catalog.entries),
            _output_rows(target),
        )
        _finish_live(executor, identity, prior=events)
        if cbf_receipt is not None:
            assert cbf_incomplete_fabio_sizes == []
            assert cbf_complete_fabio_sizes
            assert min(cbf_complete_fabio_sizes) >= cbf_required_end
            assert cbf_receipt[0].is_persisted(cbf_receipt[1])

    assert outcomes == {
        "tif": (1, 1, (1,)),
        "cbf": (1, 1, (1,)),
        "edf": (1, 1, (1,)),
    }


def test_p1b_b09_flat_series_extends_one_output_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B09: late TIFF/known-RAW members extend their exact output group."""

    _bridge_legacy_expected_target_state(monkeypatch)
    outcomes = {}
    for offset, kind in enumerate(("tif", "raw"), start=1):
        root = tmp_path / kind
        root.mkdir()
        first_path = root / f"series_0001.{kind}"
        second_path = root / f"series_0002.{kind}"
        writer = _write_tiff if kind == "tif" else _write_raw
        writer(first_path, 1)
        poni = root / "cal.poni"
        write_poni(poni)
        output = root / "processed"
        executor = StandardRunExecutor(join_timeout=2.0)
        identity = _start(
            executor,
            _live_intent(root, output, poni, suffixes=(f".{kind}",)),
            request_value=1900 + offset,
        )
        first = _drain_until(
            executor,
            lambda values: len(_frame_events(values)) >= 1,
            timeout=60.0,
        )
        first = _wait_physical_results(
            executor,
            identity,
            first,
            expected_rows=(1,),
            expected_files=(1, 0, 0, 1),
            output_root=output,
            timeout=60.0,
        )
        first_run = executor._exact_run(identity)
        assert first_run is not None and first_run.session is not None
        session_identity = first_run.session
        writer(second_path, 2)
        second = _wait_revision_outcome(
            executor, first, expected_frames=2, timeout=30.0,
        )
        events = (*first, *second)
        if len(_frame_events(events)) >= 2:
            events = _wait_physical_results(
                executor,
                identity,
                events,
                expected_rows=(1, 2),
                expected_files=(2, 0, 0, 2),
                output_root=output,
                timeout=60.0,
            )
        frames = _frame_events(events)
        target = Path(frames[0].artifact)
        current_run = executor._exact_run(identity)
        assert current_run is not None
        assert current_run.session is session_identity
        dynamic = getattr(session_identity, "_dynamic_accounting", None)
        if dynamic is not None:
            assert len(dynamic.snapshot().attempts) == 2
        catalog = executor.frame_catalog(identity)
        assert catalog is not None
        local_entries = tuple(
            key for key in catalog.entries if key.artifact == str(target)
        )
        assert len(local_entries) == 2, kind
        assert tuple(key.local_frame_label for key in local_entries) == (1, 2)
        assert frames[-1].frame_key is not None
        assert frames[-1].frame_key.local_frame_label == 2, kind
        assert frames[-1].artifact_total == 2, kind
        outcomes[kind] = (
            len(frames),
            len({event.artifact for event in frames}),
            _output_rows(target),
        )
        _finish_live(executor, identity, prior=events)

    assert outcomes == {
        "tif": (2, 1, (1, 2)),
        "raw": (2, 1, (1, 2)),
    }


def test_p1b_b10_raw_size_and_unknown_tail_are_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B10: RAW readiness is exact-size typed truth with no geometry guess."""

    import xrd_tools.io.image as image_io
    from xrd_tools.reduction import NexusSink
    from xrd_tools.sources.adapters import candidate_owner
    from xrd_tools.session.dynamic_accounting import DynamicRunAccounting

    _bridge_legacy_expected_target_state(monkeypatch)
    reads: list[Path] = []
    real_read = image_io.read_image

    def observed_read(path, *args, **kwargs):
        reads.append(Path(path).resolve())
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(image_io, "read_image", observed_read)

    accounting_owners = []
    real_init = DynamicRunAccounting.__init__

    def observed_init(owner, *args, **kwargs):
        real_init(owner, *args, **kwargs)
        accounting_owners.append(owner)

    monkeypatch.setattr(DynamicRunAccounting, "__init__", observed_init)
    graph_effects = []
    real_nexus_init = NexusSink.__init__

    def observed_nexus_init(owner, *args, **kwargs):
        real_nexus_init(owner, *args, **kwargs)
        graph_effects.append(owner)

    monkeypatch.setattr(NexusSink, "__init__", observed_nexus_init)

    poni = tmp_path / "cal.poni"
    write_poni(poni)
    exact = np.ones((195, 487), dtype=np.int32).tobytes()
    initial_payloads = {
        "known-partial": exact[:len(exact) // 3],
        "unknown-tail": np.ones((17, 19), dtype=np.int32).tobytes(),
    }
    outcomes = {}
    for offset, (domain, initial_payload) in enumerate(
        initial_payloads.items(), start=1,
    ):
        live_root = tmp_path / domain
        live_root.mkdir()
        source = live_root / f"{domain}_{1:04d}.raw"
        source.write_bytes(initial_payload)
        owner = candidate_owner(source)
        assert owner is not None
        pending = owner.probe(source)
        assert pending.state is ProbeState.IN_PROGRESS, domain
        assert pending.reason == (
            "RAW payload does not yet match one known detector"
        ), domain
        assert reads == [], domain

        output = live_root / "processed"
        graph_before = len(graph_effects)
        accounting_before = len(accounting_owners)
        executor = StandardRunExecutor(join_timeout=2.0)
        identity = _start(
            executor,
            _live_intent(live_root, output, poni, suffixes=(".raw",)),
            request_value=2000 + offset,
        )
        pending_events = _drain_until(
            executor,
            lambda values: any(
                event.kind is StandardEventKind.DISCOVERY
                and (
                    event.files_processed,
                    event.files_skipped,
                    event.files_pending,
                    event.files_discovered,
                ) == (0, 0, 1, 1)
                for event in values
            ),
        )
        run = executor._exact_run(identity)
        assert run is not None
        assert run.session is None and run.sink is None and run.records is None
        assert run.display.artifacts == {}
        assert _frame_events(pending_events) == ()
        assert len(graph_effects) == graph_before
        assert len(accounting_owners) == accounting_before
        assert reads == []
        assert not output.exists() or tuple(output.rglob("*")) == ()

        source.write_bytes(exact)
        ready_events = _drain_until(
            executor,
            lambda values: bool(_frame_events(values)),
            timeout=60.0,
        )
        all_events = _wait_physical_results(
            executor,
            identity,
            (*pending_events, *ready_events),
            expected_rows=(1,),
            expected_files=(1, 0, 0, 1),
            output_root=output,
            timeout=60.0,
        )
        assert reads and set(reads) == {source.resolve()}, domain
        frames = _frame_events(all_events)
        assert len(frames) == 1, domain
        assert len(executor.frame_catalog(identity).entries) == 1, domain
        assert len(graph_effects) == graph_before + 1, domain
        assert len(tuple(output.rglob("*.nexus"))) == 1, domain
        assert _xye_labels(output) == (1,), domain

        new_accounting = accounting_owners[accounting_before:]
        if new_accounting:
            assert len(new_accounting) == 1, domain
            deadline = time.monotonic() + 20.0
            snapshot = new_accounting[0].snapshot()
            while (
                len(_fully_durable_keys(new_accounting[0], snapshot)) != 1
                and time.monotonic() < deadline
            ):
                executor.drain_events()
                snapshot = new_accounting[0].snapshot()
                time.sleep(0.01)
            assert len(_fully_durable_keys(new_accounting[0], snapshot)) == 1
        outcomes[domain] = (
            len(frames),
            len(executor.frame_catalog(identity).entries),
            _output_rows(Path(frames[0].artifact)),
        )
        _finish_live(executor, identity, prior=all_events)
        reads.clear()

    assert outcomes == {
        "known-partial": (1, 1, (1,)),
        "unknown-tail": (1, 1, (1,)),
    }


def test_p1b_b11_nonmonotonic_revision_preserves_durable_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B11: shrink/rewrite refuses without changing the durable prefix."""

    from xrd_tools.io.append import AppendRefused

    _bridge_legacy_expected_target_state(monkeypatch)
    outcomes = {}
    for offset, kind in enumerate(("container", "flat"), start=1):
        root = tmp_path / kind
        root.mkdir()
        poni = root / "cal.poni"
        write_poni(poni)
        if kind == "container":
            source = root / "shrinking.nxs"
            _write_stack(source, 2)
            suffixes = (".nxs",)
            expected_initial = 2
        else:
            source = root / "mutating_0001.tif"
            _write_tiff(source, 1)
            suffixes = (".tif",)
            expected_initial = 1
        expected_rows = (0, 1) if kind == "container" else (1,)
        output = root / "processed"
        executor = StandardRunExecutor(join_timeout=2.0)
        identity = _start(
            executor,
            _live_intent(root, output, poni, suffixes=suffixes),
            request_value=2100 + offset,
        )
        initial = _drain_until(
            executor,
            lambda values: len(_frame_events(values)) >= expected_initial,
            timeout=60.0,
        )
        initial = _wait_physical_results(
            executor,
            identity,
            initial,
            expected_rows=expected_rows,
            expected_files=(1, 0, 0, 1),
            output_root=output,
            timeout=60.0,
        )
        frames = _frame_events(initial)
        target = Path(frames[0].artifact)
        before = target.read_bytes()
        assert before
        xye_paths = tuple(output.rglob("iq_*.xye"))
        assert len(xye_paths) == expected_initial
        assert _xye_labels(output) == expected_rows
        xye_before = _file_facts(xye_paths)
        run = executor._exact_run(identity)
        assert run is not None
        stores = {
            event.frame_key.local_frame_label:
            run.display.artifacts[event.frame_key.artifact].records
            for event in frames
            if event.frame_key is not None
        }
        assert all(store.is_persisted(label) for label, store in stores.items())

        if kind == "container":
            _grow_stack(source, 1)
        else:
            _write_tiff(source, 9)
        revised = _wait_revision_outcome(
            executor,
            initial,
            expected_frames=expected_initial + 1,
            timeout=30.0,
        )
        terminal = next(
            (event for event in revised if event.kind in _TERMINAL),
            None,
        )
        outcomes[kind] = (
            terminal,
            target.read_bytes() == before,
            _file_facts(tuple(output.rglob("iq_*.xye"))) == xye_before,
            all(store.is_persisted(label) for label, store in stores.items()),
        )
        if terminal is None:
            _finish_live(executor, identity, prior=(*initial, *revised))
        else:
            _finish_live(executor, identity, prior=revised)

    for kind, (
        terminal,
        bytes_preserved,
        xye_preserved,
        receipts_preserved,
    ) in outcomes.items():
        assert terminal is not None, f"{kind}: nonmonotonic revision was deferred"
        assert terminal.kind is StandardEventKind.FAILED, kind
        assert terminal.primary is not None, kind
        assert terminal.primary.type_qualname in {
            AppendRefused.__qualname__,
            "AppendSourceGraphRefused",
        }, kind
        assert bytes_preserved and xye_preserved and receipts_preserved, kind


def test_p1b_b12_stop_cancel_preserves_exact_durable_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B12: four Stop phases never project staged work as persisted."""

    import xrd_tools.reduction.core as reduction_core
    from xrd_tools.reduction import NexusSink
    from xrd_tools.sources.image import TiffSeriesSource

    results = {}
    for offset, phase in enumerate(
        ("read", "reduction", "staging", "publication"),
        start=1,
    ):
        case = tmp_path / phase
        case.mkdir()
        image = case / "stop_0001.tif"
        _write_tiff(image, offset)
        poni = case / "cal.poni"
        write_poni(poni)
        target = case / "stop.nexus"
        entered = Event()
        release = Event()
        with monkeypatch.context() as scoped:
            _bridge_legacy_expected_target_state(scoped)
            if phase == "read":
                real = TiffSeriesSource._read_path

                def blocked(owner, path, _real=real):
                    entered.set()
                    assert release.wait(20.0)
                    return _real(owner, path)

                scoped.setattr(TiffSeriesSource, "_read_path", blocked)
            elif phase == "reduction":
                real = reduction_core._reduce_frame

                def blocked(*args, _real=real, **kwargs):
                    entered.set()
                    assert release.wait(20.0)
                    return _real(*args, **kwargs)

                scoped.setattr(reduction_core, "_reduce_frame", blocked)
            elif phase == "staging":
                real = NexusSink.write

                def blocked(owner, frame, reduction, _real=real):
                    entered.set()
                    assert release.wait(20.0)
                    return _real(owner, frame, reduction)

                scoped.setattr(NexusSink, "write", blocked)
            else:
                real = NexusSink.finish

                def blocked(owner, result, _real=real):
                    entered.set()
                    assert release.wait(20.0)
                    return _real(owner, result)

                scoped.setattr(NexusSink, "finish", blocked)

            executor = StandardRunExecutor(join_timeout=2.0)
            identity = None
            terminal_events = ()
            try:
                try:
                    identity = _start(
                        executor,
                        _batch_intent(image, target, poni),
                        request_value=2200 + offset,
                    )
                    assert entered.wait(60.0), phase
                    early_events = executor.drain_events()
                    run = executor._exact_run(identity)
                    assert run is not None
                    dynamic = (
                        None
                        if run.session is None
                        else getattr(run.session, "_dynamic_accounting", None)
                    )
                    early_persisted = []
                    for event in _frame_events(early_events):
                        assert event.frame_key is not None
                        store = run.display.artifacts[
                            event.frame_key.artifact
                        ].records
                        early_persisted.append(store.is_persisted(
                            event.frame_key.local_frame_label
                        ))
                    stop_started = time.monotonic()
                    executor.stop(identity)
                    stop_elapsed = time.monotonic() - stop_started
                    worker_alive_after_stop = (
                        run.worker is not None and run.worker.is_alive()
                    )
                finally:
                    # Every injected phase is released even if setup/assertion
                    # fails, so no blocked worker escapes this finite row.
                    release.set()

                terminal_events = _drain_until(
                    executor,
                    lambda values: any(
                        event.kind in _TERMINAL for event in values
                    ),
                    timeout=60.0,
                )
                terminal = next(
                    event
                    for event in terminal_events
                    if event.kind in _TERMINAL
                )
                all_frames = _frame_events((*early_events, *terminal_events))
                exact_keys = tuple(dict.fromkeys(
                    event.frame_key
                    for event in all_frames
                    if event.frame_key is not None
                ))
                persisted_labels = tuple(sorted(
                    key.local_frame_label
                    for key in exact_keys
                    if run.display.artifacts[
                        key.artifact
                    ].records.is_persisted(key.local_frame_label)
                ))
                rows = _output_rows(_written(target))
                xye_labels = _xye_labels(case)
                dynamic_durable_count = len(persisted_labels)
                if dynamic is not None:
                    snapshot = dynamic.snapshot()
                    fully_durable = _fully_durable_keys(dynamic, snapshot)
                    dynamic_durable_count = len(fully_durable)
                results[phase] = (
                    terminal.kind,
                    terminal.primary,
                    tuple(early_persisted),
                    persisted_labels,
                    rows,
                    xye_labels,
                    dynamic_durable_count,
                    terminal.cleanup_status,
                    stop_elapsed,
                    worker_alive_after_stop,
                )
            finally:
                release.set()
                if identity is not None:
                    if not any(
                        event.kind in _TERMINAL for event in terminal_events
                    ):
                        executor.stop(identity)
                        try:
                            _drain_until(
                                executor,
                                lambda values: any(
                                    event.kind in _TERMINAL for event in values
                                ),
                                timeout=60.0,
                            )
                        except AssertionError:
                            pass
                    first_close = executor.close(identity)
                    second_close = executor.close(identity)
                    assert first_close.cleanup_status is CleanupStatus.CLEANED
                    assert second_close.cleanup_status is CleanupStatus.CLEANED

    for phase, result in results.items():
        (
            terminal,
            primary,
            early_persisted,
            persisted_labels,
            rows,
            xye_labels,
            dynamic_durable_count,
            cleanup,
            stop_elapsed,
            worker_alive_after_stop,
        ) = result
        assert terminal is StandardEventKind.STOPPED, (phase, primary)
        assert primary is None, phase
        assert not any(early_persisted), f"{phase}: staged record claimed persisted"
        assert persisted_labels == rows == xye_labels, phase
        assert dynamic_durable_count == len(persisted_labels), phase
        assert cleanup is CleanupStatus.CLEANED, phase
        assert stop_elapsed < 0.5, f"{phase}: Stop blocked for {stop_elapsed:.3f}s"
        assert worker_alive_after_stop, f"{phase}: Stop settled worker inline"

    # A submit already past the adapter's short attempt boundary may remain
    # backpressured under AcquisitionRuntime's command lock.  request_stop()
    # must bypass that lock, return promptly, and fence every later attempt.
    from xdart.gui.tabs.scattering.acquisition_runtime import AcquisitionRuntime

    class BlockingAdapter:
        def __init__(self) -> None:
            self.lock = Lock()
            self.stopped = False
            self.minted = 0
            self.entered = Event()
            self.release = Event()

        def submit(self, _frame) -> bool:
            with self.lock:
                if self.stopped:
                    return False
                self.minted += 1
            self.entered.set()
            assert self.release.wait(5.0)
            return True

        def stop(self) -> None:
            with self.lock:
                self.stopped = True

    runtime = AcquisitionRuntime()
    blocked = BlockingAdapter()
    submit_values = []
    submit_worker = Thread(
        target=lambda: submit_values.append(runtime.submit(blocked, object())),
        name="p1b-backpressured-submit",
    )
    stop_returned = Event()
    stop_errors = []

    def request_blocked_stop() -> None:
        try:
            runtime.request_stop(blocked)
        except BaseException as error:
            stop_errors.append(error)
        finally:
            stop_returned.set()

    stop_worker = Thread(
        target=request_blocked_stop,
        name="p1b-prompt-stop",
    )
    submit_worker.start()
    assert blocked.entered.wait(1.0)
    try:
        stop_worker.start()
        assert stop_returned.wait(0.5), "record-only Stop waited on submit"
        assert submit_worker.is_alive()
    finally:
        blocked.release.set()
        submit_worker.join(2.0)
        stop_worker.join(2.0)
    assert not submit_worker.is_alive() and not stop_worker.is_alive()
    assert stop_errors == [] and submit_values == [True]
    assert blocked.minted == 1
    assert runtime.submit(blocked, object()) is False
    assert blocked.minted == 1

    # A durable Pause clears the same runtime gate.  A worker parked there is
    # woken by request_stop(), observes the adapter fence, and mints nothing.
    class PausedSession:
        def pause(self, *, timeout: float) -> bool:
            return True

        def resume(self) -> None:
            raise AssertionError("Stop must not compensate Pause with resume")

    paused_runtime = AcquisitionRuntime()
    paused_adapter = BlockingAdapter()
    paused_runtime.pause(PausedSession(), object(), 1.0)
    paused_values = []
    paused_worker = Thread(
        target=lambda: paused_values.append(
            paused_runtime.submit(paused_adapter, object())
        ),
        name="p1b-paused-submit",
    )
    paused_worker.start()
    assert not paused_adapter.entered.wait(0.1)
    paused_runtime.request_stop(paused_adapter)
    paused_worker.join(2.0)
    assert not paused_worker.is_alive()
    assert paused_values == [False]
    assert paused_adapter.minted == 0

    # Projection-failure termination uses the same dynamic request-Stop door,
    # reopens the Pause gate, and never calls the peer core-session Stop door.
    from xdart.gui.tabs.scattering.adapters.run_executor import _StandardRun
    from xdart.gui.tabs.scattering.events import RunIdentity, detach_exception
    from xrd_tools.session.run_configuration import RunIntent

    class ProjectionOutput:
        def __init__(self) -> None:
            self.stops = 0
            self.finishes = 0

        def stop(self) -> None:
            self.stops += 1

        def finish_all(self, *, stopped: bool = False):
            assert stopped is True
            self.finishes += 1
            return ()

        def project_new_durable(self, _apply) -> None:
            return None

    class PeerSession:
        def __init__(self) -> None:
            self.stops = 0

        def stop(self) -> None:
            self.stops += 1

    projection_executor = StandardRunExecutor(join_timeout=0.2)
    projection_configuration = RunIntent().freeze()
    projection_identity = RunIdentity.from_configuration(
        projection_configuration
    )
    projection_output = ProjectionOutput()
    peer_session = PeerSession()
    projection_runtime = AcquisitionRuntime()
    projection_runtime._gate.clear()
    projection_run = _StandardRun(
        projection_configuration,
        projection_identity,
        None,
        None,
        peer_session,
        None,
        tmp_path / "projection-failure.nexus",
        sink=projection_output,
        output=projection_output,
        context_runtime=projection_runtime,
    )
    projection_executor._active = projection_run
    projection_receipt = projection_executor._terminate_projection_failure(
        projection_run,
        detach_exception(
            RuntimeError("injected display projection failure"),
            "context.pause",
        ),
    )
    assert projection_output.stops == projection_output.finishes == 1
    assert peer_session.stops == 0
    assert projection_runtime._gate.is_set()
    assert projection_receipt.cleanup_status is CleanupStatus.CLEANED
    assert projection_receipt.primary is not None
    assert projection_receipt.primary.message == (
        "injected display projection failure"
    )
    projection_terminal = projection_executor.drain_events()
    assert len(projection_terminal) == 1
    assert projection_terminal[0].kind is StandardEventKind.FAILED

    # The normal worker path (rather than Pause's explicit termination path)
    # must route its display-projection failure through the same wake door.
    from types import SimpleNamespace

    class WorkerProjectionOutput(ProjectionOutput):
        write_labels = (0,)

        def __init__(self) -> None:
            super().__init__()
            self.submits = 0

        def submit(self, _frame) -> bool:
            self.submits += 1
            return True

        def finish_current(self):
            return SimpleNamespace(failed=False, cancelled=False)

    class WorkerPeerSession(PeerSession):
        def __init__(self) -> None:
            super().__init__()
            self.starts = 0

        def start(self) -> None:
            self.starts += 1

    worker_executor = StandardRunExecutor(join_timeout=0.2)
    worker_configuration = RunIntent().freeze()
    worker_identity = RunIdentity.from_configuration(worker_configuration)
    worker_output = WorkerProjectionOutput()
    worker_session = WorkerPeerSession()
    worker_runtime = AcquisitionRuntime()
    worker_run = _StandardRun(
        worker_configuration,
        worker_identity,
        SimpleNamespace(frames=(SimpleNamespace(index=0),)),
        None,
        worker_session,
        None,
        tmp_path / "worker-projection-failure.nexus",
        sink=worker_output,
        output=worker_output,
        context_runtime=worker_runtime,
    )
    worker_executor._active = worker_run

    def fail_worker_projection(_run) -> None:
        worker_runtime._gate.clear()
        raise RuntimeError("injected worker projection failure")

    with monkeypatch.context() as worker_fault:
        worker_fault.setattr(
            worker_executor,
            "_finish_display_projection",
            fail_worker_projection,
        )
        with pytest.raises(
            RuntimeError, match="injected worker projection failure",
        ):
            worker_executor._execute_current(worker_run, construct=False)
    assert worker_run.stop_requested and worker_run.stop_signal.is_set()
    assert worker_runtime._gate.is_set()
    assert worker_output.stops == 1
    assert worker_session.stops == 0
    assert worker_session.starts == worker_output.submits == 1
    assert worker_executor._cleanup(
        worker_run
    ).cleanup_status is CleanupStatus.CLEANED

    # A projection worker can fail after the bounded queue becomes full.  Its
    # retirement sentinel must not wait forever for capacity that the failed
    # worker can no longer release.
    from queue import Empty, Queue
    import xdart.gui.tabs.scattering.adapters.run_executor as run_executor_module

    projection_entered = Event()
    projection_release = Event()
    sentinel_attempted = Event()
    finish_returned = Event()
    finish_errors = []

    class ObservedQueue(Queue):
        def put(self, item, *args, **kwargs):
            if item is run_executor_module._DISPLAY_PROJECTION_END:
                sentinel_attempted.set()
            return super().put(item, *args, **kwargs)

    queue_executor = StandardRunExecutor(join_timeout=0.2)
    queue_configuration = RunIntent().freeze()
    queue_identity = RunIdentity.from_configuration(queue_configuration)
    queue_run = _StandardRun(
        queue_configuration,
        queue_identity,
        None,
        None,
        None,
        None,
        tmp_path / "full-projection-queue.nexus",
    )

    def fail_projection(*_args) -> None:
        projection_entered.set()
        assert projection_release.wait(5.0)
        raise RuntimeError("injected full projection queue failure")

    def finish_projection() -> None:
        try:
            queue_executor._finish_display_projection(queue_run)
        except BaseException as error:
            finish_errors.append(error)
        finally:
            finish_returned.set()

    finisher = Thread(
        target=finish_projection,
        name="p1b-full-projection-queue-finish",
    )
    sentinel_was_attempted = False
    returned_without_rescue = False
    pending = None
    projection_worker = None
    try:
        with monkeypatch.context() as queue_fault:
            queue_fault.setattr(run_executor_module, "Queue", ObservedQueue)
            queue_fault.setattr(
                queue_executor, "_frame_ready_owned", fail_projection,
            )
            queue_executor._start_display_projection(queue_run)
            pending = queue_run.display_projection_queue
            projection_worker = queue_run.display_projection_worker
            assert pending is not None and projection_worker is not None
            from xrd_tools.core import FrameRecord
            queued = run_executor_module._FrameProjectionItem(
                0, FrameRecord(0), False,
            )
            pending.put(queued)
            assert projection_entered.wait(2.0)
            for _ in range(pending.maxsize):
                pending.put(queued)
            finisher.start()
            sentinel_was_attempted = sentinel_attempted.wait(1.0)
            projection_release.set()
            projection_worker.join(2.0)
            returned_without_rescue = finish_returned.wait(0.5)
    finally:
        projection_release.set()
        if projection_worker is not None:
            projection_worker.join(2.0)
        if finisher.is_alive() and pending is not None:
            try:
                pending.get_nowait()
            except Empty:
                pass
            else:
                pending.task_done()
        if finisher.ident is not None:
            finisher.join(2.0)
        if pending is not None:
            while True:
                try:
                    pending.get_nowait()
                except Empty:
                    break
                else:
                    pending.task_done()

    assert sentinel_was_attempted
    assert not finisher.is_alive()
    assert projection_worker is not None
    assert not projection_worker.is_alive()
    assert returned_without_rescue, "full queue stranded projection retirement"
    assert len(finish_errors) == 1
    assert type(finish_errors[0]) is RuntimeError
    assert str(finish_errors[0]) == (
        "injected full projection queue failure"
    )
    assert queue_run.display_projection_queue is None
    assert queue_run.display_projection_worker is None

    # A container producer can fail with one decoded item already handed to
    # the submission worker.  The producer's request-Stop fence must win before
    # that queued item mints an attempt, and the source error stays primary.
    from xdart.gui.tabs.scattering.adapters.dynamic_output import (
        DynamicOutputAdapter,
    )
    from xrd_tools.sources.nexus import NexusStackSource

    producer_case = tmp_path / "producer-failure"
    producer_case.mkdir()
    producer_source = producer_case / "source.nxs"
    producer_target = producer_case / "output.nexus"
    producer_poni = producer_case / "cal.poni"
    _write_stack(producer_source, 2)
    write_poni(producer_poni)
    submit_entered = Event()
    submit_release = Event()
    stop_seen = Event()
    real_submit = DynamicOutputAdapter.submit
    real_stop = DynamicOutputAdapter.stop

    def blocked_queued_submit(owner, frame, image):
        submit_entered.set()
        assert submit_release.wait(5.0)
        return real_submit(owner, frame, image)

    def observed_record_stop(owner) -> None:
        real_stop(owner)
        stop_seen.set()
        raise RuntimeError("injected record-Stop routing failure")

    def failing_chunks(_owner, _chunk_size):
        yield np.ones((1, 195, 487), dtype=np.uint16), [0]
        assert submit_entered.wait(5.0)
        raise OSError("injected container source read failure")

    producer_executor = StandardRunExecutor(join_timeout=2.0)
    producer_identity = None
    producer_events = ()
    with monkeypatch.context() as producer_fault:
        _bridge_legacy_expected_target_state(producer_fault)
        producer_fault.setattr(
            DynamicOutputAdapter, "submit", blocked_queued_submit,
        )
        producer_fault.setattr(
            DynamicOutputAdapter, "stop", observed_record_stop,
        )
        producer_fault.setattr(
            NexusStackSource, "iter_chunks", failing_chunks,
        )
        try:
            producer_identity = _start(
                producer_executor,
                _batch_intent(
                    producer_source, producer_target, producer_poni,
                ),
                request_value=2210,
            )
            assert submit_entered.wait(60.0)
            assert stop_seen.wait(2.0)
            producer_run = producer_executor._exact_run(producer_identity)
            assert producer_run is not None and producer_run.output is not None
            producer_accounting = producer_run.output.accounting
            producer_snapshot = producer_accounting.snapshot()
            assert sum(
                len(values) for values in producer_snapshot.attempts.values()
            ) == 0
        finally:
            submit_release.set()
        producer_events = _drain_until(
            producer_executor,
            lambda values: any(event.kind in _TERMINAL for event in values),
            timeout=60.0,
        )
    producer_terminal = next(
        event for event in producer_events if event.kind in _TERMINAL
    )
    assert producer_terminal.kind is StandardEventKind.FAILED
    assert producer_terminal.primary is not None
    assert producer_terminal.primary.type_qualname == OSError.__qualname__
    assert producer_terminal.primary.message == (
        "injected container source read failure"
    )
    assert any(
        failure.message == "injected record-Stop routing failure"
        and failure.operation == "dynamic_output.stop"
        for failure in producer_terminal.cleanup_failures
    )
    assert _frame_events(producer_events) == ()
    assert _output_rows(producer_target) == ()
    assert _xye_labels(producer_case) == ()
    assert producer_executor.close(
        producer_identity
    ).cleanup_status is CleanupStatus.CLEANED

    # Cleanup owns an exact retry when the first Live epoch publication fails.
    # Stop arriving while that transition is held is record-only; after release
    # the retry creates the first canonical heavy/light durable prefix exactly
    # once and applies the pending core Stop outside the adapter lock.
    from xrd_tools.session.scan_session import ScanSession

    epoch_case = tmp_path / "epoch-retry"
    epoch_case.mkdir()
    epoch_source = epoch_case / "epoch_0001.tif"
    epoch_poni = epoch_case / "cal.poni"
    epoch_output = epoch_case / "processed"
    _write_tiff(epoch_source, 11)
    write_poni(epoch_poni)
    epoch_entered = Event()
    epoch_release = Event()
    epoch_calls = []
    epoch_stop_calls = []
    epoch_finish_calls = []
    real_epoch = ScanSession.commit_epoch
    real_session_stop = ScanSession.stop
    real_session_finish = ScanSession.finish

    def transient_epoch(owner):
        epoch_calls.append(owner)
        if len(epoch_calls) == 1:
            epoch_entered.set()
            assert epoch_release.wait(5.0)
            raise OSError("injected transient epoch settlement failure")
        return real_epoch(owner)

    def observed_session_stop(owner) -> None:
        epoch_stop_calls.append(owner)
        return real_session_stop(owner)

    def observed_session_finish(owner, *args, **kwargs):
        epoch_finish_calls.append(owner)
        return real_session_finish(owner, *args, **kwargs)

    epoch_executor = StandardRunExecutor(join_timeout=2.0)
    epoch_identity = None
    epoch_events = ()
    early_epoch_events = ()
    with monkeypatch.context() as epoch_fault:
        _bridge_legacy_expected_target_state(epoch_fault)
        epoch_fault.setattr(ScanSession, "commit_epoch", transient_epoch)
        epoch_fault.setattr(ScanSession, "stop", observed_session_stop)
        epoch_fault.setattr(ScanSession, "finish", observed_session_finish)
        try:
            epoch_identity = _start(
                epoch_executor,
                _live_intent(
                    epoch_case,
                    epoch_output,
                    epoch_poni,
                    suffixes=(".tif",),
                ),
                request_value=2211,
            )
            assert epoch_entered.wait(60.0)
            epoch_run = epoch_executor._exact_run(epoch_identity)
            assert epoch_run is not None and epoch_run.output is not None
            epoch_adapter = epoch_run.output
            epoch_graph = next(iter(epoch_adapter._graphs.values()))
            epoch_dynamic = epoch_graph["accounting"]
            early_epoch_events = epoch_executor.drain_events()
            stop_started = time.monotonic()
            epoch_executor.stop(epoch_identity)
            epoch_stop_elapsed = time.monotonic() - stop_started
            assert epoch_stop_elapsed < 0.5
            assert epoch_run.worker is not None and epoch_run.worker.is_alive()
            assert epoch_stop_calls == []
        finally:
            epoch_release.set()
        epoch_events = _drain_until(
            epoch_executor,
            lambda values: any(event.kind in _TERMINAL for event in values),
            timeout=60.0,
        )

    epoch_all_events = (*early_epoch_events, *epoch_events)
    epoch_terminal = next(
        event for event in epoch_all_events if event.kind in _TERMINAL
    )
    epoch_frames = _frame_events(epoch_all_events)
    assert len(epoch_frames) == 1, tuple(
        (event.kind, event.detail, event.cleanup_status, event.primary)
        for event in epoch_all_events
    )
    epoch_key = epoch_frames[0].frame_key
    assert epoch_key is not None
    epoch_label = epoch_key.local_frame_label
    epoch_target = Path(epoch_frames[0].artifact)
    artifact_owner = epoch_run.display.artifacts[epoch_key.artifact]
    heavy_store = epoch_graph["record_store"]
    light_lease = artifact_owner.light_lease
    assert heavy_store is artifact_owner.records
    assert not hasattr(artifact_owner, "light_records")
    assert light_lease is not None
    assert artifact_owner.publications._light_1d is light_lease
    assert heavy_store.labels() == (epoch_label,)
    assert light_lease.keys() == (epoch_label,)
    assert heavy_store.get(epoch_label) is not None
    borrowed = light_lease.borrow(epoch_label)
    assert borrowed is not None
    borrowed.close()
    assert epoch_terminal.kind is StandardEventKind.FAILED
    assert epoch_terminal.primary is not None
    assert epoch_terminal.primary.type_qualname == OSError.__qualname__
    assert epoch_terminal.primary.message == (
        "injected transient epoch settlement failure"
    )
    assert epoch_terminal.cleanup_status is CleanupStatus.CLEANED
    assert _output_rows(epoch_target) == (epoch_label,)
    assert _xye_labels(epoch_output) == (epoch_label,)
    assert light_lease.state.value == "active"
    assert heavy_store.is_persisted(epoch_label)
    assert len(_fully_durable_keys(
        epoch_dynamic, epoch_dynamic.snapshot()
    )) == 1
    assert epoch_run.completed == epoch_terminal.completed == 1
    assert epoch_terminal.artifact_completed == 1
    assert epoch_terminal.artifacts == (str(epoch_target),)
    assert len(epoch_calls) == 2
    assert len(epoch_stop_calls) == len(epoch_finish_calls) == 1
    epoch_counts = (
        epoch_run.completed,
        len(epoch_calls),
        len(epoch_stop_calls),
        len(epoch_finish_calls),
    )
    assert epoch_executor.close(
        epoch_identity
    ).cleanup_status is CleanupStatus.CLEANED
    assert epoch_executor.close(
        epoch_identity
    ).cleanup_status is CleanupStatus.CLEANED
    assert heavy_store.is_persisted(epoch_label)
    assert light_lease.state.value == "released"
    assert light_lease.retained_count == 0
    assert (
        epoch_run.completed,
        len(epoch_calls),
        len(epoch_stop_calls),
        len(epoch_finish_calls),
    ) == epoch_counts

    # A genuine pre-settlement reduction failure has no resumable XYE epoch.
    # Cleanup must finish/abort it terminally rather than retain epoch custody
    # forever or claim either heavy or light durability.
    failed_case = tmp_path / "pre-settlement-failure"
    failed_case.mkdir()
    failed_source = failed_case / "failed_0001.tif"
    failed_poni = failed_case / "cal.poni"
    failed_output = failed_case / "processed"
    _write_tiff(failed_source, 12)
    write_poni(failed_poni)
    reduction_failed_entered = Event()
    reduction_failed_release = Event()

    def fail_reduction(*_args, **_kwargs):
        reduction_failed_entered.set()
        assert reduction_failed_release.wait(5.0)
        raise RuntimeError("injected pre-settlement reduction failure")

    failed_executor = StandardRunExecutor(join_timeout=2.0)
    failed_identity = None
    failed_events = ()
    with monkeypatch.context() as reduction_fault:
        _bridge_legacy_expected_target_state(reduction_fault)
        reduction_fault.setattr(reduction_core, "_reduce_frame", fail_reduction)
        try:
            failed_identity = _start(
                failed_executor,
                _live_intent(
                    failed_case,
                    failed_output,
                    failed_poni,
                    suffixes=(".tif",),
                ),
                request_value=2212,
            )
            assert reduction_failed_entered.wait(60.0)
            failed_run = failed_executor._exact_run(failed_identity)
            assert failed_run is not None and failed_run.output is not None
            failed_adapter = failed_run.output
            failed_graph = next(iter(failed_adapter._graphs.values()))
            failed_dynamic = failed_graph["accounting"]
        finally:
            reduction_failed_release.set()
        failed_events = _drain_until(
            failed_executor,
            lambda values: any(event.kind in _TERMINAL for event in values),
            timeout=60.0,
        )
    failed_terminal = next(
        event for event in failed_events if event.kind in _TERMINAL
    )
    assert failed_terminal.kind is StandardEventKind.FAILED
    assert failed_terminal.primary is not None
    assert failed_terminal.primary.message == (
        "injected pre-settlement reduction failure"
    )
    assert failed_terminal.cleanup_status is CleanupStatus.CLEANED
    assert _frame_events(failed_events) == ()
    assert _fully_durable_keys(
        failed_dynamic, failed_dynamic.snapshot()
    ) == frozenset()
    assert failed_run.completed == failed_terminal.completed == 0
    assert failed_graph["transition"] is None
    assert failed_graph["session"].is_running is False
    assert _xye_labels(failed_output) == ()
    assert failed_executor.close(
        failed_identity
    ).cleanup_status is CleanupStatus.CLEANED
    assert failed_executor.close(
        failed_identity
    ).cleanup_status is CleanupStatus.CLEANED


def test_p1b_b16_duplicate_revision_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B16: unchanged watcher observations mint one attempt and receipt."""

    from xrd_tools.session.dynamic_accounting import DynamicRunAccounting
    from xrd_tools.sources.directory_session import DirectoryIndexSession

    owners: list[DynamicRunAccounting] = []
    attempts = []
    real_init = DynamicRunAccounting.__init__
    real_begin = DynamicRunAccounting.begin_attempt
    real_observe = DirectoryIndexSession.observe
    observations = []

    def observed_init(owner, *args, **kwargs):
        real_init(owner, *args, **kwargs)
        owners.append(owner)

    def observed_begin(owner, key, *, source_revision):
        token = real_begin(owner, key, source_revision=source_revision)
        attempts.append(token)
        return token

    def observed_observe(owner, *args, **kwargs):
        observation = real_observe(owner, *args, **kwargs)
        observations.append(observation)
        return observation

    monkeypatch.setattr(DynamicRunAccounting, "__init__", observed_init)
    monkeypatch.setattr(DynamicRunAccounting, "begin_attempt", observed_begin)
    monkeypatch.setattr(DirectoryIndexSession, "observe", observed_observe)
    _bridge_legacy_expected_target_state(monkeypatch)

    root = tmp_path / "raw"
    root.mkdir()
    image = root / "duplicate_0001.tif"
    _write_tiff(image, 1)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    output = tmp_path / "processed"
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        _live_intent(root, output, poni, suffixes=(".tif",)),
        request_value=2301,
    )
    events = ()
    try:
        first = _drain_until(
            executor,
            lambda values: bool(_frame_events(values)),
            timeout=60.0,
        )
        first = _wait_physical_results(
            executor,
            identity,
            first,
            expected_rows=(1,),
            expected_files=(1, 0, 0, 1),
            output_root=output,
            timeout=60.0,
        )
        # Consume the already-queued tail once before taking the immutable
        # baseline. Subsequent watcher polls must be silent for this revision.
        baseline_tail = tuple(executor.drain_events())
        baseline_events = (*first, *baseline_tail)
        events = baseline_events
        frames = _frame_events(baseline_events)
        target = Path(frames[0].artifact)
        nexus_before = _file_facts((target,))
        xye_paths = tuple(output.rglob("iq_*.xye"))
        assert len(xye_paths) == 1
        xye_before = _file_facts(xye_paths)
        catalog_before = executor.frame_catalog(identity)
        assert catalog_before is not None
        assert len(catalog_before.entries) == 1
        settled = next(
            event
            for event in reversed(baseline_events)
            if event.kind is StandardEventKind.DISCOVERY
            and (
                event.files_processed,
                event.files_skipped,
                event.files_pending,
                event.files_discovered,
            ) == (1, 0, 0, 1)
        )
        progress_before = tuple(
            getattr(settled, name)
            for name in (
                "completed", "total", "artifact_completed", "artifact_total",
                "files_processed", "files_skipped", "files_pending",
                "files_discovered",
            )
        )
        assert len(owners) == 1, (
            "B16 accounting: unchanged observations coalesced, but the run did "
            "not delegate exactly-once truth to DynamicRunAccounting"
        )
        initial_snapshot = owners[0].snapshot()
        initial_attempts = tuple(attempts)
        assert len(initial_attempts) == 1
        assert len(initial_snapshot.attempts) == 1
        assert len(_fully_durable_keys(owners[0], initial_snapshot)) == 1

        # Prove the one real watcher completed at least two later unchanged
        # observations. The second call cannot begin until the first result was
        # consumed by the serial Live loop.
        observation_floor = len(observations)
        repeated_events = []
        deadline = time.monotonic() + 10.0
        while (
            len(observations) < observation_floor + 2
            and time.monotonic() < deadline
        ):
            repeated_events.extend(executor.drain_events())
            time.sleep(0.01)
        assert len(observations) >= observation_floor + 2
        repeated_events.extend(executor.drain_events())
        repeated = tuple(repeated_events)
        assert not any(
            event.kind in {
                StandardEventKind.FRAME_READY,
                StandardEventKind.DISCOVERY,
            }
            for event in repeated
        )
        events = (*baseline_events, *repeated)
        frames = _frame_events(events)
        catalog = executor.frame_catalog(identity)
        assert len(frames) == 1
        assert catalog == catalog_before
        assert _output_rows(target) == (1,)
        assert _file_facts((target,)) == nexus_before
        assert _file_facts(tuple(output.rglob("iq_*.xye"))) == xye_before
        assert progress_before == (1, 1, 1, 1, 1, 0, 0, 1)

        snapshot = owners[0].snapshot()
        assert tuple(attempts) == initial_attempts
        assert snapshot.attempts == initial_snapshot.attempts
        assert snapshot.durable == initial_snapshot.durable
        assert _fully_durable_keys(owners[0], snapshot) == _fully_durable_keys(
            owners[0], initial_snapshot,
        )
    finally:
        _finish_live(executor, identity, prior=events)
