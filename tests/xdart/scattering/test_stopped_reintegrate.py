"""Real saved-prefix Run outputs retain exact reintegration source custody."""

import copy
import hashlib
import os
from pathlib import Path
from threading import Event
import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets
import tifffile

from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xrd_tools.io.record_writer import _decode_replacement_fact
from xrd_tools.reduction import ReintegrateSuccessorPlan, run_reintegrate_successor
from xrd_tools.reduction import reintegrate
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.image import TiffSeriesSource
from xrd_tools.sources.nexus import NexusStackSource
from xrd_tools.sources.selection import image_series_spec


def _wait(app, predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return False


@pytest.mark.parametrize("source_kind", ["stopped-tiff", "completed-tiff", "stopped-eiger"])
def test_real_run_reintegrates_only_its_saved_labels(tmp_path, monkeypatch, source_kind):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    stopped = source_kind.startswith("stopped")
    eiger = source_kind.endswith("eiger")
    entered, release = Event(), Event()
    expected_count = 104 if eiger else 64 if stopped else 3
    if eiger:
        root = Path(os.environ.get("XDART_TEST_DATA", "/missing")) / "eiger"
        raw = root / "long/eiger_S069Ta_redo_eta2p0_1_scan001_master.h5"
        member = raw.with_name(raw.name.replace("master", "data_000001"))
        poni = root / "LaB6_detxn26_detyn6p5_eta4p5.poni"
        if not all(path.is_file() for path in (raw, member, poni)):
            pytest.skip("real Eiger source unavailable")
        raw_paths = (raw, member, poni)
        total = 651
        original_chunks = NexusStackSource._iter_cursor_chunks

        def paced_chunks(source, cursor, chunk_size):
            for values, labels in original_chunks(source, cursor, chunk_size):
                boundary = next((i for i, label in enumerate(labels)
                                 if label >= expected_count), len(labels))
                if boundary:
                    yield values[:boundary], labels[:boundary]
                if boundary < len(labels) or labels[-1] == expected_count - 1:
                    entered.set()
                    assert release.wait(30.0), "Eiger source pacing not released"
                if boundary < len(labels):
                    yield values[boundary:], labels[boundary:]

        monkeypatch.setattr(NexusStackSource, "_iter_cursor_chunks", paced_chunks)
    else:
        total = 80 if stopped else 3
        for label in range(1, total + 1):
            tifffile.imwrite(tmp_path / f"raw_{label:04d}.tif",
                            np.full((195, 487), label + 1, dtype=np.uint16))
        raw = tmp_path / "raw_0001.tif"
        poni = tmp_path / "calibration.poni"
        poni.write_text(
            "poni_version: 2\nDetector: Pilatus100k\nDetector_config: {}\n"
            "Distance: 0.1234\nPoni1: 0.01\nPoni2: 0.01\n"
            "Rot1: 0.0\nRot2: 0.0\nRot3: 0.0\nWavelength: 1.0e-10\n",
            encoding="utf-8",
        )
        raw_paths = (*sorted(tmp_path.glob("raw_*.tif")), poni)
        original_read = TiffSeriesSource._read_path

        def paced_read(source, path):
            if stopped and Path(path).stem == "raw_0065":
                entered.set()
                assert release.wait(30.0), "TIFF source pacing not released"
            return original_read(source, path)

        monkeypatch.setattr(TiffSeriesSource, "_read_path", paced_read)
    raw_states = {path: (state.st_dev, state.st_ino, state.st_size,
                         state.st_mtime_ns, state.st_ctime_ns)
                  for path in raw_paths for state in (path.stat(),)}
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=image_series_spec(raw), poni_file=str(poni),
            project_root=str(tmp_path), save_path=str(tmp_path / "run.nexus"),
            output_mode="Overwrite", processing_mode="Int 2D", max_cores=4 if eiger else 1,
            bai_1d_args={"npt": 16}, bai_2d_args={"npt_rad": 16, "npt_azim": 8},
        )),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
        executor=StandardRunExecutor(join_timeout=2.0),
    )
    events = []
    original_drain = page._run_executor.drain_events

    def observe_events():
        update = original_drain()
        events.extend(update)
        return update

    monkeypatch.setattr(page._run_executor, "drain_events", observe_events)
    try:
        page._shell.run_controls.startButton.click()
        if stopped:
            assert _wait(app, entered.is_set), page._notice_text
            assert _wait(app, lambda: len(page._context_controller.frame_keys) == expected_count), (
                page._notice_text, len(page._context_controller.frame_keys))
            page._shell.run_controls.stopButton.click()
            release.set()
        assert _wait(app, lambda: any(event.kind in {
            StandardEventKind.FINISHED, StandardEventKind.STOPPED, StandardEventKind.FAILED,
        } for event in events)), page._notice_text
        terminal = next(event for event in events if event.kind in {
            StandardEventKind.FINISHED, StandardEventKind.STOPPED, StandardEventKind.FAILED,
        })
        assert terminal.kind is (StandardEventKind.STOPPED if stopped else StandardEventKind.FINISHED), terminal
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        if stopped and not eiger:
            # TIFF's parallel prefetch may have submitted a few further rows
            # before Stop. The published labels, not the pacing point, govern.
            assert 0 < terminal.completed < total
            expected_count = terminal.completed
        else:
            assert terminal.completed == expected_count
        assert terminal.total == total
        assert _wait(app, lambda: page._capture_current_loaded_browse() is not None)
        capture = page._capture_current_loaded_browse()
        assert len(capture.labels) == expected_count
        artifact = Path(capture.target)
        before = hashlib.sha256(artifact.read_bytes()).hexdigest()
        with h5py.File(artifact, "r") as document:
            cake = {key: value[()] for key, value in document["entry/integrated_2d"].items()
                    if isinstance(value, h5py.Dataset)}
            fact = _decode_replacement_fact(document, capture.labels[0])
            final = fact["append_lineage"]["epochs"][-1]["source"]
            print("SOURCE_EXTENTS", source_kind, fact["source_execution"]["frame_count"],
                  final["extent"], len(capture.labels), flush=True)
            assert fact["source_execution"]["frame_count"] == total
            assert final["extent"] == expected_count
            # Exact source identity and every selected persisted label remain mandatory.
            for mismatch in ("labels", "path"):
                broken = dict(fact)
                broken["append_lineage"] = copy.deepcopy(fact["append_lineage"])
                epoch = broken["append_lineage"]["epochs"][-1]
                if mismatch == "labels":
                    epoch["labels"] = epoch["labels"][:-1]
                else:
                    epoch["source"]["path"] += ".foreign"
                with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_LINEAGE_UNQUALIFIED"):
                    reintegrate._admit_source_topology(
                        broken, full_inventory=True, selected_labels=capture.labels,
                    )
        intent = page._intents.snapshot().thaw()
        intent.bai_1d_args["npt"] = 100
        preparation = page._reintegrate_preparation(intent, "1d")
        plan = ReintegrateSuccessorPlan.from_prepared_or_artifact(
            capture.prepared_reintegrate_offer, capture.target, entry=capture.entry,
            dimension="1d", preparation=preparation, source_root=capture.request.source_root,
            expected_target_snapshot=capture.target_snapshot, expected_labels=capture.labels,
            expected_terminal_identity=capture.request.terminal_commit_identity,
        )
        assert plan.labels == capture.labels
        result = run_reintegrate_successor(plan)
        assert result.disposition == "COMMITTED", result
        assert result.committed_labels == capture.labels
        assert result.publication_dropped_labels == ()
        with h5py.File(result.output_artifact, "r") as document:
            one = document["entry/integrated_1d"]
            assert one["intensity"].shape == (expected_count, 100)
            np.testing.assert_array_equal(one["frame_index"][()], capture.labels)
            for key, expected in cake.items():
                np.testing.assert_array_equal(document[f"entry/integrated_2d/{key}"][()], expected)
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == before
        assert {path: (state.st_dev, state.st_ino, state.st_size,
                       state.st_mtime_ns, state.st_ctime_ns)
                for path in raw_paths for state in (path.stat(),)} == raw_states
    finally:
        release.set()
        assert _wait(app, lambda: page.close_workspace().cleanup_status is CleanupStatus.CLEANED)
        page.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("saved_count", [1, 3])
def test_stopped_external_prefix_keeps_the_whole_raw_graph(tmp_path, saved_count):
    from xdart.gui.tabs.scattering.contracts import (
        ExternalSourceState, SourceExecutionStamp, SourceFileState,
    )
    from xrd_tools.io.append import (
        AppendExternalMember, AppendSource, _source_dict, truncate_append_source,
    )

    members = []
    for ordinal in range(2):
        member = tmp_path / f"member_{ordinal}.h5"
        with h5py.File(member, "w") as document:
            document.create_dataset("entry/data/data", data=np.full((2, 5, 7), ordinal, np.uint16))
        members.append(ExternalSourceState(SourceFileState.capture(member),
                       "/entry/data/data", ordinal * 2, ordinal * 2 + 2, ordinal))
    master = tmp_path / "master.h5"
    selectors = tuple(f"/entry/data/data_{ordinal + 1:06d}" for ordinal in range(2))
    with h5py.File(master, "w") as document:
        for selector, member in zip(selectors, members):
            document[selector] = h5py.ExternalLink(Path(member.file.path).name, member.dataset)
    state = SourceFileState.capture(master)
    execution = SourceExecutionStamp(state, "nexus_hdf5", 4, 0,
                                    external_members=tuple(members)).as_dict()
    complete = AppendSource(
        state.path, "nexus_hdf5", state.size, state.mtime_ns, 4,
        dataset_paths=selectors,
        external_members=tuple(AppendExternalMember(
            member.file.path, member.dataset, member.file.size, member.file.mtime_ns,
            member.first, member.stop, member.epoch,
        ) for member in members),
    )
    prefix = _source_dict(truncate_append_source(complete, saved_count))
    fact = {
        "label": saved_count - 1, "path": state.path, "frame_index": saved_count - 1,
        "source_base": "", "snapshot": {
            "adapter_id": "nexus_hdf5", "size": state.size, "mtime_ns": state.mtime_ns,
            "frame_count": 4, "dataset_path": selectors[0], "self_contained": False,
        },
        "source_execution": execution,
        "append_lineage": {"epochs": [{"source": prefix, "labels": list(range(saved_count))}]},
        "metadata": {}, "geometry": {}, "background_dependency": None,
    }
    topology = reintegrate._admit_source_topology(
        fact, full_inventory=True, selected_labels=tuple(range(saved_count)),
    )
    assert tuple(topology.frame_routes) == tuple(range(saved_count))
    assert topology.external_paths == selectors
    np.testing.assert_array_equal(reintegrate._source_fact(fact, read=True, topology=topology)[3],
                                  np.full((5, 7), int(saved_count > 2), np.uint16))
    for field, bad_value in (("dataset_path", "/entry/foreign"), ("size", 0)):
        broken = copy.deepcopy(fact)
        broken["append_lineage"]["epochs"][0]["source"]["external_members"][-1][field] = bad_value
        with pytest.raises(ValueError, match="REPLACEMENT_"):
            reintegrate._admit_source_topology(
                broken, full_inventory=True, selected_labels=tuple(range(saved_count)),
            )
    # An unsaved trailing member remains part of the immutable raw-source graph.
    broken = copy.deepcopy(fact)
    broken["source_execution"]["external_members"] = broken["source_execution"]["external_members"][:1]
    with pytest.raises(ValueError, match="REPLACEMENT_HDF5_DEPENDENCY_TOPOLOGY_UNSUPPORTED"):
        reintegrate._admit_source_topology(
            broken, full_inventory=True, selected_labels=tuple(range(saved_count)),
        )
