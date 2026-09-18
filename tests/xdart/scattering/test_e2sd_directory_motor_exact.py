"""Production-shaped probes for the E2-SD Directory and motor seam."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
import os
import subprocess
import sys
import time

import fabio
import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtWidgets

from tests.xdart.scattering._e2sd_support import (
    admit_with_session,
    directory_start,
    external_eiger_capture,
    write_poni,
    write_motor_container,
)
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import (
    SourceCapture,
    SourceFileState,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
    StartCapture,
)
from xdart.gui.tabs.scattering.controls_projection import GI_MOTOR, PROJECT_ROOT
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import RequestId
from xdart.gui.tabs.scattering.output_preflight import (
    SourceRevisionChanged,
    materialize_deferred_output,
    prepare_output,
    validate_admitted_receipt,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_widgets import source_header_projection
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.core.scan import SourceKind
from xrd_tools.io import metadata as io_metadata
from xrd_tools.io.metadata import ImageMetadataRead
from xrd_tools.io.output_safety import OutputCollisionError
from xrd_tools.sources import execution_graph as source_graph
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.directory_index import DirectoryIndex
from xrd_tools.sources.selection import DirectorySourceSpec, image_series_spec


def _write_tiff(path: Path, value: int) -> None:
    fabio.tifimage.TifImage(
        data=np.full((4, 4), value, dtype=np.uint16)
    ).write(str(path))


def _write_edf(path: Path, value: int) -> None:
    fabio.edfimage.EdfImage(
        data=np.full((4, 4), value, dtype=np.uint16)
    ).write(str(path))


def _write_sidecar(path: Path, lines: tuple[str, ...]) -> None:
    path.with_suffix(".txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _tiff_directory_start(
    tmp_path: Path,
    *,
    metadata_format: str | None = "auto",
    incidence_motor: str = "th",
    recursive: bool = False,
    save_path: Path | None = None,
) -> StartCapture:
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=recursive,
        suffixes=(".tif",),
        metadata_format=metadata_format,
    )
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(
            tmp_path / "processed" if save_path is None else save_path
        ),
        output_mode="Overwrite",
        gi=GIIntent(enabled=True, incidence_motor=incidence_motor),
    )).snapshot()
    request = RequestId(700)
    return StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source),
    )


def _directory_start(
    tmp_path: Path, *, gi: bool, output_mode: str,
) -> StartCapture:
    start = _tiff_directory_start(tmp_path)
    intent = start.intent_snapshot.thaw()
    intent.gi.enabled = gi
    intent.output_mode = output_mode
    return StartCapture(
        start.request_id, start.capture_sequence,
        RunIntentStore(intent).snapshot(), start.source_capture,
    )


def _wait(qapp: QtWidgets.QApplication, predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not settle")


def _shell(workspace: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = workspace.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_numbered_directory_preview_uses_one_canonical_candidate_order(
    tmp_path: Path,
) -> None:
    for name in ("scan_2.nxs", "scan_10.nxs"):
        write_motor_container(tmp_path / name)
    source = DirectorySourceSpec(tmp_path, suffixes=(".nxs",))
    request = SourceObservationRequest(1, 0, source)
    adapter = FilesystemSourceAdapter()

    passive = adapter.observe(request)
    preview = adapter.preview_motors(request)

    assert preview.candidate_fingerprint == passive.candidate_fingerprint
    assert preview.gi_motor_choices == ("halpha",)


def _recursive_container_start(
    tmp_path: Path,
    *,
    save_path: Path | None = None,
) -> tuple[StartCapture, tuple[Path, Path], Path]:
    raw = tmp_path / "raw"
    first = raw / "data" / "scan_0001.nxs"
    second = raw / "live_test" / "scan_0001.nxs"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    write_motor_container(first)
    write_motor_container(second)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    output = save_path or (tmp_path / "processed")
    source = DirectorySourceSpec(
        raw,
        recursive=True,
        suffixes=(".nxs",),
    )
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(output),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(719)
    return (
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        (first, second),
        output,
    )


def test_large_standard_directory_defers_descriptors_and_frame_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(40):
        (raw / f"scan_{index:04d}.nxs").write_bytes(b"not opened")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(718)
    start = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source),
    )

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError(
            "Standard admission must not probe descriptors or frame counts"
        )

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", unexpected_probe)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        start,
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    try:
        assert receipt.outputs == ()
        assert receipt.deferred_directory is not None
        assert receipt.deferred_directory.discovered_file_count == 40
        assert len(receipt.deferred_directory.entries) == 40
    finally:
        sessions[0].close()


def test_valid_container_directory_probes_only_current_candidate_just_in_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(3):
        write_motor_container(raw / f"scan_{index:04d}.nxs")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(731)
    calls: list[Path] = []
    original_probe = DirectoryIndex.probe_candidate

    def counted_probe(index, candidate, **kwargs):
        calls.append(candidate.path)
        return original_probe(index, candidate, **kwargs)

    monkeypatch.setattr(DirectoryIndex, "probe_candidate", counted_probe)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        assert len(deferred.entries) == 3
        assert calls == []

        decision, ready, skipped = materialize_deferred_output(
            receipt,
            session,
            deferred.entries[0],
            cancelled=lambda: False,
        )
        assert decision is not None
        assert (ready, skipped) == (1, 0)
        assert calls == [deferred.entries[0].candidates[0].path]
    finally:
        session.close()


def test_deferred_unreadable_container_revalidates_before_skip(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    landing = raw / "still_landing.nxs"
    landing.write_bytes(b"incomplete")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(740)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        entry = deferred.entries[0]
        # Admission is name/stat-only: the container is not opened until its
        # own JIT materialization, which skips it while it is still landing.
        assert entry.skip_reason == ""
        validate_admitted_receipt(receipt, session)
        assert materialize_deferred_output(
            receipt,
            session,
            entry,
            cancelled=lambda: False,
        ) == (None, 0, 1)

        landing.write_bytes(b"new revision")

        with pytest.raises(
            SourceRevisionChanged,
            match="HDF5 dependency changed during admission: ",
        ):
            materialize_deferred_output(
                receipt,
                session,
                entry,
                cancelled=lambda: False,
            )
    finally:
        session.close()


def test_standard_directory_dependency_inventory_does_not_walk_whole_hdf5_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "scan_0001.nxs")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()

    def unexpected_walk(*_args, **_kwargs):
        raise AssertionError("admission must not recursively walk HDF5 links")

    monkeypatch.setattr(
        h5py.Group,
        "visititems_links",
        unexpected_walk,
        raising=False,
    )
    request = RequestId(719)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    try:
        assert receipt.deferred_directory is not None
    finally:
        sessions[0].close()


def test_decisive_detector_ignores_unrelated_dangling_external_link(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    selected = raw / "scan_0001.nxs"
    with h5py.File(selected, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_group("data").create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
        entry["notes"] = h5py.ExternalLink(
            "unrelated-missing.h5",
            "/notes",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(732)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        assert tuple(
            Path(value.path) for value in deferred.entries[0].protected_states
        ) == (selected,)
    finally:
        sessions[0].close()


def test_standard_directory_fails_closed_when_hdf5_dependency_proof_cannot_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    selected = raw / "scan_0001.nxs"
    write_motor_container(selected)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    real_file = h5py.File

    def refused_file(path, *args, **kwargs):
        if Path(path) == selected:
            raise OSError("locked during dependency proof")
        return real_file(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", refused_file)
    request = RequestId(723)
    sessions: list[DirectoryIndexSession] = []
    # Admission never opens the container, so a locked file cannot refuse
    # Start; the JIT probe that owns the dependency proof skips it instead of
    # executing without one.
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        validate_admitted_receipt(receipt, session)
        assert materialize_deferred_output(
            receipt,
            session,
            deferred.entries[0],
            cancelled=lambda: False,
        ) == (None, 0, 1)
    finally:
        session.close()


@pytest.mark.parametrize("count", (1, 32, 33))
def test_standard_directory_lazy_policy_is_not_size_dependent(
    tmp_path: Path,
    count: int,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(count):
        (raw / f"scan_{index:04d}.nxs").write_bytes(b"not hdf5")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(722 + count)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    try:
        assert receipt.outputs == ()
        assert receipt.deferred_directory is not None
        assert len(receipt.deferred_directory.entries) == count
    finally:
        sessions[0].close()


def _collision_target(tmp_path: Path) -> Path:
    """Pre-create scan_0001's own output so a raw link to it can be followed.

    Admission is name/stat-only, so the raw-input collision can only surface
    once the linking container's dependency proof is materialized just in
    time; that proof dereferences the link, which needs a real target.
    """

    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    target = processed / "scan_0001_int2d.nexus"
    with h5py.File(target, "w") as handle:
        handle.create_group("entry").create_group("data").create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
    return target


def _expect_jit_collision(
    tmp_path: Path,
    raw: Path,
    *,
    request: RequestId,
) -> None:
    processed = tmp_path / "processed"
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(processed),
        output_mode="Overwrite",
    )).snapshot()
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        validate_admitted_receipt(receipt, session)
        entries = {
            entry.candidates[0].path.name: entry for entry in deferred.entries
        }
        decision, ready, skipped = materialize_deferred_output(
            receipt,
            session,
            entries["scan_0001.nxs"],
            cancelled=lambda: False,
        )
        assert decision is not None
        assert (ready, skipped) == (1, 0)
        with pytest.raises(
            OutputCollisionError,
            match="same file as a raw directory input",
        ):
            materialize_deferred_output(
                receipt,
                session,
                entries["scan_0002.nxs"],
                cancelled=lambda: False,
            )
    finally:
        session.close()


def test_deferred_directory_rejects_generic_external_member_target(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "scan_0001.nxs")
    target = _collision_target(tmp_path)
    with h5py.File(raw / "scan_0002.nxs", "w") as handle:
        data = handle.create_group("entry").create_group("data")
        data["pixels"] = h5py.ExternalLink(str(target), "/entry/data/data")
    _expect_jit_collision(tmp_path, raw, request=RequestId(720))


def test_deferred_directory_soft_chain_rejects_external_target(
    tmp_path: Path,
) -> None:
    """A soft chain onto an external link still binds the linked file."""

    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "scan_0001.nxs")
    target = _collision_target(tmp_path)
    with h5py.File(raw / "scan_0002.nxs", "w") as handle:
        entry = handle.create_group("entry")
        data = entry.create_group("data")
        data["data"] = h5py.SoftLink("/hidden/raw")
        hidden = handle.create_group("hidden")
        hidden["raw"] = h5py.ExternalLink(str(target), "/entry/data/data")
    _expect_jit_collision(tmp_path, raw, request=RequestId(725))


def _write_member_linking_container(raw: Path, member_name: str) -> None:
    """scan_0002.nxs whose data lives in a sibling member file (still landing)."""

    with h5py.File(raw / "scan_0002.nxs", "w") as handle:
        data = handle.create_group("entry").create_group("data")
        data["pixels"] = h5py.ExternalLink(member_name, "/entry/data/data")


def test_deferred_directory_rejects_hard_link_to_admitted_target(
    tmp_path: Path,
) -> None:
    """A hard link carries no target spelling; only stat identity sees it."""

    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "scan_0001.nxs")
    target = _collision_target(tmp_path)
    os.link(target, raw / "scan_0002_data.h5")
    _write_member_linking_container(raw, "scan_0002_data.h5")
    _expect_jit_collision(tmp_path, raw, request=RequestId(741))


def test_deferred_directory_rejects_hard_link_to_target_written_after_admission(
    tmp_path: Path,
) -> None:
    """The residual the admission-time identity set cannot see.

    scan_0001's output does not exist at admission, so no identity is
    frozen for it; the run writes it, and only then does a raw member land
    as a hard link to that output.  The lexical and realpath keys cannot
    see a hard link and the frozen identity set never held one, so the
    member's link count is the only remaining signal.
    """

    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "scan_0001.nxs")
    _write_member_linking_container(raw, "scan_0002_data.h5")
    processed = tmp_path / "processed"
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(processed),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(742)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        assert not any(
            type(entry.fact.target_state) is tuple for entry in deferred.entries
        )
        validate_admitted_receipt(receipt, session)
        entries = {
            entry.candidates[0].path.name: entry for entry in deferred.entries
        }
        decision, ready, skipped = materialize_deferred_output(
            receipt,
            session,
            entries["scan_0001.nxs"],
            cancelled=lambda: False,
        )
        assert decision is not None
        assert (ready, skipped) == (1, 0)
        # The run has written scan_0001's output; the member now lands as a
        # hard link to it.
        target = _collision_target(tmp_path)
        assert target == entries["scan_0001.nxs"].target
        os.link(target, raw / "scan_0002_data.h5")
        with pytest.raises(
            OutputCollisionError,
            match="same file as a raw directory input",
        ):
            materialize_deferred_output(
                receipt,
                session,
                entries["scan_0002.nxs"],
                cancelled=lambda: False,
            )
    finally:
        session.close()


def test_owns_target_scans_current_identities_only_for_multiply_linked_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The residual is closed without a once-per-run inventory.

    A single-linked dependency (every ordinary raw file) is settled by its
    own stat; only a dependency with more than one link pays a stat per
    target, and a hard link to an output written after admission is then
    found through the targets' current identities.
    """

    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "scan_0001.nxs")
    write_motor_container(raw / "scan_0002.nxs")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(743)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    sessions[0].close()
    plan = receipt.deferred_directory
    assert plan is not None
    targets = set(plan.targets)
    assert len(targets) == 2
    real_stat = os.stat
    stats: list[Path] = []

    def counting_stat(path, *args, **kwargs):
        stats.append(Path(path))
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", counting_stat)

    single = raw / "scan_0001.nxs"
    assert plan.owns_target(single) is False
    assert stats == [single]

    # Multiply linked, but to another raw file: the targets are consulted
    # and none matches.
    twin = raw / "scan_0001_twin.nxs"
    os.link(single, twin)
    stats.clear()
    assert plan.owns_target(twin) is False
    assert stats[0] == twin and set(stats[1:]) == targets

    # Multiply linked to an output the run wrote after admission.
    written = next(iter(targets))
    written.parent.mkdir(exist_ok=True)
    written.write_bytes(b"processed")
    landed = raw / "scan_0003_data.h5"
    os.link(written, landed)
    stats.clear()
    assert plan.owns_target(landed) is True
    assert stats[0] == landed and set(stats[1:]) <= targets


def test_indirect_instrument_precedence_forces_complete_dependency_proof(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "scan_0001.nxs")
    target = _collision_target(tmp_path)
    landing = tmp_path / "landing"
    landing.mkdir()
    external_group = landing / "detector.h5"
    with h5py.File(external_group, "w") as handle:
        detector = handle.create_group("detector")
        detector["data"] = h5py.ExternalLink(str(target), "/entry/data/data")
    with h5py.File(raw / "scan_0002.nxs", "w") as handle:
        instrument = handle.create_group("entry").create_group("instrument")
        instrument["aaa"] = h5py.ExternalLink(
            str(external_group),
            "/detector",
        )
        instrument.create_group("zzz").create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
    _expect_jit_collision(tmp_path, raw, request=RequestId(726))


def test_external_entry_uses_landed_file_as_relative_link_base(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    landing = raw / "landing"
    raw.mkdir()
    landing.mkdir()
    master = raw / "scan_master.h5"
    entry_file = landing / "entry.h5"
    segment = landing / "segment.nexus"
    with h5py.File(segment, "w") as handle:
        handle.create_group("entry").create_group("data").create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
    with h5py.File(entry_file, "w") as handle:
        data = handle.create_group("entry").create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            segment.name,
            "/entry/data/data",
        )
    with h5py.File(master, "w") as handle:
        handle["entry"] = h5py.ExternalLink(
            "landing/entry.h5",
            "/entry",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".h5",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(727)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        # Name/stat admission freezes only the landed master; the linked
        # entry file and segment are bound by the JIT dependency proof below.
        assert tuple(
            Path(value.path) for value in deferred.entries[0].protected_states
        ) == (master,)
        validate_admitted_receipt(receipt, session)
        decision, ready, skipped = materialize_deferred_output(
            receipt,
            session,
            deferred.entries[0],
            cancelled=lambda: False,
        )
        assert decision is not None
        assert (ready, skipped) == (1, 0)
        assert decision.item.source_stamp.frame_count == 2
        assert tuple(
            Path(value.file.path)
            for value in decision.item.source_stamp.external_members
        ) == (segment,)
        assert tuple(
            Path(value.path)
            for value in decision.item.source_stamp.dependency_files
        ) == (entry_file,)
    finally:
        session.close()


def _restore_mtime(path: Path, before: os.stat_result) -> None:
    """Put mtime back after a rewrite so only ctime records the change."""

    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (
        before.st_size,
        before.st_mtime_ns,
    )
    assert after.st_ctime_ns != before.st_ctime_ns


def _same_stat_byte_rewrite(path: Path, offset: int) -> None:
    """Rewrite one raw contiguous-dataset byte through a plain descriptor.

    The JIT dependency proof holds the file open read-only through HDF5, so
    a second in-process ``h5py.File`` open is refused; an external writer
    does not go through this process's HDF5 open table at all.
    """

    before = path.stat()
    fd = os.open(path, os.O_WRONLY)
    try:
        assert os.pwrite(fd, np.uint16(7).tobytes(), offset) == 2
    finally:
        os.close(fd)
    _restore_mtime(path, before)


def _same_stat_rewrite_from_other_process(path: Path, dataset: str) -> None:
    """Rewrite one dataset element from a separate writer process.

    Chunked datasets have no fixed raw offset, so the rewrite goes through
    HDF5 in another process (without the file lock), like a detector writer.
    """

    before = path.stat()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, h5py\n"
            "with h5py.File(sys.argv[1], 'r+', locking=False) as handle:\n"
            "    handle[sys.argv[2]][0, 0, 0] = 7\n",
            str(path),
            dataset,
        ],
        env=dict(os.environ, HDF5_USE_FILE_LOCKING="FALSE"),
        check=True,
    )
    _restore_mtime(path, before)


def _race_dependency_after_first_capture(
    monkeypatch: pytest.MonkeyPatch,
    dependency: Path,
    rewrite: Callable[[], None],
) -> Callable[[], bool]:
    """Run ``rewrite`` right after the JIT proof first captures ``dependency``."""

    original_capture = source_graph._capture_source_topology
    raced = False

    def racing_capture(path, **kwargs):
        nonlocal raced
        result = original_capture(path, **kwargs)
        if not raced and Path(path).resolve() == dependency.resolve():
            raced = True
            rewrite()
        return result

    monkeypatch.setattr(
        source_graph,
        "_capture_source_topology",
        racing_capture,
    )
    return lambda: raced


def test_selected_external_entry_is_frozen_before_dereference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    landing = tmp_path / "landing"
    raw.mkdir()
    landing.mkdir()
    master = raw / "scan_master.h5"
    entry_file = landing / "entry.h5"
    with h5py.File(entry_file, "w") as handle:
        pixels = handle.create_group("entry").create_group(
            "data"
        ).create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
        offset = int(pixels.id.get_offset())
    with h5py.File(master, "w") as handle:
        handle["entry"] = h5py.ExternalLink(
            str(entry_file),
            "/entry",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".h5",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(735)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    raced = _race_dependency_after_first_capture(
        monkeypatch,
        entry_file,
        lambda: _same_stat_byte_rewrite(entry_file, offset),
    )
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        validate_admitted_receipt(receipt, session)
        with pytest.raises(
            SourceRevisionChanged,
            match="HDF5 dependency changed during admission",
        ):
            materialize_deferred_output(
                receipt,
                session,
                deferred.entries[0],
                cancelled=lambda: False,
            )
        assert raced()
        with h5py.File(entry_file, "r") as handle:
            assert handle["entry/data/data"][0, 0, 0] == 7
    finally:
        session.close()


def test_apstools_soft_external_nxdata_binds_only_selected_dependency(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    landing = tmp_path / "landing"
    raw.mkdir()
    landing.mkdir()
    master = raw / "scan_0001.nxs"
    sidecar = landing / "pixels.h5"
    missing = landing / "unrelated-missing.h5"
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
        entry["unrelated"] = h5py.ExternalLink(
            str(missing),
            "/ignored",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(736)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        protected = {
            Path(value.path)
            for value in deferred.entries[0].protected_states
        }
        # Name/stat admission freezes the master only; the selected sidecar
        # is bound by the JIT proof and the unrelated dangling link never is.
        assert protected == {master}
        assert missing not in protected

        decision, ready, skipped = materialize_deferred_output(
            receipt,
            session,
            deferred.entries[0],
            cancelled=lambda: False,
        )
        assert decision is not None
        assert (ready, skipped) == (1, 0)
        stamp = decision.item.source_stamp
        assert stamp.frame_count == 3
        # The detector dataset physically lives in the sidecar, so the proof
        # binds it as an external member under its owner-file selector
        # (tests/core: test_selected_link_owner_selector_detaches_sidecar_
        # internal_path), not as a bare dependency file.
        assert tuple(
            (Path(member.file.path), member.dataset, member.first, member.stop)
            for member in stamp.external_members
        ) == ((sidecar, "/entry/data/pixels", 0, 3),)
        assert stamp.dependency_files == ()
    finally:
        session.close()


def test_apstools_external_detector_revision_is_frozen_before_dereference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    landing = tmp_path / "landing"
    raw.mkdir()
    landing.mkdir()
    master = raw / "scan_0001.nxs"
    sidecar = landing / "pixels.h5"
    with h5py.File(sidecar, "w") as handle:
        pixels = handle.create_group("entry").create_group(
            "data"
        ).create_dataset(
            "pixels",
            data=np.ones((3, 4, 5), dtype=np.uint16),
        )
        pixels.attrs["signal_type"] = "detector"
        offset = int(pixels.id.get_offset())
    with h5py.File(master, "w") as handle:
        handle.attrs["creator"] = "NXWriter"
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset("end_time", data="complete")
        entry.create_group("instrument").create_group("bluesky")
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data["pixels"] = h5py.ExternalLink(
            str(sidecar),
            "/entry/data/pixels",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(737)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    raced = _race_dependency_after_first_capture(
        monkeypatch,
        sidecar,
        lambda: _same_stat_byte_rewrite(sidecar, offset),
    )
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        validate_admitted_receipt(receipt, session)
        with pytest.raises(
            SourceRevisionChanged,
            match="HDF5 dependency changed during admission",
        ):
            materialize_deferred_output(
                receipt,
                session,
                deferred.entries[0],
                cancelled=lambda: False,
            )
        assert raced()
        with h5py.File(sidecar, "r") as handle:
            assert handle["entry/data/pixels"][0, 0, 0] == 7
    finally:
        session.close()


def test_eager_gi_reprobes_cached_self_contained_descriptor_after_rebind(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dataclasses import replace
    from xdart.gui.tabs.scattering import output_preflight

    raw = tmp_path / "raw"
    raw.mkdir()
    selected = raw / "scan_0001.nxs"
    replacement = tmp_path / "replacement.nxs"
    write_motor_container(selected, "oldmotor")
    write_motor_container(replacement, "newmotor")
    before = selected.stat()
    replacement_state = replacement.stat()
    assert replacement_state.st_size == before.st_size
    os.utime(
        replacement,
        ns=(replacement_state.st_atime_ns, before.st_mtime_ns),
    )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
        gi=GIIntent(enabled=True, incidence_motor="Manual"),
    )).snapshot()
    request = RequestId(737)
    original_from_observation = (
        output_preflight.RunCandidatePlan.from_observation
    )
    rebound = False

    def racing_from_observation(cls, observation):
        nonlocal rebound
        plan = original_from_observation(observation)
        descriptor = plan.descriptor_for(plan.candidates[0])
        assert descriptor is not None
        assert descriptor.motor_names == ("oldmotor",)
        if not rebound:
            rebound = True
            os.replace(replacement, selected)
            after = selected.stat()
            assert (after.st_size, after.st_mtime_ns) == (
                before.st_size,
                before.st_mtime_ns,
            )
            assert after.st_ino != before.st_ino
        return plan

    monkeypatch.setattr(
        output_preflight.RunCandidatePlan,
        "from_observation",
        classmethod(racing_from_observation),
    )
    real_get_adapter = output_preflight.get_adapter
    adapter = output_preflight.candidate_owner(selected)
    assert adapter is not None
    fresh_probes: list[Path] = []

    def fresh_probe(path: Path):
        fresh_probes.append(Path(path))
        return adapter.probe(path)

    wrapped = replace(adapter, probe=fresh_probe)
    monkeypatch.setattr(
        output_preflight,
        "get_adapter",
        lambda adapter_id: (
            wrapped
            if adapter_id == wrapped.id
            else real_get_adapter(adapter_id)
        ),
    )
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    try:
        assert rebound
        assert fresh_probes == [selected]
        assert receipt.gi_motor_choices == ("newmotor",)
        assert len(receipt.outputs) == 1
        item = receipt.outputs[0].item
        assert item.descriptor is not None
        assert item.descriptor.motor_names == ("newmotor",)
        after = selected.stat()
        assert item.source_stamp.file.inode == after.st_ino
        assert item.source_stamp.file.inode != before.st_ino
    finally:
        sessions[0].close()


@pytest.mark.parametrize("layout", ("direct", "soft", "local"))
def test_soft_detector_path_materializes_with_exact_external_dependency(
    tmp_path: Path,
    layout: str,
) -> None:
    raw = tmp_path / "raw"
    landing = tmp_path / "landing"
    raw.mkdir()
    landing.mkdir()
    master = raw / "scan_0001.nxs"
    sidecar = landing / "pixels.nexus"
    with h5py.File(sidecar, "w") as handle:
        handle.create_group("detector").create_dataset(
            "pixels",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry").create_group("data")
        if layout == "soft":
            data["data"] = h5py.SoftLink("/hidden/raw")
            handle.create_group("hidden")["raw"] = h5py.ExternalLink(
                str(sidecar), "/detector/pixels",
            )
        elif layout == "local":
            data.create_dataset("data", data=np.ones((2, 4, 5), dtype=np.uint16))
        else:
            data["data"] = h5py.ExternalLink(str(sidecar), "/detector/pixels")
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(728)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        validate_admitted_receipt(receipt, session)
        decision, ready, skipped = materialize_deferred_output(
            receipt,
            session,
            deferred.entries[0],
            cancelled=lambda: False,
        )
        assert decision is not None
        assert (ready, skipped) == (1, 0)
        stamp = decision.item.source_stamp
        assert stamp.frame_count == 2
        if layout == "local":
            assert stamp.external_members == ()
        else:
            member, = stamp.external_members
            assert Path(member.file.path) == sidecar
            assert member.dataset == "/detector/pixels"
            assert (member.first, member.stop, member.epoch) == (0, 2, 0)
        assert stamp.dependency_files == ()
    finally:
        session.close()


@pytest.mark.parametrize("layout", ("ancestor", "mixed", "soft", "terminal"))
def test_external_member_capture_uses_dataset_owner_and_all_segment_offsets(
    tmp_path: Path, layout: str,
) -> None:
    """Capture contract only: these descriptors do not test adapter selection."""
    from xrd_tools.sources.execution_graph import _external_members
    from xrd_tools.sources.descriptor import ContainerDescriptor

    master, sidecar = tmp_path / "master.h5", tmp_path / "pixels.h5"
    pixels = np.ones((2, 4, 5), dtype=np.uint16)
    with h5py.File(sidecar, "w") as handle:
        handle.create_dataset("detector/pixels", data=pixels)
    with h5py.File(master, "w") as handle:
        if layout == "ancestor":
            handle["landed"] = h5py.ExternalLink(str(sidecar), "/detector")
            paths = ("/landed/pixels",)
        elif layout == "soft":
            handle["bridge"] = h5py.ExternalLink(str(sidecar), "/detector")
            handle["entry/data/data_000001"] = h5py.SoftLink(
                "/bridge/pixels",
            )
            paths = ("/entry/data/data_000001",)
        elif layout == "terminal":
            middle = tmp_path / "middle.h5"
            with h5py.File(middle, "w") as middle_handle:
                middle_handle["alias"] = h5py.ExternalLink(
                    sidecar.name, "/detector/pixels",
                )
            handle["entry/data/data_000001"] = h5py.ExternalLink(
                middle.name, "/alias",
            )
            paths = ("/entry/data/data_000001",)
        else:
            paths = tuple(f"/segment_{index}" for index in range(4))
            for index, path in enumerate(paths):
                if index % 2:
                    handle[path] = h5py.ExternalLink(str(sidecar), "/detector/pixels")
                else:
                    handle.create_dataset(path, data=pixels)
    descriptor = ContainerDescriptor(
        master, dataset_path=paths[0], segment_paths=paths,
        frame_count=2 * len(paths), frame_shape=(4, 5), dtype=pixels.dtype,
        # The descriptor's leaf-link heuristic misses an external ancestor.
        self_contained=layout == "ancestor",
    )
    members = _external_members(master, SourceFileState.capture(master), descriptor)
    assert all(Path(member.file.path) == sidecar for member in members)
    assert all(member.dataset == "/detector/pixels" for member in members)
    assert [(member.first, member.stop, member.epoch) for member in members] == (
        [(0, 2, 0)] if layout in {"ancestor", "soft", "terminal"} else [(2, 4, 1), (6, 8, 3)]
    )


def test_nested_vds_dependency_collision_is_rejected_before_execution(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_motor_container(raw / "scan_0001.nxs")
    target = _collision_target(tmp_path)
    landing = tmp_path / "landing"
    landing.mkdir()
    middle = landing / "middle.h5"
    shape = (2, 4, 5)
    with h5py.File(middle, "w", libver="latest") as handle:
        layout = h5py.VirtualLayout(shape=shape, dtype=np.uint16)
        layout[:] = h5py.VirtualSource(
            str(target),
            "/entry/data/data",
            shape=shape,
        )
        handle.create_virtual_dataset("vdata", layout)
    with h5py.File(raw / "scan_0002.nxs", "w", libver="latest") as handle:
        layout = h5py.VirtualLayout(shape=shape, dtype=np.uint16)
        layout[:] = h5py.VirtualSource(
            str(middle),
            "/vdata",
            shape=shape,
        )
        handle.create_group("entry").create_group("data").create_virtual_dataset(
            "data",
            layout,
        )
    _expect_jit_collision(tmp_path, raw, request=RequestId(729))


def test_selected_vds_refuses_existing_file_with_missing_dataset(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    middle = tmp_path / "middle.h5"
    with h5py.File(middle, "w"):
        pass
    shape = (2, 4, 5)
    with h5py.File(raw / "scan_0001.nxs", "w", libver="latest") as handle:
        layout = h5py.VirtualLayout(shape=shape, dtype=np.uint16)
        layout[:] = h5py.VirtualSource(
            str(middle),
            "/missing",
            shape=shape,
        )
        handle.create_group("entry").create_group("data").create_virtual_dataset(
            "data",
            layout,
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".nxs",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(730)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        validate_admitted_receipt(receipt, session)
        # The VDS source file exists, so this is a terminal dependency
        # failure at the JIT proof rather than a still-landing skip.
        with pytest.raises(
            ValueError,
            match="selected HDF5 dependency path is unavailable",
        ):
            materialize_deferred_output(
                receipt,
                session,
                deferred.entries[0],
                cancelled=lambda: False,
            )
    finally:
        session.close()


def test_dependency_graph_state_cannot_rebaseline_after_jit_freeze(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import output_preflight

    raw = tmp_path / "raw"
    raw.mkdir()
    landing = tmp_path / "landing"
    landing.mkdir()
    master = raw / "scan_master.h5"
    sidecar = landing / "sidecar.h5"
    first = landing / "first.h5"
    second = landing / "second.h5"
    shape = (2, 4, 5)
    for path, value in ((first, 1), (second, 2)):
        with h5py.File(path, "w") as handle:
            handle.create_dataset(
                "pixels",
                data=np.full(shape, value, dtype=np.uint16),
            )

    def write_sidecar(source: Path) -> None:
        with h5py.File(sidecar, "w", libver="latest") as handle:
            layout = h5py.VirtualLayout(shape=shape, dtype=np.uint16)
            layout[:] = h5py.VirtualSource(
                str(source),
                "/pixels",
                shape=shape,
            )
            handle.create_virtual_dataset("pixels", layout)

    write_sidecar(first)
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry").create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            str(sidecar),
            "/pixels",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".h5",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(733)
    original_items = output_preflight._directory_items
    mutated = False

    def racing_items(configuration, plan, *, cancelled):
        nonlocal mutated
        result = original_items(configuration, plan, cancelled=cancelled)
        graph = result[0].graph
        assert str(sidecar.resolve()) in {
            target.resolved_path for target in graph.stamp.canonical_targets
        }
        if not mutated:
            mutated = True
            write_sidecar(second)
        return result

    monkeypatch.setattr(
        output_preflight,
        "_directory_items",
        racing_items,
    )
    sessions: list[DirectoryIndexSession] = []
    try:
        receipt = prepare_output(
            StartCapture(request, 1, snapshot, SourceCapture(request, 1, source)),
            cancelled=lambda: False,
            session_owner=sessions.append,
        )
        with pytest.raises(SourceRevisionChanged, match="source target changed after admission"):
            materialize_deferred_output(
                receipt, sessions[0], receipt.deferred_directory.entries[0],
                cancelled=lambda: False,
            )
        assert mutated is True
    finally:
        for session in sessions:
            session.close()


def test_image_series_refuses_external_frame_growth_after_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "scan_data_000001.h5"
    master = tmp_path / "scan_master.h5"
    with h5py.File(sidecar, "w") as handle:
        handle.create_group("entry").create_group("data").create_dataset(
            "data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
            maxshape=(None, 4, 5),
            chunks=True,
        )
    with h5py.File(master, "w") as handle:
        handle.create_group("entry").create_group("data")[
            "data_000001"
        ] = h5py.ExternalLink(sidecar.name, "/entry/data/data")
    source = image_series_spec(master)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(734)
    original_describe = source_graph.describe_container_from_open
    mutated = False
    # Qualification derives the descriptor from its own open master handle
    # and then re-counts the external members through that same handle, so
    # HDF5 already holds the sidecar open read-only here.  Grow it the way a
    # detector writer would: from another process, without the file lock.
    grow = (
        "import sys, h5py\n"
        "with h5py.File(sys.argv[1], 'r+', locking=False) as handle:\n"
        "    dataset = handle['entry/data/data']\n"
        "    dataset.resize((3, 4, 5))\n"
        "    dataset[2] = 3\n"
    )

    def racing_describe(handle, **kwargs):
        nonlocal mutated
        result = original_describe(handle, **kwargs)
        if not mutated and Path(kwargs.get("path", "")) == master:
            mutated = True
            subprocess.run(
                [sys.executable, "-c", grow, str(sidecar)],
                env=dict(os.environ, HDF5_USE_FILE_LOCKING="FALSE"),
                check=True,
            )
        return result

    monkeypatch.setattr(
        source_graph,
        "describe_container_from_open",
        racing_describe,
    )

    with pytest.raises(
        ValueError,
        match="external detector frame count changed during admission",
    ):
        prepare_output(
            StartCapture(
                request,
                1,
                snapshot,
                SourceCapture(request, 1, source),
            ),
            cancelled=lambda: False,
            session_owner=lambda _session: None,
        )
    assert mutated


def test_deferred_standard_refuses_missing_eiger_dependency(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    master = raw / "scan_master.h5"
    sidecar = raw / "scan_data_000001.h5"
    with h5py.File(master, "w") as handle:
        data = handle.create_group("entry").create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            sidecar.name,
            "/entry/data/data",
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(raw, suffixes=(".h5",))
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(721)
    sessions: list[DirectoryIndexSession] = []
    # A still-landing dependency no longer refuses Start: admission is
    # name/stat-only, and the JIT proof skips the master until its sidecar
    # exists instead of executing against an incomplete series.
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        validate_admitted_receipt(receipt, session)
        assert materialize_deferred_output(
            receipt,
            session,
            deferred.entries[0],
            cancelled=lambda: False,
        ) == (None, 0, 1)
    finally:
        session.close()


def test_deferred_raw_series_uses_known_shape_and_counts_ready_members(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in (1, 2):
        np.full((195, 487), index, dtype=np.int32).tofile(
            raw / f"scan_{index:04d}.raw"
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = DirectorySourceSpec(
        raw,
        suffixes=(".raw",),
        metadata_format=None,
    )
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(726)
    sessions: list[DirectoryIndexSession] = []
    receipt = prepare_output(
        StartCapture(
            request,
            1,
            snapshot,
            SourceCapture(request, 1, source),
        ),
        cancelled=lambda: False,
        session_owner=sessions.append,
    )
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        assert len(deferred.entries) == 1
        decision, ready, skipped = materialize_deferred_output(
            receipt,
            sessions[0],
            deferred.entries[0],
            cancelled=lambda: False,
        )

        assert decision is not None
        assert decision.item.source_spec.kind is SourceKind.TIFF_SERIES
        assert decision.item.source_stamp.frame_count == 2
        assert ready == 2
        assert skipped == 0
    finally:
        sessions[0].close()


def test_recursive_same_named_containers_preserve_relative_output_directories(
    tmp_path: Path,
) -> None:
    start, (first, second), output = _recursive_container_start(tmp_path)
    sessions: list[DirectoryIndexSession] = []
    reservations: list[tuple[Path, ...]] = []

    receipt = prepare_output(
        start,
        cancelled=lambda: False,
        session_owner=sessions.append,
        targets_owner=reservations.append,
    )
    assert len(sessions) == 1
    session = sessions[0]
    try:
        deferred = receipt.deferred_directory
        assert deferred is not None
        targets = {
            entry.candidates[0].path: entry.target
            for entry in deferred.entries
        }
        assert targets == {
            # Same stem, different relative directories -- the point of this
            # test. The stable slot rides along; it does not disambiguate them.
            first: output / "data" / "scan_0001_int2d.nexus",
            second: output / "live_test" / "scan_0001_int2d.nexus",
        }
        assert len(reservations) == 1
        assert set(reservations[0]) == set(targets.values())
        assert all(not target.exists() for target in targets.values())
        validate_admitted_receipt(receipt, session)
    finally:
        session.close()


def test_recursive_explicit_target_refuses_before_reservation(
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "processed.nxs"
    start, sources, _output = _recursive_container_start(
        tmp_path,
        save_path=explicit,
    )
    sessions: list[DirectoryIndexSession] = []
    reservations: list[tuple[Path, ...]] = []
    source_bytes = tuple(path.read_bytes() for path in sources)
    try:
        with pytest.raises(ValueError, match="duplicate output target"):
            prepare_output(
                start,
                cancelled=lambda: False,
                session_owner=sessions.append,
                targets_owner=reservations.append,
            )
        assert reservations == []
        assert not explicit.exists()
        assert tuple(path.read_bytes() for path in sources) == source_bytes
    finally:
        for session in sessions:
            session.close()


def test_recursive_output_parent_symlink_cannot_escape_selected_root(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed"
    outside = tmp_path / "outside"
    output.mkdir()
    outside.mkdir()
    (output / "data").symlink_to(outside, target_is_directory=True)
    start, sources, _output = _recursive_container_start(
        tmp_path,
        save_path=output,
    )
    sessions: list[DirectoryIndexSession] = []
    reservations: list[tuple[Path, ...]] = []
    source_bytes = tuple(path.read_bytes() for path in sources)
    try:
        with pytest.raises(
            ValueError,
            match="directory output escaped selected root",
        ):
            prepare_output(
                start,
                cancelled=lambda: False,
                session_owner=sessions.append,
                targets_owner=reservations.append,
            )
        assert reservations == []
        assert tuple(outside.iterdir()) == ()
        assert tuple(path.read_bytes() for path in sources) == source_bytes
    finally:
        for session in sessions:
            session.close()


def test_tiff_metadata_motor_names_are_finite_filtered_and_disabled_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata

    image = tmp_path / "scan_0001.tif"
    calls: list[tuple[Path, str | None]] = []

    def metadata(path: Path, metadata_format: str | None):
        calls.append((Path(path), metadata_format))
        return {
            "th": 0.15,
            "eta": np.float64(0.25),
            "ROI1": 9.0,
            "sample_pd": 8.0,
            "not_numeric": "missing",
            "vector": np.array([1.0]),
            "not_a_motor": True,
            "not_finite": np.nan,
            "also_not_finite": np.inf,
        }

    monkeypatch.setattr(source_metadata, "read_image_metadata", metadata)

    assert source_metadata.image_metadata_motor_names(image, "auto") == (
        "th",
        "eta",
    )
    assert calls == [(image, "auto")]

    calls.clear()
    assert source_metadata.image_metadata_motor_names(image, None) == ()
    assert calls == []


def test_disabled_tiff_metadata_stays_known_empty_without_reading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata

    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("metadata-off must not invoke the sidecar reader")

    monkeypatch.setattr(
        source_metadata,
        "read_image_metadata",
        unexpected_read,
    )
    source = DirectorySourceSpec(
        tmp_path,
        suffixes=(".tif",),
        metadata_format=None,
    )
    preview = FilesystemSourceAdapter().preview_motors(
        SourceObservationRequest(702, 0, source)
    )

    assert preview.gi_motor_choices == ()

    receipt, session = admit_with_session(_tiff_directory_start(
        tmp_path,
        metadata_format=None,
        incidence_motor="Manual",
    ))
    try:
        assert receipt.gi_motor_choices == ()
        assert tuple(
            output.item.group.motor_names for output in receipt.outputs
        ) == ((),)
        assert all(
            output.item.source_stamp.admitted_motor_values == ()
            and output.item.source_spec.options["admitted_motor_values"] == ()
            for output in receipt.outputs
        )
    finally:
        session.close()


def test_tiff_admission_cancels_after_first_metadata_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import output_preflight
    from xrd_tools.io import metadata as metadata_io

    images = tuple(tmp_path / f"scan_{index:04d}.tif" for index in range(3))
    for index, image in enumerate(images):
        _write_tiff(image, index)
    cancelled = Event()
    calls: list[Path] = []

    def cancel_first(
        path: Path,
        _metadata_format: str | None,
        **_kwargs,
    ) -> ImageMetadataRead:
        calls.append(Path(path))
        cancelled.set()
        return ImageMetadataRead({}, None)

    monkeypatch.setattr(
        metadata_io,
        "read_image_metadata_observed",
        cancel_first,
    )
    sessions: list[DirectoryIndexSession] = []
    try:
        with pytest.raises(RuntimeError, match="admission cancelled"):
            prepare_output(
                _tiff_directory_start(
                    tmp_path,
                    incidence_motor="Manual",
                ),
                cancelled=cancelled.is_set,
                session_owner=sessions.append,
            )
        assert len(calls) == 1
    finally:
        for session in sessions:
            session.close()


def test_tiff_admission_binds_values_to_stable_second_sidecar_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import output_preflight
    from xrd_tools.io import metadata as metadata_io

    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    _write_sidecar(image, ("th=0.1", "exposure=1", "sequence=1"))
    real_read = metadata_io.read_image_metadata_observed
    calls = 0

    def replace_after_discovery(*args, **kwargs):
        nonlocal calls
        observed = real_read(*args, **kwargs)
        calls += 1
        if calls == 1:
            _write_sidecar(
                image,
                ("th=0.2", "exposure=1", "sequence=1"),
            )
        return observed

    monkeypatch.setattr(
        metadata_io,
        "read_image_metadata_observed",
        replace_after_discovery,
    )
    sessions: list[DirectoryIndexSession] = []
    try:
        receipt = prepare_output(
            _tiff_directory_start(tmp_path, incidence_motor="th"),
            cancelled=lambda: False,
            session_owner=sessions.append,
        )
        values = receipt.outputs[0].item.source_stamp.admitted_motor_values
        assert calls == 2
        assert tuple(value.value for value in values) == (0.2,)
    finally:
        for session in sessions:
            session.close()


def test_tiff_admission_rejects_sidecar_mutation_during_guarded_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import output_preflight
    from xrd_tools.io import metadata as metadata_io

    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    _write_sidecar(image, ("th=0.1", "exposure=1", "sequence=1"))
    real_read = metadata_io.read_image_metadata_observed
    calls = 0

    def replace_during_guarded_read(*args, **kwargs):
        nonlocal calls
        observed = real_read(*args, **kwargs)
        calls += 1
        if calls == 2:
            _write_sidecar(
                image,
                ("th=0.2", "exposure=1", "sequence=1"),
            )
        return observed

    monkeypatch.setattr(
        metadata_io,
        "read_image_metadata_observed",
        replace_during_guarded_read,
    )
    sessions: list[DirectoryIndexSession] = []
    try:
        with pytest.raises(
            ValueError,
            match="metadata source changed during admission",
        ):
            prepare_output(
                _tiff_directory_start(tmp_path, incidence_motor="th"),
                cancelled=lambda: False,
                session_owner=sessions.append,
            )
        assert calls == 2
    finally:
        for session in sessions:
            session.close()


@pytest.mark.parametrize("source_mode", ("series", "directory"))
def test_tiff_admission_cancels_after_first_member_state_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_mode: str,
) -> None:
    from xdart.gui.tabs.scattering import output_preflight

    raw = tmp_path / "raw"
    raw.mkdir()
    images = tuple(raw / f"scan_{index:04d}.tif" for index in range(3))
    for index, image in enumerate(images):
        _write_tiff(image, index)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = (
        image_series_spec(images[0], metadata_format=None)
        if source_mode == "series"
        else DirectorySourceSpec(
            raw,
            suffixes=(".tif",),
            metadata_format=None,
        )
    )
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(715)
    start = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source),
    )
    cancelled = Event()
    calls: list[Path] = []
    original = SourceFileState.capture

    def cancel_first(path: Path) -> SourceFileState:
        result = original(path)
        if Path(path).suffix.lower() in {".tif", ".tiff"}:
            calls.append(Path(path))
            if len(calls) == 1:
                cancelled.set()
        return result

    monkeypatch.setattr(
        output_preflight.SourceFileState,
        "capture",
        staticmethod(cancel_first),
    )
    sessions: list[DirectoryIndexSession] = []
    try:
        with pytest.raises(RuntimeError, match="admission cancelled"):
            prepare_output(
                start,
                cancelled=cancelled.is_set,
                session_owner=sessions.append,
            )
        assert calls == [images[0]]
    finally:
        for session in sessions:
            session.close()


def test_tiff_revalidation_cancels_after_first_member_state_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import output_preflight

    raw = tmp_path / "raw"
    raw.mkdir()
    images = tuple(raw / f"scan_{index:04d}.tif" for index in range(3))
    for index, image in enumerate(images):
        _write_tiff(image, index)
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    source = image_series_spec(images[0], metadata_format=None)
    snapshot = RunIntentStore(RunIntent(
        source_spec=source,
        poni_file=str(poni),
        save_path=str(tmp_path / "processed.nxs"),
        output_mode="Overwrite",
    )).snapshot()
    request = RequestId(718)
    start = StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source),
    )
    receipt = prepare_output(
        start,
        cancelled=lambda: False,
        session_owner=lambda _session: None,
    )
    # P4/OUT-1: a suffix-shaped save target keeps the operator's directory and
    # stem, and the run slot + ``.nexus`` are the shared naming owner's.
    assert receipt.outputs[0].item.target == tmp_path / "processed_int2d.nexus"
    cancelled = Event()
    calls: list[Path] = []
    original = SourceFileState.capture

    def cancel_first(path: Path) -> SourceFileState:
        result = original(path)
        if Path(path).suffix.lower() in {".tif", ".tiff"}:
            calls.append(Path(path))
            if len(calls) == 1:
                cancelled.set()
        return result

    monkeypatch.setattr(
        output_preflight.SourceFileState,
        "capture",
        staticmethod(cancel_first),
    )

    with pytest.raises(RuntimeError, match="admission cancelled"):
        validate_admitted_receipt(
            receipt,
            None,
            cancelled=cancelled.is_set,
        )
    assert calls == [images[0]]


def test_mixed_tiff_metadata_rejects_missing_selected_gi_motor_but_manual_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata
    from xrd_tools.io import metadata as metadata_io

    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    _write_sidecar(first, ("th=0.15", "exposure=1"))
    _write_sidecar(second, ("exposure=2",))

    def metadata(path: Path, _metadata_format: str | None):
        return (
            {"th": 0.15, "exposure": 1.0}
            if Path(path) == first
            else {"exposure": 2.0}
        )

    monkeypatch.setattr(source_metadata, "read_image_metadata", metadata)
    monkeypatch.setattr(
        metadata_io,
        "read_image_metadata_observed",
        lambda path, metadata_format, **_kwargs: ImageMetadataRead(
            metadata(Path(path), metadata_format),
            Path(path).with_suffix(".txt"),
        ),
    )

    sessions: list[object] = []
    try:
        with pytest.raises(
            ValueError,
            match=(
                "GI metadata motor 'th'.*finite value in every admitted TIFF"
            ),
        ):
            prepare_output(
                _tiff_directory_start(tmp_path),
                cancelled=lambda: False,
                session_owner=sessions.append,
            )
    finally:
        for session in sessions:
            session.close()

    receipt, session = admit_with_session(_tiff_directory_start(
        tmp_path,
        incidence_motor="Manual",
    ))
    try:
        assert receipt.gi_motor_choices == ("exposure",)
        gi = receipt.candidate.processing_mapping()["gi"]
        assert gi["incidence_motor"] == "Manual"
        assert gi["resolved_motor"] == "Manual"
        assert all(
            output.item.source_stamp.admitted_motor_values == ()
            and output.item.source_spec.options["admitted_motor_values"] == ()
            for output in receipt.outputs
        )
    finally:
        session.close()


def test_tiff_preview_and_admission_share_ordered_member_intersection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata
    from xrd_tools.io import metadata as metadata_io

    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    _write_sidecar(first, (
        "th=0.15",
        "exposure=1.0",
        "ROI1=99",
        "sample_pd=88",
        "nan_value=nan",
        "label=not-a-number",
    ))
    _write_sidecar(second, (
        "exposure=2.0",
        "th=0.25",
        "second_only=3.0",
    ))
    source = DirectorySourceSpec(
        tmp_path,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    calls: list[Path] = []
    original = source_metadata.read_image_metadata
    original_observed = metadata_io.read_image_metadata_observed

    def counted(path: Path, metadata_format: str | None):
        calls.append(Path(path))
        return original(path, metadata_format)

    def counted_observed(
        path: Path,
        metadata_format: str | None = None,
        **kwargs,
    ) -> ImageMetadataRead:
        calls.append(Path(path))
        return original_observed(
            path, kwargs.pop("meta_format", metadata_format), **kwargs,
        )

    monkeypatch.setattr(source_metadata, "read_image_metadata", counted)
    preview = FilesystemSourceAdapter().preview_motors(
        SourceObservationRequest(701, 0, source)
    )

    assert preview.gi_motor_choices == ("th", "exposure")
    assert calls == [first, second]

    calls.clear()
    monkeypatch.setattr(
        metadata_io,
        "read_image_metadata_observed",
        counted_observed,
    )
    receipt, session = admit_with_session(_tiff_directory_start(tmp_path))
    try:
        assert receipt.gi_motor_choices == preview.gi_motor_choices
        assert tuple(
            output.item.group.motor_names for output in receipt.outputs
        ) == (("th", "exposure"),)
        # Admission guards each discovered sidecar with a second read bound
        # between exact before/after file states.
        assert calls == [first, first, second, second]
    finally:
        session.close()


def test_recursive_tiff_preview_reads_every_nested_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata

    nested = tmp_path / "nested"
    nested.mkdir()
    direct = tmp_path / "scan_0001.tif"
    child = nested / "scan_0002.tif"
    _write_tiff(direct, 1)
    _write_tiff(child, 2)
    _write_sidecar(direct, ("th=0.15", "exposure=1", "direct_only=3"))
    _write_sidecar(child, ("exposure=2", "th=0.25", "nested_only=4"))
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    calls: list[Path] = []
    original = source_metadata.read_image_metadata

    def counted(path: Path, metadata_format: str | None):
        calls.append(Path(path))
        return original(path, metadata_format)

    monkeypatch.setattr(source_metadata, "read_image_metadata", counted)
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(703, 0, source)
    passive = adapter.observe(request)
    preview = adapter.preview_motors(request)

    assert passive.subdirectories_deferred is True
    assert preview.candidate_fingerprint == passive.candidate_fingerprint
    assert preview.direct_child_count == 1
    assert preview.subdirectories_deferred is True
    header = source_header_projection(preview)
    assert header.text == (
        "2 files (folder + 1 level) · Image Directory"
    )
    assert (
        "Deeper subfolders are outside the supported Run scope."
        in header.detail
    )
    # Recursive discovery owns one deterministic natural path order.  The
    # nested path sorts first here, so its metadata order owns the result.
    assert calls == [child, direct]
    assert preview.gi_motor_choices == ("exposure", "th")


def test_recursive_tiff_preview_bootstraps_image_owner_in_fresh_process(
    tmp_path: Path,
) -> None:
    code = r"""
from pathlib import Path
import sys

import fabio
import numpy as np

root = Path(sys.argv[1])
image = root / "scan_0001.tif"
fabio.tifimage.TifImage(
    data=np.ones((4, 4), dtype=np.uint16)
).write(str(image))

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceObservationRequest
from xrd_tools.sources.selection import DirectorySourceSpec

source = DirectorySourceSpec(
    root,
    recursive=True,
    suffixes=(".tif",),
    metadata_format=None,
)
preview = FilesystemSourceAdapter().preview_motors(
    SourceObservationRequest(1, 0, source)
)
assert preview.gi_motor_choices == (), preview

# The builtin adapters bootstrap lazily on first registry use (so the
# registry is never observably empty); in this fresh process nothing GUI-side
# registered anything, and the image owner still resolves.
from xrd_tools.sources.adapters import candidate_owner
owner = candidate_owner(image)
assert owner is not None and owner.id == "image_file", owner
"""

    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_recursive_tiff_preview_is_shallow_but_run_rejects_deeper_gi_gap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata

    immediate = tmp_path / "immediate"
    deeper = immediate / "deeper"
    immediate.mkdir()
    deeper.mkdir()
    direct = tmp_path / "scan_0001.tif"
    child = immediate / "scan_0002.tif"
    grandchild = deeper / "scan_0003.tif"
    for index, path in enumerate((direct, child, grandchild), start=1):
        _write_tiff(path, index)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    calls: list[Path] = []

    def counted(path: Path, _metadata_format: str | None):
        candidate = Path(path)
        calls.append(candidate)
        return (
            {"exposure": 3.0}
            if candidate == grandchild
            else {"th": 0.15, "exposure": 1.0}
        )

    def observed(path, metadata_format, **_kwargs):
        return ImageMetadataRead(counted(Path(path), metadata_format), None)

    monkeypatch.setattr(source_metadata, "read_image_metadata", counted)
    monkeypatch.setattr(source_metadata, "read_image_metadata_observed", observed)
    # Run admission reads sidecars through the io owner, not the GUI preview
    # module, so the same counted reader must answer there as well.
    monkeypatch.setattr(io_metadata, "read_image_metadata_observed", observed)
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(706, 0, source)

    preview = adapter.preview_motors(request)

    assert preview.observed_file_count == 2
    assert preview.gi_motor_choices == ("th", "exposure")
    assert set(calls) == {direct, child}
    assert grandchild not in calls

    calls.clear()
    sessions: list[object] = []
    try:
        receipt = prepare_output(
            _tiff_directory_start(
                tmp_path,
                recursive=True,
                save_path=tmp_path.with_name(f"{tmp_path.name}-processed"),
            ),
            cancelled=lambda: False,
            session_owner=sessions.append,
        )
        assert len(receipt.outputs) == 2
        member_path_groups = {
            tuple(
                Path(member.path).resolve()
                for member in output.item.source_stamp.members
            )
            for output in receipt.outputs
        }
        assert member_path_groups == {
            (direct.resolve(),),
            (child.resolve(),),
        }
        frame_counts = sorted(
            output.item.source_stamp.frame_count
            for output in receipt.outputs
        )
        assert frame_counts == [1, 1]
        assert sum(frame_counts) == 2
        assert set(calls) == {direct, child}
        assert grandchild not in calls
    finally:
        for session in sessions:
            session.close()


def test_recursive_tiff_preview_keeps_the_32_candidate_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata

    nested = tmp_path / "nested"
    nested.mkdir()
    for index in range(FilesystemSourceAdapter._MOTOR_PREVIEW_LIMIT + 1):
        (nested / f"scan_{index:04d}.tif").write_bytes(b"not probed")

    def unexpected_read(*_args, **_kwargs):
        raise AssertionError("over-limit preview must not read sidecars")

    monkeypatch.setattr(
        source_metadata,
        "read_image_metadata",
        unexpected_read,
    )
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(704, 0, source)
    passive = adapter.observe(request)

    preview = adapter.preview_motors(request)

    assert preview == passive
    header = source_header_projection(preview)
    assert header.text == (
        "33 files (folder + 1 level) · Image Directory"
    )
    assert header.ready
    assert "content is qualified just in time" in header.detail
    assert (
        "Deeper subfolders are outside the supported Run scope."
        in header.detail
    )


def test_recursive_over_limit_preview_never_infers_from_direct_tiffs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata

    nested = tmp_path / "nested"
    nested.mkdir()
    direct = tuple(
        tmp_path / f"scan_{index:04d}.tif"
        for index in range(1, 3)
    )
    for index, path in enumerate(direct, start=1):
        _write_tiff(path, index)
        _write_sidecar(path, (
            f"th={index / 10}",
            f"exposure={index}",
            f"direct_{index}={index}",
        ))
    for index in range(FilesystemSourceAdapter._MOTOR_PREVIEW_LIMIT - 1):
        (nested / f"nested_{index:04d}.tif").write_bytes(b"not probed")

    calls: list[Path] = []

    def counted(path: Path, _metadata_format: str | None):
        calls.append(Path(path))
        return {"th": 0.1}

    monkeypatch.setattr(source_metadata, "read_image_metadata", counted)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(705, 0, source)
    passive = adapter.observe(request)
    preview = adapter.preview_motors(request)

    assert passive.direct_child_count == 2
    assert passive.subdirectories_deferred is True
    assert preview.source == passive.source
    assert preview.candidate_fingerprint == passive.candidate_fingerprint
    assert preview.direct_child_count == passive.direct_child_count
    assert preview.subdirectories_deferred is True
    assert preview.gi_motor_choices is None
    assert calls == []
    assert adapter.project_motor_knowledge(
        source,
        passive.candidate_fingerprint,
    ) is None
    header = source_header_projection(preview)
    assert header.text == (
        "33 files (folder + 1 level) · Image Directory"
    )
    assert header.ready
    assert "content is qualified just in time" in header.detail
    assert (
        "Deeper subfolders are outside the supported Run scope."
        in header.detail
    )


def test_nonempty_recursive_tiff_marker_defers_corrupt_content_to_run(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "corrupt.tif").write_bytes(b"not a TIFF")
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(707, 0, source)

    passive = adapter.observe(request)
    preview = adapter.preview_motors(request)

    assert passive.observed_file_count == 1
    assert preview == passive
    assert preview.gi_motor_choices is None
    header = source_header_projection(preview)
    assert header.ready
    assert "content is qualified just in time" in header.detail


def test_recursive_tiff_preview_filter_matches_exact_suffix_stripping(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_tiff(nested / "scan_0001.tif", 1)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        # Exact DirectoryIndexSession filters the suffix-stripped base, so
        # this term must not match solely because the file ends in '.tif'.
        name_filter="tif",
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(708, 0, source)

    passive = adapter.observe(request)
    preview = adapter.preview_motors(request)
    session = DirectoryIndexSession(probe_candidates=False)
    try:
        session.configure(
            source.root,
            recursive=True,
            suffixes=source.suffixes,
            name_filter=source.name_filter,
        )
        exact = session.observe(refresh=True)
    finally:
        session.close()

    assert passive.direct_child_count == 0
    assert passive.observed_file_count == 0
    assert exact.discovered_snapshot.candidates == ()
    assert preview == passive
    assert preview.gi_motor_choices is None


def test_recursive_tiff_malformed_filter_is_contained_as_unavailable(
    tmp_path: Path,
) -> None:
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        name_filter="scan |",
    )

    observed = FilesystemSourceAdapter().observe(
        SourceObservationRequest(713, 0, source)
    )

    assert observed.status is SourceObservationStatus.UNAVAILABLE
    assert observed.reason == "Directory metadata is unavailable."


def test_recursive_tiff_preview_ignores_immediate_directory_symlinks(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_tiff(outside / "escaped.tif", 1)
    (selected / "alias").symlink_to(outside, target_is_directory=True)
    source = DirectorySourceSpec(
        selected,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(709, 0, source)

    passive = adapter.observe(request)
    preview = adapter.preview_motors(request)

    assert passive.direct_child_count == 0
    assert passive.observed_file_count == 0
    assert preview == passive
    assert preview.gi_motor_choices is None


def test_recursive_nested_member_change_invalidates_motor_knowledge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering import source_metadata

    nested = tmp_path / "nested"
    nested.mkdir()
    image = nested / "scan_0001.tif"
    _write_tiff(image, 1)
    monkeypatch.setattr(
        source_metadata,
        "read_image_metadata",
        lambda *_args, **_kwargs: {"th": 0.15},
    )
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(710, 0, source)

    first = adapter.preview_motors(request)
    assert first.gi_motor_choices == ("th",)
    assert adapter.project_motor_knowledge(
        source, first.candidate_fingerprint
    ) == first

    _write_tiff(image, 123)
    second = adapter.observe(SourceObservationRequest(711, 0, source))

    assert second.candidate_fingerprint != first.candidate_fingerprint
    assert adapter.project_motor_knowledge(
        source, second.candidate_fingerprint
    ) is None


def test_recursive_tiff_preview_honors_cancellation_during_sidecar_catalog(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering.adapters import source as source_adapter

    nested = tmp_path / "nested"
    nested.mkdir()
    direct = tmp_path / "scan_0001.tif"
    child = nested / "scan_0002.tif"
    _write_tiff(direct, 1)
    _write_tiff(child, 2)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(705, 0, source)
    calls: list[Path] = []

    def cancel_after_first(path: Path, _metadata_format: str | None):
        calls.append(Path(path))
        adapter.cancel_observation(request.observation_id)
        return ("th",)

    monkeypatch.setattr(
        source_adapter,
        "image_metadata_motor_names",
        cancel_after_first,
    )

    preview = adapter.preview_motors(request)

    assert preview.status is SourceObservationStatus.UNAVAILABLE
    assert preview.reason == "Motor preview cancelled."
    assert len(calls) == 1


def test_recursive_tiff_preview_cancels_during_passive_shallow_walk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(714, 0, source)
    calls: list[Path] = []
    original = SourceFileState.capture

    def cancel_first(path: Path):
        calls.append(Path(path))
        result = original(path)
        if len(calls) == 1:
            adapter.cancel_observation(request.observation_id)
        return result

    monkeypatch.setattr(
        SourceFileState,
        "capture",
        staticmethod(cancel_first),
    )

    preview = adapter.preview_motors(request)

    assert preview.status is SourceObservationStatus.UNAVAILABLE
    assert preview.reason == "Motor preview cancelled."
    assert len(calls) == 1


def test_directory_observation_cancels_during_root_listing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for index in range(3):
        _write_tiff(tmp_path / f"scan_{index:04d}.tif", index)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(717, 0, source)
    original = Path.iterdir
    yielded: list[Path] = []

    def cancellable_iterdir(path: Path):
        values = original(path)
        if path != tmp_path:
            return values

        def cancel_on_first_member():
            for value in values:
                yielded.append(value)
                if len(yielded) == 1:
                    adapter.cancel_observation(request.observation_id)
                yield value

        return cancel_on_first_member()

    monkeypatch.setattr(Path, "iterdir", cancellable_iterdir)

    observation = adapter.observe(request)

    # Cancellation is checked after the first member, whichever the
    # filesystem lists first (ext4 readdir is hash-ordered, not sorted).
    assert len(yielded) == 1
    assert yielded[0].parent == tmp_path
    assert yielded[0].name in {f"scan_{index:04d}.tif" for index in range(3)}
    assert observation.status is SourceObservationStatus.UNAVAILABLE
    assert observation.reason == "Observation cancelled."


def test_recursive_tiff_preview_cancels_during_second_shallow_walk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    for index in range(3):
        _write_tiff(nested / f"scan_{index:04d}.tif", index)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(716, 0, source)
    original = Path.iterdir
    nested_walks = 0

    def cancellable_iterdir(path: Path):
        nonlocal nested_walks
        values = original(path)
        if path != nested:
            return values
        nested_walks += 1
        if nested_walks != 2:
            return values

        def cancel_before_second_member():
            iterator = iter(values)
            first = next(iterator, None)
            if first is not None:
                yield first
            adapter.cancel_observation(request.observation_id)
            yield from iterator

        return cancel_before_second_member()

    monkeypatch.setattr(Path, "iterdir", cancellable_iterdir)

    preview = adapter.preview_motors(request)

    assert nested_walks == 2
    assert preview.status is SourceObservationStatus.UNAVAILABLE
    assert preview.reason == "Motor preview cancelled."


def test_recursive_single_tiff_cancellation_after_final_metadata_is_not_published(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering.adapters import source as source_adapter

    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(712, 0, source)

    def cancel_on_only_catalog(*_args, **_kwargs):
        adapter.cancel_observation(request.observation_id)
        return ("th",)

    monkeypatch.setattr(
        source_adapter,
        "image_metadata_motor_names",
        cancel_on_only_catalog,
    )

    preview = adapter.preview_motors(request)

    assert preview.status is SourceObservationStatus.UNAVAILABLE
    assert preview.reason == "Motor preview cancelled."
    assert adapter.project_motor_knowledge(source) is None


def test_recursive_tiff_cancellation_at_publish_is_not_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    _write_sidecar(image, ("th=0.15",))
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif",),
        metadata_format="auto",
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(715, 0, source)
    entered = Event()
    release = Event()
    original = adapter.publish_motor_knowledge
    result: list[SourceObservation] = []

    def blocked_publish(observation: SourceObservation) -> None:
        entered.set()
        assert release.wait(timeout=3.0)
        original(observation)

    monkeypatch.setattr(adapter, "publish_motor_knowledge", blocked_publish)
    worker = Thread(
        target=lambda: result.append(adapter.preview_motors(request)),
        daemon=True,
    )
    worker.start()
    assert entered.wait(timeout=3.0)
    adapter.cancel_observation(request.observation_id)
    release.set()
    worker.join(timeout=3.0)

    assert not worker.is_alive()
    assert len(result) == 1
    assert result[0].status is SourceObservationStatus.UNAVAILABLE
    assert result[0].reason == "Motor preview cancelled."
    assert adapter.project_motor_knowledge(source) is None


def test_tiff_sidecar_motor_mutation_invalidates_admission(
    tmp_path: Path,
) -> None:
    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    _write_sidecar(first, ("th=0.15", "exposure=1", "sequence=1"))
    _write_sidecar(second, ("th=0.25", "exposure=2", "sequence=2"))
    receipt, session = admit_with_session(_tiff_directory_start(tmp_path))
    try:
        assert receipt.gi_motor_choices == ("th", "exposure", "sequence")
        _write_sidecar(second, ("eta=0.25", "exposure=2", "sequence=2"))

        with pytest.raises(
            ValueError,
            match="authoritative motor knowledge changed after admission",
        ):
            validate_admitted_receipt(receipt, session)
    finally:
        session.close()


def test_tiff_sidecar_finite_motor_rewrite_invalidates_admission(
    tmp_path: Path,
) -> None:
    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    _write_sidecar(first, ("th=0.15", "exposure=1", "sequence=1"))
    _write_sidecar(second, ("th=0.25", "exposure=2", "sequence=2"))
    receipt, session = admit_with_session(_tiff_directory_start(tmp_path))
    try:
        second.with_suffix(".txt").write_text(
            "th=7.25\nexposure=2\nsequence=2\n",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="authoritative motor knowledge changed after admission",
        ):
            validate_admitted_receipt(receipt, session)
    finally:
        session.close()


def test_exact_tiff_suffix_excludes_edf_mask_candidate(tmp_path: Path) -> None:
    image = tmp_path / "scan_0001.tif"
    mask = tmp_path / "scan-mask.edf"
    _write_tiff(image, 1)
    _write_edf(mask, 0)
    _write_sidecar(image, ("th=0.15", "exposure=1", "sequence=1"))

    receipt, session = admit_with_session(_tiff_directory_start(tmp_path))
    try:
        assert len(receipt.outputs) == 1
        item = receipt.outputs[0].item
        # The run publishes its stable slot, not the bare scan name, and this
        # directory start is grazing incidence, so the family carries the marker.
        assert item.target.name == "scan_gi_int2d.nexus"
        assert item.artifact_family == "scan_gi"
        assert tuple(Path(value.path) for value in item.source_stamp.members) == (
            image,
        )
        assert all("mask" not in output.item.target.name for output in receipt.outputs)
    finally:
        session.close()


class _DelayedDirectorySource:
    def __init__(self) -> None:
        self.passive_started = Event()
        self.passive_release = Event()
        self.preview_requests: list[SourceObservationRequest] = []
        self.knowledge: SourceObservation | None = None

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.passive_started.set()
        assert self.passive_release.wait(3.0)
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "raw",
            True,
            True,
            direct_child_count=1,
            candidate_fingerprint="qualified-source-fingerprint",
        )

    def preview_motors(
        self, request: SourceObservationRequest
    ) -> SourceObservation:
        self.preview_requests.append(request)
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "raw",
            True,
            True,
            direct_child_count=1,
            gi_motor_choices=("halpha",),
            candidate_fingerprint="qualified-source-fingerprint",
        )

    def cancel_observation(self, _observation_id: int) -> None:
        return

    def publish_motor_knowledge(
        self, observation: SourceObservation
    ) -> None:
        self.knowledge = observation

    def project_motor_knowledge(
        self, source, candidate_fingerprint=None
    ):
        knowledge = self.knowledge
        if knowledge is None or knowledge.source != source:
            return None
        if (
            candidate_fingerprint is not None
            and knowledge.candidate_fingerprint != candidate_fingerprint
        ):
            return None
        return knowledge

    def capture(self, *_args):
        raise AssertionError("source capture is outside this preview probe")

    def cancel(self, *_args) -> None:
        return


def test_unrelated_edit_during_preview_eventually_populates_matching_motor(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    source_spec = DirectorySourceSpec(tmp_path, suffixes=(".nxs",))
    run_store = RunIntentStore(RunIntent(
        source_spec=source_spec,
        gi=GIIntent(enabled=True),
    ))
    source = _DelayedDirectorySource()
    workspace = ScatteringWorkspace(
        intents=run_store,
        lifecycle=ScatteringCoordinator(),
        sources=source,
    )
    try:
        controls = _shell(workspace).controls
        assert source.passive_started.wait(1.0)
        controls.fieldValueChanged.emit(PROJECT_ROOT, "/unrelated")
        assert run_store.snapshot().revision == 1
        source.passive_release.set()
        _wait(
            qapp,
            lambda: not workspace._source_selection.observing
            and bool(source.preview_requests),
        )
        projection = controls.projection
        assert projection is not None
        field = next(
            value
            for value in projection.fields
            if value.path == GI_MOTOR
        )
        assert field.choices == ("Manual", "halpha")
    finally:
        source.passive_release.set()
        workspace.close_workspace()


def test_same_stat_motor_rewrite_invalidates_admission(tmp_path: Path) -> None:
    _store, start = directory_start(tmp_path)
    receipt, session = admit_with_session(start)
    path = Path(start.source_capture.source.root) / "scan_0.nxs"
    before = path.stat()
    try:
        with h5py.File(path, "r+") as handle:
            handle.move(
                "entry/instrument/positioners/halpha",
                "entry/instrument/positioners/zzzzzz",
            )
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert (after.st_size, after.st_mtime_ns) == (
            before.st_size,
            before.st_mtime_ns,
        )
        assert after.st_ctime_ns != before.st_ctime_ns

        with pytest.raises(
            ValueError,
            match="authoritative motor knowledge changed after admission",
        ):
            validate_admitted_receipt(receipt, session)
    finally:
        session.close()


def test_same_stat_eiger_member_rewrite_before_start_binds_current_revision(
    tmp_path: Path,
) -> None:
    """A member rewritten before Start is bound as-is: there is no stale proof.

    Directory admission never opens the Eiger sidecar, so a same-stat rewrite
    between admission and Start cannot invalidate anything; the JIT proof
    captures the member when the entry is materialized, and the stamp carries
    that revision (the post-rewrite ctime), not the admission-time one.
    """

    _store, _capture, start, sidecar = external_eiger_capture(tmp_path)
    intent = start.intent_snapshot.thaw()
    intent.output_mode = "Overwrite"
    run_store = RunIntentStore(intent)
    snapshot = run_store.snapshot()
    request = RequestId(2)
    source = snapshot.thaw().source_spec
    start = StartCapture(
        request, 1, snapshot, SourceCapture(request, 1, source)
    )
    receipt, session = admit_with_session(start)
    before = sidecar.stat()
    try:
        with h5py.File(sidecar, "r+") as handle:
            handle["entry/data/data"][0, 0, 0] = 9
        _restore_mtime(sidecar, before)
        after = sidecar.stat()

        validate_admitted_receipt(receipt, session)
        deferred = receipt.deferred_directory
        assert deferred is not None
        decision, ready, skipped = materialize_deferred_output(
            receipt,
            session,
            deferred.entries[0],
            cancelled=lambda: False,
        )
        assert decision is not None
        assert (ready, skipped) == (1, 0)
        member, = decision.item.source_stamp.external_members
        assert Path(member.file.path) == sidecar
        assert member.file.ctime_ns == after.st_ctime_ns
        assert member.file.ctime_ns != before.st_ctime_ns
    finally:
        session.close()


def test_deferred_cursor_rechecks_dependency_after_start_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The JIT proof still freezes the member before dereferencing it.

    Start validation is name/stat-only, so the recheck that matters is inside
    materialization: a same-stat rewrite landing between the proof's first
    capture and its verification recapture is refused, even though only
    ctime records it.  The Eiger sidecar dataset is chunked, so the rewrite
    goes through a separate writer process rather than a raw byte offset.
    """

    _store, _capture, start, sidecar = external_eiger_capture(tmp_path)
    intent = start.intent_snapshot.thaw()
    intent.output_mode = "Overwrite"
    snapshot = RunIntentStore(intent).snapshot()
    request = RequestId(3)
    source = snapshot.thaw().source_spec
    receipt, session = admit_with_session(StartCapture(
        request,
        1,
        snapshot,
        SourceCapture(request, 1, source),
    ))
    raced = _race_dependency_after_first_capture(
        monkeypatch,
        sidecar,
        lambda: _same_stat_rewrite_from_other_process(
            sidecar, "entry/data/data"
        ),
    )
    try:
        validate_admitted_receipt(receipt, session)
        deferred = receipt.deferred_directory
        assert deferred is not None
        with pytest.raises(
            SourceRevisionChanged,
            match="HDF5 dependency changed during admission: ",
        ):
            materialize_deferred_output(
                receipt,
                session,
                deferred.entries[0],
                cancelled=lambda: False,
            )
        assert raced()
        with h5py.File(sidecar, "r") as handle:
            assert handle["entry/data/data"][0, 0, 0] == 7
    finally:
        session.close()


@pytest.mark.parametrize("output_mode", ("Overwrite", "Append"))
def test_an_automatic_gi_run_leaves_an_unsuffixed_prior_result_alone(
    tmp_path: Path, output_mode: str,
) -> None:
    """The new automatic name must not migrate, resume or replace the old one.

    Before the marker, a GI run of `scan` and a Standard run of `scan` both
    published `scan_int2d.nexus`.  A file already at that name is therefore an
    older GI result or the Standard result.  Either way the GI run now selects
    its own target, and is admitted exactly as it would be with no such file:
    Append checks the target it actually selected.
    """
    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    _write_sidecar(image, ("th=0.15", "exposure=1", "sequence=1"))
    processed = tmp_path / "processed"
    processed.mkdir()
    prior = processed / "scan_int2d.nexus"
    prior.write_bytes(b"an earlier result this run must not touch")
    before = (prior.read_bytes(), prior.stat().st_mtime_ns)

    dispositions = []
    for prior_present in (True, False):
        if not prior_present:
            prior.unlink()
        receipt, session = admit_with_session(
            _directory_start(tmp_path, gi=True, output_mode=output_mode),
        )
        try:
            assert len(receipt.outputs) == 1
            output = receipt.outputs[0]
            dispositions.append(output.disposition)
            assert output.item.target == processed / "scan_gi_int2d.nexus"
            assert not output.item.target.exists()
            if prior_present:
                assert (prior.read_bytes(), prior.stat().st_mtime_ns) == before
        finally:
            session.close()

    assert dispositions[0] == dispositions[1]
