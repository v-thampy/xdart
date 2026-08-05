"""Production-shaped probes for the E2-SD Directory and motor seam."""

from __future__ import annotations

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
    prepare_output,
    validate_admitted_receipt,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_widgets import source_header_projection
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.io.metadata import ImageMetadataRead
from xrd_tools.sources.directory_session import DirectoryIndexSession
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
        save_path=str(tmp_path / "processed"),
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
        targets = {
            item.item.source_path: item.item.target
            for item in receipt.outputs
        }
        assert targets == {
            first: output / "data" / "scan_0001.nexus",
            second: output / "live_test" / "scan_0001.nexus",
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
        output_preflight,
        "read_image_motor_metadata",
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

    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    _write_sidecar(image, ("th=0.1", "exposure=1", "sequence=1"))
    real_read = output_preflight.read_image_motor_metadata
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
        output_preflight,
        "read_image_motor_metadata",
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

    image = tmp_path / "scan_0001.tif"
    _write_tiff(image, 1)
    _write_sidecar(image, ("th=0.1", "exposure=1", "sequence=1"))
    real_read = output_preflight.read_image_motor_metadata
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
        output_preflight,
        "read_image_motor_metadata",
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
    # P4/OUT-1 retained-green: an explicit suffix-shaped save target is the
    # operator's exact choice and is preserved byte-for-byte.
    assert receipt.outputs[0].item.target == tmp_path / "processed.nxs"
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
        source_metadata,
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
    original_observed = source_metadata.read_image_metadata_observed

    def counted(path: Path, metadata_format: str | None):
        calls.append(Path(path))
        return original(path, metadata_format)

    def counted_observed(
        path: Path,
        metadata_format: str | None,
        **kwargs,
    ) -> ImageMetadataRead:
        calls.append(Path(path))
        return original_observed(path, metadata_format, **kwargs)

    monkeypatch.setattr(source_metadata, "read_image_metadata", counted)
    monkeypatch.setattr(
        source_metadata,
        "read_image_metadata_observed",
        counted_observed,
    )
    preview = FilesystemSourceAdapter().preview_motors(
        SourceObservationRequest(701, 0, source)
    )

    assert preview.gi_motor_choices == ("th", "exposure")
    assert calls == [first, second]

    calls.clear()
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
    assert "Deeper subfolders are evaluated during Run." in header.detail
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

from xrd_tools.sources.adapters import all_adapters
assert all_adapters() == (), all_adapters()

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

    monkeypatch.setattr(source_metadata, "read_image_metadata", counted)
    monkeypatch.setattr(
        source_metadata,
        "read_image_metadata_observed",
        lambda path, metadata_format, **_kwargs: ImageMetadataRead(
            counted(Path(path), metadata_format),
            None,
        ),
    )
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
        with pytest.raises(
            ValueError,
            match=(
                "GI metadata motor 'th'.*finite value in every admitted TIFF"
            ),
        ):
            prepare_output(
                _tiff_directory_start(tmp_path, recursive=True),
                cancelled=lambda: False,
                session_owner=sessions.append,
            )
        assert grandchild in calls
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
    assert not header.ready
    assert "Deeper subfolders are evaluated during Run." in header.detail


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
    assert not header.ready
    assert "Deeper subfolders are evaluated during Run." in header.detail


def test_recursive_tiff_preview_requires_at_least_one_readable_image(
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
    assert not source_header_projection(preview).ready


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

    assert yielded == [tmp_path / "scan_0000.tif"]
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
        assert item.target.name == "scan.nexus"
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
            lambda: workspace._observation is None
            and bool(source.preview_requests),
        )
        field = next(
            value
            for value in controls._bound_state.fields
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

        with pytest.raises(ValueError, match="source"):
            validate_admitted_receipt(receipt, session)
    finally:
        session.close()


def test_same_stat_eiger_member_rewrite_invalidates_admission(
    tmp_path: Path,
) -> None:
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
        os.utime(sidecar, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = sidecar.stat()
        assert (after.st_size, after.st_mtime_ns) == (
            before.st_size,
            before.st_mtime_ns,
        )
        assert after.st_ctime_ns != before.st_ctime_ns

        with pytest.raises(ValueError, match="external source member"):
            validate_admitted_receipt(receipt, session)
    finally:
        session.close()
