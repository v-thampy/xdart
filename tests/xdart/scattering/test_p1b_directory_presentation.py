"""Finite P1-B directory-depth and artifact-local presentation oracle."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

import h5py
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.e3_shell_support import make_shell_projection
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import (
    SourceCapture,
    SourceCountScope,
    SourceObservationRequest,
    StartCapture,
)
from xdart.gui.tabs.scattering.events import RequestId
from xdart.gui.tabs.scattering.output_preflight import prepare_output
from xdart.gui.tabs.scattering.shell_values import (
    ArtifactProgress,
    FrameNavigationProjection,
    ProgressProjection,
)
from xdart.gui.tabs.scattering.shell_widgets import source_header_projection
from xdart.gui.tabs.scattering.source_view import SourceStatusView
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec, image_series_spec
import xrd_tools.sources.registry  # noqa: F401  # register built-in readers


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _write_stack(path: Path, frames: int) -> None:
    with h5py.File(path, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        detector.create_dataset(
            "data",
            data=np.ones((frames, 195, 487), dtype=np.uint16),
            chunks=(1, 195, 487),
            maxshape=(None, 195, 487),
        )


def _dispose(widget: QtWidgets.QWidget, qapp: QtWidgets.QApplication) -> None:
    widget.close()
    widget.deleteLater()
    qapp.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


def test_p1b_b13_directory_count_is_physical_and_bounded(
    tmp_path: Path,
    qapp: QtWidgets.QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B13: readiness and Run share one physical root-plus-one envelope."""

    selected = tmp_path / "selected"
    immediate = selected / "immediate"
    deeper = immediate / "deeper"
    deeper.mkdir(parents=True)
    direct = selected / "direct.nxs"
    child = immediate / "child.nxs"
    excluded = deeper / "excluded.nxs"
    for path in (direct, child, excluded):
        _write_stack(path, 1)

    source = DirectorySourceSpec(
        selected,
        recursive=True,
        suffixes=(".nxs",),
        metadata_format=None,
    )
    adapter = FilesystemSourceAdapter()
    from xdart.gui.tabs.scattering.adapters import source as source_module

    opened = []
    with monkeypatch.context() as scoped:
        scoped.setattr(
            source_module,
            "candidate_owner",
            lambda path: (_ for _ in ()).throw(
                AssertionError(f"directory readiness opened {path}")
            ),
        )
        scoped.setattr(
            h5py,
            "File",
            lambda path, *args, **kwargs: opened.append(Path(path))
            or (_ for _ in ()).throw(
                AssertionError(f"directory readiness opened HDF5 {path}")
            ),
        )
        observation = adapter.observe(
            SourceObservationRequest(1301, 0, source)
        )
    assert opened == []
    assert observation.direct_child_count == 1
    assert observation.one_level_file_count == 2
    assert observation.observed_file_count == 2
    assert observation.file_count_scope is SourceCountScope.SELECTED_PLUS_IMMEDIATE

    # Explicit Image Series is the only place where this container's internal
    # frames are displayed as a count.
    stack = tmp_path / "selected-series.nxs"
    _write_stack(stack, 3)
    selected_observation = adapter.observe(
        SourceObservationRequest(1302, 0, image_series_spec(stack))
    )
    assert selected_observation.observed_file_count == 3

    poni = tmp_path / "cal.poni"
    write_poni(poni)
    intent = RunIntent(
        source_spec=source,
        poni_file=str(poni),
        project_root=str(tmp_path),
        save_path=str(tmp_path / "processed"),
        output_mode="Overwrite",
    )
    snapshot = RunIntentStore(intent).snapshot()
    request = RequestId(1303)
    sessions = []
    from xrd_tools.sources.directory_index import DirectoryIndex

    prepare_opened = []
    real_h5_file = h5py.File

    def observed_h5_file(path, *args, **kwargs):
        prepare_opened.append(Path(path).expanduser().absolute())
        return real_h5_file(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(h5py, "File", observed_h5_file)
        scoped.setattr(
            DirectoryIndex,
            "probe_candidate",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("B13 candidates must be depth-bounded before probe")
            ),
        )
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
        assert receipt.directory_discovered_file_count == 2
        assert receipt.directory_discovered_paths == (
            direct.absolute(),
            child.absolute(),
        )
        deferred = receipt.deferred_directory
        assert deferred is not None
        assert tuple(
            entry.candidates[0].path for entry in deferred.entries
        ) == (
            direct.absolute(),
            child.absolute(),
        )
        assert excluded.absolute() not in receipt.directory_discovered_paths
        assert excluded.absolute() not in prepare_opened
        assert not any(
            path == deeper.absolute()
            or path.is_relative_to(deeper.absolute())
            for path in prepare_opened
        )
    finally:
        for session in sessions:
            session.close()

    # Both retained diagnostic surfaces state the actual supported envelope;
    # neither promises a later deep-recursion pass.
    header = source_header_projection(observation)
    shell = ScatteringWorkspaceShell()
    view = SourceStatusView(shell.controls)
    shell.controls.set_source_widget(view)
    try:
        view.render(observation)
        mounted_text = " ".join((
            shell.controls.source_card.status.text(),
            view._count.text(),
            view._header.text,
            view._header.detail,
        )).casefold()
    finally:
        _dispose(shell, qapp)
    projected_text = f"{header.text} {header.detail}".casefold()
    for text in (mounted_text, projected_text):
        assert "selected folder" in text
        assert "immediate subfolders" in text
        assert "processed" in text
        assert "only" in text
        assert "deeper subfolders are evaluated during run" not in text
        assert "deeper subfolders processed during run" not in text


def test_p1b_b14_status_separates_file_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B14: physical outcomes come from the one accepted accounting owner."""

    # The four fields are an accepted E6 presentation seam.  P1-B must bind
    # their values to the canonical dynamic-attempt and durable-receipt owner,
    # rather than locally summing frames from discovered containers.
    from tests.xdart.scattering.test_p1b_output_graph import (
        _TERMINAL,
        _bridge_legacy_expected_target_state,
        _drain_until,
        _start,
        _write_tiff,
    )
    from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
    from xdart.gui.tabs.scattering.display_values import (
        StandardEventKind,
        StandardRunEvent,
    )
    from xrd_tools.session.dynamic_accounting import DynamicRunAccounting

    annotations = getattr(StandardRunEvent, "__annotations__", {})
    for field in (
        "files_discovered",
        "files_processed",
        "files_skipped",
        "files_pending",
    ):
        assert field in annotations

    captured_accounting: list[DynamicRunAccounting] = []
    attempt_calls = []
    real_init = DynamicRunAccounting.__init__
    real_begin_attempt = DynamicRunAccounting.begin_attempt

    def observed_init(owner, *args, **kwargs):
        real_init(owner, *args, **kwargs)
        captured_accounting.append(owner)

    def observed_begin_attempt(owner, key, *, source_revision):
        token = real_begin_attempt(
            owner, key, source_revision=source_revision,
        )
        attempt_calls.append(token)
        return token

    monkeypatch.setattr(DynamicRunAccounting, "__init__", observed_init)
    monkeypatch.setattr(
        DynamicRunAccounting, "begin_attempt", observed_begin_attempt,
    )
    _bridge_legacy_expected_target_state(monkeypatch)

    raw = tmp_path / "raw"
    raw.mkdir()
    good = raw / "good_0001.tif"
    partial = raw / "pending_0001.tif"
    invalid = raw / "stable-invalid.nxs"
    _write_tiff(good, 1)
    partial.write_bytes(b"partial-tiff")
    with h5py.File(invalid, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        detector = entry.create_group("instrument/detector")
        detector.create_dataset(
            "data", data=np.ones((1, 1, 2, 2), dtype=np.uint16),
        )
    poni = tmp_path / "cal.poni"
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        RunIntent(
            source_spec=DirectorySourceSpec(
                raw,
                suffixes=(".tif", ".nxs"),
                metadata_format=None,
            ),
            poni_file=str(poni),
            project_root=str(raw),
            save_path=str(tmp_path / "processed"),
            output_mode="Overwrite",
            processing_mode="Int 1D",
            live_mode=True,
            max_cores=1,
            bai_1d_args={"npt": 8, "method": "numpy"},
        ),
        request_value=1401,
    )
    try:
        events = list(_drain_until(
            executor,
            lambda values: any(
                event.kind is StandardEventKind.FRAME_READY for event in values
            ),
            timeout=60.0,
        ))
        early_index, early_frame = next(
            (index, event)
            for index, event in enumerate(events)
            if event.kind is StandardEventKind.FRAME_READY
        )
        # FRAME_READY is display/reduction truth, not a durable file receipt.
        assert early_frame.files_processed == 0
        assert all(
            event.files_processed == 0
            for event in events[:early_index + 1]
        )

        assert len(captured_accounting) == 1, (
            "B14 accounting: presentation fields exist, but the Live run did "
            "not delegate to one canonical DynamicRunAccounting owner"
        )
        accounting = captured_accounting[0]
        required = accounting.ledger.targets_by_mode

        def fully_durable(snapshot):
            return {
                key
                for key in snapshot.discovered
                if all(
                    (key, mode, target) in snapshot.durable
                    for mode, targets in required.items()
                    for target in targets
                )
            }

        deadline = time.monotonic() + 20.0
        snapshot = accounting.snapshot()
        while not fully_durable(snapshot) and time.monotonic() < deadline:
            newly_drained = tuple(executor.drain_events())
            next_snapshot = accounting.snapshot()
            if not fully_durable(next_snapshot):
                assert all(
                    event.files_processed == 0 for event in newly_drained
                )
            events.extend(newly_drained)
            snapshot = next_snapshot
            time.sleep(0.01)
        durable_frames = fully_durable(snapshot)
        assert len(durable_frames) == 1

        def settled_projection(values):
            return next((
                (index, event)
                for index, event in enumerate(values)
                if index > early_index
                and (
                    event.files_processed,
                    event.files_skipped,
                    event.files_pending,
                    event.files_discovered,
                ) == (1, 1, 1, 3)
            ), None)

        deadline = time.monotonic() + 20.0
        while settled_projection(events) is None and time.monotonic() < deadline:
            events.extend(executor.drain_events())
            snapshot = accounting.snapshot()
            time.sleep(0.01)
        settled = settled_projection(events)
        assert settled is not None
        settled_index, event = settled
        assert settled_index > early_index
        assert len(snapshot.attempts) == len(attempt_calls) == 1
        assert len(durable_frames) == event.files_processed == 1
        assert (
            event.files_discovered
            - len(durable_frames)
            - event.files_skipped
        ) == event.files_pending
    finally:
        executor.stop(identity)
        terminal = _drain_until(
            executor,
            lambda values: any(event.kind in _TERMINAL for event in values),
        )
        assert any(event.kind in _TERMINAL for event in terminal)
        executor.close(identity)


def test_p1b_b15_footer_is_exact_artifact_local(
    qapp: QtWidgets.QApplication,
) -> None:
    """B15: duplicate scan captions never mix two output artifacts."""

    state = make_shell_projection(
        frame_count=4,
        selected_index=1,
        heavy_indices=(),
        plot_mode="Single",
        source_scan="same-scan-caption",
    )
    original = state.navigation.frames
    frames = tuple(
        replace(
            frame,
            artifact=("a.nexus" if index % 2 == 0 else "b.nexus"),
            local_frame_label=(index // 2) + 1,
        )
        for index, frame in enumerate(original)
    )
    current = frames[1]
    artifact_frames = (frames[1], frames[3])
    state = replace(
        state,
        browser=replace(state.browser, frames=frames),
        scientific=replace(
            state.scientific,
            traces=tuple(
                replace(trace, frame=frames[index])
                for index, trace in enumerate(state.scientific.traces)
            ),
            heavy_available=frozenset(),
            heavy=None,
        ),
        navigation=FrameNavigationProjection(frames, current, (current,)),
        progress=ProgressProjection(
            4,
            4,
            "Ready",
            (ArtifactProgress(current.artifact, 2, 2, published=2),),
        ),
    )

    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)
        # Global retained history remains available to the browser/log owner.
        assert shell.browser.frame_model.rowCount() == 4
        assert tuple(shell.browser.frame_model.frames) == frames

        # Both footer-local calculations follow the exact current artifact.
        assert shell.scientific.frame_selector.count() == 2
        assert tuple(
            shell.scientific.frame_selector.itemData(index)
            for index in range(shell.scientific.frame_selector.count())
        ) == artifact_frames
        assert shell.scientific.progress.text() == "1/2"
    finally:
        _dispose(shell, qapp)
