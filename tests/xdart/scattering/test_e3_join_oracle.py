"""Frozen mounted-shell oracle for the E3 context/shell join.

Every operator action in this module enters through the public composition
root: a real shell click or the shell's typed command signal.  Tests may
inspect the mounted controller and executor after the fact, but never drive a
private controller beside a decorative shell.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from pathlib import Path
import shutil
from threading import Event, current_thread
import time

import fabio.tifimage
import h5py
import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.adapters import (
    browse_loader as browse_module,
)
from xdart.gui.tabs.scattering.adapters import (
    dynamic_output as output_module,
)
from xdart.gui.tabs.scattering.adapters import (
    run_executor as executor_module,
)
from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadRequest,
    BrowseLoadStatus,
)
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering import page as page_module
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xdart.modules.display_context import ContextKind
from xrd_tools.io import (
    FrameViewReader,
    ProcessedScan,
    get_raw_frame,
)
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec

from xrd_tools.integrate.calibration import detector_calibration_to_integrator


_PRODUCTION_SINK = output_module.NexusSink
_PRODUCTION_OPEN_SOURCE = executor_module.open_source


def _wait(
    app: QtWidgets.QApplication,
    predicate,
    *,
    timeout: float = 20.0,
    diagnostic=lambda: "",
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError(f"E3-J0 timed out: {diagnostic()}")


class _RecordingIntegrator:
    def __init__(
        self,
        calibration,
        facts: list[tuple[str, str]],
        *,
        delay: float,
    ) -> None:
        self._inner = detector_calibration_to_integrator(calibration)
        self._facts = facts
        self._delay = delay

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __deepcopy__(self, memo):
        copied = object.__new__(type(self))
        memo[id(self)] = copied
        copied._inner = deepcopy(self._inner, memo)
        copied._facts = self._facts
        copied._delay = self._delay
        return copied

    def integrate1d(self, *args, **kwargs):
        self._facts.append(("integrate1d", current_thread().name))
        if self._delay:
            time.sleep(self._delay)
        return self._inner.integrate1d(*args, **kwargs)

    def integrate2d(self, *args, **kwargs):
        self._facts.append(("integrate2d", current_thread().name))
        if self._delay:
            time.sleep(self._delay)
        return self._inner.integrate2d(*args, **kwargs)


class _RecordingScalarReader:
    def __init__(
        self,
        reader: FrameViewReader,
        facts: list[tuple[str, str]],
        entered: Event | None,
        release: Event | None,
        gate_path: Path | None,
    ) -> None:
        self._reader = reader
        self._facts = facts
        self._entered = entered
        self._release = release
        self._gate_path = gate_path

    def __enter__(self) -> "_RecordingScalarReader":
        if self._reader.__enter__() is not self._reader:
            raise RuntimeError("FrameViewReader changed identity")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._reader.__exit__(exc_type, exc, tb)

    def read_scalar_catalog(self, *, cancelled):
        self._facts.append(("read-scalar-catalog", current_thread().name))
        gated = (
            self._gate_path is None
            or self._reader.path.resolve() == self._gate_path.resolve()
        )
        if gated and self._entered is not None:
            self._entered.set()
        if gated and self._release is not None:
            self._release.wait(timeout=10.0)
        return self._reader.read_scalar_catalog(cancelled=cancelled)


@dataclass(slots=True)
class _MountedRig:
    app: QtWidgets.QApplication
    page: ScatteringWorkspace
    lifecycle: ScatteringCoordinator
    executor: StandardRunExecutor
    loader: BrowseLoader
    project: Path
    selected: Path
    output: Path
    facts: list[tuple[str, str]]
    browse_facts: list[tuple[str, str]]

    @property
    def shell(self) -> ScatteringWorkspaceShell:
        shell = self.page.findChild(ScatteringWorkspaceShell)
        assert shell is not None, (
            "the public ScatteringWorkspace still has no mounted E3 shell"
        )
        return shell

    @property
    def controller(self) -> ContextController:
        value = self.page._context_controller
        assert type(value) is ContextController
        return value

    def command(self, command: ShellCommand) -> None:
        self.shell.commandRequested.emit(command)
        self.app.processEvents()

    def close(self):
        receipt = self.page.close_workspace()
        if receipt is None:
            return None
        deadline = time.monotonic() + 10.0
        while (
            receipt.cleanup_status is not CleanupStatus.CLEANED
            and time.monotonic() < deadline
        ):
            self.app.processEvents()
            receipt = self.page.close_workspace()
        return receipt


def _mount(
    monkeypatch,
    root: Path,
    *,
    labels: tuple[int, ...] = tuple(range(1, 65)),
    scan_name: str = "run.with.dots",
    output_mode: str = "Overwrite",
    max_display_items: int = 2,
    reduction_delay: float = 0.0,
    browse_entered: Event | None = None,
    browse_release: Event | None = None,
    browse_gate_path: Path | None = None,
    browse_join_timeout: float = 5.0,
) -> _MountedRig:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    project = root / "project"
    raw = project / "raw"
    processed = project / "processed"
    raw.mkdir(parents=True)
    processed.mkdir(parents=True)
    members = tuple(
        raw / f"{scan_name}_{label:04d}.tif" for label in labels
    )
    for ordinal, member in enumerate(members, 1):
        fabio.tifimage.TifImage(
            data=(
                np.arange(24, dtype=np.uint16).reshape(4, 6)
                + ordinal - 1
            )
        ).write(str(member))
    selected = members[0]
    poni = project / "detector.poni"
    poni.write_text(
        'poni_version: 2.1\nDetector: Detector\n'
        'Detector_config: {"pixel1": 0.0001, "pixel2": 0.0001, "max_shape": [4, 6], "orientation": 3}\n'
        'Distance: 0.1\nPoni1: 0.0002\nPoni2: 0.0003\n'
        'Rot1: 0\nRot2: 0\nRot3: 0\nWavelength: 1e-10\n'
    )
    output = processed / "run.with.dots_int2d.nexus"
    facts: list[tuple[str, str]] = []
    browse_facts: list[tuple[str, str]] = []

    def open_source(spec):
        source = _PRODUCTION_OPEN_SOURCE(spec)
        facts.append(("source-open", current_thread().name))
        to_scan = source.to_scan
        close = getattr(source, "close", None)

        def recorded_scan(**kwargs):
            facts.append(("source-to-scan", current_thread().name))
            return to_scan(**kwargs)

        def recorded_close():
            facts.append(("source-close", current_thread().name))
            return close()

        monkeypatch.setattr(source, "to_scan", recorded_scan)
        if callable(close):
            monkeypatch.setattr(source, "close", recorded_close)
        return source

    monkeypatch.setattr(executor_module, "open_source", open_source)
    monkeypatch.setattr(
        executor_module,
        "poni_to_integrator",
        lambda calibration: _RecordingIntegrator(
            calibration, facts, delay=reduction_delay
        ),
    )

    def sink_factory(*args, **kwargs):
        facts.append(("sink-open", current_thread().name))
        return _PRODUCTION_SINK(*args, **kwargs)

    monkeypatch.setattr(output_module, "NexusSink", sink_factory)

    def open_scan(source, **kwargs):
        browse_facts.append(("open-scan", current_thread().name))
        return ProcessedScan(source, **kwargs)

    def open_reader(source, **kwargs):
        return _RecordingScalarReader(
            FrameViewReader(source, **kwargs),
            browse_facts,
            browse_entered,
            browse_release,
            browse_gate_path,
        )

    loader = BrowseLoader(
        max_items=32,
        join_timeout=browse_join_timeout,
        open_scan=open_scan,
        open_reader=open_reader,
    )
    # The parent has no loader at all.  Patching the future production factory
    # preserves a meaningful parent-red assertion: no mounted shell, rather
    # than an unexpected constructor keyword.
    monkeypatch.setattr(
        page_module, "BrowseLoader", lambda **_kwargs: loader, raising=False
    )
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor(
        max_display_items=max_display_items
    )
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(selected),
                poni_file=str(poni),
                project_root=str(project),
                save_path=str(processed / "run.with.dots.nexus"),
                output_mode=output_mode,
                max_cores=1,
                bai_1d_args={"npt": 12},
                bai_2d_args={"npt_rad": 10, "npt_azim": 8},
            )
        ),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    page.show()
    app.processEvents()
    return _MountedRig(
        app,
        page,
        lifecycle,
        executor,
        loader,
        project,
        selected,
        output,
        facts,
        browse_facts,
    )


def _run(rig: _MountedRig) -> None:
    _wait(
        rig.app,
        lambda: rig.shell.run_controls.startButton.isEnabled(),
    )
    rig.shell.run_controls.startButton.click()
    _wait(
        rig.app,
        lambda: rig.controller.acquisition_context is not None or rig.lifecycle.phase is RunPhase.FAILED,
        diagnostic=lambda: rig.page._notice_text,
    )

    assert rig.controller.acquisition_context is not None, rig.page._notice_text


def _pause(rig: _MountedRig) -> None:
    if rig.lifecycle.phase is RunPhase.RUNNING:
        _wait(
            rig.app,
            lambda: "Pause" in rig.shell.run_controls.startButton.text(),
        )
        rig.shell.run_controls.startButton.click()
    _wait(
        rig.app,
        lambda: rig.lifecycle.phase is RunPhase.PAUSED,
        diagnostic=lambda: rig.lifecycle.phase.value,
    )


def _produce_browse_artifact(monkeypatch, root: Path) -> Path:
    rig = _mount(
        monkeypatch,
        root,
        labels=(1, 2, 3),
        scan_name="browse.with.dots",
        max_display_items=3,
    )
    try:
        _run(rig)
        _wait(rig.app, lambda: rig.lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED})
        assert rig.lifecycle.phase is RunPhase.IDLE, rig.page._notice_text
        assert rig.output.is_file()
        assert rig.close().cleanup_status is CleanupStatus.CLEANED
        result = rig.output.with_name("browse.with.dots.nexus")
        if result != rig.output:
            rig.output.rename(result)
        return result
    finally:
        rig.close()
        rig.page.deleteLater()
        rig.app.processEvents()


def test_j0_01_run_mounts_exact_a_and_shell_bindings(
    monkeypatch, tmp_path: Path,
) -> None:
    rig = _mount(monkeypatch, tmp_path, reduction_delay=0.001)
    try:
        assert type(rig.shell) is ScatteringWorkspaceShell
        assert type(rig.page._context_projection) is ContextProjection
        assert (
            rig.controller._projection
            is rig.page._context_projection
        )
        assert len(
            rig.page.findChildren(ScatteringWorkspaceShell)
        ) == 1
        _run(rig)
        context = rig.controller.acquisition_context
        identity = rig.controller.run_identity
        assert context is rig.executor.acquisition_context(identity)
        assert rig.controller.selection.names(context)
        assert context.scan is context.current_display_scan
        assert context.record_store is context.publication_store
        bindings = context.display_bindings()
        assert bindings.scan is context.scan
        assert bindings.record_store is context.record_store
        assert bindings.publication_store is context.publication_store

        _wait(rig.app, lambda: bool(rig.controller.frame_keys))
        # Presentation pacing (Single + Auto Last) repaints the scientific
        # surface on hydration boundaries, not on every FRAME_READY, so the
        # selector can trail controller.frame_keys by a few frames on a slow
        # host (Linux CI).  The contract is that a reconcile mirrors the
        # exact key objects; wait for one, then pin the mirrored snapshot.
        selector = rig.shell.scientific.frame_selector

        def selector_mirrors_keys() -> bool:
            keys = rig.controller.frame_keys
            return bool(keys) and selector.count() == len(keys) and all(
                selector.itemData(index) is key
                for index, key in enumerate(keys)
            )

        _wait(
            rig.app,
            selector_mirrors_keys,
            diagnostic=lambda: (
                f"selector={selector.count()} "
                f"keys={len(rig.controller.frame_keys)} "
                f"phase={rig.lifecycle.phase}"
            ),
        )
        keys = rig.controller.frame_keys
        assert all(
            selector.itemData(index) is key
            for index, key in enumerate(keys)
        )
        assert all(
            thread == "scattering-standard"
            for operation, thread in rig.facts
            if operation in {"source-open", "source-to-scan", "sink-open"}
        )
        assert all(
            thread != current_thread().name
            for operation, thread in rig.facts
            if operation in {"integrate1d", "integrate2d"}
        )
        assert [item[0] for item in rig.facts].count("source-open") == 1
        assert [item[0] for item in rig.facts].count("sink-open") == 1
    finally:
        rig.close()


def test_j0_02_browser_footer_share_repeated_dotted_exact_keys(
    monkeypatch, tmp_path: Path,
) -> None:
    browse = _produce_browse_artifact(monkeypatch, tmp_path / "b")
    rig = _mount(
        monkeypatch,
        tmp_path / "a",
        labels=tuple(range(1, 80)),
        reduction_delay=0.001,
    )
    try:
        _run(rig)
        # _run returns once the acquisition context exists, before any frame
        # is catalogued; a Pause landing that early on a slow host (Linux CI)
        # freezes navigation with no key for frame 1.  Cores=1 delivers frames
        # in order, so the first key is frame 1 once any key exists.
        _wait(rig.app, lambda: bool(rig.controller.frame_keys))
        _pause(rig)
        a_one = next(
            key for key in rig.controller.frame_keys
            if key.local_frame_label == 1
        )
        # Production only emits SELECT_SCAN for a row the filesystem catalog
        # actually discovered, so the operator route is: open the artifact's
        # folder, let the real catalog find it, then click that row.
        monkeypatch.setattr(
            rig.page,
            "_browser_directory_chooser",
            lambda _current, _start_directory: str(browse.parent),
        )
        rig.command(
            ShellCommand(ShellCommandKind.MENU, "File:Open Folder")
        )

        def catalog_row():
            scans = rig.shell.browser.scans
            return next(
                (
                    scans.item(index)
                    for index in range(scans.count())
                    if scans.item(index).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    )
                    == str(browse)
                ),
                None,
            )

        _wait(
            rig.app,
            lambda: catalog_row() is not None,
            diagnostic=lambda: (
                f"{browse} never entered the browser catalog"
            ),
        )
        discovered = catalog_row()
        assert discovered is not None
        discovered.setSelected(True)
        rig.app.processEvents()
        _wait(
            rig.app,
            lambda: (
                rig.controller.selection is not None
                and rig.controller.selection.kind is ContextKind.BROWSE
            ),
        )
        b_one = next(
            key for key in rig.controller.frame_keys
            if key.local_frame_label == 1
        )
        assert a_one is not b_one
        assert a_one.local_frame_label == b_one.local_frame_label == 1
        assert b_one.source_scan == "browse.with.dots"
        assert rig.shell.scientific.frame_selector.currentData() is b_one
        row = rig.shell.browser.frame_model.row_for(b_one)
        assert row is not None
        assert (
            rig.shell.browser.frame_model.index(row, 0).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            is b_one
        )
    finally:
        rig.close()


def test_j0_03_pause_closes_submission_and_exposes_a(
    monkeypatch, tmp_path: Path,
) -> None:
    rig = _mount(
        monkeypatch,
        tmp_path,
        labels=tuple(range(1, 160)),
        reduction_delay=0.0015,
    )
    try:
        _run(rig)
        _wait(rig.app, lambda: len(rig.controller.frame_keys) >= 2)
        _pause(rig)
        context = rig.controller.acquisition_context
        before = len(context.record_store.catalog_snapshot().entries)
        deadline = time.monotonic() + 0.08
        while time.monotonic() < deadline:
            rig.app.processEvents()
            time.sleep(0.002)
        assert (
            len(context.record_store.catalog_snapshot().entries)
            == before
        )
        assert rig.controller.selection.names(context)
        assert "Resume" in rig.shell.run_controls.startButton.text()
        assert rig.shell.run_controls.startButton.isEnabled()
    finally:
        rig.close()


def test_j0_04_browse_b_is_off_gui_and_selected_only_after_ready(
    monkeypatch, tmp_path: Path,
) -> None:
    browse = _produce_browse_artifact(monkeypatch, tmp_path / "b")
    entered, release = Event(), Event()
    rig = _mount(
        monkeypatch,
        tmp_path / "a",
        labels=tuple(range(1, 100)),
        reduction_delay=0.001,
        browse_entered=entered,
        browse_release=release,
    )
    try:
        _run(rig)
        _pause(rig)
        acquisition = rig.controller.acquisition_context
        a_selection = rig.controller.selection
        rig.command(
            ShellCommand(
                ShellCommandKind.SELECT_SCAN, str(browse)
            )
        )
        assert entered.wait(timeout=5.0), (
            rig.browse_facts,
            None if rig.loader._active is None else rig.loader._active.outcome,
        )
        rig.app.processEvents()
        assert rig.controller.selection is a_selection
        release.set()
        _wait(
            rig.app,
            lambda: (
                rig.controller.selection is not None
                and rig.controller.selection.kind is ContextKind.BROWSE
            ),
        )
        browsed = rig.controller.browse_context
        assert browsed.record_store is not acquisition.record_store
        assert (
            browsed.publication_store
            is not acquisition.publication_store
        )
        assert rig.browse_facts == [
            ("open-scan", "scattering-browse"),
            ("read-scalar-catalog", "scattering-browse"),
        ]
        assert rig.shell.scientific.frame_selector.currentData() is (
            rig.controller.frame_keys[0]
        )
    finally:
        release.set()
        rig.close()


def test_j0_05_resume_is_an_a_pointer_move_without_b_to_a_writes(
    monkeypatch, tmp_path: Path,
) -> None:
    browse = _produce_browse_artifact(monkeypatch, tmp_path / "b")
    rig = _mount(
        monkeypatch,
        tmp_path / "a",
        labels=tuple(range(1, 120)),
        reduction_delay=0.001,
    )
    try:
        _run(rig)
        _pause(rig)
        acquisition = rig.controller.acquisition_context
        frozen_configuration = acquisition.run_configuration
        frozen_scan = acquisition.scan
        frozen_records = acquisition.record_store
        frozen_publications = acquisition.publication_store
        frozen_catalog = frozen_publications.catalog_snapshot()
        frozen_residency = frozen_publications.residency_snapshot()
        rig.command(
            ShellCommand(
                ShellCommandKind.SELECT_SCAN, str(browse)
            )
        )
        _wait(
            rig.app,
            lambda: (
                rig.controller.selection is not None
                and rig.controller.selection.kind is ContextKind.BROWSE
            ),
        )
        browsed = rig.controller.browse_context
        assert frozen_publications.catalog_snapshot() == frozen_catalog
        assert frozen_publications.residency_snapshot() == frozen_residency
        rig.shell.run_controls.startButton.click()
        _wait(rig.app, lambda: rig.lifecycle.phase is RunPhase.RUNNING)
        assert rig.controller.selection.names(acquisition)
        assert browsed.invalidated is True
        assert rig.controller.acquisition_context is acquisition
        assert acquisition.run_configuration is frozen_configuration
        assert acquisition.scan is frozen_scan
        assert acquisition.record_store is frozen_records
        assert acquisition.publication_store is frozen_publications
        resumed_catalog = frozen_publications.catalog_snapshot()
        assert len(resumed_catalog.entries) >= len(frozen_catalog.entries)
        assert all(
            resumed is paused
            for resumed, paused in zip(
                resumed_catalog.entries,
                frozen_catalog.entries,
                strict=False,
            )
        )
        assert all(
            key.artifact != browsed.requested_path
            for key in resumed_catalog.entries
        )
    finally:
        rig.close()


def test_j0_06_stop_is_fail_closed_without_orphan_work(
    monkeypatch, tmp_path: Path,
) -> None:
    browse = _produce_browse_artifact(monkeypatch, tmp_path / "b")
    entered, release = Event(), Event()
    for state in ("running", "paused", "browse", "resumed"):
        rig = _mount(
            monkeypatch,
            tmp_path / state,
            labels=tuple(range(1, 180)),
            reduction_delay=0.001,
            browse_entered=entered if state == "browse" else None,
            browse_release=release if state == "browse" else None,
            browse_join_timeout=0.02,
        )
        try:
            entered.clear()
            release.clear()
            _run(rig)
            if state != "running":
                _pause(rig)
            if state == "browse":
                rig.command(
                    ShellCommand(
                        ShellCommandKind.SELECT_SCAN, str(browse)
                    )
                )
                assert entered.wait(timeout=5.0)
            elif state == "resumed":
                rig.shell.run_controls.startButton.click()
                _wait(
                    rig.app,
                    lambda: rig.lifecycle.phase is RunPhase.RUNNING,
                )
            rig.shell.run_controls.stopButton.click()
            release.set()
            _wait(
                rig.app,
                lambda: rig.lifecycle.phase
                in {RunPhase.IDLE, RunPhase.FAILED},
            )
            # Stop settles the writer before the asynchronous final-artifact
            # Browse handoff. It must settle too, rather than remain orphaned.
            _wait(rig.app, lambda: not rig.controller.browse_pending)
            assert not rig.shell.run_controls.stopButton.isEnabled()
        finally:
            release.set()
            receipt = rig.close()
            assert receipt.cleanup_status is CleanupStatus.CLEANED
            worker = rig.loader._worker
            assert worker is None or not worker.is_alive()


def test_j0_07_b_to_c_releases_b_before_c_and_retains_failed_owner(
    monkeypatch, tmp_path: Path,
) -> None:
    b = _produce_browse_artifact(monkeypatch, tmp_path / "b")
    c = _produce_browse_artifact(monkeypatch, tmp_path / "c")
    rig = _mount(
        monkeypatch,
        tmp_path / "a",
        labels=tuple(range(1, 100)),
        reduction_delay=0.001,
    )
    try:
        _run(rig)
        _pause(rig)
        rig.command(
            ShellCommand(ShellCommandKind.SELECT_SCAN, str(b))
        )
        _wait(
            rig.app,
            lambda: (
                rig.controller.selection is not None
                and rig.controller.selection.kind is ContextKind.BROWSE
            ),
        )
        context_b = rig.controller.browse_context
        # Catalog adoption precedes the separately queued preview publication.
        _wait(rig.app, lambda: (
            rig.shell.scientific.raw.image.image is not None
            and rig.shell.scientific.cake.image.image is not None
            and bool(rig.shell.scientific.curve.listDataItems())
        ))
        frame_b = rig.controller.navigation.current
        assert frame_b is not None
        assert rig.shell.scientific.raw.image.image is not None
        assert rig.shell.scientific.cake.image.image is not None
        assert rig.shell.scientific.curve.listDataItems()
        presentations: list[
            tuple[
                DisplayFrameKey | None,
                str,
                bool,
                bool,
                int,
            ]
        ] = []
        snapshots: list[bool] = []
        apply_state = rig.shell.apply_state

        def record_presentation(
            state,
            *,
            preserve_display: bool = False,
            preserve_scientific: bool = False,
            replace_scientific_on_failure: bool = False,
        ) -> None:
            apply_state(
                state,
                preserve_display=preserve_display,
                preserve_scientific=preserve_scientific,
                replace_scientific_on_failure=(
                    replace_scientific_on_failure
                ),
            )
            snapshots.append(
                rig.shell.scientific.viewer_loading_snapshot_visible
                and rig.shell.scientific.viewer_loading_snapshot_pixels > 0
            )
            presentations.append(
                (
                    state.navigation.current,
                    rig.shell.scientific.title.text(),
                    rig.shell.scientific.raw.image.image is not None,
                    rig.shell.scientific.cake.image.image is not None,
                    len(rig.shell.scientific.curve.listDataItems()),
                )
            )

        monkeypatch.setattr(
            rig.shell, "apply_state", record_presentation
        )
        real_settle_cache = rig.loader._settle_cache
        cancelled_requests: list[BrowseLoadRequest] = []

        def pending_once(cache):
            monkeypatch.setattr(
                rig.loader, "_settle_cache", real_settle_cache
            )
            assert cache is context_b.browse_1d_cache
            operation = rig.loader._active
            assert operation is not None
            cancelled_requests.append(operation.request)
            # Fail after the real owner withdraws admission authority, while
            # it still retains B's exact cache and presentation for retry.
            raise OSError("injected Browse cache close failure")

        monkeypatch.setattr(
            rig.loader, "_settle_cache", pending_once
        )
        rig.command(
            ShellCommand(ShellCommandKind.SELECT_SCAN, str(c))
        )
        assert len(cancelled_requests) == 1
        cancelled_c = cancelled_requests[0]
        assert rig.controller.browse_context is context_b
        assert context_b.released is False
        assert sum(
            operation == "open-scan"
            for operation, _thread in rig.browse_facts
        ) == 1

        # The refused replacement remains one exact loader-owned cleanup
        # operation.  Ordinary page polling must retire it before another C
        # request can be admitted, without releasing or blanking B.
        _wait(
            rig.app,
            lambda: (
                not rig.controller.browse_pending
                and not rig.loader.owns_request(cancelled_c)
            ),
        )
        assert rig.controller._cleanup_receipt is None
        assert rig.controller.browse_context is context_b
        assert context_b.released is False

        rig.command(
            ShellCommand(ShellCommandKind.SELECT_SCAN, str(c))
        )
        _wait(
            rig.app,
            lambda: (
                rig.controller.browse_context is not None
                and rig.controller.browse_context is not context_b
                and rig.controller.browse_context.requested_path == str(c)
                and rig.controller.selection.names(
                    rig.controller.browse_context
                )
            ),
        )
        assert context_b.released is True
        assert len(context_b.record_store) == 0
        context_c = rig.controller.browse_context
        assert context_c is not None
        assert context_c.load_request is not cancelled_c
        assert (
            context_c.load_request.load_generation
            == cancelled_c.load_generation + 1
        )
        frame_c = rig.controller.navigation.current
        assert frame_c is not None and frame_c is not frame_b
        assert presentations
        assert presentations[0][0] is frame_b
        assert presentations[-1][0] is frame_c
        assert all(title for _frame, title, _raw, _cake, _traces in presentations)
        # A changed Browse identity releases foreign scientific arrays while
        # retaining the capped screen raster until the new sparse read is ready.
        assert all(
            snapshot or (raw and cake and traces)
            for (_frame, _title, raw, cake, traces), snapshot
            in zip(presentations, snapshots, strict=True)
        ), (presentations, snapshots)
        _wait(rig.app, lambda: (
            rig.shell.scientific.raw.image.image is not None
            and rig.shell.scientific.cake.image.image is not None
            and bool(rig.shell.scientific.curve.listDataItems())
            and not rig.shell.scientific.viewer_loading_snapshot_visible
        ))
    finally:
        rig.close()


def test_j0_08_evicted_hydration_is_single_flight_and_does_not_blank(
    monkeypatch, tmp_path: Path,
) -> None:
    # Fund the sixteen-frame writer checkpoint, then exceed the retained
    # heavy window while the acquisition is still running.
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "16")
    rig = _mount(
        monkeypatch,
        tmp_path,
        labels=tuple(range(1, 180)),
        max_display_items=1,
        reduction_delay=0.003,
    )
    read_entered, release_read = Event(), Event()
    try:
        _run(rig)
        _wait(rig.app, lambda: len(rig.controller.frame_keys) >= 20)
        first = rig.controller.frame_keys[0]
        # Residency may only drop frame 1 once the writer's checkpoint has
        # verified its rows; a slow host accepts twenty frames before that.
        # Wait for the eviction while the acquisition still re-arms
        # residency, so the hydration below has to read.
        _wait(
            rig.app,
            lambda: not any(
                frame is first for frame in rig.controller.resident_frame_keys
            ),
            diagnostic=lambda: (
                f"frame 1 still resident after "
                f"{len(rig.controller.frame_keys)} frames, "
                f"phase={rig.lifecycle.phase.value}"
            ),
        )
        _pause(rig)
        # A record batch written since the last sixteen-frame checkpoint
        # revokes the artifact's checkpoint-hydration authority, and the shell
        # drops an evicted-frame read silently while it is revoked.  A durable
        # Pause seals the quiesced writer itself, so the hydration below is
        # authorised on every host, not only when the pause happens to land
        # on a checkpoint boundary.
        run = rig.executor._active
        assert run is not None and run.session is not None
        records = run.display.artifacts[str(run.artifact)].records
        assert records._checkpoint_hydration_authority()[0] is not None
        assert run.session._dynamic_nexus_checkpoint_count == 0
        _wait(rig.app, lambda: rig.shell.scientific.raw.image.image is not None)
        from xdart.gui.tabs.scattering import hydration_transport

        read_labels: list[int] = []
        real_read = hydration_transport.read_frame_preview

        def counted_read(read_key, **kwargs):
            read_labels.append(int(read_key.frame_identity))
            read_entered.set()
            assert release_read.wait(5.0)
            return real_read(read_key, **kwargs)

        monkeypatch.setattr(
            hydration_transport, "read_frame_preview", counted_read
        )
        before_raw = np.array(
            rig.shell.scientific.raw.image.image, copy=True
        )
        command = ShellCommand(
            ShellCommandKind.HYDRATE_FRAME,
            frame=first,
            frames=(first,),
        )
        rig.command(command)
        assert read_entered.wait(5.0), (
            "no preview read for frame 1; resident="
            f"{any(frame is first for frame in rig.controller.resident_frame_keys)}"
            f" authorised={records._checkpoint_hydration_authority()[0] is not None}"
        )
        rig.command(command)
        np.testing.assert_array_equal(
            rig.shell.scientific.raw.image.image, before_raw
        )
        release_read.set()
        _wait(
            rig.app,
            lambda: (
                rig.shell.scientific.frame_selector.currentData()
                is first
                and rig.shell.scientific.title.text()
                == "run.with.dots_0001.tif"
            ),
        )
        # Single flight: the repeated command coalesced onto ONE exact read.
        assert read_labels.count(first.local_frame_label) == 1
    finally:
        release_read.set()
        rig.close()


def test_j0_09_equal_foreign_stale_and_late_values_are_inert(
    monkeypatch, tmp_path: Path,
) -> None:
    rig = _mount(
        monkeypatch,
        tmp_path,
        labels=tuple(range(1, 16)),
    )
    try:
        _run(rig)
        _wait(rig.app, lambda: rig.lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED})
        assert rig.lifecycle.phase is RunPhase.IDLE, rig.page._notice_text
        accepted = rig.controller.frame_keys[-1]
        prior = rig.shell.scientific.frame_selector.currentData()
        equal_foreign = replace(accepted)
        assert equal_foreign == accepted
        assert equal_foreign is not accepted
        rig.command(
            ShellCommand(
                ShellCommandKind.SELECT_FRAME,
                frame=equal_foreign,
                frames=(equal_foreign,),
            )
        )
        assert (
            rig.shell.scientific.frame_selector.currentData() is prior
        )

        foreign = DisplayFrameKey(
            replace(accepted.run_identity, fingerprint="foreign"),
            accepted.source_scan,
            accepted.artifact,
            accepted.local_frame_label,
            accepted.work_ordinal,
        )
        rig.command(
            ShellCommand(
                ShellCommandKind.HYDRATE_FRAME,
                frame=foreign,
                frames=(foreign,),
            )
        )
        assert (
            rig.shell.scientific.frame_selector.currentData() is prior
        )
    finally:
        rig.close()


def test_j0_10_close_closeevent_and_deferreddelete_share_cached_receipt(
    monkeypatch, tmp_path: Path,
) -> None:
    for entrypoint in ("close", "close_event", "deferred_delete"):
        rig = _mount(monkeypatch, tmp_path / entrypoint, labels=(1,))
        if entrypoint == "close":
            first = rig.page.close_workspace()
        elif entrypoint == "close_event":
            rig.page.close()
            first = rig.page.close_workspace()
        else:
            QtWidgets.QApplication.sendEvent(
                rig.page,
                QtCore.QEvent(QtCore.QEvent.DeferredDelete),
            )
            first = rig.page.close_workspace()
        _wait(
            rig.app,
            lambda: (
                rig.page.close_workspace().cleanup_status
                is CleanupStatus.CLEANED
            ),
        )
        terminal = rig.page.close_workspace()
        duplicate = rig.page.close_workspace()
        assert terminal is duplicate
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        if first.cleanup_identity is not None:
            assert terminal.cleanup_identity is first.cleanup_identity


def test_j0_11_relative_raw_path_survives_project_tree_move(
    monkeypatch, tmp_path: Path,
) -> None:
    rig = _mount(monkeypatch, tmp_path, labels=(1,))
    try:
        _run(rig)
        _wait(rig.app, lambda: rig.lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED})
        assert rig.lifecycle.phase is RunPhase.IDLE, rig.page._notice_text
        with h5py.File(rig.output, "r") as handle:
            source = handle["entry/frames/frame_0001/source/path"][()]
            if isinstance(source, bytes):
                source = source.decode()
            assert source == "raw/run.with.dots_0001.tif"
        rig.close()
        moved = tmp_path / "relocated"
        shutil.move(str(rig.project), str(moved))
        raw = get_raw_frame(
            moved / "processed" / rig.output.name,
            1,
            source_root=moved,
        )
        np.testing.assert_array_equal(
            np.asarray(raw),
            np.arange(24, dtype=np.uint16).reshape(4, 6),
        )
    finally:
        if not rig.page._closed:
            rig.close()


def test_j0_12_append_is_visible_and_admitted_through_real_preflight(
    monkeypatch, tmp_path: Path,
) -> None:
    rig = _mount(
        monkeypatch,
        tmp_path,
        labels=(1,),
        output_mode="Append",
    )
    try:
        state = rig.shell.run_controls
        assert state.write_mode() == "Append"
        _run(rig)
        _wait(rig.app, lambda: rig.lifecycle.phase in {RunPhase.IDLE, RunPhase.FAILED})
        assert rig.lifecycle.phase is RunPhase.IDLE, rig.page._notice_text
        assert rig.output.is_file()
        with h5py.File(rig.output, "r") as handle:
            assert list(handle["entry/frames"]) == ["frame_0001"]
        assert any(fact == "source-open" for fact, _thread in rig.facts)
        assert any(fact == "sink-open" for fact, _thread in rig.facts)
    finally:
        rig.close()
