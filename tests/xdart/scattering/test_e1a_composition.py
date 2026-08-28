"""Locked E1a composition checks for the mounted scattering workspace."""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from threading import Event, get_ident
import time
from typing import get_type_hints
import weakref

import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from xrd_tools.core.scan import SourceSpec
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.sources.selection import DirectorySourceSpec

from xdart.gui.tabs.scattering.contracts import (
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.controls_projection import (
    GI_ORIENTATION,
    PROJECT_ROOT,
    SAVE_PATH,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.page import (
    ScatteringWorkspace,
    _ObservationOperation,
)
from xdart.gui.tabs.scattering.source_view import SourceStatusView
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import ControlsPanelV2


def _wait_until(qapp: QtWidgets.QApplication, predicate: object, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.01)
    return False


def _shell(workspace: ScatteringWorkspace) -> ScatteringWorkspaceShell:
    shell = workspace.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    return shell


def _source_status(workspace: ScatteringWorkspace) -> SourceStatusView:
    status = workspace.findChild(SourceStatusView)
    assert status is not None
    return status


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class CountingStore(RunIntentStore):
    def __init__(self) -> None:
        super().__init__()
        self.commit_calls = 0
        self.make_next_commit_stale = False

    def commit(self, candidate: object, *, expected_revision: int):  # type: ignore[override]
        self.commit_calls += 1
        if self.make_next_commit_stale:
            self.make_next_commit_stale = False
            concurrent = self.snapshot().thaw()
            concurrent.project_root = "/concurrent"
            super().commit(concurrent, expected_revision=self.revision)
        return super().commit(candidate, expected_revision=expected_revision)  # type: ignore[arg-type]


class ImmediateSource:
    def __init__(self) -> None:
        self.observation_thread: int | None = None
        self.cancelled: list[int] = []
        self.capture_calls = 0
        self.motor_knowledge: SourceObservation | None = None

    def capture(self, source: object, request_id: object) -> object:
        self.capture_calls += 1
        raise AssertionError("E1a must not capture a source")

    def cancel(self, request_id: object) -> None:
        raise AssertionError("E1a must not use Start cancellation")

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.observation_thread = get_ident()
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "selected.dat",
            True,
            False,
            size_bytes=12,
        )

    def cancel_observation(self, observation_id: int) -> None:
        self.cancelled.append(observation_id)

    def publish_motor_knowledge(
        self, observation: SourceObservation
    ) -> None:
        self.motor_knowledge = observation

    def project_motor_knowledge(
        self, source, candidate_fingerprint=None
    ):
        knowledge = self.motor_knowledge
        if knowledge is None or knowledge.source != source:
            return None
        return knowledge


class BlockingSource(ImmediateSource):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()
        self.last_request: SourceObservationRequest | None = None
        self.requests: list[SourceObservationRequest] = []

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.last_request = request
        self.requests.append(request)
        self.started.set()
        assert self.release.wait(3.0)
        return super().observe(request)


class MalformedCompletionSource(ImmediateSource):
    def observe(self, request: SourceObservationRequest) -> object:
        self.observation_thread = get_ident()
        return object()


class RaisingCompletionSource(ImmediateSource):
    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.observation_thread = get_ident()
        raise RuntimeError("observation failed")


class WrongEchoSource(ImmediateSource):
    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.observation_thread = get_ident()
        return SourceObservation(
            request.observation_id + 1,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "wrong.dat",
            True,
            False,
        )


class QueuedSource(ImmediateSource):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[int] = []
        self.first_started = Event()
        self.release_first = Event()

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.calls.append(request.observation_id)
        if len(self.calls) == 1:
            self.first_started.set()
            assert self.release_first.wait(3.0)
        return super().observe(request)


class RecordingSource(ImmediateSource):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[SourceObservationRequest] = []

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.requests.append(request)
        return super().observe(request)


class SourceRecaptureStore(RunIntentStore):
    def __init__(self, source: SourceSpec) -> None:
        super().__init__()
        self._concurrent_source = source
        self.commit_calls = 0

    def commit(self, candidate: object, *, expected_revision: int):  # type: ignore[override]
        self.commit_calls += 1
        if self.commit_calls == 1:
            current = self.snapshot().thaw()
            current.source_spec = self._concurrent_source
            RunIntentStore.commit(self, current, expected_revision=self.revision)
        return RunIntentStore.commit(self, candidate, expected_revision=expected_revision)  # type: ignore[arg-type]


class RecordingFilesystemSourceAdapter(FilesystemSourceAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.observation_thread: int | None = None

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.observation_thread = get_ident()
        return super().observe(request)


def _workspace(source: object) -> tuple[ScatteringWorkspace, CountingStore, ScatteringCoordinator]:
    store = CountingStore()
    lifecycle = ScatteringCoordinator()
    return ScatteringWorkspace(intents=store, lifecycle=lifecycle, sources=source), store, lifecycle  # type: ignore[arg-type]


def test_workspace_uses_real_controls_and_one_cas_per_valid_edit(qapp: QtWidgets.QApplication) -> None:
    workspace, store, lifecycle = _workspace(ImmediateSource())
    choose_requests: list[object] = []
    workspace.sourceSelectionRequested.connect(choose_requests.append)
    try:
        shell = _shell(workspace)
        controls = shell.controls
        source_status = _source_status(workspace)
        assert type(controls) is ControlsPanelV2
        assert controls.profile is not None
        assert controls.profile.run_enabled is False
        assert (
            shell.run_controls.readinessLabel.text()
            == "Execution is unavailable"
        )
        assert lifecycle.phase.value == "idle"
        assert source_status.isHidden()
        assert not source_status._choose.isVisibleTo(workspace)
        controls.fieldValueChanged.emit(("Signal", "inp_type"), "Image Series")
        assert choose_requests == []
        assert store.commit_calls == 0

        controls.fieldValueChanged.emit(PROJECT_ROOT, "/project")
        assert store.commit_calls == 1
        assert store.snapshot().revision == 1
        assert store.snapshot().thaw().project_root == "/project"
        assert store.snapshot().thaw().save_path == (
            "/project/xdart_processed_data"
        )

        controls.fieldValueChanged.emit(PROJECT_ROOT, "/project")
        assert store.commit_calls == 1
        controls.fieldValueChanged.emit(PROJECT_ROOT, "/next-project")
        assert store.commit_calls == 2
        assert store.snapshot().thaw().save_path == (
            "/next-project/xdart_processed_data"
        )
        controls.fieldValueChanged.emit(SAVE_PATH, "/custom-output")
        controls.fieldValueChanged.emit(PROJECT_ROOT, "/third-project")
        assert store.commit_calls == 4
        assert store.snapshot().thaw().save_path == "/custom-output"
        controls.fieldDraftChanged.emit(GI_ORIENTATION, "4.5")
        assert store.commit_calls == 4
        assert "integer" in shell.scientific.status.text().lower()
        controls.fieldValueChanged.emit(GI_ORIENTATION, "4.5")
        assert store.commit_calls == 4
    finally:
        workspace.close_workspace()


def test_stale_cas_renders_returned_snapshot_without_retry(qapp: QtWidgets.QApplication) -> None:
    workspace, store, _ = _workspace(ImmediateSource())
    try:
        store.make_next_commit_stale = True
        shell = _shell(workspace)
        shell.controls.fieldValueChanged.emit(SAVE_PATH, "/wanted")
        assert store.commit_calls == 1
        assert store.snapshot().revision == 1
        assert store.snapshot().thaw().project_root == "/concurrent"
        assert store.snapshot().thaw().save_path != "/wanted"
        assert (
            shell.scientific.status.text()
            == "Edit superseded; review current value."
        )
    finally:
        workspace.close_workspace()


def test_source_selection_is_one_store_value_and_observes_off_gui_thread(
    qapp: QtWidgets.QApplication, tmp_path: Path
) -> None:
    direct = tmp_path / "direct.tif"
    direct.write_text("x")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "descendant.tif").write_text("x")
    source = RecordingFilesystemSourceAdapter()
    workspace, store, _ = _workspace(source)
    gui_thread = get_ident()
    try:
        source_status = _source_status(workspace)
        workspace.select_source(DirectorySourceSpec(tmp_path, recursive=True, suffixes=(".tif",)))
        assert store.commit_calls == 1
        assert _wait_until(qapp, lambda: source.observation_thread is not None)
        assert source.observation_thread != gui_thread
        # Selected plus immediate is the shared preview and Run boundary.
        assert _wait_until(qapp, lambda: "2 files" in source_status._header.text)
        assert source_status._header.text == "2 files (folder + 1 level) · Image Directory"
        assert source_status._header.detail.endswith(
            "Only the selected folder and immediate subfolders are processed. "
            "Deeper subfolders are outside the supported Run scope."
        )
        assert store.snapshot().thaw().source_spec == DirectorySourceSpec(
            tmp_path, recursive=True, suffixes=(".tif",)
        )
        assert not {
            "_source_spec",
            "_current_source",
            "_source_revision",
            "_source_epoch",
        } & set(vars(workspace))
        assert set(vars(source)) == {
            "_lock", "_inflight", "_cancelled", "_capture_epoch",
            "_capture_request", "_motor_knowledge", "observation_thread",
        }
    finally:
        workspace.close_workspace()


def test_foreign_callback_and_post_close_completion_are_inert(qapp: QtWidgets.QApplication) -> None:
    source = BlockingSource()
    workspace, store, lifecycle = _workspace(source)
    try:
        source_status = _source_status(workspace)
        workspace.select_source(SourceSpec("/selected.dat"))
        assert source.started.wait(1.0)
        request = source.last_request
        assert request is not None
        operation = workspace._observation
        assert operation is not None
        before = source_status._header
        foreign = Future()
        foreign.set_result(object())
        workspace._observationFinished.emit(object(), foreign)
        qapp.processEvents()
        assert source_status._header == before
        assert workspace._observation is operation
        workspace._observationFinished.emit(operation, foreign)
        qapp.processEvents()
        assert source_status._header == before
        assert workspace._observation is operation

        commits_before_close = store.commit_calls
        workspace.close_workspace()
        assert lifecycle.closed is True
        assert source.cancelled == [request.observation_id]
        assert workspace._observation_pool is None
        assert workspace._observation is None
        workspace.close_workspace()
        source.release.set()
        assert _wait_until(qapp, lambda: source.observation_thread is not None)
        qapp.processEvents()
        assert store.commit_calls == commits_before_close
    finally:
        source.release.set()
        workspace.close_workspace()


def test_unrelated_edit_accepts_existing_observation_without_restat(
    qapp: QtWidgets.QApplication,
) -> None:
    source = BlockingSource()
    workspace, store, _ = _workspace(source)
    try:
        shell = _shell(workspace)
        source_status = _source_status(workspace)
        workspace.select_source(SourceSpec("/selected.dat"))
        assert source.started.wait(1.0)
        request = source.last_request
        assert request is not None
        shell.controls.fieldValueChanged.emit(PROJECT_ROOT, "/outside")
        assert len(source.requests) == 1
        source.release.set()
        assert _wait_until(qapp, lambda: workspace._observation is None)
        assert source_status._header.text == "Single Image"
        assert source_status._header.ready is False
        assert request.intent_revision == 1
        assert store.snapshot().revision == 2
    finally:
        source.release.set()
        workspace.close_workspace()


@pytest.mark.parametrize(
    "source_type",
    [MalformedCompletionSource, RaisingCompletionSource, WrongEchoSource],
)
def test_current_bad_completion_is_consumed_as_unavailable(
    qapp: QtWidgets.QApplication,
    source_type: type[ImmediateSource],
) -> None:
    source = source_type()
    workspace, store, _ = _workspace(source)
    try:
        source_status = _source_status(workspace)
        workspace.select_source(SourceSpec("/selected.dat"))
        assert _wait_until(qapp, lambda: source.observation_thread is not None)
        assert _wait_until(qapp, lambda: workspace._observation is None)
        assert source_status._header.text == "unavailable"
        assert store.snapshot().revision == 1
    finally:
        workspace.close_workspace()


def test_deleted_page_completion_is_contained(qapp: QtWidgets.QApplication) -> None:
    store = RunIntentStore()
    workspace = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=ImmediateSource(),
    )
    page_ref = weakref.ref(workspace)
    workspace.close_workspace()
    workspace.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    qapp.processEvents()
    future = Future()
    future.set_result(object())
    operation = _ObservationOperation(
        SourceObservationRequest(1, 0, SourceSpec("/selected.dat")), future,
    )
    ScatteringWorkspace._deliver_observation(page_ref, operation, future)
    assert store.revision == 0


def test_replacement_cancels_queued_observation_before_it_starts(
    qapp: QtWidgets.QApplication,
) -> None:
    source = QueuedSource()
    workspace, _, _ = _workspace(source)
    try:
        workspace.select_source(SourceSpec("/a.dat"))
        assert source.first_started.wait(1.0)
        workspace.select_source(SourceSpec("/b.dat"))
        workspace.select_source(SourceSpec("/c.dat"))
        source.release_first.set()
        assert _wait_until(qapp, lambda: len(source.calls) == 2)
        assert source.calls == [1, 3]
        assert source.cancelled == [1]
    finally:
        source.release_first.set()
        workspace.close_workspace()


def test_source_changing_recapture_observes_only_returned_canonical_source(
    qapp: QtWidgets.QApplication,
) -> None:
    canonical = SourceSpec("/canonical.dat")
    store = SourceRecaptureStore(canonical)
    source = RecordingSource()
    workspace = ScatteringWorkspace(
        intents=store,
        lifecycle=ScatteringCoordinator(),
        sources=source,
    )
    try:
        workspace.select_source(SourceSpec("/requested.dat"))
        assert _wait_until(qapp, lambda: bool(source.requests))
        assert store.snapshot().thaw().source_spec == canonical
        assert [request.source for request in source.requests] == [canonical]
    finally:
        workspace.close_workspace()


def test_valid_draft_clears_transient_refusal_without_committing(
    qapp: QtWidgets.QApplication,
) -> None:
    workspace, store, _ = _workspace(ImmediateSource())
    try:
        shell = _shell(workspace)
        shell.controls.fieldDraftChanged.emit(GI_ORIENTATION, "4.5")
        assert shell.scientific.status.text()
        shell.controls.fieldDraftChanged.emit(GI_ORIENTATION, "4")
        assert shell.scientific.status.text() == ""
        assert store.commit_calls == 0
    finally:
        workspace.close_workspace()


def test_observation_values_are_independent_and_runtime_resolvable(tmp_path: Path) -> None:
    options = {"details": {"label": "one"}}
    source = SourceSpec(tmp_path / "sample.dat", options=options)
    request = SourceObservationRequest(1, 0, source)
    observation = SourceObservation(
        1,
        0,
        source,
        SourceObservationStatus.AVAILABLE,
        "sample.dat",
        True,
        False,
    )
    assert request.source is not source
    assert observation.source is not source
    options["details"]["label"] = "changed"
    assert request.source.options["details"]["label"] == "one"  # type: ignore[index]
    assert observation.source.options["details"]["label"] == "one"  # type: ignore[index]
    assert get_type_hints(SourceObservationRequest)
    assert get_type_hints(SourceObservation)


def test_adapter_lists_only_immediate_matching_children(tmp_path: Path) -> None:
    (tmp_path / "one.tif").write_text("x")
    (tmp_path / "two.txt").write_text("x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "hidden.tif").write_text("x")
    request = SourceObservationRequest(
        1,
        0,
        DirectorySourceSpec(tmp_path, recursive=True, suffixes=(".tif",)),
    )
    observed = FilesystemSourceAdapter().observe(request)
    assert observed.status is SourceObservationStatus.AVAILABLE
    assert observed.direct_child_count == 1
    assert observed.subdirectories_deferred is True


def test_page_adds_only_the_e1b_execution_path_without_legacy_hosts() -> None:
    source = Path(__file__).parents[3] / "src" / "xdart" / "gui" / "tabs" / "scattering" / "page.py"
    text = source.read_text()
    for forbidden in (
        ".freeze(",
        ".capture(",
        "OutputPort",
        "BrowseLoaderPort",
        "ProjectionPort",
        "staticWidget",
        "imageWrangler",
        "nexusWrangler",
        "ParameterTree",
    ):
        assert forbidden not in text
    assert "StartPipeline" in text
    assert "RunExecutorPort" in text
