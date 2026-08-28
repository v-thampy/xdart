from __future__ import annotations

from pathlib import Path
from threading import Event, Lock
import time

import pytest
from pyqtgraph.Qt import QtWidgets

from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec
from xdart.gui.tabs.scattering.contracts import (
    SourceCapture,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.controls_projection import PROJECT_ROOT
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import (
    StandardEventKind,
    StandardRunEvent,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    ExecutorClosed,
    RunIdentity,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.shell_values import (
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from tests.xdart.scattering._admission import ImmediateAdmission


@pytest.fixture
def qapp() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait(
    app: QtWidgets.QApplication,
    predicate,
    *,
    timeout: float = 3.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("page operation did not settle")


class _Executor(ImmediateAdmission):
    def __init__(self) -> None:
        self.events: list[object] = []
        self.start_calls = 0

    def start(self, _configuration, _source, identity, _admission):
        self.start_calls += 1
        return ExecutorAccepted(identity)

    def stop(self, _identity) -> None:
        return None

    def close(self, identity):
        return ExecutorClosed(identity, CleanupStatus.CLEANED)

    def pause(self, _identity) -> None:
        return None

    def resume(self, _identity) -> None:
        return None

    def drain_events(self):
        events, self.events = tuple(self.events), []
        return events


class _StatOnlySources:
    """A passive source double that never opens TIFF content."""

    def __init__(self) -> None:
        self.observe_requests: list[SourceObservationRequest] = []
        self.preview_requests: list[SourceObservationRequest] = []
        self.cancelled: list[int] = []
        self.knowledge: SourceObservation | None = None
        self.capture_epoch = 0

    def capture(self, source, request_id):
        self.capture_epoch += 1
        return SourceCapture(request_id, self.capture_epoch, source)

    def cancel(self, _request_id) -> None:
        return None

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        self.observe_requests.append(request)
        source = request.source
        assert type(source) is DirectorySourceSpec
        members = tuple(sorted(source.root.glob("*.tif")))
        fingerprint = "|".join(
            f"{path.name}:{path.stat().st_size}" for path in members
        )
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            source,
            SourceObservationStatus.AVAILABLE,
            source.root.name,
            source.root.is_dir(),
            True,
            direct_child_count=len(members),
            candidate_fingerprint=fingerprint,
        )

    def preview_motors(
        self, request: SourceObservationRequest
    ) -> SourceObservation:
        self.preview_requests.append(request)
        raise AssertionError("a Live passive refresh opened source content")

    def cancel_observation(self, observation_id: int) -> None:
        self.cancelled.append(observation_id)

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


class _BlockingSources(_StatOnlySources):
    def __init__(self, *, initial_count: int = 0) -> None:
        super().__init__()
        self.initial_count = initial_count
        self.refresh_count = initial_count
        self.refresh_fingerprint = "initial" if initial_count else ""
        self.block_refresh = False
        self.refresh_started = Event()
        self.refresh_release = Event()
        self.block_preview = False
        self.preview_started = Event()
        self.preview_release = Event()
        self._observe_lock = Lock()

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        with self._observe_lock:
            ordinal = len(self.observe_requests)
            self.observe_requests.append(request)
        if ordinal > 0 and self.block_refresh:
            self.refresh_started.set()
            assert self.refresh_release.wait(3.0)
        count = self.initial_count if ordinal == 0 else self.refresh_count
        fingerprint = (
            ("initial" if self.initial_count else "")
            if ordinal == 0
            else self.refresh_fingerprint
        )
        source = request.source
        assert type(source) is DirectorySourceSpec
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            source,
            SourceObservationStatus.AVAILABLE,
            source.root.name,
            True,
            True,
            direct_child_count=count,
            candidate_fingerprint=fingerprint,
        )

    def preview_motors(
        self, request: SourceObservationRequest
    ) -> SourceObservation:
        self.preview_requests.append(request)
        if self.block_preview:
            self.preview_started.set()
            assert self.preview_release.wait(3.0)
        source = request.source
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            source,
            SourceObservationStatus.AVAILABLE,
            source.root.name,
            True,
            True,
            direct_child_count=self.initial_count,
            gi_motor_choices=("theta",),
            candidate_fingerprint="initial",
        )


def _page(
    root: Path,
    sources: _StatOnlySources,
    *,
    live: bool = True,
) -> tuple[ScatteringWorkspace, _Executor]:
    root.mkdir(parents=True, exist_ok=True)
    source = DirectorySourceSpec(root, suffixes=(".tif",))
    executor = _Executor()
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=source,
                live_mode=live,
                poni_file=str(root.parent / "calibration.poni"),
                save_path=str(root.parent / "processed"),
                output_mode="Overwrite",
            )
        ),
        lifecycle=ScatteringCoordinator(),
        sources=sources,
        executor=executor,
    )
    return page, executor


def _start(
    page: ScatteringWorkspace,
    app: QtWidgets.QApplication,
) -> RunIdentity:
    _wait(app, lambda: not page._source_selection.observing)
    page._shell.commandRequested.emit(
        ShellCommand(ShellCommandKind.RUN_ACTION)
    )
    page._drain_executor()
    identity = page._lifecycle.active_run_identity
    assert identity is not None
    assert page._lifecycle.phase is RunPhase.RUNNING
    return identity


def _discovery(
    identity: RunIdentity,
    *,
    pending: int,
    discovered: int,
) -> StandardRunEvent:
    return StandardRunEvent(
        identity,
        StandardEventKind.DISCOVERY,
        files_pending=pending,
        files_discovered=discovered,
    )


def _dispose(
    page: ScatteringWorkspace,
    app: QtWidgets.QApplication,
) -> None:
    page.close_workspace()
    page.deleteLater()
    app.processEvents()


def test_live_discovery_passively_refreshes_source_header_without_preview(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    sources = _StatOnlySources()
    page, executor = _page(tmp_path / "raw", sources)
    try:
        identity = _start(page, qapp)
        assert page._source_status._header.text == "0 files · Image Directory"
        assert page._source_status._header.ready is False
        assert page._shell.controls.source_card.valid_marker.isHidden()

        partial = tmp_path / "raw" / "partial_0001.tif"
        partial.write_bytes(b"x" * 4096)
        executor.events.append(
            _discovery(identity, pending=9, discovered=9)
        )
        page._drain_executor()

        _wait(
            qapp,
            lambda: not page._source_selection.observing
            and page._source_status._header.text
            == "1 file · Image Directory",
        )
        assert partial.stat().st_size == 4096
        assert page._source_status._header.ready is True
        assert not page._shell.controls.source_card.valid_marker.isHidden()
        assert len(sources.observe_requests) == 2
        assert sources.preview_requests == []
    finally:
        _dispose(page, qapp)


def test_discoveries_during_preview_coalesce_to_newest_passive_request(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    sources = _BlockingSources(initial_count=1)
    sources.block_preview = True
    page, executor = _page(tmp_path / "raw", sources)
    try:
        _wait(qapp, sources.preview_started.is_set)
        page._shell.commandRequested.emit(
            ShellCommand(ShellCommandKind.RUN_ACTION)
        )
        page._drain_executor()
        identity = page._lifecycle.active_run_identity
        assert identity is not None

        for discovered in (2, 3, 4):
            sources.refresh_count = discovered
            executor.events.append(
                _discovery(
                    identity,
                    pending=discovered,
                    discovered=discovered,
                )
            )
            page._drain_executor()

        sources.refresh_count = 4
        sources.refresh_fingerprint = "initial"
        sources.preview_release.set()
        _wait(
            qapp,
            lambda: not page._source_selection.observing
            and page._source_status._header.text
            == "4 files · Image Directory",
        )

        assert [item.observation_id for item in sources.observe_requests] == [
            1,
            4,
        ]
        assert [item.observation_id for item in sources.preview_requests] == [1]
        assert page._source_selection.observation is not None
        assert page._source_selection.observation.gi_motor_choices == (
            "theta",
        )

        sources.refresh_count = 5
        sources.refresh_fingerprint = "changed"
        executor.events.append(
            _discovery(identity, pending=5, discovered=5)
        )
        page._drain_executor()
        _wait(
            qapp,
            lambda: not page._source_selection.observing
            and page._source_status._header.text
            == "5 files · Image Directory",
        )
        assert [
            item.observation_id for item in sources.observe_requests
        ] == [1, 4, 5]
        assert [item.observation_id for item in sources.preview_requests] == [1]
        assert page._source_selection.observation is not None
        assert page._source_selection.observation.gi_motor_choices is None
    finally:
        sources.preview_release.set()
        _dispose(page, qapp)


def test_discoveries_during_passive_refresh_launch_only_newest_successor(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    sources = _BlockingSources()
    page, executor = _page(tmp_path / "raw", sources)
    try:
        identity = _start(page, qapp)
        sources.block_refresh = True
        sources.refresh_fingerprint = "burst"
        executor.events.append(
            _discovery(identity, pending=1, discovered=1)
        )
        page._drain_executor()
        assert sources.refresh_started.wait(3.0)

        for discovered in (2, 3, 4):
            sources.refresh_count = discovered
            executor.events.append(
                _discovery(
                    identity,
                    pending=discovered,
                    discovered=discovered,
                )
            )
            page._drain_executor()

        sources.refresh_release.set()
        _wait(
            qapp,
            lambda: not page._source_selection.observing
            and page._source_status._header.text
            == "4 files · Image Directory",
        )

        assert [item.observation_id for item in sources.observe_requests] == [
            1,
            2,
            5,
        ]
        assert sources.preview_requests == []
    finally:
        sources.refresh_release.set()
        _dispose(page, qapp)


def test_terminal_during_passive_refresh_cannot_repaint_or_launch_pending(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    sources = _BlockingSources()
    page, executor = _page(tmp_path / "raw", sources)
    try:
        identity = _start(page, qapp)
        sources.block_refresh = True
        sources.refresh_count = 2
        sources.refresh_fingerprint = "terminal-stale"
        executor.events.append(
            _discovery(identity, pending=1, discovered=1)
        )
        page._drain_executor()
        assert sources.refresh_started.wait(3.0)
        executor.events.append(
            _discovery(identity, pending=2, discovered=2)
        )
        page._drain_executor()

        executor.events.append(
            StandardRunEvent(
                identity,
                StandardEventKind.FINISHED,
                cleanup_status=CleanupStatus.CLEANED,
            )
        )
        page._drain_executor()
        assert page._lifecycle.phase is RunPhase.IDLE
        sources.refresh_release.set()
        _wait(qapp, lambda: not page._source_selection.observing)

        assert [item.observation_id for item in sources.observe_requests] == [
            1,
            2,
        ]
        assert page._source_status._header.text == "0 files · Image Directory"
        assert page._source_status._header.ready is False
        assert page._source_selection.pending_refresh is None
    finally:
        sources.refresh_release.set()
        _dispose(page, qapp)


def test_unrelated_intent_edit_does_not_drop_blocked_passive_refresh(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    sources = _BlockingSources()
    page, executor = _page(tmp_path / "raw", sources)
    try:
        identity = _start(page, qapp)
        sources.block_refresh = True
        sources.refresh_count = 1
        sources.refresh_fingerprint = "one-arrival"
        executor.events.append(
            _discovery(identity, pending=1, discovered=1)
        )
        page._drain_executor()
        assert sources.refresh_started.wait(3.0)

        prior_revision = page._intents.snapshot().revision
        page._shell.controls.fieldValueChanged.emit(
            PROJECT_ROOT,
            str(tmp_path / "unrelated-project"),
        )
        assert page._intents.snapshot().revision == prior_revision + 1

        sources.refresh_release.set()
        _wait(
            qapp,
            lambda: not page._source_selection.observing
            and page._source_status._header.text
            == "1 file · Image Directory",
        )
        assert [
            request.observation_id for request in sources.observe_requests
        ] == [1, 2]
        assert page._source_status._header.ready is True
    finally:
        sources.refresh_release.set()
        _dispose(page, qapp)


def test_foreign_non_live_stale_and_post_close_discoveries_do_not_refresh(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    live_sources = _BlockingSources()
    live_page, live_executor = _page(tmp_path / "live", live_sources)
    non_live_sources = _BlockingSources()
    non_live_page, non_live_executor = _page(
        tmp_path / "non-live", non_live_sources, live=False
    )
    try:
        live_identity = _start(live_page, qapp)
        non_live_identity = _start(non_live_page, qapp)

        foreign = RunIdentity(
            live_identity.generation + 100,
            f"foreign-{live_identity.fingerprint}",
        )
        live_executor.events.append(
            _discovery(foreign, pending=1, discovered=1)
        )
        live_page._drain_executor()
        non_live_executor.events.append(
            _discovery(non_live_identity, pending=1, discovered=1)
        )
        non_live_page._drain_executor()
        qapp.processEvents()
        assert len(live_sources.observe_requests) == 1
        assert len(non_live_sources.observe_requests) == 1

        live_executor.events.append(
            StandardRunEvent(
                live_identity,
                StandardEventKind.FINISHED,
                cleanup_status=CleanupStatus.CLEANED,
            )
        )
        live_page._drain_executor()
        assert live_page._lifecycle.phase is RunPhase.IDLE
        live_executor.events.append(
            _discovery(live_identity, pending=1, discovered=1)
        )
        live_page._drain_executor()
        assert len(live_sources.observe_requests) == 1

        live_page.close_workspace()
        live_executor.events.append(
            _discovery(live_identity, pending=1, discovered=1)
        )
        live_page._drain_executor()
        assert len(live_sources.observe_requests) == 1
    finally:
        _dispose(live_page, qapp)
        _dispose(non_live_page, qapp)


def test_source_change_clears_blocked_refresh_and_queued_successor(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    sources = _BlockingSources()
    page, executor = _page(tmp_path / "first", sources)
    try:
        identity = _start(page, qapp)
        sources.block_refresh = True
        executor.events.append(
            _discovery(identity, pending=1, discovered=1)
        )
        page._drain_executor()
        assert sources.refresh_started.wait(3.0)
        executor.events.append(
            _discovery(identity, pending=2, discovered=2)
        )
        page._drain_executor()

        second = DirectorySourceSpec(
            tmp_path / "second", suffixes=(".tif",)
        )
        second.root.mkdir(parents=True)
        page.select_source(second)
        sources.block_refresh = False
        sources.refresh_release.set()
        _wait(
            qapp,
            lambda: not page._source_selection.observing
            and page._source_selection.observation is not None
            and page._source_selection.observation.source == second,
        )

        old_source = DirectorySourceSpec(
            tmp_path / "first", suffixes=(".tif",)
        )
        assert [
            request.source for request in sources.observe_requests
        ].count(old_source) == 2
        assert [
            request.source for request in sources.observe_requests
        ].count(second) == 1
    finally:
        sources.refresh_release.set()
        _dispose(page, qapp)


def test_close_clears_blocked_refresh_and_queued_successor(
    qapp: QtWidgets.QApplication,
    tmp_path: Path,
) -> None:
    sources = _BlockingSources()
    page, executor = _page(tmp_path / "raw", sources)
    try:
        identity = _start(page, qapp)
        sources.block_refresh = True
        executor.events.append(
            _discovery(identity, pending=1, discovered=1)
        )
        page._drain_executor()
        assert sources.refresh_started.wait(3.0)
        executor.events.append(
            _discovery(identity, pending=2, discovered=2)
        )
        page._drain_executor()

        page.close_workspace()
        sources.refresh_release.set()
        _wait(qapp, lambda: not page._source_selection.pool_open)
        qapp.processEvents()

        assert len(sources.observe_requests) == 2
        assert page._source_status._header.text == "0 files · Image Directory"
        assert page._source_status._header.ready is False
        assert page._source_selection.pending_refresh is None
        assert page._source_selection.live_source is None
    finally:
        sources.refresh_release.set()
        _dispose(page, qapp)
