"""Qt-free exact ownership checks for source selection and observation."""

from __future__ import annotations

from pathlib import Path
from threading import Event

from xrd_tools.core.scan import SourceSpec
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentRecaptureRequired,
    RunIntentStore,
)
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    image_series_spec,
    single_image_spec,
)

from xdart.gui.tabs.scattering.contracts import (
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.controls_inventory import (
    GI_MOTOR,
    SOURCE_DIRECTORY,
    SOURCE_TYPE,
)
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.source_selection import (
    SourceObservationWake,
    SourceRefreshEffect,
    SourceSelectionOwner,
    SourceStatusDirective,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase


class CountingStore(RunIntentStore):
    def __init__(self, initial: RunIntent) -> None:
        super().__init__(initial)
        self.commit_calls = 0
        self.recapture_next = False

    def commit(self, candidate: object, *, expected_revision: int):  # type: ignore[override]
        self.commit_calls += 1
        if self.recapture_next:
            self.recapture_next = False
            concurrent = self.snapshot().thaw()
            concurrent.project_root = "/concurrent"
            RunIntentStore.commit(
                self,
                concurrent,
                expected_revision=self.revision,
            )
        return RunIntentStore.commit(  # type: ignore[arg-type]
            self,
            candidate,
            expected_revision=expected_revision,
        )


class ImmediateSource:
    def __init__(self, choices: tuple[str, ...] | None = None) -> None:
        self.choices = choices
        self.requests: list[SourceObservationRequest] = []
        self.cancelled: list[int] = []
        self.knowledge: SourceObservation | None = None

    def observe(
        self,
        request: SourceObservationRequest,
    ) -> SourceObservation:
        self.requests.append(request)
        return SourceObservation(
            request.observation_id,
            request.intent_revision,
            request.source,
            SourceObservationStatus.AVAILABLE,
            "selected",
            True,
            type(request.source) is DirectorySourceSpec,
            gi_motor_choices=self.choices,
        )

    def preview_motors(
        self,
        request: SourceObservationRequest,
    ) -> SourceObservation:
        return self.observe(request)

    def cancel_observation(self, observation_id: int) -> None:
        self.cancelled.append(observation_id)

    def publish_motor_knowledge(
        self,
        observation: SourceObservation,
    ) -> None:
        self.knowledge = observation

    def project_motor_knowledge(
        self,
        source: object,
        candidate_fingerprint: str | None = None,
    ) -> SourceObservation | None:
        knowledge = self.knowledge
        if knowledge is None or knowledge.source != source:
            return None
        if (
            candidate_fingerprint is not None
            and knowledge.candidate_fingerprint != candidate_fingerprint
        ):
            return None
        return knowledge


def _owner(
    store: CountingStore,
    source: ImmediateSource,
) -> tuple[SourceSelectionOwner, list[SourceObservationWake], Event]:
    wakes: list[SourceObservationWake] = []
    delivered = Event()

    def deliver(wake: SourceObservationWake) -> None:
        wakes.append(wake)
        delivered.set()

    return SourceSelectionOwner(source, store, deliver), wakes, delivered


def _settle(
    owner: SourceSelectionOwner,
    wake: SourceObservationWake,
):
    return owner.consume(
        wake,
        active_identity=None,
        phase=RunPhase.IDLE,
        automatic_motor_permitted=True,
        terminal_progress=False,
    )


def test_observation_owns_one_automatic_motor_cas_and_one_refresh() -> None:
    selected = SourceSpec("/selected.nxs")
    store = CountingStore(RunIntent(source_spec=selected))
    source = ImmediateSource(("exposure", "chi", "th"))
    owner, wakes, delivered = _owner(store, source)
    try:
        checking = owner.request_observation(store.snapshot())
        assert checking.status is SourceStatusDirective.CHECKING
        assert checking.refresh is SourceRefreshEffect.NONE
        assert delivered.wait(1.0)

        transition = _settle(owner, wakes[-1])

        assert transition.status is SourceStatusDirective.OBSERVED
        assert transition.refresh is SourceRefreshEffect.CONTROLS
        assert transition.automatic_motor == "th"
        assert transition.intent is not None
        assert type(transition.intent.result) is IntentCommitAccepted
        assert store.commit_calls == 1
        assert store.snapshot().revision == 1
        assert store.snapshot().thaw().gi.incidence_motor == "th"
    finally:
        owner.finalize_close()


def test_begin_close_blocks_a_deferred_automatic_motor_cas() -> None:
    selected = SourceSpec("/selected.nxs")
    store = CountingStore(RunIntent(source_spec=selected))
    source = ImmediateSource(("exposure", "chi", "th"))
    owner, wakes, delivered = _owner(store, source)
    try:
        owner.request_observation(store.snapshot())
        assert delivered.wait(1.0)
        observed = owner.consume(
            wakes[-1],
            active_identity=None,
            phase=RunPhase.PREPARING,
            automatic_motor_permitted=False,
            terminal_progress=False,
        )
        assert observed.status is SourceStatusDirective.OBSERVED
        assert store.commit_calls == 0

        owner.begin_close()
        retry = owner.commit_deferred_motor_default(permitted=True)

        assert retry.refresh is SourceRefreshEffect.NONE
        assert store.commit_calls == 0
        assert store.snapshot().thaw().gi.incidence_motor == "Manual"
    finally:
        owner.finalize_close()


def test_automatic_motor_recapture_is_one_cas_without_retry() -> None:
    selected = SourceSpec("/selected.nxs")
    store = CountingStore(RunIntent(source_spec=selected))
    store.recapture_next = True
    source = ImmediateSource(("th",))
    owner, wakes, delivered = _owner(store, source)
    try:
        owner.request_observation(store.snapshot())
        assert delivered.wait(1.0)

        transition = _settle(owner, wakes[-1])

        assert transition.intent is not None
        assert type(transition.intent.result) is IntentRecaptureRequired
        assert transition.automatic_motor is None
        assert store.commit_calls == 1
        assert store.snapshot().revision == 1
        assert store.snapshot().thaw().project_root == "/concurrent"
        assert store.snapshot().thaw().gi.incidence_motor == "Manual"
    finally:
        owner.finalize_close()


def test_foreign_equal_wake_cannot_consume_the_owned_operation() -> None:
    selected = SourceSpec("/selected.nxs")
    store = CountingStore(RunIntent(source_spec=selected))
    owner, wakes, delivered = _owner(store, ImmediateSource())
    try:
        owner.request_observation(store.snapshot())
        assert delivered.wait(1.0)
        wake = wakes[-1]

        foreign = _settle(owner, SourceObservationWake(wake.token))
        assert foreign.refresh is SourceRefreshEffect.NONE
        assert owner.current_wake is wake

        accepted = _settle(owner, wake)
        assert accepted.status is SourceStatusDirective.OBSERVED
        assert owner.current_wake is None
    finally:
        owner.finalize_close()


def test_passive_refresh_never_crosses_live_run_identity() -> None:
    selected = DirectorySourceSpec(Path("/raw/live"))
    store = CountingStore(RunIntent(
        source_spec=selected,
        live_mode=True,
    ))
    source = ImmediateSource()
    owner, wakes, delivered = _owner(store, source)
    first = RunIdentity(1, "first")
    second = RunIdentity(2, "second")
    try:
        owner.set_launched_source(selected, first, live_mode=True)
        owner.queue_live_refresh(
            first,
            store.snapshot(),
            active_identity=first,
            phase=RunPhase.RUNNING,
        )
        assert delivered.wait(1.0)
        assert owner.observing
        owner.queue_live_refresh(
            first,
            store.snapshot(),
            active_identity=first,
            phase=RunPhase.RUNNING,
        )
        assert owner.pending_refresh is not None

        owner.set_launched_source(selected, second, live_mode=True)
        assert owner.pending_refresh is None
        stale = owner.consume(
            wakes[-1],
            active_identity=second,
            phase=RunPhase.RUNNING,
            automatic_motor_permitted=False,
            terminal_progress=False,
        )

        assert stale.refresh is SourceRefreshEffect.NONE
        assert len(source.requests) == 1
        assert not owner.observing
    finally:
        owner.finalize_close()


def test_automatic_motor_resets_in_same_directory_edit_cas() -> None:
    selected = DirectorySourceSpec(Path("/raw/first"))
    store = CountingStore(RunIntent(source_spec=selected))
    owner, wakes, delivered = _owner(store, ImmediateSource(("th",)))
    try:
        owner.request_observation(store.snapshot())
        assert delivered.wait(1.0)
        automatic = _settle(owner, wakes[-1])
        assert automatic.automatic_motor == "th"
        assert store.commit_calls == 1

        edited = owner.edit(SOURCE_DIRECTORY, "/raw/replacement")

        assert edited.intent is not None
        assert type(edited.intent.result) is IntentCommitAccepted
        assert store.commit_calls == 2
        intent = store.snapshot().thaw()
        assert type(intent.source_spec) is DirectorySourceSpec
        assert intent.source_spec.root == Path("/raw/replacement")
        assert intent.gi.incidence_motor == "Manual"
    finally:
        owner.finalize_close()


def test_explicit_motor_and_mode_history_are_owner_commits(
    tmp_path: Path,
) -> None:
    single = single_image_spec(tmp_path / "single.tif")
    store = CountingStore(RunIntent(source_spec=single))
    owner, _, _ = _owner(store, ImmediateSource())
    try:
        motor = owner.edit(GI_MOTOR, "th")
        assert motor.intent is not None
        assert type(motor.intent.result) is IntentCommitAccepted

        series_mode = owner.edit(SOURCE_TYPE, "Image Series")
        assert series_mode.intent is not None
        assert store.snapshot().thaw().source_spec is None
        assert owner.mode == "Image Series"

        series = image_series_spec(tmp_path / "series_0001.tif")
        selected = owner.select(series)
        assert selected.intent is not None
        assert store.snapshot().thaw().source_spec == series
        assert store.snapshot().thaw().gi.incidence_motor == "th"

        restored = owner.edit(SOURCE_TYPE, "Single Image")
        assert restored.intent is not None
        assert store.snapshot().thaw().source_spec == single
        assert owner.mode == "Single Image"
        assert store.commit_calls == 4
    finally:
        owner.finalize_close()


def test_page_has_one_source_owner_and_no_source_executor_or_port_calls() -> None:
    root = Path(__file__).parents[3]
    page = (
        root / "src/xdart/gui/tabs/scattering/page.py"
    ).read_text()
    owner = (
        root / "src/xdart/gui/tabs/scattering/source_selection.py"
    ).read_text()

    for retired in (
        "self._sources",
        "self._observation_pool",
        "self._pending_source_refresh",
        "self._live_source_refresh_source",
        "self._source_observation",
        "self._gi_manual_source",
        "self._gi_auto_motor",
        "self._observation_token",
        "self._source_mode",
        "self._source_history",
    ):
        assert retired not in page
    assert page.count("ThreadPoolExecutor(max_workers=1)") == 1
    assert owner.count("ThreadPoolExecutor(max_workers=1)") == 1
    assert owner.count("self._intents.commit(") == 1
    assert "self._sources.observe" in owner
    assert "self._sources.preview_motors" in owner
