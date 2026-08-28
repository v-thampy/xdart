"""Qt-free ownership for source selection and passive observation.

The workspace page remains the Qt/status renderer and the intent store remains
the canonical editable-run owner.  This object owns the source-specific state
that must not be split across those two boundaries: editor mode/history,
observation execution/currentness, Live refresh coalescing, and provenance for
automatic GI motor defaults.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.gi_motor import pick_default_gi_motor
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    IntentRecaptureRequired,
    RunIntentSnapshot,
    RunIntentStore,
)
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    image_series_spec,
    single_image_spec,
)

from .contracts import (
    SourceObservation,
    SourceObservationRequest,
    SourcePort,
    SourceSelection,
)
from .controls_projection import (
    EditNoChange,
    EditRefusal,
    EditResult,
    GI_MOTOR,
    SOURCE_FILE,
    SOURCE_TYPE,
    reduce_control_edit,
    reduce_source_selection,
    source_mode,
)
from .controls_inventory import SOURCE_EDIT_PATHS
from .events import RunIdentity
from .state_machine import RunPhase


_NO_DELIBERATE_MANUAL = object()
_NO_AUTOMATIC_GI_MOTOR = object()
_LIVE_SOURCE_REFRESH_PHASES = frozenset({
    RunPhase.RUNNING,
    RunPhase.PAUSING,
    RunPhase.PAUSED,
    RunPhase.RESUMING,
    RunPhase.STOPPING,
})


class SourceRefreshEffect(Enum):
    """Smallest shell repaint required by one source transition."""

    NONE = "none"
    CONTROLS = "controls"


class SourceStatusDirective(Enum):
    """Independent source-card action for one transition."""

    UNCHANGED = "unchanged"
    NO_SOURCE = "no_source"
    CHECKING = "checking"
    UNAVAILABLE = "unavailable"
    OBSERVED = "observed"


@dataclass(frozen=True, slots=True)
class SourceObservationWake:
    """Opaque exact wake token delivered to the Qt event bridge."""

    token: int

    def __post_init__(self) -> None:
        if type(self.token) is not int or self.token <= 0:
            raise ValueError("source observation wake token is invalid")


@dataclass(frozen=True, slots=True)
class SourceIntentReceipt:
    """One source-owner CAS result with its exact prior snapshot."""

    before: RunIntentSnapshot
    result: IntentCommitAccepted | IntentRecaptureRequired

    def __post_init__(self) -> None:
        if (
            type(self.before) is not RunIntentSnapshot
            or type(self.result)
            not in {IntentCommitAccepted, IntentRecaptureRequired}
        ):
            raise ValueError("source intent receipt is invalid")


@dataclass(frozen=True, slots=True)
class _SourceObservationOperation:
    """Owner-private observation future and launch provenance."""

    request: SourceObservationRequest
    future: Future[object]
    wake: SourceObservationWake
    preview: bool = False
    candidate_fingerprint: str = ""
    passive_refresh: bool = False
    refresh_identity: RunIdentity | None = None


@dataclass(frozen=True, slots=True)
class _PendingLiveRefresh:
    """Latest-only passive request bound to one exact Live run."""

    identity: RunIdentity
    request: SourceObservationRequest


@dataclass(frozen=True, slots=True)
class SourceSelectionTransition:
    """Detached result for the page's status/controls composition."""

    refresh: SourceRefreshEffect
    status: SourceStatusDirective = SourceStatusDirective.UNCHANGED
    observation: SourceObservation | None = None
    automatic_motor: str | None = None
    reset_terminal_progress: bool = False
    intent: SourceIntentReceipt | None = None
    notice: str | None = None
    preserve_terminal: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.refresh) is not SourceRefreshEffect
            or type(self.status) is not SourceStatusDirective
            or (
                self.observation is not None
                and type(self.observation) is not SourceObservation
            )
            or (
                (self.status is SourceStatusDirective.OBSERVED)
                != (self.observation is not None)
            )
            or (
                self.automatic_motor is not None
                and (
                    type(self.automatic_motor) is not str
                    or not self.automatic_motor
                    or self.observation is None
                )
            )
            or type(self.reset_terminal_progress) is not bool
            or (
                self.intent is not None
                and type(self.intent) is not SourceIntentReceipt
            )
            or (self.notice is not None and type(self.notice) is not str)
            or type(self.preserve_terminal) is not bool
        ):
            raise ValueError("source selection transition is invalid")


Delivery = Callable[[SourceObservationWake], None]


class SourceSelectionOwner:
    """Sole owner of source editor and observation lifecycle state."""

    def __init__(
        self,
        sources: SourcePort,
        intents: RunIntentStore,
        deliver: Delivery,
    ) -> None:
        if type(intents) is not RunIntentStore and not isinstance(
            intents, RunIntentStore
        ):
            raise TypeError("intents must be RunIntentStore")
        if not callable(deliver):
            raise TypeError("deliver must be callable")
        initial = intents.snapshot()
        source = initial.thaw().source_spec
        self._sources = sources
        self._intents = intents
        self._deliver = deliver
        self._pool: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=1)
        self._operation: _SourceObservationOperation | None = None
        self._pending_refresh: _PendingLiveRefresh | None = None
        self._live_source: DirectorySourceSpec | None = None
        self._live_identity: RunIdentity | None = None
        self._current_observation: SourceObservation | None = None
        self._deliberate_manual_source: object = _NO_DELIBERATE_MANUAL
        self._automatic_motor: object = _NO_AUTOMATIC_GI_MOTOR
        self._token = 0
        self._mode = source_mode(source)
        self._history: dict[str, SourceSelection] = {}
        if source is not None:
            self._history[self._mode] = source
        self._closing = False
        self._closed = False

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def history(self) -> dict[str, SourceSelection]:
        return dict(self._history)

    @property
    def observation(self) -> SourceObservation | None:
        return self._current_observation

    @property
    def observing(self) -> bool:
        return self._operation is not None

    @property
    def current_wake(self) -> SourceObservationWake | None:
        operation = self._operation
        return None if operation is None else operation.wake

    @property
    def pending_refresh(self) -> SourceObservationRequest | None:
        pending = self._pending_refresh
        return None if pending is None else pending.request

    @property
    def live_source(self) -> DirectorySourceSpec | None:
        return self._live_source

    @property
    def pool_open(self) -> bool:
        return self._pool is not None

    @property
    def closed(self) -> bool:
        return self._closed

    def project_observation(
        self,
        snapshot: RunIntentSnapshot,
    ) -> SourceObservation | None:
        """Project current or retained exact source knowledge for controls."""

        if self._closing or self._closed:
            return None
        source = snapshot.thaw().source_spec
        if source is None:
            return None
        current = self._current_observation
        if current is not None and current.source == source:
            return current
        knowledge = self._sources.project_motor_knowledge(source, None)
        return knowledge if type(knowledge) is SourceObservation else None

    @staticmethod
    def owns_edit(path: object) -> bool:
        """Return whether one complete controls value belongs to this owner."""

        return bool(
            type(path) is tuple
            and (
                path in {SOURCE_TYPE, SOURCE_FILE, GI_MOTOR}
                or path in SOURCE_EDIT_PATHS
            )
        )

    def _commit(
        self,
        candidate: object,
        *,
        expected_revision: int,
    ) -> IntentCommitAccepted | IntentRecaptureRequired:
        """Keep every source-owned intent CAS at one auditable callsite."""

        return self._intents.commit(  # type: ignore[arg-type]
            candidate,
            expected_revision=expected_revision,
        )

    @staticmethod
    def _typed_file_source(
        current: object,
        mode: str,
        value: object,
    ) -> SourceSelection | EditRefusal:
        if type(value) is not str or not value.strip():
            return EditRefusal("Choose an image file.")
        metadata_format: str | None = "auto"
        if type(current) is SourceSpec:
            candidate = current.options.get("metadata_format", "auto")
            if candidate is None or type(candidate) is str:
                metadata_format = candidate
        try:
            if mode == "Image Series":
                return image_series_spec(
                    value,
                    metadata_format=metadata_format,
                )
            if mode == "Single Image":
                return single_image_spec(
                    value,
                    metadata_format=metadata_format,
                )
        except (OSError, ValueError) as error:
            return EditRefusal(str(error))
        return EditRefusal(
            "Choose an image source before editing its file."
        )

    @staticmethod
    def _local_transition(
        *,
        notice: str,
    ) -> SourceSelectionTransition:
        return SourceSelectionTransition(
            refresh=SourceRefreshEffect.CONTROLS,
            notice=notice,
        )

    @staticmethod
    def _commit_transition(
        before: RunIntentSnapshot,
        result: IntentCommitAccepted | IntentRecaptureRequired,
        *,
        automatic_motor: str | None = None,
        preserve_terminal: bool = False,
    ) -> SourceSelectionTransition:
        return SourceSelectionTransition(
            refresh=SourceRefreshEffect.CONTROLS,
            automatic_motor=automatic_motor,
            intent=SourceIntentReceipt(before, result),
            notice=(
                ""
                if type(result) is IntentCommitAccepted
                else "Edit superseded; review current value."
            ),
            preserve_terminal=preserve_terminal,
        )

    def select(self, source: SourceSelection) -> SourceSelectionTransition:
        """Commit one complete source selection through the owner CAS."""

        if self._closing or self._closed:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        before = self._intents.snapshot()
        reduced = reduce_source_selection(
            before,
            source,
            reset_auto_gi_motor=self._automatic_motor_matches(before),
        )
        if isinstance(reduced, EditRefusal):
            return self._local_transition(notice=reduced.reason)
        if isinstance(reduced, EditNoChange):
            return self._local_transition(notice="")
        result = self._commit(
            reduced,
            expected_revision=before.revision,
        )
        self.remember_current(result.snapshot)
        return self._commit_transition(before, result)

    def edit(
        self,
        path: object,
        value: object,
    ) -> SourceSelectionTransition:
        """Validate and commit one source-owned controls edit."""

        if self._closing or self._closed:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        if not self.owns_edit(path):
            return self._local_transition(notice="Unknown source edit.")
        before = self._intents.snapshot()
        desired_mode: str | None = None
        if path == SOURCE_TYPE:
            if type(value) is not str or value not in {
                "Image Series",
                "Image Directory",
                "Single Image",
            }:
                return self._local_transition(
                    notice="Unknown source type."
                )
            desired_mode = value
            if desired_mode == self._mode:
                return self._local_transition(notice="")
            reduced = self._source_for_mode(before, desired_mode)
        elif path == SOURCE_FILE:
            source = self._typed_file_source(
                before.thaw().source_spec,
                self._mode,
                value,
            )
            if isinstance(source, EditRefusal):
                return self._local_transition(notice=source.reason)
            reduced = reduce_source_selection(
                before,
                source,
                reset_auto_gi_motor=self._automatic_motor_matches(before),
            )
        else:
            reduced = reduce_control_edit(
                before,
                path,  # type: ignore[arg-type]
                value,
                reset_auto_gi_motor=(
                    type(path) is tuple
                    and path in SOURCE_EDIT_PATHS
                    and self._automatic_motor_matches(before)
                ),
            )
        if isinstance(reduced, EditRefusal):
            return self._local_transition(notice=reduced.reason)
        if isinstance(reduced, EditNoChange):
            if path == GI_MOTOR:
                self._record_motor_edit(
                    before,
                    value,
                    automatic=False,
                )
            return self._local_transition(notice="")
        result = self._commit(
            reduced,
            expected_revision=before.revision,
        )
        accepted = type(result) is IntentCommitAccepted
        if desired_mode is not None:
            self._accept_mode_result(
                desired_mode,
                result.snapshot,
                accepted=accepted,
            )
        else:
            self.remember_current(result.snapshot)
        if path == GI_MOTOR and accepted:
            self._record_motor_edit(
                before,
                value,
                automatic=False,
            )
        return self._commit_transition(before, result)

    def _source_for_mode(
        self,
        snapshot: RunIntentSnapshot,
        desired_mode: str,
    ) -> EditResult:
        if desired_mode == self._mode:
            return EditNoChange()
        current = snapshot.thaw().source_spec
        if current is not None:
            self._history[source_mode(current)] = current
        candidate = snapshot.thaw()
        candidate.source_spec = self._history.get(desired_mode)
        if self._automatic_motor_matches(snapshot):
            candidate.gi.incidence_motor = "Manual"
        return candidate

    def _accept_mode_result(
        self,
        desired_mode: str,
        current: RunIntentSnapshot,
        *,
        accepted: bool,
    ) -> None:
        source = current.thaw().source_spec
        self._mode = desired_mode if accepted else source_mode(source)
        if source is not None:
            self._history[source_mode(source)] = source

    def remember_current(self, snapshot: RunIntentSnapshot) -> None:
        if self._closing or self._closed:
            return
        source = snapshot.thaw().source_spec
        if source is None:
            return
        self._mode = source_mode(source)
        self._history[self._mode] = source

    def _automatic_motor_matches(self, snapshot: RunIntentSnapshot) -> bool:
        marker = self._automatic_motor
        if marker is _NO_AUTOMATIC_GI_MOTOR:
            return False
        intent = snapshot.thaw()
        return marker == (intent.source_spec, intent.gi.incidence_motor)

    def _record_motor_edit(
        self,
        snapshot: RunIntentSnapshot,
        value: object,
        *,
        automatic: bool,
    ) -> None:
        source = snapshot.thaw().source_spec
        self._deliberate_manual_source = (
            source
            if not automatic and value == "Manual"
            else _NO_DELIBERATE_MANUAL
        )
        self._automatic_motor = (
            (source, value)
            if automatic
            else _NO_AUTOMATIC_GI_MOTOR
        )

    def _automatic_motor_default(
        self,
        snapshot: RunIntentSnapshot,
        *,
        permitted: bool,
    ) -> str | None:
        observation = self._current_observation
        intent = snapshot.thaw()
        choices = (
            observation.gi_motor_choices
            if observation is not None
            and observation.source == intent.source_spec
            else None
        )
        if (
            not permitted
            or not choices
            or intent.gi.incidence_motor != "Manual"
            or (
                self._deliberate_manual_source is not _NO_DELIBERATE_MANUAL
                and self._deliberate_manual_source == intent.source_spec
            )
        ):
            return None
        preferred = pick_default_gi_motor(choices)
        return None if preferred == "Manual" else preferred

    def _commit_automatic_motor_default(
        self,
        *,
        permitted: bool,
    ) -> tuple[str | None, SourceIntentReceipt | None]:
        """Commit one optional observation-owned default in the exact CAS."""

        prior = self._intents.snapshot()
        preferred = self._automatic_motor_default(
            prior,
            permitted=permitted,
        )
        if preferred is None:
            return None, None
        candidate = prior.thaw()
        candidate.gi.incidence_motor = preferred
        result = self._commit(
            candidate,
            expected_revision=prior.revision,
        )
        if type(result) is IntentCommitAccepted:
            self._record_motor_edit(
                prior,
                preferred,
                automatic=True,
            )
            return preferred, SourceIntentReceipt(prior, result)
        return None, SourceIntentReceipt(prior, result)

    def commit_deferred_motor_default(
        self,
        *,
        permitted: bool,
    ) -> SourceSelectionTransition:
        if self._closing or self._closed:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        automatic_motor, receipt = (
            self._commit_automatic_motor_default(permitted=permitted)
        )
        if receipt is None:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        return SourceSelectionTransition(
            refresh=SourceRefreshEffect.CONTROLS,
            status=SourceStatusDirective.OBSERVED,
            observation=self._current_observation,
            automatic_motor=automatic_motor,
            intent=receipt,
            preserve_terminal=True,
        )

    def request_observation(
        self,
        snapshot: RunIntentSnapshot,
    ) -> SourceSelectionTransition:
        if self._closing or self._closed:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        source = snapshot.thaw().source_spec
        if source is None:
            self._current_observation = None
            return SourceSelectionTransition(
                SourceRefreshEffect.NONE,
                SourceStatusDirective.NO_SOURCE,
            )
        if self._pool is None or self._closing:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        self.cancel_observation()
        self._token += 1
        if (
            self._current_observation is not None
            and self._current_observation.source != source
        ):
            self._current_observation = None
        request = SourceObservationRequest(
            self._token, snapshot.revision, source
        )
        self._submit(request)
        return SourceSelectionTransition(
            SourceRefreshEffect.NONE,
            SourceStatusDirective.CHECKING,
        )

    def _submit(
        self,
        request: SourceObservationRequest,
        *,
        preview: bool = False,
        candidate_fingerprint: str = "",
        passive_refresh: bool = False,
        refresh_identity: RunIdentity | None = None,
    ) -> _SourceObservationOperation | None:
        pool = self._pool
        if pool is None or self._closing:
            return None
        future = pool.submit(
            self._sources.preview_motors if preview else self._sources.observe,
            request,
        )
        wake = SourceObservationWake(request.observation_id)
        operation = _SourceObservationOperation(
            request,
            future,
            wake,
            preview=preview,
            candidate_fingerprint=candidate_fingerprint,
            passive_refresh=passive_refresh,
            refresh_identity=refresh_identity,
        )
        self._operation = operation
        future.add_done_callback(
            lambda _done: self._deliver(wake)
        )
        return operation

    def set_launched_source(
        self,
        source: SourceSelection,
        identity: RunIdentity,
        *,
        live_mode: bool,
    ) -> None:
        if self._closing or self._closed:
            return
        self._pending_refresh = None
        self._live_source = (
            source
            if live_mode and type(source) is DirectorySourceSpec
            else None
        )
        self._live_identity = identity if self._live_source is not None else None

    def clear_live_refresh(self) -> None:
        self._pending_refresh = None
        self._live_source = None
        self._live_identity = None

    def _live_refresh_is_current(
        self,
        identity: RunIdentity,
        request: SourceObservationRequest,
        snapshot: RunIntentSnapshot,
        *,
        active_identity: RunIdentity | None,
        phase: RunPhase,
    ) -> bool:
        source = self._live_source
        intent = snapshot.thaw()
        return bool(
            not self._closing
            and not self._closed
            and self._pool is not None
            and source is not None
            and identity is self._live_identity
            and request.source == source
            and active_identity is identity
            and phase in _LIVE_SOURCE_REFRESH_PHASES
            and intent.live_mode
            and type(intent.source_spec) is DirectorySourceSpec
            and intent.source_spec == source
        )

    def queue_live_refresh(
        self,
        identity: RunIdentity,
        snapshot: RunIntentSnapshot,
        *,
        active_identity: RunIdentity | None,
        phase: RunPhase,
    ) -> None:
        if self._closing or self._closed:
            return
        source = self._live_source
        if source is None:
            return
        self._token += 1
        pending = SourceObservationRequest(
            self._token, snapshot.revision, source
        )
        if not self._live_refresh_is_current(
            identity,
            pending,
            snapshot,
            active_identity=active_identity,
            phase=phase,
        ):
            return
        if self._operation is not None:
            self._pending_refresh = _PendingLiveRefresh(
                identity,
                pending,
            )
            return
        self._submit(
            pending,
            passive_refresh=True,
            refresh_identity=identity,
        )

    def _launch_pending_refresh(
        self,
        snapshot: RunIntentSnapshot,
        *,
        active_identity: RunIdentity | None,
        phase: RunPhase,
    ) -> None:
        if self._operation is not None:
            return
        pending, self._pending_refresh = self._pending_refresh, None
        if (
            pending is not None
            and self._live_refresh_is_current(
                pending.identity,
                pending.request,
                snapshot,
                active_identity=active_identity,
                phase=phase,
            )
        ):
            self._submit(
                pending.request,
                passive_refresh=True,
                refresh_identity=pending.identity,
            )

    def consume(
        self,
        wake: object,
        *,
        active_identity: RunIdentity | None,
        phase: RunPhase,
        automatic_motor_permitted: bool,
        terminal_progress: bool,
    ) -> SourceSelectionTransition:
        if (
            self._closing
            or self._closed
            or type(wake) is not SourceObservationWake
            or self._operation is None
            or wake is not self._operation.wake
        ):
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        operation = self._operation
        self._operation = None
        snapshot = self._intents.snapshot()
        try:
            transition = self._settle(
                operation,
                snapshot,
                active_identity=active_identity,
                phase=phase,
                automatic_motor_permitted=automatic_motor_permitted,
                terminal_progress=terminal_progress,
            )
        finally:
            self._launch_pending_refresh(
                self._intents.snapshot(),
                active_identity=active_identity,
                phase=phase,
            )
        return transition

    def _settle(
        self,
        operation: _SourceObservationOperation,
        snapshot: RunIntentSnapshot,
        *,
        active_identity: RunIdentity | None,
        phase: RunPhase,
        automatic_motor_permitted: bool,
        terminal_progress: bool,
    ) -> SourceSelectionTransition:
        request = operation.request
        if operation.passive_refresh:
            identity = operation.refresh_identity
            if identity is None or not self._live_refresh_is_current(
                identity,
                request,
                snapshot,
                active_identity=active_identity,
                phase=phase,
            ):
                return SourceSelectionTransition(SourceRefreshEffect.NONE)
        try:
            observation: object = operation.future.result()
        except Exception:
            return SourceSelectionTransition(
                SourceRefreshEffect.NONE,
                SourceStatusDirective.UNCHANGED
                if operation.passive_refresh
                else SourceStatusDirective.UNAVAILABLE,
            )
        if type(observation) is not SourceObservation:
            return SourceSelectionTransition(
                SourceRefreshEffect.NONE,
                SourceStatusDirective.UNCHANGED
                if operation.passive_refresh
                else SourceStatusDirective.UNAVAILABLE,
            )
        expected_fingerprint = (
            operation.candidate_fingerprint if operation.preview else None
        )
        if not observation.qualifies(request, expected_fingerprint):
            return SourceSelectionTransition(
                SourceRefreshEffect.NONE,
                SourceStatusDirective.UNCHANGED
                if operation.passive_refresh
                else SourceStatusDirective.UNAVAILABLE,
            )
        if snapshot.thaw().source_spec != request.source:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        if operation.passive_refresh:
            identity = operation.refresh_identity
            if identity is None or not self._live_refresh_is_current(
                identity,
                request,
                snapshot,
                active_identity=active_identity,
                phase=phase,
            ):
                return SourceSelectionTransition(SourceRefreshEffect.NONE)
            observation = self._retain_exact_motor_knowledge(observation)

        prior = self._current_observation
        reset_terminal = bool(
            terminal_progress
            and prior is not None
            and (
                prior.candidate_fingerprint,
                prior.status,
                prior.exists,
                prior.observed_file_count,
            )
            != (
                observation.candidate_fingerprint,
                observation.status,
                observation.exists,
                observation.observed_file_count,
            )
        )
        self._current_observation = observation
        automatic_motor, receipt = (
            self._commit_automatic_motor_default(
                permitted=automatic_motor_permitted,
            )
        )
        if operation.preview:
            self._sources.publish_motor_knowledge(observation)
        if (
            not operation.preview
            and not operation.passive_refresh
            and (
                type(request.source) is DirectorySourceSpec
                or (
                    type(request.source) is SourceSpec
                    and request.source.kind is SourceKind.TIFF_SERIES
                )
            )
            and observation.exists
            and observation.candidate_fingerprint
            and self._pool is not None
        ):
            self._submit(
                request,
                preview=True,
                candidate_fingerprint=observation.candidate_fingerprint,
            )
        return SourceSelectionTransition(
            refresh=SourceRefreshEffect.CONTROLS,
            status=SourceStatusDirective.OBSERVED,
            observation=observation,
            automatic_motor=automatic_motor,
            reset_terminal_progress=reset_terminal,
            intent=receipt,
            preserve_terminal=True,
        )

    def _retain_exact_motor_knowledge(
        self,
        observation: SourceObservation,
    ) -> SourceObservation:
        fingerprint = observation.candidate_fingerprint
        if not fingerprint or observation.gi_motor_choices is not None:
            return observation
        knowledge = self._sources.project_motor_knowledge(
            observation.source,
            fingerprint,
        )
        if (
            type(knowledge) is SourceObservation
            and knowledge.source == observation.source
            and knowledge.candidate_fingerprint == fingerprint
            and knowledge.gi_motor_choices is not None
        ):
            return replace(
                observation,
                gi_motor_choices=knowledge.gi_motor_choices,
            )
        return observation

    def cancel_observation(self) -> None:
        operation, self._operation = self._operation, None
        if operation is not None and not operation.future.cancel():
            try:
                self._sources.cancel_observation(
                    operation.request.observation_id
                )
            except Exception:
                pass

    def reconcile_source(
        self,
        prior: RunIntentSnapshot,
        current: RunIntentSnapshot,
    ) -> SourceSelectionTransition:
        if self._closing or self._closed:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        self.remember_current(current)
        if prior.thaw().source_spec == current.thaw().source_spec:
            return SourceSelectionTransition(SourceRefreshEffect.NONE)
        self.clear_live_refresh()
        self._automatic_motor = _NO_AUTOMATIC_GI_MOTOR
        self._current_observation = None
        self.cancel_observation()
        return self.request_observation(current)

    def begin_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.clear_live_refresh()
        self._token += 1
        self.cancel_observation()

    def finalize_close(self) -> None:
        if self._closed:
            return
        self.begin_close()
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
        self._closed = True


__all__ = [
    "SourceIntentReceipt",
    "SourceObservationWake",
    "SourceRefreshEffect",
    "SourceSelectionOwner",
    "SourceSelectionTransition",
    "SourceStatusDirective",
]
