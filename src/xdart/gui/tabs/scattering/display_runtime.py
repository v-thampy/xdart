"""Qt-free run-scoped display catalog and residency ownership."""

from __future__ import annotations

import logging
from collections import OrderedDict
from copy import copy
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Lock, RLock, Thread
from typing import Callable

import numpy as np

from xdart.modules.display_context import (
    HydrationOwner,
    HydrationRequest,
    Viewer2DCatalogHydrationRequest,
    Viewer2DCommitGate,
    Viewer2DFrameHydrationRequest,
)
from xdart.modules.frame_publication import (
    FramePublication,
    PublicationStore,
    _publication_has_heavy_payload,
)
from xrd_tools.core import FrameRecord, FrameView
from xrd_tools.core.frame_view import DEFAULT_MODE_KEY
from xrd_tools.core.invalid import (
    combine_detector_masks,
    detector_value_mask,
    integer_saturation_ceiling,
)
from xrd_tools.core.staging import (
    browse_publication_max_items,
    heavy_window,
    live_record_store_max_items,
)
from xrd_tools.io.frame_preview import DetectorPreviewProjection
from xrd_tools.session.frame_record_store import FrameRecordStore
from xrd_tools.session.hydration import (
    HydrationCompletion,
    HydrationOutcome,
    HydrationPurpose,
    HydrationReadKey,
    HydrationScope,
    HydrationToken,
)
from xrd_tools.session.scan_norm import (
    ScanNormAggregate,
    empty_norm_aggregate,
    fold_norm_metadata,
    next_norm_revision,
)
from xrd_tools.session import (
    DynamicRunState,
    Light1DCleanupPending,
    Light1DCustodyState,
    Light1DLeaseState,
    Light1DModeData,
    Light1DRecord,
    Light1DRetentionLease,
)

from .display_values import (
    DisplayFrameCatalog,
    DisplayFrameKey,
    DisplayNavigationDelta,
    StandardDisplayPayload,
    StandardEventKind,
    StandardRunEvent,
)
from .display_catalog import CATALOG_MAX_ITEMS, DisplayCatalogIndex
from .display_residency import (
    DisplayResidencyLimits,
    DisplayResidencySnapshot,
    RunDisplayResidency,
)
from .browse_values import canonical_browse_source_identity
from .events import RunIdentity
from .hydration_transport import HydrationTransport, PreparedHydrationCommit


logger = logging.getLogger(__name__)

THUMBNAIL_MAX_ITEMS = 512


class DetectorHydrationOutcome(str, Enum):
    FRAME_MASK_UNAVAILABLE = "frame_mask_unavailable"
    DETECTOR_UNAVAILABLE = "detector_unavailable"


@dataclass(slots=True)
class DisplayArtifact:
    artifact: Path
    source_scan: str
    records: FrameRecordStore
    publications: PublicationStore
    light_lease: Light1DRetentionLease | None = None
    light_slot: object | None = None
    light_hooks: object | None = None
    light_retry_token: object | None = None
    light_release_reason: str | None = None
    light_unsubscribe: Callable[[], None] | None = None
    mask: np.ndarray | None = None
    mask_saturation: bool = True
    measurement_mode: str = "Standard"
    gi_incidence_motor: str = ""
    gi_resolved_motor: str = ""
    gi_mode_1d: str = ""
    gi_mode_2d: str = ""
    #: The accepted detector saturation ceiling, stamped ONCE from the first
    #: live frame's native integer dtype (single writer:
    #: :meth:`RunDisplayState.stamp_saturation_ceiling`).  ``None`` means the
    #: accepted saturation input is unavailable and a toggle-ON preview
    #: fallback fails closed instead of guessing a ceiling.
    saturation_ceiling: float | None = None
    #: Session-owned first-frame value mask copied once for live, persisted,
    #: and rehydrated detector parity.  ``saturation_mask_seeded`` distinguishes
    #: a valid empty mask from a run that has not accepted its first frame.
    saturation_mask: np.ndarray | None = None
    saturation_mask_seeded: bool = False
    wavelength_m: float | None = None
    #: Monotonic authority that this artifact's output/checkpoint owner has
    #: finished and its final durable projection was applied.  This belongs to
    #: the artifact (not the overall GUI phase): prior directory artifacts may
    #: be terminal while the run continues with a later artifact.
    hydration_closed: bool = False
    #: The per-artifact whole-scan normalization aggregate (E6-NORM-N1).
    #: ``None`` until the first successful live retain; assigned only at the
    #: successful tail of :meth:`RunDisplayState.retain_frame`, so revision 0
    #: is never observable and a failed retain leaves the prior value.
    norm_aggregate: ScanNormAggregate | None = None


class RunDisplayState:
    """One run owner for catalog, payload, and all artifact residency budgets."""

    def __init__(
        self,
        identity: RunIdentity,
        *,
        max_payload_items: int,
        catalog_max_items: int | None = None,
    ) -> None:
        if type(max_payload_items) is not int or max_payload_items < 1:
            raise ValueError("max_payload_items must be positive")
        self.identity = identity
        self.max_payload_items = max_payload_items
        self.payloads: OrderedDict[
            DisplayFrameKey, StandardDisplayPayload
        ] = OrderedDict()
        self.artifacts: OrderedDict[str, DisplayArtifact] = OrderedDict()
        self._catalog_capacity_explicit = catalog_max_items is not None
        self.catalog = DisplayCatalogIndex(
            identity,
            max_items=(
                CATALOG_MAX_ITEMS
                if catalog_max_items is None
                else catalog_max_items
            ),
        )
        self._partition_count = 1
        self._partition_index = 0
        self._npt = 0
        self._frame_bytes: int | None = None
        self._record_store_factory: Callable[..., FrameRecordStore] = (
            FrameRecordStore
        )
        self._publication_store_factory: Callable[
            ..., PublicationStore
        ] = PublicationStore
        self._configured = False
        self._allocated_heavy_limit: int | None = None
        self._residency = RunDisplayResidency(
            DisplayResidencyLimits(1, 1, 1, 1)
        )
        self._lock = RLock()
        self._light_admission_lock = Lock()
        self._event_sink: Callable[[StandardRunEvent], object] | None = None
        self._transport = HydrationTransport(
            self.commit_preview, self._derive_target,
            completion_sink=self._complete_hydration,
        )
        self._frame_mask_qualified: set[DisplayFrameKey] = set()
        self._detector_outcomes: dict[
            DisplayFrameKey, DetectorHydrationOutcome
        ] = {}
        self._full_demand: tuple | None = None
        self._full_demand_generation = 0
        self._full_pending = False
        self._full_diagnostic: tuple[DisplayFrameKey, str] | None = None
        self._raw_lru: OrderedDict[DisplayFrameKey, None] = OrderedDict()
        self._retired = False

    @property
    def configured(self) -> bool:
        return self._configured

    @property
    def transport(self) -> HydrationTransport:
        """The ONE composed preview transport this run owns."""
        return self._transport

    @property
    def hydration_thread(self) -> Thread | None:
        """The transport's worker, exposed for page liveness polling."""
        return self._transport.worker

    def set_factories(
        self,
        record_store_factory: Callable[..., FrameRecordStore],
        publication_store_factory: Callable[..., PublicationStore],
    ) -> None:
        if self.artifacts:
            return
        self._record_store_factory = record_store_factory
        self._publication_store_factory = publication_store_factory

    def bind_transport(
        self,
        *,
        event_sink: Callable[[StandardRunEvent], object],
    ) -> None:
        """Bind the one repaint event sink; reads go through the transport."""
        with self._lock:
            if self._retired:
                return
            self._event_sink = event_sink

    def configure(
        self,
        *,
        partition_count: int,
        npt: int,
        frame_bytes: int | None,
    ) -> None:
        if type(partition_count) is not int or partition_count < 1:
            raise ValueError("display partition count must be positive")
        if self.artifacts:
            return
        self._partition_count = partition_count
        self._npt = npt if type(npt) is int and npt > 0 else 0
        self._frame_bytes = (
            frame_bytes
            if type(frame_bytes) is int and frame_bytes > 0
            else None
        )
        self._configured = True
        if not self._catalog_capacity_explicit:
            self.catalog.resize(
                browse_publication_max_items(self._npt)
            )
        self._residency = RunDisplayResidency(
            DisplayResidencyLimits(
                heavy_window(self._frame_bytes),
                THUMBNAIL_MAX_ITEMS,
                browse_publication_max_items(self._npt),
                live_record_store_max_items(self._npt),
            )
        )

    def bind_heavy_allocation(self, allocation: object) -> int:
        """Bind the display tier to the monotonic minimum actual heavy grant."""

        record = getattr(allocation, "record_heavy_items", None)
        publication = getattr(allocation, "publication_heavy_items", None)
        if any(type(value) is not int or value < 1
               for value in (record, publication)):
            raise ValueError("display heavy allocation grants must be positive integers")
        granted = min(record, publication)
        with self._lock:
            effective = (
                granted if self._allocated_heavy_limit is None
                else min(self._allocated_heavy_limit, granted)
            )
            self._allocated_heavy_limit = effective
            self._residency.limits = replace(
                self._residency.limits, heavy=effective,
            )
            self._residency.enforce(protected=self._raw_lru)
            return effective

    def add_artifact(
        self,
        artifact: Path,
        source_scan: str,
        *,
        mask: np.ndarray | None,
        mask_saturation: bool,
        measurement_mode: str,
        gi_incidence_motor: str = "",
        gi_resolved_motor: str = "",
        gi_mode_1d: str = "",
        gi_mode_2d: str = "",
        wavelength_m: float | None = None,
    ) -> DisplayArtifact:
        if not self._configured:
            self.configure(
                partition_count=1,
                npt=0,
                frame_bytes=None,
            )
        if self._partition_index >= self._partition_count:
            raise RuntimeError("display received more artifacts than admitted")
        records = self._record_store_factory(
            max_items=None,
            max_heavy_items=None,
        )
        publications = self._publication_store_factory(
            max_items=None,
            max_heavy_items=None,
            max_thumbnail_items=None,
        )
        publications.set_evictable_probe(records.can_release_record)
        publications.set_heavy_evictable_probe(records.can_release_heavy)
        publications.set_thumbnail_evictable_probe(records.can_release_thumbnail)
        frozen_mask = None if mask is None else np.asarray(mask, dtype=bool)
        if frozen_mask is not None:
            frozen_mask.setflags(write=False)
        owner = DisplayArtifact(
            artifact=artifact,
            source_scan=source_scan,
            records=records,
            publications=publications,
            mask=frozen_mask,
            mask_saturation=bool(mask_saturation),
            measurement_mode=measurement_mode,
            gi_incidence_motor=gi_incidence_motor,
            gi_resolved_motor=gi_resolved_motor,
            gi_mode_1d=gi_mode_1d,
            gi_mode_2d=gi_mode_2d,
            wavelength_m=(
                None if wavelength_m is None else float(wavelength_m)
            ),
        )
        self.artifacts[str(artifact)] = owner
        self._partition_index += 1
        return owner

    def bind_checkpoint_hydration(self, owner: DisplayArtifact) -> None:
        owner.records._bind_checkpoint_hydration_lineage(
            str(owner.artifact), self.identity)

    def mark_hydration_closed(self, owner: DisplayArtifact) -> None:
        """Publish one artifact's monotonic terminal hydration authority."""
        with self._lock:
            if self.artifacts.get(str(owner.artifact)) is not owner:
                raise ValueError("display hydration owner is foreign")
            owner.hydration_closed = True

    def mark_all_hydration_closed(self) -> None:
        """Close hydration for every artifact after terminal output drain."""
        with self._lock:
            for owner in self.artifacts.values():
                owner.hydration_closed = True

    @staticmethod
    def _checkpoint_authority(owner: DisplayArtifact):
        required = owner.records._checkpoint_hydration_required()
        token, gate = owner.records._checkpoint_hydration_authority()
        return required, token, gate

    def stage_light_1d(self, owner: DisplayArtifact, lease: Light1DRetentionLease,
                       *, hooks=None, slot=None) -> None:
        with self._light_admission_lock:
            if any(incoming is not None and current is not None
                   and current is not incoming for current, incoming in (
                       (owner.light_lease, lease), (owner.light_hooks, hooks),
                       (owner.light_slot, slot),
                   )):
                raise RuntimeError("display artifact light-1D custody changed")
            if slot is not None and hooks is None and owner.light_hooks is None:
                raise RuntimeError("custody slot requires staged cleanup hooks")
            owner.light_lease = lease
            owner.light_hooks = owner.light_hooks if hooks is None else hooks
            owner.light_slot = owner.light_slot if slot is None else slot

    def bind_light_1d(self, owner: DisplayArtifact, lease: Light1DRetentionLease) -> None:
        with self._light_admission_lock:
            if owner.light_lease is not lease or owner.light_hooks is None:
                raise RuntimeError("display artifact has no staged light-1D lease")

    def bind_light_custody(self, owner: DisplayArtifact, hooks: object, slot: object) -> None:
        with self._light_admission_lock:
            if owner.light_lease is None or owner.light_hooks is not hooks \
                    or owner.light_slot is not slot:
                raise RuntimeError("display artifact has no staged custody graph")

    def bind_light_subscription(self, owner: DisplayArtifact, register, callback) -> None:
        with self._light_admission_lock:
            lease, slot = owner.light_lease, owner.light_slot
            if (
                self.artifacts.get(str(owner.artifact)) is not owner
                or (lease is None and (owner.light_hooks is not None or slot is not None))
                or (lease is not None and (
                    owner.light_hooks is None or slot is None
                    or lease.state is not Light1DLeaseState.ACTIVE
                    or slot.state is not Light1DCustodyState.PENDING))
                or any(value is not None for value in (
                    owner.light_release_reason, owner.light_retry_token,
                    owner.light_unsubscribe))
            ):
                raise RuntimeError("display artifact has no active subscription owner")
            owner.light_unsubscribe = register(callback)

    def cancel_light_1d(self, owner: DisplayArtifact, commit_gate: object | None) -> None:
        with self._light_admission_lock:
            unsubscribe = owner.light_unsubscribe
        if unsubscribe is not None:
            unsubscribe()
        mutation = None
        with self._light_admission_lock:
            if owner.light_unsubscribe is unsubscribe:
                owner.light_unsubscribe = None
            if commit_gate is not None:
                mutation = self._transport.cancel_gate_detached(commit_gate)
        if mutation is not None:
            self._transport.dispatch_detached(mutation)

    def _mutate_transport_locked(self, request=None, *, gate=None, closed=False):
        if request is not None:
            return self._transport.submit_detached(request, closed=closed)
        return self._transport.cancel_gate_detached(gate) if gate is not None else None

    def _dispatch_transport(self, mutation) -> None:
        if mutation is not None:
            self._transport.dispatch_detached(mutation)

    def submit_viewer_2d(self, request, *, closed=False):
        if type(request) not in (Viewer2DCatalogHydrationRequest,
                                 Viewer2DFrameHydrationRequest) or type(
                                     request.commit_gate) is not Viewer2DCommitGate:
            return None
        with self._light_admission_lock:
            mutation = self._transport.submit_detached(request, closed=closed)
        self._transport.dispatch_detached(mutation)
        return mutation.token

    def _evict_raw_key_locked(self, key: DisplayFrameKey) -> None:
        art = self.artifacts.get(key.artifact); label = key.local_frame_label
        if art is not None: art.publications.evict_raw(label)
        payload = self.payloads.get(key)
        if payload is not None and payload.view.raw is not None:
            view = copy(payload.view); object.__setattr__(view, "raw", None)
            updated = copy(payload); object.__setattr__(updated, "view", view)
            self.payloads[key] = updated
        self._raw_lru.pop(key, None); self._residency._rearm_heavy_key(key)

    def _protected_raw_labels_locked(self, owner: DisplayArtifact, incoming=None) -> tuple:
        labels = []
        for key in tuple(self._raw_lru):
            art = self.artifacts.get(key.artifact); publication = None if art is None else art.publications.get(key.local_frame_label)
            payload = self.payloads.get(key); raw = None if publication is None else publication.view.raw
            exact = bool(art is not None and self.catalog.resolve(key) is key and art.source_scan == key.source_scan and raw is not None and publication.generation == art.publications.generation and publication.scan_key == key.source_scan and (payload is None or payload.frame_key is key and payload.view.raw is raw))
            if exact and art is owner and incoming is not None and incoming.label == key.local_frame_label:
                exact = (incoming.generation, incoming.source_identity, incoming.scan_key) == (publication.generation, publication.source_identity, publication.scan_key)
            if not exact: self._evict_raw_key_locked(key)
            elif art is owner: labels.append(key.local_frame_label)
        return tuple(labels)

    def _complete_hydration(self, completion: HydrationCompletion) -> None:
        event = sink = None
        with self._lock:
            demand = self._full_demand
            if type(completion) is not HydrationCompletion or demand is None or completion.token is not demand[1]: return
            key = demand[3]; self._full_pending = False
            art = self.artifacts.get(key.artifact); publication = None if art is None else art.publications.get(key.local_frame_label)
            resident = bool(publication is not None and publication.view.raw is not None); diagnostic = None if resident else completion.diagnostic or "Full Raw detector pixels unavailable"
            self._full_diagnostic = None if diagnostic is None else (key, diagnostic)
            if diagnostic is not None and not self._retired:
                event = StandardRunEvent(self.identity, StandardEventKind.DISPLAY_READY, artifact=key.artifact, frame_key=key, selection_generation=demand[6]); sink = self._event_sink
        if event is not None and sink is not None: sink(event)

    def cancel_viewer_2d(self, gate) -> None:
        if type(gate) is not Viewer2DCommitGate:
            return
        with self._light_admission_lock:
            mutation = self._mutate_transport_locked(gate=gate)
        self._dispatch_transport(mutation)

    def viewer_2d_retains_gate(self, gate) -> bool:
        if type(gate) is not Viewer2DCommitGate:
            return False
        return self._transport.retains_gate(gate)

    def viewer_2d_blocked_cleanup_token(self, token):
        return self._transport.blocked_cleanup_token(token)

    def retry_viewer_2d_blocked_cleanup(self, token):
        return self._transport.retry_blocked_cleanup(token)

    def verify_light_1d(self, owner: DisplayArtifact, commit_gate: object | None) -> None:
        if commit_gate is not None and self._transport.retains_gate(commit_gate):
            raise RuntimeError("shared hydration gate remains retained")

    def release_light_1d(self, owner: DisplayArtifact, *, reason: str, terminal=None) -> bool:
        reason = str(reason)
        with self._light_admission_lock:
            lease, slot = owner.light_lease, owner.light_slot
            hooks, token = owner.light_hooks, owner.light_retry_token
            unsubscribe = owner.light_unsubscribe
            if lease is None:
                if any(value is not None for value in (slot, hooks, token)):
                    return False
            else:
                state = lease.state
                pending_slot = slot is not None and slot.state is Light1DCustodyState.PENDING
                if state is Light1DLeaseState.FENCED:
                    return False
                if state is Light1DLeaseState.CLEANUP_PENDING:
                    receipt = lease.cleanup_receipt
                    if token is None or receipt is None \
                            or token is not receipt.retry_token:
                        return False
                elif state is Light1DLeaseState.ACTIVE:
                    if (slot is None or pending_slot) and terminal is None:
                        return False
                    if (slot is None or pending_slot) and type(terminal) is not DynamicRunState:
                        raise TypeError("construction release requires exact terminal intent")
                if owner.light_release_reason not in {None, reason}:
                    raise ValueError("light-1D release reason cannot change")
                owner.light_release_reason = reason
                if state is Light1DLeaseState.ACTIVE:
                    lease.fence()
                if state is Light1DLeaseState.ACTIVE and pending_slot:
                    slot.cancel(terminal=terminal)
        if lease is None:
            if unsubscribe is not None:
                unsubscribe()
                with self._light_admission_lock:
                    if owner.light_unsubscribe is unsubscribe:
                        owner.light_unsubscribe = None
            return True
        try:
            if slot is not None \
                    and slot.state is Light1DCustodyState.CLEANUP_PENDING:
                slot.retry_cleanup(token)
            elif lease.state is Light1DLeaseState.CLEANUP_PENDING:
                lease.retry_cleanup(token, hooks=hooks)
            elif slot is not None and slot.state is Light1DCustodyState.RETAINED:
                slot.release(reason=owner.light_release_reason)
            elif slot is None or slot.state is Light1DCustodyState.CANCELLED:
                if lease.state is not Light1DLeaseState.RELEASED:
                    lease.release(reason=owner.light_release_reason, hooks=hooks)
            elif slot.state is not Light1DCustodyState.RELEASED:
                raise RuntimeError("light-1D custody is not terminally releasable")
        except Light1DCleanupPending as error:
            with self._light_admission_lock:
                receipt = lease.cleanup_receipt
                if owner.light_lease is lease and receipt is not None \
                        and error.token is receipt.retry_token:
                    owner.light_retry_token = receipt.retry_token
            return False
        with self._light_admission_lock:
            clean = owner.light_lease is lease \
                and lease.state is Light1DLeaseState.RELEASED \
                and lease.authority.snapshot().reservation_count == 0 \
                and owner.publications.allocation is None \
                and owner.publications._light_1d is None \
                and owner.light_unsubscribe is None
            if not clean:
                return False
            owner.light_lease = owner.light_slot = None
            owner.light_hooks = owner.light_retry_token = None
            owner.light_release_reason = owner.light_unsubscribe = None
        return True

    def light_1d_cleanup_unresolved(self) -> bool:
        with self._light_admission_lock:
            for owner in self.artifacts.values():
                lease, slot = owner.light_lease, owner.light_slot
                if lease is None:
                    if any(value is not None for value in (
                        slot, owner.light_hooks, owner.light_retry_token,
                        owner.light_release_reason, owner.light_unsubscribe,
                        owner.publications.allocation, owner.publications._light_1d,
                    )):
                        return True
                    continue
                historical = lease.state is Light1DLeaseState.ACTIVE \
                    and slot is not None and slot.state is Light1DCustodyState.RETAINED \
                    and owner.light_hooks is not None \
                    and owner.light_release_reason is None and owner.light_retry_token is None \
                    and owner.light_unsubscribe is not None \
                    and owner.publications.allocation is not None \
                    and owner.publications._light_1d is lease \
                    and lease.authority.snapshot().reservation_count == 1
                if not historical:
                    return True
        return False

    def admit_additional_partition(self) -> None:
        """Monotonically admit one next Live artifact partition.

        A directory Live run cannot know its final artifact count at Start.
        Extending only the partition ceiling preserves the existing catalog,
        residency owner, and per-artifact stores.  The operation is idempotent
        until the newly admitted slot is consumed, so a proven source drift
        before ``add_artifact`` can retry without inflating the ceiling.
        """

        with self._lock:
            if not self._configured or self._partition_index < 1:
                raise RuntimeError(
                    "additional display partition requires one admitted artifact"
                )
            self._partition_count = max(
                self._partition_count,
                self._partition_index + 1,
            )

    def append_navigation(
        self,
        source_scan: str,
        artifact: str,
        local_frame_label: int,
    ) -> DisplayNavigationDelta:
        with self._lock:
            delta = self.catalog.append(
                source_scan, artifact, local_frame_label
            )
            for retired in delta.retired:
                self._frame_mask_qualified.discard(retired)
                self._detector_outcomes.pop(retired, None)
                self._evict_raw_key_locked(retired)
            self._residency.retire_navigation(delta.retired)
        return delta

    def seed_navigation(
        self,
        source_scan: str,
        artifact: str,
        local_frame_label: int,
    ) -> DisplayFrameKey:
        """Install one payload-free persisted key, deduplicated exactly."""

        with self._lock:
            existing = self.catalog.resolve_exact(
                artifact, local_frame_label,
            )
            if existing is not None:
                return existing
            return self.append_navigation(
                source_scan, artifact, local_frame_label,
            ).appended

    def seed_navigation_at_work_ordinal(
        self,
        source_scan: str,
        artifact: str,
        local_frame_label: int,
        work_ordinal: int,
    ) -> DisplayFrameKey:
        """Install one payload-free persisted key at an absolute ordinal."""

        return self.seed_navigation_prefix_at_work_ordinals((
            (source_scan, artifact, local_frame_label, work_ordinal),
        ))[0]

    def seed_navigation_prefix_at_work_ordinals(
        self,
        rows: tuple[tuple[str, str, int, int], ...],
    ) -> tuple[DisplayFrameKey, ...]:
        """Install one exact persisted prefix after all-row preflight."""

        with self._lock:
            deltas = self.catalog.seed_many_at_work_ordinals(rows)
            retired = tuple(
                key
                for delta in deltas
                for key in delta.retired
            )
            for key in retired:
                self._frame_mask_qualified.discard(key)
                self._detector_outcomes.pop(key, None)
                self._evict_raw_key_locked(key)
            self._residency.retire_navigation(retired)
            return tuple(delta.appended for delta in deltas)

    @property
    def navigation_capacity(self) -> int:
        return self.catalog.max_items

    def publish_light_1d(self, owner: DisplayArtifact, record: FrameRecord, *, source_identity: str) -> FramePublication:
        with self._lock, self._light_admission_lock:
            lease = owner.light_lease
            if type(lease) is not Light1DRetentionLease \
                    or lease.state is not Light1DLeaseState.ACTIVE:
                raise RuntimeError("light-1D callback has no active lineage lease")
            light_record = _light_1d_record(
                record, generation=lease.generation, source_identity=source_identity,
                scan_key=owner.source_scan)
            display_record = FrameRecord(
                record.label, results_1d=dict(record.results_1d),
                active_mode_1d=record.active_mode_1d)
            publication = FramePublication(
                display_record.active_view(), record=display_record,
                source_identity=source_identity, generation=owner.publications.generation,
                scan_key=owner.source_scan)
            protected = self._protected_raw_labels_locked(owner, publication)
            return owner.publications.publish_gui_light_1d(
                publication, light_record, protected=protected)

    def retain_frame(
        self,
        owner: DisplayArtifact,
        key: DisplayFrameKey,
        record: FrameRecord,
        publication: FramePublication,
        *,
        source_identity: str,
        frame_mask_qualified: bool,
    ) -> None:
        with self._lock:
            self._detector_outcomes.pop(key, None)
            if frame_mask_qualified:
                self._frame_mask_qualified.add(key)
            else:
                self._frame_mask_qualified.discard(key)
            draft = owner.norm_aggregate
            if draft is None:
                draft = empty_norm_aggregate((
                    self.identity.generation,
                    self.identity.fingerprint,
                    str(owner.artifact),
                    owner.source_scan,
                ))
            draft = fold_norm_metadata(
                draft, publication.view.metadata_numeric
            )
            candidate = _without_light_1d(publication)
            protected = self._protected_raw_labels_locked(owner, candidate)
            owner.publications.upsert(candidate, protected=protected)
            self._residency.observe(
                key,
                records=owner.records,
                publications=owner.publications,
            )
            self._residency.enforce(protected=self._raw_lru)
            owner.norm_aggregate = next_norm_revision(draft)

    def mark_durable(
        self,
        owner: DisplayArtifact,
        labels: tuple[int, ...],
    ) -> None:
        """Rearm this writer-bound owner's heavy keys at durability."""

        with self._lock:
            self._residency._rearm_heavy_owner(
                owner.records, owner.publications
            )
            self._residency.enforce(protected=self._raw_lru)

    def mark_checkpoint_recoverable(
        self, owner: DisplayArtifact, labels: tuple[int, ...],
    ) -> None:
        with self._lock:
            self._residency._rearm_heavy_labels(
                owner.records, owner.publications, labels)
            self._residency.enforce(protected=self._raw_lru)

    def frame_norm_aggregate(
        self, key: DisplayFrameKey
    ) -> ScanNormAggregate | None:
        """The exact-frame read of the per-artifact aggregate (E6-NORM-N1).

        A foreign run identity, artifact or source scan returns ``None``;
        callers never traverse ``artifacts`` to reconstruct normalization.
        """
        with self._lock:
            if key.run_identity != self.identity:
                return None
            owner = self.artifacts.get(key.artifact)
            if owner is None or owner.source_scan != key.source_scan:
                return None
            return owner.norm_aggregate

    def residency_snapshot(self) -> DisplayResidencySnapshot:
        return self._residency.snapshot()

    def detector_outcome(
        self, key: DisplayFrameKey
    ) -> DetectorHydrationOutcome | None:
        with self._lock:
            return self._detector_outcomes.get(key)

    def full_raw_status(self, key: DisplayFrameKey) -> tuple[bool, bool, str | None]:
        with self._lock:
            art = self.artifacts.get(key.artifact); publication = None if art is None or self.catalog.resolve(key) is not key else art.publications.get(key.local_frame_label)
            resident = bool(publication is not None and publication.view.raw is not None)
            pending = bool(self._full_pending and self._full_demand is not None and self._full_demand[3] is key)
            diagnostic = self._full_diagnostic
            return resident, pending, diagnostic[1] if diagnostic is not None and diagnostic[0] is key else None

    def invalidate_full_demand(self, *, clear_raw: bool = False) -> None:
        with self._lock:
            self._full_demand = None; self._full_pending = False; self._full_diagnostic = None
            if clear_raw:
                for key in tuple(self._raw_lru): self._evict_raw_key_locked(key)

    def request_full(self, key: DisplayFrameKey, generation: int, *,
                     owner: HydrationOwner, commit_gate: object,
                     closed: bool = False) -> HydrationToken | None:
        if (type(key) is not DisplayFrameKey or type(generation) is not int or generation < 0
                or type(owner) is not HydrationOwner or not owner.qualified or commit_gate is None): return None
        with self._lock:
            art = self.artifacts.get(key.artifact)
            if self._retired or art is None or self.catalog.resolve(key) is not key: return None
            closed = bool(closed or art.hydration_closed)
            required, checkpoint_token, checkpoint_gate = (
                (False, None, None) if closed else self._checkpoint_authority(art))
            if required and checkpoint_token is None: return None
            read_key = HydrationReadKey(HydrationScope(*owner.as_tuple()), key.artifact, key.local_frame_label, HydrationPurpose.FULL)
            serial = self._full_demand_generation = self._full_demand_generation + 1
            token = HydrationToken(read_key, serial)
            request = HydrationRequest(key.local_frame_label, HydrationPurpose.FULL, serial,
                owner, (art.records, art.publications), commit_gate, read_key=read_key, token=token)
            request = replace(
                request, checkpoint_token=checkpoint_token,
                checkpoint_gate=checkpoint_gate)
            self._full_demand = (serial, token, owner, key, HydrationPurpose.FULL, commit_gate, generation)
            self._full_diagnostic = None; publication = art.publications.get(key.local_frame_label)
            if publication is not None and publication.view.raw is not None:
                self._full_pending = False; self._raw_lru.pop(key, None); self._raw_lru[key] = None
                return token
            self._full_pending = True
        with self._light_admission_lock:
            mutation = self._transport.submit_detached(request, closed=closed)
        if mutation.token is None:
            with self._lock:
                if self._full_demand is not None and self._full_demand[1] is token: self._full_demand = None; self._full_pending = False
            return None
        self._transport.dispatch_detached(mutation)
        return mutation.token

    def resolve_frame(
        self, frame: DisplayFrameKey
    ) -> DisplayFrameKey | None:
        return self.catalog.resolve(frame)

    def complete_frame_keys(
        self, frames: tuple[DisplayFrameKey, ...],
    ) -> frozenset[DisplayFrameKey]:
        """Return acquisition frames complete for their persisted dimensions.

        Light-1D composition remains complete for an Int1D artifact.  When the
        same label has a recoverable persisted/checkpoint 2-D mode, however,
        the base publication must also carry complete resident 2-D arrays.
        The batch store queries avoid materialising every retained light pair
        during the GUI's navigation-residency projection.
        """
        if type(frames) is not tuple:
            return frozenset()
        with self._lock:
            grouped: dict[
                str, tuple[DisplayArtifact, list[tuple[DisplayFrameKey, int | str]]]
            ] = {}
            for frame in frames:
                if type(frame) is not DisplayFrameKey:
                    continue
                owner = self.artifacts.get(frame.artifact)
                if (
                    owner is None
                    or owner.source_scan != frame.source_scan
                    or self.catalog.resolve(frame) is not frame
                ):
                    continue
                group = grouped.get(frame.artifact)
                if group is None:
                    group = (owner, [])
                    grouped[frame.artifact] = group
                group[1].append((frame, frame.local_frame_label))

            resident: list[DisplayFrameKey] = []
            for owner, candidates in grouped.values():
                labels = tuple(label for _frame, label in candidates)
                complete = owner.publications.complete_labels(labels)
                require_2d = owner.records.recoverable_mode_labels(
                    complete, "2d",
                )
                complete_2d = owner.publications.complete_2d_labels(
                    require_2d,
                )
                accepted = (complete - require_2d) | complete_2d
                resident.extend(
                    frame
                    for frame, label in candidates
                    if label in accepted
                )
            return frozenset(resident)

    def project(
        self,
        frame: DisplayFrameKey,
        selection_generation: int,
        *,
        closed: bool,
        owner: HydrationOwner | None = None,
        commit_gate: object | None = None,
        require_complete: bool = True,
    ) -> StandardDisplayPayload | None:
        """Project one frame; a missing display tier submits one typed
        ``PREVIEW`` through the transport when the caller supplies its exact
        context identity (``owner``) and commit authority (``commit_gate``)."""
        with self._lock:
            if self._retired:
                return None
            key = self.catalog.resolve(frame)
            if key is None:
                return None
            payload = self.payloads.get(key)
            artifact_owner = self.artifacts.get(key.artifact)
            publication = (
                None
                if artifact_owner is None
                else artifact_owner.publications.get(key.local_frame_label)
            )
            detector_outcome = self._detector_outcomes.get(key)
            effective_closed = bool(
                closed
                or (
                    artifact_owner is not None
                    and artifact_owner.hydration_closed
                )
            )
        if artifact_owner is None:
            return self._qualified_payload(
                payload,
                key,
                selection_generation,
            )
        needs_hydration = publication_needs_hydration(
            publication,
            detector_outcome,
        ) or _artifact_2d_needs_hydration(
            artifact_owner,
            key.local_frame_label,
            publication,
        )
        if needs_hydration and require_complete:
            self._request_preview(
                artifact_owner,
                key,
                selection_generation,
                effective_closed,
                owner,
                commit_gate,
            )
            return None
        if publication is None:
            return self._qualified_payload(
                payload,
                key,
                selection_generation,
            )
        payload = self._payload_from_publication(
            artifact_owner,
            key,
            publication,
            closed=effective_closed,
        )
        return self._qualified_payload(
            payload,
            key,
            selection_generation,
        )

    def catalog_snapshot(self) -> DisplayFrameCatalog:
        return self.catalog.snapshot()

    def put_payload(self, payload: StandardDisplayPayload) -> None:
        key = payload.frame_key
        if type(key) is not DisplayFrameKey:
            return
        with self._lock:
            if self._retired:
                return
            art = self.artifacts.get(key.artifact)
            if key in self._raw_lru and art is None: self._evict_raw_key_locked(key)
            elif key in self._raw_lru and key.local_frame_label in self._protected_raw_labels_locked(art):
                publication = art.publications.get(key.local_frame_label)
                payload = replace(payload, view=replace(payload.view, raw=publication.view.raw,
                    mask_baked=publication.view.mask_baked))
            payload = replace(payload, view=replace(
                payload.view, axis_1d=None, intensity_1d=None, sigma_1d=None))
            self.payloads[key] = payload
            self.payloads.move_to_end(key)
            while len(self.payloads) > self.max_payload_items:
                self.payloads.popitem(last=False)

    def retire(self, *, join_timeout: float) -> bool:
        with self._lock:
            self._retired = True
            self.invalidate_full_demand(clear_raw=True)
        if not self._transport.retire(join_timeout=join_timeout):
            return False
        clean = True
        for owner in tuple(self.artifacts.values()):
            reason = owner.light_release_reason or (
                "construction" if owner.light_lease is not None
                and owner.light_slot is None else "run-close")
            if not self.release_light_1d(owner, reason=reason):
                clean = False
        return clean

    def _qualified_payload(
        self,
        payload: StandardDisplayPayload | None,
        key: DisplayFrameKey,
        selection_generation: int,
    ) -> StandardDisplayPayload | None:
        if payload is None or payload.frame_key is not key:
            return None
        return replace(
            payload,
            selection_generation=selection_generation,
        )

    def _payload_from_publication(
        self,
        owner: DisplayArtifact,
        key: DisplayFrameKey,
        publication: FramePublication,
        *,
        closed: bool,
    ) -> StandardDisplayPayload:
        view = publication.view
        if owner.light_lease is not None:
            view = _detached_light_1d_view(view)
        return StandardDisplayPayload(
            0,
            key,
            (
                f"{owner.measurement_mode} · {key.source_scan} · "
                f"frame {key.local_frame_label}"
            ),
            view,
            "finished" if closed else "running",
            measurement_mode=owner.measurement_mode,
            gi_incidence_motor=owner.gi_incidence_motor,
            gi_resolved_motor=owner.gi_resolved_motor,
            gi_mode_1d=owner.gi_mode_1d,
            gi_mode_2d=owner.gi_mode_2d,
            wavelength_m=owner.wavelength_m,
        )

    def stamp_saturation_ceiling(
        self, owner: DisplayArtifact, image: object
    ) -> None:
        """Stamp the dtype-derived ceiling once from the exact live frame;
        a float frame yields none and a toggle-ON fallback then fails closed
        rather than inventing one."""
        if owner.saturation_ceiling is None:
            owner.saturation_ceiling = integer_saturation_ceiling(image)

    def stamp_saturation_mask(
        self,
        owner: DisplayArtifact,
        mask: np.ndarray | None,
    ) -> None:
        """Copy the session's immutable first-frame value mask exactly once."""
        frozen = None
        if mask is not None:
            frozen = np.array(mask, dtype=bool, copy=True)
            frozen.setflags(write=False)
        if owner.saturation_mask_seeded:
            current = owner.saturation_mask
            if (current is None) != (frozen is None) or (
                current is not None
                and frozen is not None
                and not np.array_equal(current, frozen)
            ):
                raise RuntimeError("display saturation mask changed within one run")
            return
        owner.saturation_mask = frozen
        owner.saturation_mask_seeded = True

    def _request_preview(
        self,
        art: DisplayArtifact,
        key: DisplayFrameKey,
        selection_generation: int,
        closed: bool,
        owner: HydrationOwner | None,
        commit_gate: object | None,
    ) -> None:
        """Build and submit one typed acquisition ``PREVIEW`` request."""
        if (
            type(owner) is not HydrationOwner
            or not owner.qualified
            or commit_gate is None
        ):
            return
        with self._lock:
            if self.artifacts.get(str(art.artifact)) is not art:
                return
            closed = bool(closed or art.hydration_closed)
        try:
            scope = HydrationScope(*owner.as_tuple())
            read_key = HydrationReadKey(
                scope,
                key.artifact,
                key.local_frame_label,
                HydrationPurpose.PREVIEW,
            )
            token = HydrationToken(read_key, int(selection_generation))
            required, checkpoint_token, checkpoint_gate = (
                (False, None, None) if closed else self._checkpoint_authority(art))
            if required and checkpoint_token is None:
                return
            request = HydrationRequest(
                key.local_frame_label,
                HydrationPurpose.PREVIEW,
                int(selection_generation),
                owner,
                (art.records, art.publications),
                commit_gate,
                read_key=read_key,
                token=token,
                checkpoint_token=checkpoint_token,
                checkpoint_gate=checkpoint_gate,
            )
        except (TypeError, ValueError):
            return
        with self._light_admission_lock:
            lease = art.light_lease
            if lease is not None and lease.state is not Light1DLeaseState.ACTIVE:
                return
            if (len(request.stores) != 2 or request.stores[0] is not art.records
                    or request.stores[1] is not art.publications):
                return
            mutation = self._transport.submit_detached(request, closed=closed)
        if mutation is not None:
            self._transport.dispatch_detached(mutation)

    def _derive_target(self, request: HydrationRequest):
        """Derive the frozen values-only projection and exact key ONCE at
        submit from the exact carried target.  A non-owned target (a Browse
        store) gets no projection — thumbnails stay free, detector fallback
        fails closed; a frame-mask-qualified key or a toggle-ON artifact
        without an accepted ceiling derives the explicit unavailable marker."""
        artifact = request.read_key.artifact_identity
        label = request.read_key.frame_identity
        with self._lock:
            art = self.artifacts.get(artifact)
            stores = request.stores
            exact = (
                art is not None
                and len(stores) == 2
                and stores[0] is art.records
                and stores[1] is art.publications
            )
            if not exact:
                return None, None
            key = self.catalog.resolve_exact(artifact, label)
            if key is not None and key in self._frame_mask_qualified:
                return DetectorPreviewProjection.unavailable(), key
            if (
                art.mask_saturation
                and not art.saturation_mask_seeded
                and art.saturation_ceiling is None
            ):
                return DetectorPreviewProjection.unavailable(), key
            projection_mask = art.mask
            dynamic_saturation = bool(art.mask_saturation)
            if art.mask_saturation and art.saturation_mask_seeded:
                dynamic_saturation = False
                if art.saturation_mask is not None:
                    projection_mask = combine_detector_masks(
                        art.mask,
                        art.saturation_mask,
                        art.saturation_mask.shape,
                    )
            policy = {
                "mask_saturation": dynamic_saturation,
                "saturation_ceiling": (
                    art.saturation_ceiling if dynamic_saturation else None
                ),
            }
            if projection_mask is not None:
                return (
                    DetectorPreviewProjection.from_mask(projection_mask, **policy),
                    key,
                )
            return DetectorPreviewProjection.without_static_mask(**policy), key

    def commit_preview(
        self, prepared: PreparedHydrationCommit
    ) -> HydrationOutcome:
        """THE one target-port operation: validate under this run's lock plus
        the exact request-carried CommitGate, perform the idempotent
        supporting updates, publish the authoritative publication/payload
        last; rejection changes no public state."""
        if type(prepared) is not PreparedHydrationCommit:
            raise TypeError(
                "commit_preview consumes one PreparedHydrationCommit"
            )
        request = prepared.request
        preview = prepared.preview
        if preview.read_key != request.read_key:
            return HydrationOutcome.OWNER_MISMATCH
        gate = request.commit_gate
        stores = request.stores
        event: StandardRunEvent | None = None
        heavy_victim: DisplayFrameKey | None = None
        with self._lock:
            if self._retired:
                return HydrationOutcome.CANCELLED
            art = self.artifacts.get(request.read_key.artifact_identity)
            acquisition_target = (
                art is not None
                and len(stores) == 2
                and stores[0] is art.records
                and stores[1] is art.publications
            )
            browse_target = not acquisition_target and len(stores) == 1
            key = None
            if acquisition_target:
                key = prepared.key or self.catalog.resolve_exact(
                    request.read_key.artifact_identity,
                    request.read_key.frame_identity,
                )
                if key is None or self.catalog.resolve(key) is not key:
                    return HydrationOutcome.OWNER_MISMATCH
            elif not browse_target:
                return HydrationOutcome.OWNER_MISMATCH
            checkpoint_gate = request.checkpoint_gate
            checkpoint_required = bool(
                acquisition_target
                and not prepared.closed
                and art.records._checkpoint_hydration_required()
            )
            if checkpoint_required and (
                request.checkpoint_token is None
                or checkpoint_gate is not art.records._checkpoint_hydration_authority()[1]
                or request.checkpoint_token.run_lineage is not self.identity
            ):
                return HydrationOutcome.OWNER_MISMATCH
            if request.purpose is HydrationPurpose.FULL:
                demand = self._full_demand
                if (not acquisition_target or demand is None or prepared.token is not demand[1]
                        or request.owner != demand[2] or key is not demand[3]
                        or demand[4] is not HydrationPurpose.FULL or request.commit_gate is not demand[5]
                        or prepared.token.presentation_generation != demand[0]
                        or prepared.token.read_key.purpose is not HydrationPurpose.FULL
                        or prepared.token.read_key.scope != HydrationScope(*demand[2].as_tuple())):
                    return HydrationOutcome.OWNER_MISMATCH
            if not gate.enter(request.epoch):
                return (
                    HydrationOutcome.CANCELLED
                    if bool(getattr(gate, "cancelled", False))
                    else HydrationOutcome.OWNER_MISMATCH
                )
            checkpoint_entered = False
            try:
                if checkpoint_required:
                    checkpoint_entered = checkpoint_gate.enter(
                        request.checkpoint_token)
                    if not checkpoint_entered:
                        return HydrationOutcome.OWNER_MISMATCH
                if acquisition_target:
                    event, heavy_victim = self._commit_acquisition_locked(
                        art, key, prepared
                    )
                else:
                    event = self._commit_browse_locked(stores[0], prepared)
            finally:
                if checkpoint_entered:
                    checkpoint_gate.leave()
                gate.leave()
            if acquisition_target and event is not None:
                # The commit is SEALED: publication/payload landed coherently.
                # Cap enforcement is demotion-only bookkeeping by the same
                # trusted owner; a raise here must not un-publish the sealed
                # commit — it is logged and the next commit's enforce retries
                # the identical trims over current state.
                try:
                    self._residency.enforce(
                        protected=self._raw_lru,
                        heavy_victim=heavy_victim,
                    )
                except Exception:
                    logger.exception(
                        "display residency cap enforcement failed; the next "
                        "commit re-enforces over current state"
                    )
        event_sink = self._event_sink
        if event is not None and event_sink is not None:
            event_sink(event)
        return HydrationOutcome.HYDRATED

    def _commit_acquisition_locked(
        self,
        art: DisplayArtifact,
        key: DisplayFrameKey,
        prepared: PreparedHydrationCommit,
    ) -> tuple[
        StandardRunEvent | None,
        DisplayFrameKey | None,
    ]:
        preview = prepared.preview
        if prepared.token.read_key.purpose is HydrationPurpose.FULL:
            presentation_generation = self._full_demand[6]
            if preview.raw is None:
                return None, None
            publication = art.publications.install_raw(key.local_frame_label, preview.raw, mask_baked=_projection_masks_values(prepared.projection))
            if publication is None:
                return None, None
            payload = self.payloads.get(key)
            if payload is None:
                self.put_payload(replace(self._payload_from_publication(art, key, publication, closed=prepared.closed), selection_generation=presentation_generation))
            else:
                view = copy(payload.view); object.__setattr__(view, "raw", preview.raw); object.__setattr__(view, "mask_baked", publication.view.mask_baked)
                payload = copy(payload); object.__setattr__(payload, "view", view); object.__setattr__(payload, "selection_generation", presentation_generation)
                self.payloads[key] = payload
            self._raw_lru.pop(key, None); self._raw_lru[key] = None
            while len(self._raw_lru) > 8: self._evict_raw_key_locked(next(iter(self._raw_lru)))
            return (
                StandardRunEvent(
                    self.identity,
                    StandardEventKind.DISPLAY_READY,
                    artifact=key.artifact,
                    frame_key=key,
                    selection_generation=presentation_generation,
                ),
                None,
            )
        presentation_generation = prepared.token.presentation_generation
        view = preview.view
        if preview.raw is not None:
            view = replace(
                view,
                raw=preview.raw,
                mask_baked=(
                    view.mask_baked
                    or _projection_masks_values(prepared.projection)
                ),
            )
        detector_outcome = self._detector_outcomes.get(key)
        if (
            detector_outcome is None
            and view.thumbnail is None
            and preview.raw is None
        ):
            if key in self._frame_mask_qualified:
                detector_outcome = (
                    DetectorHydrationOutcome.FRAME_MASK_UNAVAILABLE
                )
            elif (
                prepared.closed
                and preview.detector_diagnostic is not None
                and has_integrated_values(view)
            ):
                detector_outcome = (
                    DetectorHydrationOutcome.DETECTOR_UNAVAILABLE
                )
        record = FrameRecord.from_view(
            view,
            mode_1d=art.gi_mode_1d or DEFAULT_MODE_KEY,
            mode_2d=art.gi_mode_2d or DEFAULT_MODE_KEY,
        )
        shell = art.publications.get_light_1d_shell(key.local_frame_label)
        source_identity = (
            shell.source_identity if shell is not None
            else f"{view.source_path or ''}#{view.source_frame_index}"
        )
        candidate = FramePublication(
            view,
            record=record,
            source_identity=source_identity,
            generation=art.publications.generation,
            raw_status="ready" if preview.raw is not None else "thumbnail" if view.thumbnail is not None else "missing",
            scan_key=art.source_scan,
        )
        raw_protected = self._protected_raw_labels_locked(art, candidate)
        heavy_before = frozenset(art.publications.heavy_labels())
        protected = _hydration_locality_protection(
            art.publications,
            key.local_frame_label,
            protected=raw_protected,
        )
        residency = self._residency.capture(key)
        try:
            self._residency.observe(
                key,
                records=art.records,
                publications=art.publications,
                incoming_heavy=_publication_has_heavy_payload(candidate),
                incoming_thumbnail=view.thumbnail is not None,
            )
            lease = art.light_lease
            if (
                type(lease) is Light1DRetentionLease
                and lease.state is Light1DLeaseState.ACTIVE
                and record.results_1d
            ):
                publication = art.publications.publish_gui_light_1d(
                    candidate,
                    _light_1d_record(
                        record,
                        generation=lease.generation,
                        source_identity=source_identity,
                        scan_key=art.source_scan,
                    ),
                    protected=protected,
                )
            else:
                publication = art.publications.upsert(candidate, protected=protected)
        except BaseException:
            self._residency.restore(residency)
            raise
        heavy_after = frozenset(art.publications.heavy_labels())
        victim_label = next(
            (
                label
                for label in heavy_before
                if label not in heavy_after
            ),
            None,
        )
        heavy_victim = (
            None
            if victim_label is None
            else self.catalog.resolve_exact(
                str(art.artifact),
                victim_label,
            )
        )
        if detector_outcome is not None:
            self._detector_outcomes[key] = detector_outcome
        payload = replace(
            self._payload_from_publication(
                art,
                key,
                publication,
                closed=prepared.closed,
            ),
            selection_generation=presentation_generation,
        )
        self.put_payload(payload)
        return (
            StandardRunEvent(
                self.identity,
                StandardEventKind.DISPLAY_READY,
                artifact=key.artifact,
                frame_key=key,
                selection_generation=presentation_generation,
            ),
            heavy_victim,
        )

    def _commit_browse_locked(
        self, store: object, prepared: PreparedHydrationCommit
    ) -> StandardRunEvent:
        """Upsert one Browse publication into exactly the carried B store."""
        preview = prepared.preview
        view = preview.view
        if preview.raw is not None:
            view = replace(
                view,
                raw=preview.raw,
                mask_baked=(
                    view.mask_baked
                    or _projection_masks_values(prepared.projection)
                ),
            )
        prior = store.get(prepared.request.label)
        prior_record = None if prior is None else prior.record
        record = FrameRecord.from_view(
            view,
            mode_1d=(
                prior_record.active_mode_1d
                if prior_record is not None and prior_record.results_1d
                else DEFAULT_MODE_KEY
            ),
            mode_2d=(
                prior_record.active_mode_2d
                if prior_record is not None and prior_record.results_2d
                else DEFAULT_MODE_KEY
            ),
        )
        source_identity = canonical_browse_source_identity(
            view,
            prepared.request.read_key.artifact_identity,
            source_base=preview.source_base,
            source_root=prepared.request.read_key.source_root,
        )
        store.upsert(
            FramePublication(
                view,
                record=record,
                source_identity=source_identity,
                scan_key=prepared.request.owner.scan_key,
            ),
            protected=_hydration_locality_protection(
                store,
                prepared.request.label,
            ),
        )
        # The Browse repaint hint: no DisplayFrameKey exists at this seam, so
        # the event carries the artifact identity and the exact presentation
        # generation; the context runtime re-projects the current selection.
        return StandardRunEvent(
            self.identity,
            StandardEventKind.DISPLAY_READY,
            artifact=prepared.request.read_key.artifact_identity,
            selection_generation=prepared.token.presentation_generation,
        )


def has_integrated_values(view: object) -> bool:
    try:
        return any(
            value is not None and np.asarray(value).size > 0
            for value in (view.intensity_1d, view.intensity_2d)
        )
    except Exception:
        return False


def _projection_masks_values(
    projection: DetectorPreviewProjection | None,
) -> bool:
    """Whether the applied projection baked any value/static mask at all."""
    return projection is not None and bool(
        projection.mask_saturation
        or projection.apply_threshold
        or (projection.mask_available and projection.mask_bytes)
    )


def _detached_light_1d_view(view: FrameView) -> FrameView:
    """Give the GUI immutable 1-D values independent of run custody."""

    axis = view.axis_1d
    if axis is not None and axis.values is not None:
        axis = replace(
            axis,
            values=np.array(axis.values, copy=True),
        )
    return replace(
        view,
        axis_1d=axis,
        intensity_1d=(
            None
            if view.intensity_1d is None
            else np.array(view.intensity_1d, copy=True)
        ),
        sigma_1d=(
            None
            if view.sigma_1d is None
            else np.array(view.sigma_1d, copy=True)
        ),
    )


def _light_1d_record(record: FrameRecord, *, generation: int,
                     source_identity: str, scan_key: str) -> Light1DRecord:
    modes = {}
    for mode, view in record.results_1d.items():
        axis = view.axis_1d
        if axis is None or axis.values is None or view.intensity_1d is None:
            raise ValueError("light-1D record has an incomplete mode")
        modes[mode] = Light1DModeData(
            np.asarray(axis.values, dtype=np.float64),
            np.asarray(view.intensity_1d, dtype=np.float64),
            None if view.sigma_1d is None
            else np.asarray(view.sigma_1d, dtype=np.float64),
        )
    return Light1DRecord(
        record.label, generation, record.active_mode_1d, modes,
        {"source_identity": source_identity, "scan_key": scan_key},
    )


def _without_light_1d(publication: FramePublication) -> FramePublication:
    return replace(
        publication,
        view=replace(publication.view, axis_1d=None, intensity_1d=None,
                     sigma_1d=None),
        record=FrameRecord(
            publication.label,
            results_2d=dict(publication.record.results_2d),
            active_mode_2d=publication.record.active_mode_2d,
        ),
        raw_ref=None,
    )


def _hydration_locality_protection(
    store: PublicationStore,
    label: int | str,
    *,
    protected: tuple[int | str, ...] | frozenset[int | str] = (),
) -> frozenset[int | str]:
    """Coordinate one target-local heavy victim for Browse or acquisition."""
    retained = store.labels()
    heavy = frozenset(store.heavy_labels())
    mandatory = frozenset(protected)
    try:
        # Browse admission requires strictly increasing labels.  Sorting the
        # retained shells recovers artifact ordinal order, including gaps in
        # the current heavy window.  Acquisition uses the same label contract.
        ordered = tuple(sorted(set(retained) | {label}))
    except TypeError:
        return mandatory | {label}
    target_ordinal = ordered.index(label)
    ranked = tuple(
        (abs(ordinal - target_ordinal), ordinal, candidate)
        for ordinal, candidate in enumerate(ordered)
        if candidate in heavy
        and candidate != label
        and candidate not in mandatory
    )
    if not ranked:
        return mandatory | {label}
    victim = max(ranked)[2]
    return frozenset((heavy - {victim}) | mandatory | {label})


def _publication_has_complete_2d(
    publication: FramePublication | None,
) -> bool:
    if publication is None or not publication.record.results_2d:
        return False
    return all(
        view.has_2d
        and view.axis_2d_x.values is not None
        and view.axis_2d_y.values is not None
        for view in publication.record.results_2d.values()
    )


def _artifact_2d_needs_hydration(
    owner: DisplayArtifact,
    label: int | str,
    publication: FramePublication | None,
) -> bool:
    return bool(
        owner.records.recoverable_mode_labels((label,), "2d")
        and not _publication_has_complete_2d(publication)
    )


def publication_needs_hydration(
    publication: FramePublication | None,
    detector_outcome: DetectorHydrationOutcome | None,
) -> bool:
    return (
        publication is None
        or publication.record.is_empty
        or any(
            view.axis_1d is None
            or view.axis_1d.values is None
            or view.intensity_1d is None
            for view in publication.record.results_1d.values()
        )
        or any(not view.has_2d or view.axis_2d_x.values is None or view.axis_2d_y.values is None for view in publication.record.results_2d.values())
    )


def browse_publication_needs_hydration(
    publication: FramePublication | None,
    detector_outcome: DetectorHydrationOutcome | None,
) -> bool:
    """Require only Browse's active scientific modes plus one detector preview.

    A light Browse record deliberately carries array-free shells for inactive
    named 2-D modes.  Those shells preserve topology but must not make the
    current frame hydrate forever.
    """

    if publication is None or publication.record.is_empty:
        return True
    record = publication.record
    active_1d = record.view_1d()
    if record.results_1d and (
        active_1d is None
        or active_1d.axis_1d is None
        or active_1d.axis_1d.values is None
        or active_1d.intensity_1d is None
    ):
        return True
    active_2d = record.view_2d()
    if record.results_2d and (
        active_2d is None
        or not active_2d.has_2d
        or active_2d.axis_2d_x.values is None
        or active_2d.axis_2d_y.values is None
    ):
        return True
    return bool(
        publication.view.raw is None
        and publication.view.thumbnail is None
        and detector_outcome
        is not DetectorHydrationOutcome.DETECTOR_UNAVAILABLE
    )


def project_detector_values(
    image: object,
    mask: np.ndarray | None,
    *,
    value_mask_enabled: bool,
) -> tuple[np.ndarray, bool]:
    """Return immutable detector values with the accepted mask baked once."""
    raw = np.asarray(image)
    resolved = detector_value_mask(
        mask,
        raw,
        enabled=value_mask_enabled,
    )
    if resolved is None:
        result = raw.copy()
    else:
        result = np.array(raw, dtype=float, copy=True)
        result[resolved] = np.nan
    result.setflags(write=False)
    return result, resolved is not None


def project_frame_detector_values(
    image: object,
    static_mask: np.ndarray | None,
    frame_mask: object,
    *,
    value_mask_enabled: bool,
    stable_value_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Project live detector values with the reduction-qualified mask union."""
    raw = np.asarray(image)
    resolved = combine_detector_masks(static_mask, frame_mask, raw.shape)
    resolved = combine_detector_masks(resolved, stable_value_mask, raw.shape)
    return project_detector_values(
        raw,
        resolved,
        value_mask_enabled=value_mask_enabled,
    )


__all__ = [
    "CATALOG_MAX_ITEMS",
    "DetectorHydrationOutcome",
    "browse_publication_needs_hydration",
    "DisplayArtifact",
    "RunDisplayState",
    "THUMBNAIL_MAX_ITEMS",
    "has_integrated_values",
    "project_detector_values",
    "project_frame_detector_values",
]
