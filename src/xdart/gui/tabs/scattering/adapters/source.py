from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np
from natsort import os_sorted
from xrd_tools.core.filters import compile_filter
from xrd_tools.core.invalid import integer_saturation_ceiling
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.adapters import candidate_owner, get_adapter
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.selection import DirectorySourceSpec
from xrd_tools.sources.directory_session import DirectoryIndexSession
from xrd_tools.sources.run_plan import RunCandidatePlan

from ..contracts import (
    SourceCapture,
    SourceFileState,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
    SourceCountScope,
    SourceSelection,
)
from ..events import RequestId
from ..source_metadata import (
    image_metadata_motor_names,
    ordered_motor_intersection,
    read_image_motor_metadata,
)

class FilesystemSourceAdapter:
    _MOTOR_PREVIEW_LIMIT = 32

    @staticmethod
    def _threshold_max_for_dtype(dtype) -> float | None:
        ceiling = integer_saturation_ceiling(np.empty(0, dtype=dtype))
        return None if ceiling is None else ceiling - 1.0

    @classmethod
    def _image_threshold_max(cls, path, options) -> float | None:
        # One header on the observation worker. No pixels, per-frame maxima,
        # or detector-family guesses; admission still qualifies the source.
        from xrd_tools.io.image import read_detector_image_layout

        try:
            layout = read_detector_image_layout(
                path, raw_dtype=str(options.get("raw_dtype", "int32")),
                raw_header_skip=int(options.get("raw_header_skip", 0)),
                detector_shape=options.get("detector_shape"),
                detector=options.get("detector"),
            )
            return cls._threshold_max_for_dtype(layout.dtype)
        except Exception:
            # Passive presence/count readiness is independent of optional
            # layout availability. Unknown limits stay automatic in the UI.
            return None

    def __init__(self) -> None:
        self._lock = Lock()
        self._inflight: set[int] = set()
        self._cancelled: set[int] = set()
        self._capture_epoch = 0
        self._capture_request: RequestId | None = None
        self._motor_knowledge: SourceObservation | None = None

    def capture(self, source: SourceSelection, request_id: RequestId) -> SourceCapture:
        if type(request_id) is not RequestId:
            raise TypeError("source capture requires a RequestId")
        if type(source) is SourceSpec and source.kind in {
            SourceKind.TIFF_SERIES,
            SourceKind.NEXUS_STACK,
            SourceKind.EIGER_MASTER,
        }:
            captured: SourceSelection = SourceSpec(
                source.uri,
                source.kind,
                metadata_uri=source.metadata_uri,
                entry=source.entry,
                options=copy.deepcopy(dict(source.options)),
            )
        elif type(source) is DirectorySourceSpec:
            captured = DirectorySourceSpec(
                source.root, source.recursive, source.suffixes,
                source.name_filter, source.generation,
                source.metadata_format,
            )
        else:
            raise ValueError(
                "E2 supports image-series, container, or directory sources."
            )
        with self._lock:
            self._capture_epoch += 1
            self._capture_request = request_id
            epoch = self._capture_epoch
        return SourceCapture(request_id, epoch, captured)

    def cancel(self, request_id: RequestId) -> None:
        if type(request_id) is not RequestId:
            return
        with self._lock:
            if request_id is self._capture_request:
                self._capture_request = None

    def observe(self, request: SourceObservationRequest) -> SourceObservation:
        with self._lock:
            if (
                self._motor_knowledge is not None
                and self._motor_knowledge.source != request.source
            ):
                self._motor_knowledge = None
            self._inflight.add(request.observation_id)
        try:
            if self._is_cancelled(request.observation_id):
                return self._unavailable(request, "Observation cancelled.")
            if type(request.source) is DirectorySourceSpec:
                result = self._observe_directory(request)
            else:
                result = self._observe_path(request)
            if self._is_cancelled(request.observation_id):
                return self._unavailable(request, "Observation cancelled.")
            return result
        finally:
            with self._lock:
                self._inflight.discard(request.observation_id)
                self._cancelled.discard(request.observation_id)

    def cancel_observation(self, observation_id: int) -> None:
        if type(observation_id) is not int or observation_id <= 0:
            return
        with self._lock:
            if observation_id in self._inflight:
                self._cancelled.add(observation_id)
                if (
                    self._motor_knowledge is not None
                    and self._motor_knowledge.observation_id == observation_id
                ):
                    self._motor_knowledge = None

    def publish_motor_knowledge(
        self, observation: SourceObservation
    ) -> None:
        if (
            type(observation) is not SourceObservation
            or not observation.candidate_fingerprint
        ):
            return
        with self._lock:
            if observation.observation_id in self._cancelled:
                return
            self._motor_knowledge = observation

    def project_motor_knowledge(
        self,
        source: SourceSelection,
        candidate_fingerprint: str | None = None,
    ) -> SourceObservation | None:
        with self._lock:
            knowledge = self._motor_knowledge
        if (
            knowledge is None
            or knowledge.source != source
            or (
                candidate_fingerprint is not None
                and knowledge.candidate_fingerprint
                != candidate_fingerprint
            )
        ):
            return None
        return knowledge

    def preview_motors(
        self,
        request: SourceObservationRequest,
    ) -> SourceObservation:
        with self._lock:
            self._inflight.add(request.observation_id)
        session = None
        try:
            passive = (
                self._observe_directory(request)
                if type(request.source) is DirectorySourceSpec
                else self._observe_path(request)
            )
            source = request.source
            if (
                type(source) is SourceSpec
                and source.kind is SourceKind.TIFF_SERIES
            ):
                return self._preview_tiff_series(request, passive)
            if type(source) is not DirectorySourceSpec:
                return passive
            if self._is_cancelled(request.observation_id):
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            if not passive.exists:
                return passive
            if source.recursive and self._is_tiff_directory(source):
                return self._preview_recursive_tiff(request, passive)
            session = DirectoryIndexSession(probe_candidates=False)
            session.configure(
                source.root,
                recursive=source.recursive,
                name_filter=source.name_filter,
                suffixes=source.suffixes,
            )
            observation = session.observe(refresh=True)
            if self._is_cancelled(request.observation_id):
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            candidates = observation.discovered_snapshot.candidates
            direct_only_preview = not source.recursive
            if len(candidates) > self._MOTOR_PREVIEW_LIMIT:
                direct_count = passive.direct_child_count
                if not (
                    source.recursive
                    and direct_count is not None
                    and 0 < direct_count <= self._MOTOR_PREVIEW_LIMIT
                ):
                    return passive
                # Deeper recursive data remain outside Run admission.  The
                # bounded preview may inspect only the already-observed direct
                # children, provided their exact passive fingerprint still
                # matches after reconfiguration.
                session.configure(
                    source.root,
                    recursive=False,
                    name_filter=source.name_filter,
                    suffixes=source.suffixes,
                )
                observation = session.observe(refresh=True)
                if self._is_cancelled(request.observation_id):
                    return self._unavailable(
                        request, "Motor preview cancelled."
                )
                candidates = observation.discovered_snapshot.candidates
                if (
                    not candidates
                    or len(candidates) > self._MOTOR_PREVIEW_LIMIT
                ):
                    return passive
                direct_only_preview = True
            fingerprint = passive.candidate_fingerprint
            if direct_only_preview:
                fingerprint = self._candidate_fingerprint(candidates)
                if fingerprint != passive.candidate_fingerprint:
                    return passive
            session.enable_probes(exclude=())
            while observation.unprobed_count:
                if self._is_cancelled(request.observation_id):
                    return self._unavailable(
                        request, "Motor preview cancelled."
                    )
                observation = session.observe(refresh=False)
            plan = RunCandidatePlan.from_observation(observation)
            catalogs = []
            for candidate in plan.candidates:
                if self._is_cancelled(request.observation_id):
                    return self._unavailable(
                        request, "Motor preview cancelled."
                    )
                descriptor = plan.descriptor_for(candidate)
                if descriptor is not None:
                    catalogs.append(descriptor.motor_names)
                    continue
                adapter = get_adapter(candidate.adapter_id)
                catalogs.append(
                    image_metadata_motor_names(
                        candidate.path,
                        source.metadata_format,
                    )
                    if adapter is not None
                    and SourceKind.IMAGE_FILE in adapter.kinds
                    else None
                )
            if self._is_cancelled(request.observation_id):
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            choices = ordered_motor_intersection(catalogs)
            result = replace(
                passive, gi_motor_choices=choices,
                # The page qualifies this bounded result against the passive
                # direct-child observation that launched it. Run admission
                # independently revalidates the same root-plus-one scope.
                candidate_fingerprint=fingerprint,
            )
            self.publish_motor_knowledge(result)
            if self._is_cancelled(request.observation_id):
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            return result
        finally:
            if session is not None:
                session.close()
            with self._lock:
                self._inflight.discard(request.observation_id)
                self._cancelled.discard(request.observation_id)

    def _preview_tiff_series(
        self,
        request: SourceObservationRequest,
        passive: SourceObservation,
    ) -> SourceObservation:
        """Read every frozen TIFF sidecar without weakening Run admission.

        The dropdown is advisory, but its inventory is still frame-complete:
        a name is offered only when every frozen series member supplied one
        finite value.  The image and sidecar facts are guarded against changes
        during the preview; any race leaves motor knowledge unknown so a later
        observation can retry.  Exact Run admission independently repeats the
        same all-frame check before execution.
        """

        source = request.source
        if (
            type(source) is not SourceSpec
            or source.kind is not SourceKind.TIFF_SERIES
            or not passive.exists
            or not passive.candidate_fingerprint
        ):
            return passive
        raw_members = source.options.get("files", ())
        if (
            type(raw_members) is not tuple
            or not raw_members
            or not all(
                type(member) is str and member for member in raw_members
            )
        ):
            return passive
        members = tuple(Path(member) for member in raw_members)
        try:
            states = tuple(
                SourceFileState.capture(member) for member in members
            )
        except (OSError, ValueError):
            return passive
        if (
            len(states) != passive.direct_child_count
            or self._fingerprint(states) != passive.candidate_fingerprint
        ):
            return passive

        metadata_format = source.options.get("metadata_format", "auto")
        meta_dir = source.options.get("meta_dir")
        catalogs: list[tuple[str, ...]] = []
        for member in members:
            if self._is_cancelled(request.observation_id):
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            try:
                discovered = read_image_motor_metadata(
                    member,
                    metadata_format,
                    meta_dir=meta_dir,
                )
                discovered_path = discovered.source_path
                before = (
                    None
                    if discovered_path is None
                    else SourceFileState.capture(discovered_path)
                )
                observed = read_image_motor_metadata(
                    member,
                    metadata_format,
                    meta_dir=meta_dir,
                )
                observed_path = observed.source_path
                after = (
                    None
                    if observed_path is None
                    else SourceFileState.capture(observed_path)
                )
            except (OSError, TypeError, ValueError):
                return passive
            if (
                (discovered_path is None) != (observed_path is None)
                or (
                    discovered_path is not None
                    and observed_path is not None
                    and Path(discovered_path).resolve(strict=False)
                    != Path(observed_path).resolve(strict=False)
                )
                or before != after
            ):
                return passive
            catalogs.append(tuple(observed.values))

        if (
            self._is_cancelled(request.observation_id)
            or not all(state.matches_disk() for state in states)
        ):
            return passive
        result = replace(
            passive,
            gi_motor_choices=ordered_motor_intersection(catalogs),
        )
        self.publish_motor_knowledge(result)
        if self._is_cancelled(request.observation_id):
            return self._unavailable(
                request, "Motor preview cancelled."
            )
        return result

    def _preview_recursive_tiff(
        self,
        request: SourceObservationRequest,
        passive: SourceObservation,
    ) -> SourceObservation:
        """Inspect one bounded TIFF level; deeper data remain unsupported."""

        source = request.source
        if type(source) is not DirectorySourceSpec:
            return passive
        shallow_count = passive.one_level_file_count
        if (
            shallow_count is None
            or shallow_count <= 0
            or shallow_count > self._MOTOR_PREVIEW_LIMIT
        ):
            # Never infer whole-shallow-scope GI knowledge from only the root
            # children.  Unknown is safer than an attractive but incomplete
            # motor catalog.
            return passive
        # This bounded path deliberately bypasses DirectoryIndexSession, whose
        # discovery step normally bootstraps the built-in format registry.
        # Bootstrap lazily here as well before asking candidate_owner().
        import xrd_tools.sources.registry  # noqa: F401

        try:
            cancelled = lambda: self._is_cancelled(request.observation_id)
            members = self._one_level_tiff_members(
                source,
                cancelled=cancelled,
            )
            if members is None:
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            captured: list[SourceFileState] = []
            for member in members:
                if cancelled():
                    return self._unavailable(
                        request, "Motor preview cancelled."
                    )
                state = SourceFileState.capture(member)
                if cancelled():
                    return self._unavailable(
                        request, "Motor preview cancelled."
                    )
                captured.append(state)
            states = tuple(captured)
        except (OSError, ValueError):
            return passive
        if (
            len(members) != shallow_count
            or self._fingerprint(states) != passive.candidate_fingerprint
        ):
            # The shallow candidate set changed after the passive observation
            # which launched this preview.  A later observation may retry it;
            # this request must not publish knowledge for mixed generations.
            return passive
        catalogs: list[tuple[str, ...]] = []
        for member in members:
            if self._is_cancelled(request.observation_id):
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            try:
                owner = candidate_owner(member)
                if owner is None or SourceKind.IMAGE_FILE not in owner.kinds:
                    continue
                probed = owner.probe(member)
            except Exception:
                continue
            if self._is_cancelled(request.observation_id):
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            if probed.state is not ProbeState.READY:
                continue
            try:
                catalog = image_metadata_motor_names(
                    member,
                    source.metadata_format,
                )
            except Exception:
                # A readable image with unavailable metadata cannot support a
                # trustworthy GI readiness claim.
                return passive
            if self._is_cancelled(request.observation_id):
                return self._unavailable(
                    request, "Motor preview cancelled."
                )
            catalogs.append(catalog)
        if not catalogs:
            return passive
        if self._is_cancelled(request.observation_id):
            return self._unavailable(
                request, "Motor preview cancelled."
            )
        if not all(state.matches_disk() for state in states):
            return passive
        result = replace(
            passive,
            gi_motor_choices=ordered_motor_intersection(catalogs),
        )
        self.publish_motor_knowledge(result)
        if self._is_cancelled(request.observation_id):
            return self._unavailable(
                request, "Motor preview cancelled."
            )
        return result

    def _observe_path(self, request: SourceObservationRequest) -> SourceObservation:
        source = request.source
        if type(source) is not SourceSpec:
            return self._unavailable(request, "Unsupported source selection.")
        if source.kind is SourceKind.TIFF_SERIES:
            members = source.options.get("files", ())
            if (
                type(members) is not tuple
                or not members
                or not all(type(member) is str and member for member in members)
            ):
                return self._missing(
                    request,
                    str(source.options.get("scan_name") or Path(source.uri).name),
                )
            try:
                captured: list[SourceFileState] = []
                for member in members:
                    if self._is_cancelled(request.observation_id):
                        return self._unavailable(
                            request, "Observation cancelled."
                        )
                    fact = SourceFileState.capture(Path(member))
                    if self._is_cancelled(request.observation_id):
                        return self._unavailable(
                            request, "Observation cancelled."
                        )
                    captured.append(fact)
                facts = tuple(captured)
            except FileNotFoundError:
                return self._missing(
                    request,
                    str(source.options.get("scan_name") or Path(source.uri).name),
                )
            except (OSError, ValueError):
                return self._unavailable(
                    request,
                    "Image-series metadata is unavailable.",
                    str(source.options.get("scan_name") or Path(source.uri).name),
                )
            return SourceObservation(
                request.observation_id,
                request.intent_revision,
                request.source,
                SourceObservationStatus.AVAILABLE,
                str(source.options.get("scan_name") or Path(source.uri).name),
                True,
                True,
                sum(fact.size for fact in facts),
                max(fact.mtime_ns for fact in facts),
                len(facts),
                candidate_fingerprint=self._fingerprint(facts),
                default_threshold_max=self._image_threshold_max(members[0], source.options),
            )
        path = Path(source.uri).expanduser()
        try:
            state = SourceFileState.capture(path)
        except FileNotFoundError:
            return self._missing(request, path.name)
        except OSError:
            return self._unavailable(request, "Source metadata is unavailable.", path.name)
        try:
            is_directory = path.is_dir()
        except OSError:
            return self._unavailable(request, "Source metadata is unavailable.", path.name)
        frame_count = None
        default_threshold_max = None
        if source.kind in {SourceKind.NEXUS_STACK, SourceKind.EIGER_MASTER}:
            # An explicitly selected container is one bounded observation, so
            # its known frame count is useful readiness truth.  Directory
            # sources deliberately stay file-count-only here: opening every
            # matching container can be prohibitively slow on beamline shares.
            try:
                owner = candidate_owner(path)
                result = None if owner is None else owner.probe(path)
                descriptor = None if result is None else result.descriptor
                if (
                    result is not None
                    and result.state is ProbeState.READY
                    and descriptor is not None
                    and descriptor.frame_count > 0
                ):
                    frame_count = int(descriptor.frame_count)
                    default_threshold_max = self._threshold_max_for_dtype(descriptor.dtype)
            except Exception:
                frame_count = None
        return SourceObservation(
            request.observation_id, request.intent_revision, request.source,
            SourceObservationStatus.AVAILABLE, path.name, True, is_directory,
            state.size, state.mtime_ns, frame_count,
            candidate_fingerprint=self._fingerprint((state,)),
            default_threshold_max=default_threshold_max,
        )

    def _observe_directory(self, request: SourceObservationRequest) -> SourceObservation:
        source = request.source
        if type(source) is not DirectorySourceSpec:
            return self._unavailable(request, "Unsupported source selection.")
        path = Path(source.root)
        try:
            metadata = path.stat()
            if self._is_cancelled(request.observation_id):
                return self._directory_cancelled(request, path, source)
            children: list[Path] = []
            for child in path.iterdir():
                if self._is_cancelled(request.observation_id):
                    return self._directory_cancelled(
                        request, path, source
                    )
                children.append(child)
            if self._is_cancelled(request.observation_id):
                return self._directory_cancelled(request, path, source)
        except FileNotFoundError:
            return self._missing(request, path.name, deferred=source.recursive)
        except OSError:
            return self._unavailable(request, "Directory metadata is unavailable.", path.name, source.recursive)
        try:
            include = self._match_predicate(source)
            facts: list[SourceFileState] = []
            for child in children:
                if self._is_cancelled(request.observation_id):
                    return self._directory_cancelled(
                        request, path, source
                    )
                if not include(child.name) or not child.is_file():
                    continue
                facts.append(SourceFileState.capture(child))
            if self._is_cancelled(request.observation_id):
                return self._directory_cancelled(request, path, source)
        except (OSError, ValueError):
            return self._unavailable(
                request,
                "Directory metadata is unavailable.",
                path.name,
                source.recursive,
            )
        one_level_file_count = None
        fingerprint_facts = tuple(facts)
        if source.recursive:
            try:
                shallow_facts = list(facts)
                for child in children:
                    if self._is_cancelled(request.observation_id):
                        return self._directory_cancelled(
                            request, path, source
                        )
                    # pathlib's recursive discovery does not descend through
                    # directory symlinks.  Keep readiness/counting on that
                    # same exact candidate universe and do not escape root.
                    if child.is_symlink() or not child.is_dir():
                        continue
                    for member in child.iterdir():
                        if self._is_cancelled(request.observation_id):
                            return self._directory_cancelled(
                                request, path, source
                            )
                        if member.is_file() and include(member.name):
                            shallow_facts.append(
                                SourceFileState.capture(member)
                            )
                one_level_file_count = len(shallow_facts)
                # Recursive TIFF motor preview owns the same bounded shallow
                # universe, so its generation fingerprint must cover it too.
                # Container motor preview retains the established direct-only
                # fallback when a recursive tree exceeds its content-probe
                # cap; widening that fingerprint would make the fallback
                # reject an otherwise unchanged direct catalog.
                if self._is_tiff_directory(source):
                    fingerprint_facts = tuple(shallow_facts)
            except (OSError, ValueError):
                # The direct observation and its fingerprint remain truthful.
                # An incomplete shallow walk is never displayed as a count.
                one_level_file_count = None
        if self._is_cancelled(request.observation_id):
            return self._directory_cancelled(request, path, source)
        return SourceObservation(
            request.observation_id, request.intent_revision, source,
            SourceObservationStatus.AVAILABLE, path.name, True, True,
            metadata.st_size, metadata.st_mtime_ns, len(facts), source.recursive,
            candidate_fingerprint=self._fingerprint(fingerprint_facts),
            one_level_file_count=one_level_file_count,
            file_count_scope=(
                SourceCountScope.SELECTED_PLUS_IMMEDIATE
                if one_level_file_count is not None
                else SourceCountScope.DIRECT_ONLY
            ),
        )

    def _is_cancelled(self, observation_id: int) -> bool:
        with self._lock:
            return observation_id in self._cancelled

    def _directory_cancelled(
        self,
        request: SourceObservationRequest,
        path: Path,
        source: DirectorySourceSpec,
    ) -> SourceObservation:
        return self._unavailable(
            request,
            "Observation cancelled.",
            path.name,
            source.recursive,
        )

    @staticmethod
    def _match_predicate(source: object):
        suffixes = tuple(getattr(source, "suffixes", ()))
        name_ok = compile_filter(getattr(source, "name_filter", None))

        def included(name: str) -> bool:
            low = name.lower()
            suffix = next(
                (item for item in suffixes if low.endswith(item)),
                "",
            )
            if suffixes and not suffix:
                return False
            # DirectoryIndexSession applies the user filter to the filename
            # with its selected suffix removed.  Preview/readiness must use
            # the same predicate as exact Run admission.
            base = name[:-len(suffix)] if suffix else name
            return name_ok(base)

        return included

    @classmethod
    def _matches(cls, name: str, source: object) -> bool:
        return cls._match_predicate(source)(name)

    @staticmethod
    def _is_tiff_directory(source: DirectorySourceSpec) -> bool:
        suffixes = tuple(
            str(suffix).casefold().lstrip(".")
            for suffix in source.suffixes
            if str(suffix)
        )
        return bool(suffixes) and all(
            suffix in {"tif", "tiff"} for suffix in suffixes
        )

    @classmethod
    def _one_level_tiff_members(
        cls,
        source: DirectorySourceSpec,
        *,
        cancelled: Callable[[], bool],
    ) -> tuple[Path, ...] | None:
        root = Path(source.root)
        children: list[Path] = []
        for child in root.iterdir():
            if cancelled():
                return None
            children.append(child)
        include = cls._match_predicate(source)
        direct: list[Path] = []
        nested: list[Path] = []
        for child in children:
            if cancelled():
                return None
            if child.is_file() and include(child.name):
                direct.append(child)
                continue
            if child.is_symlink() or not child.is_dir():
                continue
            for member in child.iterdir():
                if cancelled():
                    return None
                if member.is_file() and include(member.name):
                    nested.append(member)
        return tuple(os_sorted((*direct, *nested)))

    @staticmethod
    def _fingerprint(values: tuple[SourceFileState, ...]) -> str:
        payload = json.dumps(
            tuple(
                tuple(value.as_dict().values())
                for value in sorted(values, key=lambda item: item.path)
            ),
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _candidate_fingerprint(cls, candidates: tuple[object, ...]) -> str:
        return cls._fingerprint(tuple(
            SourceFileState.capture(Path(value.path))
            for value in candidates
        ))

    @staticmethod
    def _missing(
        request: SourceObservationRequest,
        name: str,
        deferred: bool = False,
    ) -> SourceObservation:
        return SourceObservation(
            request.observation_id, request.intent_revision, request.source,
            SourceObservationStatus.MISSING, name, False, False,
            subdirectories_deferred=deferred, reason="Selected source does not exist.",
        )

    @staticmethod
    def _unavailable(
        request: SourceObservationRequest,
        reason: str,
        name: str = "",
        deferred: bool = False,
    ) -> SourceObservation:
        return SourceObservation(
            request.observation_id, request.intent_revision, request.source,
            SourceObservationStatus.UNAVAILABLE, name, False, False,
            subdirectories_deferred=deferred, reason=reason,
        )

__all__ = ["FilesystemSourceAdapter"]
