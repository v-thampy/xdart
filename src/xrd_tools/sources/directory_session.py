# -*- coding: utf-8 -*-
"""Serialized, value-only ownership for one persistent ``DirectoryIndex``.

The index itself is mutable and therefore never leaves the session's single
worker thread.  GUI and processing callers receive immutable observations and
may share this session without racing directory polls or probe bookkeeping.
"""

from __future__ import annotations

import os
import threading
import time
import logging
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from xrd_tools.sources import directory_index
from xrd_tools.sources.directory_index import (
    DirectoryIndex,
    IndexDelta,
    Snapshot,
    StaleCandidateError,
)
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.probe import ProbeResult, ProbeState
from xrd_tools.sources.selection import SeriesBound
from xrd_tools.core.filters import compile_filter


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DirectorySessionConfig:
    root: Path
    recursive: bool = False
    name_filter: str | None = None
    suffixes: tuple[str, ...] = ()
    series: SeriesBound | None = None  # a Live Image Series: only its members

    @classmethod
    def build(cls, root, *, recursive=False, name_filter=None, suffixes=(), series=None):
        normalized = tuple(str(item).lower() for item in suffixes if item)
        return cls(
            Path(root).expanduser(), bool(recursive), name_filter or None,
            normalized, series)


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    candidate: Candidate
    result: ProbeResult

    @property
    def descriptor(self):
        """Container facts retained by this exact candidate observation."""
        return self.result.descriptor


@dataclass(frozen=True, slots=True)
class DirectoryObservation:
    """One immutable poll/probe result from a :class:`DirectoryIndexSession`."""

    request_generation: int
    discovered_snapshot: Snapshot
    ready_snapshot: Snapshot
    candidates: tuple[CandidateObservation, ...]
    delta: IndexDelta
    content_opens: int
    stale_drops: int
    elapsed_s: float
    unprobed_count: int = 0

    def result_for(self, candidate: Candidate | str | Path) -> ProbeResult | None:
        path = candidate.path if isinstance(candidate, Candidate) else Path(candidate)
        for observation in self.candidates:
            if observation.candidate.path == path:
                return observation.result
        return None

    @property
    def pending_count(self) -> int:
        return sum(
            item.result.state is ProbeState.IN_PROGRESS
            for item in self.candidates
        )

    @property
    def excluded_count(self) -> int:
        return sum(
            item.result.state is not ProbeState.READY
            and item.result.state is not ProbeState.IN_PROGRESS
            for item in self.candidates
        )

    @property
    def acquiring_count(self) -> int:
        """Probed candidates still genuinely provisional."""
        return max(0, self.pending_count - int(self.unprobed_count))

    @property
    def ready_observations(self) -> tuple[CandidateObservation, ...]:
        """READY candidates and their retained facts, in catalog order."""
        return tuple(
            item for item in self.candidates
            if item.result.state is ProbeState.READY
        )

    @property
    def ready_frame_count(self) -> int | None:
        """Total READY detector frames from this catalog generation.

        ``None`` means at least one READY adapter did not provide descriptor
        facts.  Built-in NeXus candidates always provide them; plugin adapters
        remain supported and may use their legacy file-count presentation.
        """
        total = 0
        for item in self.ready_observations:
            descriptor = item.descriptor
            if descriptor is None:
                return None
            total += max(0, int(descriptor.frame_count))
        return total


class DirectoryIndexSession:
    """Own one persistent index on one serialized worker thread.

    ``configure`` is cheap and thread-safe.  ``observe_async`` is intended for
    GUI callers; ``observe`` lets the processing worker consume the same owner.
    Neither API exposes the mutable index or any open source object.
    """

    def __init__(
        self,
        *,
        retry_deadline: float | None = None,
        max_probes_per_observation: int = 8,
        probe_time_budget_s: float = 0.75,
        probe_candidates: bool = True,
    ) -> None:
        if max_probes_per_observation < 1:
            raise ValueError("max_probes_per_observation must be at least 1")
        if probe_time_budget_s <= 0:
            raise ValueError("probe_time_budget_s must be positive")
        self._lock = threading.Lock()
        self._desired: DirectorySessionConfig | None = None
        self._request_generation = 0
        self._active: DirectorySessionConfig | None = None
        self._index: DirectoryIndex | None = None
        self._results: dict[Path, tuple[Candidate, ProbeResult]] = {}
        self._visible_candidates: tuple[Candidate, ...] = ()
        self._retry_deadline = retry_deadline
        self._max_probes_per_observation = int(max_probes_per_observation)
        self._probe_time_budget_s = float(probe_time_budget_s)
        self._probe_candidates = bool(probe_candidates)
        self._probe_excluded: set[Candidate] = set()
        # A lazy (non-probing) session returns its last observation again while
        # the index returns the same Snapshot and no probe result moved.
        self._results_version = 0
        self._last: tuple[Snapshot, int, DirectoryObservation] | None = None
        self._paths: tuple[Snapshot, Mapping[Path, Candidate]] | None = None
        # D1: the paths a Live Run watches (None: off), a forced listing, and
        # the last real listing's start with what a deferral must find unchanged.
        self._watched: tuple[Path, ...] | None = None
        self._force = False
        self._listed: tuple[float, tuple, tuple] | None = None
        self.listing_deferred = False
        # The paths the last refreshing observation saw first: a listing concurrent with
        # writes can show a newer file and miss an older one, so a Live Run admits a
        # file only once a later listing has seen it again.
        self.fresh: frozenset[Path] = frozenset()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="xdart-directory-index")
        self._closed = False

    @property
    def request_generation(self) -> int:
        with self._lock:
            return self._request_generation

    @property
    def configured(self) -> DirectorySessionConfig | None:
        with self._lock:
            return self._desired

    def configure(
        self, root, *, recursive=False, name_filter=None, suffixes=(), series=None,
    ) -> int:
        config = DirectorySessionConfig.build(
            root, recursive=recursive, name_filter=name_filter,
            suffixes=suffixes, series=series)
        with self._lock:
            if self._closed:
                raise RuntimeError("directory index session is closed")
            if config != self._desired:
                self._desired = config
                self._request_generation += 1
            return self._request_generation

    def clear(self) -> int:
        with self._lock:
            if self._closed:
                return self._request_generation
            if self._desired is not None:
                self._desired = None
                self._request_generation += 1
            return self._request_generation

    def observe_async(self, *, refresh: bool = True) -> Future:
        """Observe the configured directory.

        ``refresh=True`` performs the normal name/stat poll before draining a
        bounded probe batch.  ``refresh=False`` drains more candidates from the
        already-discovered immutable snapshot, so a large recursive tree is not
        walked again for every eight-file batch.
        """
        generation, config = self._request()
        return self._executor.submit(
            self._observe_on_owner, generation, config, bool(refresh))

    def observe(self, *, refresh: bool = True) -> DirectoryObservation:
        return self.observe_async(refresh=refresh).result()

    def probe_candidate(
        self,
        candidate: Candidate,
        *,
        refresh: bool = True,
    ) -> CandidateObservation:
        """Probe exactly one frozen candidate on the serialized owner.

        Large Standard directory runs use this cursor-shaped operation so
        content readiness and container frame counts are learned only when
        that candidate reaches execution.  The optional refresh revalidates
        its cheap ``(path, adapter, size, mtime)`` identity first; no sibling
        candidate is probed as a side effect.
        """

        if type(candidate) is not Candidate:
            raise TypeError("candidate probe requires an exact Candidate")
        generation, config = self._request()
        return self._executor.submit(
            self._probe_candidate_on_owner,
            generation,
            config,
            candidate,
            bool(refresh),
            False,
        ).result()

    def reprobe_candidate(
        self,
        candidate: Candidate,
        *,
        refresh: bool = True,
    ) -> CandidateObservation:
        """Reprobe one still-exact candidate after proven dependency drift.

        Only that candidate's terminal/result cache is discarded, on the
        serialized session owner, after its current generation, cheap stamp,
        and adapter identity have all been revalidated.
        """

        if type(candidate) is not Candidate:
            raise TypeError("candidate reprobe requires an exact Candidate")
        generation, config = self._request()
        return self._executor.submit(
            self._probe_candidate_on_owner,
            generation,
            config,
            candidate,
            bool(refresh),
            True,
        ).result()

    def enable_probes(self, *, exclude: tuple[Candidate, ...] = ()) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("directory index session is closed")
            self._probe_candidates = True
            self._probe_excluded = set(exclude)

    def _request(self) -> tuple[int, DirectorySessionConfig]:
        with self._lock:
            if self._closed:
                raise RuntimeError("directory index session is closed")
            if self._desired is None:
                raise RuntimeError("directory index session is not configured")
            return self._request_generation, self._desired

    def _observe_on_owner(
        self, generation: int, config: DirectorySessionConfig, refresh: bool,
    ) -> DirectoryObservation:
        started = time.perf_counter()
        reconfigured = self._index is None or self._active != config
        if reconfigured:
            kwargs = {}
            if self._retry_deadline is not None:
                kwargs["retry_deadline"] = float(self._retry_deadline)
            self._index = DirectoryIndex(
                config.root,
                recursive=config.recursive,
                name_filter=None,
                **kwargs,
            )
            self._active = config
            self._results.clear()
            self._results_version += 1
            self._visible_candidates = ()
            self._last = None

        index = self._index
        deferred = refresh and self._defers(index, reconfigured)
        if (refresh and not deferred) or reconfigured or index.snapshot.generation == 0:
            if self._watched is not None:
                directories = self._directories(index)
                self._listed = (time.monotonic(), directories, self._watch(directories))
            snapshot = index.poll()
        else:
            snapshot = index.snapshot
        last = self._last
        if (last is not None and not self._probe_candidates and last[0] is snapshot
                and last[1] == self._results_version and last[2].request_generation == generation):
            # Same Snapshot object, no probe result moved and nothing to probe:
            # exactly the observation a rebuild would make, with an empty delta.
            self.fresh = frozenset() if refresh else self.fresh
            return replace(last[2], delta=IndexDelta(), content_opens=0, stale_drops=0,
                           elapsed_s=time.perf_counter() - started)
        raw_snapshot = snapshot
        name_ok = compile_filter(config.name_filter)

        def included(candidate: Candidate) -> bool:
            name = candidate.path.name
            if config.series is not None and config.series.key(name) is None:
                return False
            low = name.lower()
            suffix = next(
                (item for item in config.suffixes if low.endswith(item)), "")
            if config.suffixes and not suffix:
                return False
            base = name[:-len(suffix)] if suffix else name
            return name_ok(base)

        candidates = tuple(filter(included, snapshot.candidates))
        snapshot = Snapshot(
            snapshot.generation, candidates, snapshot.root,
            snapshot.recursive, config.name_filter)
        current = snapshot.by_path()
        self._prune_results(current)

        content_opens = 0
        stale_drops = 0
        priority_paths = {
            candidate.path
            for candidate in (*index.last_delta.changed, *index.last_delta.added)
        }
        unseen = []
        retrying_candidates = []
        for candidate in snapshot.candidates:
            stored = self._results.get(candidate.path)
            retrying = index.retry_state(candidate.path) is not None
            if stored is None and candidate not in self._probe_excluded:
                unseen.append(candidate)
            elif retrying:
                retrying_candidates.append(candidate)

        # New/changed files get their first readiness observation before a
        # nascent shell is retried.  This prevents one slow writer from
        # starving later ready siblings.  Files from the current poll delta
        # retain natural order at the front of the unseen queue.
        unseen.sort(key=lambda item: item.path not in priority_paths)
        probe_started = time.perf_counter()
        if self._probe_candidates:
            for candidate in (*unseen, *retrying_candidates):
                if content_opens >= self._max_probes_per_observation:
                    break
                if (
                    content_opens
                    and time.perf_counter() - probe_started
                    >= self._probe_time_budget_s
                ):
                    break
                try:
                    content_opens += 1
                    result = index.probe_candidate(candidate)
                except StaleCandidateError:
                    stale_drops += 1
                    continue
                except Exception as exc:
                    # Cache an unexpected failure against this exact candidate
                    # identity so one defective source cannot poison every ready
                    # sibling. A later byte/owner change invalidates it normally.
                    logger.warning(
                        "Source readiness probe failed for %s: %s",
                        candidate.path,
                        exc,
                    )
                    result = ProbeResult(
                        ProbeState.INVALID,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                self._results[candidate.path] = (candidate, result)
                self._results_version += 1

        # A probe can race a file mutation or owner flip.  Re-poll before
        # publishing so no stale candidate identity reaches the GUI or Run.
        if stale_drops:
            snapshot = raw_snapshot = index.poll()
            candidates = tuple(filter(included, snapshot.candidates))
            snapshot = Snapshot(
                snapshot.generation, candidates, snapshot.root,
                snapshot.recursive, config.name_filter)
            current = snapshot.by_path()
            self._prune_results(current)

        prior_by_path = {
            candidate.path: candidate for candidate in self._visible_candidates
        }
        current_by_path = snapshot.by_path()
        delta = IndexDelta(
            added=tuple(
                candidate for candidate in snapshot.candidates
                if candidate.path not in prior_by_path),
            changed=tuple(
                candidate for candidate in snapshot.candidates
                if candidate.path in prior_by_path
                and prior_by_path[candidate.path] != candidate),
            removed=tuple(
                candidate.path for candidate in self._visible_candidates
                if candidate.path not in current_by_path),
        )
        self._visible_candidates = snapshot.candidates
        self.fresh = frozenset(candidate.path for candidate in delta.added) if refresh else self.fresh

        queued_result = ProbeResult(
            ProbeState.IN_PROGRESS,
            reason="awaiting bounded readiness probe",
        )
        observations = tuple(
            CandidateObservation(
                candidate,
                self._results[candidate.path][1]
                if candidate.path in self._results
                else queued_result,
            )
            for candidate in snapshot.candidates
        )
        unprobed_count = sum(
            candidate.path not in self._results
            and candidate not in self._probe_excluded
            for candidate in snapshot.candidates
        )
        ready_candidates = tuple(
            item.candidate for item in observations
            if item.result.state is ProbeState.READY
        )
        ready_snapshot = Snapshot(
            snapshot.generation,
            ready_candidates,
            snapshot.root,
            snapshot.recursive,
            snapshot.name_filter,
        )
        observation = DirectoryObservation(
            request_generation=generation,
            discovered_snapshot=snapshot,
            ready_snapshot=ready_snapshot,
            candidates=observations,
            delta=delta,
            content_opens=content_opens,
            stale_drops=stale_drops,
            elapsed_s=time.perf_counter() - started,
            unprobed_count=unprobed_count,
        )
        self._last = (raw_snapshot, self._results_version, observation)
        return observation

    def _prune_results(self, current: Mapping[Path, Candidate]) -> None:
        kept = {path: stored for path, stored in self._results.items() if current.get(path) == stored[0]}
        if len(kept) != len(self._results):
            self._results_version += 1
        self._results = kept

    def _probe_candidate_on_owner(
        self,
        generation: int,
        config: DirectorySessionConfig,
        candidate: Candidate,
        refresh: bool,
        force_reprobe: bool,
    ) -> CandidateObservation:
        # A lazy session is configured with probing disabled.  Reuse the same
        # owner to refresh name/stat identity, then call the index for exactly
        # the requested candidate rather than draining the directory queue.
        observation = self._observe_on_owner(
            generation,
            config,
            refresh,
        )
        return self._probe_exact(observation, candidate, force_reprobe)

    def _probe_exact(self, observation: DirectoryObservation, candidate: Candidate,
                     force_reprobe: bool) -> CandidateObservation:
        snapshot = observation.discovered_snapshot
        if self._paths is None or self._paths[0] is not snapshot:
            self._paths = (snapshot, snapshot.by_path())
        current = self._paths[1].get(candidate.path)
        if current != candidate:
            raise StaleCandidateError(
                f"{candidate.path} changed before its deferred probe"
            )
        index = self._index
        if index is None:  # pragma: no cover - owner construction invariant
            raise RuntimeError("directory index session has no active index")
        if force_reprobe:
            index.discard_probe_resolution(current)
            if self._results.pop(current.path, None) is not None:
                self._results_version += 1
        stored = self._results.get(candidate.path)
        retrying = index.retry_state(candidate.path) is not None
        if stored is not None and stored[0] == current and not retrying:
            return CandidateObservation(current, stored[1])
        result = index.probe_candidate(current)
        self._results[current.path] = (current, result)
        self._results_version += 1
        return CandidateObservation(current, result)

    def probe_candidates(self, candidates, *, refresh: bool = False, reprobe: bool = False,
                         stop_unready: bool = True, cancelled=None) -> tuple[CandidateObservation, ...]:
        """:meth:`probe_candidate` (with ``reprobe``, :meth:`reprobe_candidate`) for each candidate in
        order, against one observation in one owner call; ``stop_unready`` ends after the first that is
        not READY, a stale one raises once the earlier are recorded, and ``cancelled`` is asked before each."""
        candidates = tuple(candidates)
        if any(type(candidate) is not Candidate for candidate in candidates):
            raise TypeError("candidate probe requires an exact Candidate")
        generation, config = self._request()
        return self._executor.submit(self._probe_candidates_on_owner, generation, config, candidates,
                                     bool(refresh), bool(reprobe), bool(stop_unready), cancelled).result()

    def _probe_candidates_on_owner(self, generation, config, candidates, refresh, reprobe, stop_unready,
                                   cancelled) -> tuple[CandidateObservation, ...]:
        observation = self._observe_on_owner(generation, config, refresh)
        observed: list[CandidateObservation] = []
        for candidate in candidates:
            if cancelled is not None and cancelled():
                raise RuntimeError("admission cancelled")
            value = self._probe_exact(observation, candidate, reprobe)
            observed.append(value)
            if stop_unready and value.result.state is not ProbeState.READY:
                break
        return tuple(observed)

    #: A deferred listing (D1) is younger than this, so a real one follows
    #: within a poll and its work more: inside 1 s.
    LISTING_AGE_S = 0.75

    def defer_listing(self, watched) -> None:
        """D1, a Live Run's session: a refreshing observation keeps the last real listing while it is
        under LISTING_AGE_S old and the registrations, the watched directories and ``watched`` files
        (pending or growing in place) are unchanged; ``None`` turns it off."""
        self._watched = None if watched is None else tuple(watched)

    def force_listing(self) -> None:
        self._force = True

    @staticmethod
    def _directories(index) -> tuple:
        """The watched directories: the root, and with Subdirs its immediate subfolders (the bounded plan)."""
        if not index.snapshot.recursive:
            return (index.root,)
        try:
            with os.scandir(index.root) as entries:
                return (index.root, *sorted(entry.path for entry in entries if entry.is_dir()))
        except OSError:
            return (index.root,)

    def _watch(self, directories) -> tuple:
        from xrd_tools.sources.adapters import registry_identity
        return (registry_identity(), *map(directory_index.directory_identity, directories),
                *((path, directory_index.directory_identity(path)) for path in self._watched or ()))

    def _defers(self, index, reconfigured: bool) -> bool:
        from xrd_tools.sources.registry import name_pure_registry  # only name-pure owners may skip a listing
        forced, self._force, listed = self._force, False, self._listed
        self.listing_deferred = not (forced or reconfigured) and self._watched is not None and (
            listed is not None and index.snapshot.generation != 0 and name_pure_registry()
            and time.monotonic() - listed[0] < self.LISTING_AGE_S and listed[2] == self._watch(listed[1]))
        return self.listing_deferred

    def reuse_records(self) -> tuple[tuple, ...]:
        """One record per path kept between polls for reuse (discovery's memo, the path map)."""
        def records():
            memo = {} if self._index is None else self._index._discovery
            paths = () if self._paths is None else self._paths[1]
            return tuple((None, str(path)) for path in (
                *memo.get("candidates", ()), *memo.get("owners", ()), *paths))
        return self._executor.submit(records).result()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._desired = None
            self._request_generation += 1
        self._executor.shutdown(wait=False, cancel_futures=True)


__all__ = [
    "CandidateObservation",
    "DirectoryIndexSession",
    "DirectoryObservation",
    "DirectorySessionConfig",
]
