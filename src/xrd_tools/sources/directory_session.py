# -*- coding: utf-8 -*-
"""Serialized, value-only ownership for one persistent ``DirectoryIndex``.

The index itself is mutable and therefore never leaves the session's single
worker thread.  GUI and processing callers receive immutable observations and
may share this session without racing directory polls or probe bookkeeping.
"""

from __future__ import annotations

import threading
import time
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from xrd_tools.sources.directory_index import (
    DirectoryIndex,
    IndexDelta,
    Snapshot,
    StaleCandidateError,
)
from xrd_tools.sources.discover import Candidate
from xrd_tools.sources.probe import ProbeResult, ProbeState
from xrd_tools.core.filters import compile_filter


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DirectorySessionConfig:
    root: Path
    recursive: bool = False
    name_filter: str | None = None
    suffixes: tuple[str, ...] = ()

    @classmethod
    def build(cls, root, *, recursive=False, name_filter=None, suffixes=()):
        normalized = tuple(str(item).lower() for item in suffixes if item)
        return cls(
            Path(root).expanduser(), bool(recursive), name_filter or None,
            normalized)


@dataclass(frozen=True, slots=True)
class CandidateObservation:
    candidate: Candidate
    result: ProbeResult


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
        self, root, *, recursive=False, name_filter=None, suffixes=(),
    ) -> int:
        config = DirectorySessionConfig.build(
            root, recursive=recursive, name_filter=name_filter,
            suffixes=suffixes)
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

    def observe_async(self) -> Future:
        generation, config = self._request()
        return self._executor.submit(self._observe_on_owner, generation, config)

    def observe(self) -> DirectoryObservation:
        return self.observe_async().result()

    def _request(self) -> tuple[int, DirectorySessionConfig]:
        with self._lock:
            if self._closed:
                raise RuntimeError("directory index session is closed")
            if self._desired is None:
                raise RuntimeError("directory index session is not configured")
            return self._request_generation, self._desired

    def _observe_on_owner(
        self, generation: int, config: DirectorySessionConfig,
    ) -> DirectoryObservation:
        started = time.perf_counter()
        if self._index is None or self._active != config:
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
            self._visible_candidates = ()

        index = self._index
        snapshot = index.poll()
        name_ok = compile_filter(config.name_filter)

        def included(candidate: Candidate) -> bool:
            name = candidate.path.name
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
        self._results = {
            path: stored
            for path, stored in self._results.items()
            if current.get(path) == stored[0]
        }

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
            if stored is None:
                unseen.append(candidate)
            elif retrying:
                retrying_candidates.append(candidate)

        # New/changed files get their first readiness observation before a
        # nascent shell is retried.  This prevents one slow writer from
        # starving later ready siblings.  Files from the current poll delta
        # retain natural order at the front of the unseen queue.
        unseen.sort(key=lambda item: item.path not in priority_paths)
        probe_started = time.perf_counter()
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

        # A probe can race a file mutation or owner flip.  Re-poll before
        # publishing so no stale candidate identity reaches the GUI or Run.
        if stale_drops:
            snapshot = index.poll()
            candidates = tuple(filter(included, snapshot.candidates))
            snapshot = Snapshot(
                snapshot.generation, candidates, snapshot.root,
                snapshot.recursive, config.name_filter)
            current = snapshot.by_path()
            self._results = {
                path: stored
                for path, stored in self._results.items()
                if current.get(path) == stored[0]
            }

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
        return DirectoryObservation(
            request_generation=generation,
            discovered_snapshot=snapshot,
            ready_snapshot=ready_snapshot,
            candidates=observations,
            delta=delta,
            content_opens=content_opens,
            stale_drops=stale_drops,
            elapsed_s=time.perf_counter() - started,
        )

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
