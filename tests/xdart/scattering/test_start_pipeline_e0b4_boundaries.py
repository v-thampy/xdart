"""Exact-object adversaries for total E0b lifecycle boundaries."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy

import pytest

from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xdart.gui.tabs.scattering.contracts import SourceCapture
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.events import (
    ExecutorAccepted,
    LifecycleResult,
    LifecycleStatus,
    RequestId,
)
from xdart.gui.tabs.scattering.start_outcomes import (
    StartCapture,
    StartFailed,
    StartRefusal,
    StartRefused,
    exception_detail,
)
from xdart.gui.tabs.scattering.start_pipeline import StartPipeline
from tests.xdart.scattering._admission import admission_for


def _intent() -> RunIntent:
    return RunIntent(
        source_spec=SourceSpec(Path("/data/frame.tif"), SourceKind.IMAGE_FILE),
    )


class Source:
    def __init__(
        self,
        capture_error: Exception | None = None,
        cancel_error: Exception | None = None,
    ) -> None:
        self.capture_error = capture_error
        self.cancel_error = cancel_error
        self.capture_calls = 0
        self.captured_requests = []
        self.cancelled = []

    def capture(self, source, request_id):
        self.capture_calls += 1
        self.captured_requests.append(request_id)
        if self.capture_error is not None:
            raise self.capture_error
        return SourceCapture(request_id, 0, source, None)

    def cancel(self, request_id):
        self.cancelled.append(request_id)
        if self.cancel_error is not None:
            raise self.cancel_error


class Executor:
    def __init__(self, error: Exception | None = None, lifecycle=None) -> None:
        self.error = error
        self.lifecycle = lifecycle
        self.calls = []
        self.closed = []
        self.request_is_owned = []

    def start(self, configuration, source, run_identity, admission):
        self.calls.append(run_identity)
        if self.lifecycle is not None:
            self.request_is_owned.append(run_identity is self.lifecycle.attempt_run_identity)
        if self.error is not None:
            raise self.error
        return ExecutorAccepted(run_identity)

    def close(self, run_identity):
        self.closed.append(run_identity)

    def pause(self, run_identity): ...
    def resume(self, run_identity): ...
    def stop(self, run_identity): ...


def _pipeline(*, lifecycle=None, source=None, executor=None):
    lifecycle = lifecycle or ScatteringCoordinator()
    source = source or Source()
    executor = executor or Executor()
    return (
        StartPipeline(
            intents=RunIntentStore(_intent()),
            lifecycle=lifecycle,
            sources=source,
            executor=executor,
        ),
        lifecycle,
        source,
        executor,
    )


class ModuleNameBomb:
    def __format__(self, spec):
        raise RuntimeError("exception module formatting escaped")


class PoisonedSourceError(RuntimeError):
    __module__ = ModuleNameBomb()


class MetadataBomb(type):
    def __getattribute__(cls, name):
        if name in {"__module__", "__qualname__"}:
            raise RuntimeError("exception type metadata escaped")
        return super().__getattribute__(name)


class PoisonedMetadataError(RuntimeError, metaclass=MetadataBomb):
    pass


class MessageBomb(RuntimeError):
    def __str__(self):
        raise RuntimeError("exception message formatting escaped")


@pytest.mark.parametrize("error", [PoisonedSourceError("primary"), PoisonedMetadataError("primary"), MessageBomb()])
def test_exception_projection_is_total_on_the_real_source_refusal_path(error):
    pipeline, lifecycle, source, _ = _pipeline(source=Source(capture_error=error))

    result = pipeline.begin()

    assert type(result) is StartRefused
    assert lifecycle.phase.value == "idle"
    assert source.cancelled
    assert type(result.exception.type_module) is str
    assert type(result.exception.type_qualname) is str
    assert type(result.exception.message) is str


@pytest.mark.parametrize("error", [PoisonedMetadataError("cleanup"), MessageBomb()])
def test_exception_projection_is_total_on_the_real_source_cleanup_path(error):
    pipeline, _, source, _ = _pipeline(source=Source(cancel_error=error))
    capture = pipeline.begin()
    assert type(capture) is StartCapture

    result = pipeline.refuse(capture, StartRefusal.MISSING_SOURCE)

    assert type(result) is StartRefused
    assert source.cancelled == [capture.request_id]
    detail = result.recovery_failures[0].exception
    assert all(type(field) is str for field in (detail.type_module, detail.type_qualname, detail.message))


def test_exception_detail_contains_hostile_type_metadata_without_escaping():
    detail = exception_detail(PoisonedMetadataError("primary"))

    assert type(detail.type_module) is str
    assert type(detail.type_qualname) is str
    assert type(detail.message) is str


class CloneRequestLifecycle(ScatteringCoordinator):
    def begin_start(self):
        result = super().begin_start()
        return replace(result, request_id=RequestId(result.request_id.value))


def test_begin_rejects_equal_request_clone_before_snapshot_or_source_work():
    source = Source()
    executor = Executor()
    pipeline, lifecycle, _, _ = _pipeline(
        lifecycle=CloneRequestLifecycle(), source=source, executor=executor
    )

    result = pipeline.begin()

    assert type(result) is StartFailed
    assert lifecycle.closed is True
    assert source.capture_calls == 0
    assert source.cancelled == []
    assert executor.calls == []


def test_source_capture_and_public_capture_keep_the_owned_request_identity():
    pipeline, lifecycle, source, _ = _pipeline()

    capture = pipeline.begin()

    assert type(capture) is StartCapture
    assert source.captured_requests == [lifecycle.request_id]
    assert capture.request_id is lifecycle.request_id


class PhaseBombLifecycle(ScatteringCoordinator):
    @property
    def phase(self):
        raise RuntimeError("phase projection escaped")

    def contain_executor_failure(self, run_identity):
        return object()


class IdentityBombLifecycle(ScatteringCoordinator):
    @property
    def attempt_run_identity(self):
        raise RuntimeError("identity projection escaped")


def test_malformed_containment_and_phase_projection_still_release_owners():
    executor = Executor(error=RuntimeError("executor failed"))
    pipeline, _, source, _ = _pipeline(
        lifecycle=PhaseBombLifecycle(), executor=executor
    )
    capture = pipeline.begin()
    assert type(capture) is StartCapture

    failed = pipeline.start(admission_for(capture))

    assert type(failed) is StartFailed
    assert type(failed.lifecycle_result) is LifecycleResult
    assert source.cancelled == [capture.request_id]
    assert executor.closed == executor.calls


@pytest.mark.parametrize("lifecycle_type", [PhaseBombLifecycle, IdentityBombLifecycle])
def test_malformed_owners_closed_is_inert_when_lifecycle_projections_raise(lifecycle_type):
    pipeline, lifecycle, _, _ = _pipeline(lifecycle=lifecycle_type())
    capture = pipeline.begin()
    assert type(capture) is StartCapture

    result = pipeline.owners_closed(object())

    assert type(result) is LifecycleResult
    assert result.status is LifecycleStatus.REJECTED
    assert lifecycle.event_sequence == 1


def test_semantic_guard_tracks_keywords_starred_kwargs_and_nested_literals():
    guard = runpy.run_path("tests/xdart/scattering/test_architecture.py")["_guard_violations"]
    rejected = (
        "def decide(candidate: RunIntent):\n"
        "    sink.commit(candidate=candidate, expected_revision=0)\n",
        "def decide(candidate: RunIntent):\n"
        "    sink.publish(*[candidate])\n",
        "def decide(candidate: RunIntent):\n"
        "    sink.publish(**{'candidate': candidate})\n",
        "def decide(candidate: RunIntent):\n"
        "    sink.publish({'nested': [candidate]})\n",
    )
    for source in rejected:
        assert "raw RunIntent transfer" in guard(source)
    allowed = (
        "def decide(candidate: RunIntent, store: RunIntentStore):\n"
        "    store.commit(candidate, expected_revision=0)\n",
        "def decide(candidate: RunIntent, store: RunIntentStore):\n"
        "    store.commit(candidate=candidate, expected_revision=0)\n",
    )
    for source in allowed:
        assert not guard(source)


def test_semantic_guard_rejects_short_authority_aliases_not_capture_sequence():
    guard = runpy.run_path("tests/xdart/scattering/test_architecture.py")["_guard_violations"]

    assert "local revision authority" in guard(
        "self._intent_rev = 1\nself._source_gen = 2\nself._run_gen = 3\n"
    )
    assert "local revision authority" not in guard("self._capture_sequence = 1\n")
