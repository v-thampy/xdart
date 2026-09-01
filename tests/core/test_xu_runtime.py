from __future__ import annotations

import copy
from threading import Event, Thread

import pytest

from xrd_tools.core.geometry.xu_runtime import (
    XU_RUNTIME_LOCK,
    XuRuntimeExecutionRecord,
    xu_runtime_availability,
    xu_runtime_session,
)


def test_xu_runtime_availability_is_engine_light(monkeypatch):
    imported = []
    real_import = __import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] == "xrayutilities":
            imported.append(name)
            raise AssertionError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)
    availability = xu_runtime_availability()
    assert availability.available is True
    assert availability.code == "OK"
    assert imported == []


def test_xu_runtime_sets_one_and_restores_on_success_and_error():
    from xrayutilities import config

    before = config.NTHREADS
    session = xu_runtime_session()
    with session:
        assert config.NTHREADS == 1
        assert session.execution_record is None
        with pytest.raises(TypeError, match="not copyable"):
            copy.copy(session)
    assert config.NTHREADS == before
    record = session.execution_record
    assert type(record) is XuRuntimeExecutionRecord
    assert record.nthreads_before == before
    assert record.nthreads_effective == 1
    assert record.nthreads_restored == before
    assert record.restore_passed is True
    assert record.to_attestation()["lock_policy"] == (
        "shared_xrd_tools_xu_rlock_v1"
    )

    failed = xu_runtime_session()
    with pytest.raises(RuntimeError, match="body failure"):
        with failed:
            assert config.NTHREADS == 1
            raise RuntimeError("body failure")
    assert config.NTHREADS == before
    assert failed.execution_record.nthreads_restored == before


def test_xu_runtime_is_reentrant_and_serializes_threads():
    from xrayutilities import config

    before = config.NTHREADS
    entered_outer = Event()
    release_outer = Event()
    entered_second = Event()
    failures = []

    def outer():
        try:
            with xu_runtime_session():
                entered_outer.set()
                with xu_runtime_session():
                    assert config.NTHREADS == 1
                release_outer.wait(5)
        except BaseException as error:
            failures.append(error)

    def second():
        try:
            entered_outer.wait(5)
            with xu_runtime_session():
                entered_second.set()
        except BaseException as error:
            failures.append(error)

    first_thread = Thread(target=outer)
    second_thread = Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert entered_outer.wait(5)
    assert not entered_second.wait(0.05)
    release_outer.set()
    first_thread.join(5)
    second_thread.join(5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert entered_second.is_set()
    assert failures == []
    assert config.NTHREADS == before
    assert XU_RUNTIME_LOCK.acquire(timeout=1)
    XU_RUNTIME_LOCK.release()
