from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread

import pytest

from xrd_tools.core.geometry.xu_runtime import (
    XU_RUNTIME_EFFECTIVE_NTHREADS,
    XU_RUNTIME_LOCK,
    XU_RUNTIME_LOCK_POLICY,
    XuRuntimeExecutionRecord,
    XuRuntimeRequirements,
    XuRuntimeUnsupported,
    require_active_xu_runtime_session,
    xu_runtime_availability,
    xu_runtime_requirements_projection,
    xu_runtime_session,
)


def test_xu_runtime_requirements_are_the_exact_validated_platform_contract():
    assert XuRuntimeRequirements().distribution_version == "1.7.12"
    with pytest.raises(TypeError, match="requirements are invalid"):
        XuRuntimeRequirements(distribution_version="future")
    with pytest.raises(TypeError, match="requirements are invalid"):
        XuRuntimeRequirements(platform_machine="x86_64")


def test_xu_runtime_requirements_projection_is_the_exact_effective_contract():
    assert XU_RUNTIME_LOCK_POLICY == "shared_xrd_tools_xu_rlock_v1"
    assert XU_RUNTIME_EFFECTIVE_NTHREADS == 1
    assert xu_runtime_requirements_projection(XuRuntimeRequirements()) == (
        "1.7.12",
        "1.7.12",
        "2.5.1",
        1e-8,
        8,
        "CPython",
        "3.13.14",
        "Darwin",
        "arm64",
        XU_RUNTIME_LOCK_POLICY,
        XU_RUNTIME_EFFECTIVE_NTHREADS,
    )
    with pytest.raises(TypeError, match="must be exact"):
        xu_runtime_requirements_projection(object())


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


def test_xu_runtime_availability_normalizes_metadata_failure(monkeypatch):
    import importlib.metadata

    def fail(_name):
        raise RuntimeError("metadata backend failed")

    monkeypatch.setattr(importlib.metadata, "version", fail)
    availability = xu_runtime_availability()
    assert availability.available is False
    assert availability.code == "XU_RUNTIME_UNSUPPORTED"


def test_xu_runtime_session_normalizes_import_failure_and_releases_lock(monkeypatch):
    real_import = __import__

    def fail_xu(name, *args, **kwargs):
        if name == "xrayutilities":
            raise RuntimeError("import backend failed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fail_xu)
    with pytest.raises(XuRuntimeUnsupported) as raised:
        with xu_runtime_session():
            pass
    assert raised.value.code == "XU_RUNTIME_UNSUPPORTED"
    assert XU_RUNTIME_LOCK.acquire(timeout=1)
    XU_RUNTIME_LOCK.release()


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


def test_xu_runtime_active_modules_are_owner_thread_only_and_expire():
    session = xu_runtime_session()
    failures = []
    with session:
        assert session.active is True
        xu_module, numpy_module = session.active_modules()
        assert xu_module.__version__ == "1.7.12"
        assert numpy_module.__version__ == "2.5.1"
        assert require_active_xu_runtime_session(session) is session

        def foreign_thread():
            try:
                session.active_modules()
            except BaseException as error:
                failures.append(error)

        thread = Thread(target=foreign_thread)
        thread.start()
        thread.join(5)
        assert not thread.is_alive()
        exit_failures = []

        def foreign_exit():
            try:
                session.__exit__(None, None, None)
            except BaseException as error:
                exit_failures.append(error)

        exit_thread = Thread(target=foreign_exit)
        exit_thread.start()
        exit_thread.join(5)
        assert not exit_thread.is_alive()
        assert len(exit_failures) == 1
        assert type(exit_failures[0]) is XuRuntimeUnsupported
        assert exit_failures[0].code == "XU_RUNTIME_SESSION_INACTIVE"
        assert session.active is True
    assert session.active is False
    assert len(failures) == 1
    assert type(failures[0]) is XuRuntimeUnsupported
    assert failures[0].code == "XU_RUNTIME_SESSION_INACTIVE"
    with pytest.raises(XuRuntimeUnsupported) as raised:
        session.active_modules()
    assert raised.value.code == "XU_RUNTIME_SESSION_INACTIVE"
    with pytest.raises(TypeError, match="must be exact"):
        require_active_xu_runtime_session(object())


def test_importing_rsm_gridding_does_not_import_xrayutilities():
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.fspath(repository / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import xrd_tools.rsm.gridding; "
                "assert 'xrayutilities' not in sys.modules"
            ),
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
