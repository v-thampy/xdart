"""P-1L stable-HDF body/close precedence oracles."""

from __future__ import annotations

from pathlib import Path
from threading import Event

import h5py
import pytest

from tests.xdart.scattering.test_e6_prestamp_hdf_boundaries import (
    _adaptive_stable_open,
)
from xdart.gui.tabs.scattering import output_preflight
from xdart.gui.tabs.scattering.contracts import SourceFileState
from xrd_tools.sources import execution_graph as source_graph


_CLOSE_SENTINEL = "P-1L stable close sentinel"
_CLOSE_NOTE = (
    "secondary HDF5 close failure: "
    f"OSError: {_CLOSE_SENTINEL}"
)


class _CloseHandle:
    def __init__(
        self,
        *,
        on_close=lambda: None,
        close_error: BaseException | None = None,
        truthy_exit: bool = False,
    ) -> None:
        self._on_close = on_close
        self._close_error = close_error
        self._truthy_exit = truthy_exit
        self.enter_calls = 0
        self.exit_calls = 0
        self.close_calls = 0

    def __enter__(self):
        self.enter_calls += 1
        return self

    def close(self) -> None:
        self.close_calls += 1
        self._on_close()
        if self._close_error is not None:
            raise self._close_error

    def __exit__(self, _kind, _error, _traceback):
        self.exit_calls += 1
        self.close()
        return self._truthy_exit


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "dependency.h5"
    path.write_bytes(b"stable source bytes")
    return path


def _install_handle(
    monkeypatch: pytest.MonkeyPatch,
    handle: _CloseHandle,
) -> None:
    monkeypatch.setattr(h5py, "File", lambda *_args, **_kwargs: handle)


def _capture_after_close(
    monkeypatch: pytest.MonkeyPatch,
    handle: _CloseHandle,
) -> list[Path]:
    captures: list[Path] = []
    real_capture = source_graph._capture_source_topology

    def counted(path, *args, **kwargs):
        if handle.close_calls:
            captures.append(Path(path))
        return real_capture(path, *args, **kwargs)

    monkeypatch.setattr(output_preflight, "_capture_source_topology", counted)
    return captures


def test_body_proved_drift_precedes_failing_close_and_keeps_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L01: proved body drift wins; existing cause and close evidence survive."""

    source = _source(tmp_path)
    close_error = OSError(_CLOSE_SENTINEL)
    scientific_cause = RuntimeError("P-1L existing source cause")
    drift = output_preflight.SourceRevisionChanged("P-1L body drift")
    drift.__cause__ = scientific_cause
    handle = _CloseHandle(close_error=close_error)
    _install_handle(monkeypatch, handle)
    captures = _capture_after_close(monkeypatch, handle)

    with pytest.raises(output_preflight.SourceRevisionChanged) as caught:
        with _adaptive_stable_open(source, {}, lambda: False):
            raise drift

    assert caught.value is drift
    assert drift.__cause__ is scientific_cause
    assert getattr(drift, "__notes__", ()) == [_CLOSE_NOTE]
    assert handle.close_calls == 1
    assert captures == []


def test_body_exact_cancellation_precedes_failing_close(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L02: exact body cancellation wins by identity without recapture."""

    source = _source(tmp_path)
    close_error = OSError(_CLOSE_SENTINEL)
    cancellation = RuntimeError("admission cancelled")
    handle = _CloseHandle(close_error=close_error)
    _install_handle(monkeypatch, handle)
    captures = _capture_after_close(monkeypatch, handle)

    with pytest.raises(RuntimeError, match="^admission cancelled$") as caught:
        with _adaptive_stable_open(source, {}, lambda: False):
            raise cancellation

    assert caught.value is cancellation
    assert cancellation.__cause__ is close_error
    assert handle.close_calls == 1
    assert captures == []


def test_ordinary_body_precedes_failing_close_when_topology_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L03: stable ordinary body failure is primary and keeps close evidence."""

    source = _source(tmp_path)
    close_error = OSError(_CLOSE_SENTINEL)
    body_error = ValueError("P-1L ordinary body failure")
    handle = _CloseHandle(close_error=close_error)
    _install_handle(monkeypatch, handle)
    captures = _capture_after_close(monkeypatch, handle)

    with pytest.raises(ValueError, match="P-1L ordinary body failure") as caught:
        with _adaptive_stable_open(source, {}, lambda: False):
            raise body_error

    assert caught.value is body_error
    assert body_error.__cause__ is close_error
    assert handle.close_calls == 1
    assert captures == [source]


def test_body_proved_drift_precedes_close_stop_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L04: proved body drift precedes Stop raised by mandatory close."""

    source = _source(tmp_path)
    stopped = Event()
    close_error = OSError(_CLOSE_SENTINEL)
    drift = output_preflight.SourceRevisionChanged("P-1L body drift")
    handle = _CloseHandle(on_close=stopped.set, close_error=close_error)
    _install_handle(monkeypatch, handle)
    captures = _capture_after_close(monkeypatch, handle)

    with pytest.raises(output_preflight.SourceRevisionChanged) as caught:
        with _adaptive_stable_open(source, {}, stopped.is_set):
            raise drift

    assert stopped.is_set()
    assert caught.value is drift
    assert drift.__cause__ is close_error
    assert handle.close_calls == 1
    assert captures == []


def test_truthy_exit_cannot_suppress_ordinary_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L05: cleanup owns close directly and never enters the handle context."""

    source = _source(tmp_path)
    body_error = ValueError("P-1L unsuppressible body failure")
    handle = _CloseHandle(truthy_exit=True)
    _install_handle(monkeypatch, handle)

    with pytest.raises(ValueError, match="P-1L unsuppressible body failure") as caught:
        with _adaptive_stable_open(source, {}, lambda: False):
            raise body_error

    assert caught.value is body_error
    assert handle.enter_calls == 0
    assert handle.exit_calls == 0
    assert handle.close_calls == 1


def test_non_exception_body_precedes_failing_close_without_recapture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L06: mandatory cleanup cannot replace non-Exception control flow."""

    source = _source(tmp_path)
    close_error = OSError(_CLOSE_SENTINEL)
    interrupt = KeyboardInterrupt("P-1L keyboard interrupt")
    handle = _CloseHandle(close_error=close_error)
    _install_handle(monkeypatch, handle)
    captures = _capture_after_close(monkeypatch, handle)

    with pytest.raises(KeyboardInterrupt) as caught:
        with _adaptive_stable_open(source, {}, lambda: False):
            raise interrupt

    assert caught.value is interrupt
    assert interrupt.__cause__ is close_error
    assert handle.close_calls == 1
    assert captures == []


def test_post_close_drift_precedes_body_and_close_with_complete_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """L07: post-close drift wins while retaining body then close failures."""

    source = _source(tmp_path)
    close_error = OSError(_CLOSE_SENTINEL)
    body_error = ValueError("P-1L ordinary body failure")

    def mutate() -> None:
        source.write_bytes(source.read_bytes() + b" changed")

    handle = _CloseHandle(on_close=mutate, close_error=close_error)
    _install_handle(monkeypatch, handle)
    captures = _capture_after_close(monkeypatch, handle)

    with pytest.raises(output_preflight.SourceRevisionChanged) as caught:
        with _adaptive_stable_open(source, {}, lambda: False):
            raise body_error

    drift = caught.value
    assert drift.__cause__ is body_error
    assert body_error.__cause__ is close_error
    assert close_error.__cause__ is None
    assert drift is not body_error and body_error is not close_error
    assert handle.close_calls == 1
    assert captures == [source]


class _InnerStop:
    def __init__(self, error: RuntimeError) -> None:
        self.error = error
        self.armed = False
        self.trace: list[bool] = []

    def arm(self) -> None:
        self.armed = True

    def __call__(self) -> bool:
        if not self.armed:
            return False
        if not self.trace:
            self.trace.append(False)
            return False
        self.trace.append(True)
        raise self.error


def _trace_io_after_inner_stop(
    monkeypatch: pytest.MonkeyPatch,
    stopped: _InnerStop,
) -> list[str]:
    later: list[str] = []
    real_capture = SourceFileState.capture

    def traced_capture(path):
        if stopped.trace == [False, True]:
            later.append("SourceFileState.capture")
        return real_capture(path)

    monkeypatch.setattr(
        SourceFileState,
        "capture",
        staticmethod(traced_capture),
    )
    for name in (
        "_resolve_source_alias",
        "_capture_canonical_source_target",
        "_candidate_owner_id",
    ):
        real = getattr(source_graph, name)

        def traced(*args, _name=name, _real=real, **kwargs):
            if stopped.trace == [False, True]:
                later.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(source_graph, name, traced)
    return later


def test_inner_exact_stop_keeps_ordinary_body_and_close_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MC01: inner Stop chains from body while body retains close."""

    source = _source(tmp_path)
    cancellation = RuntimeError("admission cancelled")
    stopped = _InnerStop(cancellation)
    close_error = OSError(_CLOSE_SENTINEL)
    body_error = ValueError("P-1M ordinary body failure")
    handle = _CloseHandle(on_close=stopped.arm, close_error=close_error)
    _install_handle(monkeypatch, handle)
    later = _trace_io_after_inner_stop(monkeypatch, stopped)

    with pytest.raises(RuntimeError, match="^admission cancelled$") as caught:
        with _adaptive_stable_open(source, {}, stopped):
            raise body_error

    assert caught.value is cancellation
    assert cancellation.__cause__ is body_error
    assert body_error.__cause__ is close_error
    assert close_error.__cause__ is None
    assert handle.close_calls == 1
    assert stopped.trace == [False, True]
    assert later == []


def test_inner_exact_stop_without_primary_keeps_existing_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MC02: inner Stop without a primary is a bare identity re-raise."""

    source = _source(tmp_path)
    cancellation = RuntimeError("admission cancelled")
    stopped = _InnerStop(cancellation)
    handle = _CloseHandle(on_close=stopped.arm)
    _install_handle(monkeypatch, handle)
    later = _trace_io_after_inner_stop(monkeypatch, stopped)

    with pytest.raises(RuntimeError, match="^admission cancelled$") as caught:
        with _adaptive_stable_open(source, {}, stopped):
            pass

    assert caught.value is cancellation
    assert cancellation.__cause__ is None
    assert handle.close_calls == 1
    assert stopped.trace == [False, True]
    assert later == []


def test_inner_exact_stop_chains_from_close_only_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MC03: inner Stop chains from the retained close-only primary."""

    source = _source(tmp_path)
    cancellation = RuntimeError("admission cancelled")
    stopped = _InnerStop(cancellation)
    close_error = OSError(_CLOSE_SENTINEL)
    handle = _CloseHandle(on_close=stopped.arm, close_error=close_error)
    _install_handle(monkeypatch, handle)
    later = _trace_io_after_inner_stop(monkeypatch, stopped)

    with pytest.raises(RuntimeError, match="^admission cancelled$") as caught:
        with _adaptive_stable_open(source, {}, stopped):
            pass

    assert caught.value is cancellation
    assert cancellation.__cause__ is close_error
    assert close_error.__cause__ is None
    assert handle.close_calls == 1
    assert stopped.trace == [False, True]
    assert later == []


def test_inner_nonexact_runtime_error_keeps_identity_and_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MC04: a non-exact inner RuntimeError is never reclassified."""

    source = _source(tmp_path)
    scientific_cause = LookupError("P-1M scientific cause")
    sentinel = RuntimeError("M-C sentinel")
    sentinel.__cause__ = scientific_cause
    stopped = _InnerStop(sentinel)
    close_error = OSError(_CLOSE_SENTINEL)
    body_error = ValueError("P-1M ordinary body failure")
    handle = _CloseHandle(on_close=stopped.arm, close_error=close_error)
    _install_handle(monkeypatch, handle)
    later = _trace_io_after_inner_stop(monkeypatch, stopped)

    with pytest.raises(RuntimeError, match="^M-C sentinel$") as caught:
        with _adaptive_stable_open(source, {}, stopped):
            raise body_error

    assert caught.value is sentinel
    assert sentinel.__cause__ is scientific_cause
    assert body_error.__cause__ is close_error
    assert close_error.__cause__ is None
    assert handle.close_calls == 1
    assert stopped.trace == [False, True]
    assert later == []
