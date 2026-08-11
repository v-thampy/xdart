"""Executor-side acquisition context and durable command owner."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from threading import Condition, Event, Lock
from time import monotonic

from xdart.modules.display_context import (
    AcquisitionContext, ContextKind, new_context_token,
)

from .events import DurablePaused, RunIdentity, detach_exception


_NO_IMAGE = object()


class CommandCompensationFailure(RuntimeError, KeyboardInterrupt):
    """A command and its deterministic recovery operation both failed."""

    def __init__(self, primary: BaseException, recovery: BaseException,
                 operation: str) -> None:
        self.diagnostics = (detach_exception(primary, operation),
                            detach_exception(
                                recovery, f"{operation}.compensation"))
        super().__init__(self.diagnostics[0].message)


class TerminalPauseFailure(RuntimeError):
    """A Pause boundary that cannot truthfully recover to Running."""

    def __init__(self, primary: BaseException) -> None:
        self.primary = primary
        self.diagnostic = detach_exception(primary, "context.pause")
        self.cleanup_receipt = None
        super().__init__(self.diagnostic.message)


def _pause_compensation(session, timeout: float) -> bool:
    try:
        pause = object.__getattribute__(session, "pause")
    except AttributeError:
        paused = object.__getattribute__(session, "paused")
        if type(paused) is not bool:
            raise
        object.__setattr__(session, "paused", True)
        return True
    return bool(pause(timeout=timeout))


class AcquisitionRuntime:
    """Compose existing executor owners without copying or wrapping stores."""

    def __init__(self) -> None:
        self.context: AcquisitionContext | None = None
        self._gate = Event()
        self._gate.set()
        self._command_lock = Lock()
        self._durable_generation = 0
        self._live_condition = Condition()
        self._live_armed = self._live_terminal = self._live_retired = False
        self._live_admitting = True
        self._live_effects = self._live_pause_request = 0
        self._live_paused_session = None

    def _arm_live(self) -> None:
        self._live_armed = True

    @contextmanager
    def _live_effect(self):
        if not self._live_armed:
            yield
            return
        with self._live_condition:
            self._live_condition.wait_for(
                lambda: self._live_admitting or self._live_terminal)
            if self._live_terminal:
                raise RuntimeError("admission cancelled")
            self._live_effects += 1
        try:
            yield
        finally:
            with self._live_condition:
                self._live_effects -= 1
                retire = self.context if self._live_retired else None
                self._live_condition.notify_all()
            if retire is not None:
                retire.retire()

    def _fence_live(self) -> None:
        with self._live_condition:
            self._live_terminal, self._live_admitting = True, False
            self._live_pause_request += 1
            self._gate.clear()
            self._live_condition.notify_all()

    def _quiesce_live(self, deadline: float) -> int:
        with self._live_condition:
            if self._live_terminal:
                raise RuntimeError("acquisition is stopped")
            self._gate.clear()
            self._live_admitting = False
            self._live_pause_request += 1
            request = self._live_pause_request
            settled = self._live_condition.wait_for(
                lambda: not self._live_effects or self._live_terminal,
                max(0.0, deadline - monotonic()))
            if not settled:
                self._require_live_request(request, reopen=True)
                raise TimeoutError("Live effects did not reach durable pause")
            if self._live_terminal or request != self._live_pause_request:
                raise RuntimeError("acquisition is stopped")
            return request

    def _require_live_request(self, request: int, session=_NO_IMAGE,
                              *, reopen=False) -> None:
        with self._live_condition:
            if (self._live_terminal or request != self._live_pause_request
                    or session is not _NO_IMAGE
                    and session is not self._live_paused_session):
                raise RuntimeError("acquisition is stopped")
            if reopen:
                self._live_paused_session, self._live_admitting = None, True
                self._gate.set()
                self._live_condition.notify_all()

    def adopt(self, run, artifact, source_path: str) -> AcquisitionContext:
        context = self.context
        if context is None:
            configuration = run.configuration
            generation, fingerprint = configuration.identity
            context = AcquisitionContext(
                context_token=new_context_token(ContextKind.ACQUISITION),
                run_configuration=configuration,
                config_generation=generation,
                config_fingerprint=fingerprint,
                run_scan_key=artifact.source_scan,
                source_path=source_path,
                scan=run.scan,
                frame=None,
                frame_ids=run.display.catalog,
                frames=run.display.artifacts,
                viewer_rows_1d=(),
                viewer_rows_2d=(),
                publication_store=run.display,
                origin="scattering-standard",
                poni_identity=configuration.poni_file,
                mask_identity=configuration.mask_file,
            )
            context.adopt_record_store(run.display)
            self.context = context
        elif (context.scan_key != artifact.source_scan
              or context.source != source_path
              or context.current_display_scan is not run.scan):
            context.rescope_to(artifact.source_scan, source_path, run.scan)
        return context

    def submit(self, session, frame, image=_NO_IMAGE) -> bool:
        while True:
            self._gate.wait()
            with self._command_lock:
                if not self._gate.is_set():
                    continue
                if self._live_terminal:
                    return False
                if image is _NO_IMAGE:
                    return bool(session.submit(frame))
                return bool(session.submit(frame, image))

    def pause(
        self,
        session,
        run_identity: RunIdentity,
        timeout: float,
        *,
        drain_projection: Callable[[float], bool] | None = None,
        session_supplier: Callable[[], object | None] | None = None,
    ) -> DurablePaused:
        deadline = monotonic() + max(0.0, float(timeout))
        live = self._live_armed
        request = self._quiesce_live(deadline) if live else 0
        self._gate.clear()
        if session is None and not live:
            self._gate.set()
            raise RuntimeError("acquisition is not pausable")
        try:
            if live and session_supplier is not None:
                session = session_supplier()
            if live:
                self._require_live_request(request)
            with self._command_lock:
                drained = session is None or bool(session.pause(
                    timeout=max(0.0, deadline - monotonic())))
        except BaseException as primary:
            self._resume_after_pause_failure(session, primary, request)
        if not drained:
            primary = TimeoutError("acquisition did not reach durable pause")
            self._resume_after_pause_failure(session, primary, request)
        if live:
            self._require_live_request(request)
        if drain_projection is not None:
            try:
                projected = bool(drain_projection(max(
                    0.0, deadline - monotonic()
                )))
            except TerminalPauseFailure:
                # Projection mutation is not proven atomic.  Its exact owner
                # must terminate the run instead of compensating to Running.
                raise
            except BaseException as primary:
                self._resume_after_pause_failure(session, primary, request)
            if not projected:
                primary = TimeoutError(
                    "display projection did not reach durable pause"
                )
                self._resume_after_pause_failure(session, primary, request)
        if live:
            with self._live_condition:
                if self._live_terminal or request != self._live_pause_request:
                    raise RuntimeError("acquisition is stopped")
                self._live_paused_session = session
                self._durable_generation += 1
                return DurablePaused(run_identity, self._durable_generation)
        self._durable_generation += 1
        return DurablePaused(run_identity, self._durable_generation)

    def resume(self, session, timeout: float = 5.0) -> None:
        live = self._live_armed
        if live:
            with self._live_condition:
                if self._live_terminal:
                    raise RuntimeError("acquisition is stopped")
                request = self._live_pause_request
                session = self._live_paused_session
        if live and session is None:
            self._require_live_request(request, reopen=True)
            return
        if session is None:
            raise RuntimeError("acquisition is not resumable")
        try:
            with self._command_lock:
                session.resume()
        except BaseException as primary:
            try:
                with self._command_lock:
                    if not _pause_compensation(session, timeout):
                        raise TimeoutError(
                            "resume compensation did not reach durable pause"
                        )
            except BaseException as recovery:
                raise CommandCompensationFailure(
                    primary, recovery, "context.resume") from None
            raise
        if live:
            self._require_live_request(request, session, reopen=True)
        else:
            self._gate.set()

    def _resume_after_pause_failure(self, session,
                                    primary: BaseException, request: int = 0) -> None:
        if request:
            self._require_live_request(request)
        try:
            if session is not None:
                with self._command_lock:
                    session.resume()
        except BaseException as recovery:
            raise CommandCompensationFailure(
                primary, recovery, "context.pause") from None
        if request:
            self._require_live_request(request, reopen=True)
        else:
            self._gate.set()
        raise primary

    def stop(self, session) -> None:
        self._fence_live()
        try:
            with self._command_lock:
                session.stop()
        finally:
            self._gate.set()

    def request_stop(self, adapter) -> None:
        self._fence_live()
        try:
            if adapter is not None:
                adapter.stop()
        finally:
            self._gate.set()

    def terminal_stop(self, session) -> None:
        """Stop one failed session without reopening frame submission."""
        self.stop(session)

    def retire(self) -> None:
        self._fence_live()
        self._gate.set()
        with self._live_condition:
            self._live_retired = True
            context = self.context
        if context is not None:
            context.retire()


__all__ = [
    "AcquisitionRuntime",
    "CommandCompensationFailure",
    "TerminalPauseFailure",
]
