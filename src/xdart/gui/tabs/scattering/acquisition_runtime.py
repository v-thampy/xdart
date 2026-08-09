"""Executor-side acquisition context and durable command owner."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock
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
    ) -> DurablePaused:
        deadline = monotonic() + max(0.0, float(timeout))
        self._gate.clear()
        try:
            with self._command_lock:
                drained = bool(session.pause(timeout=max(
                    0.0, deadline - monotonic()
                )))
        except BaseException as primary:
            self._resume_after_pause_failure(session, primary)
        if not drained:
            primary = TimeoutError("acquisition did not reach durable pause")
            self._resume_after_pause_failure(session, primary)
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
                self._resume_after_pause_failure(session, primary)
            if not projected:
                primary = TimeoutError(
                    "display projection did not reach durable pause"
                )
                self._resume_after_pause_failure(session, primary)
        self._durable_generation += 1
        return DurablePaused(run_identity, self._durable_generation)

    def resume(self, session, timeout: float = 5.0) -> None:
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
        self._gate.set()

    def _resume_after_pause_failure(self, session,
                                    primary: BaseException) -> None:
        try:
            with self._command_lock:
                session.resume()
        except BaseException as recovery:
            raise CommandCompensationFailure(
                primary, recovery, "context.pause") from None
        self._gate.set()
        raise primary

    def stop(self, session) -> None:
        self._gate.set()
        with self._command_lock:
            session.stop()

    def request_stop(self, adapter) -> None:
        try:
            adapter.stop()
        finally:
            self._gate.set()

    def terminal_stop(self, session) -> None:
        """Stop one failed session without reopening frame submission."""

        self._gate.clear()
        with self._command_lock:
            session.stop()

    def retire(self) -> None:
        self._gate.set()
        if self.context is not None:
            self.context.retire()


__all__ = [
    "AcquisitionRuntime",
    "CommandCompensationFailure",
    "TerminalPauseFailure",
]
