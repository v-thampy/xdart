"""Construction-owned vNext processing-mode and Run-strip projection."""

from __future__ import annotations

import os

from xrd_tools.session.run_configuration import RunIntent

from .output_values import APPEND_UNAVAILABLE
from .shell_values import RunStripProjection, ShellPhase
from .state_machine import RunPhase


RUN_MODE_CHOICES = (
    "Int 1D",
    "Int 2D",
    "Int 1D (XYE)",
    "Stitch 1D",
    "Stitch 2D",
    "Image Viewer",
    "XYE Viewer",
)
UNOWNED_RUN_MODE_REASONS = (
    (
        "Int 1D (XYE)",
        "XYE output has no mounted vNext sink contract yet.",
    ),
    (
        "Stitch 1D",
        "Stitching has no mounted vNext operation service yet.",
    ),
    (
        "Stitch 2D",
        "Stitching has no mounted vNext operation service yet.",
    ),
    (
        "Image Viewer",
        "Image Viewer has no mounted vNext viewer context yet.",
    ),
    (
        "XYE Viewer",
        "XYE Viewer has no mounted vNext viewer context yet.",
    ),
)
_NATIVE_RUN_MODES = frozenset(("Int 1D", "Int 2D"))


def build_run_strip_projection(
    phase: RunPhase,
    intent: RunIntent,
    *,
    executor_available: bool,
    start_permitted: bool,
    start_blocker: str,
    source_count: int | None = None,
    source_count_is_files: bool = False,
    source_count_includes_immediate: bool = False,
) -> RunStripProjection:
    overwrite = (
        type(intent.output_mode) is str
        and intent.output_mode.strip().lower() == "overwrite"
    )
    missing: list[str] = []
    if intent.source_spec is None:
        missing.append("source")
    if not intent.poni_file:
        missing.append("PONI")
    if not intent.save_path:
        missing.append("output")
    mode = str(intent.processing_mode or "")
    mode_blocker = dict(UNOWNED_RUN_MODE_REASONS).get(mode)
    if mode not in _NATIVE_RUN_MODES and mode_blocker is None:
        mode_blocker = (
            f"{mode or 'Selected mode'} has no mounted vNext operation "
            "service yet."
        )
    if mode_blocker is not None:
        readiness = mode_blocker
    elif not executor_available:
        readiness = "Execution is unavailable"
    elif not overwrite:
        readiness = APPEND_UNAVAILABLE
    elif missing:
        readiness = f"Needs {', '.join(missing)}"
    elif not start_permitted:
        readiness = start_blocker or "Cleanup remains pending"
    else:
        readiness = f"Ready · {mode}"
        if source_count is not None:
            noun = (
                "file" if source_count == 1 else "files"
            ) if source_count_is_files else (
                "frame" if source_count == 1 else "frames"
            )
            count = f"{source_count} {noun}"
            if source_count_includes_immediate:
                count += " (folder + 1 level)"
            readiness += f" · {count}"
    ready = (
        executor_available
        and overwrite
        and not missing
        and start_permitted
        and mode_blocker is None
    )
    return RunStripProjection(
        phase=_shell_phase(phase),
        modes=RUN_MODE_CHOICES,
        disabled_modes=UNOWNED_RUN_MODE_REASONS,
        mode=mode,
        batch=intent.batch_mode,
        cores=max(1, intent.max_cores),
        max_cores=max(1, os.cpu_count() or 1),
        live=intent.live_mode,
        output_policy=intent.output_mode,
        readiness=readiness,
        ready=ready,
        run_enabled=(
            phase in {RunPhase.RUNNING, RunPhase.PAUSED}
            or phase in {RunPhase.IDLE, RunPhase.FAILED} and ready
        ),
        stop_enabled=phase in {
            RunPhase.STARTING,
            RunPhase.RUNNING,
            RunPhase.PAUSING,
            RunPhase.PAUSED,
            RunPhase.RESUMING,
            RunPhase.STOPPING,
            RunPhase.FINALIZING,
        },
    )


def _shell_phase(phase: RunPhase) -> ShellPhase:
    return {
        RunPhase.IDLE: ShellPhase.IDLE,
        RunPhase.PREPARING: ShellPhase.PREPARING,
        RunPhase.STARTING: ShellPhase.PREPARING,
        RunPhase.RUNNING: ShellPhase.RUNNING,
        RunPhase.PAUSING: ShellPhase.PAUSING,
        RunPhase.PAUSED: ShellPhase.PAUSED,
        RunPhase.RESUMING: ShellPhase.RUNNING,
        RunPhase.STOPPING: ShellPhase.STOPPING,
        RunPhase.FINALIZING: ShellPhase.STOPPING,
        RunPhase.FAILED: ShellPhase.FAILED,
        RunPhase.CLOSED: ShellPhase.CLOSED,
    }[phase]


__all__ = [
    "RUN_MODE_CHOICES",
    "UNOWNED_RUN_MODE_REASONS",
    "build_run_strip_projection",
]
