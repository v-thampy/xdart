"""Headless ``ScanSession`` demo — drive a scan by *commands in / events out*,
with **no Qt / GUI imports** (greenfield Difference 2).

This is the public data-ownership boundary: a caller (here a plain script; in
production the xdart GUI, or a notebook, or an autonomous loop) feeds frames to
a :class:`xrd_tools.session.ScanSession` and consumes immutable
:class:`~xrd_tools.session.FrameEvent` / ``ProgressEvent`` / ``StateChangeEvent``
callbacks — owning no reduction state itself.

What it exercises (all Qt-free):
  * build a small synthetic ``Scan`` + a real pyFAI integrator;
  * open a ``ScanSession`` over a ``NexusSink`` (one persistent streaming
    writer thread);
  * register ``on_frame_completed`` / ``on_progress`` / ``on_state_change``;
  * ``submit`` every frame, ``pause``/``resume`` mid-run, ``finish``;
  * confirm one single-result event per frame (ADR-0003) and that pause/resume
    did not bump the caller-owned ``generation`` (ADR-0004 §2).

Run it where importing ``xdart`` or Qt would fail — it must still pass::

    python examples/headless_scan_session.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from xrd_tools.reduction import (
    Frame,
    Integration1DPlan,
    Integration2DPlan,
    NexusSink,
    ReductionPlan,
    Scan,
)
from xrd_tools.session import FrameEvent, ProgressEvent, ScanSession, StateChangeEvent

N_FRAMES = 6
SHAPE = (64, 64)


def _build_integrator():
    from pyFAI.detectors import Detector
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    pixel = 100e-6
    det = Detector(pixel1=pixel, pixel2=pixel, max_shape=SHAPE)
    center = (SHAPE[0] / 2) * pixel
    return AzimuthalIntegrator(
        dist=0.1, poni1=center, poni2=center, detector=det, wavelength=1e-10,
    )


def _frames():
    rng = np.random.default_rng(0)
    return [
        Frame(index=i, image=rng.random(SHAPE) + i,
              metadata={"i0": 100.0 + i, "sample": "demo"})
        for i in range(N_FRAMES)
    ]


def main() -> int:
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=200, unit="q_A^-1"),
        integration_2d=Integration2DPlan(npt_rad=200, npt_azim=36, unit="q_A^-1"),
    )
    frames = _frames()
    scan = Scan(name="session_demo", frames=frames, integrator=_build_integrator())

    completed: list[FrameEvent] = []
    progress: list[ProgressEvent] = []
    states: list[StateChangeEvent] = []

    with tempfile.TemporaryDirectory() as tmp:
        nxs = Path(tmp) / "session_demo.nxs"

        with ScanSession(plan, scan, sink=NexusSink(nxs, overwrite=True),
                         executor=2) as session:
            session.set_generation(42)                  # the caller owns this stamp
            session.on_frame_completed(completed.append)
            session.on_progress(progress.append)
            session.on_state_change(states.append)

            # Feed the first half, pause to "browse", resume, feed the rest.
            for fr in frames[:N_FRAMES // 2]:
                session.submit(fr)
            assert session.pause(timeout=10), "writer did not quiesce"
            assert session.is_paused
            session.resume()
            for fr in frames[N_FRAMES // 2:]:
                session.submit(fr)
        # __exit__ -> finish(): writer drained, .nxs finalized.

    # --- assert the contract -------------------------------------------------
    assert len(completed) == N_FRAMES, len(completed)
    assert {e.frame_index for e in completed} == set(range(N_FRAMES))
    assert all(e.result_1d is not None for e in completed)     # single-result
    assert all(e.mode_key is None for e in completed)          # standard scan
    assert all(e.generation == 42 for e in completed), \
        "pause/resume must not bump the generation stamp (ADR-0004 §2)"
    assert progress and progress[-1].completed == N_FRAMES
    assert any(s.is_paused for s in states) and states[-1].is_running is False

    print(f"ScanSession: {len(completed)} single-result events, "
          f"submitted={progress[-1].submitted} completed={progress[-1].completed}, "
          f"paused+resumed with generation pinned at 42 — Qt-free. OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
