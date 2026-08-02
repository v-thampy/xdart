"""Frozen E2-LV-R4.2 composed-Close and detector-absence oracle."""

from __future__ import annotations

from pathlib import Path
import time

import h5py
import numpy as np

from tests.xdart.scattering.test_e2lv_r4_1_boundaries import (
    _SequencedExecutor,
    _active_page,
    _dispose,
)
from xdart.gui.tabs.scattering.display_runtime import (
    DetectorHydrationOutcome,
    RunDisplayState,
)
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    DurableFinal,
    ExecutionEnded,
    ExecutorClosed,
    RunIdentity,
)
from xdart.modules.frame_publication import FramePublication
from xrd_tools.core import Axis, FrameRecord, FrameView
from xrd_tools.io import (
    read_frame_record,
    write_frame_records,
)
from xrd_tools.io.image_source import (
    load_processed_raw_or_thumbnail,
)


class _RaiseFirstExecutor(_SequencedExecutor):
    def __init__(self) -> None:
        super().__init__([])
        self._raise_once = True

    def close(self, identity):
        self.close_calls.append(identity)
        if self._raise_once:
            self._raise_once = False
            raise RuntimeError("historical cleanup failed once")
        return ExecutorClosed(identity, CleanupStatus.CLEANED)


def _finish_run(page, lifecycle, identity: RunIdentity) -> None:
    ended = lifecycle.execution_ended(ExecutionEnded(identity))
    assert ended.phase.value == "finalizing"
    final = lifecycle.durable_final(DurableFinal(identity))
    assert final.phase.value == "idle"
    assert page._context_controller.run_identity is identity


def _wait_for(condition, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.001)
    assert condition()


def _integrated_record(label: int = 1) -> FrameRecord:
    q = np.linspace(0.1, 0.5, 5)
    chi = np.linspace(-10.0, 10.0, 3)
    return FrameRecord.from_view(
        FrameView(
            label=label,
            axis_1d=Axis("q", "1/angstrom", values=q),
            intensity_1d=np.arange(5, dtype=float) + 10,
            sigma_1d=np.arange(5, dtype=float) / 10,
            axis_2d_x=Axis("q", "1/angstrom", values=q),
            axis_2d_y=Axis("chi", "degree", values=chi),
            intensity_2d=np.arange(15, dtype=float).reshape(3, 5) + 100,
            sigma_2d=np.arange(15, dtype=float).reshape(3, 5) / 100,
        )
    )


def _closed_display_state(
    artifact: Path,
    record: FrameRecord,
    *,
    fingerprint: str,
) -> tuple[RunDisplayState, object]:
    identity = RunIdentity(1, fingerprint)
    state = RunDisplayState(identity, max_payload_items=1)
    state.configure(partition_count=1, npt=5, frame_bytes=128)
    owner = state.add_artifact(
        artifact,
        "scan",
        mask=None,
        mask_saturation=False,
        measurement_mode="Standard",
    )
    key = state.append_navigation(
        owner.source_scan, str(owner.artifact), record.label
    ).appended
    view = record.active_view()
    publication = FramePublication(
        view,
        record=record,
        source_identity=f"integrated:{record.label}",
        scan_key=owner.source_scan,
    )
    state.retain_frame(
        owner,
        key,
        record,
        publication,
        source_identity=publication.source_identity,
        frame_mask_qualified=False,
    )
    assert owner.publications.discard(record.label)
    return state, key


def test_historical_close_exception_retains_owner_for_public_retry(
    tmp_path: Path,
) -> None:
    executor = _RaiseFirstExecutor()
    qapp, page, lifecycle, identity = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, identity)
    try:
        pending = page.close_workspace()
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_identity is identity

        terminal = page.close_workspace()
        assert executor.close_calls == [identity, identity]
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
    finally:
        _dispose(page, qapp)


def test_historical_foreign_receipt_is_inert_and_retryable(
    tmp_path: Path,
) -> None:
    executor = _SequencedExecutor(["foreign", "cleaned"])
    qapp, page, lifecycle, identity = _active_page(tmp_path, executor)
    _finish_run(page, lifecycle, identity)
    try:
        pending = page.close_workspace()
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert pending.cleanup_identity is identity

        terminal = page.close_workspace()
        assert executor.close_calls == [identity, identity]
        assert terminal.cleanup_status is CleanupStatus.CLEANED
        assert terminal.cleanup_identity is identity
    finally:
        _dispose(page, qapp)
