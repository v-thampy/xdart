"""Finite P1-C L1 oracles for canonical projection and display residency."""

from __future__ import annotations

import ast
from pathlib import Path
import threading
import time

import numpy as np
from pyqtgraph import QtWidgets

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.test_p1b_comprehensive_live import (
    _batch_intent,
    _write_stack,
)
from tests.xdart.scattering.test_p1b_output_graph import (
    _TERMINAL,
    _drain_until,
    _start,
)
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.display_runtime import RunDisplayState
from xdart.gui.tabs.scattering.display_residency import DisplayResidencyLimits
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus, RunIdentity
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.modules.frame_publication import FramePublication
from xrd_tools.core import Axis, FrameRecord, FrameView, TwoDKind
from xrd_tools.session import FrameRecordStore
from xrd_tools.session.intent_store import RunIntentStore


def _owner(state: RunDisplayState):
    return state.add_artifact(
        Path("/run/p1c-l1.nexus"),
        "p1c-l1",
        mask=None,
        mask_saturation=False,
        measurement_mode="Standard",
    )


def _record(label: int) -> tuple[FrameRecord, FramePublication, str]:
    source = f"/source/p1c_{label:04d}.tif"
    view = FrameView(
        label=label,
        axis_1d=Axis("q", "1/angstrom", np.linspace(0.1, 1.0, 8)),
        intensity_1d=np.arange(8, dtype=float) + label,
        axis_2d_x=Axis("q", "1/angstrom", np.linspace(0.1, 1.0, 4)),
        axis_2d_y=Axis("chi", "degree", np.linspace(-1.0, 1.0, 3)),
        intensity_2d=np.arange(12, dtype=float).reshape(3, 4) + label,
        two_d_kind=TwoDKind.Q_CHI,
        raw=np.full((4, 4), label, dtype=np.uint16),
        thumbnail=np.full((2, 2), label, dtype=float),
        source_path=source,
        source_frame_index=label,
    )
    record = FrameRecord.from_view(view)
    source_identity = f"{source}#{label}"
    return (
        record,
        FramePublication(
            view,
            record=record,
            source_identity=source_identity,
            scan_key="p1c-l1",
        ),
        source_identity,
    )


def _publish(
    state: RunDisplayState,
    owner,
    label: int,
    *,
    hydratable: tuple[tuple[str, str], ...],
    durable: tuple[tuple[str, str], ...],
    dropped: tuple[tuple[str, str], ...] = (),
):
    record, publication, source_identity = _record(label)
    owner.records.upsert(record, source_identity=source_identity)
    owner.records.replace_projection(
        label,
        hydratable=hydratable,
        durable=durable,
        dropped=dropped,
    )
    delta = state.append_navigation(owner.source_scan, str(owner.artifact), label)
    state.retain_frame(
        owner,
        delta.appended,
        record,
        publication,
        source_identity=source_identity,
        frame_mask_qualified=False,
    )
    return record, delta.appended


def test_display_retain_does_not_rewrite_session_projection() -> None:
    state = RunDisplayState(RunIdentity(1, "p1c-projection"), max_payload_items=2)
    state.configure(partition_count=1, npt=8, frame_bytes=32)
    owner = _owner(state)
    one_d = (("1d", "default"),)
    two_d = (("2d", "default"),)

    record = _record(1)[0]
    owner.records.upsert(record, source_identity="/source/p1c_0001.tif#1")
    owner.records.replace_projection(
        1,
        hydratable=one_d,
        durable=one_d,
        dropped=two_d,
    )
    before = (
        owner.records._revisions[1],
        owner.records.hydratable_modes(1),
        owner.records.durable_modes(1),
        owner.records.dropped_modes(1),
    )
    publication = _record(1)[1]
    delta = state.append_navigation(owner.source_scan, str(owner.artifact), 1)
    state.retain_frame(
        owner,
        delta.appended,
        record,
        publication,
        source_identity="/source/p1c_0001.tif#1",
        frame_mask_qualified=False,
    )
    state.mark_durable(owner, (1,))

    assert (
        owner.records._revisions[1],
        owner.records.hydratable_modes(1),
        owner.records.durable_modes(1),
        owner.records.dropped_modes(1),
    ) == before
    assert state._residency._evict_live(delta.appended) is True


def test_nine_durable_frames_cross_real_heavy_cap_and_cleanly_demote(
    monkeypatch,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    state = RunDisplayState(RunIdentity(1, "p1c-cap"), max_payload_items=2)
    state.configure(partition_count=1, npt=8, frame_bytes=32)
    owner = _owner(state)
    modes = (("1d", "default"), ("2d", "default"))

    for label in range(1, 10):
        _publish(
            state,
            owner,
            label,
            hydratable=modes,
            durable=modes,
        )
        state.mark_durable(owner, (label,))

    snapshot = state.residency_snapshot()
    assert snapshot.heavy == snapshot.limits.heavy == 8
    assert owner.records.has_heavy_payload(1) is False
    assert owner.publications.get(1).raw_status == "thumbnail"
    assert owner.light_records.get(1) is not None


def test_owed_live_retirement_is_unchanged_then_durable_retry_is_total() -> None:
    state = RunDisplayState(RunIdentity(1, "p1c-retire"), max_payload_items=2)
    state.configure(partition_count=1, npt=8, frame_bytes=32)
    state._residency.limits = DisplayResidencyLimits(8, 8, 1, 1)
    owner = _owner(state)
    modes = (("1d", "default"), ("2d", "default"))
    record, key = _publish(
        state,
        owner,
        1,
        hydratable=modes,
        durable=modes,
    )
    before = (
        owner.records.get(1),
        owner.light_records.get(1),
        owner.publications.get(1),
        state._residency.capture(key),
    )

    assert state._residency._evict_live(key) is False
    assert (
        owner.records.get(1),
        owner.light_records.get(1),
        owner.publications.get(1),
        state._residency.capture(key),
    ) == before

    state.mark_durable(owner, (1,))
    assert state._residency._evict_live(key) is True
    assert owner.records.get(1) is None
    assert owner.light_records.get(1) is None
    assert owner.publications.get(1) is None
    assert key not in state._residency._stores
    assert key not in state._residency._heavy
    assert key not in state._residency._thumbnails
    assert key not in state._residency._browse
    assert key not in state._residency._live
    assert record.label == 1


def test_removed_store_calls_and_canonical_projection_writes_stay_absent() -> None:
    root = Path(__file__).resolve().parents[3]
    offenders = []
    for relative in (
        "src/xdart/gui/tabs/scattering/display_runtime.py",
        "src/xdart/gui/tabs/scattering/display_residency.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            ):
                continue
            receiver = ast.unparse(node.func.value)
            method = node.func.attr
            canonical = receiver.endswith(".records") and not receiver.endswith(
                ".light_records"
            )
            if (
                canonical
                and method in {
                    "upsert",
                    "mark_persisted",
                    "replace_projection",
                    "evict_heavy",
                    "discard",
                }
            ) or (
                receiver.endswith(".light_records") and method == "discard"
            ):
                offenders.append((relative, receiver, method, node.lineno))

    assert offenders == []
    assert not hasattr(FrameRecordStore, "discard")
    assert not hasattr(FrameRecordStore, "evict_heavy")


def test_real_nine_frame_container_finishes_and_executor_closes(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    source = tmp_path / "nine.nxs"
    target = tmp_path / "nine-processed.nexus"
    poni = tmp_path / "cal.poni"
    _write_stack(source, 9)
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        _batch_intent(source, target, poni),
        request_value=9101,
    )
    events = _drain_until(
        executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
        timeout=90.0,
    )
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FINISHED
    assert terminal.completed == terminal.total == 9
    run = executor._exact_run(identity)
    assert run is not None
    assert run.completed == 9
    snapshot = run.display.residency_snapshot()
    assert snapshot.heavy == snapshot.limits.heavy == 8
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED


def test_real_host_exit_action_closes_after_nine_frame_run(
    monkeypatch,
    tmp_path,
) -> None:
    from xdart import _gui_main
    from xdart.gui.pages.catalog import SCATTERING_WORKSPACE_PAGE
    from xdart.gui.pages.services import (
        DiagnosticIdentity,
        ExecutionProfile,
        HostServices,
    )
    from xdart.gui.pages.values import PageCleanup
    from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter

    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    monkeypatch.setenv("XDART_SETTINGS_FILE", str(tmp_path / "settings.ini"))
    monkeypatch.setenv("XDART_SESSION_FILE", str(tmp_path / "session.json"))
    source = tmp_path / "host-nine.nxs"
    target = tmp_path / "host-nine-processed.nexus"
    poni = tmp_path / "host-cal.poni"
    _write_stack(source, 9)
    write_poni(poni)
    intent_store = RunIntentStore(_batch_intent(source, target, poni))
    executor = StandardRunExecutor(join_timeout=2.0)
    source_port = FilesystemSourceAdapter()

    class _Status:
        def show(self, _text, timeout_ms=0):
            del timeout_ms

    class _Intents:
        def store_for(self, _key):
            return intent_store

    class _Execution:
        def executor_for(self, _key):
            return executor

    class _Sources:
        def source_port_for(self, _key):
            return source_port

    services = HostServices(
        status=_Status(),
        run_intents=_Intents(),
        execution=_Execution(),
        sources=_Sources(),
        execution_profile=ExecutionProfile.TEST,
        diagnostics=DiagnosticIdentity("tests.p1c-l1-host-close"),
    )
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = _gui_main.Main(
        page_descriptors=(SCATTERING_WORKSPACE_PAGE,),
        host_services=services,
        selected_page_key=SCATTERING_WORKSPACE_PAGE.key,
    )
    page = window.main_widget
    terminated = []
    monkeypatch.setattr(
        window,
        "_terminate_process",
        lambda: terminated.append(True),
    )
    window.show()

    def wait_for(predicate, timeout=90.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            qapp.processEvents()
            if predicate():
                return True
            time.sleep(0.01)
        return False

    try:
        window.page_handle.run_control.run_pause()
        assert wait_for(lambda: page._lifecycle.phase is RunPhase.RUNNING)
        assert wait_for(lambda: page._lifecycle.phase is RunPhase.IDLE)
        window.ui.actionExit.trigger()
        assert wait_for(lambda: bool(terminated) and not window.isVisible())
        assert window.page_handle.close().status is PageCleanup.CLEAN
        assert wait_for(
            lambda: not any(
                thread.is_alive() and thread.name.startswith("scattering-")
                for thread in threading.enumerate()
            ),
            timeout=5.0,
        )
    finally:
        window._process_exit_requested = False
        window.close()
        window.deleteLater()
        qapp.processEvents()
