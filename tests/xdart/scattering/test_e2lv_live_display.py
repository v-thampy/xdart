"""Production-shaped E2-LV live-display continuity oracle."""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Event, current_thread
from types import SimpleNamespace
import time

import fabio.tifimage
import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import (
    AcceptedScientificAssets,
    AdmittedOutput,
    AdmissionReceipt,
    OutputFact,
)
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import (
    StandardEventKind,
    StandardRunEvent,
)
from xdart.gui.tabs.scattering.output_preflight import (
    OutputCandidate,
    OutputDisposition,
    _series_item,
)
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xrd_tools.core.scan import Scan, ScanFrame
from xrd_tools.core.staging import (
    browse_publication_max_items,
    heavy_window,
    live_record_store_max_items,
)
from xrd_tools.io import (
    load_processed_raw_or_thumbnail as real_load_processed_raw,
)
from xrd_tools.io import read_frame_record as real_read_frame_record
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec


def _wait(
    qapp: QtWidgets.QApplication,
    predicate,
    *,
    timeout: float = 30.0,
    diagnostic=lambda: "",
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError(
        f"E2-LV production-shaped operation timed out: {diagnostic()}"
    )


def _mounted(
    page: ScatteringWorkspace,
) -> tuple[ScatteringWorkspaceShell, ContextController]:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    controller = page._context_controller
    assert type(controller) is ContextController
    return shell, controller


def _select_exact_frame(
    shell: ScatteringWorkspaceShell,
    frame,
) -> None:
    selector = shell.scientific.frame_selector
    index = next(
        index
        for index in range(selector.count())
        if selector.itemData(index) is frame
    )
    selector.setCurrentIndex(index)


def _shell_diagnostic(
    shell: ScatteringWorkspaceShell,
    lifecycle: ScatteringCoordinator,
) -> str:
    readiness = shell.run_controls.readinessLabel
    full_text = getattr(readiness, "full_text", None)
    summary = full_text() if callable(full_text) else readiness.text()
    return (
        f"phase={lifecycle.phase!r}, readiness={summary!r}, "
        f"status={shell.scientific.status.text()!r}"
    )


class _Integrator:
    detector = SimpleNamespace(mask=None)

    def integrate1d(self, image, npt, *, unit, **_kwargs):
        value = float(np.nanmean(np.asarray(image)))
        return SimpleNamespace(
            radial=np.linspace(0.1, 1.0, npt),
            intensity=np.linspace(value, value + 1.0, npt),
            sigma=None,
            unit=unit,
        )

    def integrate2d(
        self,
        image,
        npt_rad,
        npt_azim,
        *,
        unit,
        azimuth_range=None,
        **_kwargs,
    ):
        value = float(np.nanmean(np.asarray(image)))
        lower, upper = azimuth_range or (-180.0, 180.0)
        return SimpleNamespace(
            radial=np.linspace(0.1, 1.0, npt_rad),
            azimuthal=np.linspace(lower, upper, npt_azim),
            intensity=np.arange(
                npt_rad * npt_azim, dtype=float
            ).reshape(npt_azim, npt_rad)
            + value,
            sigma=None,
            unit=unit,
            azimuthal_unit="chi_deg",
        )


class _Source:
    def __init__(
        self,
        members: tuple[Path, ...],
        labels: tuple[int, ...],
    ) -> None:
        self._members = members
        self._selected = members[0]
        self._labels = labels

    def to_scan(self, *, poni, integrator, output_path):
        return Scan(
            self._selected.stem,
            [
                ScanFrame(
                    label,
                    image=np.arange(8, dtype=np.uint32).reshape(2, 4)
                    + label,
                    source_path=member,
                    source_frame_index=0,
                )
                for member, label in zip(
                    self._members, self._labels, strict=True
                )
            ],
            poni=poni,
            integrator=integrator,
            output_path=output_path,
        )

    def close(self) -> None:
        return None


def _write_tiff(path: Path, *, offset: int = 0) -> None:
    fabio.tifimage.TifImage(
        data=np.arange(8, dtype=np.uint16).reshape(2, 4) + offset
    ).write(str(path))


def _accepted_admission(
    labels: tuple[int, ...],
    *,
    mask: np.ndarray | None = None,
):
    def build(
        capture,
        *,
        cancelled,
        session_owner,
    ) -> AdmissionReceipt:
        del session_owner
        assert not cancelled()
        configuration = capture.intent_snapshot.thaw()
        poni_bytes = Path(configuration.poni_file).read_bytes()
        mask_bytes = None if mask is None else mask.tobytes()
        assets = AcceptedScientificAssets(
            (0.1, 0.01, 0.01, 0.0, 0.0, 0.0, 1e-10, "Detector"),
            None if mask is None else str(mask.dtype),
            None if mask is None else mask.shape,
            mask_bytes,
            hashlib.sha256(poni_bytes).hexdigest(),
            (
                None
                if mask_bytes is None
                else hashlib.sha256(mask_bytes).hexdigest()
            ),
            "{\"orientation\":3}",
        )
        candidate = OutputCandidate.from_start_capture(
            capture,
            assets,
            (),
        )
        item = _series_item(
            candidate,
            candidate.source,
            cancelled=cancelled,
        )
        assert labels == tuple(
            range(
                item.source_stamp.first_label,
                item.source_stamp.first_label
                + item.source_stamp.frame_count,
            )
        )
        return AdmissionReceipt(
            capture.request_id,
            capture.intent_snapshot.revision,
            capture.source_capture,
            candidate,
            (
                AdmittedOutput(
                    item,
                    OutputDisposition.WRITE,
                    labels,
                    OutputFact(False),
                ),
            ),
            assets,
            (),
        )

    return build


def _standard_page(
    monkeypatch,
    tmp_path: Path,
    *,
    labels: tuple[int, ...],
    mask: np.ndarray | None = None,
) -> tuple[
    QtWidgets.QApplication,
    ScatteringWorkspace,
    ScatteringCoordinator,
    StandardRunExecutor,
    Path,
]:
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    members = tuple(
        tmp_path / f"tiny_{label:04d}.tif" for label in labels
    )
    for member, label in zip(members, labels, strict=True):
        _write_tiff(member, offset=label)
    selected = members[0]
    poni = tmp_path / "tiny.poni"
    poni.write_text("accepted through immutable test assets")
    output = tmp_path / "tiny.nxs"
    monkeypatch.setattr(
        executor_module,
        "build_admission_receipt",
        _accepted_admission(labels, mask=mask),
    )
    monkeypatch.setattr(
        executor_module,
        "open_source",
        lambda _spec: _Source(members, labels),
    )
    monkeypatch.setattr(
        executor_module,
        "poni_to_integrator",
        lambda _poni: _Integrator(),
    )
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor(max_display_items=2)
    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(selected),
                poni_file=str(poni),
                project_root=str(tmp_path),
                save_path=str(output),
                output_mode="Overwrite",
                max_cores=1,
                bai_1d_args={"npt": 8},
                bai_2d_args={"npt_rad": 8, "npt_azim": 6},
            )
        ),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    return qapp, page, lifecycle, executor, output


def test_all_frames_remain_selectable_and_evicted_frame_hydrates_off_gui(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    labels = tuple(range(1, 11))
    qapp, page, lifecycle, executor, output = _standard_page(
        monkeypatch, tmp_path, labels=labels
    )
    from xdart.gui.tabs.scattering import hydration_transport

    gui_thread = current_thread()
    reads: list[tuple[str, object]] = []
    real_preview_read = hydration_transport.read_frame_preview

    def traced_preview(read_key, **kwargs):
        reads.append(("preview", current_thread()))
        return real_preview_read(read_key, **kwargs)

    monkeypatch.setattr(
        hydration_transport, "read_frame_preview", traced_preview
    )
    shell, controller = _mounted(page)
    try:
        _wait(
            qapp,
            lambda: shell.run_controls.startButton.isEnabled(),
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        shell.run_controls.startButton.click()
        _wait(
            qapp,
            lambda: lifecycle.phase is RunPhase.IDLE,
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        assert output.is_file()
        keys = controller.frame_keys
        assert tuple(key.local_frame_label for key in keys) == labels
        selector = shell.scientific.frame_selector
        assert selector.count() == len(labels)
        assert all(
            selector.itemData(index) is key
            for index, key in enumerate(keys)
        )

        first = keys[0]
        _select_exact_frame(shell, first)
        _wait(
            qapp,
            lambda: (
                selector.currentData() is first
                and shell.scientific.title.text() == "tiny_0001.tif"
            ),
        )

        assert reads
        assert all(thread is not gui_thread for _kind, thread in reads)
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
        assert shell.scientific.curve.listDataItems()
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_latest_qualified_selection_wins_over_slow_evicted_reload(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    qapp, page, lifecycle, executor, _output = _standard_page(
        monkeypatch, tmp_path, labels=tuple(range(1, 11))
    )
    reads: list[int] = []
    frame_one_entered = Event()
    release_frame_one = Event()

    from xdart.gui.tabs.scattering import hydration_transport

    real_preview_read = hydration_transport.read_frame_preview

    def delayed_record(read_key, **kwargs):
        label = int(read_key.frame_identity)
        reads.append(label)
        if label == 1:
            frame_one_entered.set()
            if not release_frame_one.wait(timeout=10.0):
                raise TimeoutError("frame-1 hydration was not released")
        return real_preview_read(read_key, **kwargs)

    monkeypatch.setattr(
        hydration_transport, "read_frame_preview", delayed_record
    )
    shell, controller = _mounted(page)
    try:
        _wait(qapp, lambda: shell.run_controls.startButton.isEnabled())
        shell.run_controls.startButton.click()
        _wait(qapp, lambda: lifecycle.phase is RunPhase.IDLE)
        first, second = controller.frame_keys[:2]
        _select_exact_frame(shell, first)
        _wait(qapp, frame_one_entered.is_set)
        _select_exact_frame(shell, second)
        release_frame_one.set()
        _wait(
            qapp,
            lambda: (
                shell.scientific.frame_selector.currentData() is second
                and shell.scientific.title.text() == "tiny_0002.tif"
            ),
            timeout=10.0,
            diagnostic=lambda: (
                f"reads={reads!r}; title={shell.scientific.title.text()!r}; "
                f"timer={page._run_timer.isActive()}; "
                f"counters={executor._active.display.transport.counters()!r}; "
                f"worker={getattr(executor._active.display, 'hydration_thread', None)!r}; "
                f"payloads={[(key.local_frame_label, value.selection_generation) for key, value in executor._active.display.payloads.items()]!r}"
            ),
        )
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.002)
        assert set(reads[:2]) == {1, 2}
        assert shell.scientific.title.text() == "tiny_0002.tif"
        assert shell.scientific.frame_selector.currentData() is second
    finally:
        release_frame_one.set()
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def _painted_values(shell):
    return (
        shell.scientific.title.text(),
        np.array(shell.scientific.raw.image.image, copy=True),
        np.array(shell.scientific.cake.image.image, copy=True),
        tuple(
            (
                np.array(item.xData, copy=True),
                np.array(item.yData, copy=True),
            )
            for item in shell.scientific.curve.listDataItems()
        ),
    )


def _assert_same_paint(shell, expected) -> None:
    title, raw, cake, traces = expected
    assert shell.scientific.title.text() == title
    np.testing.assert_array_equal(shell.scientific.raw.image.image, raw)
    np.testing.assert_array_equal(shell.scientific.cake.image.image, cake)
    rendered = shell.scientific.curve.listDataItems()
    assert len(rendered) == len(traces)
    for item, (x, y) in zip(rendered, traces, strict=True):
        np.testing.assert_array_equal(item.xData, x)
        np.testing.assert_array_equal(item.yData, y)


def test_duplicate_same_owner_context_ready_keeps_pending_display_and_request(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Queued short-artifact hints neither blank nor supersede hydration."""

    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    qapp, page, lifecycle, executor, _output = _standard_page(
        monkeypatch,
        tmp_path,
        labels=tuple(range(1, 11)),
    )
    shell, controller = _mounted(page)
    entered = Event()
    release = Event()

    from xdart.gui.tabs.scattering import hydration_transport

    real_read = hydration_transport.read_frame_preview

    def blocked_first(read_key, **kwargs):
        if int(read_key.frame_identity) == 1:
            entered.set()
            if not release.wait(timeout=10.0):
                raise TimeoutError("frame-1 hydration was not released")
        return real_read(read_key, **kwargs)

    monkeypatch.setattr(
        hydration_transport,
        "read_frame_preview",
        blocked_first,
    )

    try:
        _wait(qapp, lambda: shell.run_controls.startButton.isEnabled())
        shell.run_controls.startButton.click()
        _wait(qapp, lambda: lifecycle.phase is RunPhase.IDLE)

        first = controller.frame_keys[0]
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
        assert shell.scientific.curve.listDataItems()
        previous = _painted_values(shell)
        selection = controller.selection
        assert selection is not None

        _select_exact_frame(shell, first)
        _wait(qapp, entered.is_set)
        assert shell.scientific.frame_selector.currentData() is first
        _assert_same_paint(shell, previous)

        applied = []
        apply_state = shell.apply_state

        def record_apply(state, *, preserve_display=False):
            apply_state(state, preserve_display=preserve_display)
            applied.append(state)

        monkeypatch.setattr(shell, "apply_state", record_apply)
        identity = controller.run_identity
        assert identity is not None
        hint = StandardRunEvent(
            identity,
            StandardEventKind.CONTEXT_READY,
        )
        executor._events.put(hint)
        executor._events.put(hint)

        page._drain_executor()

        assert controller.selection is selection
        assert controller.selection.display_generation == (
            selection.display_generation
        )
        assert len(applied) == 1
        assert applied[0].navigation.current is first
        assert applied[0].scientific.heavy is None
        assert applied[0].scientific.retain_display is True
        _assert_same_paint(shell, previous)

        release.set()
        _wait(
            qapp,
            lambda: (
                len(applied) >= 2
                and applied[-1].navigation.current is first
                and applied[-1].scientific.heavy is not None
                and applied[-1].scientific.heavy.frame is first
            ),
        )
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
        assert shell.scientific.curve.listDataItems()
        assert not np.array_equal(
            shell.scientific.raw.image.image,
            previous[1],
        )
    finally:
        release.set()
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_initial_context_ready_retains_outgoing_display_until_first_frame(
    monkeypatch,
    tmp_path: Path,
) -> None:
    entered = Event()
    release = Event()
    block_next = Event()
    integrate_1d = _Integrator.integrate1d

    def gated_integrate_1d(self, *args, **kwargs):
        if block_next.is_set():
            entered.set()
            if not release.wait(timeout=10.0):
                raise TimeoutError("first-frame integration was not released")
        return integrate_1d(self, *args, **kwargs)

    monkeypatch.setattr(_Integrator, "integrate1d", gated_integrate_1d)
    qapp, page, lifecycle, executor, output = _standard_page(
        monkeypatch,
        tmp_path,
        labels=(1,),
    )
    shell, controller = _mounted(page)
    try:
        _wait(qapp, lambda: shell.run_controls.startButton.isEnabled())
        shell.run_controls.startButton.click()
        _wait(qapp, lambda: lifecycle.phase is RunPhase.IDLE)
        previous = _painted_values(shell)
        output.unlink()

        applied: list[tuple[object, bool]] = []
        apply_state = shell.apply_state

        def record_apply(state, *, preserve_display=False):
            apply_state(state, preserve_display=preserve_display)
            applied.append((state, preserve_display))

        monkeypatch.setattr(shell, "apply_state", record_apply)
        block_next.set()
        shell.run_controls.startButton.click()
        _wait(qapp, entered.is_set)
        _wait(
            qapp,
            lambda: (
                controller.run_identity is lifecycle.active_run_identity
                and not page._run_frame_seen
                and any(preserve for _state, preserve in applied)
            ),
        )

        _assert_same_paint(shell, previous)
        assert applied[-1][1] is True

        release.set()
        _wait(
            qapp,
            lambda: (
                lifecycle.phase is RunPhase.IDLE
                and any(
                    not preserve and state.scientific.heavy is not None
                    for state, preserve in applied
                )
            ),
        )
        assert shell.scientific.raw.image.image is not None
        assert shell.scientific.cake.image.image is not None
        assert shell.scientific.curve.listDataItems()
    finally:
        release.set()
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_accepted_detector_mask_is_baked_before_live_publication(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mask = np.zeros((2, 4), dtype=np.uint8)
    mask[0, 1] = 1
    qapp, page, lifecycle, _executor, _output = _standard_page(
        monkeypatch, tmp_path, labels=(1,), mask=mask
    )
    shell, _controller = _mounted(page)
    try:
        _wait(qapp, lambda: shell.run_controls.startButton.isEnabled())
        shell.run_controls.startButton.click()
        _wait(qapp, lambda: lifecycle.phase is RunPhase.IDLE)
        rendered = shell.scientific.raw.image.image
        assert rendered is not None
        assert rendered.shape == (4, 2)
        assert np.isnan(rendered[1, 1])
        assert np.isfinite(rendered[0, 1])
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_product_retention_uses_ram_aware_tiers_not_delivery_limits(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    labels = tuple(range(1, 11))
    qapp, page, lifecycle, executor, _output = _standard_page(
        monkeypatch, tmp_path, labels=labels
    )
    shell, controller = _mounted(page)
    try:
        _wait(
            qapp,
            lambda: shell.run_controls.startButton.isEnabled(),
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        shell.run_controls.startButton.click()
        _wait(
            qapp,
            lambda: lifecycle.phase is RunPhase.IDLE,
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        identity = controller.run_identity
        assert identity is not None
        run = executor._exact_run(identity)
        assert run is not None
        owners = tuple(run.display.artifacts.values())
        assert len(owners) == 1
        owner = owners[0]
        residency = run.display.residency_snapshot()
        assert residency.limits.live == live_record_store_max_items(8)
        assert residency.limits.heavy == heavy_window()
        assert residency.limits.browse == browse_publication_max_items(8)
        assert residency.limits.thumbnails == 512
        assert owner.records._max_items is None
        assert owner.records._max_heavy_items is None
        assert owner.light_records._max_items is None
        assert owner.light_records._max_heavy_items is None
        assert owner.publications._max_heavy_items is None
        assert owner.publications._max_thumbnail_items is None
        assert len(owner.light_records) == len(labels)
        assert residency.heavy <= residency.limits.heavy
        assert residency.thumbnails <= residency.limits.thumbnails
        assert residency.browse <= residency.limits.browse
        assert residency.live <= residency.limits.live
        assert len(run.display.payloads) == 2
        assert len(run.display.catalog_snapshot().entries) == len(labels)
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_historical_selection_moves_within_current_scan_footer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    qapp, page, lifecycle, _executor, _output = _standard_page(
        monkeypatch, tmp_path, labels=(1, 2, 3, 4, 5)
    )
    shell, controller = _mounted(page)
    try:
        _wait(
            qapp,
            lambda: shell.run_controls.startButton.isEnabled(),
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        shell.run_controls.startButton.click()
        _wait(
            qapp,
            lambda: lifecycle.phase is RunPhase.IDLE,
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        assert shell.scientific.progress.text() == "5/5"

        fourth = next(
            frame
            for frame in controller.frame_keys
            if frame.local_frame_label == 4
        )
        _select_exact_frame(shell, fourth)
        _wait(
            qapp,
            lambda: (
                shell.scientific.frame_selector.currentData() is fourth
                and shell.scientific.title.text() == "tiny_0004.tif"
            ),
        )

        assert shell.scientific.progress.text() == "4/5"
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()


def test_qualified_catalog_distinguishes_multi_output_repeated_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    qapp, page, lifecycle, executor, _output = _standard_page(
        monkeypatch, tmp_path, labels=(1, 2, 3)
    )
    shell, controller = _mounted(page)
    try:
        _wait(
            qapp,
            lambda: shell.run_controls.startButton.isEnabled(),
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        shell.run_controls.startButton.click()
        _wait(
            qapp,
            lambda: lifecycle.phase is RunPhase.IDLE,
            diagnostic=lambda: _shell_diagnostic(shell, lifecycle),
        )
        identity = controller.run_identity
        assert identity is not None
        catalog = executor.frame_catalog(identity)
        assert catalog is not None
        keys = controller.frame_keys
        assert len(keys) == len(catalog.entries)
        assert all(
            actual is expected
            for actual, expected in zip(
                keys, catalog.entries, strict=True
            )
        )
        assert tuple(key.local_frame_label for key in catalog.entries) == (
            1,
            2,
            3,
        )
        assert tuple(key.work_ordinal for key in catalog.entries) == (1, 2, 3)
        assert len({(key.source_scan, key.artifact) for key in catalog.entries}) == 1
        assert all(key.run_identity is identity for key in catalog.entries)
    finally:
        page.close_workspace()
        page.deleteLater()
        qapp.processEvents()
