"""Production-shaped E3 browse loading and A/B pointer transitions."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Event, current_thread
import time

import pytest

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadOutcome,
    BrowseLoadRequest,
    BrowseLoadStatus,
)
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import CleanupStatus
from xdart.gui.tabs.scattering.start_outcomes import StartLaunched
from xdart.gui.tabs.scattering.start_pipeline import StartPipeline
from xdart.modules.display_context import ContextKind, new_context_token
from xrd_tools.reduction import Integration1DPlan, ReductionPlan
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xrd_tools.sources.discover import enumerate_candidates
from xrd_tools.io import ProcessedScan, iter_frame_records

from tests.xdart.scattering._admission import await_admission
from tests.xdart.scattering.test_e2p_rapid_navigation import (
    _DroppingSink,
    _TinyIntegrator,
    _TinySource,
    _accepted_admission,
)
from tests.xdart.scattering.test_e3_context_contract import (
    _current_key,
    _running_controller,
)


_B_SHA256 = (
    "6110bedb3bb14c1c30978f84b43716339ff099856ab55f96e4ca7f2d9c56a3c6"
)


def _browse_fixture() -> Path:
    configured = os.environ.get("XDART_E3_BROWSE_FIXTURE")
    candidates = (
        Path(configured) if configured else None,
        Path(
            "/Users/vthampy/repos/tmp/codex-c3d-testdata-final/"
            "xdart_processed_data/"
            "Combi4_Angledependence_samz_4p9_03271005.nxs"
        ),
    )
    path = next(
        (item for item in candidates if item is not None and item.is_file()),
        None,
    )
    if path is None:
        pytest.skip("the accepted private E3 browse fixture is unavailable")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == _B_SHA256
    return path


def _wait_outcome(
    loader: BrowseLoader,
    request: BrowseLoadRequest,
    *,
    timeout: float = 15.0,
) -> BrowseLoadOutcome:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outcome = loader.poll(request)
        if outcome is not None:
            return outcome
        time.sleep(0.005)
    raise AssertionError("processed browse load did not finish")


def test_real_browse_load_is_off_thread_exact_and_independently_owned(
    monkeypatch,
):
    path = _browse_fixture()
    calls: list[tuple[str, str]] = []
    file_opens: list[tuple[str, str]] = []
    import h5py

    real_file = h5py.File

    def counted_file(source, *args, **kwargs):
        file_opens.append((str(source), current_thread().name))
        return real_file(source, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", counted_file)

    def open_scan(source):
        calls.append(("open_scan", current_thread().name))
        return ProcessedScan(source)

    def read_records(source):
        calls.append(("read_records", current_thread().name))
        yield from iter_frame_records(source)

    loader = BrowseLoader(
        max_items=32,
        open_scan=open_scan,
        read_records=read_records,
    )
    request = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        1,
        str(path),
    )
    assert loader.begin(request) is request
    outcome = _wait_outcome(loader, request)
    assert outcome.request is request
    assert outcome.status is BrowseLoadStatus.READY
    context = loader.consume(outcome)
    assert context is not None
    assert context.operation is request
    assert context.load_request is request
    assert context.scan_key == "Combi4_Angledependence_samz_4p9_03271005"
    assert tuple(context.frame_ids) == tuple(range(1, 17))
    publication = context.publication_store.get(1)
    assert publication is not None
    assert publication.scan_key == context.scan_key
    assert publication.view.intensity_1d is not None
    assert publication.view.intensity_2d is not None
    assert (
        publication.view.raw is not None
        or publication.view.thumbnail is not None
    )
    assert context.calibration_identity
    assert context.mask_identity == "True"
    assert context.result_identity == str(path)
    assert calls == [
        ("open_scan", "scattering-browse"),
        ("read_records", "scattering-browse"),
    ]
    assert 2 <= len(file_opens) <= 4
    assert all(
        source == str(path) and thread == "scattering-browse"
        for source, thread in file_opens
    )
    loader.release_context(context)
    assert context.released is True
    assert len(context.record_store) == 0
    loader.close()


def test_explicit_processed_nexus_browse_does_not_enter_raw_discovery() -> None:
    path = Path(
        "/Users/vthampy/repos/test_data/eiger/xdart_processed_data/"
        "Combi4_Angledependence_samz_4p9_03271005.nexus"
    )
    if not path.is_file():
        pytest.skip(
            "processed .nexus Browse capability unavailable: missing "
            f"{path}"
        )

    discovered = enumerate_candidates(path.parent)
    assert path not in {candidate.path for candidate in discovered}

    loader = BrowseLoader(max_items=32)
    request = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        1,
        str(path),
    )
    assert loader.begin(request) is request
    outcome = _wait_outcome(loader, request)
    assert outcome.status is BrowseLoadStatus.READY
    context = loader.consume(outcome)
    assert context is not None
    assert context.requested_path == str(path)
    assert context.scan_key == path.stem
    assert path not in {
        candidate.path for candidate in enumerate_candidates(path.parent)
    }
    loader.release_context(context)
    loader.close()


def test_explicit_malformed_nexus_browse_fails_truthfully(
    tmp_path: Path,
) -> None:
    path = tmp_path / "malformed.nexus"
    path.write_bytes(b"not a processed NeXus file")
    loader = BrowseLoader()
    request = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        1,
        str(path),
    )
    loader.begin(request)
    outcome = _wait_outcome(loader, request)
    assert outcome.status is BrowseLoadStatus.FAILED
    assert outcome.detail
    assert loader.consume(outcome) is None
    loader.close()


def test_real_browse_resume_keeps_exact_a_and_rejects_late_projection():
    path = _browse_fixture()
    _, lifecycle, executor, _, acquisition = _running_controller(gi=True)
    loader = BrowseLoader(max_items=32)
    controller = ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=loader,
        projection=ContextProjection(),
    )
    controller.adopt_acquisition(executor.identity)
    a_store = acquisition.publication_store
    a_catalog = a_store.catalog_snapshot()
    controller.pause()
    request = controller.begin_browse(str(path))
    deadline = time.monotonic() + 15.0
    outcome = None
    while outcome is None and time.monotonic() < deadline:
        outcome = controller.poll_browse()
        if outcome is None:
            time.sleep(0.005)
    assert outcome is not None
    assert outcome.request is request
    browse = controller.browse_context
    assert browse is not None
    assert browse.record_store is not acquisition.record_store
    assert browse.publication_store is not a_store
    browse_key = _current_key(controller)
    payload = controller.project(browse_key)
    assert payload is not None
    assert payload.frame_key.local_frame_label == 1
    assert payload.measurement_mode == "GI"
    assert payload.gi_resolved_motor == "th"
    delayed = controller.project_request(browse_key)
    selection = controller.resume()
    assert selection.names(acquisition)
    assert controller.acquisition_context is acquisition
    assert acquisition.publication_store is a_store
    assert a_store.catalog_snapshot() == a_catalog
    assert controller.resolve_projection(delayed) is None
    controller.close()


def test_cancelled_load_retires_exact_worker_after_inert_completion(
    tmp_path: Path,
):
    entered = Event()
    release = Event()

    def read_records(_source):
        entered.set()
        release.wait(timeout=5.0)
        return iter(())

    path = tmp_path / "pending.nxs"
    path.write_bytes(b"not opened by the injected reader")
    loader = BrowseLoader(
        open_scan=lambda source: object(),
        read_records=read_records,
    )
    request = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        1,
        str(path),
    )
    loader.begin(request)
    assert entered.wait(timeout=2.0)
    loader.cancel(request)
    loader.cancel(request)
    worker = loader._worker
    assert worker is not None
    release.set()
    worker.join(timeout=2.0)
    cleaned = loader.cancel(request)
    assert cleaned.request is request
    assert cleaned.cleanup_status is CleanupStatus.CLEANED
    assert loader.poll(request) is None
    assert loader._active is None
    assert loader._queued is None
    assert loader._worker is None
    second = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        2,
        str(path),
    )
    assert loader.begin(second) is second
    failed = _wait_outcome(loader, second)
    assert failed.status is BrowseLoadStatus.FAILED
    assert loader.consume(failed) is None
    loader.close()


@pytest.mark.parametrize(
    "gi_enabled",
    (False, True),
    ids=("standard", "gi"),
)
def test_real_executor_pauses_at_durable_store_boundary(
    monkeypatch,
    tmp_path: Path,
    gi_enabled: bool,
):
    # Use the accepted 651-frame production-shaped source so Pause cannot race
    # a five-frame run to completion.  Construction and reduction still travel
    # through StandardRunExecutor, ScanSession, and the real lifecycle.
    qapp = (
        QtWidgets.QApplication.instance()
        or QtWidgets.QApplication([])
    )
    assert qapp is not None
    selected = tmp_path / "tiny_0001.tif"
    import fabio
    import numpy as np

    fabio.tifimage.TifImage(
        data=np.ones((2, 2), dtype=np.uint16)
    ).write(str(selected))
    poni = tmp_path / "tiny.poni"
    poni.write_text("deterministic test calibration")
    output = tmp_path / "tiny.nxs"
    source_facts = []
    monkeypatch.setattr(
        executor_module, "build_admission_receipt", _accepted_admission
    )
    monkeypatch.setattr(
        executor_module,
        "open_source",
        lambda _spec: _TinySource(selected, source_facts),
    )
    monkeypatch.setattr(
        executor_module,
        "poni_to_integrator",
        lambda _poni: _TinyIntegrator(),
    )
    monkeypatch.setattr(
        executor_module,
        "build_native_int_reduction_plan_from_args",
        lambda *_args, **_kwargs: ReductionPlan(
            integration_1d=Integration1DPlan(npt=2),
            integration_2d=None,
        ),
    )
    monkeypatch.setattr(
        executor_module,
        "NexusSink",
        lambda *_args, **_kwargs: _DroppingSink(),
    )
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor(max_display_items=4)
    pipeline = StartPipeline(
        intents=RunIntentStore(
            RunIntent(
                source_spec=image_series_spec(selected),
                poni_file=str(poni),
                project_root=str(tmp_path),
                save_path=str(output),
                output_mode="Overwrite",
                max_cores=1,
                gi=(
                    {
                        "enabled": True,
                        "incidence_motor": "Manual",
                        "th_val": 0.1,
                        "mode_1d": "q_total",
                        "mode_2d": "qip_qoop",
                    }
                    if gi_enabled
                    else {}
                ),
            )
        ),
        lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(),
        executor=executor,
    )
    capture = pipeline.begin()
    admission = await_admission(executor, capture)
    launched = pipeline.start(admission)
    assert isinstance(launched, StartLaunched)
    deadline = time.monotonic() + 5.0
    context = None
    while time.monotonic() < deadline:
        context = executor.acquisition_context(launched.run_identity)
        if (
            context is not None
            and context.record_store.catalog_snapshot().entries
        ):
            break
        time.sleep(0.002)
    assert context is not None
    controller = ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=BrowseLoader(),
        projection=ContextProjection(),
    )
    controller.adopt_acquisition(launched.run_identity)
    first_key = context.record_store.catalog_snapshot().entries[-1]
    first_payload = controller.project(first_key)
    assert first_payload is not None
    assert first_payload.measurement_mode == (
        "GI" if gi_enabled else "Standard"
    )
    if gi_enabled:
        assert first_payload.gi_resolved_motor == "Manual"
    frozen_gi = context.run_configuration.gi
    paused = controller.pause()
    before = len(context.record_store.catalog_snapshot().entries)
    time.sleep(0.03)
    assert len(context.record_store.catalog_snapshot().entries) == before
    assert paused.run_identity is launched.run_identity
    assert lifecycle.phase.value == "paused"
    assert controller.selection.names(context)
    controller.resume()
    deadline = time.monotonic() + 3.0
    while (
        len(context.record_store.catalog_snapshot().entries) <= before
        and time.monotonic() < deadline
    ):
        time.sleep(0.002)
    assert len(context.record_store.catalog_snapshot().entries) > before
    assert context.run_configuration.gi is frozen_gi
    resumed_payload = controller.project(first_key)
    assert resumed_payload is not None
    np.testing.assert_array_equal(
        first_payload.view.axis_1d.values,
        resumed_payload.view.axis_1d.values,
    )
    controller.stop()
    deadline = time.monotonic() + 10.0
    terminal = ()
    while time.monotonic() < deadline:
        terminal += executor.drain_events()
        if any(
            event.kind
            in {
                StandardEventKind.FINISHED,
                StandardEventKind.STOPPED,
                StandardEventKind.FAILED,
            }
            for event in terminal
        ):
            break
        time.sleep(0.005)
    assert any(
        event.kind is StandardEventKind.STOPPED for event in terminal
    )
    executor.close(launched.run_identity)
    controller.close()
