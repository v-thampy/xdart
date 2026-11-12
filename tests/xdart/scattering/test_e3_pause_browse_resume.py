"""Production-shaped E3 browse loading and A/B pointer transitions."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Event, current_thread
import time

import pytest

from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters import browse_loader as browse_module
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
from xrd_tools.core import SourceKind, TwoDKind
from xrd_tools.core.containers import IntegrationResult1D
import xrd_tools.reduction.core as reduction_core
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import (
    FrozenRunConfiguration,
    RunIntent,
)
from xrd_tools.sources.selection import image_series_spec
from xrd_tools.sources.discover import discover_scans, enumerate_candidates
from xrd_tools.io import (
    FrameScalarCatalog,
    FrameScalarRow,
    FrameViewReader,
    ProcessedScan,
)

from tests.xdart.scattering._admission import await_admission
from tests.xdart.scattering.test_e2p_rapid_navigation import (
    _TinyIntegrator,
    _write_synthetic_series,
)
from tests.xdart.scattering.test_e3_context_contract import (
    _current_key,
    _running_controller,
)


#: The 16-frame grazing-incidence Combi4 scan in the real test data (XDART_TEST_DATA), reduced once per module by the
#: real executor. Until 1.2.10 the fixture defaulted to a personal scratch file that no longer exists, so both rows
#: that use it skipped everywhere (review M6).
_B_RAW = Path("Tiff/Combi4_Angledependence_samz_4p9_03271005_0001.tif")
_B_PONI = Path("Tiff/LaB6_detz190_dety72_th5_03261554_0001.poni")
_EXPLICIT_NEXUS_RELATIVE = Path(
    "eiger/xdart_processed_data/"
    "Combi4_Angledependence_samz_4p9_03271005.nexus"
)
_EXPLICIT_NEXUS_SIZE = 10_449_060
_EXPLICIT_NEXUS_SHA256 = (
    "f41fcbdc0f341ded301987662a508427ed14981ee05134a71ee9a86b6a59a4b9"
)


@pytest.fixture(scope="module")
def _browse_fixture(tmp_path_factory) -> Path:
    """The Combi4 scan reduced with GI (incidence motor th) into a fresh Save Path, through the same StartPipeline and
    StandardRunExecutor a Run uses; returns the output the admission named."""
    data_root = os.environ.get("XDART_TEST_DATA")
    if not data_root:
        pytest.skip(f"XDART_TEST_DATA is unset; expected {_B_RAW}")
    raw, poni = Path(data_root) / _B_RAW, Path(data_root) / _B_PONI
    if not (raw.is_file() and poni.is_file()):
        pytest.skip(f"{_B_RAW} or {_B_PONI} is missing from XDART_TEST_DATA")
    folder = tmp_path_factory.mktemp("e3-browse")
    intent = RunIntent(
        source_spec=image_series_spec(raw), processing_mode="Int 2D", poni_file=str(poni),
        project_root=str(raw.parent.parent), save_path=str(folder), output_mode="Overwrite", max_cores=4,
        bai_1d_args={"npt": 500, "method": "csr"}, bai_2d_args={"npt_rad": 200, "npt_azim": 200, "method": "csr"},
    )
    intent.gi.enabled = True
    intent.gi.incidence_motor = "th"
    executor = StandardRunExecutor(join_timeout=5.0)
    pipeline = StartPipeline(intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
                             sources=FilesystemSourceAdapter(), executor=executor)
    admission = await_admission(executor, pipeline.begin())
    launched = pipeline.start(admission)
    assert isinstance(launched, StartLaunched)
    target = Path(admission.outputs[0].item.target)
    deadline, events = time.monotonic() + 180.0, []
    try:
        while not any(event.kind is StandardEventKind.FINISHED for event in events):
            assert time.monotonic() < deadline, [event.kind for event in events][-5:]
            assert not any(event.kind is StandardEventKind.FAILED for event in events), events[-1]
            events.extend(executor.drain_events())
            time.sleep(0.02)
    finally:
        executor.close(launched.run_identity)
    assert target.is_file()
    return target


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


def test_default_browse_retains_all_651_scalar_rows_without_eager_frames(
    tmp_path, monkeypatch,
) -> None:
    """Browse owns all scalar rows without eagerly hydrating frame arrays."""

    from tests.xdart.scattering.test_e4_preview_transport import (
        _write_processed,
    )
    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    labels = tuple(range(1, 652))
    processed, _raw = _write_processed(tmp_path, labels=labels)
    loader = BrowseLoader()
    context = None
    try:
        request = BrowseLoadRequest(
            new_context_token(ContextKind.BROWSE),
            1,
            str(processed.resolve()),
        )
        loader.begin(request)
        outcome = _wait_outcome(loader, request)
        assert outcome.status is BrowseLoadStatus.READY
        context = loader.consume(outcome)
        assert context is not None
        catalog = context.scalar_catalog
        assert type(catalog) is FrameScalarCatalog
        assert catalog.labels == labels
        assert context.frame_ids is catalog.labels
        assert context.loaded_labels is catalog.labels
        assert context.publication_store._max_items >= len(labels)
        assert context.publication_store.labels() == ()
        assert context.publication_store._heavy_labels == []
        assert context.record_store.labels() == ()
        assert context.browse_1d_cache.resident_keys == ()

        # The open read no frame facts; they are read on demand, frame by frame.
        assert context.frame_facts.record_count == 0
        shown = (labels[0], labels[len(labels) // 2], labels[-1])
        with FrameViewReader(catalog.artifact_path, resolve_source=False) as reader:
            facts = reader.read_frame_facts(tuple(sorted(set(shown))))
        by_label = {item.label: item for item in facts}
        for label in shown:
            row = catalog.row(label)
            assert type(row) is FrameScalarRow
            assert row.active_mode_1d in row.modes_1d
            assert row.active_mode_2d in row.modes_2d
            item = by_label[label]
            assert item.source_path is not None
            assert Path(item.source_path).name == _raw.name
            assert item.source_frame_index == 0
            assert item.has_thumbnail is True
            assert item.mask_baked is True
    finally:
        if context is not None:
            loader.release_context(context)
        loader.close()


def test_browse_heavy_hydration_preserves_named_active_modes() -> None:
    """Light Browse hydration must not invent a default mode for GI data."""

    from dataclasses import replace

    from tests.xdart.scattering.test_e3_context_contract import _browse
    from xdart.gui.tabs.scattering.browse_hydration import (
        _BrowseHydrationOwner,
    )
    from xdart.gui.tabs.scattering.browse_values import (
        canonical_browse_source_identity,
    )
    from xdart.gui.tabs.scattering.display_runtime import (
        browse_publication_needs_hydration,
    )
    from xdart.gui.tabs.scattering.hydration_transport import (
        PreparedHydrationCommit,
    )
    from xdart.modules.display_context import HydrationRequest
    from xdart.modules.frame_publication import FramePublication
    from xrd_tools.core import FrameRecord
    from xrd_tools.io.frame_preview import FramePreview
    from xrd_tools.session.hydration import (
        HydrationOutcome,
        HydrationPurpose,
        HydrationReadKey,
        HydrationScope,
        HydrationToken,
    )

    request, browse = _browse(
        new_context_token(ContextKind.BROWSE),
        1,
        scan_key="named-gi",
        scalar_row=FrameScalarRow(
            1,
            modes_1d=("q_ip", "q_oop"),
            modes_2d=("q_ip_q_oop", "q_chi"),
            active_mode_1d="q_ip",
            active_mode_2d="q_ip_q_oop",
            two_d_kinds=(
                ("q_ip_q_oop", TwoDKind.QIP_QOOP),
                ("q_chi", TwoDKind.Q_CHI),
            ),
        ),
        axes_1d=(
            ("q_ip", "q_ip", "A^-1", False),
            ("q_oop", "q_oop", "A^-1", False),
        ),
    )
    store = browse.publication_store
    initial = store.get(1)
    assert initial is not None
    full_view = replace(initial.view, source_frame_index=0)
    light_view = replace(
        full_view,
        intensity_2d=None,
        raw=None,
        thumbnail=None,
    )
    named = FrameRecord(
        label=1,
        results_1d={
            "q_ip": light_view,
            "q_oop": light_view,
        },
        results_2d={
            "q_ip_q_oop": light_view,
            "q_chi": light_view,
        },
        active_mode_1d="q_ip",
        active_mode_2d="q_ip_q_oop",
    )
    assert store.discard(1) is True
    store.upsert(FramePublication(
        light_view,
        record=named,
        source_identity=canonical_browse_source_identity(
            light_view, browse.requested_path,
        ),
        scan_key=browse.scan_key,
    ))

    owner = _BrowseHydrationOwner(browse)
    hydration_owner = browse.hydration_owner
    read_key = HydrationReadKey(
        HydrationScope(*hydration_owner.as_tuple()),
        browse.requested_path,
        1,
        HydrationPurpose.PREVIEW,
    )
    token = HydrationToken(read_key, 1)
    hydration = HydrationRequest(
        1,
        HydrationPurpose.PREVIEW,
        1,
        hydration_owner,
        (store,),
        browse.commit_gate,
        read_key=read_key,
        token=token,
    )
    view = replace(full_view, raw=None)
    preview = FramePreview(
        read_key,
        view,
        view.thumbnail,
        None,
        view.source_path,
        None,
        view.source_frame_index,
        None,
        "q_ip",
        "q_ip_q_oop",
    )
    prepared = PreparedHydrationCommit(
        hydration,
        token,
        None,
        False,
        preview,
        None,
    )
    try:
        assert owner.commit_preview(prepared) is HydrationOutcome.HYDRATED
        committed = store.get(1)
        assert committed is not None
        assert committed.record.active_mode_1d == "q_ip"
        assert committed.record.active_mode_2d == "q_ip_q_oop"
        assert set(committed.record.results_1d) == {"q_ip", "q_oop"}
        assert set(committed.record.results_2d) == {
            "q_ip_q_oop", "q_chi",
        }
        assert committed.record.results_2d["q_chi"].has_2d is False
        assert browse_publication_needs_hydration(committed, None) is False
    finally:
        owner.retire()


class _KnownBrowseOpenCount(AssertionError):
    """E3-BROWSE-ROWS-STALE-1210: Combi4 opens five times, against the old bound of four."""


class _KnownBrowseMotorFallback(AssertionError):
    """E3-BROWSE-ROWS-STALE-1210: Browse shows Manual for the generated GI output."""


@pytest.mark.xfail(strict=True, raises=_KnownBrowseOpenCount, reason="E3-BROWSE-ROWS-STALE-1210")
def test_real_browse_load_is_off_thread_exact_and_independently_owned(
    monkeypatch, _browse_fixture,
):
    path = _browse_fixture
    calls: list[tuple[str, str]] = []
    file_opens: list[tuple[str, str]] = []
    import h5py

    real_file = h5py.File

    def counted_file(source, *args, **kwargs):
        file_opens.append((str(source), current_thread().name))
        return real_file(source, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", counted_file)

    def open_scan(source, **kwargs):
        calls.append(("open_scan", current_thread().name))
        return ProcessedScan(source, **kwargs)

    class RecordingReader:
        def __init__(self, source, **kwargs):
            self._reader = FrameViewReader(source, **kwargs)

        def __enter__(self):
            if self._reader.__enter__() is not self._reader:
                raise RuntimeError("FrameViewReader changed identity")
            return self

        def read_scalar_catalog(self, *, cancelled):
            calls.append(("read_scalar_catalog", current_thread().name))
            return self._reader.read_scalar_catalog(cancelled=cancelled)

        def __exit__(self, exc_type, exc, tb):
            return self._reader.__exit__(exc_type, exc, tb)

    loader = BrowseLoader(
        max_items=32,
        open_scan=open_scan,
        open_reader=RecordingReader,
    )
    request = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        1,
        str(path),
    )
    context = None
    try:
        assert loader.begin(request) is request
        outcome = _wait_outcome(loader, request)
        assert outcome.request is request
        assert outcome.status is BrowseLoadStatus.READY
        context = loader.consume(outcome)
        assert context is not None
        assert context.operation is request
        assert context.load_request is request
        assert context.scan_key == path.stem
        assert tuple(context.frame_ids) == tuple(range(1, 17))
        assert context.scalar_catalog is not None
        assert context.scalar_catalog.labels == tuple(range(1, 17))
        assert len(context.publication_store) == 0
        assert len(context.record_store) == 0
        assert context.calibration_identity
        assert context.requested_path == str(path)
        assert not hasattr(context, "mask_identity")
        assert not hasattr(context, "result_identity")
        assert calls == [
            ("open_scan", "scattering-browse"),
            ("read_scalar_catalog", "scattering-browse"),
        ]
        assert all(source == str(path) and thread == "scattering-browse" for source, thread in file_opens)
    finally:
        released = None if context is None else loader.release_context(context)
        closed = loader.close()
    assert released is not None and released.cleanup_status is CleanupStatus.CLEANED
    assert closed.cleanup_status is CleanupStatus.CLEANED
    assert context.released is True
    assert len(context.record_store) == 0
    if len(file_opens) == 5:
        raise _KnownBrowseOpenCount(file_opens)
    assert 2 <= len(file_opens) <= 4


def test_explicit_processed_nexus_browse_reload_does_not_enter_raw_discovery(
) -> None:
    data_root = os.environ.get("XDART_TEST_DATA")
    if not data_root:
        pytest.skip(
            "processed .nexus Browse capability unavailable: "
            "XDART_TEST_DATA is unset; expected relative path "
            f"{_EXPLICIT_NEXUS_RELATIVE} with sha256 "
            f"{_EXPLICIT_NEXUS_SHA256}"
        )
    path = Path(data_root) / _EXPLICIT_NEXUS_RELATIVE
    if not path.is_file():
        pytest.skip(
            "processed .nexus Browse capability unavailable: missing "
            f"{path}; expected size {_EXPLICIT_NEXUS_SIZE} and sha256 "
            f"{_EXPLICIT_NEXUS_SHA256}"
        )
    assert path.stat().st_size == _EXPLICIT_NEXUS_SIZE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        _EXPLICIT_NEXUS_SHA256
    )

    discovered = enumerate_candidates(path.parent)
    candidate = next(item for item in discovered if item.path == path)
    assert candidate.adapter_id == "nexus_hdf5"
    assert path not in {
        Path(item.uri)
        for item in discover_scans(path.parent, SourceKind.NEXUS_STACK)
    }
    assert path in {
        Path(item.uri)
        for item in discover_scans(path.parent, SourceKind.PROCESSED_NEXUS)
    }

    loader = BrowseLoader(max_items=32)
    contexts = []
    for generation in (1, 2):
        request = BrowseLoadRequest(
            new_context_token(ContextKind.BROWSE),
            generation,
            str(path),
        )
        assert loader.begin(request) is request
        outcome = _wait_outcome(loader, request)
        assert outcome.status is BrowseLoadStatus.READY
        context = loader.consume(outcome)
        assert context is not None
        contexts.append(context)
        assert context.load_generation == generation
        assert context.requested_path == str(path)
        assert context.scan_key == path.stem
        assert path in {
            Path(item.uri)
            for item in discover_scans(path.parent, SourceKind.PROCESSED_NEXUS)
        }
        loader.release_context(context)
        assert context.released is True
    assert contexts[0] is not contexts[1]
    loader.close()


def test_explicit_malformed_nexus_browse_is_refused_by_admission(
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
    assert outcome.request is request
    assert outcome.status is BrowseLoadStatus.REFUSED
    assert outcome.detail == "No processed-scan format owns this path."
    assert loader.consume(outcome) is None
    loader.close()


@pytest.mark.xfail(strict=True, raises=_KnownBrowseMotorFallback, reason="E3-BROWSE-ROWS-STALE-1210")
def test_real_browse_resume_keeps_exact_a_and_rejects_late_projection(_browse_fixture):
    from xrd_tools.io.browse_presentation import read_ring_record

    path = _browse_fixture
    saved = read_ring_record(path).run_configuration
    assert saved is not None and saved["gi"]["resolved_motor"] == "th"
    _, lifecycle, executor, _, acquisition = _running_controller(gi=True)
    loader = BrowseLoader(max_items=32)
    controller = ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=loader,
        projection=ContextProjection(),
    )
    try:
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
        deadline = time.monotonic() + 15.0
        while payload is None and time.monotonic() < deadline:
            controller.poll_browse_preview()
            payload = controller.project(browse_key)
            if payload is None:
                time.sleep(0.005)
        assert payload is not None
        assert payload.frame_key.local_frame_label == 1
        assert payload.measurement_mode == "GI"
        delayed = controller.project_request(browse_key)
        selection = controller.resume()
        assert selection.names(acquisition)
        assert controller.acquisition_context is acquisition
        assert acquisition.publication_store is a_store
        assert a_store.catalog_snapshot() == a_catalog
        assert controller.resolve_projection(delayed) is None
    finally:
        closed = controller.close()
    assert closed.cleanup_status is CleanupStatus.CLEANED
    if payload.gi_resolved_motor == "Manual":
        raise _KnownBrowseMotorFallback(payload.gi_resolved_motor)
    assert payload.gi_resolved_motor == "th"


def test_cancelled_load_retires_exact_worker_after_inert_completion(
    tmp_path: Path,
    monkeypatch,
):
    entered = Event()
    release = Event()

    class BlockingReader:
        def __init__(self, source, *, resolve_source):
            assert resolve_source is False
            self._path = str(Path(source).resolve())

        def __enter__(self):
            return self

        def read_scalar_catalog(self, *, cancelled):
            entered.set()
            release.wait(timeout=5.0)
            if cancelled():
                raise InterruptedError("Browse scalar catalog read cancelled")
            return FrameScalarCatalog(self._path, "entry", ())

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    path = tmp_path / "pending.nexus"
    path.write_bytes(b"not opened by the injected reader")
    monkeypatch.setattr(
        browse_module,
        "canonical_browse_scan_key",
        lambda source: Path(source).stem,
    )
    loader = BrowseLoader(
        open_scan=lambda source: object(),
        open_reader=BlockingReader,
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


@pytest.mark.parametrize("at_pi", (True, False), ids=("chi_disc_at_pi", "chi_disc_at_zero"))
def test_tiny_integrator_copies_the_real_chi_discontinuity(tmp_path: Path, at_pi: bool):
    # Run construction's ring geometry reads the integrator's chi convention;
    # the shared tiny adapter must carry the calibrated value, never assume pi.
    from tests.xdart.scattering._e2sd_support import write_poni
    from xrd_tools.integrate.calibration import (
        detector_calibration_to_integrator,
        load_detector_calibration,
    )

    poni = tmp_path / "tiny.poni"
    write_poni(poni)
    calibrated = detector_calibration_to_integrator(load_detector_calibration(poni))
    if at_pi:
        calibrated.setChiDiscAtPi()
    else:
        calibrated.setChiDiscAtZero()
    assert _TinyIntegrator(calibrated).chiDiscAtPi is calibrated.chiDiscAtPi is at_pi


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
    import numpy as np

    members = _write_synthetic_series(selected)
    poni = tmp_path / "tiny.poni"
    from tests.xdart.scattering._e2sd_support import write_poni
    write_poni(poni)
    output = tmp_path / "tiny.nexus"
    source_facts = []
    plan_configurations = []
    native_plan = executor_module.native_int_reduction_plan
    from xrd_tools.session import readiness as readiness_module
    native_binding = readiness_module.bind_native_int_source_energy

    def bind_tiny_integrator(scan, calibration, **options):
        effective = native_binding(scan, calibration, **options)
        if not gi_enabled:
            scan.integrator = _TinyIntegrator(scan.integrator)
        return effective

    monkeypatch.setattr(
        readiness_module, "bind_native_int_source_energy", bind_tiny_integrator,
    )

    def tiny_plan(configuration, *, energy_selection=None, **options):
        assert type(configuration) is FrozenRunConfiguration
        plan_configurations.append(configuration)
        # An ordinary Run asks the translation to declare its companion maps.
        assert options == {"companion_modes_2d": True}
        plan = native_plan(configuration, energy_selection=energy_selection, **options)
        assert plan.integration_1d is not None
        assert plan.integration_1d.npt == 2
        assert plan.integration_2d is None
        return plan

    monkeypatch.setattr(
        reduction_core,
        "poni_to_fiber_integrator",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        reduction_core,
        "integrate_gi_polar_1d",
        lambda image, _integrator, **kwargs: IntegrationResult1D(
            radial=np.linspace(0.0, 1.0, kwargs["npt"]),
            intensity=np.full(kwargs["npt"], float(np.asarray(image).mean())),
            sigma=None,
            unit=kwargs["unit"],
        )
    )

    monkeypatch.setattr(
        executor_module,
        "native_int_reduction_plan",
        tiny_plan,
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
                processing_mode="Int 1D",
                output_mode="Overwrite",
                max_cores=1,
                bai_1d_args={"npt": 2},
                bai_2d_args={},
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
    observed = ()
    while time.monotonic() < deadline:
        observed += executor.drain_events()
        context = executor.acquisition_context(launched.run_identity)
        if (
            context is not None
            and context.record_store.catalog_snapshot().entries
        ):
            break
        time.sleep(0.002)
    assert context is not None, tuple(
        (event.kind, event.detail, event.primary, event.cleanup_failures)
        for event in observed
    )
    assert len(plan_configurations) == 1
    assert plan_configurations[0] is context.run_configuration
    controller = ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=BrowseLoader(),
        projection=ContextProjection(),
    )
    controller.adopt_acquisition(launched.run_identity)
    entries = context.record_store.catalog_snapshot().entries
    assert entries, tuple(
        (event.kind, event.detail, event.primary, event.cleanup_failures)
        for event in observed + executor.drain_events()
    )
    first_key = entries[-1]
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
    terminal = observed
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
