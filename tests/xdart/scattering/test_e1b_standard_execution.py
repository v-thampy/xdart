"""Frozen real-data E1b Standard executor and durability oracle."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
from threading import Event, Thread, current_thread
import time
from types import SimpleNamespace

import numpy as np
import pytest

from xrd_tools.core.provenance import read_provenance
from xrd_tools.io.frame_view import read_frame_record
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent, ThresholdIntent
from xrd_tools.sources.selection import image_series_spec
from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor, _StandardRun
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.context_controller import ContextController
from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import ExecutorStartFailed, RequestId, RunIdentity
from xdart.gui.tabs.scattering.contracts import SourceCapture
from xdart.gui.tabs.scattering.start_outcomes import StartCapture, StartLaunched
from xdart.gui.tabs.scattering.start_pipeline import StartPipeline
from tests.xdart.scattering._admission import admission_for, await_admission
from tests.xdart.scattering.test_e4_preview_transport import _wait_transport_idle


def _fixture_root() -> Path:
    root = os.environ.get("XDART_TEST_DATA")
    if not root:
        pytest.skip("XDART_TEST_DATA is required for the E1b Standard fixture")
    tiff = Path(root) / "Tiff"
    selected = tiff / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
    poni = tiff / "LaB6_detz190_dety72_th5_03261554_0001.poni"
    if not selected.is_file() or not poni.is_file():
        pytest.skip("the frozen E1b Standard fixture is unavailable")
    return tiff


def _wait_for_terminal(executor: StandardRunExecutor, timeout: float = 180.0):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(executor.drain_events())
        if any(event.kind in {StandardEventKind.FINISHED, StandardEventKind.FAILED,
                              StandardEventKind.STOPPED} for event in events):
            return tuple(events)
        time.sleep(0.02)
    raise AssertionError("Standard executor did not reach a terminal event")


def test_real_standard_run_is_headless_durable_and_projectable(tmp_path: Path) -> None:
    tiff = _fixture_root()
    selected = tiff / "Combi4_Angledependence_samz_4p9_03271002_0001.tif"
    output = tmp_path / "standard.nxs"
    intent = RunIntent(
        source_spec=image_series_spec(selected),
        poni_file=str(tiff / "LaB6_detz190_dety72_th5_03261554_0001.poni"),
        project_root=str(tiff),
        save_path=str(output),
        output_mode="Overwrite",
        max_cores=1,
        bai_1d_args={"npt": 128, "method": "csr"},
        bai_2d_args={"npt_rad": 128, "npt_azim": 64, "method": "csr"},
    )
    lifecycle = ScatteringCoordinator()
    executor = StandardRunExecutor(max_display_items=2)
    pipeline = StartPipeline(
        intents=RunIntentStore(intent), lifecycle=lifecycle,
        sources=FilesystemSourceAdapter(), executor=executor,
    )

    capture = pipeline.begin()
    assert isinstance(capture, StartCapture)
    admission = await_admission(executor, capture)
    output = admission.outputs[0].item.target
    launched = pipeline.start(admission)

    assert isinstance(launched, StartLaunched)
    try:
        _assert_headless_run_is_durable_and_projectable(
            executor, lifecycle, launched, admission, output,
        )
    finally:
        receipt = executor.close(launched.run_identity)
    assert receipt.cleanup_status.value == "cleaned"


def _assert_headless_run_is_durable_and_projectable(
    executor, lifecycle, launched, admission, output,
) -> None:
    assert launched.configuration.generation == launched.run_identity.generation
    events = _wait_for_terminal(executor)
    assert events[-1].kind is StandardEventKind.FINISHED
    frame_events = [event for event in events if event.kind is StandardEventKind.FRAME_READY]
    assert all(event.frame_key is not None for event in frame_events)
    assert [
        event.frame_key.local_frame_label
        for event in frame_events
        if event.frame_key is not None
    ] == [1, 2, 3, 4, 5]
    assert output.is_file()
    scan_name = frame_events[-1].frame_key.source_scan
    xye_directory = output.parent / scan_name
    assert xye_directory.is_dir()
    assert tuple(path.name for path in sorted(xye_directory.iterdir())) == tuple(
        f"iq_{scan_name}_{frame:04d}.xye"
        for frame in range(1, 6)
    )

    controller = ContextController(
        lifecycle=lifecycle,
        executor=executor,
        browse_loader=object(),
        projection=ContextProjection(),
    )
    selection = controller.adopt_acquisition(launched.run_identity)
    assert controller.run_identity is launched.run_identity
    assert controller.acquisition_context is executor.acquisition_context(
        launched.run_identity
    )
    assert controller.frame_keys
    final_key = controller.frame_keys[-1]
    assert final_key.local_frame_label == 5
    assert controller.select_navigation(final_key, (final_key,))
    payload = controller.project(final_key)
    assert payload is not None
    assert payload.frame_key.run_identity is launched.run_identity
    assert payload.frame_key is final_key
    assert payload.selection_generation == selection.display_generation
    # The executor publishes light payloads; Full Raw is an explicit
    # on-demand read through the acquisition's own preview transport.
    assert controller.request_full_current() is not None
    assert _wait_transport_idle(controller.acquisition_context.publication_store)
    assert controller.full_raw_status() == (True, False, None)
    payload = controller.project(final_key)
    assert payload is not None
    assert payload.frame_key is final_key
    assert payload.selection_generation == selection.display_generation
    assert payload.view.raw is not None
    assert payload.view.intensity_1d is not None
    assert payload.view.intensity_2d is not None
    assert payload.view.raw.flags.writeable is False
    first_key = controller.frame_keys[0]
    assert controller.select_navigation(first_key, (first_key,))
    first_payload = controller.project(first_key)
    assert first_payload is not None
    assert first_payload.frame_key is first_key
    assert first_payload.frame_key.local_frame_label == 1
    foreign = RunIdentity(launched.run_identity.generation, launched.run_identity.fingerprint)
    foreign_key = replace(first_key, run_identity=foreign)
    with pytest.raises(RuntimeError, match="selection"):
        controller.project(foreign_key)

    reloaded = read_frame_record(output, 5).active_view()
    np.testing.assert_allclose(reloaded.axis_1d.values, payload.view.axis_1d.values)
    np.testing.assert_allclose(reloaded.intensity_1d, payload.view.intensity_1d)
    np.testing.assert_allclose(reloaded.axis_2d_x.values, payload.view.axis_2d_x.values)
    np.testing.assert_allclose(reloaded.axis_2d_y.values, payload.view.axis_2d_y.values)
    np.testing.assert_allclose(reloaded.intensity_2d, payload.view.intensity_2d)
    assert reloaded.source_path == str(payload.view.source_path)
    provenance = read_provenance(str(output))["config"]["run_configuration"]
    signature = provenance["scientific_signature"]
    assert {key: value for key, value in provenance.items()
            if key != "scientific_signature"} == launched.configuration.as_provenance()
    assert signature == admission.candidate.processing_mapping()


def test_executor_refuses_incomplete_configuration_before_constructing_output(tmp_path: Path) -> None:
    source = image_series_spec(tmp_path / "raw_0001.tif")
    configuration = RunIntent(source_spec=source, save_path=str(tmp_path / "out.nxs")).freeze()
    identity = RunIdentity.from_configuration(configuration)

    result = StandardRunExecutor().start(
        configuration, SourceCapture(RequestId(1), 1, source), identity, object(),
    )

    assert isinstance(result, ExecutorStartFailed)
    assert result.cleanup_status.value == "cleaned"
    assert not (tmp_path / "out.nxs").exists()


@pytest.mark.parametrize("trailing_separator", [False, True])
def test_real_16_frame_run_publishes_with_a_project_folder_separator(
    tmp_path: Path, trailing_separator: bool,
) -> None:
    import h5py

    tiff = _fixture_root()
    selected = tiff / "Combi4_Angledependence_samz_4p9_03271005_0001.tif"
    if not selected.is_file():
        pytest.skip("the 16-frame Combi4 fixture is unavailable")
    root = str(tiff.parent) + (os.sep if trailing_separator else "")
    intent = RunIntent(
        source_spec=image_series_spec(selected),
        processing_mode="Int 2D",
        poni_file=str(tiff / "LaB6_detz190_dety72_th5_03261554_0001.poni"),
        project_root=root,
        save_path=str(tmp_path / "combi.nexus"),
        output_mode="Overwrite", max_cores=4,
        threshold=ThresholdIntent(apply_threshold=True, threshold_min=0,
                                  threshold_max=65534),
        bai_1d_args={"npt": 1000, "method": "csr", "unit": "q_A^-1",
                     "radial_range": (0, 5), "azimuth_range": (-180, 180)},
        bai_2d_args={"npt_rad": 500, "npt_azim": 500, "method": "csr",
                     "unit": "q_A^-1", "radial_range": (0, 5),
                     "azimuth_range": (-180, 180)},
    )
    executor = StandardRunExecutor(max_display_items=8)
    pipeline = StartPipeline(
        intents=RunIntentStore(intent), lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(), executor=executor,
    )
    started = time.perf_counter()
    capture = pipeline.begin()
    admission = await_admission(executor, capture)
    admitted = time.perf_counter()
    launched = pipeline.start(admission)
    assert isinstance(launched, StartLaunched)
    try:
        events = _wait_for_terminal(executor)
    finally:
        receipt = executor.close(launched.run_identity)
    closed = time.perf_counter()
    print(f"COMBI16 admission_s={admitted - started:.3f} "
          f"run_and_close_s={closed - admitted:.3f} "
          f"total_s={closed - started:.3f} output={admission.outputs[0].item.target}")
    assert receipt.cleanup_status.value == "cleaned"
    terminal = events[-1]
    assert terminal.kind is StandardEventKind.FINISHED, (terminal.detail, terminal.primary)
    assert terminal.completed == 16
    assert terminal.cleanup_status.value == "cleaned"
    ready = [event for event in events if event.kind is StandardEventKind.FRAME_READY]
    assert [event.frame_key.local_frame_label for event in ready] == list(range(1, 17))
    output = admission.outputs[0].item.target
    with h5py.File(output, "r") as handle:
        for dimension in ("integrated_1d", "integrated_2d"):
            np.testing.assert_array_equal(
                handle[f"entry/{dimension}/frame_index"][:], np.arange(1, 17),
            )
    for label in (1, 8, 16):
        view = read_frame_record(output, label).active_view()
        assert view.intensity_1d.shape == (1000,)
        assert view.intensity_2d.shape == (500, 500)
        # Empty integration bins are NaN; each result must still contain data.
        assert np.isfinite(view.intensity_1d).any()
        assert np.isfinite(view.intensity_2d).any()


def test_stop_during_admitted_revalidation_returns_stopped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = image_series_spec(tmp_path / "raw_0001.tif")
    configuration = RunIntent(
        source_spec=source,
        poni_file=str(tmp_path / "calibration.poni"),
        save_path=str(tmp_path / "out.nxs"),
        output_mode="Overwrite",
    ).freeze()
    identity = RunIdentity.from_configuration(configuration)
    request = RequestId(1)
    capture = SourceCapture(request, 1, source)
    start = StartCapture(
        request,
        1,
        RunIntentStore(RunIntent.from_frozen(configuration)).snapshot(),
        capture,
    )
    receipt = admission_for(start)
    resources = SimpleNamespace(
        admission=receipt,
        directory_session=None,
    )
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        Path(configuration.save_path),
        resources=resources,
    )
    calls = 0

    def cancel_during_validation(
        accepted,
        session,
        *,
        cancelled,
    ) -> None:
        nonlocal calls
        calls += 1
        assert accepted is receipt
        assert session is None
        assert not cancelled()
        run.stop_requested = True
        assert cancelled()
        raise RuntimeError("admission cancelled")

    monkeypatch.setattr(
        executor_module,
        "validate_admitted_receipt",
        cancel_during_validation,
    )

    assert StandardRunExecutor()._execute_admitted(run) is True
    assert calls == 1


def test_executor_constructs_and_releases_each_standard_owner_once(monkeypatch, tmp_path: Path) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.reduction import NexusSink
    from xrd_tools.session import ScanSession
    from xrd_tools.sources.image import TiffSeriesSource
    from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run

    executor, run, _admission = _prepared_run(tmp_path)
    calls, sessions, sinks = {}, [], []
    originals = (
        (executor_module, "open_source"),
        (executor_module, "poni_to_integrator"),
        (executor_module, "native_int_reduction_plan"),
        (TiffSeriesSource, "to_scan"),
        (NexusSink, "begin"),
        (NexusSink, "finish"),
        (ScanSession, "start"),
        (ScanSession, "submit"),
        (ScanSession, "finish"),
    )
    for owner, name in originals:
        original = getattr(owner, name)
        key = (owner, name)
        def counted(*args, _original=original, _key=key, **kwargs):
            calls[_key] = calls.get(_key, 0) + 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(owner, name, counted)
    real_open = dynamic_output.open_headless_scan_session
    def open_session(*args, **kwargs):
        session = real_open(*args, **kwargs)
        sessions.append(session)
        sinks.append(kwargs["sink"])
        return session
    monkeypatch.setattr(dynamic_output, "open_headless_scan_session", open_session)
    closed_sources = []
    monkeypatch.setattr(TiffSeriesSource, "close", lambda source: closed_sources.append(source), raising=False)
    executor._run(run)
    terminal = executor.drain_events()[-1]
    assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
    assert len(sessions) == len(sinks) == len(closed_sources) == 1
    for owner, name in originals:
        # The executor asks for terminal truth twice; the real ScanSession
        # caches it and settles the sink exactly once.
        assert calls[(owner, name)] == (2 if (owner, name) == (ScanSession, "finish") else 1)
    assert sessions[0].terminal_result.commit_identity is not None
    assert executor.close(run.identity).cleanup_status.value == "cleaned"


def test_eiger_submission_uses_bounded_reads_and_stops_before_next_chunk(
    monkeypatch,
) -> None:
    identity = RunIdentity(1, "c" * 64)
    requested_chunks: list[int] = []
    yielded_chunks: list[int] = []

    class Source:
        frame_indices = tuple(range(16))
        def iter_chunks(self, size: int):
            requested_chunks.append(size)
            yielded_chunks.append(1)
            yield np.ones((8, 2, 2)), tuple(range(8))
            yielded_chunks.append(2)
            yield np.ones((8, 2, 2)), tuple(range(8, 16))

        def close(self) -> None:
            return None

    class Session:
        frames_completed = 0

        def start(self) -> None:
            return None

        def submit(self, _frame, _image) -> bool:
            return False

        def finish(self, **_kwargs):
            return SimpleNamespace(
                failed=False, cancelled=True, n_processed=0
            )

    class Scan:
        frames = tuple(SimpleNamespace(index=index) for index in range(16))

    monkeypatch.setattr(executor_module, "NexusStackSource", Source)
    run = _StandardRun(
        RunIntent().freeze(),
        identity,
        Scan(),
        Source(),
        Session(),
        None,
        Path("out.nxs"),
    )
    run.frames_by_label = {int(frame.index): frame for frame in run.scan.frames}
    StandardRunExecutor._submit_container_source(run, run.session)
    assert requested_chunks == [8]
    assert yielded_chunks == [1]


def test_eiger_source_read_overlaps_submit_backpressure(monkeypatch) -> None:
    identity = RunIdentity(1, "d" * 64)
    submit_entered = Event()
    release_submit = Event()
    second_read_started = Event()
    source_threads: list[str] = []
    submit_threads: list[str] = []

    class Source:
        frame_indices = (0, 1)
        def iter_chunks(self, _size: int):
            source_threads.append(current_thread().name)
            yield np.ones((1, 2, 2)), (0,)
            source_threads.append(current_thread().name)
            second_read_started.set()
            yield np.ones((1, 2, 2)), (1,)

    class Session:
        def submit(self, _frame, _image) -> bool:
            submit_threads.append(current_thread().name)
            if len(submit_threads) == 1:
                submit_entered.set()
                assert release_submit.wait(timeout=2.0)
            return True

        def stop(self) -> None:
            return None

    class Scan:
        frames = tuple(SimpleNamespace(index=index) for index in range(2))

    monkeypatch.setattr(executor_module, "NexusStackSource", Source)
    run = _StandardRun(
        RunIntent().freeze(),
        identity,
        Scan(),
        Source(),
        Session(),
        None,
        Path("out.nxs"),
    )
    run.frames_by_label = {
        int(frame.index): frame for frame in run.scan.frames
    }

    errors: list[BaseException] = []

    def execute() -> None:
        try:
            StandardRunExecutor._submit_container_source(
                run, run.session
            )
        except BaseException as error:
            errors.append(error)

    owner = Thread(target=execute, name="source-owner")
    owner.start()
    assert submit_entered.wait(timeout=2.0)
    assert second_read_started.wait(timeout=2.0)
    release_submit.set()
    owner.join(timeout=2.0)

    assert not owner.is_alive()
    assert errors == []
    assert source_threads == ["source-owner", "source-owner"]
    assert submit_threads == [
        "scattering-source-submit",
        "scattering-source-submit",
    ]


def test_eiger_source_read_failure_releases_waiting_submitter(monkeypatch) -> None:
    identity = RunIdentity(1, "e" * 64)

    class Source:
        frame_indices = ()
        def iter_chunks(self, _size: int):
            if False:
                yield
            raise ValueError("source read failed")

    class Session:
        def submit(self, _frame, _image) -> bool:
            raise AssertionError("no frame should be submitted")

        def stop(self) -> None:
            return None

    monkeypatch.setattr(executor_module, "NexusStackSource", Source)
    run = _StandardRun(
        RunIntent().freeze(),
        identity,
        SimpleNamespace(frames=()),
        Source(),
        Session(),
        None,
        Path("out.nxs"),
    )

    with pytest.raises(ValueError, match="source read failed"):
        StandardRunExecutor._submit_container_source(run, run.session)


def test_executor_failure_stop_and_close_release_resources_once(tmp_path, monkeypatch) -> None:
    from xrd_tools.reduction import NexusSink
    from xrd_tools.sources.image import TiffSeriesSource
    from tests.xdart.scattering.test_e1b1_terminal_cleanup import _prepared_run
    from tests.xdart.scattering.test_e1b2_executor_exact_review import _writer_begin_failure

    with monkeypatch.context() as patch:
        executor, run, _admission = _prepared_run(tmp_path / "failure")
        sinks, aborts, _held = _writer_begin_failure(patch)
        closed = []
        patch.setattr(TiffSeriesSource, "close", lambda source: closed.append(source), raising=False)
        executor._run(run)
        terminal = executor.drain_events()[-1]
        assert terminal.kind is StandardEventKind.FAILED
        assert len(closed) == 1 and aborts == sinks and len(sinks) == 1
        assert executor.close(run.identity).cleanup_status.value == "cleaned"
        assert len(closed) == len(aborts) == 1

    for action in ("stop", "close"):
        executor, run, admission = _prepared_run(tmp_path / action)
        decision = admission.outputs[0]
        executor._construct(run, item=decision.item, decision=decision)
        session, output = run.session, run.output
        source = run.source
        closed, finished = [], []
        real_finish = NexusSink.finish
        with monkeypatch.context() as patch:
            patch.setattr(TiffSeriesSource, "close", lambda owner: closed.append(owner), raising=False)
            def finish(owner, result):
                finished.append(owner)
                return real_finish(owner, result)
            patch.setattr(NexusSink, "finish", finish)
            # Preserve one genuine completed row before Stop/Close, so the
            # final processed record remains a nonempty current record.
            assert output.submit(run.scan.frames[0])
            assert session.pause(timeout=5.0)
            if action == "stop":
                executor.stop(run.identity)
                executor._run(run)
                terminal = executor.drain_events()[-1]
                assert terminal.kind is StandardEventKind.STOPPED, terminal.detail
            receipt = executor.close(run.identity)
            if action == "close":
                # This run has no source worker: Close settles its writer
                # after its initial display-retirement receipt was pending.
                assert receipt.cleanup_status.value == "cleanup_pending"
                assert run.output is run.session is run.sink is None
            else:
                assert receipt.cleanup_status.value == "cleaned"
            assert executor.close(run.identity).cleanup_status.value == "cleaned"
            assert closed == [source] and len(finished) == 1
            assert session.terminal_result.commit_identity is not None


def test_complete_eiger_submission_uses_cancel_aware_route_and_publishes_one_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from queue import Queue
    from xrd_tools.sources.eiger_direct_chunk import EigerDirectChunkFact
    fact = EigerDirectChunkFact("master.h5", True, "selected", 1, 8, 8, 2, 0, 4); queue_depths: list[int] = []; real_queue = Queue

    class Source:
        frame_indices = (0, 1)
        allocation = SimpleNamespace(queue_depth=1)
        _canonical_iter_chunks = None

        def __init__(self, direct_fact):
            self.direct_fact = direct_fact
            self.private_calls = 0
            self.public_calls: list[int] = []
            self.take_calls = 0
            self.cancelled = None

        def iter_chunks(self, _size):
            raise AssertionError("canonical public route bypassed the private route")

        def _iter_vnext_chunks(self, size, cancelled):
            self.private_calls += 1
            self.cancelled = cancelled
            assert size == 8 and not cancelled()
            yield np.arange(8).reshape(2, 2, 2), (0, 1)

        def load_frame(self, _label):
            raise AssertionError("complete selection used conventional random access")

        def take_direct_chunk_fact(self):
            self.take_calls += 1
            value, self.direct_fact = self.direct_fact, None
            return value

    Source._canonical_iter_chunks = Source.iter_chunks
    monkeypatch.setattr(executor_module, "NexusStackSource", Source)
    monkeypatch.setattr(
        executor_module, "Queue",
        lambda maxsize: queue_depths.append(maxsize) or real_queue(maxsize),
    )
    for override in (False, True):
        source = Source(fact if not override else None)
        if override:
            source.iter_chunks = lambda size: (
                source.public_calls.append(size)
                or iter(((np.arange(8).reshape(2, 2, 2), (0, 1)),))
            )
        submitted: list[int] = []
        run = SimpleNamespace(
            configuration=RunIntent().freeze(), source=source,
            frames_by_label={i: SimpleNamespace(index=i) for i in (0, 1)},
            stop_requested=False, stop_signal=Event(), context_runtime=None,
            resource_facts=[], cleanup_failures=[], perf_enabled=False,
        )
        output = SimpleNamespace(
            submit=lambda frame, _image: submitted.append(frame.index) or True,
        )
        StandardRunExecutor._submit_container_source(run, output)
        assert submitted == [0, 1] and source.take_calls == 1
        if override:
            assert source.public_calls == [8] and source.private_calls == 0
            assert run.resource_facts == []
        else:
            assert source.private_calls == 1 and source.public_calls == []
            assert callable(source.cancelled) and run.resource_facts == [fact]
    assert queue_depths == [1, 1]


@pytest.mark.parametrize("mode", ("success", "submit-false", "load-error", "pre-stop"))
def test_equal_length_noncomplete_eiger_bypass_counts_only_observed_fallbacks(
    mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrd_tools.sources.eiger_direct_chunk import EigerDirectChunkFact
    reason = "non-complete source selection uses random access"
    scenarios = (
        ("submit-false", "submit-error", "cancel-next")
        if mode == "submit-false" else (mode,)
    )
    load_error = RuntimeError("load failed"); submit_error = RuntimeError("submit failed")

    class Source:
        frame_indices = (0, 1)

        def __init__(self, scenario):
            self.scenario = scenario
            self.loads: list[int] = []
            self.notes: list[tuple[str, int]] = []
            self.fact = None
            self.take_calls = 0

        def load_frame(self, label):
            self.loads.append(label)
            if self.scenario == "load-error" and len(self.loads) == 2:
                raise load_error
            return np.full((2, 2), label)

        def note_direct_chunk_bypass(self, observed_reason, count):
            self.notes.append((observed_reason, count))
            self.fact = EigerDirectChunkFact(
                "master.h5", False, observed_reason, 1, 8, 8, 0, count, 0,
            )

        def take_direct_chunk_fact(self):
            self.take_calls += 1
            value, self.fact = self.fact, None
            return value

    monkeypatch.setattr(executor_module, "NexusStackSource", Source)
    for scenario in scenarios:
        source = Source(scenario)
        frames = {i: SimpleNamespace(index=i) for i in (1, 0)}
        run = SimpleNamespace(
            configuration=RunIntent().freeze(), source=source, frames_by_label=frames,
            stop_requested=scenario == "pre-stop", stop_signal=Event(),
            context_runtime=None, resource_facts=[], cleanup_failures=[],
        )

        def submit(_frame, _image):
            if scenario == "submit-error":
                raise submit_error
            if scenario == "cancel-next":
                run.stop_requested = True
            return scenario != "submit-false"

        expected_error = (
            load_error if scenario == "load-error"
            else submit_error if scenario == "submit-error" else None
        )
        if expected_error is None:
            StandardRunExecutor._submit_container_source(
                run, SimpleNamespace(submit=submit),
            )
        else:
            with pytest.raises(RuntimeError) as caught:
                StandardRunExecutor._submit_container_source(
                    run, SimpleNamespace(submit=submit),
                )
            assert caught.value is expected_error
        expected = 2 if scenario == "success" else 0 if scenario == "pre-stop" else 1
        assert tuple(frames) == (1, 0) and len(frames) == len(source.frame_indices)
        assert source.notes == [(reason, expected)] and source.take_calls == 1
        assert len(run.resource_facts) == 1
        observed = run.resource_facts[0]
        assert not observed.selected and observed.direct_frames == 0
        assert observed.reason == reason and observed.fallback_frames == expected
