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
from xrd_tools.session.run_configuration import RunIntent
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
    launched = pipeline.start(admission)

    assert isinstance(launched, StartLaunched)
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
    source = image_series_spec(tmp_path / "raw_0001.tif")
    configuration = RunIntent(
        source_spec=source, poni_file=str(tmp_path / "calibration.poni"),
        save_path=str(tmp_path / "out.nxs"), output_mode="Overwrite",
        max_cores=1,
    ).freeze()
    identity = RunIdentity.from_configuration(configuration)
    calls: dict[str, int] = {}

    def counted(name: str) -> None:
        calls[name] = calls.get(name, 0) + 1

    class Scan:
        name = "Standard"
        frames = (SimpleNamespace(index=1),)

        def __len__(self) -> int:
            return 1

    class Opened:
        def to_scan(self, **_kwargs):
            counted("scan")
            return Scan()

        def close(self) -> None:
            counted("source.close")

    sink_options: dict[str, object] = {}

    class Sink:
        def __init__(self, *_args, **kwargs) -> None:
            counted("sink")
            sink_options.update(kwargs)

        def finish(self, _result) -> None:
            counted("sink.finish")

    class Session:
        def __init__(self, _plan, _scan, sink, **_kwargs) -> None:
            counted("session")
            self._sink = sink
            self.frames_completed = 1

        def on_frame_completed(self, _callback) -> None:
            return None

        def start(self) -> None:
            counted("session.start")

        def submit(self, _frame) -> bool:
            counted("session.submit")
            return True

        def finish(self, **_kwargs):
            counted("session.finish")
            self._sink.finish(None)
            return SimpleNamespace(failed=False, cancelled=False, n_processed=1)

    opened = Opened()
    monkeypatch.setattr(executor_module, "open_source", lambda _source: (counted("source.open"), opened)[1])
    monkeypatch.setattr(executor_module, "load_poni", lambda _path: (counted("poni"), object())[1])
    monkeypatch.setattr(executor_module, "poni_to_integrator", lambda _poni: (counted("integrator"), object())[1])
    monkeypatch.setattr(
        executor_module,
        "build_native_int_reduction_plan_from_args",
        lambda *_args, **_kwargs: (counted("plan"), object())[1],
    )
    monkeypatch.setattr(executor_module, "NexusSink", Sink)
    monkeypatch.setattr(executor_module, "ScanSession", Session)

    executor = StandardRunExecutor()
    run = _StandardRun(
        configuration, identity, None, None, None, None, Path(configuration.save_path),
        capture=SourceCapture(RequestId(1), 1, source),
    )
    executor._construct(run)
    assert sink_options["flush_every"] == 8
    executor._run(run)

    assert calls == {
        "source.open": 1,
        "poni": 1,
        "integrator": 1,
        "scan": 1,
        "plan": 1,
        "sink": 1,
        "session": 1,
        "session.start": 1,
        "session.submit": 1,
        "session.finish": 1,
        "sink.finish": 1,
        "source.close": 1,
    }
    assert executor.drain_events()[-1].kind is StandardEventKind.FINISHED


def test_eiger_submission_uses_bounded_reads_and_stops_before_next_chunk(
    monkeypatch,
) -> None:
    identity = RunIdentity(1, "c" * 64)
    requested_chunks: list[int] = []
    yielded_chunks: list[int] = []

    class Source:
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
        None,
        identity,
        Scan(),
        Source(),
        Session(),
        None,
        Path("out.nxs"),
    )

    stopped = StandardRunExecutor()._execute_current(
        run, construct=False
    )

    assert stopped is True
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
        None,
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
        None,
        identity,
        SimpleNamespace(frames=()),
        Source(),
        Session(),
        None,
        Path("out.nxs"),
    )

    with pytest.raises(ValueError, match="source read failed"):
        StandardRunExecutor._submit_container_source(run, run.session)


def test_executor_failure_stop_and_close_release_resources_once() -> None:
    identity = RunIdentity(1, "f" * 64)

    class Source:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class Sink:
        def __init__(self) -> None:
            self.abort_calls = 0

        def abort(self, _result) -> None:
            self.abort_calls += 1

    class Session:
        def __init__(self, sink: Sink, *, submit: bool, cancelled: bool = False) -> None:
            self.sink = sink
            self.submit_result = submit
            self.cancelled = cancelled
            self.frames_completed = 0
            self.finish_calls = 0
            self.stop_calls = 0

        def start(self) -> None:
            return None

        def submit(self, _frame) -> bool:
            if self.submit_result:
                raise RuntimeError("reduction failed")
            return False

        def finish(self, **_kwargs):
            self.finish_calls += 1
            self.sink.abort(None)
            return SimpleNamespace(failed=False, cancelled=self.cancelled, n_processed=0)

        def stop(self) -> None:
            self.stop_calls += 1

    def run_for(source: Source, session: Session) -> _StandardRun:
        class Scan:
            name = "Standard"
            frames = (SimpleNamespace(index=1),)

            def __len__(self) -> int:
                return 1

        return _StandardRun(
            None, identity, Scan(), source, session, None, Path("out.nxs"),
        )

    failure_source, failure_sink = Source(), Sink()
    failure_session = Session(failure_sink, submit=True)
    failure_executor = StandardRunExecutor()
    failure_executor._run(run_for(failure_source, failure_session))
    assert failure_session.finish_calls == failure_sink.abort_calls == failure_source.close_calls == 1
    assert failure_executor.drain_events()[-1].kind is StandardEventKind.FAILED

    stop_source, stop_sink = Source(), Sink()
    stop_session = Session(stop_sink, submit=False, cancelled=True)
    stop_executor = StandardRunExecutor()
    stop_executor._run(run_for(stop_source, stop_session))
    assert stop_session.finish_calls == stop_sink.abort_calls == stop_source.close_calls == 1
    assert stop_executor.drain_events()[-1].kind is StandardEventKind.STOPPED

    close_source, close_sink = Source(), Sink()
    close_session = Session(close_sink, submit=False, cancelled=True)
    close_executor = StandardRunExecutor()
    close_run = run_for(close_source, close_session)
    close_executor._active = close_run
    close_executor.close(identity)
    close_executor.close(identity)
    assert close_session.stop_calls == close_session.finish_calls == close_sink.abort_calls == close_source.close_calls == 1
