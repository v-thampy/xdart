from pathlib import Path
from types import SimpleNamespace

from xdart.gui.tabs.scattering.adapters import run_executor as executor_module
from xdart.gui.tabs.scattering.adapters.run_executor import (
    StandardRunExecutor,
    _StandardRun,
)
from xdart.gui.tabs.scattering.contracts import (
    PlannedOutput,
    SourceCapture,
    SourceExecutionStamp,
    SourceFileState,
)
from xdart.gui.tabs.scattering.events import RequestId, RunIdentity
from xrd_tools.sources.execution_graph import freeze_source_execution_graph
from xrd_tools.sources.selection import image_series_spec

from tests.xdart.scattering.test_e3_context_contract import _configuration


def test_construct_loop_adopts_two_artifacts_as_one_atomic_current_scope(
    monkeypatch,
):
    configuration = _configuration()
    identity = RunIdentity.from_configuration(configuration)
    capture = SourceCapture(
        RequestId(1), 1, configuration.thaw_source_spec()
    )
    run = _StandardRun(
        configuration,
        identity,
        None,
        None,
        None,
        None,
        Path(configuration.save_path),
        capture=capture,
    )
    run.display.configure(partition_count=2, npt=2, frame_bytes=32)

    paths = (
        Path("/data/a/a_0001.tif"),
        Path("/data/b/b_0001.tif"),
    )
    items = tuple(
        PlannedOutput(
            freeze_source_execution_graph(
                image_series_spec(path), image_series_spec(path),
                source_path=path, group_key=path.stem,
                file=SourceFileState(str(path), 1, 1, 1, 1, index),
                adapter_id="tiff_series", frame_count=1, first_label=1,
                detector_shape=None, native_dtype=None,
            ),
            Path(f"/out/{name}.nxs"),
        )
        for index, (path, name) in enumerate(
            zip(paths, ("artifact-a", "artifact-b")), start=1
        )
    )

    scans = []
    closed_sources = []

    class Frame:
        index = 1
        image = None

    class Source:
        def __init__(self, spec):
            self.spec = spec

        def to_scan(self, **_kwargs):
            scan = SimpleNamespace(
                name=Path(self.spec.uri).name,
                frames=(Frame(),),
                gi_config=None,
            )
            scans.append(scan)
            return scan

        def close(self):
            closed_sources.append(self)

    class Session:
        def __init__(self, *_args, **_kwargs):
            self.frames_completed = 0
            self._completed = None

        def on_frame_completed(self, callback):
            self._completed = callback

        def start(self):
            pass

        def submit(self, _frame):
            return True

        def finish(self, **_kwargs):
            self.frames_completed = 1
            assert self._completed is not None
            self._completed(SimpleNamespace(frame_index=1))
            return SimpleNamespace(
                failed=False, cancelled=False, n_processed=1
            )

        monkeypatch.setattr(
            executor_module,
            "validate_planned_source",
            lambda _item, **_kwargs: None,
        )
    monkeypatch.setattr(executor_module, "open_source", Source)
    monkeypatch.setattr(executor_module, "load_poni", lambda _path: object())
    monkeypatch.setattr(
        executor_module, "poni_to_integrator", lambda _poni: object()
    )
    monkeypatch.setattr(
        executor_module,
        "build_native_int_reduction_plan_from_args",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        executor_module,
        "NexusSink",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(executor_module, "ScanSession", Session)

    executor = StandardRunExecutor()
    monkeypatch.setattr(
        executor,
        "_frame_ready_owned",
        lambda owned_run, _event, _image, _session: setattr(
            owned_run,
            "current_published",
            owned_run.current_published + 1,
        ),
    )
    executor._construct(run, item=items[0])
    context = run.context_runtime.context
    scan_a = scans[0]
    scope_a = context.current_scope
    assert scope_a.display_scan is scan_a
    assert context.scan is scan_a
    executor._execute_current(run, construct=False)

    executor._construct(run, item=items[1])
    scan_b = scans[1]
    scope_b = context.current_scope
    assert scope_b is not scope_a
    assert (
        scope_b.scan_key,
        scope_b.source,
        scope_b.display_scan,
        scope_b.commit_epoch,
    ) == (
        Path(items[1].source_spec.uri).name,
        str(items[1].source_spec.uri),
        scan_b,
        scope_a.commit_epoch + 1,
    )
    assert context.scan is scan_a
    assert context.commit_gate.epoch == scope_b.commit_epoch
    assert context.hydration_owner.scan_key == scope_b.scan_key
    assert context.hydration_owner.source == scope_b.source
    assert context.hydration_owner.epoch == scope_b.commit_epoch
    assert context.display_bindings().scan is scan_b
    assert tuple(run.display.artifacts) == tuple(
        str(item.target) for item in items
    )
    executor._execute_current(run, construct=False)
    assert len(closed_sources) == 2
