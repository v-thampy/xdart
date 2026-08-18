"""Finite P1-B output-graph and public-facade acceptance oracle."""

from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
import time
from types import SimpleNamespace

import fabio
import h5py
import numpy as np
import pytest

from tests.core._vnext_p0_c2_bridge_support import append_intent, live_scan
from tests.xdart.scattering._admission import await_admission
from tests.xdart.scattering._e2sd_support import write_poni
from xdart.gui.tabs.scattering.adapters.run_executor import StandardRunExecutor
from xdart.gui.tabs.scattering.contracts import (
    AdmissionFailure,
    AdmissionReceipt,
    SourceCapture,
    StartCapture,
)
from xdart.gui.tabs.scattering.display_values import StandardEventKind
from xdart.gui.tabs.scattering.events import (
    CleanupStatus,
    ExecutorAccepted,
    RequestId,
    RunIdentity,
)
from xdart.gui.tabs.scattering.output_values import APPEND_UNAVAILABLE
from xdart.gui.tabs.scattering.run_mode_projection import (
    UNOWNED_RUN_MODE_REASONS,
    build_run_strip_projection,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import image_series_spec
from xrd_tools.sources.selection import DirectorySourceSpec


ROOT = Path(__file__).resolve().parents[3]
SCATTERING = ROOT / "src/xdart/gui/tabs/scattering"
HOST_PAGE = ROOT / "src/xdart/gui/pages/scattering_workspace.py"
DYNAMIC_OUTPUT = SCATTERING / "adapters/dynamic_output.py"
RUN_EXECUTOR = SCATTERING / "adapters/run_executor.py"
TARGET_RESERVATION = SCATTERING / "adapters/target_reservation.py"
DISPLAY_RUNTIME = SCATTERING / "display_runtime.py"
PAGE = SCATTERING / "page.py"
HEADLESS_FACADE = ROOT / "src/xdart/modules/reduction.py"
_TERMINAL = {
    StandardEventKind.FINISHED,
    StandardEventKind.FAILED,
    StandardEventKind.STOPPED,
}


def _frame_events(events) -> tuple[object, ...]:
    return tuple(
        event
        for event in events
        if event.kind is StandardEventKind.FRAME_READY
    )


def _browser_captions(frames) -> tuple[str, ...]:
    from pyqtgraph.Qt import QtCore

    from xdart.gui.tabs.scattering.browser_model import FrameListModel

    model = FrameListModel()
    model.reconcile(tuple(frames))
    return tuple(
        str(model.data(
            model.index(row, 0),
            int(QtCore.Qt.ItemDataRole.DisplayRole),
        ))
        for row in range(model.rowCount())
    )


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _tree(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                bindings[alias.asname or alias.name] = (
                    f"{module}.{alias.name}" if module else alias.name
                )
        elif isinstance(node, ast.ClassDef):
            # Local authority clones are still constructor sites, but arbitrary
            # local variables are not assumed to name themselves.
            bindings.setdefault(node.name, node.name)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = node.value
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                value = node.value
                targets = (node.target,)
            else:
                continue
            resolved = _resolve_expression(value, bindings)
            if resolved is None:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id not in bindings
                ):
                    bindings[target.id] = resolved
                    changed = True
    return bindings


def _resolve_expression(
    node: ast.expr, bindings: dict[str, str],
) -> str | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _resolve_expression(node.value, bindings)
        return None if base is None else f"{base}.{node.attr}"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        base = _resolve_expression(node.args[0], bindings)
        return (
            None
            if base is None
            else f"{base}.{node.args[1].value}"
        )
    return None


def _constructor_sites(name: str) -> set[str]:
    roots = tuple(SCATTERING.rglob("*.py")) + (HOST_PAGE,)
    sites: set[str] = set()
    for path in roots:
        tree = _tree(path)
        bindings = _import_bindings(tree)
        if any(
            isinstance(node, ast.Call)
            and (
                (resolved := _resolve_expression(node.func, bindings)) == name
                or resolved is not None
                and resolved.endswith(f".{name}")
            )
            for node in ast.walk(tree)
        ):
            sites.add(str(path.relative_to(ROOT)))
    return sites


def _class_names(path: Path) -> set[str]:
    return {
        node.name
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ClassDef)
    }


def _literal_true_persisted_calls(
    path: Path, *, class_name: str, function_name: str,
) -> tuple[int, ...]:
    tree = _tree(path)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    function = next(
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    return tuple(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and any(
            keyword.arg == "persisted"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    )


def _configured_intent(*, mode: str, output: str) -> RunIntent:
    return RunIntent(
        source_spec=image_series_spec("/tmp/p1b-source.tif"),
        poni_file="/tmp/p1b.poni",
        save_path="/tmp/p1b-output.nexus",
        processing_mode=mode,
        output_mode=output,
    )


def _write_tiff(path: Path, value: int) -> None:
    fabio.tifimage.TifImage(
        data=np.full((195, 487), value, dtype=np.uint16)
    ).write(str(path))


def _write_eiger(master: Path, member: Path, frames: int) -> None:
    with h5py.File(member, "w") as handle:
        handle.create_dataset(
            "entry/data/data",
            data=np.ones((frames, 195, 487), dtype=np.uint16),
            chunks=(1, 195, 487),
            maxshape=(None, 195, 487),
        )
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data["data_000001"] = h5py.ExternalLink(
            str(member), "/entry/data/data",
        )


def _nexus_rows(path: Path) -> tuple[int, ...]:
    with h5py.File(path, "r") as handle:
        return tuple(
            int(value)
            for value in handle["entry/integrated_1d/frame_index"][()]
        )


def _intent(
    source: Path,
    target: Path,
    poni: Path,
    *,
    output_mode: str = "Overwrite",
    processing_mode: str = "Int 1D",
    npt: int = 8,
    live_mode: bool = False,
) -> RunIntent:
    return RunIntent(
        source_spec=image_series_spec(source, metadata_format=None),
        poni_file=str(poni),
        project_root=str(source.parent),
        save_path=str(target),
        output_mode=output_mode,
        processing_mode=processing_mode,
        live_mode=live_mode,
        max_cores=1,
        bai_1d_args={"npt": int(npt), "method": "numpy"},
        bai_2d_args={"npt_rad": 8, "npt_azim": 4, "method": "numpy"},
    )


def _live_directory_intent(
    root: Path,
    output_root: Path,
    poni: Path,
    *,
    processing_mode: str,
    output_mode: str = "Overwrite",
) -> RunIntent:
    return RunIntent(
        source_spec=DirectorySourceSpec(
            root,
            suffixes=(".tif",),
            metadata_format=None,
        ),
        poni_file=str(poni),
        project_root=str(root),
        save_path=str(output_root),
        output_mode=output_mode,
        processing_mode=processing_mode,
        live_mode=True,
        max_cores=1,
        bai_1d_args={"npt": 8, "method": "numpy"},
        bai_2d_args={"npt_rad": 8, "npt_azim": 4, "method": "numpy"},
    )


def _bridge_legacy_expected_target_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach P1-B's intended RED past one unrelated aggregate-parent skew."""

    from xrd_tools.reduction import NexusSink

    if DYNAMIC_OUTPUT.is_file():
        return
    if "expected_target_state" in inspect.signature(NexusSink).parameters:
        return
    real_init = NexusSink.__init__

    def compatible_init(owner, *args, **kwargs):
        kwargs.pop("expected_target_state", None)
        # The aggregate parent also feeds the older writer a provenance shape
        # containing E6's newer ``path`` field. That skew is not B02/B03.
        kwargs.pop("source_execution_provenance", None)
        kwargs.pop("source_snapshots_provenance", None)
        real_init(owner, *args, **kwargs)

    monkeypatch.setattr(NexusSink, "__init__", compatible_init)


def _spy_output_enumeration(
    monkeypatch: pytest.MonkeyPatch,
    output_root: Path,
) -> list[tuple[str, Path]]:
    calls: list[tuple[str, Path]] = []
    output_root = output_root.resolve()
    for method_name in ("iterdir", "glob", "rglob"):
        original = getattr(Path, method_name)

        def observed(owner, *args, _name=method_name, _original=original, **kwargs):
            candidate = owner.resolve(strict=False)
            if candidate == output_root or candidate.is_relative_to(output_root):
                calls.append((_name, candidate))
            return _original(owner, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, observed)
    return calls


def _capture(
    executor: StandardRunExecutor,
    intent: RunIntent,
    *,
    request_value: int,
) -> tuple[StartCapture, SourceCapture]:
    snapshot = RunIntentStore(intent).snapshot()
    request = RequestId(request_value)
    source = intent.source_spec
    source_capture = SourceCapture(request, 1, source)
    return (
        StartCapture(request, 1, snapshot, source_capture),
        source_capture,
    )


def _admit(
    executor: StandardRunExecutor,
    intent: RunIntent,
    *,
    request_value: int,
    timeout: float = 20.0,
) -> tuple[
    AdmissionReceipt | AdmissionFailure,
    SourceCapture,
    object,
]:
    capture, source_capture = _capture(
        executor, intent, request_value=request_value,
    )
    token = executor.begin_admission(capture)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = executor.poll_admission(token)
        if type(result) in {AdmissionReceipt, AdmissionFailure}:
            return result, source_capture, token
        time.sleep(0.01)
    raise AssertionError("P1-B output admission did not settle")


def _start(
    executor: StandardRunExecutor,
    intent: RunIntent,
    *,
    request_value: int,
) -> RunIdentity:
    capture, source_capture = _capture(
        executor, intent, request_value=request_value,
    )
    receipt = await_admission(executor, capture, timeout=20.0)
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)
    outcome = executor.start(
        configuration, source_capture, identity, receipt,
    )
    assert type(outcome) is ExecutorAccepted
    return identity


def _start_admitted(
    executor: StandardRunExecutor,
    intent: RunIntent,
    receipt: AdmissionReceipt,
    source_capture: SourceCapture,
) -> RunIdentity:
    configuration = intent.freeze()
    identity = RunIdentity.from_configuration(configuration)
    outcome = executor.start(
        configuration, source_capture, identity, receipt,
    )
    assert type(outcome) is ExecutorAccepted
    return identity


def _drain_until(
    executor: StandardRunExecutor,
    predicate,
    *,
    timeout: float = 30.0,
):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        events.extend(executor.drain_events())
        if predicate(events):
            return tuple(events)
        time.sleep(0.01)
    raise AssertionError("P1-B executor event did not arrive")


def _run_to_terminal(
    intent: RunIntent,
    *,
    request_value: int,
) -> tuple[StandardRunExecutor, RunIdentity, tuple[object, ...]]:
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(executor, intent, request_value=request_value)
    events = _drain_until(
        executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
    )
    return executor, identity, events


def test_p1b_b01_one_dynamic_graph_and_removal_census() -> None:
    """B01: one adapter constructs the graph; legacy owners are absent."""

    expected = {
        "src/xdart/gui/tabs/scattering/adapters/dynamic_output.py"
    }
    assert DYNAMIC_OUTPUT.is_file()
    assert _constructor_sites("ScanSession") == set()
    assert _constructor_sites("XYESink") == set()
    assert _constructor_sites("TargetLease") == set()
    assert _constructor_sites("TargetLease.acquire") == set()
    assert _constructor_sites("NexusSink") == expected
    assert _constructor_sites("TransactionalXYESink") == expected
    assert _constructor_sites("CompositeSink") == expected
    assert _constructor_sites("DynamicRunAccounting") == expected
    assert _constructor_sites("open_headless_scan_session") == expected
    assert _constructor_sites("DirectoryIndexSession") == {
        "src/xdart/gui/tabs/scattering/adapters/source.py",
        "src/xdart/gui/tabs/scattering/output_preflight.py",
    }

    reservation = _source(TARGET_RESERVATION)
    executor = _source(RUN_EXECUTOR)
    host = _source(HOST_PAGE)
    assert "TargetLease" not in _class_names(TARGET_RESERVATION)
    assert "target_lease" not in reservation
    assert "_reserved" not in reservation
    assert "_LIVE_SINK_FLUSH_EVERY" not in executor
    assert "_AppendHeldWriteMode" not in host
    for legacy_authority in ("owned_targets", "deferred_live_revisions"):
        assert legacy_authority not in executor, legacy_authority
    assert "current_published + 1" not in executor
    assert _literal_true_persisted_calls(
        DISPLAY_RUNTIME,
        class_name="RunDisplayState",
        function_name="retain_frame",
    ) == ()


def test_p1b_b02_overwrite_native_durable_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B02: FRAME_READY is not durable; exact receipts make it durable."""

    from xrd_tools.reduction import NexusSink

    _bridge_legacy_expected_target_state(monkeypatch)

    raw = tmp_path / "native_0001.tif"
    poni = tmp_path / "cal.poni"
    target = tmp_path / "native.nexus"
    _write_tiff(raw, 7)
    write_poni(poni)

    finish_entered = Event()
    release_finish = Event()
    real_finish = NexusSink.finish

    def held_finish(owner, result):
        finish_entered.set()
        assert release_finish.wait(timeout=10.0)
        return real_finish(owner, result)

    monkeypatch.setattr(NexusSink, "finish", held_finish)
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        _intent(raw, target, poni),
        request_value=1202,
    )
    try:
        events = _drain_until(
            executor,
            lambda values: (
                finish_entered.is_set()
                and any(
                    event.kind is StandardEventKind.FRAME_READY
                    for event in values
                )
            ) or any(event.kind in _TERMINAL for event in values),
            timeout=60.0,
        )
        early_terminal = next(
            (event for event in events if event.kind in _TERMINAL),
            None,
        )
        assert early_terminal is None, getattr(early_terminal, "primary", None)
        frame_event = next(
            event
            for event in events
            if event.kind is StandardEventKind.FRAME_READY
        )
        assert frame_event.frame_key is not None
        run = executor._exact_run(identity)
        assert run is not None
        owner = run.display.artifacts[frame_event.frame_key.artifact]
        records = owner.records
        label = frame_event.frame_key.local_frame_label
        persisted_before_terminal = records.is_persisted(label)
    finally:
        release_finish.set()

    terminal_events = _drain_until(
        executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
    )
    terminal = next(event for event in terminal_events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FINISHED
    assert persisted_before_terminal is False
    assert records.is_persisted(label) is True
    assert target.is_file()
    xye_files = tuple((tmp_path / "native").glob("*.xye"))
    assert [path.name for path in xye_files] == ["iq_native_0001.xye"]
    executor.close(identity)

    # A large container is fully discovered/revision-pinned at construction,
    # but attempts are minted only at each submit boundary. The named lineage
    # ceiling, rather than reduction in-flight, bounds the unsettled epoch.
    import xdart.gui.tabs.scattering.adapters.dynamic_output as dynamic_output
    from xdart.gui.tabs.scattering.output_preflight import execution_plan_values
    from xrd_tools.integrate.calibration import poni_to_integrator
    from xrd_tools.session.frame_record_store import FrameRecordStore
    from xrd_tools.session.readiness import (
        build_native_int_reduction_plan_from_args,
    )
    from xrd_tools.sources import open_source

    large = tmp_path / "large.nxs"
    with h5py.File(large, "w") as handle:
        detector = handle.create_group("entry/instrument/detector")
        detector.create_dataset(
            "data",
            shape=(129, 195, 487),
            maxshape=(None, 195, 487),
            chunks=(1, 195, 487),
            dtype=np.uint16,
            fillvalue=1,
        )
    large_poni = tmp_path / "large.poni"
    write_poni(large_poni)
    large_intent = _intent(large, tmp_path / "large-output.nexus", large_poni)
    admission_executor = StandardRunExecutor()
    large_admission, _large_capture, large_token = _admit(
        admission_executor, large_intent, request_value=1203,
    )
    assert type(large_admission) is AdmissionReceipt
    large_configuration = large_intent.freeze()
    large_decision = large_admission.outputs[0]
    source = open_source(large_decision.item.source_spec)
    accepted_poni = large_admission.scientific_assets.poni
    assert accepted_poni is not None
    scan = source.to_scan(
        poni=accepted_poni,
        integrator=poni_to_integrator(accepted_poni),
        output_path=large_decision.item.target,
    )
    args_1d, args_2d, values = execution_plan_values(
        large_configuration,
        large_admission.scientific_assets.mask,
    )
    plan = build_native_int_reduction_plan_from_args(
        args_1d, args_2d, **values,
    )

    adapter = dynamic_output.DynamicOutputAdapter(large_configuration)
    try:
        session, created = adapter.activate(
            scan,
            plan,
            large_decision.item,
            large_decision,
            record_store=FrameRecordStore(),
            run_provenance={
                **large_configuration.as_provenance(),
                "scientific_signature": (
                    large_admission.candidate.processing_mapping()
                ),
            },
        )
        assert created and session is adapter.session
        snapshot = adapter.accounting.snapshot()
        assert len(snapshot.discovered) == 129
        assert sum(len(value) for value in snapshot.attempts.values()) == 0
        session.start()
        import xrd_tools.reduction.core as reduction_core

        reduction_entered = Event()
        reduction_release = Event()
        real_reduce = reduction_core._reduce_frame

        def hold_first_reduction(*args, **kwargs):
            if not reduction_entered.is_set():
                reduction_entered.set()
                assert reduction_release.wait(5.0)
            return real_reduce(*args, **kwargs)

        monkeypatch.setattr(reduction_core, "_reduce_frame", hold_first_reduction)
        first = scan.frames[0]
        assert adapter.submit(first, source.load_frame(first.index))
        assert reduction_entered.wait(2.0)
        try:
            with pytest.raises(
                ValueError,
                match="dynamic live attempt must be failed or cancelled",
            ):
                adapter.submit(first, source.load_frame(first.index))
        finally:
            reduction_release.set()
        for frame in scan.frames[1:]:
            assert adapter.submit(frame, source.load_frame(frame.index))
        snapshot = adapter.accounting.snapshot()
        assert sum(len(value) for value in snapshot.attempts.values()) == 129
        assert {
            token.source_revision
            for attempts in snapshot.attempts.values()
            for token in attempts
        } == {1}
        result = adapter.finish_current()
        assert not result.failed
        settlements = []
        adapter.project_new_durable(
            lambda target, durable, delta: settlements.append(
                (target, durable, delta)
            ),
        )
        assert settlements == [
            (
                str(large_decision.item.target),
                tuple(range(129)),
                tuple(range(129)),
            )
        ]
        durable_labels = settlements[0][1]
        assert durable_labels == tuple(range(129))
        assert _nexus_rows(large_decision.item.target) == tuple(range(129))
        assert len(tuple((tmp_path / scan.name).glob("*.xye"))) == 129
        assert len(adapter._graphs) == 1

        # The shared supported-lineage check runs before target inspection,
        # label materialization, source decoding, or graph construction.
        import xdart.gui.tabs.scattering.output_preflight as output_preflight

        oversized_target = tmp_path / "oversized-output.nexus"
        oversized_stamp = replace(
            large_decision.item.source_stamp,
            frame_count=1_000_001,
        )
        oversized_item = replace(
            large_decision.item,
            target=oversized_target,
            source_stamp=oversized_stamp,
        )
        with monkeypatch.context() as bounded:
            bounded.setattr(
                output_preflight,
                "_path_state",
                lambda _path: pytest.fail("oversized target was inspected"),
            )
            with pytest.raises(
                ValueError,
                match="supported 1000000-frame ceiling",
            ):
                output_preflight.inspect_output(
                    oversized_item,
                    large_configuration,
                )
        assert not oversized_target.exists()
    finally:
        source.close()
        admission_executor.cancel_admission(large_token)


def test_p1b_b03_native_cross_run_append_missing_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B03: missing/no-op/mismatch use H23 before source pixel reads."""

    import xrd_tools.sources.image as image_source
    import xdart.gui.tabs.scattering.adapters.run_executor as run_executor_module

    _bridge_legacy_expected_target_state(monkeypatch)

    output_adapters = []
    real_output_adapter = run_executor_module.DynamicOutputAdapter

    def observed_output_adapter(configuration):
        owner = real_output_adapter(configuration)
        output_adapters.append(owner)
        return owner

    monkeypatch.setattr(
        run_executor_module, "DynamicOutputAdapter", observed_output_adapter,
    )

    raw1 = tmp_path / "series_0001.tif"
    raw2 = tmp_path / "series_0002.tif"
    raw3 = tmp_path / "series_0003.tif"
    poni = tmp_path / "cal.poni"
    target = tmp_path / "append.nexus"
    _write_tiff(raw1, 1)
    _write_tiff(raw2, 2)
    write_poni(poni)

    first_executor, _identity, first_events = _run_to_terminal(
        _intent(raw1, target, poni), request_value=1301,
    )
    assert next(event for event in first_events if event.kind in _TERMINAL).kind \
        is StandardEventKind.FINISHED
    first_executor.close(_identity)
    assert len(output_adapters) == 1
    assert output_adapters[0].persisted_prefix_labels == ()
    assert _nexus_rows(target) == (1, 2)
    xye_directory = tmp_path / "series"
    seeded_xye = tuple(sorted(xye_directory.glob("*.xye")))
    assert [path.name for path in seeded_xye] == [
        "iq_series_0001.xye",
        "iq_series_0002.xye",
    ]

    def xye_facts(paths):
        return {
            path.name: (
                path.read_bytes(),
                path.stat().st_dev,
                path.stat().st_ino,
                path.stat().st_mtime_ns,
            )
            for path in paths
        }

    seeded_xye_facts = xye_facts(seeded_xye)

    reads: list[Path] = []
    real_read = image_source.read_image

    def counted_read(path, *args, **kwargs):
        reads.append(Path(path).resolve())
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(image_source, "read_image", counted_read)

    from xrd_tools.session.dynamic_accounting import DynamicRunAccounting

    accounting_owners: list[DynamicRunAccounting] = []
    real_accounting_init = DynamicRunAccounting.__init__

    def observed_accounting_init(owner, *args, **kwargs):
        real_accounting_init(owner, *args, **kwargs)
        accounting_owners.append(owner)

    monkeypatch.setattr(
        DynamicRunAccounting, "__init__", observed_accounting_init,
    )

    # missing: a new member is the only source payload opened.
    _write_tiff(raw3, 3)
    missing_intent = _intent(raw1, target, poni, output_mode="Append")
    missing_executor = StandardRunExecutor(join_timeout=2.0)
    missing_admission, missing_capture, missing_token = _admit(
        missing_executor, missing_intent, request_value=1302,
    )
    if type(missing_admission) is AdmissionFailure:
        reason = missing_admission.reason
        missing_executor.cancel_admission(missing_token)
        pytest.fail(f"B03 missing: {reason}")
    missing_identity = _start_admitted(
        missing_executor,
        missing_intent,
        missing_admission,
        missing_capture,
    )
    missing_events = _drain_until(
        missing_executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
    )
    assert next(event for event in missing_events if event.kind in _TERMINAL).kind \
        is StandardEventKind.FINISHED, "B03 missing"
    assert reads == [raw3.resolve()], "B03 missing"
    assert _nexus_rows(target) == (1, 2, 3), "B03 missing"
    missing_frames = _frame_events(missing_events)
    assert len(missing_frames) == 1, "B03 missing"
    missing_frame = missing_frames[0]
    assert missing_frame.frame_key is not None
    assert missing_frame.frame_key.local_frame_label == 3
    missing_run = missing_executor._exact_run(missing_identity)
    assert missing_run is not None
    missing_catalog = missing_executor.frame_catalog(missing_identity)
    assert missing_catalog is not None
    assert tuple(
        key.local_frame_label for key in missing_catalog.entries
    ) == (1, 2, 3), "B03 missing persisted-prefix catalog"
    assert tuple(
        key.work_ordinal for key in missing_catalog.entries
    ) == (1, 2, 3), "B03 missing persisted-prefix ordinals"
    assert _browser_captions(missing_catalog.entries) == (
        "1", "2", "3",
    ), "B03 missing persisted-prefix browser"
    missing_context = missing_executor.acquisition_context(missing_identity)
    assert missing_context is not None
    assert missing_context.frame_ids.entries == missing_catalog.entries
    assert missing_run.display.artifacts[
        missing_frame.frame_key.artifact
    ].records.is_persisted(3)
    assert len(accounting_owners) == 1
    accounting = accounting_owners[0]
    accounting_snapshot = accounting.snapshot()
    required_targets = accounting.ledger.targets_by_mode
    fully_durable = {
        key
        for key in accounting_snapshot.discovered
        if all(
            (key, mode, output_target) in accounting_snapshot.durable
            for mode, targets in required_targets.items()
            for output_target in targets
        )
    }
    assert len(fully_durable) == 1
    missing_xye = tuple(sorted(xye_directory.glob("*.xye")))
    assert [path.name for path in missing_xye] == [
        "iq_series_0001.xye",
        "iq_series_0002.xye",
        "iq_series_0003.xye",
    ], "B03 missing"
    assert {
        name: xye_facts(missing_xye)[name]
        for name in seeded_xye_facts
    } == seeded_xye_facts, "B03 missing"
    persisted_key = missing_catalog.entries[0]
    assert missing_run.display.project(
        persisted_key,
        1,
        closed=missing_run.closed,
        owner=missing_context.hydration_owner,
        commit_gate=missing_context.commit_gate,
    ) is None
    hydrated_events = _drain_until(
        missing_executor,
        lambda values: any(
            event.kind is StandardEventKind.DISPLAY_READY
            and event.frame_key is persisted_key
            for event in values
        ),
    )
    assert not _frame_events(hydrated_events)
    hydrated = missing_run.display.project(
        persisted_key,
        1,
        closed=missing_run.closed,
        owner=missing_context.hydration_owner,
        commit_gate=missing_context.commit_gate,
    )
    assert hydrated is not None
    assert hydrated.frame_key is persisted_key
    assert hydrated.view.axis_1d is not None
    assert hydrated.view.intensity_1d is not None
    assert len(output_adapters) == 2
    assert output_adapters[1].persisted_prefix_labels == (1, 2)
    missing_executor.close(missing_identity)

    # no-op: exact committed content performs no source read or replacement.
    reads.clear()
    before = target.read_bytes()
    before_stat = target.stat()
    before_xye = xye_facts(tuple(sorted(xye_directory.glob("*.xye"))))
    noop_executor, noop_identity, noop_events = _run_to_terminal(
        _intent(raw1, target, poni, output_mode="Append"),
        request_value=1303,
    )
    assert next(event for event in noop_events if event.kind in _TERMINAL).kind \
        is StandardEventKind.FINISHED, "B03 no-op"
    after_stat = target.stat()
    assert reads == [], "B03 no-op"
    assert target.read_bytes() == before, "B03 no-op"
    assert (after_stat.st_dev, after_stat.st_ino) == (
        before_stat.st_dev,
        before_stat.st_ino,
    ), "B03 no-op"
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns, "B03 no-op"
    assert xye_facts(tuple(sorted(xye_directory.glob("*.xye")))) \
        == before_xye, "B03 no-op"
    noop_catalog = noop_executor.frame_catalog(noop_identity)
    assert noop_catalog is not None
    assert tuple(
        key.local_frame_label for key in noop_catalog.entries
    ) == (1, 2, 3), "B03 no-op persisted-prefix catalog"
    assert tuple(
        key.work_ordinal for key in noop_catalog.entries
    ) == (1, 2, 3), "B03 no-op persisted-prefix ordinals"
    assert _browser_captions(noop_catalog.entries) == (
        "1", "2", "3",
    ), "B03 no-op persisted-prefix browser"
    noop_context = noop_executor.acquisition_context(noop_identity)
    assert noop_context is not None
    assert noop_context.frame_ids.entries == noop_catalog.entries
    assert _frame_events(noop_events) == (), "B03 no-op synthetic frame event"
    assert len(output_adapters) == 3
    assert output_adapters[2].persisted_prefix_labels == (1, 2, 3)
    noop_executor.close(noop_identity)

    # Same-path scientific-asset drift is part of the signed science identity,
    # not a path-only match.  It refuses at preview before source pixels open.
    reads.clear()
    before = target.read_bytes()
    before_stat = target.stat()
    before_xye = xye_facts(tuple(sorted(xye_directory.glob("*.xye"))))
    original_poni = poni.read_bytes()
    changed_poni = original_poni.replace(b"0.1234", b"0.2234")
    assert changed_poni != original_poni
    poni.write_bytes(changed_poni)
    asset_executor = StandardRunExecutor()
    asset_admission, _asset_capture, asset_token = _admit(
        asset_executor,
        _intent(raw1, target, poni, output_mode="Append"),
        request_value=1305,
    )
    try:
        assert type(asset_admission) is AdmissionFailure, "B03 PONI drift"
        assert "Append" in asset_admission.reason, "B03 PONI drift"
        assert reads == [], "B03 PONI drift"
        after_stat = target.stat()
        assert target.read_bytes() == before, "B03 PONI drift"
        assert (after_stat.st_dev, after_stat.st_ino) == (
            before_stat.st_dev,
            before_stat.st_ino,
        ), "B03 PONI drift"
        assert xye_facts(tuple(sorted(xye_directory.glob("*.xye")))) \
            == before_xye, "B03 PONI drift"
    finally:
        asset_executor.cancel_admission(asset_token)
        poni.write_bytes(original_poni)

    # mismatch: foreign science refuses before the first pixel loader call.
    reads.clear()
    before = target.read_bytes()
    before_stat = target.stat()
    before_xye = xye_facts(tuple(sorted(xye_directory.glob("*.xye"))))
    mismatch = _intent(
        raw1, target, poni, output_mode="Append", npt=9,
    )
    mismatch_executor = StandardRunExecutor()
    admission, _capture_value, token = _admit(
        mismatch_executor, mismatch, request_value=1304,
    )
    try:
        assert type(admission) is AdmissionFailure, "B03 mismatch"
        assert "Append" in admission.reason, "B03 mismatch"
        assert reads == [], "B03 mismatch"
        after_stat = target.stat()
        assert target.read_bytes() == before, "B03 mismatch"
        assert (after_stat.st_dev, after_stat.st_ino) == (
            before_stat.st_dev,
            before_stat.st_ino,
        ), "B03 mismatch"
        assert xye_facts(tuple(sorted(xye_directory.glob("*.xye")))) \
            == before_xye, "B03 mismatch"
    finally:
        mismatch_executor.cancel_admission(token)

    # A very large committed prefix seeds only the retained navigation tail;
    # an exact key already in that tail is reused rather than duplicated.
    from xdart.gui.tabs.scattering.display_runtime import RunDisplayState

    display = RunDisplayState(
        RunIdentity(1, "append-prefix-tail"),
        max_payload_items=1,
        catalog_max_items=3,
    )
    artifact = str(Path("/processed/series.nexus"))
    owner = SimpleNamespace(source_scan="series", artifact=Path(artifact))
    run = SimpleNamespace(display=display)
    display.seed_navigation(owner.source_scan, artifact, 999_998)

    calls = []
    real_seed = display.seed_navigation

    def observed_seed(source_scan, target, label):
        calls.append((source_scan, target, label))
        return real_seed(source_scan, target, label)

    monkeypatch.setattr(display, "seed_navigation", observed_seed)
    StandardRunExecutor._seed_persisted_prefix_navigation(
        run,
        owner,
        tuple(range(1, 1_000_001)),
    )

    assert calls == [
        ("series", artifact, 999_998),
        ("series", artifact, 999_999),
        ("series", artifact, 1_000_000),
    ]
    entries = display.catalog_snapshot().entries
    assert tuple(key.local_frame_label for key in entries) == (
        999_998, 999_999, 1_000_000,
    )
    assert tuple(key.work_ordinal for key in entries) == (1, 2, 3)


def test_p1b_b04_xye_only_prefix_and_append_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B04: closed prefixes and executable XYE-only supported envelope."""

    from xrd_tools.io.output_transaction import OutputTransactionCoordinator
    from xrd_tools.reduction import TransactionalXYESink

    _bridge_legacy_expected_target_state(monkeypatch)
    prepare_calls: list[Path] = []
    real_prepare = OutputTransactionCoordinator.prepare_xye

    def observed_prepare(owner, directory, *, run_owner):
        prepare_calls.append(Path(directory).resolve())
        return real_prepare(owner, directory, run_owner=run_owner)

    monkeypatch.setattr(
        OutputTransactionCoordinator,
        "prepare_xye",
        observed_prepare,
    )

    assert "prefix" in inspect.signature(TransactionalXYESink).parameters
    allowed = ("iq", "itth", "iqip", "iqoop", "iexit")
    for prefix in allowed:
        sink = TransactionalXYESink(
            tmp_path / f"xye-{prefix}",
            prefix=prefix,
        )
        assert sink.pattern == f"{prefix}_{{scan}}_{{frame:04d}}.xye", prefix
        sink._scan_name = "bounded"
        assert sink._path_for(
            SimpleNamespace(index=7, label=7)
        ).name == f"{prefix}_bounded_0007.xye", prefix
        sink.abort(None)

    prepared_allowed = len(prepare_calls)
    for invalid in ("unknown", "../iq", "iq/name"):
        target = tmp_path / f"invalid-{invalid.replace('/', '-') }"
        with pytest.raises(ValueError, match="prefix"):
            TransactionalXYESink(target, prefix=invalid)
        assert not target.exists(), invalid
        assert len(prepare_calls) == prepared_allowed, invalid

    reasons = dict(UNOWNED_RUN_MODE_REASONS)
    assert "Int 1D (XYE)" not in reasons
    overwrite = build_run_strip_projection(
        RunPhase.IDLE,
        _configured_intent(mode="Int 1D (XYE)", output="Overwrite"),
        executor_available=True,
        start_permitted=True,
        start_blocker="",
    )
    append = build_run_strip_projection(
        RunPhase.IDLE,
        _configured_intent(mode="Int 1D (XYE)", output="Append"),
        executor_available=True,
        start_permitted=True,
        start_blocker="",
    )
    assert overwrite.ready
    assert not append.ready
    assert "XYE" in append.readiness and "Append" in append.readiness

    # Enter through vNext's public Live executor. The second member lands only
    # after frame one is writer-boundary durable, so this is a real same-session
    # continuation rather than a fixed two-frame batch.
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw1 = raw_root / "xye_0001.tif"
    raw2 = raw_root / "xye_0002.tif"
    poni = tmp_path / "cal.poni"
    output_root = tmp_path / "processed"
    _write_tiff(raw1, 1)
    write_poni(poni)
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(
        executor,
        _live_directory_intent(
            raw_root,
            output_root,
            poni,
            processing_mode="Int 1D (XYE)",
        ),
        request_value=1401,
    )

    def wait_xye_settled(prior, *, expected: int):
        values = list(prior)
        expected_names = tuple(
            f"iq_xye_{label:04d}.xye"
            for label in range(1, expected + 1)
        )
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            values.extend(executor.drain_events())
            frames = tuple(
                event
                for event in values
                if event.kind is StandardEventKind.FRAME_READY
            )
            last_frame = max(
                (
                    index
                    for index, event in enumerate(values)
                    if event.kind is StandardEventKind.FRAME_READY
                ),
                default=-1,
            )
            settled = any(
                index > last_frame
                and event.kind is StandardEventKind.DISCOVERY
                and (
                    event.files_processed,
                    event.files_skipped,
                    event.files_pending,
                    event.files_discovered,
                ) == (expected, 0, 0, expected)
                for index, event in enumerate(values)
            )
            paths = tuple(sorted(tmp_path.rglob("iq_xye_*.xye")))
            if (
                len(frames) >= expected
                and settled
                and tuple(path.name for path in paths) == expected_names
                and all(path.stat().st_size > 0 for path in paths)
            ):
                run = executor._exact_run(identity)
                assert run is not None
                assert all(
                    event.frame_key is not None
                    and run.display.artifacts[
                        event.frame_key.artifact
                    ].records.is_persisted(
                        event.frame_key.local_frame_label
                    )
                    for event in frames[:expected]
                )
                return tuple(values)
            time.sleep(0.01)
        raise AssertionError("B04 XYE-only group did not physically settle")

    events = _drain_until(
        executor,
        lambda values: any(
            event.kind is StandardEventKind.FRAME_READY
            for event in values
        ),
    )
    events = wait_xye_settled(events, expected=1)

    _write_tiff(raw2, 2)
    second_events = _drain_until(
        executor,
        lambda values: sum(
            event.kind is StandardEventKind.FRAME_READY
            for event in (*events, *values)
        ) >= 2,
    )
    combined = wait_xye_settled((*events, *second_events), expected=2)
    all_frame_events = [
        event
        for event in combined
        if event.kind is StandardEventKind.FRAME_READY
    ]
    assert len(all_frame_events) == 2
    executor.stop(identity)
    terminal_events = _drain_until(
        executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
    )
    assert next(
        event for event in terminal_events if event.kind in _TERMINAL
    ).kind is StandardEventKind.STOPPED
    assert len(all_frame_events) == 2
    assert tuple(tmp_path.rglob("*.nexus")) == ()
    assert [path.name for path in sorted(tmp_path.rglob("*.xye"))] == [
        "iq_xye_0001.xye",
        "iq_xye_0002.xye",
    ]
    executor.close(identity)

    # Cross-run XYE-only Append refuses before a transaction or any filename
    # enumeration under the output root.
    prepare_calls.clear()
    enumeration = _spy_output_enumeration(monkeypatch, output_root)
    append_executor = StandardRunExecutor()
    refusal, _, token = _admit(
        append_executor,
        _live_directory_intent(
            raw_root,
            output_root,
            poni,
            processing_mode="Int 1D (XYE)",
            output_mode="Append",
        ),
        request_value=1402,
    )
    try:
        assert type(refusal) is AdmissionFailure
        assert refusal.reason != APPEND_UNAVAILABLE
        assert "XYE" in refusal.reason and "Append" in refusal.reason
        assert prepare_calls == []
        assert enumeration == []
    finally:
        append_executor.cancel_admission(token)


def test_p1b_b17_collision_zero_frame_and_xye_append_refuse_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B17: real H23 collision/zero truth and typed XYE Append refusal."""

    from xdart.modules.reduction import open_live_scan_nexus_session
    from xrd_tools.io import (
        AppendPreflightState,
        LeaseOwner,
        LeaseUnavailable,
        OwnerToken,
        get_output_transaction_coordinator,
    )

    # Collision: the one H23 coordinator owns and types this outcome.
    coordinator = get_output_transaction_coordinator()
    target = tmp_path / "collision.nexus"
    transaction_owner = OwnerToken("p1b-collision-transaction")
    target_owner = OwnerToken("p1b-collision-target")
    transaction = coordinator.admit(
        target,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
    )
    owners = {
        role: OwnerToken(f"p1b-collision-{role.value}")
        for role in LeaseOwner
    }
    lease = transaction.acquire_lease(
        admission=transaction.admission,
        transaction_owner=transaction_owner,
        target_owner=target_owner,
        owners=owners,
    )
    contender_transaction_owner = OwnerToken("p1b-contender-transaction")
    contender_target_owner = OwnerToken("p1b-contender-target")
    contender = coordinator.admit(
        target,
        transaction_owner=contender_transaction_owner,
        target_owner=contender_target_owner,
    )
    contender_owners = {
        role: OwnerToken(f"p1b-contender-{role.value}")
        for role in LeaseOwner
    }
    with pytest.raises(LeaseUnavailable, match="already leased"):
        contender.acquire_lease(
            admission=contender.admission,
            transaction_owner=contender_transaction_owner,
            target_owner=contender_target_owner,
            owners=contender_owners,
        )
    transaction.abandon(lease)
    for role in LeaseOwner:
        transaction.release_lease_owner(lease, role, owners[role])
    contender_lease = contender.acquire_lease(
        admission=contender.admission,
        transaction_owner=contender_transaction_owner,
        target_owner=contender_target_owner,
        owners=contender_owners,
    )
    contender.abandon(contender_lease)
    for role in LeaseOwner:
        contender.release_lease_owner(
            contender_lease, role, contender_owners[role],
        )

    # Zero-frame Replace binds the valid lineage/target before frame zero and
    # aborts through the accepted H23 owner; vNext must preserve this truth.
    zero_target = tmp_path / "zero.nexus"
    zero_intent = append_intent(
        tmp_path,
        extent=1,
        labels=(0,),
        generation=0,
        source_identity="p1b/zero",
    )
    zero_session = open_live_scan_nexus_session(
        live_scan(zero_target, zero_intent, ()),
        replace=True,
    )
    zero_session.flush(force=True)
    assert zero_target.exists()
    zero_session.abort()

    zero_transaction_owner = OwnerToken("p1b-zero-reacquire-transaction")
    zero_target_owner = OwnerToken("p1b-zero-reacquire-target")
    zero_transaction = coordinator.admit(
        zero_target,
        transaction_owner=zero_transaction_owner,
        target_owner=zero_target_owner,
    )
    zero_owners = {
        role: OwnerToken(f"p1b-zero-reacquire-{role.value}")
        for role in LeaseOwner
    }
    zero_lease = zero_transaction.acquire_lease(
        admission=zero_transaction.admission,
        transaction_owner=zero_transaction_owner,
        target_owner=zero_target_owner,
        owners=zero_owners,
    )
    zero_transaction.abandon(zero_lease)
    for role in LeaseOwner:
        zero_transaction.release_lease_owner(
            zero_lease, role, zero_owners[role],
        )

    # Construction custody: a preflight acquired at JIT remains reachable
    # until the graph is registered, even when a later constructor fails.
    import xdart.gui.tabs.scattering.adapters.dynamic_output as dynamic_output

    custody_root = tmp_path / "custody"
    custody_root.mkdir()
    custody_raw1 = custody_root / "custody_0001.tif"
    custody_raw2 = custody_root / "custody_0002.tif"
    custody_poni = tmp_path / "custody.poni"
    custody_target = tmp_path / "custody.nexus"
    _write_tiff(custody_raw1, 1)
    write_poni(custody_poni)
    seed_executor, seed_identity, seed_events = _run_to_terminal(
        _intent(custody_raw1, custody_target, custody_poni),
        request_value=1704,
    )
    seed_terminal = next(
        event for event in seed_events if event.kind in _TERMINAL
    )
    assert seed_terminal.kind is StandardEventKind.FINISHED, seed_terminal.primary
    assert seed_executor.close(
        seed_identity
    ).cleanup_status is CleanupStatus.CLEANED
    _write_tiff(custody_raw2, 2)
    custody_before = (
        custody_target.read_bytes(),
        custody_target.stat().st_dev,
        custody_target.stat().st_ino,
        custody_target.stat().st_mtime_ns,
    )

    def assert_custody_target_reacquirable(label: str) -> None:
        transaction_authority = OwnerToken(f"{label}-transaction")
        target_authority = OwnerToken(f"{label}-target")
        admitted = coordinator.admit(
            custody_target,
            transaction_owner=transaction_authority,
            target_owner=target_authority,
        )
        lease_owners = {
            role: OwnerToken(f"{label}-{role.value}")
            for role in LeaseOwner
        }
        reacquired = admitted.acquire_lease(
            admission=admitted.admission,
            transaction_owner=transaction_authority,
            target_owner=target_authority,
            owners=lease_owners,
        )
        admitted.abandon(reacquired)
        for role in LeaseOwner:
            admitted.release_lease_owner(
                reacquired, role, lease_owners[role],
            )

    real_prepare = dynamic_output.prepare_append_preflight
    prebind_owners = []

    def capture_prebind(*args, **kwargs):
        owner = real_prepare(*args, **kwargs)
        prebind_owners.append(owner)
        return owner

    def fail_xye_construction(*_args, **_kwargs):
        raise RuntimeError("injected pre-bind XYE construction failure")

    with monkeypatch.context() as fault:
        fault.setattr(
            dynamic_output, "prepare_append_preflight", capture_prebind,
        )
        fault.setattr(
            dynamic_output, "TransactionalXYESink", fail_xye_construction,
        )
        prebind_executor, prebind_identity, prebind_events = _run_to_terminal(
            _intent(
                custody_raw1,
                custody_target,
                custody_poni,
                output_mode="Append",
            ),
            request_value=1705,
        )
    prebind_terminal = next(
        event for event in prebind_events if event.kind in _TERMINAL
    )
    assert prebind_terminal.kind is StandardEventKind.FAILED
    assert prebind_terminal.primary is not None
    assert prebind_terminal.primary.message == (
        "injected pre-bind XYE construction failure"
    )
    assert prebind_terminal.cleanup_status is CleanupStatus.CLEANED
    assert _frame_events(prebind_events) == ()
    assert len(prebind_owners) == 1
    assert prebind_owners[0].snapshot.state is AppendPreflightState.ABORTED
    assert prebind_executor.close(
        prebind_identity
    ).cleanup_status is CleanupStatus.CLEANED
    assert (
        custody_target.read_bytes(),
        custody_target.stat().st_dev,
        custody_target.stat().st_ino,
        custody_target.stat().st_mtime_ns,
    ) == custody_before
    assert_custody_target_reacquirable("p1b-prebind-reacquire")

    # A later Composite child can fail after Nexus has bound the preflight.
    # Two exact cleanup failures keep that authority reachable through the
    # executor; close retries the same sink and only then reports CLEANED.
    from xrd_tools.reduction import NexusSink, TransactionalXYESink

    begin_owners = []

    def capture_begin(*args, **kwargs):
        owner = real_prepare(*args, **kwargs)
        begin_owners.append(owner)
        return owner

    real_xye_begin = TransactionalXYESink.begin
    real_nexus_abort = NexusSink.abort
    abort_attempts = []

    def fail_later_begin(owner, scan, plan):
        real_xye_begin(owner, scan, plan)
        raise RuntimeError("injected later-child XYE begin failure")

    def retryable_nexus_abort(owner, result):
        if (
            Path(owner.path).resolve() == custody_target.resolve()
            and len(abort_attempts) < 2
        ):
            abort_attempts.append(owner)
            raise OSError("injected Nexus cleanup retry")
        return real_nexus_abort(owner, result)

    with monkeypatch.context() as fault:
        fault.setattr(
            dynamic_output, "prepare_append_preflight", capture_begin,
        )
        fault.setattr(TransactionalXYESink, "begin", fail_later_begin)
        fault.setattr(NexusSink, "abort", retryable_nexus_abort)
        begin_executor, begin_identity, begin_events = _run_to_terminal(
            _intent(
                custody_raw1,
                custody_target,
                custody_poni,
                output_mode="Append",
            ),
            request_value=1706,
        )
        begin_terminal = next(
            event for event in begin_events if event.kind in _TERMINAL
        )
        assert begin_terminal.kind is StandardEventKind.FAILED
        assert begin_terminal.primary is not None
        assert begin_terminal.primary.message == (
            "injected later-child XYE begin failure"
        )
        assert begin_terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert _frame_events(begin_events) == ()
        assert len(begin_owners) == 1
        assert begin_owners[0].snapshot.state in {
            AppendPreflightState.BOUND,
            AppendPreflightState.RETRYABLE,
        }
        assert len(abort_attempts) == 2
    assert begin_executor.close(
        begin_identity
    ).cleanup_status is CleanupStatus.CLEANED
    assert begin_owners[0].snapshot.state is AppendPreflightState.ABORTED
    assert (
        custody_target.read_bytes(),
        custody_target.stat().st_dev,
        custody_target.stat().st_ino,
        custody_target.stat().st_mtime_ns,
    ) == custody_before
    assert_custody_target_reacquirable("p1b-later-begin-reacquire")

    # Durable display projection is acknowledged independently per graph and
    # only after its callback succeeds. A partial callback failure must retain
    # B without replaying already-applied A on cleanup re-entry.
    from xrd_tools.session import DynamicFrameIdentity, ResultMode

    projection_adapter = dynamic_output.DynamicOutputAdapter(
        RunIntent().freeze()
    )
    projection_mode = ResultMode.one_d()
    projection_a = tmp_path / "projection-a.nexus"
    projection_b = tmp_path / "projection-b.nexus"
    key_a = DynamicFrameIdentity(
        ("p1b", "projection-a"), (str(projection_a), 7),
    )
    key_b = DynamicFrameIdentity(
        ("p1b", "projection-b"), (str(projection_b), 11),
    )

    class ProjectionAccounting:
        def __init__(self, key, target: str) -> None:
            self.key = key
            self.ledger = SimpleNamespace(
                targets_by_mode={projection_mode: (target,)}
            )
            self.target = target

        def snapshot(self):
            return SimpleNamespace(
                discovered=(self.key,),
                durable=frozenset({(
                    self.key, projection_mode, self.target,
                )}),
            )

    graph_a = {
        "accounting": ProjectionAccounting(key_a, "nexus:a"),
        "settled_labels": set(),
        "projection_pending": True,
        "item": SimpleNamespace(target=projection_a),
    }
    graph_b = {
        "accounting": ProjectionAccounting(key_b, "nexus:b"),
        "settled_labels": set(),
        "projection_pending": True,
        "item": SimpleNamespace(target=projection_b),
    }
    projection_adapter._graphs = {"a": graph_a, "b": graph_b}
    projection_adapter._current = graph_a
    projected = []
    fail_b_once = [True]

    def partially_project(target, durable_labels, new_labels) -> None:
        projected.append((target, durable_labels, new_labels))
        if target == str(projection_b) and fail_b_once[0]:
            fail_b_once[0] = False
            raise RuntimeError("injected graph-B projection failure")

    with pytest.raises(
        RuntimeError, match="injected graph-B projection failure",
    ):
        projection_adapter.project_new_durable(partially_project)
    assert projected == [
        (str(projection_a), (7,), (7,)),
        (str(projection_b), (11,), (11,)),
    ]
    assert graph_a["settled_labels"] == {7}
    assert not graph_a["projection_pending"]
    assert graph_b["settled_labels"] == set()
    assert graph_b["projection_pending"]
    projected.clear()
    projection_adapter.project_new_durable(partially_project)
    assert projected == [(str(projection_b), (11,), (11,))]
    assert graph_b["settled_labels"] == {11}
    assert not graph_b["projection_pending"]
    projected.clear()
    projection_adapter.project_new_durable(partially_project)
    assert projected == []
    graph_a["projection_pending"] = True
    projection_adapter.project_new_durable(partially_project)
    assert projected == [(str(projection_a), (7,), ())]
    assert not graph_a["projection_pending"]
    projected.clear()
    projection_adapter.project_new_durable(partially_project)
    assert projected == []

    # Cleanup projects the durable prefix from every graph even when a later
    # graph's finish raises.  Re-entry retries only the unsettled graph's
    # nonempty delta and preserves the original run failure independently.
    from xdart.gui.tabs.scattering.adapters.run_executor import _StandardRun
    from xdart.gui.tabs.scattering.events import detach_exception

    class SettlementAccounting:
        def __init__(self, key, target: str, settled) -> None:
            self.key = key
            self.target = target
            self.settled = settled
            self.ledger = SimpleNamespace(
                targets_by_mode={projection_mode: (target,)}
            )

        def snapshot(self):
            durable = (
                frozenset({(
                    self.key, projection_mode, self.target,
                )})
                if self.settled[0]
                else frozenset()
            )
            return SimpleNamespace(
                discovered=(self.key,),
                durable=durable,
            )

    class FinishSession:
        is_running = False

        def __init__(self, settled, *, fail_once: bool = False) -> None:
            self.settled = settled
            self.fail_once = fail_once
            self.calls = 0

        def finish(self, *, raise_on_failure: bool = False):
            assert raise_on_failure is False
            self.calls += 1
            if self.fail_once:
                self.fail_once = False
                raise OSError("injected graph-B finish failure")
            self.settled[0] = True
            return SimpleNamespace(failed=False, cancelled=False)

    class CleanupProjectionAdapter(dynamic_output.DynamicOutputAdapter):
        def __init__(self, configuration) -> None:
            super().__init__(configuration)
            self.projections = []

        def project_new_durable(self, apply) -> None:
            def observed(artifact, durable_labels, new_labels) -> None:
                self.projections.append((
                    artifact, durable_labels, new_labels,
                ))
                apply(artifact, durable_labels, new_labels)

            super().project_new_durable(observed)

    settled_a = [False]
    settled_b = [False]
    session_a = FinishSession(settled_a)
    session_b = FinishSession(settled_b, fail_once=True)
    cleanup_adapter = CleanupProjectionAdapter(RunIntent().freeze())
    cleanup_graph_a = {
        "session": session_a,
        "transition": "finish",
        "stop_requested": False,
        "projection_pending": False,
        "accounting": SettlementAccounting(
            key_a, "nexus:a", settled_a,
        ),
        "settled_labels": set(),
        "item": SimpleNamespace(target=projection_a),
    }
    cleanup_graph_b = {
        "session": session_b,
        "transition": "finish",
        "stop_requested": False,
        "projection_pending": False,
        "accounting": SettlementAccounting(
            key_b, "nexus:b", settled_b,
        ),
        "settled_labels": set(),
        "item": SimpleNamespace(target=projection_b),
    }
    cleanup_adapter._graphs = {
        "a": cleanup_graph_a,
        "b": cleanup_graph_b,
    }
    cleanup_adapter._current = cleanup_graph_a
    cleanup_configuration = RunIntent().freeze()
    cleanup_identity = RunIdentity.from_configuration(
        cleanup_configuration
    )
    cleanup_run = _StandardRun(
        cleanup_configuration,
        cleanup_identity,
        None,
        None,
        object(),
        None,
        projection_b,
        sink=cleanup_adapter,
        output=cleanup_adapter,
        total=2,
        current_total=1,
    )
    cleanup_run.display.configure(
        partition_count=2, npt=2, frame_bytes=48,
    )
    cleanup_run.display.add_artifact(
        projection_a,
        "projection-a",
        mask=None,
        mask_saturation=False,
        measurement_mode="Standard",
    )
    cleanup_run.display.add_artifact(
        projection_b,
        "projection-b",
        mask=None,
        mask_saturation=False,
        measurement_mode="Standard",
    )
    cleanup_executor = StandardRunExecutor()
    original_failure = detach_exception(
        RuntimeError("injected original run failure"), "run",
    )

    first_cleanup = cleanup_executor._cleanup(
        cleanup_run, original_failure,
    )
    assert first_cleanup.cleanup_status is CleanupStatus.CLEANUP_PENDING
    assert first_cleanup.primary is not None
    assert first_cleanup.primary.message == "injected original run failure"
    assert any(
        failure.message == "injected graph-B finish failure"
        for failure in first_cleanup.cleanup_failures
    )
    assert cleanup_run.completed == 1
    assert cleanup_run.artifacts == [projection_a]
    assert cleanup_graph_a["settled_labels"] == {7}
    assert cleanup_graph_b["settled_labels"] == set()
    assert cleanup_run.output is cleanup_adapter
    assert cleanup_adapter.projections == [
        (str(projection_a), (7,), (7,)),
    ]

    second_cleanup = cleanup_executor._cleanup(cleanup_run)
    stable_cleanup = (
        cleanup_run.completed,
        tuple(cleanup_run.artifacts),
        tuple(cleanup_graph_a["settled_labels"]),
        tuple(cleanup_graph_b["settled_labels"]),
        tuple(cleanup_adapter.projections),
    )
    third_cleanup = cleanup_executor._cleanup(cleanup_run)
    assert second_cleanup.cleanup_status is CleanupStatus.CLEANED
    assert second_cleanup.primary is not None
    assert second_cleanup.primary.message == "injected original run failure"
    assert cleanup_run.completed == 2
    assert cleanup_run.artifacts == [projection_a, projection_b]
    assert cleanup_graph_a["settled_labels"] == {7}
    assert cleanup_graph_b["settled_labels"] == {11}
    assert cleanup_run.current_completed == 1
    assert cleanup_run.current_published == 1
    assert cleanup_run.output is None
    assert cleanup_run.session is None
    assert cleanup_run.sink is None
    assert cleanup_adapter.projections == [
        (str(projection_a), (7,), (7,)),
        (str(projection_a), (7,), ()),
        (str(projection_b), (11,), (11,)),
    ]
    assert third_cleanup.cleanup_status is CleanupStatus.CLEANED
    assert stable_cleanup == (
        cleanup_run.completed,
        tuple(cleanup_run.artifacts),
        tuple(cleanup_graph_a["settled_labels"]),
        tuple(cleanup_graph_b["settled_labels"]),
        tuple(cleanup_adapter.projections),
    )

    # Enter the same collision through the public vNext executor. Holding the
    # first writer at finish keeps both the legacy parent reservation and the
    # candidate's H23 lease live. The candidate must delegate the contender's
    # exact typed failure to H23 instead of guessing path state at admission.
    _bridge_legacy_expected_target_state(monkeypatch)
    first_root = tmp_path / "collision-first"
    second_root = tmp_path / "collision-second"
    first_root.mkdir()
    second_root.mkdir()
    first_raw = first_root / "shared_0001.tif"
    second_raw = second_root / "shared_0001.tif"
    shared_target = tmp_path / "executor-collision.nexus"
    collision_poni = tmp_path / "collision.poni"
    _write_tiff(first_raw, 1)
    _write_tiff(second_raw, 2)
    write_poni(collision_poni)

    finish_entered = Event()
    release_finish = Event()
    real_finish = NexusSink.finish
    held_owner = []

    def held_finish(owner, result):
        if (
            Path(owner.path).resolve() == shared_target.resolve()
            and not held_owner
        ):
            held_owner.append(owner)
            finish_entered.set()
            release_finish.wait(60.0)
        return real_finish(owner, result)

    monkeypatch.setattr(NexusSink, "finish", held_finish)
    first_executor = StandardRunExecutor(join_timeout=2.0)
    second_executor = StandardRunExecutor(join_timeout=2.0)
    first_identity = None
    second_identity = None
    second_token = None
    second_events = ()
    first_terminal = ()
    try:
        first_identity = _start(
            first_executor,
            _intent(
                first_raw,
                shared_target,
                collision_poni,
                processing_mode="Int 2D",
            ),
            request_value=1702,
        )
        assert finish_entered.wait(60.0)
        contender_intent = _intent(
            second_raw,
            shared_target,
            collision_poni,
            processing_mode="Int 2D",
        )
        contender_admission, contender_capture, second_token = _admit(
            second_executor,
            contender_intent,
            request_value=1703,
        )
        assert type(contender_admission) is AdmissionReceipt, (
            "B17 routed collision was guessed during vNext admission instead "
            "of being typed by the canonical H23 lease owner"
        )
        second_identity = _start_admitted(
            second_executor,
            contender_intent,
            contender_admission,
            contender_capture,
        )
        second_events = _drain_until(
            second_executor,
            lambda values: any(event.kind in _TERMINAL for event in values),
            timeout=60.0,
        )
        collision_terminal = next(
            event for event in second_events if event.kind in _TERMINAL
        )
        normalized_target = os.path.normcase(os.path.abspath(shared_target))
        assert collision_terminal.kind is StandardEventKind.FAILED
        assert collision_terminal.primary is not None
        assert collision_terminal.primary.type_module == (
            "xrd_tools.io.output_transaction"
        )
        assert collision_terminal.primary.type_qualname == (
            LeaseUnavailable.__qualname__
        )
        assert collision_terminal.primary.message == (
            f"target already leased: {normalized_target}"
        )
        assert collision_terminal.primary.operation == ""
        assert collision_terminal.cleanup_status is CleanupStatus.CLEANED
        assert _frame_events(second_events) == ()
        assert collision_terminal.artifacts == ()
        contender_catalog = second_executor.frame_catalog(second_identity)
        assert contender_catalog is not None
        assert contender_catalog.entries == ()
    finally:
        if second_token is not None and second_identity is None:
            second_executor.cancel_admission(second_token)
        release_finish.set()
        if first_identity is not None:
            try:
                try:
                    first_terminal = _drain_until(
                        first_executor,
                        lambda values: any(
                            event.kind in _TERMINAL for event in values
                        ),
                        timeout=60.0,
                    )
                except AssertionError:
                    first_executor.stop(first_identity)
                    first_terminal = _drain_until(
                        first_executor,
                        lambda values: any(
                            event.kind in _TERMINAL for event in values
                        ),
                        timeout=60.0,
                    )
            finally:
                assert first_executor.close(
                    first_identity
                ).cleanup_status is CleanupStatus.CLEANED
        if second_identity is not None:
            if not any(event.kind in _TERMINAL for event in second_events):
                second_executor.stop(second_identity)
                second_events = _drain_until(
                    second_executor,
                    lambda values: any(
                        event.kind in _TERMINAL for event in values
                    ),
                    timeout=60.0,
                )
            assert second_executor.close(
                second_identity
            ).cleanup_status is CleanupStatus.CLEANED

    # XYE-only cross-run Append is not the old global Append-unavailable hold;
    # it is a specific pre-effect capability refusal with no output bytes.
    from xrd_tools.io.output_transaction import OutputTransactionCoordinator

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "xye_append_0001.tif"
    poni = tmp_path / "cal.poni"
    _write_tiff(raw, 1)
    write_poni(poni)
    output_root = tmp_path / "processed"
    prepare_calls: list[Path] = []
    real_prepare = OutputTransactionCoordinator.prepare_xye

    def observed_prepare(owner, directory, *, run_owner):
        prepare_calls.append(Path(directory).resolve())
        return real_prepare(owner, directory, run_owner=run_owner)

    monkeypatch.setattr(
        OutputTransactionCoordinator, "prepare_xye", observed_prepare,
    )
    enumeration = _spy_output_enumeration(monkeypatch, output_root)
    append_executor = StandardRunExecutor()
    refusal, _, token = _admit(
        append_executor,
        _intent(
            raw,
            output_root / "xye-append.nexus",
            poni,
            output_mode="Append",
            processing_mode="Int 1D (XYE)",
        ),
        request_value=1701,
    )
    try:
        assert type(refusal) is AdmissionFailure
        assert refusal.reason != APPEND_UNAVAILABLE
        assert "XYE" in refusal.reason and "Append" in refusal.reason
        assert prepare_calls == []
        assert enumeration == []
        assert not output_root.exists()
    finally:
        append_executor.cancel_admission(token)


@pytest.mark.parametrize(
    ("pipeline", "expected"),
    (
        (None, (8, 16, None, 56, 8, 4, 4)),
        ((16, 16, 48), (16, 16, 48, 48, 16, 4, 4)),
        ((1, 4, 56), (1, 4, 56, 56, 1, 4, 4)),
        ((1, 8, 16, 56), (1, 16, 56, 56, 1, 4, 4)),
        ((1, 8, 8, 64, 72), (1, 8, 64, 64, 1, 4, 4)),
    ),
    ids=(
        "default", "diagnostic-16-16-48", "diagnostic-1-4-56",
        "v2-settlement1-record8-inflight16-checkpoint56",
        "v2-funded-staging72-checkpoint64",
    ),
)
def test_post_g2_pipeline_option_and_absent_defaults_plumb_exact_owned_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog,
    pipeline: tuple[int, ...] | None, expected: tuple[int, ...],
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    raw = tmp_path / "pipeline_0001.tif"
    target = tmp_path / "pipeline.nexus"
    poni = tmp_path / "pipeline.poni"
    _write_tiff(raw, 7)
    write_poni(poni)
    intent = _intent(raw, target, poni)
    intent.max_cores = 4
    if pipeline is not None:
        if len(pipeline) == 3:
            intent.run_options["_post_g2_pipeline"] = {
                "writer_batch_size": pipeline[0],
                "reduction_inflight": pipeline[1],
                "checkpoint_frame_cap": pipeline[2],
            }
        else:
            intent.run_options["_post_g2_pipeline_v2"] = {
                "writer_settlement_batch_size": pipeline[0],
                "nexus_record_batch_size": pipeline[1],
                "reduction_inflight": pipeline[2],
                "semantic_checkpoint_frame_cap": pipeline[3],
            }
            if len(pipeline) == 5:
                intent.run_options["_post_g2_pipeline_v2"][
                    "staging_frame_cap"
                ] = pipeline[4]
    resolve = dynamic_output.resolve_session_policy
    open_session = dynamic_output.open_headless_scan_session
    observed = []
    v2_observed = []

    def fixed_envelope(requirements, **kwargs):
        kwargs["envelope_bytes"] = 64 * 1024 ** 3
        return resolve(requirements, **kwargs)

    def capture_session(*args, **kwargs):
        session = open_session(*args, **kwargs)
        observed.append((
            kwargs["sink"].writer_batch_size, kwargs["inflight_max"],
            kwargs["dynamic_nexus_checkpoint_threshold"],
            session._dynamic_nexus_checkpoint_threshold,
            session._session._writer_batch_size,
            kwargs["executor"], session._session._worker._max_workers,
        ))
        if pipeline is not None and len(pipeline) in (4, 5):
            children = kwargs["sink"].output_sink_children
            (nexus,) = tuple(
                child for child in children
                if type(child).__name__ == "NexusSink"
            )
            v2_observed.append((
                session._session._writer_batch_size,
                session._policy.allocation.reduction_inflight,
                session._session.inflight_max,
                kwargs["executor"], session._session._worker._max_workers,
                nexus.nexus_record_batch_size,
                tuple(type(child).__name__ for child in children),
            ))
        return session

    monkeypatch.setattr(
        dynamic_output, "resolve_session_policy", fixed_envelope,
    )
    monkeypatch.setattr(
        dynamic_output, "open_headless_scan_session", capture_session,
    )
    caplog.set_level(logging.INFO, logger=dynamic_output.__name__)
    executor, identity, events = _run_to_terminal(
        intent, request_value=1710,
    )
    try:
        terminal = next(event for event in events if event.kind in _TERMINAL)
        assert terminal.kind is StandardEventKind.FINISHED
        assert _nexus_rows(target) == (1,)
        assert observed == [expected]
        assert v2_observed == ([] if pipeline is None or len(pipeline) == 3 else [
            (
                1,
                pipeline[2],
                pipeline[2],
                4,
                4,
                8,
                ("TransactionalXYESink", "NexusSink"),
            ),
        ])
        facts = [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith("[RUN-PIPELINE]")
        ]
        expected_facts = (
            [] if pipeline is None or len(pipeline) in (4, 5) else [
                f"[RUN-PIPELINE] requested-batch={pipeline[0]} "
                f"effective-batch={pipeline[0]} "
                f"requested-inflight={pipeline[1]} "
                f"effective-inflight={pipeline[1]} "
                f"requested-checkpoint={pipeline[2]} "
                f"effective-checkpoint={pipeline[2]}"
            ]
        )
        assert facts == expected_facts
        v2_facts = [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith("[RUN-PIPELINE-V2]")
        ]
        if pipeline is None or len(pipeline) == 3:
            assert v2_facts == []
        else:
            assert len(v2_facts) == 1
            assert (
                f"requested-inflight={pipeline[2]} "
                f"effective-inflight={pipeline[2]}"
            ) in v2_facts[0]
            assert (
                f"requested-checkpoint={pipeline[3]} "
                f"effective-checkpoint={pipeline[3]}"
            ) in v2_facts[0]
            assert (
                "requested-staging=64 effective-staging=64"
                if len(pipeline) == 4
                else "requested-staging=72 effective-staging=72"
            ) in v2_facts[0]
    finally:
        executor.close(identity)


def test_post_g2_funded_staging_partial_grant_refuses_before_output_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    raw = tmp_path / "funded_0001.tif"
    target = tmp_path / "funded.nexus"
    poni = tmp_path / "funded.poni"
    _write_tiff(raw, 7)
    write_poni(poni)
    intent = _intent(raw, target, poni)
    intent.max_cores = 4
    intent.run_options["_post_g2_pipeline_v2"] = {
        "writer_settlement_batch_size": 1,
        "nexus_record_batch_size": 8,
        "reduction_inflight": 8,
        "semantic_checkpoint_frame_cap": 10_000,
        "staging_frame_cap": 10_008,
    }
    resolve = dynamic_output.resolve_session_policy

    def constrained(requirements, **kwargs):
        kwargs["envelope_bytes"] = 5 * 1024 ** 3
        return resolve(requirements, **kwargs)

    monkeypatch.setattr(dynamic_output, "resolve_session_policy", constrained)
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(executor, intent, request_value=1711)
    events = _drain_until(
        executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
    )
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FAILED
    assert "staging request was not granted exactly" in terminal.detail
    assert "requested=10008" in terminal.detail
    assert not target.exists()
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED

    intent.run_options["_post_g2_pipeline_v2"] = {
        "writer_settlement_batch_size": 1,
        "nexus_record_batch_size": 8,
        "reduction_inflight": 8,
        "semantic_checkpoint_frame_cap": 1_000,
        "staging_frame_cap": 1_008,
    }
    identity = _start(executor, intent, request_value=1712)
    events = _drain_until(
        executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
    )
    terminal = next(event for event in events if event.kind in _TERMINAL)
    assert terminal.kind is StandardEventKind.FAILED
    assert "checkpoint retention was not funded" in terminal.detail
    assert "checkpoint=1000" in terminal.detail
    assert not target.exists()
    assert executor.close(identity).cleanup_status is CleanupStatus.CLEANED

    intent.run_options["_post_g2_pipeline_v2"] = {
        "writer_settlement_batch_size": 1,
        "nexus_record_batch_size": 8,
        "reduction_inflight": 8,
        "semantic_checkpoint_frame_cap": 56,
        "staging_frame_cap": 64,
    }
    identity = _start(executor, intent, request_value=1714)
    events = _drain_until(
        executor,
        lambda values: any(event.kind in _TERMINAL for event in values),
    )
    try:
        assert (
            next(event for event in events if event.kind in _TERMINAL).kind
            is StandardEventKind.FINISHED
        )
        assert _nexus_rows(target) == (1,)
    finally:
        assert (
            executor.close(identity).cleanup_status is CleanupStatus.CLEANED
        )


def test_post_g2_unsafe_unfunded_staging_accepts_exact_eiger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog,
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    master = tmp_path / "unsafe_master.h5"
    member = tmp_path / "unsafe_data_000001.h5"
    target = tmp_path / "unsafe.nexus"
    poni = tmp_path / "unsafe.poni"
    _write_eiger(master, member, 1)
    write_poni(poni)
    intent = _intent(
        master, target, poni,
        processing_mode="Int 2D",
    )
    intent.max_cores = 4
    intent.run_options["heavy_window"] = 64
    intent.run_options["_post_g2_pipeline_v2"] = {
        "writer_settlement_batch_size": 1,
        "nexus_record_batch_size": 8,
        "reduction_inflight": 8,
        "semantic_checkpoint_frame_cap": 10_000,
        "staging_frame_cap": 10_008,
    }
    marker = {
        "mode": "UNSAFE_UNFUNDED",
        "checkpoint": 10_000,
        "staging_frame_cap": 10_008,
        "max_frames": 3_621,
    }
    intent.run_options[
        "_post_g2_unfunded_staging_diagnostic_v1"
    ] = marker
    resolve = dynamic_output.resolve_session_policy
    open_session = dynamic_output.open_headless_scan_session
    observed = []

    def fixed_envelope(requirements, **kwargs):
        kwargs["envelope_bytes"] = 64 * 1024 ** 3
        return resolve(requirements, **kwargs)

    def capture_session(*args, **kwargs):
        session = open_session(*args, **kwargs)
        observed.append((
            session._policy.allocation.staging_items,
            session._policy.flush.cap,
            session._dynamic_nexus_checkpoint_threshold,
        ))
        return session

    monkeypatch.setenv("XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC", "1")
    monkeypatch.setattr(
        dynamic_output, "total_physical_ram_bytes", lambda: 128 * 1024 ** 3,
    )
    monkeypatch.setattr(
        dynamic_output, "resolve_session_policy", fixed_envelope,
    )
    monkeypatch.setattr(
        dynamic_output, "open_headless_scan_session", capture_session,
    )
    caplog.set_level(logging.INFO, logger=dynamic_output.__name__)
    executor, identity, events = _run_to_terminal(
        intent, request_value=1713,
    )
    try:
        assert (
            next(event for event in events if event.kind in _TERMINAL).kind
            is StandardEventKind.FINISHED
        )
        assert observed == [(64, 10_008, 10_000)]
        with h5py.File(target, "r") as handle:
            raw = handle["entry/reduction/config/run_configuration"].asstr()[()]
            configuration = json.loads(raw)
        assert configuration["run_options"][
            "_post_g2_unfunded_staging_diagnostic_v1"
        ] == marker
        assert configuration[
            "unsafe_unfunded_staging_diagnostic"
        ]["mode"] == "UNSAFE_UNFUNDED"
        facts = tuple(record.getMessage() for record in caplog.records)
        assert any(
            "[RUN-STAGING] mode=UNSAFE_UNFUNDED" in fact
            and "requested-staging=10008 funded-staging=64" in fact
            for fact in facts
        )
    finally:
        assert (
            executor.close(identity).cleanup_status is CleanupStatus.CLEANED
        )


def test_post_g2_output_diagnostics_disable_only_xye_and_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog,
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.io import output_transaction

    raw = tmp_path / "diagnostics_0001.tif"
    target = tmp_path / "diagnostics.nexus"
    poni = tmp_path / "diagnostics.poni"
    _write_tiff(raw, 7)
    write_poni(poni)
    intent = _intent(raw, target, poni)
    intent.max_cores = 4
    intent.run_options["_post_g2_pipeline_v2"] = {
        "writer_settlement_batch_size": 1,
        "nexus_record_batch_size": 8,
        "reduction_inflight": 16,
        "semantic_checkpoint_frame_cap": 56,
    }
    intent.run_options["_post_g2_output_diagnostics_v1"] = {
        "save_xye": False,
        "durable_fsync": False,
    }
    resolve = dynamic_output.resolve_session_policy
    open_session = dynamic_output.open_headless_scan_session
    observed = []

    def fixed_envelope(requirements, **kwargs):
        kwargs["envelope_bytes"] = 64 * 1024 ** 3
        return resolve(requirements, **kwargs)

    def capture_session(*args, **kwargs):
        sink = kwargs["sink"]
        children = getattr(sink, "output_sink_children", (sink,))
        (nexus,) = tuple(children)
        session = open_session(*args, **kwargs)
        observed.append((
            tuple(type(child).__name__ for child in children),
            nexus.durable_fsync,
            nexus,
        ))
        return session

    fsync_calls = []
    monkeypatch.setattr(
        dynamic_output, "resolve_session_policy", fixed_envelope,
    )
    monkeypatch.setattr(
        dynamic_output, "open_headless_scan_session", capture_session,
    )
    monkeypatch.setattr(output_transaction.os, "fsync", fsync_calls.append)
    caplog.set_level(logging.INFO, logger=dynamic_output.__name__)
    executor, identity, events = _run_to_terminal(
        intent, request_value=1711,
    )
    try:
        terminal = next(event for event in events if event.kind in _TERMINAL)
        assert terminal.kind is StandardEventKind.FINISHED, terminal.detail
        assert _nexus_rows(target) == (1,)
        assert tuple(tmp_path.rglob("*.xye")) == ()
        assert len(observed) == 1
        child_names, durable_fsync, nexus = observed[0]
        assert child_names == ("NexusSink",)
        assert durable_fsync is False
        assert nexus._transaction._durable_fsync is False
        assert fsync_calls == []
        assert [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith("[RUN-OUTPUT-DIAGNOSTICS]")
        ] == [
            "[RUN-OUTPUT-DIAGNOSTICS] save-xye=off durable-fsync=off "
            "durability=UNSAFE_SIMULATED"
        ]
    finally:
        executor.close(identity)


def test_p1b_b18_headless_and_single_owner_census(tmp_path: Path) -> None:
    """B18: exact Scan identity survives one Qt-free public facade."""

    import xdart.modules.reduction as reduction_facade
    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import Integration1DPlan, MemorySink, ReductionPlan

    loads: list[int] = []

    def loader(frame: ScanFrame) -> np.ndarray:
        loads.append(int(frame.index))
        return np.ones((2, 2), dtype=float)

    frame7 = ScanFrame(
        7,
        metadata={"identity": "seven"},
        source_path=tmp_path / "seven.tif",
        source_frame_index=3,
        source_identity="p1b/exact-seven",
        loader=loader,
    )
    frame9 = ScanFrame(
        9,
        metadata={"identity": "nine"},
        source_path=tmp_path / "nine.tif",
        source_frame_index=5,
        source_identity="p1b/exact-nine",
        loader=loader,
    )
    scan = Scan(
        "exact-p1b",
        [frame7, frame9],
        integrator=object(),
        motors={"theta": np.asarray([0.7, 0.9])},
        geometry=object(),
        extra={"identity": "must-survive"},
    )
    facade = getattr(reduction_facade, "open_headless_scan_session", None)
    assert callable(facade), "exact-Scan public facade is absent"
    session = facade(
        scan,
        ReductionPlan(
            integration_1d=Integration1DPlan(npt=8),
            integration_2d=None,
        ),
        sink=MemorySink(),
        executor=1,
    )
    assert session.scan is scan
    assert session.scan.frames[0] is frame7
    assert session.scan.frames[1] is frame9
    assert session.scan.frames[0].metadata is frame7.metadata
    assert session.scan.frames[1].metadata is frame9.metadata
    assert session.scan.motors["theta"] is scan.motors["theta"]
    assert session.scan.geometry is scan.geometry
    assert session.scan.extra is scan.extra
    assert loads == []
    session.stop()
    session.finish(raise_on_failure=False)
    assert loads == []

    # Probe a fresh interpreter so a prior Qt import cannot hide transitive
    # coupling in the exact public facade module.
    code = """
import sys
import xdart.modules.reduction as reduction
assert callable(getattr(reduction, 'open_headless_scan_session', None))
for name in tuple(sys.modules):
    assert not (name == 'qtpy' or name.startswith(('qtpy.', 'PyQt', 'PySide', 'pyqtgraph'))), name
"""
    env = dict(os.environ)
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + prior if prior else ""
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert _constructor_sites("ScanSession") == set()
    assert _constructor_sites("TargetLease") == set()
    assert _constructor_sites("TargetLease.acquire") == set()
    assert _constructor_sites("XYESink") == set()
    assert _constructor_sites("DynamicRunAccounting") == {
        "src/xdart/gui/tabs/scattering/adapters/dynamic_output.py"
    }
