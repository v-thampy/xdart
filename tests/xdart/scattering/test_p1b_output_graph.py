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


def _written(target: Path, processing_mode: str = "Int 1D") -> Path:
    """Thin alias for the SHARED spelling; see `_output_slots.written`."""
    from tests.xdart.scattering._output_slots import written

    return written(target, processing_mode)


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
    # The ARTIFACT, not the anchor.  Fable F3 on `4fe073e8`: left as `target`
    # this assertion failed FIRST and masked b02's own pre-existing cause
    # (`activation requires one prepared admission`), quietly giving a test
    # already labelled pre-existing a second, range-introduced reason to be red.
    assert _written(target).is_file()
    xye_files = tuple((tmp_path / "native").glob("*.xye"))
    assert [path.name for path in xye_files] == ["iq_native_0001.xye"]
    executor.close(identity)

    # A large container is fully discovered/revision-pinned at construction,
    # but attempts are minted only at each submit boundary. The named lineage
    # ceiling, rather than reduction in-flight, bounds the unsettled epoch.
    import xdart.gui.tabs.scattering.adapters.dynamic_output as dynamic_output
    from xdart.gui.tabs.scattering.output_preflight import native_int_reduction_plan
    from xrd_tools.integrate.calibration import poni_to_integrator
    from xrd_tools.session.frame_record_store import FrameRecordStore
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
    plan = replace(
        native_int_reduction_plan(large_configuration),
        mask=large_admission.scientific_assets.mask,
    )

    adapter = dynamic_output.DynamicOutputAdapter(large_configuration)
    try:
        from xdart.gui.tabs.scattering.adapters.run_executor import (
            _qualify_background_bindings,
        )
        stop_signal = Event()
        preparation = adapter.prepare_admission(
            scan, plan, large_decision.item, large_decision, stop_signal,
            qualify=lambda policy, prior: _qualify_background_bindings(
                large_configuration, scan, large_decision.item,
                large_decision, policy, prior, stop_signal,
            ),
        )
        # The direct headless facade validates the admitted allocation against
        # an actual native frame, as the coordinated GUI constructor does.
        scan.frames[0].load_image()
        session, created = adapter.activate(
            preparation,
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
            graph=replace(large_decision.item.graph, stamp=oversized_stamp),
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


def test_persisted_catalog_seed_requires_exact_monotonic_work_ordinal() -> None:
    from xdart.gui.tabs.scattering.display_catalog import DisplayCatalogIndex

    catalog = DisplayCatalogIndex(
        RunIdentity(1, "persisted-ordinal-contract"), max_items=2,
    )
    for arguments in (
        (None, "/out/a.nxs", 1, 10),
        ("", "/out/a.nxs", 1, 10),
        ("scan", None, 1, 10),
        ("scan", "", 1, 10),
        ("scan", "/out/a.nxs", True, 10),
        ("scan", "/out/a.nxs", 1.0, 10),
    ):
        with pytest.raises(TypeError):
            catalog.seed_at_work_ordinal(*arguments)
        assert catalog.entries == ()
    for invalid, error in (
        (True, TypeError),
        (1.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ):
        with pytest.raises(error):
            catalog.seed_at_work_ordinal("scan", "/out/a.nxs", 1, invalid)
        assert catalog.entries == ()

    first = catalog.seed_at_work_ordinal(
        "scan", "/out/a.nxs", 1, 10,
    ).appended
    reused = catalog.seed_at_work_ordinal(
        "scan", "/out/a.nxs", 1, 10,
    )
    assert reused.appended is first
    assert reused.retired == ()
    prior = catalog.entries
    with pytest.raises(ValueError, match="source scan conflicts"):
        catalog.seed_at_work_ordinal("foreign", "/out/a.nxs", 1, 10)
    with pytest.raises(ValueError, match="conflicts"):
        catalog.seed_at_work_ordinal("scan", "/out/a.nxs", 1, 11)
    with pytest.raises(ValueError, match="advance high-water"):
        catalog.seed_at_work_ordinal("scan", "/out/a.nxs", 2, 9)
    assert catalog.entries == prior
    assert catalog.resolve_exact("/out/a.nxs", 1) is first

    second = catalog.seed_at_work_ordinal(
        "scan", "/out/a.nxs", 2, 12,
    ).appended
    seeded_third = catalog.seed_at_work_ordinal(
        "scan", "/out/a.nxs", 3, 13,
    )
    assert seeded_third.retired == (first,)
    assert catalog.resolve(first) is None
    live = catalog.append("scan", "/out/a.nxs", 4).appended
    assert second.work_ordinal == 12
    assert seeded_third.appended.work_ordinal == 13
    assert live.work_ordinal == 14
    assert catalog.entries == (seeded_third.appended, live)

    label_contract = DisplayCatalogIndex(
        RunIdentity(1, "persisted-label-contract"), max_items=1,
    )
    negative = label_contract.seed_at_work_ordinal(
        "scan", "/out/negative.nxs", -1, 1,
    ).appended
    assert negative.local_frame_label == -1


def test_persisted_catalog_prefix_preflight_is_atomic() -> None:
    from xdart.gui.tabs.scattering.display_catalog import DisplayCatalogIndex

    catalog = DisplayCatalogIndex(
        RunIdentity(1, "persisted-prefix-atomic"), max_items=2,
    )
    catalog.seed_many_at_work_ordinals((
        ("scan", "/out/a.nxs", 1, 10),
        ("scan", "/out/a.nxs", 2, 11),
    ))

    def exact_state():
        return (
            catalog.entries,
            catalog._work_ordinal,
            tuple((key, id(value)) for key, value in catalog._exact.items()),
            tuple(
                (key, id(value))
                for key, value in catalog._by_value.items()
            ),
        )

    prior = exact_state()
    cases = (
        (
            (
                ("scan", "/out/a.nxs", 3, 12),
                ("scan", "/out/a.nxs", 4, 11),
            ),
            "advance high-water",
        ),
        (
            (
                ("scan", "/out/a.nxs", 3, 12),
                ("scan", "/out/a.nxs", 3, 12),
            ),
            "duplicate",
        ),
    )
    for rows, message in cases:
        with pytest.raises(ValueError, match=message):
            catalog.seed_many_at_work_ordinals(rows)
        assert exact_state() == prior

    with pytest.raises(TypeError, match="exact integer"):
        catalog.seed_many_at_work_ordinals((
            ("scan", "/out/a.nxs", 3, 12),
            ("scan", "/out/a.nxs", True, 13),
        ))
    assert exact_state() == prior


@pytest.mark.parametrize(
    ("capacity", "expected_labels", "expected_ordinals"),
    (
        (3, (103, 104, 105), (10, 11, 12)),
        (5, (101, 102, 103, 104, 105), (8, 9, 10, 11, 12)),
        (8, (101, 102, 103, 104, 105), (8, 9, 10, 11, 12)),
    ),
    ids=("shorter", "equal", "larger"),
)
def test_executor_persisted_prefix_seeds_absolute_run_ordinals(
    capacity: int,
    expected_labels: tuple[int, ...],
    expected_ordinals: tuple[int, ...],
) -> None:
    from xdart.gui.tabs.scattering.display_runtime import RunDisplayState

    display = RunDisplayState(
        RunIdentity(1, f"persisted-prefix-{capacity}"),
        max_payload_items=1,
        catalog_max_items=capacity,
    )
    artifact = "/processed/series.nexus"
    owner = SimpleNamespace(source_scan="series", artifact=Path(artifact))
    run = SimpleNamespace(
        display=display,
        completed=7,
        current_completed=5,
    )
    labels = (101, 102, 103, 104, 105)
    StandardRunExecutor._seed_persisted_prefix_navigation(
        run, owner, labels, newly_adopted=True,
    )
    entries = display.catalog_snapshot().entries
    assert tuple(key.local_frame_label for key in entries) == expected_labels
    assert tuple(key.work_ordinal for key in entries) == expected_ordinals

    prior = entries
    run.current_completed = 4
    with pytest.raises(ValueError, match="cardinality"):
        StandardRunExecutor._seed_persisted_prefix_navigation(
            run, owner, labels, newly_adopted=True,
        )
    assert display.catalog_snapshot().entries == prior

    appended = display.append_navigation(
        owner.source_scan, artifact, 106,
    ).appended
    assert appended.work_ordinal == 13


@pytest.mark.parametrize(
    "same_run_labels",
    ((), (101, 102, 103, 104, 105)),
    ids=("overwrite-empty-prefix", "append-represented-prefix"),
)
def test_executor_prefix_seed_distinguishes_reused_and_new_owner(
    same_run_labels: tuple[int, ...],
) -> None:
    from xdart.gui.tabs.scattering.display_runtime import RunDisplayState

    artifact = "/processed/series.nexus"
    owner = SimpleNamespace(source_scan="series", artifact=Path(artifact))
    reused = RunDisplayState(
        RunIdentity(1, f"same-run-{len(same_run_labels)}"),
        max_payload_items=1,
        catalog_max_items=3,
    )
    reused.seed_navigation_prefix_at_work_ordinals((
        (owner.source_scan, artifact, 8, 8),
        (owner.source_scan, artifact, 9, 9),
        (owner.source_scan, artifact, 10, 10),
    ))
    prior = reused.catalog_snapshot().entries
    same_run = SimpleNamespace(
        display=reused,
        completed=10,
        current_completed=5,
    )
    StandardRunExecutor._seed_persisted_prefix_navigation(
        same_run,
        owner,
        same_run_labels,
        newly_adopted=False,
    )
    assert reused.catalog_snapshot().entries == prior
    assert reused.append_navigation(
        owner.source_scan, artifact, 11,
    ).appended.work_ordinal == 11

    newly_adopted = RunDisplayState(
        RunIdentity(1, f"cross-run-{len(same_run_labels)}"),
        max_payload_items=1,
        catalog_max_items=3,
    )
    dormant_noop = SimpleNamespace(
        display=newly_adopted,
        completed=7,
        current_completed=5,
    )
    StandardRunExecutor._seed_persisted_prefix_navigation(
        dormant_noop,
        owner,
        (101, 102, 103, 104, 105),
        newly_adopted=True,
    )
    entries = newly_adopted.catalog_snapshot().entries
    assert tuple(key.local_frame_label for key in entries) == (103, 104, 105)
    assert tuple(key.work_ordinal for key in entries) == (10, 11, 12)


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
    real_output_init = real_output_adapter.__init__

    def observed_output_init(owner, configuration):
        real_output_init(owner, configuration)
        output_adapters.append(owner)

    monkeypatch.setattr(
        real_output_adapter, "__init__", observed_output_init,
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
    assert _nexus_rows(_written(target)) == (1, 2)
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
    assert _nexus_rows(_written(target)) == (1, 2, 3), "B03 missing"
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
    before = _written(target).read_bytes()
    before_stat = _written(target).stat()
    before_xye = xye_facts(tuple(sorted(xye_directory.glob("*.xye"))))
    noop_executor, noop_identity, noop_events = _run_to_terminal(
        _intent(raw1, target, poni, output_mode="Append"),
        request_value=1303,
    )
    assert next(event for event in noop_events if event.kind in _TERMINAL).kind \
        is StandardEventKind.FINISHED, "B03 no-op"
    after_stat = _written(target).stat()
    assert reads == [], "B03 no-op"
    assert _written(target).read_bytes() == before, "B03 no-op"
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
    before = _written(target).read_bytes()
    before_stat = _written(target).stat()
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
        after_stat = _written(target).stat()
        assert _written(target).read_bytes() == before, "B03 PONI drift"
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
    before = _written(target).read_bytes()
    before_stat = _written(target).stat()
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
        after_stat = _written(target).stat()
        assert _written(target).read_bytes() == before, "B03 mismatch"
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
    run = SimpleNamespace(
        display=display,
        completed=0,
        current_completed=1_000_000,
    )
    display.seed_navigation_at_work_ordinal(
        owner.source_scan, artifact, 999_998, 999_998,
    )

    calls = []
    real_seed = display.seed_navigation_prefix_at_work_ordinals

    def observed_seed(rows):
        calls.append(rows)
        return real_seed(rows)

    monkeypatch.setattr(
        display, "seed_navigation_prefix_at_work_ordinals", observed_seed,
    )
    StandardRunExecutor._seed_persisted_prefix_navigation(
        run,
        owner,
        tuple(range(1, 1_000_001)),
        newly_adopted=True,
    )

    assert calls == [
        (
            ("series", artifact, 999_998, 999_998),
            ("series", artifact, 999_999, 999_999),
            ("series", artifact, 1_000_000, 1_000_000),
        ),
    ]
    entries = display.catalog_snapshot().entries
    assert tuple(key.local_frame_label for key in entries) == (
        999_998, 999_999, 1_000_000,
    )
    assert tuple(key.work_ordinal for key in entries) == (
        999_998, 999_999, 1_000_000,
    )
    appended = display.append_navigation(
        owner.source_scan, artifact, 1_000_001,
    ).appended
    assert appended.work_ordinal == 1_000_001


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
                # XYE has no NeXus record writer to mark a FrameRecordStore.
                # Require the exact current attempt's receipts on every
                # applicable XYE target, after the physical files settled.
                assert run.output is not None
                for event in frames[:expected]:
                    assert event.frame_key is not None
                    graph = run.output._graphs[event.frame_key.artifact]
                    accounting = graph["accounting"]
                    key = graph["keys"][event.frame_key.local_frame_label]
                    receipts = {
                        (key, mode, target)
                        for mode, targets in accounting.ledger.targets_by_mode.items()
                        for target in targets
                    }
                    assert receipts
                    assert receipts.issubset(accounting.snapshot().durable)
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


def test_p1b_b17_collision_custody_and_xye_append_refuse_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B17: real H23 collision/custody and typed XYE Append refusal."""

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
    # The run wrote its SLOT, not the requested name. Custody -- leases and
    # byte/inode fingerprints alike -- is about the artifact that exists.
    custody_written = _written(custody_target)
    custody_before = (
        custody_written.read_bytes(),
        custody_written.stat().st_dev,
        custody_written.stat().st_ino,
        custody_written.stat().st_mtime_ns,
    )

    def assert_custody_target_reacquirable(label: str) -> None:
        transaction_authority = OwnerToken(f"{label}-transaction")
        target_authority = OwnerToken(f"{label}-target")
        admitted = coordinator.admit(
            custody_written,
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
        custody_written.read_bytes(),
        custody_written.stat().st_dev,
        custody_written.stat().st_ino,
        custody_written.stat().st_mtime_ns,
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
            Path(owner.path).resolve() == custody_written.resolve()
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
        custody_written.read_bytes(),
        custody_written.stat().st_dev,
        custody_written.stat().st_ino,
        custody_written.stat().st_mtime_ns,
    ) == custody_before
    assert_custody_target_reacquirable("p1b-later-begin-reacquire")

    # Durable display projection is acknowledged independently per graph and
    # only after its callback succeeds. A partial callback failure must retain
    # B without replaying already-applied A on cleanup re-entry.
    from xrd_tools.session import (
        DynamicFrameIdentity,
        DynamicRunState,
        ResultMode,
    )

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
                state=(
                    DynamicRunState.FINISHED
                    if self.settled[0]
                    else DynamicRunState.ACTIVE
                ),
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
    owner_a = cleanup_run.display.add_artifact(
        projection_a,
        "projection-a",
        mask=None,
        mask_saturation=False,
        measurement_mode="Standard",
    )
    owner_b = cleanup_run.display.add_artifact(
        projection_b,
        "projection-b",
        mask=None,
        mask_saturation=False,
        measurement_mode="Standard",
    )
    cleanup_graph_a["display_owner"] = owner_a
    cleanup_graph_b["display_owner"] = owner_b
    cleanup_executor = StandardRunExecutor()
    original_failure = detach_exception(
        RuntimeError("injected original run failure"), "run",
    )
    assert cleanup_adapter.finalized_display_owners() == ()
    assert (owner_a.hydration_closed, owner_b.hydration_closed) == (
        False,
        False,
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
    assert cleanup_adapter.finalized_display_owners() == (owner_a,)
    assert (owner_a.hydration_closed, owner_b.hydration_closed) == (
        True,
        False,
    )
    assert cleanup_graph_a["transition"] is None
    assert cleanup_graph_b["transition"] == "finish"

    second_cleanup = cleanup_executor._cleanup(cleanup_run)
    stable_cleanup = (
        cleanup_run.completed,
        tuple(cleanup_run.artifacts),
        tuple(cleanup_graph_a["settled_labels"]),
        tuple(cleanup_graph_b["settled_labels"]),
        tuple(cleanup_adapter.projections),
        owner_a.hydration_closed,
        owner_b.hydration_closed,
        cleanup_adapter.finalized_display_owners(),
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
    assert cleanup_adapter.finalized_display_owners() == (
        owner_a,
        owner_b,
    )
    assert (owner_a.hydration_closed, owner_b.hydration_closed) == (
        True,
        True,
    )
    assert cleanup_graph_a["transition"] is None
    assert cleanup_graph_b["transition"] is None
    assert third_cleanup.cleanup_status is CleanupStatus.CLEANED
    assert stable_cleanup == (
        cleanup_run.completed,
        tuple(cleanup_run.artifacts),
        tuple(cleanup_graph_a["settled_labels"]),
        tuple(cleanup_graph_b["settled_labels"]),
        tuple(cleanup_adapter.projections),
        owner_a.hydration_closed,
        owner_b.hydration_closed,
        cleanup_adapter.finalized_display_owners(),
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
    # Both contenders REQUEST the anchor and both resolve the one `Int 2D`
    # slot, so the artifact -- not the anchor -- is what the writer holds
    # and what H23 leases. Hold and expect the file that is really there.
    collision_written = _written(shared_target, "Int 2D")
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
            Path(owner.path).resolve() == collision_written.resolve()
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
        normalized_target = os.path.normcase(
            os.path.abspath(collision_written)
        )
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
        (None, (1, 8, 16, 16, 1, 4, 4)),
        ((1, 8, 16, 56, 64), (1, 16, 56, 56, 1, 4, 4)),
        ((1, 8, 8, 64, 72), (1, 8, 64, 64, 1, 4, 4)),
    ),
    ids=(
        "default", "v2-settlement1-record8-inflight16-checkpoint56",
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
        intent.run_options["_post_g2_pipeline_v2"] = {
            "writer_settlement_batch_size": pipeline[0],
            "nexus_record_batch_size": pipeline[1],
            "reduction_inflight": pipeline[2],
            "semantic_checkpoint_frame_cap": pipeline[3],
            "staging_frame_cap": pipeline[4],
        }
    expected_pipeline = (
        (1, 8, 8, 16, 64) if pipeline is None else pipeline
    )
    resolve = dynamic_output.resolve_session_policy
    open_session = dynamic_output.open_headless_scan_session
    observed = []
    checkpoint_observed = []
    v2_observed = []

    def fixed_envelope(requirements, **kwargs):
        kwargs["envelope_bytes"] = 64 * 1024 ** 3
        return resolve(requirements, **kwargs)

    def capture_session(*args, **kwargs):
        session = open_session(*args, **kwargs)
        checkpoint_observed.append((
            kwargs["dynamic_nexus_checkpoint"],
            kwargs["dynamic_nexus_checkpoint_threshold"],
            session._dynamic_nexus_checkpoint_threshold,
            session._policy.flush.hard_threshold(),
        ))
        observed.append((
            kwargs["sink"].writer_batch_size, kwargs["inflight_max"],
            kwargs["dynamic_nexus_checkpoint_threshold"],
            session._dynamic_nexus_checkpoint_threshold,
            session._session._writer_batch_size,
            kwargs["executor"], session._session._worker._max_workers,
        ))
        if len(expected_pipeline) == 5:
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

    # The worker count pinned below is the Cores=4 request itself.  The pool
    # cap also clamps to the host (``min(cores, cpu_count)``, 2 below 16 GiB
    # RAM), so fix the host the way ``fixed_envelope`` fixes the envelope:
    # a 16 GiB / 4 vCPU CI runner otherwise reports 2 workers.
    from xrd_tools.core import staging
    monkeypatch.setattr(staging, "total_physical_ram_bytes", lambda: 64 * 1024 ** 3)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
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
        assert _nexus_rows(_written(target)) == (1,)
        assert observed == [expected]
        assert checkpoint_observed == [(
            True,
            expected_pipeline[3],
            expected_pipeline[3],
            expected_pipeline[4] - 8,
        )]
        assert v2_observed == ([] if len(expected_pipeline) == 3 else [
            (
                expected_pipeline[0],
                expected_pipeline[2],
                expected_pipeline[2],
                4,
                4,
                expected_pipeline[1],
                ("TransactionalXYESink", "NexusSink"),
            ),
        ])
        facts = [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith("[RUN-PIPELINE]")
        ]
        expected_facts = (
            [] if len(expected_pipeline) in (4, 5) else [
                f"[RUN-PIPELINE] requested-batch={expected_pipeline[0]} "
                f"effective-batch={expected_pipeline[0]} "
                f"requested-inflight={expected_pipeline[1]} "
                f"effective-inflight={expected_pipeline[1]} "
                f"requested-checkpoint={expected_pipeline[2]} "
                f"effective-checkpoint={expected_pipeline[2]}"
            ]
        )
        assert facts == expected_facts
        v2_facts = [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith("[RUN-PIPELINE-V2]")
        ]
        if len(expected_pipeline) == 3:
            assert v2_facts == []
        else:
            assert len(v2_facts) == 1
            assert (
                f"requested-inflight={expected_pipeline[2]} "
                f"effective-inflight={expected_pipeline[2]}"
            ) in v2_facts[0]
            assert (
                f"requested-checkpoint={expected_pipeline[3]} "
                f"effective-checkpoint={expected_pipeline[3]}"
            ) in v2_facts[0]
            staging = (
                expected_pipeline[4]
                if len(expected_pipeline) == 5 else 64
            )
            assert (
                f"requested-staging={staging} effective-staging={staging}"
            ) in v2_facts[0]
    finally:
        executor.close(identity)


def test_live_gui_checkpoint_uses_the_exact_policy_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw = raw_root / "live_checkpoint_0001.tif"
    poni = tmp_path / "live_checkpoint.poni"
    _write_tiff(raw, 7)
    write_poni(poni)
    intent = _live_directory_intent(
        raw_root,
        tmp_path / "processed",
        poni,
        processing_mode="Int 1D",
    )
    resolve = dynamic_output.resolve_session_policy
    open_session = dynamic_output.open_headless_scan_session
    observed = []

    def fixed_envelope(requirements, **kwargs):
        kwargs["envelope_bytes"] = 64 * 1024 ** 3
        return resolve(requirements, **kwargs)

    def capture_session(*args, **kwargs):
        session = open_session(*args, **kwargs)
        observed.append((
            kwargs["dynamic_nexus_checkpoint"],
            kwargs["dynamic_nexus_checkpoint_threshold"],
            session._dynamic_nexus_checkpoint_threshold,
            session._policy.flush.hard_threshold(),
        ))
        return session

    monkeypatch.setattr(
        dynamic_output, "resolve_session_policy", fixed_envelope,
    )
    monkeypatch.setattr(
        dynamic_output, "open_headless_scan_session", capture_session,
    )
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(executor, intent, request_value=1715)
    try:
        _drain_until(
            executor,
            lambda values: any(
                event.kind is StandardEventKind.FRAME_READY
                for event in values
            ),
        )
        assert observed == [(True, None, 56, 56)]
    finally:
        executor.stop(identity)
        terminal_events = _drain_until(
            executor,
            lambda values: any(event.kind in _TERMINAL for event in values),
        )
        assert next(
            event for event in terminal_events if event.kind in _TERMINAL
        ).kind is StandardEventKind.STOPPED
        assert (
            executor.close(identity).cleanup_status
            is CleanupStatus.CLEANED
        )


def test_live_gui_checkpoint_crosses_policy_then_stop_commits_exact_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.reduction import NexusTerminalDisposition
    from xrd_tools.session import FrameRecordStore
    from xrd_tools.session.scan_session import ScanSession

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    poni = tmp_path / "bounded_live.poni"
    for label in range(1, 9):
        _write_tiff(raw_root / f"bounded_live_{label:04d}.tif", label)
    write_poni(poni)
    intent = _live_directory_intent(
        raw_root,
        tmp_path / "processed",
        poni,
        processing_mode="Int 1D",
    )
    intent.run_options["heavy_window"] = 16
    open_session = dynamic_output.open_headless_scan_session
    sessions = []
    stores = []
    checkpoints = []
    clear_calls = []
    epoch_entered = Event()
    release_epoch = Event()
    commit_epoch = ScanSession.commit_epoch
    clear_checkpoint_recoverable = (
        FrameRecordStore.clear_checkpoint_recoverable
    )

    def observed_clear(owner):
        clear_calls.append((
            owner,
            tuple(
                owner.checkpoint_recoverable_modes(label)
                for label in range(1, 9)
            ),
        ))
        return clear_checkpoint_recoverable(owner)

    def capture_session(*args, **kwargs):
        session = open_session(*args, **kwargs)
        session.on_checkpoint_recoverable(checkpoints.append)
        sessions.append(session)
        stores.append(kwargs["record_store"])
        return session

    def held_epoch(owner):
        # Observe checkpoint custody before the independent live-epoch
        # terminal transition can close the writer and consume its receipts.
        epoch_entered.set()
        assert release_epoch.wait(10.0)
        return commit_epoch(owner)

    monkeypatch.setattr(
        dynamic_output, "open_headless_scan_session", capture_session,
    )
    monkeypatch.setattr(
        FrameRecordStore,
        "clear_checkpoint_recoverable",
        observed_clear,
    )
    monkeypatch.setattr(ScanSession, "commit_epoch", held_epoch)
    executor = StandardRunExecutor(join_timeout=2.0)
    identity = _start(executor, intent, request_value=1716)
    try:
        _drain_until(
            executor,
            lambda values: (
                sum(
                    event.kind is StandardEventKind.FRAME_READY
                    for event in values
                ) >= 8
                and bool(checkpoints)
                and epoch_entered.is_set()
            ),
        )
        assert len(sessions) == len(stores) == 1
        session, store = sessions[0], stores[0]
        expected_rows = tuple(range(1, 9))
        assert tuple(session.scan.frame_indices) == expected_rows
        assert session._dynamic_nexus_checkpoint_threshold == 8
        assert session._dynamic_nexus_checkpoint_count == 0
        assert checkpoints[0].labels == expected_rows
        target = session._dynamic_nexus_sink.path
        # The checkpoint still belongs to the active writer. Read its exact
        # HDF5 handle under the writer lock; an independent file open races
        # that exclusive custody. The post-Stop read below uses a fresh open.
        writer = session._dynamic_nexus_sink._writer
        with writer._boundary():
            assert tuple(writer._h5["entry/integrated_1d/frame_index"][()]) == expected_rows

        staged = session._dynamic_accounting.snapshot()
        assert len(staged.pending_durable) == 8
        assert staged.durable == frozenset()
        expected_modes = frozenset({("1d", "default")})
        assert all(
            store.checkpoint_recoverable_modes(label) == expected_modes
            for label in expected_rows
        )

        executor.stop(identity)
        release_epoch.set()
        terminal_events = _drain_until(
            executor,
            lambda values: any(event.kind in _TERMINAL for event in values),
        )
        assert next(
            event for event in terminal_events if event.kind in _TERMINAL
        ).kind is StandardEventKind.STOPPED
        assert (
            session.terminal_result.disposition
            is NexusTerminalDisposition.COMMITTED
        )
        settled = session._dynamic_accounting.snapshot()
        assert settled.state.value == "stopped"
        assert settled.pending_durable == frozenset()
        ledger = session._dynamic_accounting.ledger
        assert settled.durable == frozenset(
            (key, mode, target_name)
            for key in settled.discovered
            for mode in ledger.required_modes
            for target_name in ledger.targets_by_mode[mode]
        )
        assert _nexus_rows(target) == expected_rows
        assert clear_calls == [
            (store, tuple(expected_modes for _label in expected_rows)),
            (store, tuple(frozenset() for _label in expected_rows)),
        ]
        assert all(
            store.checkpoint_recoverable_modes(label) == frozenset()
            for label in expected_rows
        )
    finally:
        release_epoch.set()
        assert (
            executor.close(identity).cleanup_status
            is CleanupStatus.CLEANED
        )


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
    assert not _written(target).exists()
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
    assert not _written(target).exists()
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
        assert _nexus_rows(_written(target)) == (1,)
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
        with h5py.File(_written(target, "Int 2D"), "r") as handle:
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


@pytest.mark.parametrize("save_xye", (False, True))
@pytest.mark.parametrize("durable_fsync", (False, True))
def test_post_g2_output_diagnostics_disable_only_xye_and_fsync(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog,
    save_xye: bool,
    durable_fsync: bool,
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.io import output_transaction
    from xrd_tools.reduction import NexusSink

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
        "staging_frame_cap": 64,
    }
    intent.run_options["_post_g2_output_diagnostics_v1"] = {
        "save_xye": save_xye,
        "durable_fsync": durable_fsync,
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
        (nexus,) = tuple(child for child in children if isinstance(child, NexusSink))
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
        assert _nexus_rows(_written(target)) == (1,)
        assert len(tuple(tmp_path.rglob("*.xye"))) == int(save_xye)
        assert len(observed) == 1
        child_names, observed_fsync, nexus = observed[0]
        assert set(child_names) == (
            {"NexusSink", "TransactionalXYESink"} if save_xye else {"NexusSink"}
        )
        assert observed_fsync is durable_fsync
        assert nexus._transaction._durable_fsync is durable_fsync
        assert bool(fsync_calls) is durable_fsync
        assert [
            record.getMessage() for record in caplog.records
            if record.getMessage().startswith("[RUN-OUTPUT-DIAGNOSTICS]")
        ] == [
            f"[RUN-OUTPUT-DIAGNOSTICS] save-xye={'on' if save_xye else 'off'} "
            f"durable-fsync={'on' if durable_fsync else 'off'} "
            f"durability={'DURABLE' if durable_fsync else 'UNSAFE_SIMULATED'}"
        ]
    finally:
        executor.close(identity)


@pytest.mark.parametrize("verification_recovers", (True, False))
def test_no_xye_terminal_verification_failure_preserves_terminal_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verification_recovers,
) -> None:
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.io.record_writer import NexusRecordWriter

    raw = tmp_path / "verification_0001.tif"
    target = tmp_path / "verification.nexus"
    poni = tmp_path / "verification.poni"
    _write_tiff(raw, 7)
    write_poni(poni)
    original_raw = raw.read_bytes()
    intent = _intent(raw, target, poni)
    intent.run_options["_post_g2_output_diagnostics_v1"] = {
        "save_xye": False, "durable_fsync": True,
    }
    calls = []
    sessions = []
    verify = NexusRecordWriter._verify_fast_integrated_results
    open_session = dynamic_output.open_headless_scan_session

    def fail_once(writer):
        calls.append(writer)
        if not verification_recovers or len(calls) == 1:
            raise OSError("injected terminal science readback failure")
        return verify(writer)

    def capture_session(*args, **kwargs):
        session = open_session(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(NexusRecordWriter, "_verify_fast_integrated_results", fail_once)
    monkeypatch.setattr(dynamic_output, "open_headless_scan_session", capture_session)
    executor, identity, events = _run_to_terminal(intent, request_value=1712)
    try:
        assert calls, "the terminal verification fault must actually execute"
        assert len(sessions) == 1
        terminal = next(event for event in events if event.kind in _TERMINAL)
        assert terminal.kind is StandardEventKind.FAILED, terminal.detail
        assert "injected terminal science readback failure" in str(terminal)
        assert len({id(writer) for writer in calls}) == 1
        snapshot = sessions[0]._dynamic_accounting.snapshot()
        if verification_recovers:
            # Cleanup retries the same writer's real proof; the executor
            # still reports its original failure despite verified publication.
            assert len(calls) >= 2
            assert snapshot.state.value == "finished"
            assert snapshot.durable
            assert _nexus_rows(_written(target)) == (1,)
            assert calls[0]._h5 is None
        else:
            assert not snapshot.durable
            # Same-run streaming owns a visible provisional path. Without
            # successful verification it has no typed terminal commit, and
            # the exact writer handle remains held for cleanup retry.
            assert sessions[0].terminal_result is None
            assert calls[0]._h5 is not None and calls[0]._h5.id.valid
            assert terminal.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert raw.read_bytes() == original_raw
    finally:
        monkeypatch.setattr(NexusRecordWriter, "_verify_fast_integrated_results", verify)
        receipt = executor.close(identity)
        if receipt.cleanup_status is CleanupStatus.CLEANUP_PENDING:
            # Display retirement preceded the successful writer retry; close
            # the same retained display now that its terminal lease settled.
            assert calls[0]._h5 is None
            receipt = executor.close(identity)
        assert receipt.cleanup_status is CleanupStatus.CLEANED
        assert calls[0]._h5 is None


def test_p1b_b18_headless_and_single_owner_census(tmp_path: Path) -> None:
    """B18: exact Scan identity survives one Qt-free public facade."""

    from xrd_tools.core.scan import Scan, ScanFrame
    from xrd_tools.reduction import Integration1DPlan, MemorySink, ReductionPlan
    from xrd_tools.session import open_headless_scan_session

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
    session = open_headless_scan_session(
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
from xrd_tools.session import open_headless_scan_session
assert callable(open_headless_scan_session)
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
