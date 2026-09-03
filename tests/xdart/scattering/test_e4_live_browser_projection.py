from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
from threading import Event, Thread
import time
from types import SimpleNamespace

import numpy as np
from pyqtgraph.Qt import QtCore, QtTest, QtWidgets
import pytest

from xdart.gui.tabs.scattering.adapters.source import (
    FilesystemSourceAdapter,
)
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering import browser_catalog as browser_catalog_module
from xdart.gui.tabs.scattering import browser_view as browser_view_module
from xdart.gui.tabs.scattering.browser_catalog import (
    BrowserCatalogEntry,
    DirectoryModifiedCache,
    enumerate_processed_artifacts,
)
from xdart.gui.tabs.scattering.browser_view import BrowserView
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.processed_browser import (
    browser_suffixes_for_mode,
)
from xdart.gui.tabs.scattering.shell_projection import (
    build_browser_scan_index,
    build_browser_projection,
)
from xdart.gui.tabs.scattering.shell_values import (
    BrowserScanIndex,
    FrameNavigationProjection,
    ShellCommand,
    ShellCommandKind,
)
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xdart.gui.themes import render_qss
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import RunIntent

from tests.xdart.scattering.e3_shell_support import make_shell_projection


def _one_frame_dynamic_admission(tmp_path):
    from xdart.gui.tabs.scattering.contracts import (
        AdmittedOutput, OutputDisposition, OutputFact, PlannedOutput,
    )
    from xrd_tools.core.scan import Scan, ScanFrame, SourceKind
    from xrd_tools.sources.descriptor import ContainerDescriptor
    from xrd_tools.sources.execution_graph import SourceExecutionStamp, SourceFileState
    from xrd_tools.sources.selection import image_series_spec

    source = tmp_path / "frame_0001.tif"
    source.write_bytes(b"raw")
    spec = image_series_spec(source, metadata_format=None)
    state = SourceFileState.capture(source)
    stamp = SourceExecutionStamp(state, "tiff_series", 1, 1, members=(state,))
    descriptor = ContainerDescriptor(
        source, kind=SourceKind.TIFF_SERIES, frame_count=1,
        frame_shape=(4, 4), dtype=np.dtype("uint16"),
    )
    item = PlannedOutput(
        spec, source, tmp_path / "result.nexus", stamp, descriptor=descriptor,
    )
    decision = AdmittedOutput(
        item, OutputDisposition.WRITE, (1,), OutputFact(False),
    )
    return Scan(
        "scan", [ScanFrame(1, source_path=source, source_frame_index=0)],
    ), item, decision


def test_heavy_residency_menu_is_exclusive_and_edits_only_next_intent():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    intents = RunIntentStore(RunIntent())
    lifecycle = ScatteringCoordinator()
    page = ScatteringWorkspace(
        intents=intents, lifecycle=lifecycle, sources=FilesystemSourceAdapter()
    )
    frozen = intents.snapshot().thaw().freeze()
    lifecycle._active_run_identity = RunIdentity.from_configuration(frozen)
    try:
        button = page._shell.browser.findChild(
            QtWidgets.QToolButton, "configMenuButton"
        )
        heavy = next(
            action.menu() for action in button.menu().actions()
            if action.menu() is not None
            and action.menu().title().startswith("Heavy residency")
        )
        choices = {action.text(): action for action in heavy.actions()}
        assert choices["Auto"].isChecked()
        assert sum(action.isChecked() for action in choices.values()) == 1
        revision = intents.revision
        choices["32"].trigger()
        app.processEvents()
        assert intents.revision == revision + 1
        assert intents.snapshot().thaw().run_options == {"heavy_window": 32}
        assert frozen.run_options == {}
        assert choices["32"].isChecked()
        assert "next run" in heavy.title().lower()
    finally:
        lifecycle._active_run_identity = None
        page.close_workspace()
        page.deleteLater()
        app.processEvents()


def test_explicit_heavy_residency_requests_all_three_bounds_without_env_mutation(
    monkeypatch,
):
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    monkeypatch.setenv("XDART_HEAVY_WINDOW", "8")
    before = dict(os.environ)
    observed = {}
    original = dynamic_output.resolve_session_policy

    def capture(requirements, **kwargs):
        observed.update(kwargs)
        return original(requirements, envelope_bytes=8 * 1024 ** 3, **kwargs)

    monkeypatch.setattr(dynamic_output, "resolve_session_policy", capture)
    configuration = RunIntent(
        max_cores=1, run_options={"heavy_window": 64}
    ).freeze()
    descriptor = SimpleNamespace(
        frame_shape=(4, 4), dtype=dynamic_output.np.dtype("uint16")
    )
    item = SimpleNamespace(
        descriptor=descriptor,
        source_stamp=SimpleNamespace(frame_count=1),
    )
    scan = SimpleNamespace(frames=(SimpleNamespace(index=1, image=None),))
    plan = SimpleNamespace(integration_1d=None, integration_2d=None, gi=None)
    policy, *_ = dynamic_output._light_policy_layout(
        configuration, plan, item, scan, (1,), heavy_request=64,
        env=dict(os.environ),
    )
    assert observed["requests"] == {
        "staging_items": 64,
        "record_heavy_items": 64,
        "publication_heavy_items": 64,
    }
    assert policy.allocation.staging_items == 64
    assert dict(os.environ) == before


def test_dynamic_policy_widens_only_standard_four_worker_grants(monkeypatch):
    from xdart.gui.tabs.scattering.adapters import dynamic_output
    from xrd_tools.reduction import (
        Integration1DPlan, Integration2DPlan, ReductionPlan,
    )

    original = dynamic_output.resolve_session_policy
    calls = []
    clamp_inflight = [False]

    def capture(requirements, **kwargs):
        kwargs.setdefault("envelope_bytes", 32 * 1024 ** 3)
        policy = original(requirements, **kwargs)
        if clamp_inflight[0]:
            policy = replace(policy, allocation=replace(
                policy.allocation, counts={
                    **policy.allocation.counts, "reduction_inflight": 15,
                },
            ))
        calls.append((kwargs, policy.allocation))
        return policy

    monkeypatch.setattr(dynamic_output, "resolve_session_policy", capture)
    descriptor = SimpleNamespace(
        frame_shape=(2167, 2070), dtype=dynamic_output.np.dtype("uint32")
    )
    item = SimpleNamespace(
        descriptor=descriptor,
        source_stamp=SimpleNamespace(frame_count=1),
    )
    scan = SimpleNamespace(frames=(SimpleNamespace(index=1, image=None),))
    plan = ReductionPlan(
        integration_1d=Integration1DPlan(npt=1000),
        integration_2d=Integration2DPlan(npt_rad=500, npt_azim=500),
    )
    frozen_env = {"XDART_REDUCTION_WORKERS": "4"}

    standard, *_ = dynamic_output._light_policy_layout(
        RunIntent(max_cores=4).freeze(), plan, item, scan, (1,),
        env=frozen_env,
    )
    assert [(a.workers, a.reduction_inflight) for _, a in calls] == [
        (4, 8), (4, 16),
    ]
    assert calls[1][0]["envelope_bytes"] == calls[0][1].envelope_bytes
    assert calls[0][0]["env"] is calls[1][0]["env"]
    assert {
        name: calls[1][1].categories[name] - calls[0][1].categories[name]
        for name in calls[0][1].categories
    } == {"source_native": 143_542_080, "staging": 0, "records": 0,
          "publication": 0, "worker": 16_192_000}
    assert standard.allocation is calls[1][1]

    calls.clear()
    live, *_ = dynamic_output._light_policy_layout(
        RunIntent(max_cores=4, live_mode=True).freeze(), plan, item, scan, (1,),
        env=frozen_env,
    )
    assert [(a.workers, a.reduction_inflight) for _, a in calls] == [(4, 8)]
    assert live.allocation is calls[0][1]

    calls.clear()
    clamp_inflight[0] = True
    with pytest.raises(RuntimeError, match="not granted exactly"):
        dynamic_output._light_policy_layout(
            RunIntent(max_cores=4).freeze(), plan, item, scan, (1,),
            reduction_inflight=16, env=frozen_env,
        )
    assert len(calls) == 1
    assert calls[0][0]["requests"] == {"reduction_inflight": 16}


def test_post_g2_pipeline_v2_absent_default_is_eligibility_bounded():
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    ineligible = (
        (RunIntent(output_mode="Overwrite", live_mode=True), True),
        (RunIntent(output_mode="Overwrite", batch_mode=True), True),
        (RunIntent(output_mode="Append"), True),
        (RunIntent(
            output_mode="Overwrite", processing_mode="Int 1D (XYE)",
        ), True),
        (RunIntent(output_mode="Overwrite"), False),
    )
    for intent, coordinated in ineligible:
        assert dynamic_output._post_g2_pipeline_v2_choice(
            intent.freeze(), coordinated=coordinated,
        ) is None

    choice = dynamic_output._post_g2_pipeline_v2_choice(
        RunIntent(output_mode="Overwrite").freeze(), coordinated=True,
    )
    assert choice is not None
    assert (
        choice.writer_settlement_batch_size,
        choice.nexus_record_batch_size,
        choice.reduction_inflight,
        choice.semantic_checkpoint_frame_cap,
        choice.staging_frame_cap,
    ) == (1, 8, 8, 16, 64)


def test_post_g2_pipeline_choice_is_named_bounded_and_nonlive():
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    fields = (
        "writer_settlement_batch_size", "nexus_record_batch_size",
        "reduction_inflight", "semantic_checkpoint_frame_cap",
        "staging_frame_cap",
    )
    values = (1, 8, 4, 56, 64)
    complete = dict(zip(fields, values, strict=True))
    choice = dynamic_output._post_g2_pipeline_v2_choice(
        RunIntent(
            output_mode="Overwrite",
            run_options={"_post_g2_pipeline_v2": complete},
        ).freeze(),
        coordinated=True,
    )
    assert choice.writer_settlement_batch_size == 1
    assert choice.nexus_record_batch_size == 8
    assert choice.reduction_inflight == 4
    assert choice.semantic_checkpoint_frame_cap == 56
    assert choice.staging_frame_cap == 64
    assert choice.buffers_nexus_records_across_calls is True
    with pytest.raises(FrozenInstanceError):
        choice.nexus_record_batch_size = 1

    invalid = (
        {**complete, "writer_settlement_batch_size": True},
        {**complete, "nexus_record_batch_size": 0},
        {**complete, "nexus_record_batch_size": 17},
        {**complete, "writer_settlement_batch_size": 17},
        {**complete, "reduction_inflight": 65},
        {**complete, "semantic_checkpoint_frame_cap": 0},
        {**complete, "writer_settlement_batch_size": 5},
        {**complete, "nexus_record_batch_size": 9,
         "semantic_checkpoint_frame_cap": 8},
        {**complete, "writer_settlement_batch_size": 3,
         "semantic_checkpoint_frame_cap": 56},
    )
    for value in invalid:
        with pytest.raises((TypeError, ValueError)):
            dynamic_output._post_g2_pipeline_v2_choice(
                RunIntent(run_options={"_post_g2_pipeline_v2": value}).freeze(),
                coordinated=True,
            )

    for intent, message in (
        (RunIntent(live_mode=True), "non-Live"),
        (RunIntent(batch_mode=True), "non-batch"),
        (RunIntent(output_mode="Append"), "Overwrite"),
        (RunIntent(output_mode="Inspect"), "Overwrite"),
        (RunIntent(
            output_mode="Overwrite", processing_mode="Int 1D (XYE)",
        ), "Nexus"),
    ):
        intent.run_options["_post_g2_pipeline_v2"] = complete
        with pytest.raises(ValueError, match=message):
            dynamic_output._post_g2_pipeline_v2_choice(
                intent.freeze(), coordinated=True,
            )
    with pytest.raises(ValueError, match="coordinated"):
        dynamic_output._post_g2_pipeline_v2_choice(
            RunIntent(
                output_mode="Overwrite",
                run_options={"_post_g2_pipeline_v2": complete},
            ).freeze(),
            coordinated=False,
        )

    five_fields = {
        **complete,
        "semantic_checkpoint_frame_cap": 1_000,
        "staging_frame_cap": 1_008,
    }
    five_choice = dynamic_output._post_g2_pipeline_v2_choice(
        RunIntent(
            output_mode="Overwrite",
            run_options={"_post_g2_pipeline_v2": five_fields},
        ).freeze(),
        coordinated=True,
    )
    assert five_choice.semantic_checkpoint_frame_cap == 1_000
    assert five_choice.staging_frame_cap == 1_008
    assert dynamic_output._unsafe_unfunded_staging_requested(
        choice,
        run_options={},
        env={"XDART_UNSAFE_UNFUNDED_STAGING_DIAGNOSTIC": "1"},
        frame_count=651,
    ) is False
    with pytest.raises(ValueError, match="staging"):
        dynamic_output._post_g2_pipeline_v2_choice(
            RunIntent(
                output_mode="Overwrite",
                run_options={"_post_g2_pipeline_v2": {
                    **complete,
                    "semantic_checkpoint_frame_cap": 1_000,
                }},
            ).freeze(),
            coordinated=True,
        )

def test_post_g2_output_diagnostics_are_exact_private_v2_values():
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    pipeline = {
        "writer_settlement_batch_size": 1,
        "nexus_record_batch_size": 8,
        "reduction_inflight": 16,
        "semantic_checkpoint_frame_cap": 56,
        "staging_frame_cap": 64,
    }
    configured = RunIntent(
        output_mode="Overwrite",
        run_options={
            "_post_g2_pipeline_v2": pipeline,
            "_post_g2_output_diagnostics_v1": {
                "save_xye": False,
                "durable_fsync": False,
            },
        },
    ).freeze()
    choice = dynamic_output._post_g2_output_diagnostics_choice(
        configured,
        pipeline_v2=dynamic_output._post_g2_pipeline_v2_choice(
            configured,
            coordinated=True,
        ),
    )
    assert choice.save_xye is False
    assert choice.durable_fsync is False
    assert choice.explicit is True

    defaults = dynamic_output._post_g2_output_diagnostics_choice(
        RunIntent().freeze(),
        pipeline_v2=None,
    )
    assert (defaults.save_xye, defaults.durable_fsync) == (True, True)
    assert defaults.explicit is False

    for value in (
        {"save_xye": False},
        {"save_xye": False, "durable_fsync": False, "extra": True},
        {"save_xye": 0, "durable_fsync": True},
        {"save_xye": True, "durable_fsync": 1},
    ):
        with pytest.raises((TypeError, ValueError)):
            dynamic_output._post_g2_output_diagnostics_choice(
                RunIntent(run_options={
                    "_post_g2_pipeline_v2": pipeline,
                    "_post_g2_output_diagnostics_v1": value,
                }).freeze(),
                pipeline_v2=object(),
            )

    with pytest.raises(ValueError, match="V2"):
        dynamic_output._post_g2_output_diagnostics_choice(
            RunIntent(run_options={
                "_post_g2_output_diagnostics_v1": {
                    "save_xye": True,
                    "durable_fsync": True,
                },
            }).freeze(),
            pipeline_v2=None,
        )


def test_post_g2_pipeline_refuses_before_dynamic_activation_effects(
    monkeypatch,
    tmp_path,
):
    from xdart.gui.tabs.scattering.adapters import dynamic_output

    touched = []
    monkeypatch.setattr(
        dynamic_output, "_science_projection",
        lambda *_args: touched.append("science"),
    )
    fields = (
        "writer_settlement_batch_size", "nexus_record_batch_size",
        "reduction_inflight", "semantic_checkpoint_frame_cap",
        "staging_frame_cap",
    )
    valid = dict(zip(fields, (1, 8, 8, 16, 64), strict=True))
    configurations = (
        RunIntent(run_options={
            "_post_g2_pipeline_v2": {
                key: valid[key] for key in fields[:-1]
            },
        }).freeze(),
        RunIntent(live_mode=True, run_options={
            "_post_g2_pipeline_v2": valid,
        }).freeze(),
        RunIntent(run_options={
            "_post_g2_pipeline_v2": {
                **valid, "semantic_checkpoint_frame_cap": False,
            },
        }).freeze(),
    )
    scan, item, decision = _one_frame_dynamic_admission(tmp_path)
    plan = SimpleNamespace(integration_1d=None, integration_2d=None, gi=None)
    for configuration in configurations:
        adapter = dynamic_output.DynamicOutputAdapter(configuration)
        with pytest.raises((TypeError, ValueError)):
            adapter.prepare_admission(
                scan,
                plan,
                item,
                decision,
                Event(),
                qualify=lambda _policy, prior: prior,
            )
        assert adapter._science_identity is None
        assert adapter._graphs == {}
        assert adapter._pending_preflights == []
        assert adapter._pending_nexus == []
        assert adapter._pending_xye == []
    assert touched == []


def test_auto_fact_survives_post_allocation_failure_and_reaches_terminal(
    monkeypatch, tmp_path, caplog,
):
    import logging
    from xdart.gui.tabs.scattering.adapters import dynamic_output, run_executor as executor_module
    from xdart.gui.tabs.scattering.display_runtime import RunDisplayState; from xdart.gui.tabs.scattering.display_values import StandardEventKind
    from xdart.gui.tabs.scattering.events import CleanupStatus, ExecutorClosed
    from xrd_tools.core import staging

    configuration = RunIntent(output_mode="Overwrite").freeze(); identity = RunIdentity.from_configuration(configuration)
    state = RunDisplayState(identity, max_payload_items=1); state.configure(partition_count=2, npt=10, frame_bytes=1024)
    monkeypatch.delenv("XDART_HEAVY_WINDOW", raising=False); monkeypatch.setattr(staging, "total_physical_ram_bytes", lambda: 64 * 1024 ** 3)
    monkeypatch.setattr(dynamic_output, "total_physical_ram_bytes", lambda: 64 * 1024 ** 3)
    real_policy = dynamic_output.resolve_session_policy
    monkeypatch.setattr(dynamic_output, "resolve_session_policy", lambda req, **kw: real_policy(req, envelope_bytes=8 * 1024 ** 3, **kw))
    monkeypatch.setattr(dynamic_output, "_science_projection", lambda *_: {})
    monkeypatch.setattr(dynamic_output, "_stable_lineage", lambda *_: ("auto",))
    monkeypatch.setattr(dynamic_output, "_append_intent", lambda *_a, **_k: SimpleNamespace(modes=()))
    monkeypatch.setattr(dynamic_output, "required_result_modes", lambda *_: ())
    scan, item, decision = _one_frame_dynamic_admission(tmp_path)
    plan = SimpleNamespace(integration_1d=None, integration_2d=SimpleNamespace(npt_rad=2, npt_azim=2, error_model=None), gi=None); facts = []
    bound = []
    def fail_bind(allocation):
        bound.append(allocation)
        raise RuntimeError("injected post-allocation failure")
    caplog.set_level(logging.INFO)
    adapter = dynamic_output.DynamicOutputAdapter(configuration)
    preparation = adapter.prepare_admission(
        scan,
        plan,
        item,
        decision,
        Event(),
        qualify=lambda _policy, prior: prior,
    )
    try:
        adapter.activate(
            preparation,
            record_store=object(), run_provenance={}, publication_store=object(), display_owner=object(),
            display_state=state, source_owner=SimpleNamespace(bind_allocation=fail_bind), gui_thread_id=1,
            light_cancel=lambda: None, light_drain=lambda: None, light_verify=lambda: None,
            on_frame_completed=lambda _event: None, resource_fact_sink=facts.append,
            on_checkpoint_recoverable=lambda _event: None,
        )
    except RuntimeError as error: assert str(error) == "injected post-allocation failure"
    else: raise AssertionError("post-allocation failure was not injected")
    assert len(bound) == 1
    assert len(facts) == 1
    fact = facts[0]
    assert (fact.choice, fact.resolution_source, fact.requested_heavy_bound) == ("auto", "auto", 64)
    for grants, expected in (((32, 16), 16), ((64, 64), 16), ((8, 12), 8)):
        assert state.bind_heavy_allocation(SimpleNamespace(record_heavy_items=grants[0], publication_heavy_items=grants[1])) == expected
    run = executor_module._StandardRun(configuration, identity, None, None, None, None, item.target, resource_facts=facts)
    executor_module.StandardRunExecutor()._terminal_event(
        run, StandardEventKind.FINISHED,
        ExecutorClosed(identity, CleanupStatus.CLEANED), 1, 1)
    messages = [record.getMessage() for record in caplog.records]
    assert [value for value in messages if value.startswith("[RUN-RESOURCES]")] == [fact.log_line("[RUN-RESOURCES]")]
    assert [value for value in messages if value.startswith("[PERF-RESOURCES]")] == [fact.log_line("[PERF-RESOURCES]")]


def test_processed_catalog_includes_parent_and_child_directory_navigation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "scan_10.nexus").touch()
    (root / "scan_2.nexus").touch()
    (root / "ignored.txt").touch()

    catalog = enumerate_processed_artifacts(str(root))

    assert tuple(
        (entry.label, entry.is_directory)
        for entry in catalog
    ) == (
        ("..", True),
        ("nested/", True),
        ("scan_2.nexus", False),
        ("scan_10.nexus", False),
    )
    assert catalog[0].artifact == str(tmp_path)
    assert catalog[1].artifact == str(root / "nested")


def test_browser_catalog_uses_exact_normal_and_viewer_suffix_policies(
    tmp_path: Path,
) -> None:
    from xrd_tools.io.viewer_1d import SUPPORTED_VIEWER_1D_SUFFIXES
    from xrd_tools.io.viewer_2d import SUPPORTED_VIEWER_SUFFIXES

    root = tmp_path / "mixed"
    root.mkdir()
    (root / "nested").mkdir()
    for name in ("scan.nexus", "curve.xye", "image.tif", "ignored.txt"):
        (root / name).touch()

    labels = lambda policy: tuple(
        entry.label for entry in enumerate_processed_artifacts(
            str(root), accepted_suffixes=policy,
        )
    )
    assert labels(None) == ("..", "nested/", "scan.nexus")
    assert labels(SUPPORTED_VIEWER_1D_SUFFIXES) == (
        "..", "curve.xye", "nested/",
    )
    assert labels(SUPPORTED_VIEWER_SUFFIXES) == (
        "..", "image.tif", "nested/", "scan.nexus",
    )
    assert browser_suffixes_for_mode("1D Viewer") == SUPPORTED_VIEWER_1D_SUFFIXES
    assert browser_suffixes_for_mode("2D Viewer") == SUPPORTED_VIEWER_SUFFIXES
    assert browser_suffixes_for_mode("Int 2D") is None


def test_deleted_processed_directory_retains_parent_navigation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    root.rmdir()

    catalog = enumerate_processed_artifacts(str(root))

    assert tuple(
        (entry.label, entry.artifact, entry.is_directory)
        for entry in catalog
    ) == (("..", str(tmp_path), True),)


def test_processed_catalog_naturally_interleaves_directories_and_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    (root / "scan_2").mkdir()
    (root / "scan_20").mkdir()
    (root / "scan_1.nexus").touch()
    (root / "scan_3.nexus").touch()
    (root / "scan_10.nexus").touch()

    catalog = enumerate_processed_artifacts(str(root))

    assert tuple(entry.label for entry in catalog) == (
        "..",
        "scan_1.nexus",
        "scan_2/",
        "scan_3.nexus",
        "scan_10.nexus",
        "scan_20/",
    )


def test_processed_directory_maps_explicit_nexus_artifact_to_parent(tmp_path):
    from xdart.gui.tabs.scattering.browser_catalog import processed_directory

    target = tmp_path / "out" / "scan.nexus"
    assert processed_directory(str(target)) == str(tmp_path / "out")


def test_browser_time_sort_interleaves_directories_and_artifacts() -> None:
    catalog = (
        BrowserCatalogEntry("/out/old-dir", "old-dir/", 10, True),
        BrowserCatalogEntry("/out/new-file.nxs", "new-file.nxs", 40),
        BrowserCatalogEntry("/", "..", 50, True),
        BrowserCatalogEntry("/out/new-dir", "new-dir/", 30, True),
        BrowserCatalogEntry("/out/old-file.nxs", "old-file.nxs", 20),
    )

    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=FrameNavigationProjection(),
        browser_directory="/out",
        date_sorted=True,
        auto_last=True,
        catalog=catalog,
    )

    assert tuple(scan.label for scan in projected.scans) == (
        "..",
        "new-file.nxs",
        "new-dir/",
        "old-file.nxs",
        "old-dir/",
    )


def test_browser_projection_reuses_prevalidated_static_scan_tuple() -> None:
    catalog = (
        BrowserCatalogEntry("/out/old.nexus", "old.nexus", 10),
        BrowserCatalogEntry("/out/new.nexus", "new.nexus", 20),
    )
    index = build_browser_scan_index(catalog, True)
    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=FrameNavigationProjection(),
        browser_directory="/out",
        date_sorted=True,
        auto_last=True,
        catalog_index=index,
    )

    assert projected.scans is index.scans
    assert tuple(scan.label for scan in projected.scans) == (
        "new.nexus",
        "old.nexus",
    )
    with pytest.raises(TypeError, match="scan index"):
        BrowserScanIndex(
            index.scans,
            frozenset({"/out/not-present.nexus"}),
            True,
        )
    with pytest.raises(TypeError, match="inconsistent"):
        build_browser_projection(
            contexts=(),
            selection=None,
            navigation=FrameNavigationProjection(),
            browser_directory="/out",
            date_sorted=False,
            auto_last=True,
            catalog_index=index,
        )
    with pytest.raises(TypeError, match="inconsistent"):
        build_browser_projection(
            contexts=(),
            selection=None,
            navigation=FrameNavigationProjection(),
            browser_directory="/out",
            date_sorted=True,
            auto_last=True,
            catalog=catalog,
            catalog_index=index,
        )


def test_browser_time_sort_derives_directory_time_from_immediate_children(
    tmp_path: Path,
) -> None:
    """Date order follows scan contents, not directory inode churn."""

    root = tmp_path / "processed"
    root.mkdir()
    inode_newer = root / "inode-newer"
    content_newer = root / "content-newer"
    inode_newer.mkdir()
    content_newer.mkdir()
    old_child = inode_newer / "old.nexus"
    new_child = content_newer / "new.nexus"
    old_child.touch()
    new_child.touch()
    nested = inode_newer / "nested"
    nested.mkdir()
    grandchild = nested / "must-not-count.nexus"
    grandchild.touch()

    os.utime(old_child, ns=(100, 100))
    os.utime(new_child, ns=(300, 300))
    os.utime(grandchild, ns=(900, 900))
    os.utime(nested, ns=(50, 50))
    # Deliberately contradict the content order at the inode layer.
    os.utime(inode_newer, ns=(800, 800))
    os.utime(content_newer, ns=(200, 200))

    catalog = enumerate_processed_artifacts(
        str(root),
        inspect_directory_contents=True,
    )
    by_label = {entry.label: entry for entry in catalog}

    assert by_label["inode-newer/"].modified_ns == 100
    assert by_label["content-newer/"].modified_ns == 300
    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=FrameNavigationProjection(),
        browser_directory=str(root),
        date_sorted=True,
        auto_last=False,
        catalog=catalog,
    )
    assert tuple(scan.label for scan in projected.scans[:3]) == (
        "..",
        "content-newer/",
        "inode-newer/",
    )


def test_directory_time_cache_clear_during_scan_cannot_repopulate_stale_value(
    tmp_path: Path,
    monkeypatch,
) -> None:
    directory = tmp_path / "scan"
    directory.mkdir()
    entered = Event()
    release = Event()
    calls: list[int] = []

    def blocked_scan(_directory: Path, _own_mtime_ns: int):
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=2.0)
        return 100 + len(calls), True

    monkeypatch.setattr(
        browser_catalog_module,
        "_directory_modified_ns",
        blocked_scan,
    )
    cache = DirectoryModifiedCache(ttl_s=60.0)
    results: list[int] = []
    worker = Thread(
        target=lambda: results.append(cache.modified_ns(directory, 1)),
    )
    worker.start()
    assert entered.wait(timeout=2.0)
    cache.clear()
    release.set()
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert results == [101]
    assert cache.modified_ns(directory, 1) == 102
    assert cache.modified_ns(directory, 1) == 102
    assert calls == [1, 2]


def _keys() -> tuple[DisplayFrameKey, ...]:
    identity = RunIdentity(9, "e4-live-browser")
    return (
        DisplayFrameKey(identity, "scan-a", "/out/a.nxs", 0, 1),
        DisplayFrameKey(identity, "scan-a", "/out/a.nxs", 1, 2),
        DisplayFrameKey(identity, "scan-b", "/out/b.nxs", 0, 3),
        DisplayFrameKey(identity, "scan-b", "/out/b.nxs", 1, 4),
    )


def test_browser_projection_borrows_only_inflight_artifact_keys() -> None:
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[3], (keys[3],))
    index = build_browser_scan_index((), False)

    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        catalog_index=index,
        transient_frame=keys[3],
    )

    assert tuple(scan.identifier for scan in projected.scans) == (
        "/out/b.nxs",
    )
    assert tuple(scan.label for scan in projected.scans) == ("b.nxs",)
    assert projected.selected_scan == "/out/b.nxs"
    assert getattr(projected, "frames", ()) == keys[2:]
    assert all(
        actual is expected
        for actual, expected in zip(
            projected.frames,
            keys[2:],
            strict=True,
        )
    )
    assert navigation.frames is keys
    assert index.scans == () and index.identifiers == frozenset()


def test_catalog_directory_fact_reaches_exact_activation_command() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    navigation = FrameNavigationProjection()
    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        catalog=(
            BrowserCatalogEntry("/out/subdir", "subdir/", 2, True),
            BrowserCatalogEntry("/out/result.nxs", "result.nxs", 1),
        ),
    )
    assert tuple(scan.is_directory for scan in projected.scans) == (True, False)

    browser = BrowserView()
    commands = []
    browser.commandRequested.connect(commands.append)
    try:
        viewer_projected = replace(
            projected,
            selected_scan="/out/result.nxs",
            selected_artifacts=("/out/result.nxs",),
            multi_artifact_selection=True,
        )
        browser.reconcile(viewer_projected, navigation, plot_mode="Single")
        assert browser.scans.selectionMode() == (
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        assert tuple(
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in browser.scans.selectedItems()
        ) == ("/out/result.nxs",)
        assert browser.scans.currentItem().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) == "/out/result.nxs"
        for row, expected in enumerate(("directory", "artifact")):
            blocker = QtCore.QSignalBlocker(browser.scans)
            browser.scans.setCurrentRow(row)
            del blocker
            browser._scan_selected()
            assert commands[-1] == ShellCommand(
                ShellCommandKind.SELECT_SCAN,
                projected.scans[row].identifier,
                path=(expected,),
                artifacts=(
                    ()
                    if expected == "directory"
                    else (projected.scans[row].identifier,)
                ),
            )
    finally:
        browser.deleteLater()
        app.processEvents()


def test_unchanged_artifact_selection_reconcile_touches_no_scan_rows(
    monkeypatch,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    touches: list[tuple[str, bool]] = []
    item_type = QtWidgets.QListWidgetItem

    class TrackedItem(item_type):
        def setSelected(self, selected: bool) -> None:
            touches.append((str(self.data(QtCore.Qt.ItemDataRole.UserRole)), selected))
            super().setSelected(selected)

    monkeypatch.setattr(
        browser_view_module.QtWidgets, "QListWidgetItem", TrackedItem,
    )
    paths = tuple(
        f"/out/scan-{index:04d}.nexus" for index in range(651)
    )
    catalog = tuple(
        BrowserCatalogEntry(path, Path(path).name, index)
        for index, path in enumerate(paths)
    )
    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=FrameNavigationProjection(),
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        catalog=catalog,
        selected_artifacts=(paths[0],),
        current_artifact=paths[0],
        multi_artifact_selection=True,
    )
    browser = BrowserView()
    navigation = FrameNavigationProjection()

    def apply(state) -> int:
        touches.clear()
        browser.reconcile(state, navigation, plot_mode="Single")
        return len(touches)

    try:
        assert apply(projected) == len(paths)
        assert apply(projected) == 0

        changed = replace(
            projected,
            selected_scan=paths[2],
            selected_artifacts=(paths[0], paths[2]),
        )
        assert apply(changed) == len(paths)
        assert apply(changed) == 0

        cleared = replace(
            changed, selected_scan="", selected_artifacts=(),
        )
        assert apply(cleared) == len(paths)
        assert browser.scans.currentItem() is None
        assert browser.scans.selectedItems() == []
        assert apply(cleared) == 0

        browser.scans.setCurrentItem(
            browser.scans.item(1),
            QtCore.QItemSelectionModel.SelectionFlag.NoUpdate,
        )
        assert apply(cleared) == len(paths)
        assert browser.scans.currentItem() is None

        # A real Qt-side selection drift is still repaired even when the
        # authoritative projection value itself is unchanged.
        apply(changed)
        browser.scans.item(1).setSelected(True)
        assert apply(changed) == len(paths)
        assert tuple(
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in browser.scans.selectedItems()
        ) == (paths[0], paths[2])

        extended = replace(
            changed,
            scans=(*changed.scans, replace(changed.scans[-1],
                identifier="/out/d.nexus", label="d.nexus")),
        )
        assert apply(extended) == len(paths) + 1
    finally:
        browser.deleteLater()
        app.processEvents()


def test_viewer_1d_artifact_modifier_gestures_emit_clicked_current_and_catalog_order(
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paths = tuple(f"/out/{name}.xye" for name in ("a", "b", "c"))
    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=FrameNavigationProjection(),
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        catalog=tuple(
            BrowserCatalogEntry(path, Path(path).name, index)
            for index, path in enumerate(paths)
        ),
        selected_artifacts=(paths[0],),
        current_artifact=paths[0],
        multi_artifact_selection=True,
    )
    browser = BrowserView()
    browser.resize(640, 480)
    browser.show()
    commands = []
    browser.commandRequested.connect(commands.append)
    try:
        browser.reconcile(projected, FrameNavigationProjection(), plot_mode="Single")
        app.processEvents()

        third = browser.scans.visualItemRect(browser.scans.item(2)).center()
        QtTest.QTest.mouseClick(
            browser.scans.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
            third,
        )
        app.processEvents()
        assert commands[-1] == ShellCommand(
            ShellCommandKind.SELECT_SCAN,
            paths[2],
            path=("artifact",),
            artifacts=(paths[0], paths[2]),
        )

        browser.reconcile(projected, FrameNavigationProjection(), plot_mode="Single")
        commands.clear()
        app.processEvents()
        QtTest.QTest.mouseClick(
            browser.scans.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
            third,
        )
        app.processEvents()
        assert commands[-1] == ShellCommand(
            ShellCommandKind.SELECT_SCAN,
            paths[2],
            path=("artifact",),
            artifacts=paths,
        )

        projected_all = replace(
            projected,
            selected_scan=paths[2],
            selected_artifacts=paths,
        )
        browser.reconcile(
            projected_all, FrameNavigationProjection(), plot_mode="Single"
        )
        commands.clear()
        middle = browser.scans.visualItemRect(browser.scans.item(1)).center()
        QtTest.QTest.mouseClick(
            browser.scans.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
            middle,
        )
        app.processEvents()
        assert commands[-1] == ShellCommand(
            ShellCommandKind.SELECT_SCAN,
            paths[2],
            path=("artifact",),
            artifacts=(paths[0], paths[2]),
        )

        projected_pair = replace(
            projected,
            selected_scan=paths[2],
            selected_artifacts=(paths[0], paths[2]),
        )
        browser.reconcile(
            projected_pair, FrameNavigationProjection(), plot_mode="Single"
        )
        commands.clear()
        QtTest.QTest.mouseClick(
            browser.scans.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
            third,
        )
        app.processEvents()
        assert commands[-1] == ShellCommand(
            ShellCommandKind.SELECT_SCAN,
            paths[0],
            path=("artifact",),
            artifacts=(paths[0],),
        )

        browser.reconcile(projected, FrameNavigationProjection(), plot_mode="Single")
        commands.clear()
        first = browser.scans.visualItemRect(browser.scans.item(0)).center()
        QtTest.QTest.mouseClick(
            browser.scans.viewport(),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.ControlModifier,
            first,
        )
        app.processEvents()
        assert commands == []
        assert tuple(
            item.data(QtCore.Qt.ItemDataRole.UserRole)
            for item in browser.scans.selectedItems()
        ) == (paths[0],)
    finally:
        browser.deleteLater()
        app.processEvents()


def test_authoritative_catalog_does_not_resurrect_historical_navigation(
) -> None:
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[3], (keys[3],))
    retained_context = SimpleNamespace(
        source="/out/missing.nxs",
        context_token="retained-browse",
        scan_key="missing",
    )

    projected = build_browser_projection(
        contexts=(retained_context,),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        catalog=(BrowserCatalogEntry("/", "..", 1, True),),
    )

    assert tuple(scan.identifier for scan in projected.scans) == ("/",)
    assert projected.selected_scan == ""
    assert projected.frames == ()
    assert navigation.frames is keys


def test_catalog_artifact_remains_selectable_after_transient_identity_ends(
) -> None:
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[3], (keys[3],))

    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        catalog=(BrowserCatalogEntry("/out/b.nxs", "b.nxs", 2),),
    )

    assert tuple(scan.identifier for scan in projected.scans) == (
        "/out/b.nxs",
    )
    assert projected.selected_scan == "/out/b.nxs"
    assert all(
        actual is expected
        for actual, expected in zip(
            projected.frames,
            keys[2:],
            strict=True,
        )
    )


def test_equal_distinct_frame_cannot_lend_a_transient_browser_artifact(
) -> None:
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[3], (keys[3],))
    latest = keys[3]
    equal_distinct = DisplayFrameKey(
        latest.run_identity,
        latest.source_scan,
        latest.artifact,
        latest.local_frame_label,
        latest.work_ordinal,
    )

    projected = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        transient_frame=equal_distinct,
    )

    assert equal_distinct == latest
    assert equal_distinct is not latest
    assert projected.scans == ()
    assert projected.selected_scan == ""


def test_browser_menu_theme_uses_canonical_indicator_and_font_contract() -> None:
    font_selector = (
        "QToolButton#fileMenuButton,\n"
        "QToolButton#configMenuButton,\n"
        "QToolButton#analysisMenuButton,\n"
        "QToolButton#helpMenuButton {"
    )
    indicator_selector = (
        "QToolButton#fileMenuButton::menu-indicator,\n"
        "QToolButton#configMenuButton::menu-indicator,\n"
        "QToolButton#analysisMenuButton::menu-indicator,\n"
        "QToolButton#helpMenuButton::menu-indicator {"
    )
    default = render_qss("dark", font_scale="default")
    extra_large = render_qss("dark", font_scale="extra_large")

    assert indicator_selector in default
    default_block = default[
        default.index(font_selector):default.index(
            "}", default.index(font_selector)
        )
    ]
    extra_large_block = extra_large[
        extra_large.index(font_selector):extra_large.index(
            "}", extra_large.index(font_selector)
        )
    ]
    assert "font-size: 13px;" in default_block
    assert "font-size: 15px;" in extra_large_block


def test_browser_frame_list_uses_uniform_item_sizes_for_long_live_prefix() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    browser = BrowserView()
    try:
        assert browser.frames.uniformItemSizes() is True
    finally:
        browser.deleteLater()
        app.processEvents()


def test_footer_and_left_frames_reconcile_to_same_exact_key() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    keys = _keys()
    navigation = FrameNavigationProjection(keys, keys[1], (keys[1],))
    browser = build_browser_projection(
        contexts=(),
        selection=None,
        navigation=navigation,
        browser_directory="/out",
        date_sorted=False,
        auto_last=True,
        transient_frame=keys[1],
    )
    base = make_shell_projection(
        frame_count=1,
        selected_index=0,
        plot_mode="Single",
    )
    state = replace(
        base,
        browser=browser,
        navigation=navigation,
        scientific=replace(
            base.scientific,
            traces=(),
            heavy=None,
            heavy_available=frozenset(),
            retain_display=True,
        ),
    )
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)
        browser_model = shell.browser.frame_model
        assert [
            browser_model.index(index, 0).data(
                QtCore.Qt.ItemDataRole.DisplayRole
            )
            for index in range(browser_model.rowCount())
        ] == ["1", "2"]
        assert [
            shell.scientific.frame_selector.itemText(index)
            for index in range(shell.scientific.frame_selector.count())
        ] == ["1", "2"]
        assert [
            (
                browser_model.index(index, 0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
            ).local_frame_label
            for index in range(browser_model.rowCount())
        ] == [0, 1]
        assert [
            shell.scientific.frame_selector.itemData(
                index
            ).local_frame_label
            for index in range(shell.scientific.frame_selector.count())
        ] == [0, 1]
        selected = shell.browser.frames.selectionModel().selectedRows()
        assert len(selected) == 1
        assert selected[0].data(QtCore.Qt.ItemDataRole.UserRole) is keys[1]
        assert shell.browser.frames.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is keys[1]
        assert shell.scientific.frame_selector.currentData() is keys[1]

        # Simulate the live failure class: Qt's painted selection has drifted
        # while the passive view's identity cache still describes navigation.
        selection = shell.browser.frames.selectionModel()
        blocker = QtCore.QSignalBlocker(selection)
        selection.clearSelection()
        selection.setCurrentIndex(
            shell.browser.frame_model.index(0, 0),
            QtCore.QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QtCore.QItemSelectionModel.SelectionFlag.Rows,
        )
        del blocker
        shell.apply_state(replace(state, revision=state.revision + 1))
        selected = selection.selectedRows()
        assert len(selected) == 1
        assert selected[0].data(QtCore.Qt.ItemDataRole.UserRole) is keys[1]
        assert selection.currentIndex().data(
            QtCore.Qt.ItemDataRole.UserRole
        ) is keys[1]

        next_navigation = FrameNavigationProjection(
            keys,
            keys[0],
            (keys[0],),
        )
        selection_changes: list[tuple[object, object]] = []
        selection.selectionChanged.connect(
            lambda selected, deselected: selection_changes.append(
                (selected, deselected)
            )
        )
        shell.apply_state(
            replace(
                state,
                revision=state.revision + 2,
                browser=build_browser_projection(
                    contexts=(),
                    selection=None,
                    navigation=next_navigation,
                    browser_directory="/out",
                    date_sorted=False,
                    auto_last=True,
                    transient_frame=keys[1],
                ),
                navigation=next_navigation,
            )
        )
        selected = shell.browser.frames.selectionModel().selectedRows()
        assert len(selected) == 1
        assert selected[0].data(QtCore.Qt.ItemDataRole.UserRole) is keys[0]
        assert shell.scientific.frame_selector.currentData() is keys[0]
        assert selection_changes, (
            "the passive reconcile must notify Qt's selection view so the "
            "highlight repaints in the same event-loop turn"
        )
    finally:
        shell.close()
        shell.deleteLater()
        app.processEvents()


def test_open_folder_and_refresh_publish_processed_catalog(
    tmp_path: Path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "initial.nexus").touch()
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "scan_10.nexus").touch()
    (selected / "scan_2.nexus").touch()
    nested = selected / "nested"
    nested.mkdir()
    (nested / "inside.nexus").touch()
    chooser_calls: list[str] = []

    def choose(current: str, _start_directory: str) -> str:
        chooser_calls.append(current)
        return str(selected)

    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(
                project_root=str(tmp_path / "raw"),
                save_path=str(configured),
            )
        ),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        browser_directory_chooser=choose,
    )
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("processed browser catalog did not settle")

    try:
        wait_for(
            lambda: shell.browser.directory_label._path
            == str(configured)
        )
        menu_buttons = {
            button.objectName(): button.text()
            for button in shell.browser.findChildren(QtWidgets.QToolButton)
            if button.objectName() in {
                "fileMenuButton",
                "configMenuButton",
                "analysisMenuButton",
                "helpMenuButton",
            }
        }
        assert menu_buttons == {
            "fileMenuButton": "File",
            "configMenuButton": "Config",
            "analysisMenuButton": "Analysis",
            "helpMenuButton": "Help",
        }
        file_menu = shell.browser.findChild(
            QtWidgets.QToolButton,
            "fileMenuButton",
        )
        assert file_menu is not None
        open_folder = next(
            action
            for action in file_menu.menu().actions()
            if action.text() == "Open Folder"
        )
        open_folder.trigger()
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(selected)
                and shell.browser.scans.count() == 4
            )
        )
        assert chooser_calls == [str(configured)]
        artifacts = [
            shell.browser.scans.item(index).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            for index in range(shell.browser.scans.count())
        ]
        assert artifacts == [
            str(tmp_path),
            str(nested),
            str(selected / "scan_2.nexus"),
            str(selected / "scan_10.nexus"),
        ]

        (selected / "scan_1.nexus").touch()
        shell.browser.refresh.click()
        wait_for(lambda: shell.browser.scans.count() == 5)
        assert [
            shell.browser.scans.item(index).data(
                QtCore.Qt.ItemDataRole.UserRole
            )
            for index in range(shell.browser.scans.count())
        ][2] == str(selected / "scan_1.nexus")

        # An idle external deletion/recreation has no shell command or run
        # event to drive a refresh.  The background catalog poll must evict
        # and later restore the exact filesystem member on its own.
        (selected / "scan_10.nexus").unlink()
        wait_for(
            lambda: (
                shell.browser.scans.count() == 4
                and all(
                    shell.browser.scans.item(index).data(
                        QtCore.Qt.ItemDataRole.UserRole
                    ) != str(selected / "scan_10.nexus")
                    for index in range(shell.browser.scans.count())
                )
            )
        )
        (selected / "scan_10.nexus").touch()
        wait_for(
            lambda: any(
                shell.browser.scans.item(index).data(
                    QtCore.Qt.ItemDataRole.UserRole
                ) == str(selected / "scan_10.nexus")
                for index in range(shell.browser.scans.count())
            )
        )

        nested_item = next(
            shell.browser.scans.item(index)
            for index in range(shell.browser.scans.count())
            if shell.browser.scans.item(index).data(
                QtCore.Qt.ItemDataRole.UserRole
            ) == str(nested)
        )
        shell.browser.scans.setCurrentItem(nested_item)
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(nested)
                and shell.browser.scans.count() == 2
            )
        )

        (nested / "inside.nexus").unlink()
        nested.rmdir()
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(nested)
                and shell.browser.scans.count() == 1
                and shell.browser.scans.item(0).text() == ".."
                and shell.browser.scans.item(0).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                == str(selected)
            )
        )
        shell.browser.scans.setCurrentItem(shell.browser.scans.item(0))
        wait_for(
            lambda: shell.browser.directory_label._path == str(selected)
        )
    finally:
        page.close_workspace()
        assert not page._browser_catalog_timer.isActive()
        page.deleteLater()
        app.processEvents()


def test_page_time_sort_and_refresh_use_immediate_child_mtime(
    tmp_path: Path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    processed = tmp_path / "processed"
    processed.mkdir()
    first = processed / "first"
    second = processed / "second"
    first.mkdir()
    second.mkdir()
    first_child = first / "frame.nexus"
    second_child = second / "frame.nexus"
    first_child.touch()
    second_child.touch()
    base = time.time_ns() - 10_000_000_000
    os.utime(first_child, ns=(base + 300, base + 300))
    os.utime(second_child, ns=(base + 100, base + 100))
    # Contradict the content order at the folder-inode layer.
    os.utime(first, ns=(base + 100, base + 100))
    os.utime(second, ns=(base + 300, base + 300))

    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(save_path=str(processed))),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None

    def labels() -> tuple[str, ...]:
        return tuple(
            shell.browser.scans.item(index).text()
            for index in range(shell.browser.scans.count())
        )

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("page Time-sort catalog did not settle")

    try:
        wait_for(lambda: shell.browser.scans.count() == 3)
        # Keep the finite row attributable to the two explicit commands. A
        # later 1.5 s poll must not hide a missing Refresh-cache invalidation
        # after the two-second TTL expires.
        page._browser_catalog_timer.stop()
        shell.browser.date_sort.click()
        wait_for(lambda: labels() == ("..", "first/", "second/"))

        os.utime(second_child, ns=(base + 500, base + 500))
        shell.browser.refresh.click()
        wait_for(lambda: labels() == ("..", "second/", "first/"))
    finally:
        page.close_workspace()
        page.deleteLater()
        app.processEvents()


def test_published_artifact_auto_follows_until_user_opens_folder(
    tmp_path: Path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configured = tmp_path / "configured"
    configured.mkdir()
    produced = tmp_path / "processed"
    produced.mkdir()
    first_artifact = produced / "scan_2.nexus"
    first_artifact.touch()
    selected = tmp_path / "selected"
    selected.mkdir()
    selected_artifact = selected / "browse.nexus"
    selected_artifact.touch()
    other = tmp_path / "other"
    other.mkdir()
    later_artifact = other / "later.nexus"
    later_artifact.touch()

    page = ScatteringWorkspace(
        intents=RunIntentStore(
            RunIntent(save_path=str(configured))
        ),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
        browser_directory_chooser=(
            lambda _current, _start_directory: str(selected)
        ),
    )
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("processed browser auto-follow did not settle")

    identity = RunIdentity(10, "browser-auto-follow")
    first = DisplayFrameKey(
        identity,
        "scan_2",
        str(first_artifact),
        0,
        1,
    )
    later = DisplayFrameKey(
        identity,
        "later",
        str(later_artifact),
        0,
        2,
    )
    try:
        page._follow_processed_artifact(first)
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(produced)
                and shell.browser.scans.count() == 2
                and shell.browser.scans.item(1).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                == str(first_artifact)
            )
        )

        page._handle_shell_command(
            ShellCommand(
                ShellCommandKind.MENU,
                "File:Open Folder",
            )
        )
        wait_for(
            lambda: shell.browser.directory_label._path == str(selected)
        )
        page._follow_processed_artifact(later)
        wait_for(lambda: shell.browser.scans.count() == 2)
        assert shell.browser.directory_label._path == str(selected)
        assert shell.browser.scans.item(1).data(
            QtCore.Qt.ItemDataRole.UserRole
        ) == str(selected_artifact)

        next_identity = RunIdentity(11, "next-browser-auto-follow")
        next_frame = DisplayFrameKey(
            next_identity,
            "later",
            str(later_artifact),
            0,
            1,
        )
        page._processed_browser.begin_follow(next_identity)
        page._follow_processed_artifact(next_frame)
        wait_for(
            lambda: (
                shell.browser.directory_label._path == str(other)
                and shell.browser.scans.count() == 2
            )
        )
        assert shell.browser.scans.item(1).data(
            QtCore.Qt.ItemDataRole.UserRole
        ) == str(later_artifact)
    finally:
        page.close_workspace()
        page.deleteLater()
        app.processEvents()


def test_terminal_transient_is_retained_until_catalog_barrier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configured = tmp_path / "configured"
    configured.mkdir()
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(save_path=str(configured))),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("browser terminal catalog barrier did not settle")

    identity = RunIdentity(12, "browser-terminal-barrier")
    frame = DisplayFrameKey(
        identity,
        "atomic-output",
        str(configured / "atomic-output.nexus"),
        0,
        1,
    )
    try:
        wait_for(lambda: page._processed_browser.active_request is None)
        page._browser_catalog_timer.stop()
        refreshes: list[bool] = []
        refresh_shell = page._refresh_shell

        def record_refresh(*, preserve_scientific=False, **kwargs):
            refreshes.append(preserve_scientific)
            return refresh_shell(
                preserve_scientific=preserve_scientific,
                **kwargs,
            )

        monkeypatch.setattr(page, "_refresh_shell", record_refresh)
        page._processed_browser.set_transient_frame(frame)
        request = page._request_browser_catalog()
        assert request is not None
        page._processed_browser.mark_transient_catalog_barrier(request)

        # The terminal refresh is asynchronous.  Do not create a one-turn
        # blank browser/display seam by dropping the only exact transient
        # owner before the authoritative filesystem result arrives.
        assert page._processed_browser.transient_frame is frame
        wait_for(lambda: page._processed_browser.active_request is None)
        assert page._processed_browser.transient_frame is None
        assert refreshes and refreshes[-1] is True
    finally:
        page.close_workspace()
        page.deleteLater()
        app.processEvents()


def test_browser_catalog_cooperatively_stops_mid_enumeration(
    tmp_path, monkeypatch,
) -> None:
    cancelled = Event()
    stats = []

    class _Entry:
        path = str(tmp_path / "first.nxs")
        def stat(self):
            stats.append(self.path)
            cancelled.set()
            return SimpleNamespace(
                st_mode=browser_catalog_module.stat.S_IFREG,
                st_mtime_ns=1,
            )

    class _Scan:
        def __enter__(self):
            return iter((_Entry(), _Entry()))
        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(browser_catalog_module.os, "scandir", lambda _root: _Scan())

    assert enumerate_processed_artifacts(
        str(tmp_path), accepted_suffixes=frozenset({".nxs"}),
        cancelled=cancelled,
    ) == ()
    assert stats == [str(tmp_path / "first.nxs")]


def test_browser_catalog_active_plus_latest_queued_never_publishes_stale(
    tmp_path, monkeypatch,
) -> None:
    import xdart.gui.tabs.scattering.processed_browser as browser_owner_module

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    entered = Event()
    release = Event()
    calls = []

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("catalog latest-only handoff did not settle")

    def catalog(directory, **_kwargs):
        calls.append(directory)
        if directory.endswith("/a"):
            entered.set()
            assert release.wait(timeout=5.0)
        if directory == str(tmp_path):
            return ()
        return (BrowserCatalogEntry(
            str(Path(directory) / "result.nexus"), Path(directory).name, 1,
        ),)

    monkeypatch.setattr(
        browser_owner_module, "enumerate_processed_artifacts", catalog,
    )
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(save_path=str(tmp_path))),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
    )
    try:
        wait_for(lambda: page._processed_browser.active_request is None)
        page._browser_catalog_timer.stop()
        calls.clear()
        page._set_browser_directory(str(tmp_path / "a"), explicit=False)
        assert entered.wait(timeout=5.0)
        page._set_browser_directory(str(tmp_path / "b"), explicit=False)
        page._set_browser_directory(str(tmp_path / "c"), explicit=False)
        latest = page._processed_browser.queued_request
        assert latest is not None
        assert latest.directory == str(tmp_path / "c")
        release.set()
        wait_for(lambda: (
            page._processed_browser.active_request is None
            and page._processed_browser.queued_request is None
        ))
        assert calls == [str(tmp_path / "a"), str(tmp_path / "c")]
        assert tuple(
            entry.label for entry in page._processed_browser.catalog
        ) == ("c",)
    finally:
        release.set()
        page.close_workspace()
        page.deleteLater()
        app.processEvents()


def test_browser_catalog_cancel_is_nonblocking_until_exact_future_retires(
    tmp_path, monkeypatch,
) -> None:
    from xdart.gui.tabs.scattering.events import CleanupStatus
    import xdart.gui.tabs.scattering.processed_browser as browser_owner_module

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    selected = tmp_path / "blocked"
    selected.mkdir()
    entered = Event()
    release = Event()

    def wait_for(predicate) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("catalog cancellation did not settle")

    def blocked(directory, **_kwargs):
        if directory == str(selected):
            entered.set()
            assert release.wait(timeout=5.0)
        return ()

    monkeypatch.setattr(
        browser_owner_module, "enumerate_processed_artifacts", blocked,
    )
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(save_path=str(tmp_path))),
        lifecycle=ScatteringCoordinator(), sources=FilesystemSourceAdapter(),
    )
    try:
        wait_for(lambda: page._processed_browser.active_request is None)
        page._browser_catalog_timer.stop()
        page._set_browser_directory(str(selected), explicit=False)
        assert entered.wait(timeout=5.0)
        started = time.monotonic()
        pending = page.close_workspace()
        assert pending.cleanup_status is CleanupStatus.CLEANUP_PENDING
        assert time.monotonic() - started < 0.2
        release.set()
        wait_for(lambda: page._processed_browser.retry_close())
        cleaned = page.close_workspace()
        assert cleaned.cleanup_status is CleanupStatus.CLEANED
        assert page._processed_browser.closed
    finally:
        release.set()
        page.close_workspace()
        page.deleteLater()
        app.processEvents()
