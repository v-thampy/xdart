from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import subprocess

from xdart.gui.tabs.scattering.context_projection import ContextProjection
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.shell_projection import (
    RUN_MODE_CHOICES,
    ScientificPreferences,
    UNOWNED_RUN_MODE_REASONS,
    build_run_strip_projection,
)
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.shell_values import (
    DirectoryFileProgress,
    ProgressProjection,
)
from xrd_tools.session.display_logic import Mode
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.readiness import ProcessingPage, Tool, tool_from_mode_text
from xrd_tools.sources.selection import DirectorySourceSpec

from tests.xdart.scattering.e3_shell_support import make_shell_projection


_EXPECTED_MODES = (
    "Int 1D",
    "Int 2D",
    "Int 1D (XYE)",
    "2D Viewer",
    "1D Viewer",
)
_EXPECTED_DISABLED_REASONS = ()


def _configured_intent(mode: str) -> RunIntent:
    return RunIntent(
        source_spec=DirectorySourceSpec(
            Path("/data/raw"),
            suffixes=(".h5",),
        ),
        processing_mode=mode,
        poni_file="/project/detector.poni",
        save_path="/processed",
        output_mode="Overwrite",
    )


def _run_strip(intent: RunIntent, **changes):
    values = {"executor_available": False, "start_permitted": True, "start_blocker": "", **changes}
    return build_run_strip_projection(RunPhase.IDLE, intent, **values)


def test_viewer_aliases_share_one_headless_parser_and_2d_is_mounted() -> None:
    aliases = ("2D Viewer", "Image Viewer", "1D Viewer", "XYE Viewer")
    assert tuple(tool_from_mode_text(value) for value in aliases) == (
        Tool.IMAGE_VIEWER, Tool.IMAGE_VIEWER, Tool.XYE_VIEWER, Tool.XYE_VIEWER)
    viewer, legacy, one_d, legacy_1d = (
        _run_strip(RunIntent(processing_mode=value, output_mode=""))
        for value in ("2D Viewer", "Image Viewer", "1D Viewer", "XYE Viewer"))
    assert (viewer.mode, viewer.ready, viewer.run_enabled, viewer.readiness) == (
        "2D Viewer", True, True, "Ready · 2D Viewer")
    assert (legacy.mode, one_d.mode, one_d.ready, one_d.run_enabled,
            legacy_1d.mode, legacy_1d.ready) == (
        "2D Viewer", "1D Viewer", True, True, "1D Viewer", True)
    assert "Image Viewer" not in legacy.modes
    store = RunIntentStore(RunIntent())
    snapshot = store.snapshot()
    candidate = snapshot.thaw()
    candidate.processing_mode = "2D Viewer"
    snapshot = store.commit(candidate, expected_revision=snapshot.revision).snapshot
    controls = project_controls(snapshot, None, RunPhase.IDLE, advanced_editor_available=True)
    one_d_candidate = snapshot.thaw()
    one_d_candidate.processing_mode = "1D Viewer"
    one_d_snapshot = store.commit(
        one_d_candidate, expected_revision=snapshot.revision).snapshot
    one_d_controls = project_controls(
        one_d_snapshot, None, RunPhase.IDLE, advanced_editor_available=True)
    assert (snapshot.thaw().processing_mode, Tool.IMAGE_VIEWER.value, Tool.XYE_VIEWER.value, Mode.IMAGE_VIEWER.value, Mode.XYE_VIEWER.value, ProcessingPage.VIEWER.value) == (
        "2D Viewer", "image_viewer", "xye_viewer", "image_viewer", "xye_viewer", "viewer")
    assert controls.processing_page is ProcessingPage.VIEWER
    assert all(not field.enabled for field in controls.fields)
    assert all(not action.enabled for actions in controls.section_actions.values()
               for action in actions)
    assert one_d_controls.processing_page is ProcessingPage.VIEWER
    assert all(not field.enabled for field in one_d_controls.fields)
    assert {field.reason for field in one_d_controls.fields} == {"1D Viewer has no acquisition authority."}
    assert all(not action.enabled for actions in one_d_controls.section_actions.values()
               for action in actions)
    assert RUN_MODE_CHOICES == _EXPECTED_MODES
    assert UNOWNED_RUN_MODE_REASONS == _EXPECTED_DISABLED_REASONS


def test_projection_owns_modes_and_refuses_run_readiness_for_unowned_mode() -> None:
    native = _run_strip(_configured_intent("Int 1D"), executor_available=True)
    unsupported = _run_strip(
        _configured_intent("Int 1D (XYE)"), executor_available=True)

    assert native.modes == RUN_MODE_CHOICES
    assert native.disabled_modes == UNOWNED_RUN_MODE_REASONS
    assert native.ready
    assert native.run_enabled
    assert native.readiness == "Ready · Int 1D"
    counted = _run_strip(_configured_intent("Int 2D"), executor_available=True,
                         source_count=651)
    directory = _run_strip(
        _configured_intent("Int 2D"), executor_available=True, source_count=8,
        source_count_is_files=True, source_count_includes_immediate=True)
    assert counted.readiness == "Ready · Int 2D · 651 frames"
    assert directory.readiness == (
        "Ready · Int 2D · 8 files (folder + 1 level)"
    )
    assert _configured_intent("Int 1D").freeze().skip_2d is True
    assert _configured_intent("Int 2D").freeze().skip_2d is False

    assert unsupported.modes == RUN_MODE_CHOICES
    assert unsupported.disabled_modes == UNOWNED_RUN_MODE_REASONS
    assert unsupported.mode == "Int 1D (XYE)"
    assert unsupported.ready
    assert unsupported.run_enabled
    assert unsupported.readiness == "Ready · Int 1D (XYE)"


def test_retired_analysis_mode_remains_visible_and_fail_loud() -> None:
    expected = {
        "Stitch 1D": "open Analysis > Stitching",
        "Stitch 2D": "open Analysis > Stitching",
        "RSM": "open Analysis > Reciprocal Space Map",
    }
    for mode, guidance in expected.items():
        projected = _run_strip(
            _configured_intent(mode), executor_available=True,
        )
        assert projected.mode == mode
        assert projected.modes == (*RUN_MODE_CHOICES, mode)
        assert dict(projected.disabled_modes) == {
            mode: projected.readiness,
        }
        assert guidance in projected.readiness
        assert not projected.ready
        assert not projected.run_enabled


def test_directory_strip_qualifies_paused_and_failed_file_progress() -> None:
    base = make_shell_projection(plot_mode="Single")

    def project(
        phase: RunPhase,
        progress: ProgressProjection,
        *,
        start_permitted: bool = True,
        start_blocker: str = "",
    ):
        return ContextProjection().build_shell(
            revision=1,
            controls=base.controls,
            controls_readiness=base.controls_readiness,
            phase=phase,
            intent=_configured_intent("Int 2D"),
            contexts=(),
            selection=None,
            navigation=base.navigation,
            payloads=(),
            resident_frames=frozenset(),
            progress=progress,
            preferences=ScientificPreferences(plot_mode="Single"),
            browser_directory="",
            date_sorted=False,
            auto_last=True,
            executor_available=True,
            start_permitted=start_permitted,
            start_blocker=start_blocker,
            notice="",
        )

    files = DirectoryFileProgress(2, 1, 3, 6)
    paused = project(
        RunPhase.PAUSED,
        ProgressProjection(directory_files=files),
    )
    failed = project(
        RunPhase.FAILED,
        ProgressProjection(
            detail="reader failed",
            directory_files=files,
            terminal=True,
        ),
    )
    cleanup_blocked = project(
        RunPhase.FAILED,
        ProgressProjection(
            detail="reader failed",
            directory_files=files,
            terminal=True,
        ),
        start_permitted=False,
        start_blocker="Cleanup remains pending",
    )

    assert paused.run.readiness == (
        "Paused · 2 processed · 1 skipped · 3 pending · 6 discovered"
    )
    assert failed.run.readiness == (
        "Failed · 2 processed · 1 skipped · 3 pending · 6 discovered"
    )
    assert cleanup_blocked.run.readiness == "Cleanup remains pending"


def test_native_processing_mode_immediately_owns_mounted_center_layout() -> None:
    base = make_shell_projection(plot_mode="Single")

    def project(mode: str, revision: int):
        return ContextProjection().build_shell(
            revision=revision,
            controls=base.controls,
            controls_readiness=base.controls_readiness,
            phase=RunPhase.IDLE,
            intent=_configured_intent(mode),
            contexts=(),
            selection=None,
            navigation=base.navigation,
            payloads=(),
            resident_frames=frozenset(),
            progress=base.progress,
            preferences=ScientificPreferences(plot_mode="Single"),
            browser_directory="",
            date_sorted=False,
            auto_last=True,
            executor_available=True,
            start_permitted=True,
            start_blocker="",
            notice="",
        )

    full = project("Int 2D", 1)
    one_d = project("Int 1D", 2)
    xye = project("Int 1D (XYE)", 3)
    restored = project("Int 2D", 4)
    assert full.scientific.processing_mode == "Int 2D"
    assert one_d.scientific.processing_mode == "Int 1D"
    assert xye.run.mode == "Int 1D (XYE)"
    assert xye.scientific.processing_mode == "Int 1D"
    assert restored.scientific.processing_mode == "Int 2D"


def test_xye_output_uses_the_existing_int1d_trace_and_center_layout(
    monkeypatch,
) -> None:
    from tests.xdart.scattering.test_e3_context_contract import _view
    from tests.xdart.scattering.test_p2a1_viewer_context_page import (
        _fake_scientific,
    )
    from xdart.gui.tabs.scattering import shell_projection
    from xdart.gui.tabs.scattering.display_values import (
        DisplayFrameKey,
        StandardDisplayPayload,
    )
    from xdart.gui.tabs.scattering.events import RunIdentity
    from xdart.gui.tabs.scattering.scientific_view import ScientificView
    from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection

    identity = RunIdentity(8, "xye-presentation")
    frame = DisplayFrameKey(identity, "run.xye", "/out/xye.nxs", 1, 1)
    view = _view(1, 2.0)
    payload = StandardDisplayPayload(
        0, frame, "XYE frame 1", view, measurement_mode="Standard",
    )
    navigation = FrameNavigationProjection((frame,), frame, (frame,))
    base = make_shell_projection(plot_mode="Single")
    allow_cake: list[bool] = []
    real_trace_projection = shell_projection.trace_projection

    def observe_trace(*args, **kwargs):
        allow_cake.append(kwargs["allow_cake"])
        return real_trace_projection(*args, **kwargs)

    monkeypatch.setattr(
        shell_projection, "trace_projection", observe_trace,
    )
    intent = _configured_intent("Int 1D (XYE)")
    projected = ContextProjection().build_shell(
        revision=4,
        controls=base.controls,
        controls_readiness=base.controls_readiness,
        phase=RunPhase.IDLE,
        intent=intent,
        contexts=(),
        selection=None,
        navigation=navigation,
        payloads=(payload,),
        resident_frames=frozenset((frame,)),
        progress=base.progress,
        preferences=ScientificPreferences(plot_mode="Single"),
        browser_directory="",
        date_sorted=False,
        auto_last=True,
        executor_available=True,
        start_permitted=True,
        start_blocker="",
        notice="",
    )

    assert projected.run.mode == "Int 1D (XYE)"
    assert projected.run.readiness == "Ready · Int 1D (XYE)"
    assert projected.scientific.processing_mode == "Int 1D"
    assert len(projected.scientific.traces) == 1
    assert projected.scientific.traces[0].intensity is view.intensity_1d
    assert allow_cake and set(allow_cake) == {False}

    mounted = _fake_scientific()
    ScientificView._apply_processing_layout(
        mounted, projected.scientific.processing_mode,
    )
    assert mounted.image_splitter.hidden
    assert not mounted.raw_popup_button.hidden
    assert mounted.detector_controls.hidden

    controls = project_controls(
        RunIntentStore(intent).snapshot(), None, RunPhase.IDLE,
    )
    paths = {field.path for field in controls.fields}
    assert ("Int1D", "axis") in paths
    assert ("Mask", "Threshold") in paths
    assert ("Signal", "series_average") not in paths


def test_viewer_1d_authority_identifier_delta_is_frozen() -> None:
    from tests.xdart.scattering.test_p2a1_viewer_context_page import _ast_facts
    root, parent, identifiers, baseline = Path(__file__).parents[3], "bf999b35184ba42229da6d545478b7270f1ac0b1", Counter(), Counter()
    paths = tuple("src/xdart/gui/tabs/scattering/" + name for name in ("context_controller.py", "context_runtime.py",
        "context_projection.py", "controls_projection.py", "run_mode_projection.py", "page.py", "scientific_view.py"))
    for path in paths:
        _, _, current = _ast_facts((root / path).read_text())
        source = subprocess.check_output(("git", "-C", str(root), "show", f"{parent}:{path}"), text=True)
        _, _, prior = _ast_facts(source)
        identifiers.update(current); baseline.update(prior)
    terms = ("target", "port", "provider", "generation", "worker", "thread", "timer", "queue", "scheduler", "cache", "watcher", "store", "lease", "owner",
        "holder", "borrow", "claim", "custody", "authority", "lock", "transport", "writer", "output", "durability", "accounting", "calibration",
        "mask", "integration", "rsm", "descriptor", "archive", "parser", "mmap", "callback", "resource")
    identifiers.subtract(baseline)
    delta = {name: count for name, count in identifiers.items()
             if count and any(term in name.lower() for term in terms)}
    assert delta == {"HydrationOwner": 1, "_OneDViewerOwner": 1, "_display_generation": 7, "_ensure_timer": 2, "_release_browse_for_viewer": 1,
        "_release_viewer_1d_holder": 2, "_viewer_1d_provider": 1, "_viewer_2d_lock": 14, "_viewer_2d_provider": 1, "admission_generation": 1,
        "admitted_provider_identity": 3, "blocked_cleanup_token": 2, "borrow": 3, "display_generation": 2, "generation": 33, "holder": 33, "owner": 135,
        "owner_holder": 1, "owner_identity": 6, "owner_request_claim": 6, "port": 1, "presentation_generation": 1, "provider": 46, "publication_store": 1, "release": 1, "retry_blocked_cleanup": 2, "retry_holder": 3, "transport_token": 1, "viewer_1d_owner": 4}
