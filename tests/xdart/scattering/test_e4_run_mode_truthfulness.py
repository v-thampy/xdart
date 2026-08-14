from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
    "Stitch 1D",
    "Stitch 2D",
    "2D Viewer",
    "1D Viewer",
)
_EXPECTED_DISABLED_REASONS = (
    (
        "Stitch 1D",
        "Stitching has no mounted vNext operation service yet.",
    ),
    (
        "Stitch 2D",
        "Stitching has no mounted vNext operation service yet.",
    ),
    (
        "1D Viewer",
        "1D Viewer has no mounted vNext viewer context yet.",
    ),
)


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
    viewer, legacy, legacy_1d = (
        _run_strip(RunIntent(processing_mode=value, output_mode=""))
        for value in ("2D Viewer", "Image Viewer", "XYE Viewer"))
    assert (viewer.mode, viewer.ready, viewer.run_enabled, viewer.readiness) == (
        "2D Viewer", True, True, "Ready · 2D Viewer")
    assert (legacy.mode, legacy_1d.mode, legacy_1d.ready) == (
        "2D Viewer", "1D Viewer", False)
    assert "Image Viewer" not in legacy.modes
    store = RunIntentStore(RunIntent())
    snapshot = store.snapshot()
    candidate = snapshot.thaw()
    candidate.processing_mode = "2D Viewer"
    snapshot = store.commit(candidate, expected_revision=snapshot.revision).snapshot
    controls = project_controls(snapshot, None, RunPhase.IDLE, advanced_editor_available=True)
    assert (snapshot.thaw().processing_mode, Tool.IMAGE_VIEWER.value, Tool.XYE_VIEWER.value, Mode.IMAGE_VIEWER.value, Mode.XYE_VIEWER.value, ProcessingPage.VIEWER.value) == (
        "2D Viewer", "image_viewer", "xye_viewer", "image_viewer", "xye_viewer", "viewer")
    assert controls.profile.processing_page is ProcessingPage.VIEWER
    assert all(not field.enabled for field in controls.bound_controls.fields)
    assert all(not action.enabled for actions in controls.profile.section_actions.values()
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
    restored = project("Int 2D", 3)
    assert full.scientific.processing_mode == "Int 2D"
    assert one_d.scientific.processing_mode == "Int 1D"
    assert restored.scientific.processing_mode == "Int 2D"
