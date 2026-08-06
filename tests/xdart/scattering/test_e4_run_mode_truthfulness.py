from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from xdart.gui.tabs.scattering.context_projection import ContextProjection
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
from xdart.gui.tabs.scattering.workspace_shell import (
    ScatteringWorkspaceShell,
)
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

from tests.xdart.scattering.e3_shell_support import make_shell_projection


_EXPECTED_MODES = (
    "Int 1D",
    "Int 2D",
    "Int 1D (XYE)",
    "Stitch 1D",
    "Stitch 2D",
    "Image Viewer",
    "XYE Viewer",
)
_EXPECTED_DISABLED_REASONS = (
    (
        "Int 1D (XYE)",
        "XYE output has no mounted vNext sink contract yet.",
    ),
    (
        "Stitch 1D",
        "Stitching has no mounted vNext operation service yet.",
    ),
    (
        "Stitch 2D",
        "Stitching has no mounted vNext operation service yet.",
    ),
    (
        "Image Viewer",
        "Image Viewer has no mounted vNext viewer context yet.",
    ),
    (
        "XYE Viewer",
        "XYE Viewer has no mounted vNext viewer context yet.",
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


def test_mode_menu_lists_future_surfaces_but_disables_unowned_services() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = ScatteringWorkspaceShell()
    state = make_shell_projection(plot_mode="Single")

    try:
        assert RUN_MODE_CHOICES == _EXPECTED_MODES
        assert UNOWNED_RUN_MODE_REASONS == _EXPECTED_DISABLED_REASONS
        shell.apply_state(
            replace(
                state,
                run=replace(
                    state.run,
                    modes=RUN_MODE_CHOICES,
                    disabled_modes=UNOWNED_RUN_MODE_REASONS,
                ),
            )
        )
        combo = shell.run_controls.modeCombo
        assert tuple(
            combo.itemText(index) for index in range(combo.count())
        ) == _EXPECTED_MODES
        reasons = dict(UNOWNED_RUN_MODE_REASONS)
        model = combo.model()
        for row, label in enumerate(_EXPECTED_MODES):
            enabled = bool(
                model.flags(model.index(row, 0))
                & QtCore.Qt.ItemFlag.ItemIsEnabled
            )
            assert enabled is (label not in reasons)
            assert (
                combo.itemData(row, QtCore.Qt.ItemDataRole.ToolTipRole)
                == reasons.get(label)
            )
    finally:
        shell.close()
        app.processEvents()


def test_projection_owns_modes_and_refuses_run_readiness_for_unowned_mode() -> None:
    native = build_run_strip_projection(
        RunPhase.IDLE,
        _configured_intent("Int 1D"),
        executor_available=True,
        start_permitted=True,
        start_blocker="",
    )
    unsupported = build_run_strip_projection(
        RunPhase.IDLE,
        _configured_intent("Int 1D (XYE)"),
        executor_available=True,
        start_permitted=True,
        start_blocker="",
    )

    assert native.modes == RUN_MODE_CHOICES
    assert native.disabled_modes == UNOWNED_RUN_MODE_REASONS
    assert native.ready
    assert native.run_enabled
    assert native.readiness == "Ready · Int 1D"
    counted = build_run_strip_projection(
        RunPhase.IDLE,
        _configured_intent("Int 2D"),
        executor_available=True,
        start_permitted=True,
        start_blocker="",
        source_count=651,
    )
    directory = build_run_strip_projection(
        RunPhase.IDLE,
        _configured_intent("Int 2D"),
        executor_available=True,
        start_permitted=True,
        start_blocker="",
        source_count=8,
        source_count_is_files=True,
        source_count_includes_immediate=True,
    )
    assert counted.readiness == "Ready · Int 2D · 651 frames"
    assert directory.readiness == (
        "Ready · Int 2D · 8 files (folder + 1 level)"
    )
    assert _configured_intent("Int 1D").freeze().skip_2d is True
    assert _configured_intent("Int 2D").freeze().skip_2d is False

    assert unsupported.modes == RUN_MODE_CHOICES
    assert unsupported.disabled_modes == UNOWNED_RUN_MODE_REASONS
    assert unsupported.mode == "Int 1D (XYE)"
    assert not unsupported.ready
    assert not unsupported.run_enabled
    assert unsupported.readiness == dict(UNOWNED_RUN_MODE_REASONS)[
        "Int 1D (XYE)"
    ]


def test_directory_strip_qualifies_paused_and_failed_file_progress() -> None:
    base = make_shell_projection(plot_mode="Single")

    def project(phase: RunPhase, progress: ProgressProjection):
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
            start_permitted=False,
            start_blocker="",
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

    assert paused.run.readiness == (
        "Paused · 2 processed · 1 skipped · 3 pending · 6 discovered"
    )
    assert failed.run.readiness == (
        "Failed · 2 processed · 1 skipped · 3 pending · 6 discovered"
    )


def test_native_processing_mode_immediately_owns_mounted_center_layout() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    shell = ScatteringWorkspaceShell()
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
    shell.resize(1400, 900)
    shell.show()
    try:
        shell.apply_state(full)
        app.processEvents()
        assert not shell.scientific.image_splitter.isHidden()
        assert not shell.scientific.raw.isHidden()
        assert not shell.scientific.cake.isHidden()
        assert not shell.scientific.curve.isHidden()
        assert not shell.scientific.plot_toolbar.isHidden()

        shell.apply_state(one_d)
        app.processEvents()
        assert one_d.scientific.processing_mode == "Int 1D"
        assert shell.scientific.image_splitter.isHidden()
        assert not shell.scientific.curve.isHidden()
        assert not shell.scientific.plot_toolbar.isHidden()
        assert shell.scientific.image_axis.isHidden()
        assert shell.scientific.share_axis.isHidden()
        assert shell.scientific.slice.isHidden()

        shell.apply_state(restored)
        app.processEvents()
        assert restored.scientific.processing_mode == "Int 2D"
        assert not shell.scientific.image_splitter.isHidden()
        assert not shell.scientific.raw.isHidden()
        assert not shell.scientific.cake.isHidden()
        assert not shell.scientific.image_axis.isHidden()
        assert not shell.scientific.share_axis.isHidden()
        assert not shell.scientific.slice.isHidden()
    finally:
        shell.close()
        app.processEvents()
