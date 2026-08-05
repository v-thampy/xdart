from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering import shell_widgets
from xdart.gui.tabs.scattering.adapters import source as source_adapter_module
from xdart.gui.tabs.scattering.contracts import (
    SourceCountScope,
    SourceObservation,
    SourceObservationRequest,
    SourceObservationStatus,
)
from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.controls_projection import project_controls
from xdart.gui.tabs.scattering.controls_readiness import (
    ControlsReadinessProjection,
)
from xdart.gui.tabs.scattering.shell_widgets import (
    experiment_header_projection,
    processing_header_projection,
    project_header_projection,
    source_header_projection,
)
from xdart.gui.tabs.scattering.source_view import SourceStatusView
from xdart.gui.tabs.scattering.state_machine import RunPhase
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.tabs.static_scan.ui.controls_panel_v2 import ControlsPanelV2
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.readiness import (
    BoundControlState,
    ControlPanelRenderState,
)
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    image_series_spec,
    single_image_spec,
)

from tests.xdart.scattering._e2sd_support import write_poni
from tests.xdart.scattering.e3_shell_support import make_shell_projection


def _qapp() -> QtWidgets.QApplication:
    return (
        QtWidgets.QApplication.instance()
        or QtWidgets.QApplication([])
    )


def _dispose(widget: QtWidgets.QWidget) -> None:
    widget.close()


def _controls_readiness(
    state: ControlPanelRenderState,
) -> ControlsReadinessProjection:
    return ControlsReadinessProjection(
        project_header_projection(state),
        experiment_header_projection(state),
        processing_header_projection(state),
    )


def _directory_observation(
    *,
    count: int,
    status: SourceObservationStatus = SourceObservationStatus.AVAILABLE,
    exists: bool = True,
    reason: str = "",
    probed: bool = False,
) -> SourceObservation:
    source = DirectorySourceSpec(
        Path("/data/raw"),
        recursive=False,
        suffixes=(".tif",),
    )
    return SourceObservation(
        1,
        0,
        source,
        status,
        "raw",
        exists,
        exists,
        direct_child_count=count if exists else None,
        reason=reason,
        gi_motor_choices=() if probed else None,
    )


def test_source_observation_projects_into_header_not_large_summary_card() -> None:
    _qapp()
    panel = ControlsPanelV2(show_section_numbers=False)
    source = SourceStatusView(panel)
    panel.set_source_widget(source, visible=False)
    try:
        source.render(_directory_observation(count=8, probed=True))

        assert source.layout().count() == 1
        assert source.layout().itemAt(0).widget() is source._choose
        assert panel.source_widget_visible() is False
        assert source.isHidden()
        assert not source._choose.isVisibleTo(panel)
        assert panel.source_card.status.text() == (
            "8 files · Image Directory"
        )
        assert not panel.source_card.valid_marker.isHidden()
        assert panel.source_card.valid_marker.toolTip() == (
            "Source preview found at least one readable candidate."
        )

    finally:
        _dispose(panel)


def test_typed_image_series_counts_its_frozen_members_as_ready_frames(
    tmp_path: Path,
) -> None:
    members = tuple(tmp_path / f"scan_{index:04d}.tif" for index in range(1, 4))
    for member in members:
        member.write_bytes(bytes([member.name.__len__()]))
    source = image_series_spec(members[1])

    observed = FilesystemSourceAdapter().observe(
        SourceObservationRequest(1, 0, source)
    )
    header = source_header_projection(observed)

    assert observed.direct_child_count == 3
    assert observed.candidate_fingerprint
    assert header.text == "3 frames · Image Series"
    assert header.ready
    assert header.detail == "Every frozen image-series member is present."


def test_explicit_single_image_is_not_reclassified_from_member_count(
    tmp_path: Path,
) -> None:
    selected = tmp_path / "single_0001.tif"
    selected.write_bytes(b"frame")
    single = single_image_spec(selected)
    series = image_series_spec(selected)
    adapter = FilesystemSourceAdapter()

    single_header = source_header_projection(adapter.observe(
        SourceObservationRequest(1, 0, single)
    ))
    series_header = source_header_projection(adapter.observe(
        SourceObservationRequest(2, 0, series)
    ))

    assert single_header.text == "1 file · Single Image"
    assert single_header.ready
    assert single_header.detail == "The selected source image is present."
    assert series_header.text == "1 frame · Image Series"
    assert series_header.ready
    assert series_header.detail == (
        "Every frozen image-series member is present."
    )


@pytest.mark.parametrize(
    ("suffix", "kind"),
    ((".h5", SourceKind.EIGER_MASTER), (".nxs", SourceKind.NEXUS_STACK)),
)
def test_container_directory_counts_files_while_explicit_series_counts_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    kind: SourceKind,
) -> None:
    path = tmp_path / f"scan_master{suffix}"
    path.write_bytes(b"container")

    class _Owner:
        @staticmethod
        def probe(candidate: Path) -> object:
            assert candidate == path
            return SimpleNamespace(
                state=ProbeState.READY,
                descriptor=SimpleNamespace(frame_count=651),
            )

    monkeypatch.setattr(
        source_adapter_module, "candidate_owner", lambda candidate: _Owner()
    )
    adapter = FilesystemSourceAdapter()
    directory = DirectorySourceSpec(
        tmp_path,
        recursive=False,
        suffixes=(suffix,),
    )
    source = SourceSpec(str(path), kind)
    directory_observed = adapter.observe(
        SourceObservationRequest(1, 0, directory)
    )
    series_observed = adapter.observe(
        SourceObservationRequest(2, 0, source)
    )

    assert directory_observed.direct_child_count == 1
    assert source_header_projection(directory_observed).text == (
        "1 file · Image Directory"
    )
    assert series_observed.direct_child_count == 651
    assert source_header_projection(series_observed).text == (
        "651 frames · Image Series"
    )


@pytest.mark.parametrize("suffix", (".h5", ".nxs"))
def test_recursive_container_count_includes_only_immediate_subfolders_without_probing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
) -> None:
    (tmp_path / f"direct{suffix}").write_bytes(b"one")
    immediate = tmp_path / "immediate"
    deeper = immediate / "deeper"
    immediate.mkdir()
    deeper.mkdir()
    (immediate / f"nested{suffix.upper()}").write_bytes(b"two")
    (deeper / f"too_deep{suffix}").write_bytes(b"three")
    (immediate / "ignore.txt").write_bytes(b"sidecar")
    monkeypatch.setattr(
        source_adapter_module,
        "candidate_owner",
        lambda _candidate: pytest.fail("directory observation opened a container"),
    )
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(suffix,),
    )
    observed = FilesystemSourceAdapter().observe(
        SourceObservationRequest(1, 0, source)
    )
    direct = FilesystemSourceAdapter().observe(
        SourceObservationRequest(
            2,
            0,
            DirectorySourceSpec(tmp_path, recursive=False, suffixes=(suffix,)),
        )
    )

    assert observed.direct_child_count == 1
    assert observed.one_level_file_count == 2
    assert observed.observed_file_count == 2
    assert (
        observed.file_count_scope
        is SourceCountScope.SELECTED_PLUS_IMMEDIATE
    )
    assert observed.candidate_fingerprint == direct.candidate_fingerprint
    assert source_header_projection(observed).text == (
        "2 files (folder + 1 level) · Image Directory"
    )


def test_recursive_tiff_count_includes_only_immediate_subfolders(
    tmp_path: Path,
) -> None:
    (tmp_path / "direct.tif").write_bytes(b"direct")
    (tmp_path / "ignore.txt").write_bytes(b"sidecar")
    first = tmp_path / "first"
    second = tmp_path / "second"
    grandchild = first / "deeper"
    first.mkdir()
    second.mkdir()
    grandchild.mkdir()
    (first / "child_1.tif").write_bytes(b"child-1")
    (first / "child_2.tiff").write_bytes(b"child-2")
    (second / "ignore.edf").write_bytes(b"not-a-tiff")
    (grandchild / "too_deep.tif").write_bytes(b"deeper")
    source = DirectorySourceSpec(
        tmp_path,
        recursive=True,
        suffixes=(".tif", ".tiff"),
    )

    observed = FilesystemSourceAdapter().observe(
        SourceObservationRequest(1, 0, source)
    )
    header = source_header_projection(observed)

    assert observed.direct_child_count == 1
    assert observed.one_level_file_count == 3
    assert observed.observed_file_count == 3
    assert (
        observed.file_count_scope
        is SourceCountScope.SELECTED_PLUS_IMMEDIATE
    )
    assert header.text == "3 files (folder + 1 level) · Image Directory"
    assert (
        "Count includes the selected folder and immediate subfolders."
        in header.detail
    )
    assert "Deeper subfolders are evaluated during Run." in header.detail


def test_source_header_never_claims_unobserved_or_empty_directory_ready() -> None:
    _qapp()
    panel = ControlsPanelV2(show_section_numbers=False)
    source = SourceStatusView(panel)
    panel.set_source_widget(source)
    try:
        source.show_checking("raw")
        assert panel.source_card.status.text() == "checking…"
        assert panel.source_card.valid_marker.isHidden()

        source.render(_directory_observation(count=0))
        assert panel.source_card.status.text() == (
            "0 files · Image Directory"
        )
        assert panel.source_card.valid_marker.isHidden()

        source.render(_directory_observation(count=8))
        assert panel.source_card.status.text() == (
            "8 files · Image Directory"
        )
        assert panel.source_card.valid_marker.isHidden()
        assert panel.source_card.status.toolTip() == (
            "Matching names are present; content readiness has not "
            "been observed."
        )

        source.render(
            _directory_observation(
                count=0,
                status=SourceObservationStatus.UNAVAILABLE,
                exists=False,
                reason="Directory metadata is unavailable.",
            )
        )
        assert panel.source_card.status.text() == "unavailable"
        assert panel.source_card.status.toolTip() == (
            "Directory metadata is unavailable."
        )
        assert panel.source_card.valid_marker.isHidden()
    finally:
        _dispose(panel)


def test_recursive_nested_only_beyond_shallow_preview_is_not_ready(
) -> None:
    source = DirectorySourceSpec(
        Path("/data/raw"),
        recursive=True,
        suffixes=(".tif",),
    )
    observed = SourceObservation(
        1,
        0,
        source,
        SourceObservationStatus.AVAILABLE,
        "raw",
        True,
        True,
        direct_child_count=0,
        subdirectories_deferred=True,
        candidate_fingerprint="direct-children-empty",
        one_level_file_count=0,
        file_count_scope=SourceCountScope.SELECTED_PLUS_IMMEDIATE,
    )

    header = source_header_projection(observed)

    assert header.text == (
        "0 files (folder + 1 level) · Image Directory"
    )
    assert not header.ready
    assert "Deeper subfolders are evaluated during Run." in header.detail


def test_project_header_requires_existing_project_and_creatable_save_target(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    save_path = project / "xdart_processed_data"
    intent = RunIntent(
        project_root=str(project),
        save_path=str(save_path),
    )

    state = project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    )
    header = project_header_projection(state)

    assert not save_path.exists()
    assert header.ready
    assert header.detail == (
        "Project folder and save directory target are ready."
    )

    intent.project_root = str(tmp_path / "missing-project")
    missing_project = project_header_projection(project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    ))
    assert not missing_project.ready

    intent.project_root = str(project)
    blocker = project / "not-a-directory"
    blocker.write_text("occupied")
    intent.save_path = str(blocker / "processed")
    blocked_save = project_header_projection(project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    ))
    assert not blocked_save.ready

    dotted_directory = project / "processed.v1"
    dotted_directory.mkdir()
    intent.save_path = str(dotted_directory)
    dotted_save = project_header_projection(project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    ))
    assert dotted_save.ready

    intent.save_path = str(project / "legacy-output.nxs")
    file_shaped_save = project_header_projection(project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    ))
    assert not file_shaped_save.ready

    intent.save_path = str(project / "new-output.nexus")
    assert not project_header_projection(project_controls(
        RunIntentStore(intent).snapshot(), None, RunPhase.IDLE,
    )).ready


def test_experiment_header_requires_readable_parseable_poni(
    tmp_path: Path,
) -> None:
    poni = tmp_path / "detector.poni"
    write_poni(poni)
    intent = RunIntent(poni_file=str(poni))

    valid = experiment_header_projection(project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    ))
    assert valid.ready
    assert valid.detail == "Detector calibration is readable and parseable."

    poni.write_text(":\n  - [")
    malformed = experiment_header_projection(project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    ))
    assert not malformed.ready

    intent.poni_file = str(tmp_path / "missing.poni")
    missing = experiment_header_projection(project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    ))
    assert not missing.ready


def test_shell_reapplies_observed_source_and_typed_processing_readiness() -> None:
    _qapp()
    shell = ScatteringWorkspaceShell()
    source = SourceStatusView(shell.controls)
    shell.controls.set_source_widget(source)
    try:
        source.render(_directory_observation(count=8, probed=True))
        state = make_shell_projection(revision=1)
        shell.apply_state(state)

        assert shell.controls.source_card.status.text() == (
            "8 files · Image Directory"
        )
        assert not shell.controls.source_card.valid_marker.isHidden()
        assert shell.controls.processing_card.status.text() == "int 2d"
        assert not shell.controls.processing_card.valid_marker.isHidden()
        assert shell.controls.processing_card.valid_marker.toolTip() == (
            "Typed processing inputs are configured."
        )

        controls = state.controls
        bound = controls.bound_controls
        assert bound is not None
        incomplete = BoundControlState(tuple(
            field
            for field in bound.fields
            if field.path != ("Int2D", "azim_points")
        ))
        incomplete_controls = ControlPanelRenderState(
            controls.profile,
            incomplete,
        )
        shell.apply_state(replace(
            state,
            revision=2,
            controls=incomplete_controls,
            controls_readiness=_controls_readiness(
                incomplete_controls
            ),
        ))

        assert shell.controls.processing_card.status.text() == "int 2d"
        assert shell.controls.processing_card.valid_marker.isHidden()
        assert not shell.controls.source_card.valid_marker.isHidden()
    finally:
        _dispose(shell)


def test_shell_applies_project_and_experiment_readiness_without_changing_other_markers(
    tmp_path: Path,
) -> None:
    _qapp()
    project = tmp_path / "project"
    project.mkdir()
    poni = project / "detector.poni"
    write_poni(poni)
    controls = project_controls(
        RunIntentStore(RunIntent(
            project_root=str(project),
            save_path=str(project / "xdart_processed_data"),
            poni_file=str(poni),
        )).snapshot(),
        None,
        RunPhase.IDLE,
    )
    state = replace(
        make_shell_projection(revision=1),
        controls=controls,
        controls_readiness=_controls_readiness(controls),
    )
    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)

        assert not shell.controls.project_card.valid_marker.isHidden()
        assert shell.controls.project_card.valid_marker.toolTip() == (
            "Project folder and save directory target are ready."
        )
        assert not shell.controls.experiment_card.valid_marker.isHidden()
        assert shell.controls.experiment_card.valid_marker.toolTip() == (
            "Detector calibration is readable and parseable."
        )
        assert shell.controls.source_card.valid_marker.isHidden()
        assert not shell.controls.processing_card.valid_marker.isHidden()

        invalid_controls = project_controls(
            RunIntentStore(RunIntent(
                project_root=str(project / "missing"),
                save_path=str(project / "xdart_processed_data"),
                poni_file=str(project / "missing.poni"),
            )).snapshot(),
            None,
            RunPhase.IDLE,
        )
        shell.apply_state(replace(
            state,
            revision=2,
            controls=invalid_controls,
            controls_readiness=_controls_readiness(invalid_controls),
        ))

        assert shell.controls.project_card.valid_marker.isHidden()
        assert shell.controls.experiment_card.valid_marker.isHidden()
        assert shell.controls.source_card.valid_marker.isHidden()
        assert not shell.controls.processing_card.valid_marker.isHidden()
    finally:
        _dispose(shell)


def test_passive_shell_reconcile_consumes_frozen_readiness_without_reprobing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A heartbeat repaint must never turn into synchronous filesystem I/O."""

    _qapp()
    project = tmp_path / "project"
    project.mkdir()
    poni = project / "detector.poni"
    write_poni(poni)
    controls = project_controls(
        RunIntentStore(RunIntent(
            project_root=str(project),
            save_path=str(project / "xdart_processed_data"),
            poni_file=str(poni),
        )).snapshot(),
        None,
        RunPhase.IDLE,
    )
    readiness = _controls_readiness(controls)
    state = replace(
        make_shell_projection(revision=1),
        controls=controls,
        controls_readiness=readiness,
    )

    def _unexpected_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("passive shell reconcile re-probed readiness")

    monkeypatch.setattr(
        shell_widgets,
        "project_header_projection",
        _unexpected_probe,
    )
    monkeypatch.setattr(
        shell_widgets,
        "experiment_header_projection",
        _unexpected_probe,
    )
    monkeypatch.setattr(
        shell_widgets,
        "processing_header_projection",
        _unexpected_probe,
    )

    shell = ScatteringWorkspaceShell()
    try:
        shell.apply_state(state)
        shell.apply_state(replace(state, revision=2))

        assert not shell.controls.project_card.valid_marker.isHidden()
        assert not shell.controls.experiment_card.valid_marker.isHidden()
        assert not shell.controls.processing_card.valid_marker.isHidden()
    finally:
        _dispose(shell)


def test_detector_header_reports_mounted_poni_detector_facts(
    tmp_path: Path,
) -> None:
    _qapp()
    poni = tmp_path / "LaB6_detector.poni"
    poni.write_text(
        "Detector: RayonixMx225\n"
        "Distance: 0.17939120815373186\n"
        "Poni1: 0.1\n"
        "Poni2: 0.2\n"
    )
    intent = RunIntent(
        source_spec=DirectorySourceSpec(
            Path("/data/raw"),
            suffixes=(".tif",),
        ),
        poni_file=str(poni),
        save_path="/processed",
        output_mode="Overwrite",
    )
    state = project_controls(
        RunIntentStore(intent).snapshot(),
        None,
        RunPhase.IDLE,
    )
    panel = ControlsPanelV2(
        experiment_title="Configuration",
        show_section_numbers=False,
    )
    try:
        assert state.profile.detector_summary == (
            "RayonixMx225 · 179.4mm · fitted"
        )
        panel.set_state(state)
        detector = next(
            child
            for child in panel.experiment_card.findChildren(
                QtWidgets.QFrame
            )
            if (
                hasattr(child, "title")
                and child.title.text() == "Detector"
            )
        )
        assert detector.status.text() == (
            "RayonixMx225 · 179.4mm · fitted"
        )
    finally:
        _dispose(panel)
