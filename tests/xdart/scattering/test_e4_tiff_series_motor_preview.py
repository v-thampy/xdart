from __future__ import annotations

from pathlib import Path
import time

import fabio
import numpy as np
from pyqtgraph.Qt import QtWidgets

from xdart.gui.tabs.scattering.adapters.source import FilesystemSourceAdapter
from xdart.gui.tabs.scattering.contracts import SourceObservationRequest
from xdart.gui.tabs.scattering.controls_projection import GI_MOTOR
from xdart.gui.tabs.scattering.coordinator import ScatteringCoordinator
from xdart.gui.tabs.scattering.page import ScatteringWorkspace
from xdart.gui.tabs.scattering.workspace_shell import ScatteringWorkspaceShell
from xdart.gui.widgets.controls_panel import FormRow
from xrd_tools.session.intent_store import RunIntentStore
from xrd_tools.session.run_configuration import GIIntent, RunIntent
from xrd_tools.sources.selection import image_series_spec


def _write_tiff(path: Path, value: int) -> None:
    fabio.tifimage.TifImage(
        data=np.full((4, 4), value, dtype=np.uint16)
    ).write(str(path))


def _write_sidecar(path: Path, *lines: str) -> None:
    path.with_suffix(".txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _series(tmp_path: Path, second_metadata: tuple[str, ...] | None = None):
    first = tmp_path / "scan_0001.tif"
    second = tmp_path / "scan_0002.tif"
    _write_tiff(first, 1)
    _write_tiff(second, 2)
    _write_sidecar(first, "th=0.15", "exposure=1", "sequence=1")
    if second_metadata is not None:
        _write_sidecar(second, *second_metadata)
    return image_series_spec(first)


def _motor_field(page: ScatteringWorkspace):
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    projection = shell.controls.projection
    assert projection is not None
    return next(
        field
        for field in projection.fields
        if field.path == GI_MOTOR
    )


def _visible_motor_choices(page: ScatteringWorkspace) -> tuple[str, ...]:
    shell = page.findChild(ScatteringWorkspaceShell)
    assert shell is not None
    row = next(
        row
        for row in shell.controls.findChildren(FormRow)
        if row.path == GI_MOTOR
    )
    return tuple(
        row.editor.itemText(index)
        for index in range(row.editor.count())
    )


def _wait(app: QtWidgets.QApplication, predicate) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("TIFF Image Series motor preview did not settle")


def test_tiff_series_preview_intersects_every_frozen_frame(tmp_path: Path) -> None:
    source = _series(
        tmp_path,
        ("th=0.25", "exposure=2", "second_only=3"),
    )
    adapter = FilesystemSourceAdapter()
    request = SourceObservationRequest(1, 0, source)

    passive = adapter.observe(request)
    preview = adapter.preview_motors(request)

    assert preview.candidate_fingerprint == passive.candidate_fingerprint
    assert preview.gi_motor_choices == ("th", "exposure")


def test_tiff_series_preview_never_borrows_a_missing_motor(tmp_path: Path) -> None:
    source = _series(
        tmp_path,
        ("exposure=2", "sequence=2", "eta=0.25"),
    )
    adapter = FilesystemSourceAdapter()

    preview = adapter.preview_motors(
        SourceObservationRequest(1, 0, source)
    )

    assert preview.gi_motor_choices == ("exposure", "sequence")
    assert "th" not in preview.gi_motor_choices


def test_tiff_series_selection_updates_mounted_motor_dropdown(
    tmp_path: Path,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    source = _series(
        tmp_path,
        ("th=0.25", "exposure=2", "sequence=2"),
    )
    page = ScatteringWorkspace(
        intents=RunIntentStore(RunIntent(
            source_spec=source,
            gi=GIIntent(enabled=True),
        )),
        lifecycle=ScatteringCoordinator(),
        sources=FilesystemSourceAdapter(),
    )
    try:
        _wait(
            app,
            lambda: (
                not page._source_selection.observing
                and _motor_field(page).choices
                == ("Manual", "th", "exposure", "sequence")
            ),
        )
        assert _visible_motor_choices(page) == (
            "Manual",
            "th",
            "exposure",
            "sequence",
        )
    finally:
        page.close_workspace()
