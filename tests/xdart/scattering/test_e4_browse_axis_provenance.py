from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadStatus,
)
from xdart.gui.tabs.scattering.context_projection import (
    ContextProjection,
    ProjectionRequest,
)
from xdart.gui.tabs.scattering.display_values import DisplayFrameKey
from xdart.gui.tabs.scattering.events import RunIdentity
from xdart.gui.tabs.scattering.shell_projection import (
    ScientificPreferences,
    build_scientific_projection,
)
from xdart.gui.tabs.scattering.shell_values import FrameNavigationProjection
from xdart.modules.display_context import (
    ContextKind,
    DisplaySelection,
    new_context_token,
)
from xrd_tools.core import IntegrationResult1D, IntegrationResult2D
from xrd_tools.reduction import NexusSink
from xrd_tools.reduction.core import Frame, FrameReduction, Scan

from tests.xdart.scattering.test_e3_context_contract import _browse


def _project_browse(context, *, identity: RunIdentity | None = None):
    identity = identity or RunIdentity(1, "browse-wavelength")
    selection = DisplaySelection.for_context(context, 3)
    frame = DisplayFrameKey(
        identity,
        context.scan_key,
        context.requested_path,
        1,
        1,
    )
    # This oracle exercises the retained 1-D trace/wavelength projection; the
    # independently hydrated current-frame cake is not part of its contract.
    request = ProjectionRequest(identity, selection, frame, False)
    return ContextProjection().project(
        context,
        request,
        selection,
        identity,
        {id(frame): frame},
    )


def test_processed_file_round_trip_preserves_wavelength_and_converts_trace(
    tmp_path: Path, monkeypatch,
) -> None:
    import tifffile
    from pyqtgraph.Qt import QtWidgets
    from tests.xdart.scattering.test_p3_experiment_operation_composition import _page
    from test_browse_selected_slices import _wait, _ready, _close
    path = tmp_path / "processed.nexus"
    raw = tmp_path / "raw.tif"
    image = np.arange(16, dtype=float).reshape(4, 4)
    tifffile.imwrite(raw, image)
    frame = Frame(index=1, image=image, source_path=str(raw), source_frame_index=0)
    scan = Scan(
        name="persisted-wavelength",
        frames=[frame],
        wavelength=0.7293,
    )
    q = np.linspace(0.0, 5.0, 8)
    sink = NexusSink(path=path, overwrite=True)
    sink.begin(scan, plan=None)
    sink.write(
        frame,
        FrameReduction(
            frame_index=1,
            result_1d=IntegrationResult1D(
                radial=q,
                intensity=np.linspace(1.0, 2.0, q.size),
                unit="q_A^-1",
            ),
            result_2d=IntegrationResult2D(
                radial=q,
                azimuthal=np.linspace(-90.0, 90.0, 4),
                intensity=np.ones((q.size, 4)),
                unit="q_A^-1",
                azimuthal_unit="chi_deg",
            ),
        ),
    )
    sink.finish(result=None)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    page, _ = _page(tmp_path, monkeypatch)
    projection = None
    try:
        controller = page._context_controller
        controller.begin_browse(str(path))
        assert _wait(app, controller.poll_browse, page).status is BrowseLoadStatus.READY
        assert json.loads(controller.browse_context.calibration_identity)["wavelength_m"] \
            == pytest.approx(0.7293e-10)
        _wait(app, lambda: _ready(page, 1), page)
        axis = page._shell.scientific.plot_axis
        axis.setCurrentIndex(axis.findData("2theta"))
        projection = _wait(app, lambda: _ready(page, 1), page)
        assert projection.plot_axis == "2theta"
        assert projection.traces[0].axis.unit == "2th_deg"
        np.testing.assert_allclose(projection.traces[0].axis.values,
            2.0 * np.rad2deg(np.arcsin(q * 0.7293 / (4.0 * np.pi))))
    finally:
        projection = None
        _close(page, app)


@pytest.mark.parametrize(
    "calibration",
    (
        "",
        "not-json",
        '{"poni_file":"/calibration/a.poni"}',
        '{"wavelength_m":"0.7293e-10"}',
        '{"wavelength_m":-1.0}',
        '{"wavelength_m":NaN}',
    ),
)
def test_browse_projection_refuses_absent_or_malformed_wavelength_provenance(
    calibration: str,
) -> None:
    token = new_context_token(ContextKind.BROWSE)
    _, context = _browse(token, 1)
    context.stamp_provenance(calibration=calibration)

    payload = _project_browse(context)

    assert payload is not None
    assert payload.wavelength_m is None
