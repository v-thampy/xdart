from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pytest

from xdart.gui.tabs.scattering.adapters.browse_loader import BrowseLoader
from xdart.gui.tabs.scattering.browse_values import (
    BrowseLoadRequest,
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


def _wait_outcome(
    loader: BrowseLoader,
    request: BrowseLoadRequest,
):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        outcome = loader.poll(request)
        if outcome is not None:
            return outcome
        time.sleep(0.002)
    raise AssertionError("browse loader did not finish")


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
        (frame,),
    )


def test_processed_file_round_trip_preserves_wavelength_and_converts_trace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "processed.nxs"
    frame = Frame(index=1, image=np.arange(16, dtype=float).reshape(4, 4))
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

    loader = BrowseLoader()
    request = BrowseLoadRequest(
        new_context_token(ContextKind.BROWSE),
        1,
        str(path),
    )
    loader.begin(request)
    outcome = _wait_outcome(loader, request)
    assert outcome.status is BrowseLoadStatus.READY
    context = loader.consume(outcome)
    assert context is not None
    assert json.loads(context.calibration_identity)["wavelength_m"] \
        == pytest.approx(0.7293e-10)

    payload = _project_browse(context)
    assert payload is not None
    assert payload.wavelength_m == pytest.approx(0.7293e-10)
    frame_key = payload.frame_key
    navigation = FrameNavigationProjection(
        (frame_key,),
        frame_key,
        (frame_key,),
    )
    projection = build_scientific_projection(
        (payload,),
        navigation,
        frozenset({frame_key}),
        ScientificPreferences(plot_axis="2theta"),
        "",
    )
    assert projection.plot_axis == "2theta"
    assert len(projection.traces) == 1
    assert projection.traces[0].axis.unit == "2th_deg"
    np.testing.assert_allclose(
        projection.traces[0].axis.values,
        2.0 * np.rad2deg(
            np.arcsin(q * 0.7293 / (4.0 * np.pi))
        ),
    )

    loader.release_context(context)
    loader.close()


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
