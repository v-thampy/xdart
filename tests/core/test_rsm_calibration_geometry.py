"""AUD-9 stage 2b C7b: RSM's instrument geometry from a calibration made in the calibration window (the maintainer:
"the idea is to reuse what was developed for stitch, not use the calibration that was done in stitch").

The window saves a pyFAI file and its xrayutilities partner, which describes the image Stitch turns into its own
camera. RSM reads raw images, so it takes the raw camera and maps the beam pixel, pitch and shape back by the
inverse turn. Row E is the oracle for all of it: Stitch's own xrayutilities provider, in the partner's frame,
against RSM's raw-frame map at the same physical pixel.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.rsm_geometry_asset import (
    RSMGeometryAssetRefused,
    _raw_frame_pixel,
    capture_rsm_calibration_geometry,
    lower_rsm_effective_geometry,
    rsm_effective_pixel_q_map,
)
from xrd_tools.analysis.xu_stitch_calibration import authored_surface_calibration
from xrd_tools.core.geometry.camera import CAMERA_AXES, DetectorCamera, stitch_orientation
from xrd_tools.core.geometry.diffractometer import ImageOrientation
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.xu_stitch import XuPowderQProvider, resolve_xu_stitch_effective_geometry

ENERGY = 17000.004
ANGLES = ((0.0, 0.0), (12.5, -3.0), (31.0, 7.25), (-18.0, 21.5))  # (del, nu)


def _write_partner(root: Path, *, rotation: int, cch: tuple[float, float], name: str = "cal",
                   offsets=(1.7, -0.8), shape=(9, 13), motors=None) -> Path:
    from pyFAI.detectors import Detector
    with xu_runtime_session() as session:
        wavelength = float(session.xu.en2lam(ENERGY))
    held = [circle for circle, motor in (motors or {}).items() if motor is None]
    domain = {circle: ([0., 0.] if circle in held else limits)
              for circle, limits in (("del", [-50., 50.]), ("nu", [-35., 35.]))}
    value = authored_surface_calibration(
        detector=Detector(pixel1=172e-6, pixel2=75e-6, max_shape=shape),
        cch1=cch[0], cch2=cch[1], distance_m=.39,
        del_offset_deg=offsets[0], nu_offset_deg=offsets[1], energy_eV=ENERGY,
        xu_wavelength_A=wavelength, source_motors=motors or {"del": "del", "nu": "nu"},
        rotation=rotation, control_domain=domain, domain=domain,
        rms_q=.001, condition=10., n_frames=5, n_points=40,
    )
    partner = root / f"{name}.xu.json"
    partner.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    (root / f"{name}.json").write_text("{}")  # the pyFAI file's place on the Calibration row
    return root / f"{name}.json"


def test_a_pyfai_file_without_its_partner_is_refused(tmp_path):
    (tmp_path / "lonely.json").write_text("{}")
    with pytest.raises(RSMGeometryAssetRefused) as refused:
        capture_rsm_calibration_geometry("lonely.json", project_root=tmp_path)
    assert refused.value.code == "RSM_CALIBRATION_UNSUPPORTED" and "lonely.xu.json" in str(refused.value)


def test_a_single_two_theta_arm_is_not_psic(tmp_path):
    pyfai = _write_partner(tmp_path, rotation=0, cch=(4.0, 6.0), motors={"del": "tth", "nu": None})
    with pytest.raises(RSMGeometryAssetRefused) as refused:
        capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path)
    assert refused.value.code == "RSM_CALIBRATION_UNSUPPORTED" and "single 2θ arm" in str(refused.value)


@pytest.mark.parametrize("camera", [(rows, columns) for rows in CAMERA_AXES for columns in CAMERA_AXES
                                    if rows[0] != columns[0]])
def test_the_beam_pixel_maps_back_to_the_raw_frame_for_every_camera(camera):
    """The inverse turn is the index permutation ``orientation.apply`` performs: every turned pixel's raw
    coordinates, exactly."""
    orientation = stitch_orientation(DetectorCamera(*camera))
    turned = (9, 13)
    raw_shape = (turned[1], turned[0]) if orientation.swaps_axes else turned
    rows, columns = np.indices(raw_shape)
    turned_rows, turned_columns = orientation.apply(rows), orientation.apply(columns)
    for i in range(turned[0]):
        for j in range(turned[1]):
            assert _raw_frame_pixel(orientation, raw_shape, (i, j)) == (turned_rows[i, j], turned_columns[i, j])


def _row_e(tmp_path, rotation, cch):
    pyfai = _write_partner(tmp_path, rotation=rotation, cch=cch)
    receipt = capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path)
    effective = lower_rsm_effective_geometry(receipt)
    assert effective.source == "calibration"
    mapper = rsm_effective_pixel_q_map(effective)
    value = receipt.calibration.projection.value
    acquisition = value["acquisition"]
    offsets = acquisition["detector_offsets_deg"]
    orientation = ImageOrientation(rotation=90 * acquisition["image_rotation_quarter_turns"])
    worst = 0.0
    with xu_runtime_session() as session:
        stitch = XuPowderQProvider(resolve_xu_stitch_effective_geometry(receipt.calibration, session),
                                   acquisition)
        for delta, nu in ANGLES:
            with stitch.frame(nu=nu, del_=delta) as lease:
                reference = np.stack(lease._values)
            raw = mapper.pixel_q([np.zeros(1)] * 4 + [np.array([nu - offsets["nu"]]),
                                                      np.array([delta - offsets["del"]])],
                                 ENERGY, runtime_session=session)
            turned = np.stack([orientation.apply(component[0]) for component in raw])
            assert turned.shape == reference.shape
            worst = max(worst, float(np.max(np.abs(turned - reference))))
            if rotation in (0, 90) and float(cch[0]).is_integer() and float(cch[1]).is_integer():
                assert np.array_equal(turned, reference), (rotation, delta, nu)
    return worst


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("cch", [(4.0, 6.0), (3.25, 7.5)])
def test_row_e_rsm_q_equals_stitch_q_at_the_same_physical_pixel(tmp_path, rotation, cch):
    """Row E (design §3.3): RSM's raw-frame q equals Stitch's xrayutilities backend's at the same physical pixel,
    within 1e-12 Å⁻¹. Measured: bit for bit at whole-number beam pixels for the 0° and 90° turns; at 180° and 270°
    xrayutilities' arithmetic along a reversed camera direction differs in the last place (the design expected
    bit-for-bit there too). Unit 1 (no refinement): the angles carry the calibration's offsets explicitly; unit 2
    applies them itself."""
    worst = _row_e(tmp_path, rotation, cch)
    print("row E", rotation, cch, "max |dq|", worst)
    assert worst <= 1e-12


# Row B (design §3.3): a calibration in the window's format encoding the canonical geometry (raw camera z-, x-;
# beam pixel 97, 243; 1.0147173 m; 172 µm; zero offsets), read through Stitch's reader, against the canonical
# asset on real scan 43. The design expected numerical agreement only (the reader takes the pitch as
# 0.000172 × 1e3 = 0.17200000000000001 mm); measured, the result is bit for bit the reference (2026-10-01).
from tests.core.test_rsm_operation_r2_real_data import _form, rsm_r2_project  # noqa: E402,F401


def _canonical_partner(project: Path) -> str:
    from pyFAI.detectors import Detector
    raw_shape, raw_cch = (195, 487), (97.0, 243.0)
    orientation = stitch_orientation(DetectorCamera("z-", "x-"))
    turned = (raw_shape[1], raw_shape[0]) if orientation.swaps_axes else raw_shape
    origin = np.array(_raw_frame_pixel(orientation, raw_shape, (0.0, 0.0)))
    matrix = np.column_stack([np.array(_raw_frame_pixel(orientation, raw_shape, point)) - origin
                              for point in ((1.0, 0.0), (0.0, 1.0))])
    turned_cch = np.linalg.solve(matrix, np.array(raw_cch) - origin)
    assert _raw_frame_pixel(orientation, raw_shape, tuple(turned_cch)) == raw_cch
    with xu_runtime_session() as session:
        wavelength = float(session.xu.en2lam(13_000.007))
    domain = {"del": [-50., 50.], "nu": [-35., 35.]}
    value = authored_surface_calibration(
        detector=Detector(pixel1=172e-6, pixel2=172e-6, max_shape=turned),
        cch1=float(turned_cch[0]), cch2=float(turned_cch[1]), distance_m=1.0147173,
        del_offset_deg=0.0, nu_offset_deg=0.0, energy_eV=13_000.007, xu_wavelength_A=wavelength,
        source_motors={"del": "del", "nu": "nu"}, rotation=orientation.rotation,
        control_domain=domain, domain=domain, rms_q=.001, condition=10., n_frames=5, n_points=40,
    )
    folder = project / "calibration"
    folder.mkdir(exist_ok=True)
    (folder / "rsm_canonical.xu.json").write_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    (folder / "rsm_canonical.json").write_text("{}")
    return "calibration/rsm_canonical.json"


@pytest.mark.slow
def test_row_b_a_calibration_encoding_the_canonical_geometry_agrees_with_the_reference(rsm_r2_project):
    from dataclasses import replace

    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
    from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2
    from xdart.gui.tools.rsm_values import prepare_rsm_tool_v2

    locator = _canonical_partner(rsm_r2_project)
    canonical_form = _form(rsm_r2_project, (43,), 40)
    calibration_form = replace(
        canonical_form,
        geometry_asset=rsm_calibration_geometry_input(locator),
        output_path=rsm_r2_project / "rsm-r2-43-40-calibration.nexus",
    )
    payloads = []
    try:
        for form in (canonical_form, calibration_form):
            request = prepare_rsm_tool_v2(form).request
            result = run_rsm_operation_v2(request)
            assert result.terminal.disposition is ModuleDisposition.COMMITTED
            payloads.append(result.payload)
    finally:
        for form in (canonical_form, calibration_form):
            if Path(form.output_path).exists():
                Path(form.output_path).unlink()
    canonical, calibrated = payloads
    assert canonical.result_fingerprint == "8a88826f85eda95bf9fb4eb9cd043b7b822e0827e4ec3d285203f9256f43b324"
    finite = np.isfinite(canonical.intensity), np.isfinite(calibrated.intensity)
    assert int(finite[0].sum()) == int(finite[1].sum()) == 5531
    for (_name, axis), (_other, other) in zip(canonical.axes, calibrated.axes, strict=True):
        np.testing.assert_allclose(other, axis, rtol=1e-12, atol=0)
    differ = finite[0] & ~np.isclose(canonical.intensity, calibrated.intensity, rtol=1e-12, atol=0)
    print("row B finite", int(finite[0].sum()), "axes max rel",
          max(float(np.max(np.abs(other - axis) / np.abs(axis))) for (_n, axis), (_o, other)
              in zip(canonical.axes, calibrated.axes)), "voxels differing", int(differ.sum()),
          "fingerprint", calibrated.result_fingerprint)
    assert int(differ.sum()) == 0
    assert calibrated.result_fingerprint == canonical.result_fingerprint  # measured: bit for bit
