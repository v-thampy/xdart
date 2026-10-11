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


def _row_e(tmp_path, rotation, cch, refinement=True):
    pyfai = _write_partner(tmp_path, rotation=rotation, cch=cch)
    receipt = capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path, refinement=refinement)
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
            # with the refinement the map takes the motors' angles and subtracts the offsets itself
            shift = (0.0, 0.0) if refinement else (offsets["nu"], offsets["del"])
            raw = mapper.pixel_q([np.zeros(1)] * 4 + [np.array([nu - shift[0]]), np.array([delta - shift[1]])],
                                 ENERGY, runtime_session=session)
            turned = np.stack([orientation.apply(component[0]) for component in raw])
            assert turned.shape == reference.shape
            worst = max(worst, float(np.max(np.abs(turned - reference))))
            if rotation == 0 and float(cch[0]).is_integer() and float(cch[1]).is_integer():
                assert np.array_equal(turned, reference), (rotation, delta, nu)
    return worst


@pytest.mark.parametrize("refinement", [True, False])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("cch", [(4.0, 6.0), (3.25, 7.5)])
def test_row_e_rsm_q_equals_stitch_q_at_the_same_physical_pixel(tmp_path, rotation, cch, refinement):
    """Row E (design §3.3): RSM's raw-frame q equals Stitch's xrayutilities backend's at the same physical pixel,
    within 1e-12 Å⁻¹. Measured: bit for bit at whole-number beam pixels for the 0° turn, and for the 90° turn on
    macOS but not on x86 Linux; at 180° and 270° xrayutilities' arithmetic along a reversed camera direction
    differs in the last place (the design expected bit-for-bit there too). Only the 0° turn is pinned bit for bit
    (maintainer, 2026-10-11). With the refinement (on by default) the map applies the calibration's del/nu offsets
    itself; with it off the motors' angles carry them explicitly, which is the plain geometry."""
    worst = _row_e(tmp_path, rotation, cch, refinement)
    print("row E", rotation, cch, refinement, "max |dq|", worst)
    assert worst <= 1e-12


# Row B (design §3.3): a calibration in the window's format encoding the canonical geometry (raw camera z-, x-;
# beam pixel 97, 243; 1.0147173 m; 172 µm; zero offsets), read through Stitch's reader, against the canonical
# asset on real scan 43. The design expected numerical agreement only (the reader takes the pitch as
# 0.000172 × 1e3 = 0.17200000000000001 mm); measured, the result is bit for bit the reference (2026-10-01).
from tests.core.test_rsm_operation_r2_real_data import _form, rsm_r2_project  # noqa: E402,F401


def _canonical_partner(project: Path) -> str:
    from pyFAI.detectors import Detector
    raw_shape, raw_cch = (195, 487), (97.0, 243.0)
    # This partner deliberately represents the frozen canonical asset's recorded camera frame.
    orientation = stitch_orientation(DetectorCamera("z-", "x-"), frame=DetectorCamera("x-", "z+"))
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
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2

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
            prepared = prepare_rsm_tool_v2(form)
            request = prepared.request
            if form is calibration_form:  # the Preview names the source and the energy it maps with
                from xrd_tools.analysis.rsm_values import rsm_preflight_summary_lines
                lines = rsm_preflight_summary_lines(prepared.summary)
                assert any(line.startswith("Geometry: the calibration on the Calibration row, refinement on")
                           for line in lines)
                assert "  energy: recorded 13000.007 eV, offset +0 eV, used 13000.007 eV" in lines
                # decision 2: #G3 checked against its own reflections in the instrument's (here standard) model
                assert any(line.startswith("  UB: #G3; recomputed from (1 1 4) and (-1 0 4) in the instrument's "
                                           "directions it agrees to") for line in lines)
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


# The refinement on a version-3 transfer (a real pyFAI extended arm, tilted mount and offsets), on every one of
# the 8 cameras (a turn and a mirror), with an energy offset: RSM's raw-frame q equals Stitch's own xrayutilities
# path at the same physical pixel. The calibrant's sample offset is never taken by RSM, so the comparison zeroes
# it in the asset (Stitch would apply it).
from tests.core.test_xu_extended_arm import _record  # noqa: E402
from tests.core.test_stacked_arm_extension import ANGLES as ARM_ANGLES  # noqa: E402

ENERGY_OFFSET = 3.0


@pytest.fixture(scope="module")
def v3_asset():
    from xrd_tools.integrate.xu_transfer import transfer_arm_to_xu
    asset, _diagnostics = transfer_arm_to_xu(_record(), validation_angles=ARM_ANGLES)
    return asset


def _v3_partner(root: Path, asset, quarter: int, flip: bool) -> tuple[Path, dict]:
    from xrd_tools.analysis.xu_stitch_calibration import _canonical_bytes, parse_xu_stitch_calibration_bytes
    value = json.loads(json.dumps(asset))
    acquisition = value["acquisition"]
    acquisition["image_rotation_quarter_turns"] = quarter
    acquisition.pop("image_flip", None)
    if flip:
        acquisition["image_flip"] = True
    acquisition["sampledis_mm"] = [0.0, 0.0, 0.0]
    acquisition["source_energy_eV"] = acquisition["energy_eV"] - ENERGY_OFFSET
    raw = _canonical_bytes(value)
    parse_xu_stitch_calibration_bytes(raw)
    name = f"arm_{quarter}_{int(flip)}"
    (root / f"{name}.xu.json").write_bytes(raw)
    (root / f"{name}.json").write_text("{}")
    return root / f"{name}.json", value


@pytest.mark.parametrize("quarter", [0, 1, 2, 3])
@pytest.mark.parametrize("flip", [False, True])
def test_row_e_the_refinement_matches_stitch_on_every_camera(tmp_path, v3_asset, quarter, flip):
    """Rows D and E: the offsets, the mount (its azimuth carried to the raw camera) and the energy offset apply,
    and RSM's q then equals Stitch's at the same physical pixel within 1e-12 Å⁻¹, for all 8 cameras."""
    pyfai, value = _v3_partner(tmp_path, v3_asset, quarter, flip)
    receipt = capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path)
    effective = lower_rsm_effective_geometry(receipt)
    assert effective.refinement and any(effective.area_mount) and any(effective.angle_offsets)
    assert effective.energy_offset_eV == pytest.approx(ENERGY_OFFSET, abs=1e-9)
    mapper = rsm_effective_pixel_q_map(effective)
    acquisition = value["acquisition"]
    orientation = ImageOrientation(rotation=90 * quarter, flip_horizontal=flip)
    recorded = acquisition["source_energy_eV"]  # the scan's recorded energy; used = recorded + offset
    worst = 0.0
    with xu_runtime_session() as session:
        stitch = XuPowderQProvider(resolve_xu_stitch_effective_geometry(receipt.calibration, session),
                                   acquisition)
        for delta, nu in ANGLES:
            with stitch.frame(nu=nu, del_=delta) as lease:
                reference = np.stack(lease._values)
            raw = mapper.pixel_q([np.zeros(1)] * 4 + [np.array([nu]), np.array([delta])], recorded,
                                 runtime_session=session)
            turned = np.stack([orientation.apply(component[0]) for component in raw])
            worst = max(worst, float(np.max(np.abs(turned - reference))))
            # row D: without the refinement the q differs (the mount, offsets and energy offset all apply)
            plain = rsm_effective_pixel_q_map(lower_rsm_effective_geometry(capture_rsm_calibration_geometry(
                pyfai.name, project_root=tmp_path, refinement=False)))
            unrefined = plain.pixel_q([np.zeros(1)] * 4 + [np.array([nu]), np.array([delta])], recorded,
                                      runtime_session=session)
            assert float(np.max(np.abs(np.stack(unrefined) - np.stack(raw)))) > 1e-4
    print("row E v3", quarter, flip, "max |dq|", worst)
    assert worst <= 1e-12


def test_the_energy_offset_scales_q_as_its_energy(tmp_path, v3_asset):
    """The energy-offset known answer: a scan recorded at E0 maps with E0 + offset, so with the geometry fixed
    every |q| scales by (E0 + offset) / E0."""
    from dataclasses import replace
    pyfai, value = _v3_partner(tmp_path, v3_asset, 0, False)
    mapper = rsm_effective_pixel_q_map(lower_rsm_effective_geometry(
        capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path)))
    recorded = 13_000.0
    angles = [np.zeros(1)] * 4 + [np.array([7.0]), np.array([21.0])]
    with xu_runtime_session() as session:
        used = np.stack(mapper.pixel_q(angles, recorded, runtime_session=session))
        explicit = np.stack(replace(mapper, energy_offset_eV=0.0).pixel_q(
            angles, recorded + mapper.energy_offset_eV, runtime_session=session))
        base = np.stack(replace(mapper, energy_offset_eV=0.0).pixel_q(angles, recorded, runtime_session=session))
    assert np.array_equal(used, explicit)
    ratio = np.linalg.norm(used, axis=0) / np.linalg.norm(base, axis=0)
    np.testing.assert_allclose(ratio, (recorded + ENERGY_OFFSET) / recorded, rtol=1e-12, atol=0)


def test_the_preview_energy_lines_and_the_far_energy_warning():
    """The Preview carries each scan's recorded energy, the offset and the energy used, and warns when a scan was
    recorded more than 2 % from the calibration's energy (the maintainer's offset decision; design §3.2)."""
    from xrd_tools.analysis.rsm_values import _rsm_member_energy_lines
    assert _rsm_member_energy_lines(13_000.0, None) == ("  energy: 13000 eV",)
    assert _rsm_member_energy_lines(13_000.0, (False, 3.0, 13_000.0)) == ("  energy: 13000 eV",)
    near = _rsm_member_energy_lines(13_100.0, (True, 3.0, 13_000.0))
    assert near == ("  energy: recorded 13100 eV, offset +3 eV, used 13103 eV",)
    far = _rsm_member_energy_lines(8_000.0, (True, 3.0, 13_000.0))
    assert far[0] == "  energy: recorded 8000 eV, offset +3 eV, used 8003 eV"
    assert far[1].startswith("  ⚠ recorded 38.5 % from the calibration's 13000 eV")


def test_the_polarization_weight_uses_the_energy_q_is_mapped_with(tmp_path, v3_asset):
    """A refined energy offset maps q at E0 + offset; the polarization weight's wavelength must be that energy's
    too, or the scatter is not elastic (found by the S3b self-review: the weights took the recorded energy)."""
    from dataclasses import replace
    from types import SimpleNamespace

    from xrd_tools.analysis.rsm_operation import _rsm_chunk_weights
    from xrd_tools.corrections.frame_preparation import polarization_axes
    pyfai, value = _v3_partner(tmp_path, v3_asset, 0, False)
    effective = lower_rsm_effective_geometry(capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path))
    mapper = rsm_effective_pixel_q_map(effective)
    diffractometer = effective.diffractometer_projection
    state = SimpleNamespace(
        bound=SimpleNamespace(corrections=SimpleNamespace(polarization_factor=0.99)), solid=None, flat=None,
        axes=polarization_axes(diffractometer.r_i, diffractometer.detector_circles[0]))
    angles = [np.zeros(1)] * 4 + [np.array([7.0]), np.array([21.0])]
    with xu_runtime_session() as session:
        refined = _rsm_chunk_weights(state, mapper, session, angles, 13_000.0, (0, -1, 0, -1))
        explicit = _rsm_chunk_weights(state, replace(mapper, energy_offset_eV=0.0), session, angles,
                                      13_000.0 + mapper.energy_offset_eV, (0, -1, 0, -1))
    assert np.array_equal(refined, explicit)


@pytest.mark.parametrize("quarter", [0, 1, 2, 3])
@pytest.mark.parametrize("flip", [False, True])
def test_the_solid_angle_follows_the_calibrations_mount_as_stitchs_arm(tmp_path, v3_asset, quarter, flip):
    """S3b's leftover (S3c unit 4a): with the refinement, a version-3 calibration's solid-angle weight is Stitch's
    own (``_extended_solid_angle``), the calibrant's sample offset at zero as RSM's q takes it, in the raw frame.
    pyFAI's relative solid angle is (dist / r)³ about its PONI, the perpendicular foot from the sample; the tilted
    mount moves that foot off xrayutilities' beam pixel, about which the untilted formula is centred (2 % apart on
    this fixture: logs/s3c-solid-angle-arm.log). Without the sample offset the weight is the same at every pose,
    so the run computes it once."""
    from types import SimpleNamespace

    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_solid_angle
    from xrd_tools.analysis.rsm_operation import RSMBoundCorrections, RSMFrameCorrections, _rsm_correction_state
    from xrd_tools.corrections.frame_preparation import flat_detector_solid_angle
    from xrd_tools.integrate.xu_stitch import _extended_solid_angle

    pyfai, value = _v3_partner(tmp_path, v3_asset, quarter, flip)
    receipt = capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path)
    effective = lower_rsm_effective_geometry(receipt)
    header = effective.detector_header
    plan = SimpleNamespace(corrections=RSMBoundCorrections(RSMFrameCorrections(solid_angle=True), None, None, None))
    state = _rsm_correction_state(plan, header, effective.diffractometer_projection, receipt)
    assert np.array_equal(state.solid, rsm_calibration_solid_angle(receipt)) and not state.solid.flags.writeable
    turned = ImageOrientation(rotation=90 * quarter, flip_horizontal=flip).apply(state.solid)
    stitch = _extended_solid_angle(value["corrections"], tuple(value["detector"]["shape"]), sample_offset=False)
    for del_, nu in ANGLES:
        assert np.array_equal(turned, stitch(nu, del_))  # Stitch's weight, bit for bit, the same at every pose
    with_offset = _extended_solid_angle(value["corrections"], tuple(value["detector"]["shape"]))
    assert not np.array_equal(with_offset(*ANGLES[1][::-1]), turned)  # Stitch's own run takes the offset
    untilted = flat_detector_solid_angle((int(header.Nch1), int(header.Nch2)), distance=float(header.distance),
                                         cch=(float(header.cch1), float(header.cch2)),
                                         pixel_size=(float(header.pwidth1), float(header.pwidth2)))
    assert float(np.max(np.abs(state.solid / untilted - 1))) > 1e-2
    # the untilted formula stays for the refinement off, a version-2 partner and the canonical asset
    off = capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path, refinement=False)
    v2 = capture_rsm_calibration_geometry(_write_partner(tmp_path, rotation=0, cch=(4., 6.)).name,
                                          project_root=tmp_path)
    assert all(rsm_calibration_solid_angle(item) is None for item in (off, v2, None))
    # the Preview names the same source (G8b), by the same rule
    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_solid_angle_source
    assert rsm_calibration_solid_angle_source(receipt).startswith("the refined calibration's arm")
    assert {rsm_calibration_solid_angle_source(item) for item in (off, v2, None)} == {
        "the untilted formula about the beam pixel"}
    off_state = _rsm_correction_state(plan, lower_rsm_effective_geometry(off).detector_header,
                                      effective.diffractometer_projection, off)
    assert np.array_equal(off_state.solid, untilted)


@pytest.mark.slow
def test_the_run_hands_its_calibration_to_the_solid_angle(rsm_r2_project, monkeypatch):
    """The run's correction state takes the plan's own calibration (row B's version-2 partner: no arm, so the
    untilted formula, and the run commits)."""
    from dataclasses import replace

    from xrd_tools.analysis import rsm_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
    from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, run_rsm_operation_v2
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2

    seen, real = [], rsm_operation.rsm_calibration_solid_angle

    def spy(receipt):
        seen.append(type(receipt).__name__)
        return real(receipt)

    monkeypatch.setattr(rsm_operation, "rsm_calibration_solid_angle", spy)
    form = replace(_form(rsm_r2_project, (43,), 40),
                   geometry_asset=rsm_calibration_geometry_input(_canonical_partner(rsm_r2_project)),
                   corrections=RSMFrameCorrections(solid_angle=True),
                   output_path=rsm_r2_project / "rsm-s3c-solid-angle.nexus")
    try:
        result = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
    finally:
        Path(form.output_path).unlink(missing_ok=True)
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert seen == ["RSMCalibrationGeometryReceipt"]


# Unit 3: an Auto threshold's ceiling from the calibrated detector (decision 39, as Stitch passes it). RSM decodes
# every frame to float64, so the data type's rule had nothing to find there (C4's disclosure): Auto left out only
# negatives and the UINT32 dummy.
def _members(*options):
    from types import SimpleNamespace
    return tuple(SimpleNamespace(detector_shape=(2, 3), source_options_json=json.dumps(value)) for value in options)


def test_an_auto_threshold_takes_the_calibrated_detectors_counter_limit():
    from types import SimpleNamespace

    from xrd_tools.analysis.rsm_operation import (
        RSMBoundCorrections, RSMFrameCorrections, _apply_rsm_frame_corrections, _bind_rsm_corrections,
    )
    from xrd_tools.core.invalid import PILATUS_COUNTER_LIMIT
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning

    spec, narrow, nexus = ({"read_image_kwargs": {"raw_dtype": "<i4"}},
                           {"read_image_kwargs": {"raw_dtype": "<u2"}},  # 65535 < the limit: its own rule
                           {"entry": "entry1"})  # learned only at the run: unknown, so the limit applies
    auto = RSMFrameCorrections(threshold=StitchConditioning(mode="auto"))
    bound = _bind_rsm_corrections(auto, _members(spec, narrow, nexus), 1 << 20, detector="Pilatus300kw")
    limit = PILATUS_COUNTER_LIMIT
    assert bound.saturation == (limit, None, limit)
    assert bound.to_provenance()["threshold"]["saturation"] == [limit, None, limit]
    assert bound.identity()[-1] == ("saturation", (limit, None, limit))
    plain = RSMBoundCorrections(auto, None, None, None).identity()
    for detector, options, corrections in (
            (None, (spec,), auto),  # the canonical asset names no detector
            ("Eiger2CdTe_1M", (spec,), auto),  # no documented limit
            ("Pilatus300kw", (narrow,), auto),  # the data type's own rule
            ("Pilatus300kw", (spec,), RSMFrameCorrections(threshold=StitchConditioning(mode="off")))):
        other = _bind_rsm_corrections(corrections, _members(*options), 1 << 20, detector=detector)
        assert other.saturation is None and "saturation" not in other.to_provenance()["threshold"]
        if corrections is auto:
            assert other.identity() == plain  # an identity without a ceiling is the one it was
    raw = np.array([[[0.0, limit - 1.0, limit], [5.0, -1.0, 2.0e6]]])
    for saturation, expected in ((None, [False, False, False, False, True, False]),
                                 (limit, [False, False, True, False, True, True])):
        conditioned = raw.copy()
        _apply_rsm_frame_corrections(raw, conditioned, SimpleNamespace(bound=bound, invalid=None, dark=None),
                                     np.ones(1), saturation)
        assert np.isnan(conditioned[0]).ravel().tolist() == expected


def test_the_calibration_names_its_detector(tmp_path):
    """(The canonical asset's None is the run row's, on its real receipt.)"""
    from xrd_tools.analysis.rsm_operation import _rsm_calibration_detector
    receipt = capture_rsm_calibration_geometry(_write_partner(tmp_path, rotation=0, cch=(4., 6.)).name,
                                               project_root=tmp_path)
    assert _rsm_calibration_detector(receipt) == "Detector"  # pyFAI's name for the partner's detector


@pytest.mark.slow
def test_the_run_applies_each_members_ceiling_from_the_calibration(rsm_r2_project, monkeypatch):
    """Preview passes the calibration's detector to the binding, and each chunk its member's ceiling (row B's
    calibration, its detector named a Pilatus so that one applies; the name's lookup is the row above)."""
    from dataclasses import replace

    from xrd_tools.analysis import rsm_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import RSMCalibrationGeometryReceipt, rsm_calibration_geometry_input
    from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, run_rsm_operation_v2
    from xrd_tools.core.invalid import PILATUS_COUNTER_LIMIT
    from xrd_tools.integrate.stitch_conditioning import StitchConditioning
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2

    looked_up, applied = [], []
    original_lookup, original_apply = rsm_operation._rsm_calibration_detector, rsm_operation._apply_rsm_frame_corrections

    def lookup(receipt):
        looked_up.append((type(receipt).__name__, original_lookup(receipt)))
        return "Pilatus100k" if type(receipt) is RSMCalibrationGeometryReceipt else None

    def apply(raw, conditioned, state, divisors, saturation=None, point_images=None):
        applied.append(saturation)
        return original_apply(raw, conditioned, state, divisors, saturation, point_images)

    monkeypatch.setattr(rsm_operation, "_rsm_calibration_detector", lookup)
    monkeypatch.setattr(rsm_operation, "_apply_rsm_frame_corrections", apply)
    auto = RSMFrameCorrections(threshold=StitchConditioning(mode="auto"))
    canonical = replace(_form(rsm_r2_project, (43,), 40), corrections=auto,
                        output_path=rsm_r2_project / "rsm-s3b-saturation-canonical.nexus")
    calibrated = replace(canonical, geometry_asset=rsm_calibration_geometry_input(_canonical_partner(rsm_r2_project)),
                         output_path=rsm_r2_project / "rsm-s3b-saturation-calibration.nexus")
    assert prepare_rsm_tool_v2(canonical).request.plan.corrections.saturation is None
    request = prepare_rsm_tool_v2(calibrated).request
    assert request.plan.corrections.saturation == (PILATUS_COUNTER_LIMIT,)
    assert looked_up == [("RSMGeometryAssetReceipt", None), ("RSMCalibrationGeometryReceipt", "Detector")]
    try:
        result = run_rsm_operation_v2(request)
    finally:
        if Path(calibrated.output_path).exists():
            Path(calibrated.output_path).unlink()
    assert result.terminal.disposition is ModuleDisposition.COMMITTED
    assert applied and set(applied) == {PILATUS_COUNTER_LIMIT}


# Unit 4, the R4 residual: the dark and flat usability check leaves out the detector's own mask (module gaps), as
# Int's preflight passes ``detector_mask``, in the capture Stitch and RSM share; RSM passes its calibration's.
def test_a_gap_masked_flat_raises_no_false_masked_fraction_warning(tmp_path):
    from xrd_tools.analysis.stitch_native import capture_detector_image
    from xrd_tools.corrections.detector import check_detector_image

    gaps = np.zeros((20, 10), dtype=bool)
    gaps[8:10] = True  # two module-gap rows: 10 % of the detector
    for role, image in (("flat", np.where(gaps, 0.0, 1.0)), ("dark", np.where(gaps, np.nan, 3.0))):
        path = tmp_path / f"{role}.npy"
        np.save(path, image)
        blind, _pixels = capture_detector_image(path, role, 1 << 20)
        aware, _pixels = capture_detector_image(path, role, 1 << 20, detector_mask=gaps)
        assert any("masks 10.0 %" in warning for warning in blind.warnings)  # the gaps counted as masked
        assert aware.warnings == () == check_detector_image(image, role, detector_mask=gaps).warnings
        assert aware.fingerprint == blind.fingerprint  # the identity is the file's, not its counts'
        if role == "flat":
            assert (blind.flat_masked_pixels, aware.flat_masked_pixels) == (20, 0)
            assert aware.flat_median == blind.flat_median == 1.0


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_the_calibrations_detector_mask_maps_back_to_the_raw_frame(tmp_path, rotation):
    from pyFAI.detectors import Pilatus300k

    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_detector_mask

    with xu_runtime_session() as session:
        wavelength = float(session.xu.en2lam(ENERGY))
    domain = {"del": [-50., 50.], "nu": [-35., 35.]}
    detector = Pilatus300k()
    value = authored_surface_calibration(
        detector=detector, cch1=300., cch2=240., distance_m=.39, del_offset_deg=0., nu_offset_deg=0.,
        energy_eV=ENERGY, xu_wavelength_A=wavelength, source_motors={"del": "del", "nu": "nu"},
        rotation=rotation, control_domain=domain, domain=domain, rms_q=.001, condition=10., n_frames=5,
        n_points=40)
    (tmp_path / "p300k.xu.json").write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    (tmp_path / "p300k.json").write_text("{}")
    raw = rsm_calibration_detector_mask(capture_rsm_calibration_geometry("p300k.json", project_root=tmp_path))
    turned = np.asarray(detector.calc_mask(), dtype=bool)
    assert raw.shape == (turned.shape[::-1] if rotation in (90, 270) else turned.shape)
    assert not raw.flags.writeable and 0 < int(raw.sum()) == int(turned.sum())
    np.testing.assert_array_equal(ImageOrientation(rotation=rotation).apply(raw), turned)
    generic = _write_partner(tmp_path, rotation=rotation, cch=(4., 6.))  # pyFAI's generic detector: no mask
    assert rsm_calibration_detector_mask(capture_rsm_calibration_geometry(generic.name, project_root=tmp_path)) is None


@pytest.mark.slow
def test_rsm_checks_its_flat_against_the_calibrations_mask_and_the_run_recaptures_it_alike(
        rsm_r2_project, tmp_path, monkeypatch):
    """Row B's calibration (scan 43), its detector given a 12-row gap and a flat that is 0 there: Preview's
    capture leaves the gap out, and the run's re-capture, with the bound mask, is Preview's (no false refusal)."""
    from dataclasses import replace

    from xrd_tools.analysis import rsm_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
    from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, run_rsm_operation_v2
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2

    gaps = np.zeros((195, 487), dtype=bool)
    gaps[60:72] = True  # 6.2 % of the detector
    gaps.setflags(write=False)
    flat = tmp_path / "gap-flat.npy"
    np.save(flat, np.where(gaps, 0.0, 1.0))
    form = replace(_form(rsm_r2_project, (43,), 40),
                   geometry_asset=rsm_calibration_geometry_input(_canonical_partner(rsm_r2_project)),
                   corrections=RSMFrameCorrections(flat_path=str(flat)),
                   output_path=rsm_r2_project / "rsm-s3b-gap-flat.nexus")
    blind = prepare_rsm_tool_v2(form).request.plan.corrections  # row B's generic detector has no mask
    assert blind.detector_mask is None and any("masks 6.2 %" in item for item in blind.flat_receipt.warnings)
    built = []

    def mask(receipt):
        built.append(type(receipt).__name__)
        return gaps

    monkeypatch.setattr(rsm_operation, "rsm_calibration_detector_mask", mask)
    request = prepare_rsm_tool_v2(form).request
    bound = request.plan.corrections
    assert built == ["RSMCalibrationGeometryReceipt"] and bound.detector_mask is gaps
    assert bound.flat_receipt.warnings == () and bound.flat_receipt.flat_masked_pixels == 0
    try:
        result = run_rsm_operation_v2(request)
    finally:
        if Path(form.output_path).exists():
            Path(form.output_path).unlink()
    assert result.terminal.disposition is ModuleDisposition.COMMITTED


def _with_record(pyfai: Path, reversed_circles) -> None:
    """The instrument record the calibration window saves in the partner's provenance."""
    from xrd_tools.session.instrument import InstrumentGeometry
    record = InstrumentGeometry("psic", reversed_circles=tuple(reversed_circles)).to_record()
    partner = pyfai.with_suffix(".xu.json")
    partner.with_suffix(partner.suffix + ".provenance.json").write_text(json.dumps({"instrument_geometry": record}))


def test_the_calibrations_instrument_record_gives_every_circle_its_direction(tmp_path):
    """Decision 2 (the maintainer: "Xu provides a way to specify the actual circle directions on the instrument.
    Shouldn't that be used?"): the sample circles turn as the calibration's instrument record says, the detector
    circles as the record and the partner agree. Without a record, psic's standard directions, as before."""
    pyfai = _write_partner(tmp_path, rotation=0, cch=(4.0, 6.0))
    plain = lower_rsm_effective_geometry(capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path))
    assert plain.diffractometer_projection.sample_circles == ("x+", "z-", "y+", "z-")
    assert plain.circle_directions == "psic's standard (the calibration records no instrument)"
    _with_record(pyfai, ("mu", "chi"))
    receipt = capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path)
    effective = lower_rsm_effective_geometry(receipt)
    assert effective.diffractometer_projection.sample_circles == ("x-", "z-", "y-", "z-")
    assert effective.diffractometer_projection.detector_circles == ("x+", "z-")
    assert effective.circle_directions == "the calibration's instrument record: psic, reversed mu, chi"
    assert effective.fingerprint != plain.fingerprint


def test_a_record_whose_detector_circles_the_partner_was_not_fitted_in_is_refused(tmp_path):
    """A version-2 partner's arm is fitted in psic's standard detector directions: a record reversing nu cannot
    be its instrument."""
    pyfai = _write_partner(tmp_path, rotation=0, cch=(4.0, 6.0))
    _with_record(pyfai, ("mu", "chi", "nu"))  # the 17-2 record
    receipt = capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path)
    with pytest.raises(RSMGeometryAssetRefused) as refused:
        lower_rsm_effective_geometry(receipt)
    assert refused.value.code == "RSM_CALIBRATION_UNSUPPORTED" and "save the calibration again" in str(refused.value)


def test_a_record_changed_after_preview_is_refused_at_the_run(tmp_path):
    from xrd_tools.analysis.rsm_geometry_asset import revalidate_rsm_geometry_source
    pyfai = _write_partner(tmp_path, rotation=0, cch=(4.0, 6.0))
    _with_record(pyfai, ("mu",))
    receipt = capture_rsm_calibration_geometry(pyfai.name, project_root=tmp_path)
    revalidate_rsm_geometry_source(receipt)
    _with_record(pyfai, ("mu", "chi"))
    with pytest.raises(RSMGeometryAssetRefused) as refused:
        revalidate_rsm_geometry_source(receipt)
    assert refused.value.code == "RSM_GEOMETRY_ASSET_IDENTITY_MISMATCH"


@pytest.mark.slow
def test_scan_43_under_a_partial_reversal_of_mu_and_chi_misindexes_specs_labels(rsm_r2_project):
    """A record reversing mu and chi but not nu is not a mirror (a version-2 partner's arm is fitted in psic's
    standard detector directions), so RSM maps it as recorded, recomputing the UB from scan 43's reflections
    (1 1 4) and (-1 0 4) as SPEC labelled them. Through the product's path, at the STO substrate's (113) that SPEC
    centred on the detector (frame 22, pixel (95, 244)):
    - psic's standard directions put it 0.049 from (1 1 3), where SPEC's own #Q puts it;
    - the partial record puts it 0.17 off: SPEC's labels were assigned in its own psic model, whose handedness this
      model does not share (nu is 0 throughout, so only mu and chi enter).
    This is about SPEC's labels under a partial reversal, not evidence about 17-2's motors (this row's first
    reading, 2026-10-01, was retracted the same day). 17-2's whole record reverses nu too: psic's mirror image,
    which no set of reflection angles can tell from psic, and which RSM maps through the mirror
    (test_scan_43_with_17_2s_whole_record_maps_through_the_mirror_as_spec_does). The remaining 0.049 is the
    orientation reflections' own disagreement: or1 sits about 0.9° off or0 and the (113) under every direction set."""
    from dataclasses import replace

    from silx.io.specfile import SpecFile

    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2, rsm_preflight_summary_lines

    locator = _canonical_partner(rsm_r2_project)
    pyfai = rsm_r2_project / locator
    provenance = pyfai.with_suffix(".xu.json.provenance.json")
    form = replace(_form(rsm_r2_project, (43,), 40), geometry_asset=rsm_calibration_geometry_input(locator))
    scan = SpecFile(str(rsm_r2_project / "STO_align"))["43.1"]
    fixed = dict(zip(("del", "eta", "chi", "phi", "nu", "mu"),
                     (float(value) for value in scan.scan_header_dict["P0"].split()[:6])))
    eta = float(np.asarray(scan.data_column_by_name("eta"), dtype=float)[22])
    angles = [np.array([value]) for value in (fixed["mu"], eta, fixed["chi"], fixed["phi"], fixed["nu"],
                                              fixed["del"])]
    off = {}
    try:
        for name, reversals in (("standard", ()), ("17-2", ("mu", "chi"))):
            if reversals:
                _with_record(pyfai, reversals)
            prepared = prepare_rsm_tool_v2(form)
            member = prepared.request.preflight.members[0]
            mapper = rsm_effective_pixel_q_map(prepared.request.preflight.effective_geometry)
            h, k, l = mapper.pixel_q(angles, member.energy_eV, UB=np.asarray(member.ub))
            at = np.array([h[0, 95, 244], k[0, 95, 244], l[0, 95, 244]])
            off[name] = float(np.linalg.norm(at - (1.0, 1.0, 3.0)))
            lines = rsm_preflight_summary_lines(prepared.summary)
            if reversals:
                assert "Circle directions: the calibration's instrument record: psic, reversed mu, chi" in lines
                assert any(line.startswith("  UB: recomputed from (1 1 4) and (-1 0 4) in the instrument's "
                                           "directions; #G3 differs by") and line.endswith("there: chi, mu")
                           for line in lines)
            else:
                assert any(line.startswith("  UB: #G3; recomputed") for line in lines)
    finally:
        provenance.unlink(missing_ok=True)
    print("scan 43 (113) |off|", off)
    assert off["standard"] <= 0.06 and off["17-2"] >= 0.10, off


def _mirror_partner(project: Path, asset, reversed_circles) -> str:
    """Row B's canonical geometry (scan 43) as 17-2 describes it, psic's mirror image: nu turns the other way (a
    version-3 file; a version-2 file's arm is fitted in psic's standard detector directions), the raw camera's
    left-right axis is flipped (rows z+, columns x-, for row B's z-, x-), and the record reverses mu, chi and nu,
    with any other circle in ``reversed_circles``. Seen in the mirror it is row B's file exactly."""
    import math

    from pyFAI.detectors import Detector

    from xrd_tools.analysis.xu_stitch_calibration import extended_surface_calibration
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    raw_shape, raw_cch = (195, 487), (97.0, 243.0)
    orientation = stitch_orientation(DetectorCamera("z+", "x-"))
    turned = (raw_shape[1], raw_shape[0]) if orientation.swaps_axes else raw_shape
    origin = np.array(_raw_frame_pixel(orientation, raw_shape, (0.0, 0.0)))
    matrix = np.column_stack([np.array(_raw_frame_pixel(orientation, raw_shape, point)) - origin
                              for point in ((1.0, 0.0), (0.0, 1.0))])
    turned_cch = np.linalg.solve(matrix, np.array(raw_cch) - origin)
    assert _raw_frame_pixel(orientation, raw_shape, tuple(turned_cch)) == raw_cch
    with xu_runtime_session() as session:
        wavelength = float(session.xu.en2lam(13_000.007))
    domain = {"del": [-50., 50.], "nu": [-35., 35.]}
    validation = asset["validation"]
    # the solid-angle arm (off in these forms) as this file's own: nu's scale reversed with the detector axes
    arm = json.loads(json.dumps(asset["corrections"]["solid_angle_arm"]))
    arm.update(detector="Detector", nu_scale=math.pi / 180, del_scale=-math.pi / 180, beam_rotation=0.0,
               wavelength_m=energy_eV_to_wavelength_m(13_000.007), source_motors={"del_angle": "del", "nu_angle": "nu"})
    value = extended_surface_calibration(
        detector=Detector(pixel1=172e-6, pixel2=172e-6, max_shape=turned),
        xu_parameters={"cch1_px": float(turned_cch[0]), "cch2_px": float(turned_cch[1]), "distance_m": 1.0147173,
                       "del_offset_deg": 0.0, "nu_offset_deg": 0.0, "detrot_deg": 0.0, "tilt_deg": 0.0,
                       "tiltazimuth_deg": 0.0, "sampledis_mm": [0.0, 0.0, 0.0]},
        arm=arm, energy_eV=13_000.007, xu_wavelength_A=wavelength,
        source_motors={"del": "del", "nu": "nu"}, rotation=orientation.rotation, control_domain=domain,
        domain=domain, fit_frames=validation["transfer_frame_count"],
        validation_frames=validation["validation_frame_count"],
        validation_pixels=validation["validation_pixel_count"], max_dq=validation["transfer_max_abs_dq_A^-1"],
        detector_axes=("x-", "z-"), flip=orientation.flip_horizontal)
    folder = project / "calibration"
    folder.mkdir(exist_ok=True)
    name = "rsm_17_2_" + "_".join(reversed_circles)
    (folder / f"{name}.xu.json").write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    (folder / f"{name}.json").write_text("{}")
    _with_record(folder / f"{name}.json", reversed_circles)
    return f"calibration/{name}.json"


def test_17_2s_whole_record_is_a_mirror_the_map_goes_through(tmp_path, v3_asset):
    """Lowering: the whole 17-2 record (mu, chi and nu reversed) sets the mirror, names it in Preview's circle
    directions with what stays reversed there, and enters the fingerprint (so an output a mirror-less map made from
    the same files is not current); the polarization weights stay the physical lab frame's, bit for bit."""
    from dataclasses import replace
    from types import SimpleNamespace

    from xrd_tools.analysis.rsm_operation import _rsm_chunk_weights
    from xrd_tools.corrections.frame_preparation import polarization_axes
    lowered = {}
    for circles, residual in ((("mu", "chi", "nu"), "psic's standard"),
                              (("mu", "chi", "phi", "nu"), "psic with phi reversed")):
        locator = _mirror_partner(tmp_path, v3_asset, circles)
        effective = lower_rsm_effective_geometry(capture_rsm_calibration_geometry(locator, project_root=tmp_path))
        lowered[circles] = locator, effective
        assert effective.mirror
        assert effective.circle_directions == (
            "the calibration's instrument record: psic, reversed " + ", ".join(circles)
            + "; mapped through the mirror into SPEC's handedness: " + residual)
        mapper = rsm_effective_pixel_q_map(effective)
        assert mapper.mirror and effective.diffractometer_projection.detector_circles == ("x-", "z-")
        diffractometer = effective.diffractometer_projection
        state = SimpleNamespace(
            bound=SimpleNamespace(corrections=SimpleNamespace(polarization_factor=0.99)), solid=None, flat=None,
            axes=polarization_axes(diffractometer.r_i, diffractometer.detector_circles[0]))
        angles = [np.zeros(2)] * 4 + [np.array([7.0, -12.0]), np.array([21.0, 33.0])]
        with xu_runtime_session() as session:
            through = _rsm_chunk_weights(state, mapper, session, angles, 13_000.007, (0, -1, 0, -1))
            physical = _rsm_chunk_weights(state, replace(mapper, mirror=False), session, angles, 13_000.007,
                                          (0, -1, 0, -1))
        assert np.array_equal(through, physical)
    locator, mirrored = lowered[("mu", "chi", "nu")]
    (tmp_path / locator).with_suffix(".xu.json.provenance.json").unlink()  # no record: the partner's own directions
    plain = lower_rsm_effective_geometry(capture_rsm_calibration_geometry(locator, project_root=tmp_path))
    assert not plain.mirror and plain.fingerprint != mirrored.fingerprint
    assert plain.diffractometer_projection.sample_circles == ("x+", "z-", "y+", "z-")


@pytest.mark.slow
@pytest.mark.parametrize("circles", [("mu", "chi", "nu"), ("mu", "chi", "phi", "nu")],
                         ids=["pure-mirror", "phi-reversed-too"])
def test_scan_43_with_17_2s_whole_record_maps_through_the_mirror_as_spec_does(rsm_r2_project, v3_asset, circles):
    """The fix (approved 2026-10-01): with 17-2's whole record, RSM maps scan 43 through the mirror. SPEC's #G3 is
    kept (its reflections give it there to the header's rounding), the (113) lands where psic's standard model
    puts it (0.049 from (1 1 3), where SPEC's #Q puts it, against 0.171 for the partial record mapped as
    recorded), and the run's grid and intensities are row B's, the canonical reference's. With phi reversed too
    (17-2 now) nothing moves: phi is 0 throughout scan 43."""
    from dataclasses import replace

    from silx.io.specfile import SpecFile

    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
    from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2, rsm_preflight_summary_lines

    canonical_form = _form(rsm_r2_project, (43,), 40)
    mirror_form = replace(canonical_form,
                          geometry_asset=rsm_calibration_geometry_input(
                              _mirror_partner(rsm_r2_project, v3_asset, circles)),
                          output_path=rsm_r2_project / f"rsm-17-2-{len(circles)}.nexus")
    scan = SpecFile(str(rsm_r2_project / "STO_align"))["43.1"]
    fixed = dict(zip(("del", "eta", "chi", "phi", "nu", "mu"),
                     (float(value) for value in scan.scan_header_dict["P0"].split()[:6])))
    eta = float(np.asarray(scan.data_column_by_name("eta"), dtype=float)[22])
    angles = [np.array([value]) for value in (fixed["mu"], eta, fixed["chi"], fixed["phi"], fixed["nu"],
                                              fixed["del"])]
    peaks, payloads, ubs, lines = [], [], [], []
    try:
        for form in (canonical_form, mirror_form):
            prepared = prepare_rsm_tool_v2(form)
            member = prepared.request.preflight.members[0]
            mapper = rsm_effective_pixel_q_map(prepared.request.preflight.effective_geometry)
            h, k, l = mapper.pixel_q(angles, member.energy_eV, UB=np.asarray(member.ub))
            peaks.append(np.array([h[0, 95, 244], k[0, 95, 244], l[0, 95, 244]]))
            ubs.append(member.ub)
            lines = rsm_preflight_summary_lines(prepared.summary)
            result = run_rsm_operation_v2(prepared.request)
            assert result.terminal.disposition is ModuleDisposition.COMMITTED
            payloads.append(result.payload)
    finally:
        for form in (canonical_form, mirror_form):
            if Path(form.output_path).exists():
                Path(form.output_path).unlink()
    off = float(np.linalg.norm(peaks[1] - (1.0, 1.0, 3.0)))
    print("scan 43 (113) through the mirror", peaks[1], "|off|", off, "max |d| vs canonical",
          float(np.max(np.abs(peaks[1] - peaks[0]))))
    np.testing.assert_allclose(peaks[1], peaks[0], rtol=0, atol=1e-12)
    assert abs(off - 0.04917069605879838) < 1e-9
    assert ubs[1] == ubs[0]  # SPEC's #G3, its own bytes
    residual = "psic's standard" if "phi" not in circles else "psic with phi reversed"
    assert (f"Circle directions: the calibration's instrument record: psic, reversed {', '.join(circles)}; "
            f"mapped through the mirror into SPEC's handedness: {residual}") in lines
    assert any(line.startswith("  UB: #G3; recomputed from (1 1 4) and (-1 0 4) in the instrument's directions "
                               "through the mirror it agrees to") for line in lines)
    canonical, mirrored = payloads
    assert canonical.result_fingerprint == "8a88826f85eda95bf9fb4eb9cd043b7b822e0827e4ec3d285203f9256f43b324"
    finite = np.isfinite(canonical.intensity), np.isfinite(mirrored.intensity)
    assert int(finite[0].sum()) == int(finite[1].sum()) == 5531 and np.array_equal(finite[0], finite[1])
    for (_name, axis), (_other, other) in zip(canonical.axes, mirrored.axes, strict=True):
        np.testing.assert_allclose(other, axis, rtol=1e-12, atol=0)
    differ = finite[0] & ~np.isclose(canonical.intensity, mirrored.intensity, rtol=1e-12, atol=0)
    print("through the mirror: voxels differing", int(differ.sum()), "fingerprint", mirrored.result_fingerprint)
    assert int(differ.sum()) == 0
    assert mirrored.result_fingerprint == canonical.result_fingerprint  # measured: bit for bit


# S3c unit 4b/4c: the calibrated detector's own mask (module gaps) is every frame's, whatever the corrections, beside
# the Mask File; the plan holds it, so every charge of a chunk counts it (Preview's Auto cap, the user's cap, the
# run's memory plan and each pass's load check).
def test_the_detector_mask_is_charged_in_every_pass_and_in_auto():
    from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, rsm_v2_auto_chunking, rsm_v2_chunk_bytes

    for shape in ((195, 487), (2167, 2070)):
        for static_hot in (False, True):
            for corrections in (None, RSMFrameCorrections(solid_angle=True, polarization_factor=0.99)):
                plain = rsm_v2_chunk_bytes(shape, shape, 3, static_hot=static_hot, corrections=corrections)
                held = rsm_v2_chunk_bytes(shape, shape, 3, static_hot=static_hot, corrections=corrections,
                                          detector_mask=True)
                assert held - plain == shape[0] * shape[1]  # one bool a pixel, in whichever pass is larger
    big, mib = (2167, 2070), 1024 * 1024  # an Eiger 4M: one frame's charge sets Auto's cap
    one = rsm_v2_chunk_bytes(big, big, 1, static_hot=False, detector_mask=True)
    assert rsm_v2_auto_chunking(big, big, 4, static_hot=False, detector_mask=True) == (1, -(-one // mib) * mib)
    assert rsm_v2_auto_chunking(big, big, 4, static_hot=False)[1] < -(-one // mib) * mib


@pytest.mark.slow
def test_the_calibrations_detector_mask_masks_every_frame_beside_the_mask_file(rsm_r2_project, tmp_path,
                                                                                monkeypatch):
    """Row B's calibration, its detector given a 12-row gap: gaps with a Mask File grid exactly as the Mask File
    of their union alone; with every correction off the gaps still mask (row B's unmasked reference moves); and
    each charge of a chunk counts the mask (Auto's cap and a typed cap, then the run's memory plan)."""
    from dataclasses import replace

    from xrd_tools.analysis import rsm_operation
    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
    from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, run_rsm_operation_v2
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2
    from tests.core.test_rsm_operation_r2_real_data import _SCIENCE

    gaps = np.zeros((195, 487), dtype=bool)
    gaps[60:72] = True
    gaps.setflags(write=False)
    file_mask = np.zeros((195, 487), dtype=bool)
    file_mask[:, 200:230] = True
    np.save(tmp_path / "file.npy", file_mask)
    np.save(tmp_path / "union.npy", gaps | file_mask)
    flags, charge = [], rsm_operation.rsm_v2_chunk_bytes

    def counted(*args, **kwargs):
        flags.append(kwargs.get("detector_mask", False))
        return charge(*args, **kwargs)

    monkeypatch.setattr(rsm_operation, "rsm_v2_chunk_bytes", counted)
    base = replace(_form(rsm_r2_project, (43,), 40),
                   geometry_asset=rsm_calibration_geometry_input(_canonical_partner(rsm_r2_project)))

    def run(name, corrections, mask, max_chunk_bytes=None):
        monkeypatch.setattr(rsm_operation, "rsm_calibration_detector_mask", lambda receipt: mask)
        form = replace(base, corrections=corrections, max_chunk_bytes=max_chunk_bytes,
                       output_path=rsm_r2_project / f"rsm-s3c-{name}.nexus")
        flags.clear()
        request = prepare_rsm_tool_v2(form).request
        assert request.plan.detector_mask is mask
        try:
            result = run_rsm_operation_v2(request)
        finally:
            Path(form.output_path).unlink(missing_ok=True)
        assert result.terminal.disposition is ModuleDisposition.COMMITTED
        assert flags and all(flag is (mask is not None) for flag in flags)
        return result.payload.result_fingerprint

    with_file = run("gaps-file", RSMFrameCorrections(mask_path=str(tmp_path / "file.npy")), gaps)
    assert with_file == run("union", RSMFrameCorrections(mask_path=str(tmp_path / "union.npy")), None)
    assert run("gaps", RSMFrameCorrections(), gaps, 256 * 1024 * 1024) != _SCIENCE[((43,), 40)]["result"]


@pytest.mark.slow
def test_scan_43_in_q_with_17_2s_whole_record_is_q_in_specs_handedness(rsm_r2_project, v3_asset):
    """Q coordinates take the same mirror (the maintainer accepted it with the fix: a scan's Q and HKL maps stay
    in one handedness, SPEC's): through the mirror, scan 43's Q grid and intensities are the canonical Q map's,
    including the exact q-bounds pass, which hands xrayutilities the mirror for its identity matrix."""
    from dataclasses import replace

    from xrd_tools.analysis.module_transaction import ModuleDisposition
    from xrd_tools.analysis.rsm_geometry_asset import rsm_calibration_geometry_input
    from xrd_tools.analysis.rsm_operation import run_rsm_operation_v2
    from xrd_tools.rsm import RSMCoordinateFrame
    from xrd_tools.analysis.rsm_values import prepare_rsm_tool_v2

    canonical_form = replace(_form(rsm_r2_project, (43,), 40),
                             coordinate_frame=RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU,
                             output_path=rsm_r2_project / "rsm-q-canonical.nexus")
    mirror_form = replace(canonical_form,
                          geometry_asset=rsm_calibration_geometry_input(
                              _mirror_partner(rsm_r2_project, v3_asset, ("mu", "chi", "nu"))),
                          output_path=rsm_r2_project / "rsm-q-17-2.nexus")
    payloads = []
    try:
        for form in (canonical_form, mirror_form):
            result = run_rsm_operation_v2(prepare_rsm_tool_v2(form).request)
            assert result.terminal.disposition is ModuleDisposition.COMMITTED
            payloads.append(result.payload)
    finally:
        for form in (canonical_form, mirror_form):
            if Path(form.output_path).exists():
                Path(form.output_path).unlink()
    canonical, mirrored = payloads
    finite = np.isfinite(canonical.intensity), np.isfinite(mirrored.intensity)
    assert int(finite[0].sum()) > 0 and np.array_equal(finite[0], finite[1])
    for (_name, axis), (_other, other) in zip(canonical.axes, mirrored.axes, strict=True):
        np.testing.assert_allclose(other, axis, rtol=1e-12, atol=0)
    differ = finite[0] & ~np.isclose(canonical.intensity, mirrored.intensity, rtol=1e-12, atol=0)
    print("Q through the mirror: finite", int(finite[0].sum()), "voxels differing", int(differ.sum()),
          "fingerprints", canonical.result_fingerprint, mirrored.result_fingerprint)
    assert int(differ.sum()) == 0
    assert mirrored.result_fingerprint == canonical.result_fingerprint  # measured: bit for bit
