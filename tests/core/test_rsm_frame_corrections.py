"""RSM's per-frame corrections, shared with Stitch (AUD-9 stage 2b C4; design §4).

The maintainer, 2026-10-01: "masks should be applied when specified. Other processing options like thresholding,
monitor etc should also apply"; decision 5, solid angle and polarization (on by default in the form). A plan
without corrections keeps its identity and its science: row A is scan 43 with every correction explicitly off,
reproducing the reference ``8a88826f…b324``.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.module_transaction import ModuleDisposition
from xrd_tools.analysis.rsm_operation import RSMFrameCorrections, run_rsm_operation_v2
from xrd_tools.corrections.frame_preparation import flat_detector_solid_angle
from xdart.gui.tools.rsm_values import RSMToolPreflightRefused, prepare_rsm_tool_v2
from tests.core.test_rsm_operation_r2_real_data import _form, rsm_r2_project  # noqa: F401 (the fixture)

REFERENCE = "8a88826f85eda95bf9fb4eb9cd043b7b822e0827e4ec3d285203f9256f43b324"


def test_the_solid_angle_is_pyfais_cos_cubed_for_an_untilted_detector():
    """pyFAI's relative solid angle from xrayutilities' header (poni = (cch + 0.5) × pixel): cos³α, α the angle
    between a pixel's ray and the detector normal, 1 at the direct-beam pixel. pyFAI integrates each pixel's area,
    so a point pixel's cos³α agrees to 6e-10 at 0.172 mm and 1 m (measured 5.6e-10)."""
    shape, cch, pixel, distance = (195, 487), (97.0, 243.0), (0.172, 0.172), 1014.7173
    values = flat_detector_solid_angle(shape, distance=distance, cch=cch, pixel_size=pixel)
    rows, columns = np.mgrid[:shape[0], :shape[1]]
    cos = distance / np.sqrt(distance ** 2 + ((rows - cch[0]) * pixel[0]) ** 2 + ((columns - cch[1]) * pixel[1]) ** 2)
    np.testing.assert_allclose(values, cos ** 3, rtol=1e-9, atol=0)
    assert values[97, 243] == pytest.approx(1.0, rel=1e-15, abs=0)


def _request(project, name, corrections=None, **form_changes):
    form = replace(_form(project, (43,), 40), output_path=Path(project) / f"rsm-c4-{name}.nexus", **form_changes)
    if corrections is not None:
        form = replace(form, corrections=corrections)
    return prepare_rsm_tool_v2(form).request


@pytest.mark.slow
def test_the_polarization_weight_takes_the_nu_circle_as_the_vertical(rsm_r2_project):
    """The canonical psic asset turns nu about x and del about z, with the beam along y: x is the vertical. The
    direct-beam pixel scatters by 2θ = δ in the vertical plane at del = δ (P = ½(1 + cos²δ + f·sin²δ)) and in
    the horizontal one at nu = δ (P = ½(1 + cos²δ − f·sin²δ)); a wrong vertical swaps them."""
    from xrd_tools.analysis.rsm_geometry_asset import rsm_effective_pixel_q_map
    from xrd_tools.analysis.rsm_operation import _rsm_chunk_weights, _rsm_correction_state
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    request = _request(rsm_r2_project, "pol", RSMFrameCorrections(polarization_factor=0.99))
    effective = request.plan.effective_geometry
    header = effective.detector_header
    state = _rsm_correction_state(request.plan, header, effective.diffractometer_projection)
    mapper = rsm_effective_pixel_q_map(effective)
    delta, factor = 30.0, 0.99
    row, column = int(header.cch1), int(header.cch2)
    assert (row, column) == (header.cch1, header.cch2)
    expected = {"del": 0.5 * (1 + math.cos(math.radians(delta)) ** 2 + factor * math.sin(math.radians(delta)) ** 2),
                "nu": 0.5 * (1 + math.cos(math.radians(delta)) ** 2 - factor * math.sin(math.radians(delta)) ** 2)}
    single = {}
    with xu_runtime_session(effective.runtime_requirements) as session:
        for circle, (nu, dl) in (("del", (0.0, delta)), ("nu", (delta, 0.0))):
            angles = tuple(np.zeros(1) for _ in range(4)) + (np.array([nu]), np.array([dl]))
            weights = _rsm_chunk_weights(state, mapper, session, angles, 13000.007, None)
            assert not weights.flags.writeable
            assert weights[row, column] == pytest.approx(expected[circle], rel=1e-9, abs=0), circle
            single[circle] = weights
        # a chunk at two detector poses: one weight per frame, which the gridder takes
        from xrd_tools.rsm.gridding import RSMGridChunkLease, StreamingGridder
        angles = tuple(np.zeros(2) for _ in range(4)) + (np.array([0.0, delta]), np.array([delta, 0.0]))
        weights = _rsm_chunk_weights(state, mapper, session, angles, 13000.007, None)
        assert weights.shape == (2,) + single["del"].shape and weights.flags.c_contiguous
        np.testing.assert_array_equal(weights[0], single["del"])
        np.testing.assert_array_equal(weights[1], single["nu"])
        gridder = StreamingGridder(mapper, (4, 4, 4), runtime_session=session,
                                   coordinate_frame=request.plan.coordinate_frame)
        gridder.set_bounds((-8.0, 8.0), (-8.0, 8.0), (-8.0, 8.0))
        frames = np.ones((2, int(header.Nch1), int(header.Nch2)))
        lease = RSMGridChunkLease.from_arrays(frames.copy(), frames.copy(), 2)
        gridder.add_leased(lease, angles, 13000.007, UB=np.eye(3), weight=weights)
        assert gridder.n_frames_processed == 2


@pytest.mark.slow
def test_corrections_enter_identity_and_provenance_only_when_set(rsm_r2_project):
    plain = _request(rsm_r2_project, "plain")
    off = _request(rsm_r2_project, "off", RSMFrameCorrections())  # every field at its default: none
    on = _request(rsm_r2_project, "on", RSMFrameCorrections(solid_angle=True, polarization_factor=0.99))
    assert plain.plan.fingerprint == off.plan.fingerprint != on.plan.fingerprint
    assert "frame_corrections" not in plain.provenance["plan"]
    assert on.provenance["plan"]["frame_corrections"] == {
        "mask": None, "threshold": None, "dark": None, "flat": None, "solid_angle": True,
        "polarization_factor": 0.99}
    assert "authored-arbitrary-detector-masks-and-mask-editing-ui" not in plain.provenance["holds"]


@pytest.mark.slow
def test_a_mask_or_correction_image_of_another_shape_or_an_unusable_dark_is_refused(rsm_r2_project, tmp_path):
    mask = tmp_path / "small.npy"
    np.save(mask, np.zeros((4, 5), dtype=bool))
    with pytest.raises(RSMToolPreflightRefused) as refused:
        _request(rsm_r2_project, "mask", RSMFrameCorrections(mask_path=str(mask)))
    assert refused.value.code == "RSM_MASK_INVALID"
    import tifffile
    dark = tmp_path / "dark.tif"
    tifffile.imwrite(dark, np.full((195, 487), np.nan, dtype=np.float32))
    with pytest.raises(RSMToolPreflightRefused) as refused:
        _request(rsm_r2_project, "dark", RSMFrameCorrections(dark_path=str(dark)))
    assert refused.value.code == "RSM_CORRECTION_IMAGE_INVALID" and "no finite pixel" in str(refused.value)


@pytest.mark.slow
def test_row_a_scan_43_with_every_correction_off_is_the_reference_and_the_corrections_change_it(
        rsm_r2_project, tmp_path):
    """Row A: scan 43 at 40³ with every correction explicitly off reproduces 8a88826f…b324 bit for bit. With the
    solid angle and polarization the result differs everywhere it is defined and covers the same voxels; a Mask
    File removes the masked pixels' voxels."""
    results = {}
    mask = tmp_path / "half.npy"
    values = np.zeros((195, 487), dtype=bool)
    values[:, :243] = True
    np.save(mask, values)
    for name, corrections in (("off", RSMFrameCorrections()),
                              ("on", RSMFrameCorrections(solid_angle=True, polarization_factor=0.99)),
                              ("mask", RSMFrameCorrections(mask_path=str(mask)))):
        result = run_rsm_operation_v2(_request(rsm_r2_project, f"run-{name}", corrections))
        assert result.terminal.disposition is ModuleDisposition.COMMITTED, result.terminal
        results[name] = result.payload
    assert results["off"].result_fingerprint == REFERENCE
    off, on = (np.asarray(results[name].intensity) for name in ("off", "on"))
    assert np.array_equal(np.isfinite(off), np.isfinite(on))
    assert not np.allclose(off[np.isfinite(off)], on[np.isfinite(on)])
    assert 0 < np.isfinite(np.asarray(results["mask"].intensity)).sum() < np.isfinite(off).sum()
    assert results["on"].result_fingerprint != REFERENCE
    assert json.loads(results["on"].provenance_json)["plan"]["frame_corrections"]["solid_angle"] is True
