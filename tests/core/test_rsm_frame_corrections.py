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


@pytest.mark.slow
def test_the_corrections_are_charged_in_the_memory_plan(rsm_r2_project):
    """S2: the per-frame corrections' allocations are charged with each science chunk, at or above what the run
    allocates. Measured with tracemalloc at Pilatus 100K, every correction on: one frame at one pose 6,899,301 B,
    four frames at four poses 9,159,084 B (S2.2, probe s2_weights_peak.py); with S2.3's banded weights (probe
    s23_weights_peak.py, ``logs/s23-weights-peak.log``) 7,654,033 B and 9,912,980 B (one band covers the crop
    there; the weight array is allocated before the band). A chunk cap between the plain and the corrected
    charge refuses the corrected plan only."""
    from types import SimpleNamespace
    from xrd_tools.analysis.rsm_operation import RSMBoundCorrections, _rsm_v2_correction_bytes
    member = SimpleNamespace(detector_shape=(195, 487), cropped_shape=(194, 486))
    full = RSMBoundCorrections(RSMFrameCorrections(solid_angle=True, polarization_factor=0.99), None, object(),
                               object())
    plan = SimpleNamespace(corrections=full)
    assert _rsm_v2_correction_bytes(plan, 1, member) >= 7_654_033
    assert _rsm_v2_correction_bytes(plan, 4, member) >= 9_912_980
    assert _rsm_v2_correction_bytes(SimpleNamespace(corrections=None), 4, member) == 0
    plain = _request(rsm_r2_project, "mem-plain")
    from xrd_tools.analysis.rsm_operation import _require_rsm_v2_memory_plan
    _require_rsm_v2_memory_plan(plain)
    corrected = _request(rsm_r2_project, "mem-on", RSMFrameCorrections(solid_angle=True, polarization_factor=0.99))
    _require_rsm_v2_memory_plan(corrected)
    cap = 56 * 1024 * 1024  # the preset chunk of 8 frames: plain 49,381,800 B, corrected 63,813,381 B (S2.3)
    tight_plain = _request(rsm_r2_project, "mem-plain-tight", None, max_chunk_bytes=cap)
    _require_rsm_v2_memory_plan(tight_plain)
    with pytest.raises(RSMToolPreflightRefused) as refused:  # the memory plan at Preview (S3a), before any decode
        _request(rsm_r2_project, "mem-on-tight", RSMFrameCorrections(solid_angle=True, polarization_factor=0.99),
                 max_chunk_bytes=cap)
    assert refused.value.code == "Q_MEMORY_LIMIT_EXCEEDED"
    assert "one chunk of 8 frames needs 61 MiB; Max chunk MiB is 56" in str(refused.value)


@pytest.mark.slow
def test_banded_polarization_weights_are_bit_identical_to_one_whole_detector_call(rsm_r2_project, monkeypatch):
    """S2.3: the polarization is computed in bands of rows, each through ``pixel_q``'s ``rows`` (xrayutilities'
    own per-call roi, the crop's beam centre unchanged), so every weight equals the unbanded formula's (C4's:
    the whole crop's q, ``polarization_weight``, ``frame_weights``) bit for bit. A non-integer beam centre (an
    integer one would pass by any shifting), crops with an offset and with negative ends, and a chunk whose
    poses repeat."""
    from xrd_tools.analysis import rsm_operation
    from xrd_tools.analysis.rsm_geometry_asset import rsm_effective_pixel_q_map
    from xrd_tools.corrections.frame_preparation import frame_weights, polarization_weight
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    flat = Path(rsm_r2_project) / "s23-flat.npy"
    np.save(flat, 1.0 + 0.01 * np.arange(195 * 487, dtype=np.float64).reshape(195, 487) / (195 * 487))
    request = _request(rsm_r2_project, "band", RSMFrameCorrections(solid_angle=True, polarization_factor=0.99,
                                                                   flat_path=str(flat)))
    effective = request.plan.effective_geometry
    state = rsm_operation._rsm_correction_state(request.plan, effective.detector_header,
                                                effective.diffractometer_projection)
    mapper = rsm_effective_pixel_q_map(effective)
    mapper = replace(mapper, header=replace(mapper.header, cch1=mapper.header.cch1 + 0.37,
                                            cch2=mapper.header.cch2 + 0.61))
    angles = tuple(np.zeros(3) for _ in range(4)) + (np.array([0.0, 2.5, 0.0]), np.array([20.0, 20.0, 20.0]))
    monkeypatch.setattr(rsm_operation, "_RSM_WEIGHT_BAND_PIXELS", 1000)  # two or three rows a band
    with xu_runtime_session(effective.runtime_requirements) as session:
        xu_module, _numpy = session.active_modules()
        wavelength = float(xu_module.en2lam(13000.007))
        for roi in (None, (5, 190, 7, 480), (0, -1, 0, -1)):
            banded = rsm_operation._rsm_chunk_weights(state, mapper, session, angles, 13000.007, roi)
            r0, r1, c0, c1 = roi if roi is not None else (0, 195, 0, 487)
            r1, c1 = 195 + r1 if r1 < 0 else r1, 487 + c1 if c1 < 0 else c1
            whole = []
            for nu, delta in zip(angles[4], angles[5]):
                lab = tuple(np.zeros(1) for _ in range(4)) + (np.array([nu]), np.array([delta]))
                qx, qy, qz = mapper.pixel_q(lab, 13000.007, UB=np.eye(3), roi=roi, runtime_session=session)
                polarization = polarization_weight(np.asarray(qx)[0], np.asarray(qy)[0], np.asarray(qz)[0],
                                                   factor=0.99, wavelength_A=wavelength, beam=state.axes[0],
                                                   vertical=state.axes[1])
                whole.append(frame_weights(state.solid[r0:r1, c0:c1], polarization=polarization,
                                           flat=state.flat[r0:r1, c0:c1]))
            assert banded.shape == (3, r1 - r0, c1 - c0) and not banded.flags.writeable, roi
            assert banded.tobytes() == np.stack(whole).tobytes(), roi
            assert banded[0].tobytes() == banded[2].tobytes()  # a repeated pose, computed once


@pytest.mark.slow
def test_weights_without_polarization_are_one_array_for_every_pose(rsm_r2_project):
    """S2.3: the solid angle and the flat do not depend on the detector pose, so a chunk at several poses takes
    one detector array (the gridder broadcasts it), not one copy per frame."""
    from xrd_tools.analysis import rsm_operation
    from xrd_tools.analysis.rsm_geometry_asset import rsm_effective_pixel_q_map
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.rsm.gridding import RSMGridChunkLease, StreamingGridder
    request = _request(rsm_r2_project, "nopol", RSMFrameCorrections(solid_angle=True))
    effective = request.plan.effective_geometry
    header = effective.detector_header
    state = rsm_operation._rsm_correction_state(request.plan, header, effective.diffractometer_projection)
    mapper = rsm_effective_pixel_q_map(effective)
    angles = tuple(np.zeros(2) for _ in range(4)) + (np.array([0.0, 2.5]), np.array([20.0, 30.0]))
    with xu_runtime_session(effective.runtime_requirements) as session:
        weights = rsm_operation._rsm_chunk_weights(state, mapper, session, angles, 13000.007, None)
        assert weights.shape == (195, 487) and not weights.flags.writeable and weights.flags.c_contiguous
        assert weights.tobytes() == state.solid.tobytes() and weights is not state.solid
        gridder = StreamingGridder(mapper, (4, 4, 4), runtime_session=session,
                                   coordinate_frame=request.plan.coordinate_frame)
        gridder.set_bounds((-8.0, 8.0), (-8.0, 8.0), (-8.0, 8.0))
        frames = np.ones((2, int(header.Nch1), int(header.Nch2)))
        lease = RSMGridChunkLease.from_arrays(frames.copy(), frames.copy(), 2)
        gridder.add_leased(lease, angles, 13000.007, UB=np.eye(3), weight=weights)
        assert gridder.n_frames_processed == 2


def test_the_default_corrections_charge_one_band_not_the_detector_at_eiger_size():
    """S2.3: at an Eiger-4M-size detector (2167 x 2070) the default corrections (solid angle, polarization 0.99)
    charge the resident solid angle and mask, one weight array and ONE band of the polarization's working set
    (``_RSM_WEIGHT_BAND_PIXELS``), not the detector: 97,122,330 B for one frame, where S2.2's charge was
    435,111,930 B (``logs/s23-eiger-charge-*.log``). Measured (``logs/s23-weights-peak.log``): the banded
    weights' traced peak 54,962,644 B, with the resident solid angle 35,885,520 B. The plain one-frame chunk alone
    is 296,055,540 B there, above the preset's 256 MiB cap: an Eiger-4M RSM needs Max chunk MiB raised whatever
    the corrections."""
    from types import SimpleNamespace
    from xrd_tools.analysis.rsm_operation import _RSM_WEIGHT_BAND_PIXELS, _rsm_v2_correction_bytes
    shape = (2167, 2070)
    pixels = shape[0] * shape[1]
    member = SimpleNamespace(detector_shape=shape, cropped_shape=shape)
    default = RSMFrameCorrections(solid_angle=True, polarization_factor=0.99)
    plan = SimpleNamespace(corrections=SimpleNamespace(corrections=default, dark_receipt=None, flat_receipt=None))
    one = _rsm_v2_correction_bytes(plan, 1, member)
    resident_and_weights = 9 * pixels + 8 * pixels  # the mask and solid angle, one weight array
    assert one - resident_and_weights <= 80 * _RSM_WEIGHT_BAND_PIXELS
    assert one <= 100_000_000
    assert one >= 54_962_644 + 35_885_520  # the measured peak, with the resident solid angle


@pytest.mark.slow
def test_pixel_q_rows_is_the_whole_calls_band_byte_for_byte(rsm_r2_project):
    """S2.3's ``PixelQMap.pixel_q(rows=...)`` contract, for any caller: a band of rows equals the same rows of
    the whole call byte for byte (a non-integer beam centre, an offset crop, two frames at two detector poses),
    and a band that is empty or out of range, or a configured ``roi`` or ``Nav``, is refused."""
    from xrd_tools.analysis.rsm_geometry_asset import rsm_effective_pixel_q_map
    from xrd_tools.core.geometry import DetectorHeader, DiffractometerConfig, PixelQMap
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    request = _request(rsm_r2_project, "rows")
    effective = request.plan.effective_geometry
    mapper = rsm_effective_pixel_q_map(effective)
    mapper = replace(mapper, header=replace(mapper.header, cch1=mapper.header.cch1 + 0.37,
                                            cch2=mapper.header.cch2 + 0.61))
    angles = (np.array([1.0, 2.0]), np.array([10.0, 11.0]), np.array([90.0, 91.0]), np.array([0.0, 5.0]),
              np.array([0.0, 2.5]), np.array([20.0, 30.0]))
    roi = (5, 190, 7, 480)
    with xu_runtime_session(effective.runtime_requirements) as session:
        whole = mapper.pixel_q(angles, 13000.007, UB=np.eye(3), roi=roi, runtime_session=session)
        for start, stop in ((0, 3), (40, 41), (100, 185)):
            band = mapper.pixel_q(angles, 13000.007, UB=np.eye(3), roi=roi, runtime_session=session,
                                  rows=(start, stop))
            for axis, (part, full) in enumerate(zip(band, whole)):
                assert part.shape == (2, stop - start, 473), (start, axis)
                assert part.tobytes() == np.ascontiguousarray(full[:, start:stop]).tobytes(), (start, axis)
    header = DetectorHeader(2.5, 3.5, 0.1, 0.1, 100.0, 8, 9)
    for config, rows in ((DiffractometerConfig(), (3, 3)), (DiffractometerConfig(), (0, 9)),
                         (DiffractometerConfig(ang2q_kwargs={"Nav": (2, 2)}), (0, 4)),
                         (DiffractometerConfig(ang2q_kwargs={"roi": [0, 4, 0, 9]}), (0, 4))):
        with pytest.raises(ValueError, match="one band"):
            PixelQMap(config, header).pixel_q([np.array([0.0])], 12000.0, rows=rows)


def test_auto_chunking_is_the_preset_at_pilatus_and_one_frame_at_eiger_size():
    """S3a: Auto Max chunk MiB is the preset's 256 MiB, or one frame's charge rounded up to the next MiB when that
    is larger, and the chunk then holds as many frames as fit (at most Chunk). Scan 43's Pilatus 100K crop keeps
    the preset (eight frames, 256 MiB) whatever the corrections; an Eiger 4M (2167 x 2070) takes one frame a chunk
    and a cap sized to it, where the 256 MiB preset refused even a plain frame (282.3 MiB with all-frame hot).
    Figures from ``logs/s3a-auto-chunking.log``."""
    from xrd_tools.analysis.rsm_operation import rsm_v2_auto_chunking, rsm_v2_chunk_bytes
    default = RSMFrameCorrections(solid_angle=True, polarization_factor=0.99)
    mib = 1024 * 1024
    for corrections in (None, default):
        for static_hot in (True, False):
            assert rsm_v2_auto_chunking((195, 487), (194, 486), 8, static_hot=static_hot,
                                        corrections=corrections) == (8, 256 * mib)
    cases = {(True, None): 283, (True, True): 375, (False, None): 279, (False, True): 371}
    for (static_hot, corrected), expected_mib in cases.items():
        corrections = default if corrected else None
        frames, cap = rsm_v2_auto_chunking((2167, 2070), (2167, 2070), 8, static_hot=static_hot,
                                           corrections=corrections)
        one = rsm_v2_chunk_bytes((2167, 2070), (2167, 2070), 1, static_hot=static_hot, corrections=corrections)
        two = rsm_v2_chunk_bytes((2167, 2070), (2167, 2070), 2, static_hot=static_hot, corrections=corrections)
        assert (frames, cap) == (1, expected_mib * mib) and one <= cap < two


@pytest.mark.slow
def test_auto_at_preview_is_the_presets_plan_at_pilatus_and_a_small_cap_names_its_need(rsm_r2_project):
    """S3a: Auto (``max_chunk_bytes=None``) resolves at Preview to the same plan as the preset's explicit 256 MiB
    at Pilatus size (plan fingerprints equal), so a default Pilatus run is unchanged; a user's cap below what a
    chunk needs is refused at Preview, not after the sources open, naming the MiB it needs."""
    from xrd_tools.analysis.rsm_operation import rsm_v2_chunk_bytes
    explicit = _request(rsm_r2_project, "auto-explicit")
    auto = _request(rsm_r2_project, "auto", max_chunk_bytes=None)
    assert (auto.plan.chunk_size, auto.plan.max_chunk_bytes) == (8, 256 * 1024 * 1024)
    assert auto.plan.fingerprint == explicit.plan.fingerprint
    member, bound = auto.preflight.members[0], auto.plan.corrections
    one = rsm_v2_chunk_bytes(member.detector_shape, member.cropped_shape, 1,
                             static_hot=auto.plan.conditioning.static_hot_threshold is not None,
                             corrections=None if bound is None else bound.corrections)
    with pytest.raises(RSMToolPreflightRefused) as refused:  # before any member is read
        _request(rsm_r2_project, "tiny", max_chunk_bytes=1024 * 1024)
    assert refused.value.code == "Q_MEMORY_LIMIT_EXCEEDED"
    assert f"one frame needs {-(-one // (1024 * 1024))} MiB; Max chunk MiB is 1" in str(refused.value)


@pytest.mark.slow
def test_the_correction_states_charge_bounds_what_the_run_allocates_before_the_chunks(rsm_r2_project):
    """S3a: the run decodes its mask, dark and flat and computes pyFAI's solid angle once, before the chunks
    (``_rsm_correction_state``); that transient is not a chunk's, so the memory plan charges it in the run's
    resident peak (``_rsm_v2_correction_state_bytes``). Measured here with tracemalloc on the real state at
    scan 43's Pilatus 100K: the charge is at or above it (the solid angle alone measured 48.1 B/px there and
    48.0 at 2167 x 2070, ``logs/s3a-solid-angle-transient.log``; charged 49)."""
    import tracemalloc
    from xrd_tools.analysis.rsm_operation import _rsm_correction_state, _rsm_v2_correction_state_bytes
    request = _request(rsm_r2_project, "state", RSMFrameCorrections(solid_angle=True, polarization_factor=0.99))
    effective = request.plan.effective_geometry
    _rsm_correction_state(request.plan, effective.detector_header, effective.diffractometer_projection)  # warm
    tracemalloc.start()
    state = _rsm_correction_state(request.plan, effective.detector_header, effective.diffractometer_projection)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert state.solid is not None
    charged = _rsm_v2_correction_state_bytes(request.plan, request.preflight.members)
    assert charged >= peak, (charged, peak)
    assert _rsm_v2_correction_state_bytes(_request(rsm_r2_project, "state-off").plan,
                                          request.preflight.members) == 0
