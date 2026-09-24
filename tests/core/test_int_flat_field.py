"""Int flat field: pyFAI applies ``ReductionPlan.flat`` in the normalization on every
integration path; unusable flat pixels are NaN, which pyFAI masks."""
import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.corrections import prepare_flatfield
from xrd_tools.integrate.calibration import poni_to_integrator
from xrd_tools.reduction import (
    Frame, GIMode, Integration1DPlan, Integration2DPlan, ReductionPlan, Scan, run_reduction,
)

SHAPE = (195, 487)
_PONI = PONI(dist=0.15, poni1=0.002, poni2=0.04, wavelength=1.0e-10, detector="Pilatus100k")
_IMAGE = 50 + np.random.default_rng(7).poisson(40, SHAPE).astype(float)


def _reduce(plan):
    scan = Scan("flat", [Frame(0, image=_IMAGE.copy(), metadata={})], poni=_PONI,
                integrator=poni_to_integrator(_PONI))
    result = run_reduction(plan, scan, executor=1)
    assert not result.failed, result.error
    return result.frames[0]


def test_prepare_flatfield_masks_unusable_pixels_without_rescaling():
    flat = np.full((4, 5), 2.0)
    flat[0, 0], flat[0, 1], flat[0, 2], flat[0, 3] = 0.0, -1.0, np.nan, 0.15
    prepared = prepare_flatfield(flat)
    assert np.isnan(prepared[0, :4]).all()          # zero, negative, NaN, below 10 % of median
    assert np.all(prepared[1:] == 2.0)              # usable pixels unchanged, not normalised
    assert np.isfinite(prepare_flatfield(flat, min_fraction=0)[0, 3])
    with pytest.raises(ValueError):
        prepare_flatfield(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        prepare_flatfield(flat, min_fraction=1.5)


def test_one_frame_with_a_flat_equals_the_integrator_with_that_flat():
    from xrd_tools.integrate.single import integrate_1d
    rng = np.random.default_rng(3)
    flat = rng.uniform(0.8, 1.2, SHAPE)
    flat[10:20, 30:60] = 0.0
    prepared = prepare_flatfield(flat)
    reduced = _reduce(ReductionPlan(integration_1d=Integration1DPlan(npt=60, error_model="poisson"),
                                    integration_2d=None, flat=prepared)).result_1d
    direct = integrate_1d(_IMAGE, poni_to_integrator(_PONI), npt=60, method="csr",
                          error_model="poisson", flat=prepared)
    np.testing.assert_allclose(reduced.intensity, direct.intensity, rtol=1e-12)
    np.testing.assert_allclose(reduced.sigma, direct.sigma, rtol=1e-12)


@pytest.mark.parametrize("output", ("1d", "2d", "chi", "gi_q_total", "gi_q_ip"))
def test_a_flat_of_two_halves_every_output(output):
    gi = None
    one_d, two_d = Integration1DPlan(npt=40), None
    if output == "2d":
        one_d, two_d = None, Integration2DPlan(npt_rad=30, npt_azim=24)
    elif output == "chi":
        one_d = Integration1DPlan(npt=40, unit="chi_deg", radial_range=(0.5, 2.0))
    elif output.startswith("gi_"):
        gi = GIMode(incident_angle=0.2, mode_1d=output[3:])
    plain = _reduce(ReductionPlan(integration_1d=one_d, integration_2d=two_d, gi=gi))
    halved = _reduce(ReductionPlan(integration_1d=one_d, integration_2d=two_d, gi=gi,
                                   flat=np.full(SHAPE, 2.0)))
    before = plain.result_2d if two_d is not None else plain.result_1d
    after = halved.result_2d if two_d is not None else halved.result_1d
    finite = np.isfinite(before.intensity)
    assert finite.any()
    np.testing.assert_allclose(after.intensity[finite], before.intensity[finite] / 2, rtol=1e-6)


def test_a_flat_turns_off_the_float32_fast_path_and_corrects_thumbnails():
    from xrd_tools.reduction.core import _can_use_owned_float32_csr, _thumbnail_corrected_image
    raw = np.full(SHAPE, 100, dtype=np.uint16)
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40, method="csr"), integration_2d=None)
    assert _can_use_owned_float32_csr(raw, None, plan, integrator_input_safe=True)
    flat = np.full(SHAPE, 4.0)
    flat[0, 0] = np.nan
    with_flat = ReductionPlan(integration_1d=Integration1DPlan(npt=40, method="csr"),
                              integration_2d=None, flat=flat)
    assert not _can_use_owned_float32_csr(raw, None, with_flat, integrator_input_safe=True)
    thumbnail = _thumbnail_corrected_image(raw, None, flat)
    assert thumbnail[1, 1] == 25.0 and np.isnan(thumbnail[0, 0])


def test_average_applies_records_and_authenticates_its_flat(tmp_path):
    import hashlib

    import h5py
    import tifffile

    from xrd_tools.io import get_1d
    from xrd_tools.reduction import AverageScanRecipe
    from xrd_tools.reduction import average as module
    from xrd_tools.session.experiment_state import (
        CalibrationState, FactStatus, DetectorImageState, PoniValues,
    )
    from xrd_tools.sources.selection import image_series_spec

    flat_path = tmp_path / "flat.npy"
    np.save(flat_path, np.full((4, 4), 2.0))
    digest = hashlib.sha256(flat_path.read_bytes()).hexdigest()

    def run(name, flat):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(1, 4):
            tifffile.imwrite(folder / f"scan_{index:04d}.tif", np.full((4, 4), 20 * index, np.uint32))
        calibration = CalibrationState(
            PoniValues(0.2, 0.001, 0.001, 0.0, 0.0, 0.0, 1e-10), "Detector",
            {"pixel1": 1e-4, "pixel2": 1e-4, "max_shape": [4, 4], "orientation": 3},
            status=FactStatus.PRESENT, flat=flat)
        runner = module.AverageScanRunner(AverageScanRecipe(
            image_series_spec(folder / "scan_0001.tif", metadata_format=None),
            folder / "scan.nexus",
            ReductionPlan(integration_1d=Integration1DPlan(npt=4, method="numpy")),
            calibration=calibration))
        result = runner.start()
        runner.close()
        return result

    plain = run("plain", DetectorImageState.absent())
    flat = run("flat", DetectorImageState(str(flat_path), digest, (4, 4), FactStatus.PRESENT))
    assert plain.disposition == flat.disposition == "COMMITTED", (plain, flat)
    before, after = get_1d(plain.target, frame=1), get_1d(flat.target, frame=1)
    finite = np.isfinite(before.intensity)
    assert finite.any()
    np.testing.assert_allclose(after.intensity[finite], before.intensity[finite] / 2, rtol=1e-6)
    with h5py.File(flat.target, "r") as handle:
        stored = handle["entry/instrument/detector/flatfield"]
        assert stored.attrs["sha256"] == hashlib.sha256(
            np.full((4, 4), 2.0, np.float32).tobytes()).hexdigest()
    with h5py.File(plain.target, "r") as handle:
        assert "flatfield" not in handle["entry/instrument/detector"]

    changed = run("changed", DetectorImageState(str(flat_path), "0" * 64, (4, 4), FactStatus.PRESENT))
    assert changed.disposition != "COMMITTED"
    assert "AVERAGE_FLAT_DIGEST_CHANGED" in f"{changed.diagnostic_code} {changed.diagnostic}"


def test_a_read_only_flat_is_accepted():
    flat = np.full(SHAPE, 2.0, dtype=np.float32)
    flat.setflags(write=False)
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40), integration_2d=None, flat=flat)
    assert plan.flat.flags.writeable
    halved = _reduce(plan).result_1d
    plain = _reduce(ReductionPlan(integration_1d=Integration1DPlan(npt=40), integration_2d=None)).result_1d
    np.testing.assert_allclose(halved.intensity, plain.intensity / 2, rtol=1e-6)
    with pytest.raises(ValueError):
        ReductionPlan(flat=np.ones(4))
