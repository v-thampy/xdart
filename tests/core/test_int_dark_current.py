"""Int dark current: pyFAI subtracts ``ReductionPlan.dark`` on every integration path;
non-finite dark pixels are NaN, which pyFAI masks. With the Poisson error model the
dark's counts add to the variance, as in the MultiGeometry Stitch."""
import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.corrections import prepare_dark
from xrd_tools.integrate.calibration import poni_to_integrator
from xrd_tools.reduction import (
    Frame, GIMode, Integration1DPlan, Integration2DPlan, ReductionPlan, Scan, run_reduction,
)

SHAPE = (195, 487)
_PONI = PONI(dist=0.15, poni1=0.002, poni2=0.04, wavelength=1.0e-10, detector="Pilatus100k")
_IMAGE = 50 + np.random.default_rng(7).poisson(40, SHAPE).astype(float)


def _reduce(plan, image=_IMAGE, **frame):
    scan = Scan("dark", [Frame(0, image=image.copy(), metadata={}, **frame)], poni=_PONI,
                integrator=poni_to_integrator(_PONI))
    result = run_reduction(plan, scan, executor=1)
    assert not result.failed, result.error
    return result.frames[0]


def test_prepare_dark_masks_only_non_finite_pixels():
    dark = np.full((3, 4), 5.0)
    dark[0, 0], dark[0, 1], dark[0, 2] = np.nan, np.inf, -2.0
    prepared = prepare_dark(dark)
    assert np.isnan(prepared[0, :2]).all()
    assert prepared[0, 2] == -2.0 and np.all(prepared[1:] == 5.0)  # not clipped or rescaled
    with pytest.raises(ValueError):
        prepare_dark(np.full((2, 2), np.nan))
    with pytest.raises(ValueError):
        prepare_dark(np.ones(4))


def test_one_frame_with_a_dark_equals_the_integrator_with_that_dark_and_its_noise():
    from xrd_tools.integrate.single import integrate_1d
    dark = prepare_dark(np.random.default_rng(3).uniform(5.0, 15.0, SHAPE))
    reduced = _reduce(ReductionPlan(integration_1d=Integration1DPlan(npt=60, error_model="poisson"),
                                    integration_2d=None, dark=dark)).result_1d
    direct = integrate_1d(_IMAGE, poni_to_integrator(_PONI), npt=60, method="csr", dark=dark,
                          variance=np.maximum(_IMAGE, 1.0) + dark)
    np.testing.assert_allclose(reduced.intensity, direct.intensity, rtol=1e-12)
    np.testing.assert_allclose(reduced.sigma, direct.sigma, rtol=1e-12)
    # pyFAI alone treats the dark as noiseless
    noiseless = integrate_1d(_IMAGE, poni_to_integrator(_PONI), npt=60, method="csr", dark=dark,
                             error_model="poisson")
    finite = np.isfinite(reduced.sigma) & (reduced.sigma > 0)
    assert finite.any() and np.all(reduced.sigma[finite] > noiseless.sigma[finite])


def test_dark_and_background_noise_add_up():
    from xrd_tools.integrate.single import integrate_1d
    dark = np.full(SHAPE, 8.0)
    background = np.full(SHAPE, 12.0)
    reduced = _reduce(ReductionPlan(integration_1d=Integration1DPlan(npt=60, error_model="poisson"),
                                    integration_2d=None, dark=dark),
                      background=background, background_variance_factor=1.0).result_1d
    direct = integrate_1d(_IMAGE - background, poni_to_integrator(_PONI), npt=60, method="csr",
                          dark=dark, variance=np.maximum(_IMAGE, 1.0) + dark + background)
    np.testing.assert_allclose(reduced.intensity, direct.intensity, rtol=1e-12)
    np.testing.assert_allclose(reduced.sigma, direct.sigma, rtol=1e-12)


@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_a_background_with_a_dark_loses_its_own_dark(scale):
    """raw − dark − scale·(background − dark): a background image holds the dark too.

    The measured dark then enters with weight 1 − scale, so at scale 1 it cancels.
    """
    from xrd_tools.integrate.single import integrate_1d
    dark = np.full(SHAPE, 8.0)
    background = np.full(SHAPE, 20.0)  # the background image, its dark included
    reduced = _reduce(ReductionPlan(integration_1d=Integration1DPlan(npt=60, error_model="poisson"),
                                    integration_2d=None, dark=dark),
                      background=scale * background, background_variance_factor=scale,
                      background_scale=scale).result_1d
    variance = np.maximum(_IMAGE, 1.0) + (1 - scale) ** 2 * dark + scale * scale * background
    direct = integrate_1d(_IMAGE - scale * (background - dark), poni_to_integrator(_PONI), npt=60,
                          method="csr", dark=dark, variance=variance)
    np.testing.assert_allclose(reduced.intensity, direct.intensity, rtol=1e-12)
    np.testing.assert_allclose(reduced.sigma, direct.sigma, rtol=1e-12)


@pytest.mark.parametrize("output", ("1d", "2d", "chi", "gi_q_total", "gi_q_ip"))
def test_a_dark_is_subtracted_from_every_output(output):
    gi = None
    one_d, two_d = Integration1DPlan(npt=40), None
    if output == "2d":
        one_d, two_d = None, Integration2DPlan(npt_rad=30, npt_azim=24)
    elif output == "chi":
        one_d = Integration1DPlan(npt=40, unit="chi_deg", radial_range=(0.5, 2.0))
    elif output.startswith("gi_"):
        gi = GIMode(incident_angle=0.2, mode_1d=output[3:])
    subtracted = _reduce(ReductionPlan(integration_1d=one_d, integration_2d=two_d, gi=gi),
                         image=_IMAGE - 30.0)
    with_dark = _reduce(ReductionPlan(integration_1d=one_d, integration_2d=two_d, gi=gi,
                                      dark=np.full(SHAPE, 30.0)))
    before = subtracted.result_2d if two_d is not None else subtracted.result_1d
    after = with_dark.result_2d if two_d is not None else with_dark.result_1d
    finite = np.isfinite(before.intensity)
    assert finite.any()
    np.testing.assert_allclose(after.intensity[finite], before.intensity[finite], rtol=1e-6)


def test_a_dark_turns_off_the_float32_fast_path_and_corrects_thumbnails():
    from xrd_tools.reduction.core import _can_use_owned_float32_csr, _thumbnail_corrected_image
    raw = np.full(SHAPE, 100, dtype=np.uint16)
    dark = np.full(SHAPE, 20.0)
    dark[0, 0] = np.nan
    with_dark = ReductionPlan(integration_1d=Integration1DPlan(npt=40, method="csr"),
                              integration_2d=None, dark=dark)
    assert not _can_use_owned_float32_csr(raw, None, with_dark, integrator_input_safe=True)
    thumbnail = _thumbnail_corrected_image(raw, 30.0, np.full(SHAPE, 2.0), dark)
    assert thumbnail[1, 1] == 25.0 and np.isnan(thumbnail[0, 0])  # (100 - 20 - 30) / 2


def test_a_read_only_dark_is_accepted():
    dark = np.full(SHAPE, 30.0, dtype=np.float32)
    dark.setflags(write=False)
    plan = ReductionPlan(integration_1d=Integration1DPlan(npt=40), integration_2d=None, dark=dark)
    assert plan.dark.flags.writeable
    with pytest.raises(ValueError):
        ReductionPlan(dark=np.ones(4))


def test_average_applies_records_and_authenticates_its_dark(tmp_path):
    import hashlib

    import h5py
    import tifffile

    from xrd_tools.io import get_1d
    from xrd_tools.reduction import AverageScanRecipe
    from xrd_tools.reduction import average as module
    from xrd_tools.session.experiment_state import (
        CalibrationState, DetectorImageState, FactStatus, PoniValues,
    )
    from xrd_tools.sources.selection import image_series_spec

    dark_path = tmp_path / "dark.npy"
    np.save(dark_path, np.full((4, 4), 10.0))
    digest = hashlib.sha256(dark_path.read_bytes()).hexdigest()

    def run(name, dark):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(1, 4):
            tifffile.imwrite(folder / f"scan_{index:04d}.tif", np.full((4, 4), 20 * index, np.uint32))
        calibration = CalibrationState(
            PoniValues(0.2, 0.001, 0.001, 0.0, 0.0, 0.0, 1e-10), "Detector",
            {"pixel1": 1e-4, "pixel2": 1e-4, "max_shape": [4, 4], "orientation": 3},
            status=FactStatus.PRESENT, dark=dark)
        runner = module.AverageScanRunner(AverageScanRecipe(
            image_series_spec(folder / "scan_0001.tif", metadata_format=None),
            folder / "scan.nexus",
            ReductionPlan(integration_1d=Integration1DPlan(npt=4, method="numpy", error_model="poisson")),
            calibration=calibration))
        result = runner.start()
        runner.close()
        return result

    plain = run("plain", DetectorImageState.absent())
    dark = run("dark", DetectorImageState(str(dark_path), digest, (4, 4), FactStatus.PRESENT))
    assert plain.disposition == dark.disposition == "COMMITTED", (plain, dark)
    before, after = get_1d(plain.target, frame=1), get_1d(dark.target, frame=1)
    finite = np.isfinite(before.intensity)
    assert finite.any()
    # mean of 20, 40, 60 is 40; the dark leaves 30
    np.testing.assert_allclose(after.intensity[finite], before.intensity[finite] * 0.75, rtol=1e-6)
    # variance of the mean, 40 / 3, plus the one dark's 10 counts
    np.testing.assert_allclose((after.sigma / before.sigma)[finite],
                               np.sqrt((40 / 3 + 10) / (40 / 3)), rtol=1e-6)
    with h5py.File(dark.target, "r") as handle:
        stored = handle["entry/instrument/detector/darkcurrent"]
        assert stored.attrs["sha256"] == hashlib.sha256(
            np.full((4, 4), 10.0, np.float32).tobytes()).hexdigest()
    with h5py.File(plain.target, "r") as handle:
        assert "darkcurrent" not in handle["entry/instrument/detector"]

    changed = run("changed", DetectorImageState(str(dark_path), "0" * 64, (4, 4), FactStatus.PRESENT))
    assert changed.disposition != "COMMITTED"
    assert "AVERAGE_DARK_DIGEST_CHANGED" in f"{changed.diagnostic_code} {changed.diagnostic}"


def test_a_run_with_background_and_dark_records_how_they_combine(tmp_path):
    from dataclasses import replace
    from xrd_tools.reduction import FrameBackgroundPlan
    from xrd_tools.session.run_configuration import RunIntent

    background = tmp_path / "background.tif"
    background.write_bytes(b"background")
    dark = str(tmp_path / "dark.npy")
    plain = FrameBackgroundPlan(mode="Single BG File", locator=str(background))
    subtracted = RunIntent(background=plain, dark_file=dark).freeze()
    as_is = RunIntent(background=replace(plain, dark_corrected=True), dark_file=dark).freeze()
    assert subtracted.as_provenance()["background_dark"] == "subtracted"
    assert as_is.as_provenance()["background_dark"] == "as_is"
    assert subtracted.fingerprint != as_is.fingerprint
    for alone in (RunIntent(background=plain).freeze(), RunIntent(dark_file=dark).freeze()):
        assert "background_dark" not in alone.as_provenance()
    # without a dark the mark changes nothing, so it is not recorded
    unmarked = RunIntent(background=replace(plain, dark_corrected=True)).freeze()
    assert "dark_corrected" not in unmarked.as_provenance()["background"]
    assert unmarked.fingerprint == RunIntent(background=plain).freeze().fingerprint
