"""Int's GI q_ip cut: its window (the range Int labels Qoop) selects pixels by q_oop.

The q_ip mode passed its unit as pyFAI's out-of-plane unit, so pyFAI binned a q_ip × q_ip grid and
the window filtered q_ip (found 2026-09-29; fix approved by the maintainer). Real pyFAI throughout.
"""
import numpy as np
import pytest

from xrd_tools.core.containers import PONI
from xrd_tools.integrate.detector_mask import mask_with_detector
from xrd_tools.integrate.gid import _effective_gi_params, create_fiber_integrator
from xrd_tools.reduction.core import GI1DMode, GIMode, Integration1DPlan, _run_gi_1d

SHAPE = (195, 487)  # Pilatus 100k
ANGLE = .2
NPT = 40


def _fiber():
    poni = PONI(dist=.15, poni1=.002, poni2=.04, wavelength=1.0e-10, detector="Pilatus100k")
    return create_fiber_integrator(poni, incident_angle=ANGLE)


def _cut(fi, image, **ranges):
    """pyFAI's 1-D fiber cut with q_ip in plane and q_oop out of plane, called as Int calls it."""
    angle, tilt, orientation = _effective_gi_params(fi, None, None, None)
    return fi.integrate1d_grazing_incidence(
        image, npt_ip=NPT, npt_oop=NPT, unit_ip="qip_A^-1", unit_oop="qoop_A^-1", method="no",
        mask=mask_with_detector(fi, None), incident_angle=angle, tilt_angle=tilt,
        sample_orientation=orientation, **ranges)


def test_the_q_ip_windows_filter_q_oop():
    image = np.random.default_rng(3).uniform(10., 100., SHAPE)
    fi = _fiber()
    q_oop = np.asarray(_cut(fi, image, vertical_integration=True).integrated)  # the frame's q_oop extent
    q_ip = np.asarray(_cut(fi, image, vertical_integration=False).integrated)
    window = (float(q_oop.min()), float(np.median(q_oop)))  # the lower half of the frame in q_oop
    band = (float(q_ip.min()), float(q_ip.max()))
    gi = GIMode(mode_1d=GI1DMode.Q_IP, method="no")

    def run(window):
        plan = Integration1DPlan(npt=NPT, radial_range=band, azimuth_range=window)
        return _run_gi_1d(image, fi, plan, gi, mask=None, incident_angle=ANGLE, normalization_factor=None)

    result = run(window)
    reference = _cut(fi, image, ip_range=band, oop_range=window, vertical_integration=False)
    assert result.unit == "qip_A^-1"
    np.testing.assert_array_equal(result.radial, np.asarray(reference.integrated))
    filled = np.isfinite(result.intensity)
    assert filled.any()
    np.testing.assert_array_equal(result.intensity[filled], np.asarray(reference.intensity)[filled])
    # a window below every pixel's q_oop leaves the cut empty
    assert np.isnan(run((window[0] - 2., window[0] - 1.)).intensity).all()


@pytest.mark.parametrize("method", ["no", "cython", "csr", "bbox", "splitpixel"])
def test_an_explicit_window_counts_only_its_own_pixels_as_stitchs_cuts(method):
    """Deferred item 8 (GI-WIN): pyFAI's histogram admits pixels up to one bin below an explicit range
    and a hair above it; Int's GI cuts now mask exactly the closed window, as Stitch's (8548d780). With
    1000 outside the window, both cuts equal those of an image that is 1 everywhere. Before, every method
    but the plain histogram ("no") counted pixels far outside the window (ratios near 500)."""
    fi = _fiber()
    angle, tilt, orientation = _effective_gi_params(fi, None, None, None)
    from pyFAI.units import get_unit_fiber
    coordinates = {}
    for name in ("qip_A^-1", "qoop_A^-1"):  # pyFAI's coordinates at this frame's incidence and tilt
        unit = get_unit_fiber(name, incident_angle=angle, tilt_angle=tilt, sample_orientation=orientation)
        fi.reset_integrator(unit.incident_angle, unit.tilt_angle, unit.sample_orientation)
        coordinates[name] = np.asarray(fi.array_from_unit(SHAPE, "center", unit, scale=True))
    q_ip, q_oop = coordinates["qip_A^-1"], coordinates["qoop_A^-1"]
    band = (float(np.quantile(q_ip, .3)), float(np.quantile(q_ip, .7)))
    window = (float(np.quantile(q_oop, .3)), float(np.quantile(q_oop, .7)))
    inside = (q_ip >= band[0]) & (q_ip <= band[1]) & (q_oop >= window[0]) & (q_oop <= window[1])
    image = np.where(inside, 1., 1000.)
    for mode in (GI1DMode.Q_IP, GI1DMode.Q_OOP):
        plan = Integration1DPlan(npt=NPT, radial_range=band, azimuth_range=window)  # ip, oop

        def cut(values):
            return _run_gi_1d(values, fi, plan, GIMode(mode_1d=mode, method=method), mask=None,
                              incident_angle=ANGLE, normalization_factor=None)
        result, ones = cut(image), cut(np.ones(SHAPE))
        filled = np.isfinite(ones.intensity)
        assert filled.sum() > NPT // 2, mode
        np.testing.assert_array_equal(np.isfinite(result.intensity), filled)
        np.testing.assert_allclose(result.intensity[filled], ones.intensity[filled], rtol=1e-12,
                                   err_msg=str(mode))
