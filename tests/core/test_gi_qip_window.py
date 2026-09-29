"""Int's GI q_ip cut: its window (the range Int labels Qoop) selects pixels by q_oop.

The q_ip mode passed its unit as pyFAI's out-of-plane unit, so pyFAI binned a q_ip × q_ip grid and
the window filtered q_ip (found 2026-09-29; fix approved by the maintainer). Real pyFAI throughout.
"""
import numpy as np

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
