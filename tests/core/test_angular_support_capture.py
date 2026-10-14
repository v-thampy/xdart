"""COVERAGE-SUPPORT-20261013 step-3 regressions of the Run capture's reuse (found by its own resource measurements).

An energy scan is one geometry whose radial line table is rebuilt a bounded number of times in either wavelength
direction; a grazing-incidence incidence scan reuses its detector-plane cut and matches a fresh footprint; workers
sharing one Run compute a new geometry and its table once. Spies wrap the real kernel functions; nothing is faked.
"""
import math
import threading
from unittest import mock

import numpy as np
import pytest

from tests.core.test_angular_support import CONVERGE, _planar


def _band_mask(shape):
    mask = np.zeros(shape, bool)
    mask[:, 140:143] = True
    mask[300:303, :] = True
    return mask


@pytest.mark.parametrize("direction", (1, -1))
def test_an_energy_scan_is_one_geometry_with_a_bounded_number_of_table_builds(direction):
    from xrd_tools.analysis import angular_support as kernel
    from xrd_tools.analysis.angular_support import (
        AngularSupportPolicy, AngularSupportRequest, RunSupportCapture, SupportPose, admitted_window)
    ai = _planar()
    mask = _band_mask((400, 500))
    digest = SupportPose.from_integrator(ai, mask).mask_digest
    centres = np.linspace(1.0, 9.0, 400)
    edges = np.r_[centres - .5 * (centres[1] - centres[0]), centres[-1] + .5 * (centres[1] - centres[0])]
    window = admitted_window(None, reducer="int_standard")
    capture = RunSupportCapture(AngularSupportPolicy())
    wavelengths = np.linspace(1.0e-10, 1.1e-10, 40)[::direction]
    with mock.patch.object(kernel, "support_line_table", wraps=kernel.support_line_table) as built:
        profiles = []
        for wavelength in wavelengths:
            ai.wavelength = float(wavelength)
            profiles.append(capture.profile(integrator=ai, mask=mask, mask_digest=digest, convention="detector_chi",
                                            radial_unit="q_A^-1", centres=centres, wavelength_m=ai.wavelength,
                                            window=window))
    assert capture.geometry_keys == 1
    assert built.call_count <= 3, f"{built.call_count} table builds for one energy scan"
    # a 10 % wavelength range narrows the needed spacing by at most ~10 %; one halving covers it
    first = built.call_args_list[0].kwargs["spacing"]
    assert all(call.kwargs["spacing"] >= .4 * first for call in built.call_args_list), "the spacing collapsed"
    for wavelength, profile in zip(wavelengths, profiles):
        direct = kernel.angular_support(AngularSupportRequest(
            poses=(SupportPose.from_integrator(ai, mask),), convention="detector_chi", radial_unit="q_A^-1",
            bin_edges=edges, wavelength_m=float(wavelength), window=window))
        assert profile.available and direct.available
        assert np.max(np.abs(profile.supported_deg - direct.supported_deg)) <= CONVERGE * 360


@pytest.mark.parametrize("orientation", (1, 4))
def test_an_incidence_scan_reuses_its_detector_cut_and_matches_a_fresh_footprint(orientation):
    from xrd_tools.analysis import angular_support as kernel
    from xrd_tools.analysis.angular_support import SupportGeometryCache, SupportPose, support_footprint
    ai = _planar(shape=(300, 400), poni=(0.005013, 0.020007))
    mask = np.zeros((300, 400), bool)
    mask[:, 300:305] = True
    cache = SupportGeometryCache()
    poses = [SupportPose.from_integrator(ai, mask, gi=(incident, 0.0, orientation)) for incident in (0.2, 1.0, 2.0)]
    with mock.patch.object(kernel, "_gi_pieces", wraps=kernel._gi_pieces) as cuts, \
            mock.patch.object(kernel, "_integrator", wraps=kernel._integrator) as loads:
        scanned = [support_footprint(pose, "gi_chi", cache=cache) for pose in poses]
        assert cuts.call_count == 1 and loads.call_count == 1, "one cut and one geometry for the whole scan"
        tilted = SupportPose.from_integrator(ai, mask, gi=(1.0, 4.0, orientation))
        support_footprint(tilted, "gi_chi", cache=cache)
        assert cuts.call_count == 2, "a tilted pose's cut depends on the incidence angle and is never reused"
    for pose, cached in zip(poses, scanned):
        fresh = support_footprint(pose, "gi_chi")
        assert cached.reason is None and fresh.reason is None
        for name in ("t0", "p0", "t1", "p1", "sign"):
            assert np.array_equal(getattr(cached, name), getattr(fresh, name)), (pose.gi, name)


def test_workers_sharing_a_run_compute_a_new_geometry_and_its_table_once():
    from xrd_tools.analysis import angular_support as kernel
    from xrd_tools.analysis.angular_support import AngularSupportPolicy, RunSupportCapture, SupportPose, admitted_window
    ai = _planar()
    mask = _band_mask((400, 500))
    digest = SupportPose.from_integrator(ai, mask).mask_digest
    centres = np.linspace(1.0, 9.0, 400)
    capture = RunSupportCapture(AngularSupportPolicy())
    start = threading.Barrier(4)
    results, errors = [None] * 4, []

    def worker(index):
        try:
            start.wait()
            results[index] = capture.profile(
                integrator=ai, mask=mask, mask_digest=digest, convention="detector_chi", radial_unit="q_A^-1",
                centres=centres, wavelength_m=ai.wavelength, window=admitted_window(None, reducer="int_standard"))
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    with mock.patch.object(kernel, "support_footprint", wraps=kernel.support_footprint) as footprints, \
            mock.patch.object(kernel, "support_line_table", wraps=kernel.support_line_table) as tables:
        threads = [threading.Thread(target=worker, args=(index,)) for index in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    assert not errors and all(result is not None and result.available for result in results)
    assert footprints.call_count == 1 and tables.call_count == 1
    for result in results[1:]:
        assert np.array_equal(result.supported_deg, results[0].supported_deg)
    assert math.isclose(results[0].window_deg, 360.0)
