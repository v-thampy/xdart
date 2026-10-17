"""COVERAGE-SUPPORT-20261013 step-3 regression: XU support values against an independent projective ring model.

XU's detector is flat, so its pixel-corner directions are a central projection of a uniform grid: a projective map
fitted to XU's own corners (DLT) sends any direction to detector pixel coordinates. Each radial line's support is
then the measured share of densely sampled ring directions that land on the detector, compared with the kernel's
exact line table. Low-angle poses whose pixel edges span more than 0.25° of χ were refused before step 3.
"""
import json
import math

import numpy as np
import pytest


def _directions(two_theta, chi):
    return np.stack((np.cos(two_theta), np.sin(two_theta) * np.cos(chi), np.sin(two_theta) * np.sin(chi)), axis=-1)


def _projective_map(two_theta, chi, stride=5):
    """3x3 H with H·v ∝ (column, row, 1) for every corner direction v (DLT on a strided subset, normalised)."""
    rows, cols = np.mgrid[0:two_theta.shape[0]:stride, 0:two_theta.shape[1]:stride]
    v = _directions(two_theta[rows, cols], chi[rows, cols]).reshape(-1, 3)
    x, y = cols.ravel() / 1000.0, rows.ravel() / 1000.0
    zero = np.zeros_like(v)
    system = np.concatenate((np.hstack((v, zero, -x[:, None] * v)), np.hstack((zero, v, -y[:, None] * v))))
    h = np.linalg.svd(system, full_matrices=False)[2][-1].reshape(3, 3)
    return np.diag([1000.0, 1000.0, 1.0]) @ h


def _ring_coverage_deg(h, shape, lines, samples=72000):
    chi = np.linspace(-math.pi, math.pi, samples, endpoint=False) + math.pi / samples
    out = []
    for line in lines:
        mapped = _directions(np.full(samples, line), chi) @ h.T
        w = mapped[:, 2]
        col, row = mapped[:, 0] / w, mapped[:, 1] / w
        on = (w * np.sign(h[2] @ np.array([1.0, 0.0, 0.0])) > 0) & (col >= 0) & (col < shape[1]) & (row >= 0) \
            & (row < shape[0])
        out.append(360.0 * on.mean())
    return np.array(out)


@pytest.mark.parametrize("nu, delta", ((-5.0, 6.0), (-10.0, 14.0)))
def test_xu_line_support_matches_a_projective_ring_model(tmp_path, nu, delta):
    from tests.core.test_stitch_operation import _xu_prepared
    from xrd_tools.analysis.angular_support import (admitted_window, footprint_from_corners, support_line_table,
                                                    xu_corner_grid)
    from xrd_tools.analysis.xu_stitch_calibration import canonical_surface_resource_bytes
    from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
    from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
    request = _xu_prepared(tmp_path)
    asset = json.loads(canonical_surface_resource_bytes())
    with xu_runtime_session() as runtime:
        geometry = resolve_xu_stitch_effective_geometry(request.plan.calibration, runtime)
        two_theta, chi = xu_corner_grid(geometry, asset["acquisition"], nu=nu, del_=delta)
    footprint = footprint_from_corners("pose", two_theta, chi, None)
    assert footprint.reason is None, footprint.detail
    h = _projective_map(two_theta, chi)
    corners = _directions(two_theta, chi).reshape(-1, 3) @ h.T
    rows, cols = np.mgrid[0:two_theta.shape[0], 0:two_theta.shape[1]]
    fit = np.hypot(corners[:, 0] / corners[:, 2] - cols.ravel(), corners[:, 1] / corners[:, 2] - rows.ravel())
    assert fit.max() < 1e-3, "fixture: XU's corners are a flat detector's projective grid"
    lo, hi = float(two_theta.min()), float(two_theta.max())
    spacing = (hi - lo) / 400
    table = support_line_table((footprint,), admitted_window(None, reducer="stitch_xu"), spacing=spacing,
                               start=lo - spacing, stop=hi + spacing)
    picks = np.linspace(0, table.lines.size - 1, 40).astype(int)
    expected = _ring_coverage_deg(h, (two_theta.shape[0] - 1, two_theta.shape[1] - 1), table.lines[picks])
    got = np.degrees(table.admitted[picks])
    assert np.max(np.abs(got - expected)) <= 0.05, np.max(np.abs(got - expected))
