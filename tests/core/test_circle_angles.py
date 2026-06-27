"""P6.3 — assemble_circle_angles: circle_motors → per-frame xu angle vector.

The shared per-frame sample-angle assembly (the "one wiring task" both the RSM
pipeline and the deferred xu_hist stitch need).  The circle ORDER + signs — the
q-convention — are carried in the preset's circle_motors, NOT invented here; the
gate pins that the assembly faithfully reproduces the legacy explicit-diff_motors
path (mechanics), with the absolute convention validated against real data later.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.core.geometry import (
    AngleMapping,
    DetectorHeader,
    Diffractometer,
    PixelQMap,
    assemble_circle_angles,
)

pd = pytest.importorskip("pandas")


def _scan(n=4):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "mu": rng.uniform(0, 1, n), "eta": rng.uniform(0, 0.5, n),
        "chi": rng.uniform(-1, 1, n), "phi": rng.uniform(0, 2, n),
        "nu": rng.uniform(0, 10, n), "del": rng.uniform(0, 20, n),
    })

    class _Scan:
        scan_data = df
        frame_indices = list(range(n))
    return _Scan()


_PSIC_MOTORS = ("mu", "eta", "chi", "phi", "nu", "del")


def test_matches_legacy_explicit_diff_motors_path():
    from xrd_tools.rsm.pipeline import _angles_for_indices
    scan = _scan()
    psic = Diffractometer.psic()
    assert tuple(m.source_motor for m in psic.circle_motors) == _PSIC_MOTORS

    got = assemble_circle_angles(psic, scan)
    ref = _angles_for_indices(scan, _PSIC_MOTORS)
    assert len(got) == len(ref) == 6
    for g, r in zip(got, ref):
        np.testing.assert_array_equal(g, r)


def test_index_selection_matches_legacy():
    from xrd_tools.rsm.pipeline import _angles_for_indices
    scan = _scan()
    psic = Diffractometer.psic()
    idx = [3, 0, 2]
    got = assemble_circle_angles(psic, scan, indices=idx)
    ref = _angles_for_indices(scan, _PSIC_MOTORS, idx)
    for g, r in zip(got, ref):
        np.testing.assert_array_equal(g, r)


def test_applies_sign_and_offset_from_circle_motors():
    """A non-trivial AngleMapping (sign/offset) is honoured — sign·motor+offset."""
    scan = _scan()
    psic = Diffractometer.psic()
    # flip eta's sign and offset it (the kind of fitted refinement a preset carries)
    new_circles = list(psic.circle_motors)
    new_circles[1] = AngleMapping(source_motor="eta", sign=-1.0, offset=5.0)
    diff = Diffractometer(preset="psic", sample_circles=psic.sample_circles,
                          detector_circles=psic.detector_circles,
                          circle_motors=tuple(new_circles))
    got = assemble_circle_angles(diff, scan)
    np.testing.assert_allclose(got[1], -1.0 * scan.scan_data["eta"].to_numpy() + 5.0)


def test_q_identity_through_pixel_q():
    """The assembled angles drive pixel_q identically to the explicit list — the
    convention identity THROUGH the actual q-conversion."""
    pytest.importorskip("xrayutilities")
    from xrd_tools.rsm.pipeline import _angles_for_indices
    scan = _scan()
    psic = Diffractometer.psic()
    h = DetectorHeader(cch1=20, cch2=24, pwidth1=0.172, pwidth2=0.172,
                       distance=500.0, Nch1=32, Nch2=40)
    mapper = PixelQMap(diff_config=psic, header=h)
    UB = np.eye(3)
    qa = mapper.pixel_q(assemble_circle_angles(psic, scan), 10000.0, UB=UB)
    qb = mapper.pixel_q(_angles_for_indices(scan, _PSIC_MOTORS), 10000.0, UB=UB)
    for a, b in zip(qa, qb):
        np.testing.assert_array_equal(a, b)


def test_empty_circle_motors_raises():
    diff = Diffractometer(preset="custom")  # bare: circle_motors empty
    with pytest.raises(ValueError, match="circle_motors is empty"):
        assemble_circle_angles(diff, _scan())


def test_missing_motor_raises():
    scan = _scan()
    del_col = scan.scan_data.drop(columns=["nu"])

    class _Scan:
        scan_data = del_col
        frame_indices = list(range(len(del_col)))
    with pytest.raises(KeyError, match="nu"):
        assemble_circle_angles(Diffractometer.psic(), _Scan())
