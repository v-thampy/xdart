"""P6.7 — RSMVolume ↔ NeXus persistence (write_rsm / read_rsm).

Mirrors the stitched persistence (P5): the gridded volume is a scan-level NXdata
group (h/k/l axes + 3D intensity) + a provenance_json blob (the RSMPlan + applied
CorrectionStack).  Schema-registered + capability-gated + feature-detected.
"""
from __future__ import annotations

import numpy as np
import pytest

from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
from xrd_tools.rsm.volume import RSMVolume


def _volume(seed=0):
    rng = np.random.default_rng(seed)
    h = np.linspace(-1.0, 1.0, 5)
    k = np.linspace(0.0, 2.0, 6)
    l = np.linspace(-0.5, 0.5, 7)
    return RSMVolume(h=h, k=k, l=l, intensity=rng.random((5, 6, 7)))


def test_write_read_rsm_roundtrips(tmp_path):
    import h5py
    from xrd_tools.io.nexus import read_rsm, write_rsm

    vol = _volume(1)
    p = tmp_path / "rsm.nxs"
    with h5py.File(p, "w") as f:
        write_rsm(f.create_group("entry"), vol)

    out = read_rsm(p)
    assert isinstance(out, RSMVolume)
    assert out.shape == (5, 6, 7)
    np.testing.assert_allclose(out.h, vol.h, rtol=1e-6)
    np.testing.assert_allclose(out.k, vol.k, rtol=1e-6)
    np.testing.assert_allclose(out.l, vol.l, rtol=1e-6)
    np.testing.assert_allclose(out.intensity, vol.intensity, rtol=1e-6)
    assert out.provenance is None


def test_write_rsm_rejects_mismatched_intensity_shape(tmp_path):
    import h5py
    from types import SimpleNamespace
    from xrd_tools.io.nexus import write_rsm

    bad = SimpleNamespace(
        h=np.linspace(-1.0, 1.0, 5),
        k=np.linspace(0.0, 2.0, 6),
        l=np.linspace(-0.5, 0.5, 7),
        intensity=np.zeros((6, 5, 7)),
    )
    with h5py.File(tmp_path / "bad_rsm.nxs", "w") as f:
        with pytest.raises(ValueError, match="rsm.intensity shape"):
            write_rsm(f.create_group("entry"), bad)


def test_rsm_provenance_roundtrips(tmp_path):
    import h5py
    from xrd_tools.analysis.plans import RSMPlan
    from xrd_tools.corrections.stack import CorrectionStack
    from xrd_tools.io.nexus import read_rsm, write_rsm

    plan = RSMPlan(mapper=None, bins=(5, 6, 7), diff_motors=("mu", "del"),
                   UB=np.eye(3),
                   energy=10000.0, q_bounds=((-1, 1), (0, 2), (-0.5, 0.5)),
                   corrections=CorrectionStack(solid_angle=True,
                                               polarization_factor=0.93))
    prov = plan.provenance()
    assert prov["kind"] == "rsm" and prov["bins"] == [5, 6, 7]
    assert prov["coordinate_frame"] == "hkl"
    assert prov["corrections"]["polarization_factor"] == 0.93

    p = tmp_path / "rsm_prov.nxs"
    with h5py.File(p, "w") as f:
        write_rsm(f.create_group("entry"), _volume(2), provenance=prov)

    out = read_rsm(p)
    assert out.provenance["kind"] == "rsm"
    assert out.provenance["bins"] == [5, 6, 7]
    assert out.provenance["corrections"]["solid_angle"] is True


def test_cartesian_q_volume_roundtrips_without_hkl_aliases(tmp_path):
    import h5py
    from xrd_tools.io.nexus import read_rsm, write_rsm

    frame = RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
    axes = (
        ("qx", np.linspace(-1.0, 1.0, 3)),
        ("qy", np.linspace(-2.0, 2.0, 4)),
        ("qz", np.linspace(0.0, 3.0, 5)),
    )
    volume = RSMVolume.from_axes(
        frame,
        axes,
        np.arange(60, dtype=float).reshape(3, 4, 5),
    )
    path = tmp_path / "rsm-q.nexus"
    with h5py.File(path, "w") as handle:
        write_rsm(
            handle.create_group("entry"),
            volume,
            provenance="{}",
            bounded_artifact=True,
        )

    result = read_rsm(path)
    assert result.coordinate_frame is frame
    assert tuple(name for name, _values in result.axes) == frame.axis_names
    assert result.axis_units == tuple(
        zip(frame.axis_names, frame.axis_units, strict=True)
    )
    for (_name, expected), observed in zip(
        axes,
        result.axis_values,
        strict=True,
    ):
        np.testing.assert_allclose(observed, expected.astype(np.float32))
    np.testing.assert_allclose(result.intensity, volume.intensity.astype(np.float32))
    with pytest.raises(AttributeError):
        _ = result.h


def test_read_rsm_refuses_mixed_or_unattested_cartesian_axes(tmp_path):
    import h5py
    from xrd_tools.io.nexus import read_rsm

    missing_frame = tmp_path / "q-without-frame.nexus"
    with h5py.File(missing_frame, "w") as handle:
        group = handle.create_group("entry/rsm")
        group.create_dataset("intensity", data=np.zeros((2, 2, 2)))
        for name in ("qx", "qy", "qz"):
            dataset = group.create_dataset(name, data=np.arange(2))
            dataset.attrs["units"] = "q_A^-1"
    with pytest.raises(ValueError, match="frame descriptor"):
        read_rsm(missing_frame)

    mixed = tmp_path / "mixed-axes.nexus"
    with h5py.File(mixed, "w") as handle:
        group = handle.create_group("entry/rsm")
        group.create_dataset("intensity", data=np.zeros((2, 2, 2)))
        for name in ("h", "k", "l", "qx"):
            group.create_dataset(name, data=np.arange(2))
    with pytest.raises(ValueError, match="incomplete or mixed"):
        read_rsm(mixed)


def test_rsm_group_is_registered_capability(tmp_path):
    import h5py
    from xrd_tools.io.nexus import (
        read_rsm, validate_group_against_schema, write_rsm)
    from xrd_tools.io.schema import CAPABILITIES, SCHEMA, detect_capabilities

    assert "rsm" in CAPABILITIES and "rsm" in SCHEMA.groups

    p = tmp_path / "cap.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_rsm(e, _volume(3), provenance={"kind": "rsm"})
        assert "rsm" in detect_capabilities(e)
        assert validate_group_against_schema(e["rsm"], "rsm") == []

    # a file without the group: no capability, read_rsm raises (optional)
    q = tmp_path / "no_rsm.nxs"
    with h5py.File(q, "w") as f:
        assert "rsm" not in detect_capabilities(f.create_group("entry"))
    with pytest.raises(KeyError):
        read_rsm(q)
