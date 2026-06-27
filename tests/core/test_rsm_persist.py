"""P6.7 — RSMVolume ↔ NeXus persistence (write_rsm / read_rsm).

Mirrors the stitched persistence (P5): the gridded volume is a scan-level NXdata
group (h/k/l axes + 3D intensity) + a provenance_json blob (the RSMPlan + applied
CorrectionStack).  Schema-registered + capability-gated + feature-detected.
"""
from __future__ import annotations

import numpy as np
import pytest

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


def test_rsm_provenance_roundtrips(tmp_path):
    import h5py
    from xrd_tools.analysis.plans import RSMPlan
    from xrd_tools.corrections.stack import CorrectionStack
    from xrd_tools.io.nexus import read_rsm, write_rsm

    plan = RSMPlan(mapper=None, bins=(5, 6, 7), diff_motors=("mu", "del"),
                   energy=10000.0, q_bounds=((-1, 1), (0, 2), (-0.5, 0.5)),
                   corrections=CorrectionStack(solid_angle=True,
                                               polarization_factor=0.93))
    prov = plan.provenance()
    assert prov["kind"] == "rsm" and prov["bins"] == [5, 6, 7]
    assert prov["corrections"]["polarization_factor"] == 0.93

    p = tmp_path / "rsm_prov.nxs"
    with h5py.File(p, "w") as f:
        write_rsm(f.create_group("entry"), _volume(2), provenance=prov)

    out = read_rsm(p)
    assert out.provenance["kind"] == "rsm"
    assert out.provenance["bins"] == [5, 6, 7]
    assert out.provenance["corrections"]["solid_angle"] is True


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
