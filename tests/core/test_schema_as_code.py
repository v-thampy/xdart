"""6b — schema-as-code: the declarative SCHEMA is the single source of truth
for the processed-scan layout, and the writers/validators/readers consume it.

Every string pinned here is PERSISTED in existing user files.  If one of
these assertions fails, the change breaks reading of every file already on
disk — fix the code, never the pin.
"""
import h5py
import numpy as np

from xrd_tools.io import SCHEMA
from xrd_tools.io.schema import (
    DTYPE_ATTR,
    INTEGRATED_ROW_ALIGNED,
    MONOTONIC_ATTR,
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
    SOURCE_BASE_ATTR,
)


# ── persisted strings are frozen ─────────────────────────────────────────────

def test_persisted_attribute_keys_are_frozen():
    assert SCHEMA_NAME_ATTR == "ssrl_schema"
    assert SCHEMA_VERSION_ATTR == "ssrl_schema_version"
    assert DTYPE_ATTR == "ssrl_dtype"
    assert MONOTONIC_ATTR == "_frame_index_strictly_increasing"
    assert SOURCE_BASE_ATTR == "source_base"


def test_schema_identity_and_back_compat_names():
    assert SCHEMA.version == PROCESSED_SCHEMA_VERSION == 2
    assert SCHEMA.name == PROCESSED_SCHEMA_NAME == "xrd_tools.processed_scan"
    # files written before the monorepo rename carry the old name
    assert "ssrl_xrd_tools.processed_scan" in SCHEMA.accepted_names
    assert SCHEMA.name in SCHEMA.accepted_names


def test_row_aligned_and_axis_declarations():
    assert INTEGRATED_ROW_ALIGNED == {"frame_index", "intensity", "sigma"}
    assert SCHEMA.groups["integrated_1d"].axes == ("q",)
    assert SCHEMA.groups["integrated_2d"].axes == ("q", "chi")
    for g in ("integrated_1d", "integrated_2d"):
        assert SCHEMA.groups[g].row_aligned == INTEGRATED_ROW_ALIGNED
        # axis datasets are shared across rows, never row-sliced
        assert not set(SCHEMA.groups[g].axes) & SCHEMA.groups[g].row_aligned


# ── the writer stamps exactly what SCHEMA declares ───────────────────────────

def test_writer_stamp_matches_schema(tmp_path):
    from xrd_tools.io import open_nexus_writer

    p = tmp_path / "s.nxs"
    f = open_nexus_writer(p, overwrite=True)
    try:
        e = f["entry"]
        assert e.attrs[SCHEMA_NAME_ATTR] == SCHEMA.name
        assert int(e.attrs[SCHEMA_VERSION_ATTR]) == SCHEMA.version
    finally:
        f.close()


# ── row surgery slices exactly the schema's row-aligned set ──────────────────

def test_drop_integrated_rows_slices_only_schema_row_set(tmp_path):
    from xrd_tools.io.nexus_record import drop_integrated_rows

    p = tmp_path / "s.nxs"
    with h5py.File(p, "w") as f:
        g = f.create_group("entry/integrated_1d")
        g.create_dataset("frame_index", data=np.array([0, 1, 2], dtype=np.int64))
        g.create_dataset("intensity", data=np.arange(15.0).reshape(3, 5))
        g.create_dataset("sigma", data=np.ones((3, 5)))
        g.create_dataset("q", data=np.linspace(0.0, 1.0, 5))
        # row-shaped but NOT in the schema's row set: must survive unsliced
        g.create_dataset("not_per_frame", data=np.arange(3.0))
        drop_integrated_rows(f, "entry/integrated_1d", [1])
        g = f["entry/integrated_1d"]
        np.testing.assert_array_equal(g["frame_index"][()], [0, 2])
        assert g["intensity"].shape == (2, 5)
        assert g["sigma"].shape == (2, 5)
        assert g["q"].shape == (5,)              # shared axis untouched
        assert g["not_per_frame"].shape == (3,)  # not declared -> not sliced


# ── 2a: full DatasetSpec declarations ────────────────────────────────────────

def test_dataset_specs_agree_with_legacy_row_sets():
    """The per-dataset declarations and the legacy fast sets must say the
    same thing — one source of truth, two views."""
    for gname, g in SCHEMA.groups.items():
        declared_rows = {n for n, d in g.datasets.items() if d.row_aligned}
        assert declared_rows == set(g.row_aligned), gname
        # every axis is declared, non-row, role="axis"
        for ax in g.axes:
            spec = g.datasets[ax]
            assert spec.role == "axis" and not spec.row_aligned, (gname, ax)


def test_dataset_spec_facts_are_frozen():
    """Persisted facts: names, dtypes, required-ness of the integrated
    stacks.  Changing any of these breaks files already on disk."""
    g1 = SCHEMA.groups["integrated_1d"].datasets
    g2 = SCHEMA.groups["integrated_2d"].datasets
    assert set(g1) == {"intensity", "q", "frame_index", "sigma"}
    assert set(g2) == {"intensity", "q", "chi", "frame_index", "sigma"}
    for g in (g1, g2):
        assert g["intensity"].dtype == "float32" and g["intensity"].compressed
        assert g["frame_index"].dtype == "int64"
        assert not g["sigma"].required          # optional on disk
        assert g["sigma"].compressed
    assert g1["q"].units_from == "radial_unit"
    assert g2["chi"].units_from == "azimuthal_unit"
    geo = SCHEMA.groups["per_frame_geometry"].datasets
    assert geo["incident_angle"].units_from == "deg"
    assert all(geo[k].units_from == "rad" for k in ("rot1", "rot2", "rot3"))


def test_group_nx_attrs_declared():
    assert SCHEMA.groups["integrated_1d"].nx_attrs["axes"] == (
        "frame_index", "q")
    assert SCHEMA.groups["integrated_2d"].nx_attrs["axes"] == (
        "frame_index", "chi", "q")
    assert SCHEMA.groups["per_frame_geometry"].nx_attrs["NX_class"] == (
        "NXcollection")
