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


# ── 2d/2e: on-disk validator + schema-driven fixture factory ─────────────────

def test_factory_output_is_schema_conformant(tmp_path):
    """The real writer's output validates against the declarations — and
    the validator catches every class of declared drift."""
    from tests.core.v2_fixture_factory import (
        assert_v2_conformant, make_v2_entry,
    )

    with h5py.File(tmp_path / "v2.nxs", "w") as f:
        e = make_v2_entry(f)
        assert_v2_conformant(e)


def test_validator_flags_declared_drift(tmp_path):
    from xrd_tools.io.nexus import validate_group_against_schema
    from tests.core.v2_fixture_factory import make_v2_entry

    with h5py.File(tmp_path / "v2.nxs", "w") as f:
        e = make_v2_entry(f, with_2d=False)
        g = e["integrated_1d"]

        # wrong dtype
        del g["frame_index"]
        g.create_dataset("frame_index", data=np.array([0, 1, 2], np.int32))
        probs = validate_group_against_schema(g, "integrated_1d")
        assert any("dtype" in p for p in probs)

        # row-count mismatch
        del g["frame_index"]
        g.create_dataset("frame_index", data=np.array([0, 1], np.int64))
        probs = validate_group_against_schema(g, "integrated_1d")
        assert any("row count" in p for p in probs)

        # missing required dataset
        del g["intensity"]
        probs = validate_group_against_schema(g, "integrated_1d")
        assert any("required, missing" in p for p in probs)

        # optional dataset absent is FINE
        del g["sigma"]
        del g["frame_index"]
        g.create_dataset("frame_index", data=np.array([0, 1, 2], np.int64))
        g.create_dataset("intensity",
                         data=np.zeros((3, 8), np.float32))
        assert validate_group_against_schema(g, "integrated_1d") == []


def test_validator_flags_nx_attr_drift(tmp_path):
    """The nx_attrs pass (Q-C2): the schema-routed writer's group validates
    clean, and the validator flags a mutated NX attr value AND a deleted
    required NX attr — by strict equality, naming the attr."""
    from xrd_tools.io.nexus import validate_group_against_schema, write_stitched
    from xrd_tools.core.containers import IntegrationResult1D

    s1 = IntegrationResult1D(radial=np.linspace(0.5, 5.0, 8),
                             intensity=np.ones(8, np.float32),
                             sigma=None, unit="q_A^-1")
    with h5py.File(tmp_path / "stitch.nxs", "w") as f:
        e = f.create_group("entry")
        write_stitched(e, stitched_1d=s1)
        g = e["stitched_1d"]

        # the real (schema-routed) writer's group is conformant
        assert validate_group_against_schema(g, "stitched_1d") == []

        # a mutated NX attr value is flagged, naming the attr
        g.attrs["signal"] = "bogus"
        probs = validate_group_against_schema(g, "stitched_1d")
        assert any("signal" in p for p in probs), probs
        g.attrs["signal"] = "intensity"          # restore to isolate the next

        # a deleted required NX attr is flagged as missing
        del g.attrs["NX_class"]
        probs = validate_group_against_schema(g, "stitched_1d")
        assert any("NX_class" in p and "missing" in p for p in probs), probs


#: Byte-identity tripwire (Q-C1).  SHA-256 of the canonical h5_content_signature
#: (group attrs + dataset names/dtypes/shapes/storage/value digests) of a
#: FIXED-SEED stitched+rsm write.  The schema-as-code routing of
#: write_stitched/write_rsm was proven byte-identical to the hand-written form
#: at refactor time (an empty before/after signature diff); this pin keeps every
#: FUTURE writer change byte-identical too.  Re-pin ONLY for an intentional,
#: additive format change (see CLAUDE.md — the persisted format is frozen +
#: additive-only).
_STITCH_RSM_SIGNATURE_SHA256 = (
    "207e82b24b951b2c889135aa5de3dbad1d1a11f08618aa52a4445f1b9ddfe08c"
)


def test_stitched_rsm_write_is_byte_identical(tmp_path):
    """Pin the stitched+rsm on-disk signature so a writer change that alters the
    bytes (names/dtypes/shapes/values/attrs) trips loudly."""
    import hashlib
    import json

    from tests.core.h5sig import h5_content_signature
    from xrd_tools.core.containers import (
        IntegrationResult1D, IntegrationResult2D)
    from xrd_tools.io.nexus import write_rsm, write_stitched
    from xrd_tools.rsm.volume import RSMVolume

    rng = np.random.default_rng(20260629)
    s1 = IntegrationResult1D(radial=np.linspace(0.5, 5.0, 6),
                             intensity=rng.random(6), sigma=rng.random(6),
                             unit="q_A^-1")
    s2 = IntegrationResult2D(radial=np.linspace(0.5, 5.0, 6),
                             azimuthal=np.linspace(-180, 180, 4, endpoint=False),
                             intensity=rng.random((6, 4)), sigma=None,
                             unit="q_A^-1", azimuthal_unit="chi_deg")
    rv = RSMVolume(h=np.linspace(-1.0, 1.0, 3), k=np.linspace(0.0, 2.0, 4),
                   l=np.linspace(-0.5, 0.5, 5), intensity=rng.random((3, 4, 5)))
    prov = {"plan": "stitch", "corrections": ["solid_angle"]}

    p = tmp_path / "stitch_rsm.nxs"
    with h5py.File(p, "w") as f:
        e = f.create_group("entry")
        write_stitched(e, stitched_1d=s1, stitched_2d=s2, provenance=prov)
        write_rsm(e, rv, provenance=prov)

    sig = h5_content_signature(p)
    digest = hashlib.sha256(
        json.dumps(sig, sort_keys=True).encode()).hexdigest()
    assert digest == _STITCH_RSM_SIGNATURE_SHA256, (
        "byte-identity signature drift in write_stitched/write_rsm — a writer "
        "changed the on-disk format.  Re-pin ONLY if intentional.\n"
        f"got: {digest}\n{json.dumps(sig, indent=1, sort_keys=True)}"
    )


# ── 2f: capability registry + feature detection (ADR-0002) ───────────────────

def test_capabilities_detected_on_real_files(tmp_path):
    from xrd_tools.io.schema import detect_capabilities
    from tests.core.v2_fixture_factory import make_v2_entry

    with h5py.File(tmp_path / "full.nxs", "w") as f:
        e = make_v2_entry(f, with_2d=True, with_sigma=True)
        caps = detect_capabilities(e)
        assert {"sigma_1d", "two_d_kind"} <= caps
        assert "per_frame_geometry" not in caps     # not written here
        assert "source_base" not in caps

    with h5py.File(tmp_path / "lean.nxs", "w") as f:
        e = make_v2_entry(f, with_2d=False, with_sigma=False)
        caps = detect_capabilities(e)
        assert "sigma_1d" not in caps and "two_d_kind" not in caps


def test_get_metadata_reports_capabilities(tmp_path):
    """The additive 'capabilities' key: notebooks can ask a file what
    optional features it carries instead of probing datasets."""
    from xrd_tools.io import get_metadata
    from tests.core.v2_fixture_factory import make_v2_entry

    p = tmp_path / "caps.nxs"
    with h5py.File(p, "w") as f:
        make_v2_entry(f)
    caps = get_metadata(p)["capabilities"]
    assert "sigma_1d" in caps and "two_d_kind" in caps


def test_accepted_schema_names_consumed():
    from xrd_tools.io.schema import is_known_schema_name

    assert is_known_schema_name("xrd_tools.processed_scan")
    assert is_known_schema_name(b"ssrl_xrd_tools.processed_scan")
    assert not is_known_schema_name("somebody.else")


def test_capability_registry_is_additive_only():
    """Registry hygiene: every capability says where it lives and is a
    known kind; introduced versions never exceed the current schema."""
    from xrd_tools.io.schema import CAPABILITIES

    for name, cap in CAPABILITIES.items():
        assert cap.kind in ("attr", "group", "dataset"), name
        assert cap.meaning, name
        assert cap.introduced <= PROCESSED_SCHEMA_VERSION, name
