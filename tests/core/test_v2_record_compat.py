"""6a gate: the refactored writer preserves the normalized v2 record contract.

The committed fixture is a normalized legacy signature rematerialized by the
headless fixture builder.  It is not claimed to be a byte-identical historical
artifact.  Every normalized scientific/storage fact stays identical; the test
separately asserts the intentional shared-writer metadata ownership deltas.
"""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

FIXTURE = (
    Path(__file__).parent / "fixtures" /
    "v2_record_signature_normalized_legacy.json"
)


def test_migration_names_the_truthful_normalized_legacy_fixture():
    migration = Path(__file__).parents[2] / "MIGRATION.md"
    text = migration.read_text(encoding="utf-8")
    assert "v2_record_signature_pre6a.json" not in text
    assert "v2_record_signature_normalized_legacy.json" in text


def test_v2_reference_fixture_is_headless_and_xdart_independent(tmp_path):
    """The frozen core fixture must neither name nor import GUI-side owners."""
    fixture_module = Path(__file__).parent / "_v2_record_fixture.py"
    tree = ast.parse(fixture_module.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    forbidden = sorted(
        name for name in imported
        if name == "xdart" or name.startswith("xdart.")
        or name == "tests.xdart" or name.startswith("tests.xdart.")
    )
    assert forbidden == []

    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class RejectGuiImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (fullname == "xdart" or fullname.startswith("xdart.")
                        or fullname == "tests.xdart"
                        or fullname.startswith("tests.xdart.")):
                    raise RuntimeError(f"forbidden core-fixture import: {fullname}")
                return None

        sys.meta_path.insert(0, RejectGuiImports())
        from tests.core._v2_record_fixture import write_reference_scan
        write_reference_scan(sys.argv[1], sys.argv[2])
        forbidden = sorted(
            name for name in sys.modules
            if name == "xdart" or name.startswith("xdart.")
            or name == "tests.xdart" or name.startswith("tests.xdart.")
        )
        if forbidden:
            raise RuntimeError(f"forbidden modules loaded: {forbidden}")
        """
    )
    repo = Path(__file__).parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "headless.nexus"),
         str(tmp_path / "source")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_v2_record_content_matches_normalized_legacy_signature(tmp_path):
    import h5py

    from tests.core._v2_record_fixture import write_reference_scan
    from tests.core.h5sig import h5_content_signature

    out = write_reference_scan(str(tmp_path / "ref.nexus"),
                               str(tmp_path / "proj"))
    now = h5_content_signature(out)
    ref = json.loads(FIXTURE.read_text())

    def _text(value):
        return value.decode() if isinstance(value, bytes) else str(value)

    # Preserve every frozen root provenance key, but name the actual shared
    # writer instead of impersonating nexusformat.  Assert that one truthful
    # identity delta plus the additive schema/program stamps positively, then
    # compare every remaining normalized legacy fact as one immutable signature.
    with h5py.File(out, "r") as h5:
        assert set(h5.attrs) == {
            "HDF5_Version", "creator", "creator_version", "file_name",
            "file_time", "h5py_version",
        }
        assert _text(h5.attrs["creator"]) == "xrd_tools"
        assert _text(h5.attrs["creator_version"])
        assert Path(_text(h5.attrs["file_name"])).resolve() == Path(out).resolve()
        entry = h5["entry"]
        assert _text(entry.attrs["default"]) == "integrated_1d"
        assert _text(entry.attrs["ssrl_schema"]) == "xrd_tools.processed_scan"
        assert int(entry.attrs["ssrl_schema_version"]) == 3
        assert tuple(entry["integrated_1d"].attrs["axes"]) == ("frame_index", "axis_1")
        assert tuple(entry["integrated_2d"].attrs["axes"]) == (
            "frame_index", "axis_2", "axis_1",
        )
        assert _text(h5["entry/reduction"].attrs["program"]) == "ssrl_xrd_tools"
        for frame in h5["entry/frames"].values():
            assert bool(frame.attrs["mask_baked"])
            assert bool(frame["thumbnail"].attrs["mask_baked"])

    comparable = copy.deepcopy(now)
    # Explicitly approved v3 deltas only. Keep the historical fixture unchanged
    # so every numeric value, dtype, chunk and unrelated attribute stays pinned.
    for group, names in (
        ("integrated_1d", {"axis_1": "q"}),
        ("integrated_2d", {"axis_1": "q", "axis_2": "chi"}),
    ):
        group_path = f"entry/{group}"
        comparable[group_path]["attrs"]["axes"] = ref[group_path]["attrs"]["axes"]
        for neutral, old in names.items():
            value = comparable.pop(f"{group_path}/{neutral}")
            assert "long_name" in value["attrs"]
            del value["attrs"]["long_name"]
            comparable[f"{group_path}/{old}"] = value
    assert set(comparable["/"]["attrs"]) == set(ref["/"]["attrs"])
    assert comparable["/"]["attrs"]["creator"] != ref["/"]["attrs"]["creator"]
    comparable["/"]["attrs"]["creator"] = ref["/"]["attrs"]["creator"]
    del comparable["entry"]["attrs"]["ssrl_schema"]
    del comparable["entry"]["attrs"]["ssrl_schema_version"]
    del comparable["entry/reduction"]["attrs"]["program"]
    for key, value in comparable.items():
        if key.startswith("entry/frames/frame_"):
            value["attrs"].pop("mask_baked", None)

    missing = sorted(set(ref) - set(comparable))
    added = sorted(set(comparable) - set(ref))
    assert not missing and not added, (
        f"tree changed: missing={missing[:6]} added={added[:6]}"
    )
    diffs = [k for k in sorted(ref) if ref[k] != comparable[k]]
    assert diffs == [], (
        "content changed at: " + ", ".join(diffs[:8]) + "\n"
        + "\n".join(
            f"  {k}: ref={ref[k]} now={comparable[k]}" for k in diffs[:3]
        )
    )


# ── C1/C2: readable storage-layout invariants ────────────────────────────────
# h5_content_signature now pins the storage layout too; these additive tests keep
# the critical writer promises obvious when the fixture diff is too large to scan.

def test_v2_record_storage_layout_frozen(tmp_path):
    """The integrated stacks stay chunked + resizable (the streaming append
    depends on both); the label/axis columns keep their shape; and NOTHING
    re-emits raw lzf (the ARM64 bus-error codec — never re-emitted, see the
    avoid-lzf policy).  All mode-independent (holds for lz4/gzip/none)."""
    import h5py
    from tests.core._v2_record_fixture import write_reference_scan

    out = write_reference_scan(str(tmp_path / "ref.nexus"), str(tmp_path / "proj"))
    with h5py.File(out, "r") as f:
        # appendable integrated stacks: chunked + resizable along frame axis
        for p in ("entry/integrated_1d/intensity", "entry/integrated_1d/sigma",
                  "entry/integrated_2d/intensity"):
            ds = f[p]
            assert ds.chunks is not None, f"{p}: lost chunking"
            assert ds.maxshape[0] is None, f"{p}: not resizable (append broken)"
        # frame_index: chunked + resizable for append, but never compressed
        for p in ("entry/integrated_1d/frame_index",
                  "entry/integrated_2d/frame_index"):
            ds = f[p]
            assert ds.chunks is not None and ds.maxshape[0] is None
            assert ds.compression is None
        # fixed axes: not chunked, not compressed
        for p in ("entry/integrated_1d/axis_1", "entry/integrated_2d/axis_1",
                  "entry/integrated_2d/axis_2"):
            ds = f[p]
            assert ds.chunks is None and ds.compression is None
        # the ARM64 guard: no dataset re-emits raw lzf anywhere in the tree
        lzf = []
        f.visititems(lambda n, o: lzf.append(n) if isinstance(o, h5py.Dataset)
                     and o.compression == "lzf" else None)
        assert lzf == [], f"raw lzf re-emitted (ARM64 bus-error risk): {lzf}"


def test_v2_record_integrated_stacks_compress(tmp_path):
    """The integrated stacks honor the resolved codec: ``gzip`` yields a portable
    (stock-h5py-readable) file; the default compresses (lz4, or a gzip fallback
    when hdf5plugin is absent).  Both keep chunking + resizability; neither
    re-emits raw lzf.  The fixture passes the selected codec explicitly to the
    shared writer, so this does not depend on GUI import-time state."""
    import h5py
    from xrd_tools.io.nexus import resolve_stack_compression
    from tests.core._v2_record_fixture import write_reference_scan

    stacks = ("entry/integrated_1d/intensity", "entry/integrated_1d/sigma",
              "entry/integrated_2d/intensity")
    # portable gzip
    out = write_reference_scan(
        str(tmp_path / "g.nexus"), str(tmp_path / "gp"), compression="gzip"
    )
    with h5py.File(out, "r") as f:
        for p in stacks:
            assert f[p].compression == "gzip", f"{p}: not portable gzip"
            assert f[p].chunks is not None and f[p].maxshape[0] is None
    # default: compressed (lz4, or gzip when hdf5plugin missing), never raw lzf
    out = write_reference_scan(
        str(tmp_path / "d.nexus"), str(tmp_path / "dp"),
        compression=resolve_stack_compression("lz4"),
    )
    with h5py.File(out, "r") as f:
        for p in stacks:
            assert f[p].compression not in (None, "lzf"), f"{p}: lost default compression"


def test_v2_record_gi_scan_writes_gi_provenance(tmp_path):
    """S-7: a GI scan's written .nexus carries the data-derived GI provenance under
    /entry/reduction/config (gi=True, + gi_config for the freeze/mode).  The
    pre-6a parity fixture uses a NON-GI reference scan (gi=False), so it never
    exercised this branch and its "byte-identical to pre-6a" claim silently
    ignored the added block -- this asserts the gi-bearing write explicitly (the
    block is ADDITIVE provenance; reload stays compatible; MIGRATION discloses it).
    """
    import h5py
    from tests.core._v2_record_fixture import (
        write_gi_reference_scan,
        write_reference_scan,
    )

    def _gi_flag(f):
        # the flag is persisted as a string ("true"/"false"), not an h5 bool
        v = f["entry/reduction/config/gi"][()]
        if isinstance(v, bytes):
            v = v.decode()
        return str(v).strip().lower() in ("true", "1")

    out = str(tmp_path / "gi.nexus")
    write_gi_reference_scan(out, str(tmp_path / "gip"))
    with h5py.File(out, "r") as f:
        assert "entry/reduction/config/gi" in f, "GI scan lost /reduction/config/gi"
        assert _gi_flag(f) is True
        assert set(f["entry/integrated_1d"].attrs["multi_result_modes"]) == {
            "q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"
        }
        assert set(f["entry/integrated_2d"].attrs["multi_result_modes"]) == {
            "qip_qoop", "q_chi", "exit_angles"
        }
        for mode in ("q_ip", "q_oop", "exit_angle", "chi_gi"):
            assert f[f"entry/integrated_1d/{mode}/sigma"].shape == (3, 32)
        for mode in ("q_chi", "exit_angles"):
            assert f[f"entry/integrated_2d/{mode}/intensity"].shape == (3, 16, 32)
        gi_config = json.loads(f["entry/reduction/config/gi_config"][()].decode())
        assert gi_config["gi_mode_1d"] == "q_total"
        assert gi_config["gi_mode_2d"] == "qip_qoop"

    # the non-GI reference records gi=False (present, not absent) -> a reader can
    # always tell GI from standard.
    ref = write_reference_scan(str(tmp_path / "std.nexus"), str(tmp_path / "sp"))
    with h5py.File(ref, "r") as f:
        assert "entry/reduction/config/gi" in f
        assert _gi_flag(f) is False
