"""Tests for ssrl_xrd_tools.core.provenance.

Covers the NXprocess writer/reader pair used by the v2 NeXus schema
(``/entry/reduction/``).  Tests are written against a fresh h5py file
fixture per case — no xdart dependency, no captured-file dependency.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from ssrl_xrd_tools.core.provenance import (
    CANONICAL_PACKAGES,
    capture_versions,
    read_provenance,
    write_provenance,
)


# ---------------------------------------------------------------------------
# capture_versions
# ---------------------------------------------------------------------------

class TestCaptureVersions:
    def test_returns_canonical_packages_plus_python(self):
        v = capture_versions()
        for pkg in CANONICAL_PACKAGES:
            assert pkg in v
        assert "python" in v

    def test_python_version_format(self):
        v = capture_versions()
        # "3.X.Y" — three numeric segments
        parts = v["python"].split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_missing_package_records_empty_string(self):
        # 'definitely_not_a_real_package_xyz' won't be installed
        v = capture_versions(extra_packages=["definitely_not_a_real_package_xyz"])
        assert v["definitely_not_a_real_package_xyz"] == ""

    def test_known_installed_package_has_version(self):
        # numpy and h5py are obviously installed since they're imported above
        v = capture_versions()
        assert v["numpy"]  # non-empty string
        assert v["h5py"]


# ---------------------------------------------------------------------------
# write/read round-trip
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_h5(tmp_path):
    p = tmp_path / "out.nxs"
    return p


class TestWriteRead:
    def test_minimal_write_then_read(self, fresh_h5):
        with h5py.File(fresh_h5, "w") as f:
            write_provenance(
                f,
                program="xdart",
                program_version="0.37.0-dev0",
                date="2026-05-11T00:00:00Z",
            )
        out = read_provenance(fresh_h5)
        assert out["program"] == "xdart"
        assert out["version"] == "0.37.0-dev0"
        assert out["date"] == "2026-05-11T00:00:00Z"
        # Versions captured from importlib.metadata
        assert "versions" in out
        assert "python" in out["versions"]

    def test_nxprocess_attr_set(self, fresh_h5):
        with h5py.File(fresh_h5, "w") as f:
            write_provenance(f, program_version="0.37.0-dev0")
            assert (
                f["entry/reduction"].attrs["NX_class"] == "NXprocess"
                or f["entry/reduction"].attrs["NX_class"] == b"NXprocess"
            )
            assert f["entry"].attrs.get("NX_class") in ("NXentry", b"NXentry")

    def test_config_block_is_jsonized_per_field(self, fresh_h5):
        cfg = {
            "bai_1d_args": {"npt": 2000, "unit": "q_A^-1", "method": "csr"},
            "gi_config": {"flavor": "psic", "incidence_motor": "eta"},
        }
        with h5py.File(fresh_h5, "w") as f:
            write_provenance(f, program_version="0.37.0-dev0", config=cfg)
            cfg_grp = f["entry/reduction/config"]
            raw = cfg_grp["bai_1d_args"][()]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            # Round-trippable JSON
            parsed = json.loads(raw)
            assert parsed["npt"] == 2000
        # Reader auto-parses JSON
        out = read_provenance(fresh_h5)
        assert out["config"]["bai_1d_args"]["npt"] == 2000
        assert out["config"]["gi_config"]["flavor"] == "psic"

    def test_geometry_config_is_structured_subgroup(self, fresh_h5):
        # The 'geometry' key inside config is special — it's a subgroup
        # with structured fields rather than a single JSON blob (see
        # plan §3.3: convention / mapping_json / motor_sources).
        geom = {
            "convention": "psic",
            "mapping_json": '{"rot1":{"source_motor":"nu"}}',
            "motor_sources": {"eta": "eta", "del": "del"},
        }
        with h5py.File(fresh_h5, "w") as f:
            write_provenance(
                f,
                program_version="0.37.0-dev0",
                config={"geometry": geom},
            )
            # Convention is stored as a plain string (no JSON wrapping)
            v = f["entry/reduction/config/geometry/convention"][()]
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            assert v == "psic"
        out = read_provenance(fresh_h5)
        g_out = out["config"]["geometry"]
        assert g_out["convention"] == "psic"
        # mapping_json was a string passed in — should round-trip parsed
        assert isinstance(g_out["mapping_json"], dict) or isinstance(
            g_out["mapping_json"], str
        )
        # motor_sources was a dict — JSON-encoded then decoded
        assert g_out["motor_sources"]["eta"] == "eta"

    def test_inputs_with_string_list(self, fresh_h5):
        inputs = {
            "raw_files": ["frame_0001.tif", "frame_0002.tif", "frame_0003.tif"],
            "meta_file": "scan.spec",
        }
        with h5py.File(fresh_h5, "w") as f:
            write_provenance(f, program_version="0.37.0-dev0", inputs=inputs)
        out = read_provenance(fresh_h5)
        assert out["inputs"]["meta_file"] == "scan.spec"
        rf = out["inputs"]["raw_files"]
        assert rf == ["frame_0001.tif", "frame_0002.tif", "frame_0003.tif"]

    def test_host_default_is_current_machine(self, fresh_h5):
        import socket as _socket

        with h5py.File(fresh_h5, "w") as f:
            write_provenance(f, program_version="0.37.0-dev0")
        out = read_provenance(fresh_h5)
        assert out["host"] == _socket.gethostname()

    def test_host_can_be_suppressed(self, fresh_h5):
        with h5py.File(fresh_h5, "w") as f:
            write_provenance(f, program_version="0.37.0-dev0", host="")
        out = read_provenance(fresh_h5)
        assert "host" not in out

    def test_extra_field_jsonized(self, fresh_h5):
        with h5py.File(fresh_h5, "w") as f:
            write_provenance(
                f,
                program_version="0.37.0-dev0",
                extra={"cli_args": ["xdart", "--mode", "batch"]},
            )
        out = read_provenance(fresh_h5)
        assert out["cli_args"] == ["xdart", "--mode", "batch"]

    def test_rewrite_replaces_scalars_not_appended(self, fresh_h5):
        with h5py.File(fresh_h5, "w") as f:
            write_provenance(f, program_version="0.37.0-dev0", host="alpha")
        with h5py.File(fresh_h5, "a") as f:
            write_provenance(f, program_version="0.37.0-dev0", host="beta")
        out = read_provenance(fresh_h5)
        assert out["host"] == "beta"

    def test_missing_entry_returns_empty(self, tmp_path):
        # An h5 file with no /entry/reduction/ at all
        p = tmp_path / "blank.h5"
        with h5py.File(p, "w") as f:
            f.create_group("entry")  # NXentry exists but no reduction
        out = read_provenance(p)
        assert out == {}


# ---------------------------------------------------------------------------
# Reader robustness against not-quite-spec files
# ---------------------------------------------------------------------------

class TestReaderRobustness:
    def test_plain_non_json_scalar_returned_as_is(self, fresh_h5):
        with h5py.File(fresh_h5, "w") as f:
            f.create_group("entry").create_group("reduction")
            f["entry/reduction/program"] = "xdart"
        out = read_provenance(fresh_h5)
        assert out["program"] == "xdart"

    def test_numeric_dataset_returned_as_python_scalar(self, fresh_h5):
        with h5py.File(fresh_h5, "w") as f:
            f.create_group("entry").create_group("reduction")
            f["entry/reduction/numeric_field"] = np.int64(42)
        out = read_provenance(fresh_h5)
        assert out["numeric_field"] == 42
