# -*- coding: utf-8 -*-
"""R1 — name-only candidate enumeration (:func:`enumerate_candidates`).

Proves the enumeration contract the DirectoryIndex depends on: no HDF5 opens,
correct adapter ownership/exclusion per format, shared Filter grammar, and
natural ordering independent of filesystem enumeration order.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from xrd_tools.sources.discover import Candidate, enumerate_candidates


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.write_bytes(data)
    return path


def _nxs_with_detector(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        inst = entry.create_group("instrument")
        det = inst.create_group("detector")
        det.create_dataset("data", data=np.zeros((2, 3, 4)))
    return path


# ---- format ownership / exclusion ---------------------------------------


def test_nxs_h5_hdf5_are_nexus_family_candidates(tmp_path):
    for name in ("a.nxs", "b.h5", "c.hdf5"):
        _touch(tmp_path / name)

    candidates = enumerate_candidates(tmp_path)

    assert {c.path.name for c in candidates} == {"a.nxs", "b.h5", "c.hdf5"}
    assert {c.adapter_id for c in candidates} == {"nexus_hdf5"}


def test_dot_nexus_output_extension_is_excluded_from_raw_candidates(tmp_path):
    """.nexus is a reserved future xdart OUTPUT extension (O/H23) — openable
    explicitly (see test_source_format_adapters / registry.guess_source_kind)
    but never a raw directory candidate."""
    _touch(tmp_path / "raw.nxs")
    _touch(tmp_path / "processed.nexus")

    candidates = enumerate_candidates(tmp_path)

    assert [c.path.name for c in candidates] == ["raw.nxs"]


def test_r1r4_cxi_is_a_nexus_family_candidate_but_nexus_is_not(tmp_path):
    """Gate 7: .cxi is a NeXus-family raw candidate (parity with
    guess_source_kind/discover_scans/_NEXUS_EXTS — the held defect returned 0
    .cxi candidates); .nexus stays excluded from raw discovery."""
    _touch(tmp_path / "scan.cxi")
    _touch(tmp_path / "raw.nxs")
    _touch(tmp_path / "out.nexus")

    candidates = enumerate_candidates(tmp_path)

    assert {c.path.name for c in candidates} == {"scan.cxi", "raw.nxs"}
    assert {c.adapter_id for c in candidates} == {"nexus_hdf5"}
    # the .cxi candidate is owned by the nexus adapter, same as .nxs/.h5/.hdf5
    cxi = next(c for c in candidates if c.path.name == "scan.cxi")
    assert cxi.adapter_id == "nexus_hdf5"


def test_eiger_data_sidecar_excluded_but_master_included(tmp_path):
    _touch(tmp_path / "run_data_000001.h5")
    _touch(tmp_path / "run_master.h5")

    candidates = enumerate_candidates(tmp_path)

    assert [c.path.name for c in candidates] == ["run_master.h5"]


def test_exact_eiger_sidecar_shape_preserves_legitimate_data_names(tmp_path):
    excluded = {
        "data_00001.h5", "data_000001.h5",
        "run_data_00001.h5", "run_data_000001.H5",
    }
    included = {
        "aa_data_00001.nxs", "bb_data_000001.cxi",
        "cc_data_00001.hdf5", "dd_data_000001.hdf5",
        "ee_data_0001.h5", "ff_data_0000001.h5",
        "gg_data_00001_tail.h5", "metadata_00001.h5",
        "zz_data_\u0660\u0660\u0660\u0660\u0661.h5",
    }
    for name in excluded | included:
        _touch(tmp_path / name)

    candidates = enumerate_candidates(tmp_path)
    assert {c.path.name: c.adapter_id for c in candidates} == {
        name: "nexus_hdf5" for name in included}


def test_tiff_files_are_image_file_candidates(tmp_path):
    _touch(tmp_path / "frame_1.tif")
    _touch(tmp_path / "frame_2.tiff")

    candidates = enumerate_candidates(tmp_path)

    assert {c.path.name for c in candidates} == {"frame_1.tif", "frame_2.tiff"}
    assert {c.adapter_id for c in candidates} == {"image_file"}


def test_unrelated_files_are_not_candidates(tmp_path):
    _touch(tmp_path / "readme.txt")
    _touch(tmp_path / "notes.md")
    _touch(tmp_path / "archive.zip")

    assert enumerate_candidates(tmp_path) == []


def test_no_adapter_double_claims_a_file(tmp_path):
    """Every built-in-claimed file is claimed by exactly one adapter (the
    partition-by-extension invariant the built-in adapters rely on)."""
    for name in ("a.nxs", "b.tif", "c.h5", "run_master.h5"):
        _touch(tmp_path / name)

    candidates = enumerate_candidates(tmp_path)
    paths_seen = [c.path for c in candidates]
    assert len(paths_seen) == len(set(paths_seen))


# ---- shared Filter grammar ------------------------------------------------


def test_name_filter_grammar_is_applied(tmp_path):
    _touch(tmp_path / "sample_bg.nxs")
    _touch(tmp_path / "sample_scan.nxs")

    candidates = enumerate_candidates(tmp_path, name_filter="sample -bg")

    assert [c.path.name for c in candidates] == ["sample_scan.nxs"]


# ---- recursive vs non-recursive ------------------------------------------


def test_recursive_flag_controls_subdirectory_descent(tmp_path):
    _touch(tmp_path / "top.nxs")
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "nested.nxs")

    shallow = enumerate_candidates(tmp_path, recursive=False)
    deep = enumerate_candidates(tmp_path, recursive=True)

    assert [c.path.name for c in shallow] == ["top.nxs"]
    assert {c.path.name for c in deep} == {"top.nxs", "nested.nxs"}


# ---- natural ordering ------------------------------------------------------


def test_natural_order_independent_of_filesystem_enumeration_order(tmp_path):
    # Create in an order that would sort differently lexicographically
    # (scan_10 before scan_2) than naturally.
    for name in ("scan_9.nxs", "scan_10.nxs", "scan_1.nxs", "scan_2.nxs"):
        _touch(tmp_path / name)

    candidates = enumerate_candidates(tmp_path)

    assert [c.path.name for c in candidates] == [
        "scan_1.nxs", "scan_2.nxs", "scan_9.nxs", "scan_10.nxs",
    ]


def test_candidate_identity_is_stable_across_repeated_enumeration(tmp_path):
    _touch(tmp_path / "a.nxs")
    _touch(tmp_path / "b.nxs")

    first = enumerate_candidates(tmp_path)
    second = enumerate_candidates(tmp_path)

    assert [(c.path, c.adapter_id, c.version_stamp) for c in first] == [
        (c.path, c.adapter_id, c.version_stamp) for c in second
    ]


# ---- version stamp ----------------------------------------------------------


def test_version_stamp_matches_size_and_mtime(tmp_path):
    p = _touch(tmp_path / "a.nxs", data=b"hello world")
    st = p.stat()

    (candidate,) = enumerate_candidates(tmp_path)

    assert candidate.version_stamp == (st.st_size, st.st_mtime_ns)
    assert isinstance(candidate, Candidate)


# ---- zero HDF5 opens --------------------------------------------------------


def test_enumeration_never_opens_hdf5_files(tmp_path, monkeypatch):
    """Name-only enumeration must not call h5py.File — even for files that,
    if opened, would resolve real detector datasets."""
    _nxs_with_detector(tmp_path / "a.nxs")
    _touch(tmp_path / "b.h5")
    _touch(tmp_path / "run_master.h5")
    _touch(tmp_path / "frame.tif")

    calls = []
    real_file = h5py.File

    def _counting_file(*args, **kwargs):
        calls.append(args)
        return real_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", _counting_file)

    first = enumerate_candidates(tmp_path)
    assert len(first) == 4
    assert calls == []

    # An "unchanged" repeat poll must also open zero HDF5 files.
    second = enumerate_candidates(tmp_path)
    assert len(second) == 4
    assert calls == []
