"""Headless safety seams for the v1.1.2 NXS directory review (Tranche A).

Two production-path failures are pinned here at the ``xrd_tools`` layer (the GUI
wiring that drives them is in ``tests/xdart/test_nxs_directory_safety.py``):

* **F-NXS-1** — the pure source/output path guard
  (:func:`xrd_tools.io.output_safety.check_output_not_source`): an output that
  would overwrite, or be re-ingested as, one of its own inputs is refused
  *before* any writer opens.
* **F-NXS-2** — a processed xdart scan file must never resolve through the RAW
  detector finder; its ``integrated_2d`` cake must not be handed back as a
  detector stack.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from xrd_tools.io.output_safety import (
    OutputCollisionError,
    check_output_not_source,
    path_within_dir,
    paths_same_file,
)
from xrd_tools.io.processed_scan_id import (
    ProcessedXdartInputError,
    is_processed_xdart_file,
    is_processed_xdart_path,
)
from xrd_tools.io.nexus import (
    find_nexus_image_dataset,
    find_nexus_image_dataset_in_open_file,
    open_nexus_image_stack,
)
from xrd_tools.io.image import _find_hdf5_image_dataset


# ---------------------------------------------------------------------------
# fixtures — a real processed xdart scan file and a real raw detector container
# ---------------------------------------------------------------------------

N_FRAMES = 5
N_Q = 40
N_CHI = 24
H, W = 12, 16


def _write_processed_xdart(path, *, stamp_schema=True):
    """Hand-craft a processed xdart scan file: the reduced ``integrated_1d`` /
    ``integrated_2d`` stacks and (optionally) the ``ssrl_schema`` stamp.

    Mirrors the on-disk layout of a real processed ``.nxs`` — critically an
    ``integrated_2d/intensity`` of shape ``(N, chi, q)``, the 3-D cake the
    unfixed largest-3-D finder mis-selected as a detector stack (F-NXS-2).  A
    real field file omits the schema stamp (nexusformat round-trip), so
    ``stamp_schema=False`` covers the legacy content-only case.
    """
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        if stamp_schema:
            from xrd_tools.io.schema import (
                PROCESSED_SCHEMA_NAME,
                PROCESSED_SCHEMA_VERSION,
                SCHEMA_NAME_ATTR,
                SCHEMA_VERSION_ATTR,
            )
            e.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
            e.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
        g1 = e.create_group("integrated_1d")
        g1.attrs["NX_class"] = "NXdata"
        g1.create_dataset("intensity",
                          data=np.ones((N_FRAMES, N_Q), dtype=np.float32))
        g1.create_dataset("frame_index",
                          data=np.arange(1, N_FRAMES + 1, dtype=np.int64))
        g1.create_dataset("q", data=np.linspace(0.5, 5.0, N_Q, dtype=np.float32))
        g2 = e.create_group("integrated_2d")
        g2.attrs["NX_class"] = "NXdata"
        # The (N, chi, q) cake — 3-D, and the largest 3-D dataset in the file.
        g2.create_dataset(
            "intensity",
            data=np.ones((N_FRAMES, N_CHI, N_Q), dtype=np.float32),
        )
        g2.create_dataset("frame_index",
                          data=np.arange(1, N_FRAMES + 1, dtype=np.int64))
        g2.create_dataset("q", data=np.linspace(0.5, 5.0, N_Q, dtype=np.float32))
        g2.create_dataset("chi",
                          data=np.linspace(-90, 90, N_CHI, dtype=np.float32))
    return path


def _write_raw_detector_nxs(path):
    """A raw detector container: a real 3-D detector stack under the canonical
    ``/entry/instrument/detector/data`` path.  No integrated groups."""
    with h5py.File(path, "w") as f:
        e = f.create_group("entry")
        e.attrs["NX_class"] = "NXentry"
        instr = e.create_group("instrument")
        det = instr.create_group("detector")
        det.create_dataset(
            "data",
            data=np.arange(N_FRAMES * H * W, dtype=np.uint32).reshape(N_FRAMES, H, W),
        )
    return path


# ---------------------------------------------------------------------------
# F-NXS-2: the processed classifier and the raw finder's rejection
# ---------------------------------------------------------------------------

class TestProcessedClassifier:
    def test_classifier_true_for_processed(self, tmp_path):
        p = _write_processed_xdart(tmp_path / "proc.nxs")
        assert is_processed_xdart_path(p) is True
        with h5py.File(p, "r") as f:
            assert is_processed_xdart_file(f) is True

    def test_classifier_true_for_legacy_no_schema_stamp(self, tmp_path):
        # Real field files (nexusformat round-trip) carry the integrated groups
        # but NO ssrl_schema stamp — the content signal must still classify them.
        p = _write_processed_xdart(tmp_path / "legacy.nxs", stamp_schema=False)
        assert is_processed_xdart_path(p) is True

    def test_classifier_false_for_raw(self, tmp_path):
        p = _write_raw_detector_nxs(tmp_path / "raw.nxs")
        assert is_processed_xdart_path(p) is False

    def test_classifier_false_for_unreadable(self, tmp_path):
        p = tmp_path / "torn.nxs"
        p.write_bytes(b"\x89HDF\r\n not really hdf5")
        assert is_processed_xdart_path(p) is False


class TestFinderRejectsProcessed:
    def test_find_nexus_image_dataset_raises_on_processed(self, tmp_path):
        # THE F-NXS-2 repro: the unfixed finder returned
        # /entry/integrated_2d/intensity (a cake) as a detector dataset.
        p = _write_processed_xdart(tmp_path / "proc.nxs")
        with pytest.raises(ProcessedXdartInputError):
            find_nexus_image_dataset(p)

    def test_open_file_finder_raises_before_largest_3d(self, tmp_path):
        p = _write_processed_xdart(tmp_path / "proc.nxs")
        with h5py.File(p, "r") as f:
            with pytest.raises(ProcessedXdartInputError):
                find_nexus_image_dataset_in_open_file(f, "entry")

    def test_open_nexus_image_stack_rejects_processed(self, tmp_path):
        # Protects the OTHER finder consumer (NexusStackSource, LiveFrame raw
        # load, headless Scan raw reads) — a processed file can't be read raw.
        p = _write_processed_xdart(tmp_path / "proc.nxs")
        with pytest.raises(ProcessedXdartInputError):
            open_nexus_image_stack(p)

    def test_image_py_finder_still_rejects_and_is_valueerror(self, tmp_path):
        # The dedup: image.py's resolver shares the classifier and raises the
        # typed error, which stays a ValueError subclass for back-compat.
        p = _write_processed_xdart(tmp_path / "proc.nxs")
        with h5py.File(p, "r") as f:
            with pytest.raises(ValueError) as ei:
                _find_hdf5_image_dataset(f)
        assert isinstance(ei.value, ProcessedXdartInputError)

    def test_raw_detector_resolves_unaffected(self, tmp_path):
        # Regression guard: a genuine raw container still resolves normally.
        p = _write_raw_detector_nxs(tmp_path / "raw.nxs")
        assert find_nexus_image_dataset(p) == "/entry/instrument/detector/data"
        with open_nexus_image_stack(p) as stack:
            assert stack.shape == (N_FRAMES, H, W)


# ---------------------------------------------------------------------------
# F-NXS-1: the pure source/output path guard
# ---------------------------------------------------------------------------

class TestCheckOutputNotSource:
    def test_exact_file_collision_existing(self, tmp_path):
        raw = _write_raw_detector_nxs(tmp_path / "acq_00001.nxs")
        with pytest.raises(OutputCollisionError):
            check_output_not_source(raw, input_files=[raw])

    def test_exact_file_collision_via_dotdot_normalization(self, tmp_path):
        # Nonexistent target reached by a different spelling of the same path
        # (`..` normalization) must still collide — resolved-normcase fallback.
        d = tmp_path / "data"
        d.mkdir()
        out = tmp_path / "data" / ".." / "data" / "scan.nxs"
        src = tmp_path / "data" / "scan.nxs"
        with pytest.raises(OutputCollisionError):
            check_output_not_source(out, input_files=[src])

    def test_windows_case_fold_uses_normcase(self, monkeypatch):
        # On a case-insensitive platform (Windows) `A.NXS` and `a.nxs` are the
        # same file.  Drive the helper's normcase seam directly (POSIX normcase
        # is identity) so the case-fold path is covered on every host.
        monkeypatch.setattr(os.path, "normcase", str.lower)
        assert paths_same_file("/beamline/Data/Scan_A.NXS",
                               "/beamline/data/scan_a.nxs") is True
        with pytest.raises(OutputCollisionError):
            check_output_not_source(
                "/beamline/Data/Scan_A.NXS",
                input_files=["/beamline/data/scan_a.nxs"],
            )

    def test_container_same_directory_rejected(self, tmp_path):
        d = tmp_path / "raw"
        d.mkdir()
        with pytest.raises(OutputCollisionError):
            check_output_not_source(
                d / "scan.nxs",
                watched_dirs=[d],
                container_directory_mode=True,
            )

    def test_separate_directories_ok(self, tmp_path):
        watch = tmp_path / "raw"
        out = tmp_path / "out"
        watch.mkdir()
        out.mkdir()
        # Must NOT raise.
        check_output_not_source(
            out / "scan.nxs",
            watched_dirs=[watch],
            container_directory_mode=True,
        )

    def test_recursive_descendant_rejected(self, tmp_path):
        watch = tmp_path / "raw"
        sub = watch / "processed"
        watch.mkdir()
        sub.mkdir()
        with pytest.raises(OutputCollisionError):
            check_output_not_source(
                sub / "scan.nxs",
                watched_dirs=[watch],
                recursive=True,
                container_directory_mode=True,
            )

    def test_non_recursive_subdir_ok(self, tmp_path):
        # A non-recursive watch never descends into the subdir, so an output
        # there cannot be re-ingested.
        watch = tmp_path / "raw"
        sub = watch / "processed"
        watch.mkdir()
        sub.mkdir()
        check_output_not_source(
            sub / "scan.nxs",
            watched_dirs=[watch],
            recursive=False,
            container_directory_mode=True,
        )

    def test_non_container_same_dir_ok(self, tmp_path):
        # A per-file image series (e.g. TIFF) writes a .nxs that can never be a
        # .tif source and is not re-discovered — same-dir is allowed.
        d = tmp_path / "tiffs"
        d.mkdir()
        check_output_not_source(
            d / "scan.nxs",
            watched_dirs=[d],
            container_directory_mode=False,
        )

    def test_sibling_prefix_not_treated_as_within(self):
        # /data-bar must not count as inside /data.
        assert path_within_dir("/data-bar/x.nxs", "/data") is False
        assert path_within_dir("/data/sub/x.nxs", "/data") is True

    def test_no_inputs_no_watched_is_safe(self, tmp_path):
        check_output_not_source(tmp_path / "scan.nxs")
