"""E4-S bounded one-frame preview read contract."""

from __future__ import annotations

import ast
from dataclasses import fields, replace
from importlib import import_module
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest
import tifffile

from xrd_tools.core import FrameView, IntegrationResult1D, IntegrationResult2D
from xrd_tools.io.nexus import write_integrated_stack
from xrd_tools.io.nexus_record import (
    ensure_frames_container,
    stamp_source_base,
    write_frame_record,
)


def _api():
    return import_module("xrd_tools.io.frame_preview")


def _hydration():
    return import_module("xrd_tools.session.hydration")


def _write_processed(
    root: Path,
    *,
    frame: int = 17,
    thumbnail: bool,
    raw: np.ndarray | None = None,
    detector_shape: np.ndarray | None = None,
) -> tuple[Path, Path]:
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4) if raw is None else raw
    raw_path = root / "raw" / "image.tif"
    raw_path.parent.mkdir(parents=True)
    tifffile.imwrite(raw_path, raw)
    processed = root / "xdart_processed_data" / "scan.nexus"
    processed.parent.mkdir()
    one_d = IntegrationResult1D(
        radial=np.array([0.1, 0.2, 0.3]),
        intensity=np.array([1.0, 2.0, 3.0]),
        unit="q_A^-1",
    )
    two_d = IntegrationResult2D(
        radial=np.array([0.1, 0.2, 0.3]),
        azimuthal=np.array([-1.0, 1.0]),
        intensity=np.arange(6, dtype=float).reshape(3, 2),
        unit="q_A^-1",
        azimuthal_unit="chi_deg",
    )
    with h5py.File(processed, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["ssrl_schema"] = "xrd_tools.processed_scan"
        entry.attrs["ssrl_schema_version"] = 2
        if detector_shape is not None:
            entry.create_dataset(
                "instrument/detector/detector_shape", data=detector_shape
            )
        write_integrated_stack(
            entry,
            frame_indices=[frame],
            results_1d=[one_d],
            results_2d=[two_d],
        )
        base = stamp_source_base(entry, root)
        write_frame_record(
            ensure_frames_container(entry),
            f"frame_{frame:04d}",
            thumbnail=(raw[::2, ::2] if thumbnail else None),
            source_path=raw_path,
            source_frame_index=0,
            source_base=base,
        )
    return processed, raw_path


def _read_key(path: Path, purpose, *, source_root: Path | None = None):
    hydration = _hydration()
    return hydration.HydrationReadKey(
        hydration.HydrationScope("context-a", "scan-7", "/raw/source", 3),
        str(path),
        17,
        purpose,
        None if source_root is None else str(source_root),
    )


def _instrument_reads(monkeypatch, processed):
    api = _api()
    frame_view = import_module("xrd_tools.io.frame_view")
    original_open = frame_view.h5py.File
    original_detector_read = api.read_image
    counts = {"processed": 0, "detector": 0, "detector_paths": []}

    def counted_open(path, *args, **kwargs):
        if Path(path) == Path(processed):
            counts["processed"] += 1
        return original_open(path, *args, **kwargs)

    def counted_detector_read(path, *args, **kwargs):
        counts["detector"] += 1
        counts["detector_paths"].append(Path(path))
        return original_detector_read(path, *args, **kwargs)

    monkeypatch.setattr(frame_view.h5py, "File", counted_open)
    monkeypatch.setattr(api, "read_image", counted_detector_read)
    return counts


def test_thumbnail_preview_opens_processed_once_and_never_reads_detector(
    tmp_path, monkeypatch
):
    processed, raw_path = _write_processed(tmp_path, thumbnail=True)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    purpose = _hydration().HydrationPurpose
    result = api.read_frame_preview(
        _read_key(processed, purpose.PREVIEW),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(),
    )
    assert counts == {"processed": 1, "detector": 0, "detector_paths": []}
    assert result.read_key.frame_identity == 17
    assert result.view.has_1d and result.view.has_2d
    assert result.thumbnail is not None
    assert result.raw is None
    assert result.detector_fallback_used is False
    assert result.raw_locator == "raw/image.tif"
    assert result.source_base == str(tmp_path)
    assert (result.view.source_path, result.view.source_frame_index) == (
        str(raw_path.resolve()), 0)
    for array in (
        result.thumbnail,
        result.view.intensity_1d,
        result.view.intensity_2d,
        result.view.axis_1d.values,
        result.view.axis_2d_x.values,
        result.view.axis_2d_y.values,
    ):
        assert array.flags.writeable is False


def test_raw_absent_thumbnail_preview_keeps_portable_projection(
    tmp_path, monkeypatch
):
    processed, raw_path = _write_processed(tmp_path, thumbnail=True)
    raw_path.unlink()
    counts = _instrument_reads(monkeypatch, processed)
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )

    assert counts == {"processed": 1, "detector": 0, "detector_paths": []}
    assert result.thumbnail is not None and result.raw is None
    assert result.detector_diagnostic is None
    assert (result.raw_locator, result.source_base) == (
        "raw/image.tif", str(tmp_path)
    )
    assert (result.view.source_path, result.view.source_frame_index) == (
        "raw/image.tif", 0
    )
    assert result.view.intensity_1d.flags.writeable is False
    assert result.view.intensity_2d.flags.writeable is False


def test_flattened_basename_preview_uses_exact_resolver_winner(
    tmp_path, monkeypatch
):
    processed, raw_path = _write_processed(tmp_path, thumbnail=True)
    first = processed.parent / raw_path.name
    raw_path.replace(first)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )

    assert counts == {"processed": 1, "detector": 0, "detector_paths": []}
    assert result.thumbnail is not None and result.raw is None
    assert result.raw_locator == "raw/image.tif"
    assert result.source_base == str(tmp_path)
    assert result.view.source_path == str(first.resolve())
    assert api._source_projection_matches(
        result.raw_locator,
        result.view.source_path,
        source_base=result.source_base,
        artifact=processed,
    ) == (True, False)

    later = processed.parent.parent / first.name
    shutil.copy2(first, later)
    assert api._source_projection_matches(
        result.raw_locator,
        str(later.resolve()),
        source_base=result.source_base,
        artifact=processed,
    ) == (False, False)

    weak_raw = np.ones((2, 2))
    weak_raw.setflags(write=False)
    with pytest.raises(ValueError, match="raw provenance"):
        replace(
            result,
            read_key=_read_key(processed, _hydration().HydrationPurpose.FULL),
            raw=weak_raw,
            detector_diagnostic=None,
        )


def test_symlinked_project_preview_uses_canonical_resolver_truth(
    tmp_path, monkeypatch
):
    real = tmp_path / "real"
    alias = tmp_path / "alias"
    real.mkdir()
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    processed, raw_path = _write_processed(alias, thumbnail=True)
    counts = _instrument_reads(monkeypatch, processed)
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )

    assert counts == {"processed": 1, "detector": 0, "detector_paths": []}
    assert result.thumbnail is not None and result.raw is None
    assert result.raw_locator == "raw/image.tif"
    assert result.source_base == str(alias)
    assert result.view.source_path == str(raw_path.resolve())


def test_no_thumbnail_preview_never_reads_detector_and_reports_missing_thumbnail(
    tmp_path, monkeypatch
):
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4)
    raw[3, 3] = np.iinfo(np.uint16).max
    processed, _ = _write_processed(tmp_path, thumbnail=False, raw=raw)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(),
    )
    assert counts == {"processed": 1, "detector": 0, "detector_paths": []}
    assert result.view.has_1d and result.view.has_2d
    assert result.thumbnail is result.raw is None
    assert result.detector_fallback_used is False
    assert result.detector_diagnostic == "stored thumbnail unavailable"


def test_no_thumbnail_without_exact_mask_truth_fails_closed(
    tmp_path, monkeypatch
):
    processed, _ = _write_processed(tmp_path, thumbnail=False)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )
    assert counts["processed"] == 1
    assert counts["detector"] == 0
    assert result.view.has_1d and result.view.has_2d
    assert result.thumbnail is None
    assert result.raw is None
    assert result.detector_fallback_used is False


def test_saturation_projection_uses_the_accepted_detector_ceiling(
    tmp_path, monkeypatch
):
    raw = np.arange(16, dtype=np.uint8).reshape(4, 4)
    raw[2:, 2:] = np.iinfo(np.uint8).max
    processed, _ = _write_processed(tmp_path, thumbnail=False, raw=raw)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    projection = api.DetectorPreviewProjection.without_static_mask(
        mask_saturation=True,
        saturation_ceiling=np.iinfo(np.uint8).max,
    )
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=projection,
    )
    assert counts["detector"] == 1
    assert np.isnan(result.raw[2:, 2:]).all()


def test_detector_value_toggle_uses_native_values_and_accepted_science(
    tmp_path, monkeypatch
):
    raw = np.array([[-1, 2], [3, 4294967295]], dtype=np.int64)
    processed, _ = _write_processed(tmp_path, thumbnail=False, raw=raw)
    api = _api()
    observed_dtypes = []
    original_mask = api._masked_detector

    def observed_mask(source, projection):
        observed_dtypes.append(np.asarray(source).dtype)
        return original_mask(source, projection)

    monkeypatch.setattr(api, "_masked_detector", observed_mask)
    disabled = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(
            mask_saturation=False
        ),
    )
    enabled = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(
            mask_saturation=True,
            saturation_ceiling=float(np.iinfo(np.int64).max),
        ),
    )
    assert observed_dtypes == [np.dtype(np.int64), np.dtype(np.int64)]
    assert disabled.raw[0, 0] == -1
    assert disabled.raw[1, 1] == 4294967295
    assert np.isnan(enabled.raw[0, 0])
    assert np.isnan(enabled.raw[1, 1])


def test_upper_threshold_is_independent_of_detector_value_toggle(
    tmp_path, monkeypatch
):
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4)
    processed, _ = _write_processed(tmp_path, thumbnail=False, raw=raw)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(
            apply_threshold=True,
            threshold_max=10,
            mask_saturation=False,
        ),
    )
    assert counts["detector"] == 1
    assert result.raw[2, 2] == 10
    assert np.isnan(result.raw[2, 3:]).all()
    assert np.isnan(result.raw[3]).all()


def test_full_is_distinct_after_thumbnail_preview(tmp_path, monkeypatch):
    processed, _ = _write_processed(tmp_path, thumbnail=True)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    purpose = _hydration().HydrationPurpose
    preview = api.read_frame_preview(_read_key(processed, purpose.PREVIEW))
    projection = api.DetectorPreviewProjection.without_static_mask(
        mask_saturation=False
    )
    full = api.read_frame_preview(
        _read_key(processed, purpose.FULL),
        detector_projection=projection,
    )
    assert preview.raw is None
    assert full.raw is not None
    assert full.detector_fallback_used is False
    assert counts["processed"] == 2
    assert counts["detector"] == 1


def test_preview_refuses_numeric_string_and_absent_processed_frame(tmp_path):
    processed, _ = _write_processed(tmp_path, thumbnail=True)
    hydration = _hydration()
    scope = hydration.HydrationScope("context-a", "scan-7", "/raw/source", 3)
    with pytest.raises((TypeError, ValueError)):
        _api().read_frame_preview(
            hydration.HydrationReadKey(
                scope,
                str(processed),
                "0017",
                hydration.HydrationPurpose.PREVIEW,
            )
        )
    with pytest.raises((KeyError, ValueError)):
        _api().read_frame_preview(
            hydration.HydrationReadKey(
                scope,
                str(processed),
                99,
                hydration.HydrationPurpose.PREVIEW,
            )
        )
    with pytest.raises((TypeError, ValueError)):
        _api().read_frame_preview(
            hydration.HydrationReadKey(
                scope,
                str(processed),
                -1,
                hydration.HydrationPurpose.PREVIEW,
            )
        )
    with h5py.File(processed, "r+") as handle:
        handle["entry/frames"].create_group("frame_0099")
    with pytest.raises((KeyError, ValueError)):
        _api().read_frame_preview(
            hydration.HydrationReadKey(
                scope,
                str(processed),
                99,
                hydration.HydrationPurpose.PREVIEW,
            )
        )


@pytest.mark.parametrize("source_index", [None, -1])
def test_missing_or_negative_detector_frame_fails_before_detector_io(
    tmp_path, monkeypatch, source_index
):
    processed, _ = _write_processed(tmp_path, thumbnail=False)
    with h5py.File(processed, "r+") as handle:
        source = handle["entry/frames/frame_0017/source"]
        if source_index is None:
            del source["frame_index"]
        else:
            source["frame_index"][...] = source_index
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(),
    )
    assert counts["detector"] == 0
    assert result.raw is None
    assert result.source_frame_index is None
    assert result.detector_fallback_used is False
    assert result.detector_diagnostic


def test_one_frame_preview_reads_only_target_maps_and_metadata_row(
    tmp_path, monkeypatch
):
    processed, _ = _write_processed(tmp_path, thumbnail=True)
    with h5py.File(processed, "r+") as handle:
        scan_data = handle["entry"].create_group("scan_data")
        scan_data["frame_index"] = np.arange(4096, dtype=np.int64)
        scan_data["motor"] = np.arange(4096, dtype=np.float64)
    frame_view = import_module("xrd_tools.io.frame_view")
    original_values = frame_view._dataset_values
    original_map = frame_view._frame_map
    full_column_reads = []
    map_targets = []

    def observed_values(dataset):
        if dataset.shape == (4096,):
            full_column_reads.append(dataset.name)
        return original_values(dataset)

    def observed_map(group, *args, **kwargs):
        target = args[0] if args else kwargs.get("target_frame")
        map_targets.append(target)
        return original_map(group, *args, **kwargs)

    monkeypatch.setattr(frame_view, "_dataset_values", observed_values)
    monkeypatch.setattr(frame_view, "_frame_map", observed_map)
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )
    assert result.view.metadata_raw["motor"] == 17
    assert full_column_reads == []
    assert map_targets and set(map_targets) == {17}


def test_preview_honors_exact_persisted_hdf_dataset_path(tmp_path):
    processed, _ = _write_processed(tmp_path, thumbnail=False)
    raw_path = tmp_path / "raw" / "two-stacks.h5"
    with h5py.File(raw_path, "w") as handle:
        handle.create_dataset("/entry/data/data", data=np.full((1, 2, 2), 11))
        handle.create_dataset(
            "/entry/instrument/detector/right",
            data=np.full((1, 2, 2), 99),
        )
    with h5py.File(processed, "r+") as handle:
        frame = handle["entry/frames/frame_0017"]
        del frame["source"]
        source = frame.create_group("source")
        source["path"] = str(raw_path)
        source["frame_index"] = 0
        source.attrs["dataset_path"] = "/entry/instrument/detector/right"
    api = _api()
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(),
    )
    assert result.raw_dataset_path == "/entry/instrument/detector/right"
    assert np.all(result.raw == 99)


def test_eiger_anchor_reads_global_index_across_segments_once(
    tmp_path, monkeypatch
):
    processed, _ = _write_processed(tmp_path, thumbnail=False)
    master = tmp_path / "raw" / "multi_master.h5"
    with h5py.File(master, "w") as handle:
        data = handle.create_group("/entry/data")
        data.create_dataset(
            "data_000001", data=np.full((2, 2, 2), 11, np.uint16)
        )
        data.create_dataset(
            "data_000002", data=np.full((2, 2, 2), 99, np.uint16)
        )
    with h5py.File(processed, "r+") as handle:
        source = handle["entry/frames/frame_0017/source"]
        source["path"][...] = str(master)
        source["frame_index"][...] = 2
        source.attrs["dataset_path"] = "/entry/data/data_000001"

    image = import_module("xrd_tools.io.image")
    original_open = image.h5py.File
    source_opens = 0

    def counted_open(path, *args, **kwargs):
        nonlocal source_opens
        if Path(path) == master:
            source_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(image.h5py, "File", counted_open)
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=_api().DetectorPreviewProjection.without_static_mask(),
    )
    assert source_opens == 1
    assert result.detector_diagnostic is None
    assert result.raw_dataset_path == "/entry/data/data_000001"
    assert np.all(result.raw == 99)


@pytest.mark.parametrize(
    ("values", "source_index", "expected"),
    [((11,), 0, 11), ((11, 99), 2, 99)],
    ids=["frame-zero", "later-segment"],
)
def test_eiger_external_link_anchor_keeps_master_identity_and_opens_once(
    tmp_path, monkeypatch, values, source_index, expected
):
    processed, _ = _write_processed(tmp_path, thumbnail=False)
    raw_dir = tmp_path / "raw"
    master = raw_dir / "external_master.h5"
    for number, value in enumerate(values, 1):
        segment = raw_dir / f"external_data_{number:06d}.h5"
        with h5py.File(segment, "w") as handle:
            handle["/entry/data/data"] = np.full(
                (2, 2, 2), value, dtype=np.uint16
            )
    with h5py.File(master, "w") as handle:
        group = handle.create_group("/entry/data")
        for number in range(1, len(values) + 1):
            group[f"data_{number:06d}"] = h5py.ExternalLink(
                f"external_data_{number:06d}.h5", "/entry/data/data"
            )
    with h5py.File(master, "r") as handle:
        assert isinstance(
            handle["/entry/data"].get("data_000001", getlink=True),
            h5py.ExternalLink,
        )
    with h5py.File(processed, "r+") as handle:
        source = handle["entry/frames/frame_0017/source"]
        source["path"][...] = str(master)
        source["frame_index"][...] = source_index
        source.attrs["dataset_path"] = "/entry/data/data_000001"

    image = import_module("xrd_tools.io.image")
    original_open = image.h5py.File
    source_opens = 0

    def counted_open(path, *args, **kwargs):
        nonlocal source_opens
        if Path(path) == master:
            source_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(image.h5py, "File", counted_open)
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=_api().DetectorPreviewProjection.without_static_mask(),
    )
    assert source_opens == 1
    assert result.detector_diagnostic is None
    assert result.raw_dataset_path == "/entry/data/data_000001"
    assert np.all(result.raw == expected)


def test_unmarked_nonmonotonic_legacy_map_uses_bounded_scan(tmp_path):
    processed = tmp_path / "legacy.nxs"
    with h5py.File(processed, "w") as handle:
        scan_data = handle.create_group("entry/scan_data")
        scan_data["frame_index"] = np.array([17, 3, 9], dtype=np.int64)
        scan_data["motor"] = np.array([123.0, 3.0, 9.0])
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )
    assert result.view.metadata_raw["motor"] == 123.0


@pytest.mark.parametrize("source_kind", ["hdf", "fabio"])
def test_exact_single_frame_source_rejects_nonzero_index(tmp_path, source_kind):
    processed, raw_path = _write_processed(tmp_path, thumbnail=False)
    dataset_path = None
    if source_kind == "hdf":
        raw_path = tmp_path / "raw" / "single.h5"
        dataset_path = "/entry/data/data"
        with h5py.File(raw_path, "w") as handle:
            handle.create_dataset(
                dataset_path, data=np.full((2, 2), 77, np.uint16)
            )
    with h5py.File(processed, "r+") as handle:
        source = handle["entry/frames/frame_0017/source"]
        source["path"][...] = str(raw_path)
        source["frame_index"][...] = 7
        if dataset_path is not None:
            source.attrs["dataset_path"] = dataset_path
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=_api().DetectorPreviewProjection.without_static_mask(),
    )
    assert result.raw is None
    assert result.detector_fallback_used is False
    assert result.detector_diagnostic


def test_hdf_fallback_without_dataset_identity_fails_before_detector_io(
    tmp_path, monkeypatch
):
    processed, _ = _write_processed(tmp_path, thumbnail=False)
    raw_path = tmp_path / "raw" / "two-stacks.h5"
    with h5py.File(raw_path, "w") as handle:
        handle.create_dataset("/entry/data/data", data=np.full((1, 2, 2), 11))
        handle.create_dataset("/entry/right", data=np.full((1, 2, 2), 99))
    with h5py.File(processed, "r+") as handle:
        source = handle["entry/frames/frame_0017/source"]
        source["path"][...] = str(raw_path)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(),
    )
    assert counts["detector"] == 0
    assert result.raw is None
    assert result.detector_diagnostic == "raw dataset identity unavailable"


def test_preview_preserves_explicit_unmasked_thumbnail_truth(tmp_path):
    processed, _ = _write_processed(tmp_path, thumbnail=True)
    with h5py.File(processed, "r+") as handle:
        handle["entry/frames/frame_0017/thumbnail"].attrs["mask_baked"] = False
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )
    assert result.thumbnail is not None
    assert result.view.mask_baked is False


def test_legacy_thumbnail_without_mask_attribute_keeps_true_default(tmp_path):
    processed, _ = _write_processed(tmp_path, thumbnail=True)
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )
    assert result.view.mask_baked is True


def test_relative_raw_locator_uses_moved_project_root_not_cwd(
    tmp_path, monkeypatch
):
    original = tmp_path / "original"
    processed, _ = _write_processed(original, thumbnail=False)
    moved = tmp_path / "moved"
    shutil.move(str(original), moved)
    moved_processed = moved / "xdart_processed_data" / processed.name
    decoy = tmp_path / "cwd"
    decoy.mkdir()
    decoy_raw = decoy / "raw" / "image.tif"
    decoy_raw.parent.mkdir()
    tifffile.imwrite(decoy_raw, np.full((4, 4), 999, dtype=np.uint16))
    monkeypatch.chdir(decoy)
    counts = _instrument_reads(monkeypatch, moved_processed)
    api = _api()
    result = api.read_frame_preview(
        _read_key(
            moved_processed,
            _hydration().HydrationPurpose.FULL,
            source_root=moved,
        ),
        detector_projection=api.DetectorPreviewProjection.without_static_mask(),
    )
    assert result.raw_locator == "raw/image.tif"
    assert result.source_base == str(original)
    assert counts["detector_paths"] == [moved / "raw" / "image.tif"]
    assert all(not str(path).startswith(str(original)) for path in counts["detector_paths"])


def test_preview_values_are_reference_free_and_reject_ndarray_construction():
    api = _api()
    names = {field.name for field in fields(api.DetectorPreviewProjection)}
    assert names.isdisjoint({"mask", "array", "handle", "store", "callback"})
    with pytest.raises(TypeError):
        api.DetectorPreviewProjection(
            mask_available=True,
            mask_bytes=np.zeros(4, dtype=bool),
            mask_dtype="bool",
            mask_shape=(2, 2),
        )
    with pytest.raises(ValueError, match="accepted detector ceiling"):
        api.DetectorPreviewProjection.without_static_mask(mask_saturation=True)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("threshold_min", True),
        ("threshold_max", "10"),
        ("saturation_ceiling", False),
        ("threshold_min", type("IntSubclass", (int,), {})(3)),
        ("threshold_max", type("FloatSubclass", (float,), {})(3)),
        ("saturation_ceiling", type("Coercible", (), {"__float__": lambda _: 3.0})()),
    ),
)
def test_scientific_scalars_reject_coercive_types(field, value):
    with pytest.raises(TypeError):
        _api().DetectorPreviewProjection.without_static_mask(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("threshold_min", np.nan),
        ("threshold_max", np.inf),
        ("saturation_ceiling", -np.inf),
        ("saturation_ceiling", 0),
        ("saturation_ceiling", -1),
    ),
)
def test_scientific_scalars_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        _api().DetectorPreviewProjection.without_static_mask(**{field: value})


def _preview_result(
    purpose,
    *,
    thumbnail=False,
    raw=False,
    locator=None,
    dataset_path=None,
    source_index=None,
    fallback=False,
    diagnostic=None,
):
    thumb = np.ones((1, 1)) if thumbnail else None
    detector = np.ones((2, 2)) if raw else None
    for array in (thumb, detector):
        if array is not None:
            array.setflags(write=False)
    key = _read_key(Path("/processed.nxs"), purpose)
    view = FrameView(
        label=17,
        thumbnail=thumb,
        source_path=locator,
        source_frame_index=source_index,
    )
    return _api().FramePreview(
        key,
        view,
        view.thumbnail,
        detector,
        locator,
        dataset_path,
        source_index,
        None,
        fallback,
        diagnostic,
    )


def test_frame_preview_accepts_complete_purpose_truth_table(tmp_path):
    purpose = _hydration().HydrationPurpose
    assert _preview_result(purpose.ONE_D).raw is None
    assert _preview_result(purpose.PREVIEW, thumbnail=True).raw is None
    assert _preview_result(
        purpose.PREVIEW, diagnostic="stored thumbnail unavailable"
    ).detector_diagnostic
    raw_path = tmp_path / "image.tif"
    tifffile.imwrite(raw_path, np.ones((2, 2), dtype=np.uint16))
    assert _preview_result(
        purpose.FULL,
        raw=True,
        locator=str(raw_path.resolve()),
        source_index=0,
    ).raw is not None
    assert _preview_result(
        purpose.FULL, diagnostic="detector unavailable"
    ).detector_diagnostic


@pytest.mark.parametrize(
    ("purpose_name", "values"),
    (
        ("ONE_D", {"raw": True, "locator": "/raw/image.tif", "source_index": 0}),
        ("ONE_D", {"fallback": True}),
        ("ONE_D", {"diagnostic": "unexpected"}),
        (
            "PREVIEW",
            {"raw": True, "locator": "/raw/image.tif", "source_index": 0},
        ),
        (
            "PREVIEW",
            {
                "thumbnail": True,
                "raw": True,
                "locator": "/raw/image.tif",
                "source_index": 0,
                "fallback": True,
            },
        ),
        ("PREVIEW", {}),
        ("PREVIEW", {"thumbnail": True, "diagnostic": "unexpected"}),
        (
            "FULL",
            {
                "raw": True,
                "locator": "/raw/image.tif",
                "source_index": 0,
                "fallback": True,
            },
        ),
        ("FULL", {}),
        (
            "FULL",
            {
                "raw": True,
                "locator": "/raw/image.tif",
                "source_index": 0,
                "diagnostic": "failed",
            },
        ),
        ("FULL", {"raw": True}),
        (
            "FULL",
            {
                "raw": True,
                "locator": "/raw/source.h5",
                "source_index": 0,
            },
        ),
    ),
)
def test_frame_preview_rejects_contradictory_purpose_facts(
    purpose_name, values
):
    purpose = getattr(_hydration().HydrationPurpose, purpose_name)
    with pytest.raises(ValueError):
        _preview_result(purpose, **values)


def test_wrong_mask_shape_never_exposes_unmasked_raw(tmp_path, monkeypatch):
    processed, _ = _write_processed(tmp_path, thumbnail=False)
    counts = _instrument_reads(monkeypatch, processed)
    api = _api()
    projection = api.DetectorPreviewProjection.from_mask(
        np.zeros((2, 2), dtype=bool)
    )
    result = api.read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.FULL),
        detector_projection=projection,
    )
    assert counts["detector"] == 1
    assert result.raw is None
    assert result.detector_fallback_used is False


@pytest.mark.parametrize(
    ("shape", "expected"),
    [(np.array([4, 6], dtype=np.int64), (4, 6)),
     (np.array([[4, 6]], dtype=np.int64), None)],
    ids=("exact-positive-pair", "malformed-rank"),
)
def test_preview_carries_only_exact_persisted_detector_shape(
    tmp_path, shape, expected
):
    processed, _ = _write_processed(
        tmp_path, thumbnail=True, detector_shape=shape
    )
    result = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )
    assert result.view.extra.get("detector_shape") == expected


def test_frame_preview_result_rejects_malformed_fields_and_label_aliases(
    tmp_path,
):
    processed, _ = _write_processed(tmp_path, thumbnail=True)
    valid = _api().read_frame_preview(
        _read_key(processed, _hydration().HydrationPurpose.PREVIEW)
    )
    malformed = (
        {"read_key": object()},
        {"view": object()},
        {"raw_locator": 3},
        {"raw_locator": ""},
        {"raw_locator": "/different/raw.tif"},
        {"view": replace(valid.view, source_path=str(tmp_path / "wrong" / "raw" / "image.tif"))},
        {"source_frame_index": 1},
        {"raw_dataset_path": 3},
        {"source_frame_index": -1},
        {"source_frame_index": True},
        {"source_base": 3},
        {"detector_fallback_used": 1},
        {"detector_fallback_used": True},
        {"detector_diagnostic": object()},
        {"view": replace(valid.view, label=18)},
    )
    for changes in malformed:
        with pytest.raises((TypeError, ValueError)):
            replace(valid, **changes)


def test_projection_guard_requires_exact_first_winner_and_native_spelling(
    tmp_path, monkeypatch
):
    later_root = tmp_path / "later"
    processed, later_raw = _write_processed(later_root, thumbnail=True)
    first_root = tmp_path / "first"
    first_raw = first_root / "raw" / later_raw.name
    first_raw.parent.mkdir(parents=True)
    shutil.copy2(later_raw, first_raw)
    with h5py.File(processed, "r+") as handle:
        handle["entry"].attrs["source_base"] = str(first_root)

    api = _api()
    purpose = _hydration().HydrationPurpose
    valid = api.read_frame_preview(_read_key(processed, purpose.PREVIEW))
    assert valid.view.source_path == str(first_raw.resolve())
    with pytest.raises(ValueError, match="raw provenance"):
        replace(
            valid,
            view=replace(valid.view, source_path=str(later_raw.resolve())),
        )

    monkeypatch.setattr(
        api,
        "resolve_source_master",
        lambda *args, **kwargs: Path("C:/project/raw/image.tif"),
    )
    native_spelling = replace(
        valid,
        view=replace(valid.view, source_path="C:/project/raw/image.tif"),
    )
    assert native_spelling.view.source_path == "C:/project/raw/image.tif"

    def resolver_failure(*args, **kwargs):
        raise OSError("resolver unavailable")

    monkeypatch.setattr(api, "resolve_source_master", resolver_failure)
    with pytest.raises(ValueError, match="raw provenance"):
        replace(valid, view=replace(valid.view, source_path=valid.raw_locator))
    monkeypatch.undo()

    with pytest.raises(ValueError, match="raw provenance"):
        replace(valid, raw_locator="raw/../raw/image.tif")
    with pytest.raises(ValueError, match="raw provenance"):
        replace(valid, source_frame_index=1)

    weak_raw = np.ones((2, 2))
    weak_raw.setflags(write=False)
    weak_locator = "absent/unique-image.tif"
    with pytest.raises(ValueError, match="raw provenance"):
        replace(
            valid,
            read_key=_read_key(processed, purpose.FULL),
            view=replace(valid.view, source_path=weak_locator),
            raw=weak_raw,
            raw_locator=weak_locator,
            detector_diagnostic=None,
        )


def test_preview_value_fields_and_scientific_import_boundary_are_exact():
    api = _api()
    assert tuple(field.name for field in fields(api.DetectorPreviewProjection)) == (
        "mask_available",
        "mask_bytes",
        "mask_dtype",
        "mask_shape",
        "apply_threshold",
        "threshold_min",
        "threshold_max",
        "mask_saturation",
        "saturation_ceiling",
    )
    assert api.DetectorPreviewProjection.__annotations__ == {
        "mask_available": "bool",
        "mask_bytes": "bytes",
        "mask_dtype": "str",
        "mask_shape": "tuple[int, ...]",
        "apply_threshold": "bool",
        "threshold_min": "float | None",
        "threshold_max": "float | None",
        "mask_saturation": "bool",
        "saturation_ceiling": "float | None",
    }
    assert tuple(field.name for field in fields(api.FramePreview)) == (
        "read_key",
        "view",
        "thumbnail",
        "raw",
        "raw_locator",
        "raw_dataset_path",
        "source_frame_index",
        "source_base",
        "detector_fallback_used",
        "detector_diagnostic",
    )
    assert api.FramePreview.__annotations__ == {
        "read_key": "HydrationReadKey",
        "view": "FrameView",
        "thumbnail": "np.ndarray | None",
        "raw": "np.ndarray | None",
        "raw_locator": "str | None",
        "raw_dataset_path": "str | None",
        "source_frame_index": "int | None",
        "source_base": "str | None",
        "detector_fallback_used": "bool",
        "detector_diagnostic": "str | None",
    }
    source = Path(api.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden_roots = (
        "pyFAI",
        "xdart",
        "xrd_tools.analysis",
        "xrd_tools.session.run_configuration",
    )
    assert not {
        module
        for module in imports
        if any(
            module == root or module.startswith(f"{root}.")
            for root in forbidden_roots
        )
    }
    lowered = source.lower()
    for forbidden in ("poni", "run_intent", "gui_field", "mutable_provider"):
        assert forbidden not in lowered
