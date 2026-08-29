from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from xrd_tools.core.scan import SourceKind
from xrd_tools.io.image import (
    read_detector_image_layout,
    read_image,
    read_nexus_frame,
)
from xrd_tools.io.image_source import ImageSourceKind, classify_image_source
from xrd_tools.io.nexus import (
    _NexusDatasetOwnerSlot,
    _bind_nexus_stack_from_entry,
    find_nexus_image_dataset,
    NexusImageStack,
    read_nexus,
    read_scan,
    read_scan_metadata,
    read_stitched,
)
from xrd_tools.io.output_path import resolve_output_target
from xrd_tools.io.processed_scan_id import (
    ProcessedXdartInputError,
    has_processed_output_markers_path,
    is_current_processed_xdart_path,
    require_current_processed,
    require_current_processed_groups,
    require_raw_input,
)
from xrd_tools.io.frame_view import FrameViewReader
from xrd_tools.io.read import get_1d, get_metadata, resolve_source_master
from xrd_tools.io.read import ProcessedScan, open_scan
from xrd_tools.io.schema import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    REINTEGRATE_SHADOW_COMPLETE_ATTR,
    REINTEGRATE_SHADOW_SUFFIX,
    SCHEMA_NAME_ATTR,
    SCHEMA_VERSION_ATTR,
)
from xrd_tools.sources.descriptor import describe_container
from xrd_tools.sources.probe import ProbeState
from xrd_tools.sources.nexus import ProcessedNexusSource
from xrd_tools.sources.registry import guess_source_kind, open_source
from xrd_tools.core.scan import SourceSpec
from xrd_tools.session.experiment_reload import ExperimentRecordReader, ReloadStatus


def _raw_detector(path: Path) -> Path:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.create_dataset(
            "instrument/detector/data",
            data=np.ones((2, 4, 5), dtype=np.uint16),
        )
    return path


def _processed(
    path: Path,
    *,
    stamp: bool,
    results: bool,
    detector: bool = False,
    schema_version: object = PROCESSED_SCHEMA_VERSION,
) -> Path:
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        if stamp:
            entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
            entry.attrs[SCHEMA_VERSION_ATTR] = schema_version
        if results:
            group = entry.create_group("integrated_2d")
            group.attrs["NX_class"] = "NXdata"
            group.attrs["signal"] = "intensity"
            group.attrs["axes"] = ("frame_index", "chi", "q")
            group.create_dataset(
                "intensity",
                data=np.ones((2, 3, 4), dtype=np.float32),
                chunks=(1, 3, 4),
                maxshape=(None, 3, 4),
            )
            group.create_dataset(
                "frame_index",
                data=np.arange(2, dtype=np.int64),
                chunks=(2,),
                maxshape=(None,),
            )
            group.create_dataset("chi", data=np.arange(3, dtype=np.float32))
            group.create_dataset("q", data=np.arange(4, dtype=np.float32))
        if detector:
            entry.create_dataset(
                "instrument/detector/data",
                data=np.ones((2, 4, 5), dtype=np.uint16),
            )
    return path


@pytest.mark.parametrize("suffix", [".nxs", ".nexus"])
def test_raw_detector_suffixes_remain_structure_discovered(
    tmp_path: Path, suffix: str
) -> None:
    path = _raw_detector(tmp_path / f"raw{suffix}")
    assert has_processed_output_markers_path(path) is False
    assert is_current_processed_xdart_path(path) is False
    assert guess_source_kind(path) is SourceKind.NEXUS_STACK
    assert describe_container(path).state is ProbeState.READY
    assert read_image(
        path,
        frame=0,
        dataset_path="/entry/instrument/detector/data",
        exact_frame=True,
    ).shape == (4, 5)


def test_historical_integrated_nxs_is_negative_only(tmp_path: Path) -> None:
    path = _processed(tmp_path / "historical.nxs", stamp=False, results=True)
    assert has_processed_output_markers_path(path) is True
    assert is_current_processed_xdart_path(path) is False
    assert describe_container(path).state is ProbeState.INVALID


def test_current_stamped_nexus_is_positive_processed(tmp_path: Path) -> None:
    path = _processed(tmp_path / "current.nexus", stamp=True, results=True)
    assert has_processed_output_markers_path(path) is True
    assert is_current_processed_xdart_path(path) is True
    assert describe_container(path).state is ProbeState.PROCESSED_OUTPUT


def test_interrupted_current_output_is_never_raw(tmp_path: Path) -> None:
    path = _processed(tmp_path / "interrupted.nexus", stamp=True, results=False)
    assert has_processed_output_markers_path(path) is True
    assert is_current_processed_xdart_path(path) is False
    assert describe_container(path).state is ProbeState.INVALID


@pytest.mark.parametrize(
    "stamp,results",
    [(True, False), (False, True)],
    ids=["stamped-partial", "integrated-hybrid"],
)
def test_processed_markers_precede_canonical_detector_admission(
    tmp_path: Path, stamp: bool, results: bool
) -> None:
    suffix = ".nexus" if stamp else ".nxs"
    path = _processed(
        tmp_path / f"hybrid{suffix}",
        stamp=stamp,
        results=results,
        detector=True,
    )
    assert has_processed_output_markers_path(path) is True
    assert classify_image_source(path).kind is ImageSourceKind.UNKNOWN
    assert describe_container(path).state is ProbeState.INVALID
    with pytest.raises(ProcessedXdartInputError):
        find_nexus_image_dataset(path)
    with pytest.raises(ProcessedXdartInputError):
        read_image(
            path,
            frame=0,
            dataset_path="/entry/instrument/detector/data",
            exact_frame=True,
        )
    with pytest.raises(ProcessedXdartInputError):
        read_nexus_frame(
            path,
            frame=0,
            dataset_path="/entry/instrument/detector/data",
        )
    with pytest.raises(ProcessedXdartInputError):
        read_detector_image_layout(path)


def test_processed_markers_precede_external_link_binding(tmp_path: Path) -> None:
    member = _raw_detector(tmp_path / "member.h5")
    master = tmp_path / "hybrid_master.h5"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_group("integrated_1d")
        data = entry.create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            member.name,
            "/entry/instrument/detector/data",
        )
    with h5py.File(master, "r") as handle:
        slot = _NexusDatasetOwnerSlot()
        with pytest.raises(ProcessedXdartInputError):
            _bind_nexus_stack_from_entry(
                handle["entry"],
                declared_entry_name="entry",
                prefer_apstools_flat=False,
                owner_slot=slot,
            )
        assert slot.owner is None
    assert classify_image_source(master).kind is ImageSourceKind.UNKNOWN


def test_external_link_target_processed_markers_precede_every_raw_binder(
    tmp_path: Path,
) -> None:
    target = _processed(
        tmp_path / "processed-target.nexus", stamp=True, results=True,
    )
    master = tmp_path / "raw_master.h5"
    selector = "/entry/data/data_000001"
    with h5py.File(master, "w") as handle:
        entry = handle.create_group("entry")
        data = entry.create_group("data")
        data["data_000001"] = h5py.ExternalLink(
            target.name,
            "/entry/integrated_2d/intensity",
        )

    with pytest.raises(ProcessedXdartInputError):
        find_nexus_image_dataset(master)
    with pytest.raises(ProcessedXdartInputError):
        read_image(master, frame=0, dataset_path=selector, exact_frame=True)
    with pytest.raises(ProcessedXdartInputError):
        read_nexus_frame(master, frame=0, dataset_path=selector)
    with h5py.File(master, "r") as handle:
        with pytest.raises(ProcessedXdartInputError):
            NexusImageStack(handle, [selector])
    with h5py.File(master, "r") as handle:
        slot = _NexusDatasetOwnerSlot()
        with pytest.raises(ProcessedXdartInputError):
            _bind_nexus_stack_from_entry(
                handle["entry"],
                declared_entry_name="entry",
                prefer_apstools_flat=False,
                owner_slot=slot,
            )
        assert slot.owner is not None
        assert slot.owner.state == "CLOSED"
    assert classify_image_source(master).kind is ImageSourceKind.UNKNOWN
    assert describe_container(master).state is ProbeState.INVALID


def test_nested_group_in_processed_owner_cannot_bind_external_raw_stack(
    tmp_path: Path,
) -> None:
    member = _raw_detector(tmp_path / "raw-member.h5")
    processed = _processed(
        tmp_path / "processed-with-proxy.nexus", stamp=True, results=True,
    )
    with h5py.File(processed, "r+") as handle:
        data = handle["entry"].create_group("proxy/data")
        data["data_000001"] = h5py.ExternalLink(
            member.name,
            "/entry/instrument/detector/data",
        )

    with h5py.File(processed, "r") as handle:
        slot = _NexusDatasetOwnerSlot()
        with pytest.raises(ProcessedXdartInputError):
            _bind_nexus_stack_from_entry(
                handle["entry/proxy"],
                declared_entry_name="proxy",
                prefer_apstools_flat=False,
                owner_slot=slot,
            )
        assert slot.owner is None


def test_public_raw_nexus_readers_reject_current_processed_input(
    tmp_path: Path,
) -> None:
    path = _processed(tmp_path / "current.nexus", stamp=True, results=True)
    with pytest.raises(ProcessedXdartInputError):
        read_nexus(path)
    with h5py.File(path, "r") as handle:
        with pytest.raises(ProcessedXdartInputError):
            NexusImageStack(handle, ["/entry/integrated_2d/intensity"])


def test_processed_markers_precede_apstools_nxwriter_fast_path(
    tmp_path: Path,
) -> None:
    for name, stamped in (("partial.nexus", True), ("hybrid.nxs", False)):
        path = tmp_path / name
        with h5py.File(path, "w") as handle:
            handle.attrs["creator"] = "NXWriter"
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            entry.create_group("instrument/bluesky")
            data = entry.create_group("data")
            data.attrs["NX_class"] = "NXdata"
            detector = data.create_dataset(
                "detector", data=np.ones((2, 4, 5), dtype=np.uint16)
            )
            detector.attrs["signal_type"] = "detector"
            if stamped:
                entry.attrs[SCHEMA_NAME_ATTR] = PROCESSED_SCHEMA_NAME
                entry.attrs[SCHEMA_VERSION_ATTR] = PROCESSED_SCHEMA_VERSION
            else:
                entry.create_group("integrated_1d")
        assert describe_container(path).state is ProbeState.INVALID
        assert classify_image_source(path).kind is ImageSourceKind.UNKNOWN


def test_processed_openers_require_current_structure_not_suffix_or_kind(
    tmp_path: Path,
) -> None:
    malformed = _processed(
        tmp_path / "malformed.nexus", stamp=True, results=False
    )
    with h5py.File(malformed, "r+") as handle:
        handle["entry"].create_group("integrated_1d")
    assert not is_current_processed_xdart_path(malformed)
    spec = SourceSpec(malformed, SourceKind.PROCESSED_NEXUS)
    for opener in (
        lambda: ProcessedScan(malformed),
        lambda: open_scan(malformed),
        lambda: ProcessedNexusSource(malformed),
        lambda: open_source(spec),
    ):
        with pytest.raises(ValueError, match="current xdart"):
            opener()


def test_current_admission_rejects_a_malformed_present_result_sibling(
    tmp_path: Path,
) -> None:
    path = _processed(
        tmp_path / "mixed-results.nexus", stamp=True, results=True,
    )
    with h5py.File(path, "r+") as handle:
        handle["entry"].create_group("integrated_1d")
    assert has_processed_output_markers_path(path) is True
    assert is_current_processed_xdart_path(path) is False
    with pytest.raises(ValueError, match="current xdart"):
        ProcessedScan(path)

    missing_axis = _processed(
        tmp_path / "missing-2d-axis.nexus", stamp=True, results=True,
    )
    with h5py.File(missing_axis, "r+") as handle:
        group = handle["entry/integrated_2d"]
        del group["chi"]
        del group["intensity"]
        group.create_dataset(
            "intensity", data=np.ones((2, 4), dtype=np.float32),
        )
    assert is_current_processed_xdart_path(missing_axis) is False


def test_current_admission_recovers_only_complete_reintegration_shadow(
    tmp_path: Path,
) -> None:
    complete = _processed(
        tmp_path / "complete-shadow.nexus", stamp=True, results=True,
    )
    with h5py.File(complete, "r+") as handle:
        entry = handle["entry"]
        shadow_name = f"integrated_2d{REINTEGRATE_SHADOW_SUFFIX}"
        entry.move("integrated_2d", shadow_name)
        entry[shadow_name].attrs[REINTEGRATE_SHADOW_COMPLETE_ATTR] = True
    assert is_current_processed_xdart_path(complete) is True
    require_current_processed(complete)

    incomplete = _processed(
        tmp_path / "incomplete-shadow.nexus", stamp=True, results=True,
    )
    with h5py.File(incomplete, "r+") as handle:
        entry = handle["entry"]
        entry.move(
            "integrated_2d",
            f"integrated_2d{REINTEGRATE_SHADOW_SUFFIX}",
        )
    assert is_current_processed_xdart_path(incomplete) is False
    with pytest.raises(ValueError, match="current xdart"):
        require_current_processed(incomplete)


def test_public_scan_readers_consume_the_admitted_complete_shadow(
    tmp_path: Path,
) -> None:
    path = _processed(
        tmp_path / "complete-shadow-readers.nexus",
        stamp=True,
        results=True,
    )
    with h5py.File(path, "r+") as handle:
        entry = handle["entry"]
        shadow_name = f"integrated_2d{REINTEGRATE_SHADOW_SUFFIX}"
        entry.move("integrated_2d", shadow_name)
        entry[shadow_name].attrs[REINTEGRATE_SHADOW_COMPLETE_ATTR] = True

    scan = read_scan(path)
    metadata = read_scan_metadata(path)
    public_metadata = get_metadata(path)

    assert scan["intensity_2d"].shape == (2, 3, 4)
    assert np.array_equal(scan["frame"].values, np.arange(2))
    assert np.array_equal(metadata["q_2d"].values, np.arange(4))
    assert np.array_equal(metadata["chi"].values, np.arange(3))
    assert public_metadata["has_1d"] is False
    assert public_metadata["has_2d"] is True


def test_current_admission_uses_the_caller_container_and_local_entry(
    tmp_path: Path,
) -> None:
    target = _processed(
        tmp_path / "target.nexus",
        stamp=True,
        results=True,
    )
    with h5py.File(target, "r") as handle:
        assert require_current_processed_groups(handle).entry.name == "/entry"
        with pytest.raises(ValueError, match="current xdart"):
            require_current_processed_groups(
                handle,
                container=tmp_path / "alias.h5",
            )

    external_entry = tmp_path / "external-entry.nexus"
    with h5py.File(external_entry, "w") as handle:
        handle["entry"] = h5py.ExternalLink(target.name, "/entry")
    with h5py.File(external_entry, "r") as handle:
        assert isinstance(handle.get("entry", getlink=True), h5py.ExternalLink)
        assert isinstance(handle["entry"], h5py.Group)
    assert is_current_processed_xdart_path(external_entry) is False
    for reader in (read_scan, read_scan_metadata):
        with pytest.raises(ValueError, match="current xdart"):
            reader(external_entry)

    external_ancestor = tmp_path / "external-ancestor.nexus"
    with h5py.File(external_ancestor, "w") as handle:
        handle["foreign"] = h5py.ExternalLink(target.name, "/")
    with h5py.File(external_ancestor, "r") as handle:
        with pytest.raises(ValueError, match="current xdart"):
            require_current_processed_groups(handle, "foreign/entry")


def test_processed_owner_markers_reject_an_unmarked_hardlink_alias_binder(
    tmp_path: Path,
) -> None:
    processed = _processed(
        tmp_path / "processed-alias.nexus",
        stamp=True,
        results=True,
    )
    with h5py.File(processed, "r+") as handle:
        alias = handle.create_group("alias")
        alias["pixels"] = handle["entry/integrated_2d/intensity"]
    with h5py.File(processed, "r") as handle:
        with pytest.raises(ProcessedXdartInputError):
            require_raw_input(handle["alias/pixels"])
        slot = _NexusDatasetOwnerSlot()
        with pytest.raises(ProcessedXdartInputError):
            _bind_nexus_stack_from_entry(
                handle["alias"],
                declared_entry_name="alias",
                prefer_apstools_flat=False,
                owner_slot=slot,
            )
        assert slot.owner is None or slot.owner.state == "CLOSED"

    raw = tmp_path / "raw-alias.h5"
    with h5py.File(raw, "w") as handle:
        alias = handle.create_group("alias")
        alias.create_dataset(
            "pixels",
            data=np.ones((2, 3, 4), dtype=np.uint16),
        )
    with h5py.File(raw, "r") as handle:
        require_raw_input(handle["alias/pixels"])
        slot = _NexusDatasetOwnerSlot()
        binding = _bind_nexus_stack_from_entry(
            handle["alias"],
            declared_entry_name="alias",
            prefer_apstools_flat=False,
            owner_slot=slot,
        )
        assert binding.paths == ["/alias/pixels"]
        binding.close()


@pytest.mark.parametrize(
    "marker",
    (
        "true",
        np.asarray([True], dtype=np.bool_),
        np.int8(1),
        np.int8(2),
        np.float64(1.0),
    ),
    ids=("text", "array", "numeric-one", "numeric-other", "float-one"),
)
def test_complete_shadow_marker_requires_exact_scalar_boolean(
    tmp_path: Path,
    marker: object,
) -> None:
    path = _processed(
        tmp_path / "malformed-shadow-marker.nexus",
        stamp=True,
        results=True,
    )
    with h5py.File(path, "r+") as handle:
        entry = handle["entry"]
        shadow_name = f"integrated_2d{REINTEGRATE_SHADOW_SUFFIX}"
        entry.move("integrated_2d", shadow_name)
        entry[shadow_name].attrs[REINTEGRATE_SHADOW_COMPLETE_ATTR] = marker
    assert is_current_processed_xdart_path(path) is False
    with pytest.raises(ValueError, match="current xdart"):
        require_current_processed(path)


def _replace_result_dataset_with_indirect_storage(
    path: Path, *, dataset_name: str, storage: str,
) -> None:
    dataset_path = f"entry/integrated_2d/{dataset_name}"
    with h5py.File(path, "r+") as handle:
        group = handle["entry/integrated_2d"]
        if dataset_name == "sigma" and dataset_name not in group:
            group.create_dataset(
                dataset_name,
                data=np.ones(group["intensity"].shape, dtype=np.float32),
            )
        original = np.asarray(handle[dataset_path][()])

    if storage == "virtual":
        source = path.with_name(f"{path.stem}-{dataset_name}-source.h5")
        with h5py.File(source, "w") as handle:
            handle.create_dataset("payload", data=original)
        layout = h5py.VirtualLayout(shape=original.shape, dtype=original.dtype)
        layout[...] = h5py.VirtualSource(
            str(source), "/payload", shape=original.shape,
        )
        with h5py.File(path, "r+") as handle:
            group = handle["entry/integrated_2d"]
            del group[dataset_name]
            group.create_virtual_dataset(dataset_name, layout)
        return

    if storage != "external":
        raise AssertionError(storage)
    backing = path.with_name(f"{path.stem}-{dataset_name}.raw")
    with h5py.File(path, "r+") as handle:
        group = handle["entry/integrated_2d"]
        del group[dataset_name]
        dataset = group.create_dataset(
            dataset_name,
            shape=original.shape,
            dtype=original.dtype,
            external=[(str(backing), 0, int(original.nbytes))],
        )
        dataset[...] = original


@pytest.mark.parametrize("storage", ("virtual", "external"))
@pytest.mark.parametrize(
    "dataset_name", ("frame_index", "q", "chi", "intensity", "sigma"),
)
def test_current_admission_rejects_indirect_result_storage(
    tmp_path: Path, storage: str, dataset_name: str,
) -> None:
    path = _processed(
        tmp_path / f"{storage}-{dataset_name}.nexus",
        stamp=True,
        results=True,
    )
    _replace_result_dataset_with_indirect_storage(
        path,
        dataset_name=dataset_name,
        storage=storage,
    )

    assert is_current_processed_xdart_path(path) is False
    with pytest.raises(ValueError, match="current xdart"):
        require_current_processed(path)


def test_direct_processed_readers_reject_unstamped_historical_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        group = entry.create_group("integrated_1d")
        group.attrs["NX_class"] = "NXdata"
        group.attrs["signal"] = "intensity"
        group.attrs["axes"] = ("frame_index", "q")
        group.create_dataset("frame_index", data=np.arange(2, dtype=np.int64))
        group.create_dataset("q", data=np.arange(4, dtype=np.float32))
        group.create_dataset(
            "intensity", data=np.ones((2, 4), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="current xdart"):
        get_1d(path)
    with pytest.raises(ValueError, match="current xdart"):
        with FrameViewReader(path):
            pass
    for reader in (read_scan, read_scan_metadata, read_stitched):
        with pytest.raises(ValueError, match="current xdart"):
            reader(path)
    experiment = ExperimentRecordReader().read(path)
    assert experiment.status is ReloadStatus.ABSENT
    assert "current xdart" in experiment.reason


@pytest.mark.parametrize(
    "malformed",
    ["2", b"2", 2.0, 2.9, True, [2]],
    ids=["text", "bytes", "float-equal", "float-truncated", "bool", "array"],
)
def test_current_schema_version_requires_exact_integral_scalar(
    tmp_path: Path, malformed: object
) -> None:
    path = _processed(
        tmp_path / "malformed-version.nexus",
        stamp=True,
        results=True,
        schema_version=malformed,
    )
    assert is_current_processed_xdart_path(path) is False
    assert has_processed_output_markers_path(path) is True
    assert describe_container(path).state is ProbeState.INVALID


def test_processed_target_never_reuses_legacy_nxs(tmp_path: Path) -> None:
    legacy = tmp_path / "scan_007.nxs"
    legacy.write_bytes(b"historical")
    target = resolve_output_target(
        tmp_path,
        "scan_007",
        mode="Append",
    )
    assert target == tmp_path / "scan_007.nexus"
    assert legacy.read_bytes() == b"historical"


def test_moved_source_resolves_only_against_selected_project_root(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old-project"
    new_root = tmp_path / "new-project"
    relative = Path("raw/run/scan_master.h5")
    moved = new_root / relative
    moved.parent.mkdir(parents=True)
    moved.write_bytes(b"raw")
    processed = tmp_path / "processed" / "scan.nexus"
    processed.parent.mkdir()

    assert resolve_source_master(
        relative.as_posix(),
        scan_file=processed,
        source_base=old_root.as_posix(),
    ) is None
    assert resolve_source_master(
        relative.as_posix(),
        scan_file=processed,
        source_base=old_root.as_posix(),
        source_root=new_root,
    ) == moved
    old_copy = old_root / relative
    old_copy.parent.mkdir(parents=True)
    old_copy.write_bytes(b"stale")
    assert resolve_source_master(
        relative.as_posix(),
        scan_file=processed,
        source_base=old_root.as_posix(),
        source_root=new_root,
    ) == moved
    assert resolve_source_master(
        "../outside.h5",
        scan_file=processed,
        source_base=old_root.as_posix(),
        source_root=new_root,
    ) is None
    assert resolve_source_master(
        relative.as_posix(),
        scan_file=processed,
        source_base=old_root.as_posix(),
        source_root=new_root / "nested" / "..",
    ) is None


def test_relative_source_symlink_escape_is_rejected_by_all_consumers(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    raw = outside / "raw.h5"
    raw.write_bytes(b"raw")
    try:
        (project / "jump").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlink unavailable: {error}")
    locator = "jump/raw.h5"
    root = str(project.resolve())
    assert resolve_source_master(
        locator,
        scan_file=project / "processed.nexus",
        source_root=root,
    ) is None

    from xdart.gui.tabs.scattering.browse_values import (
        canonical_browse_source_identity,
    )
    from xrd_tools.reduction.reintegrate import _resolve_source_locator

    with pytest.raises(ValueError, match="REPLACEMENT_SOURCE_RELOCATION"):
        _resolve_source_locator(locator, root)
    with pytest.raises(ValueError, match="escapes Project root"):
        canonical_browse_source_identity(
            SimpleNamespace(
                label=1,
                source_path=locator,
                source_frame_index=0,
            ),
            str(project / "processed.nexus"),
            source_root=root,
        )
