from xrd_tools.io.metadata import (
    read_image_metadata,
    read_pdi_metadata,
    read_txt_metadata,
)
from xrd_tools.io.image import (
    load_mask,
    read_image,
    read_image_stack,
    read_images_parallel,
    read_nexus_frame,
    nexus_info,
    find_image_files,
    apply_rotation,
    get_detector_mask,
    resolve_detector_shape,
    count_frames,
    SUPPORTED_EXTS,
)
from xrd_tools.io.spec import (
    get_scan_path_info,
    get_energy_and_UB,
    get_spec_scan_type,
    get_from_spec_file,
    get_angles,
)
from xrd_tools.io.export import (
    read_xye,
    write_h5,
    write_xye,
)
from xrd_tools.io.schema import (
    PROCESSED_SCHEMA_VERSION,
    SCHEMA,
)
from xrd_tools.io.nexus import (
    find_nexus_image_dataset,
    list_entries,
    open_nexus_writer,
    read_nexus,
    read_scan,
    read_scan_metadata,
    read_stitched,
    upsert_per_frame_geometry,
    upsert_positioners,
    upsert_scan_metadata,
    validate_integrated_stack_write,
    write_frame_records,
    write_integrated_stack,
    write_nexus,
    write_nexus_frame,
    write_per_frame_geometry,
    write_positioners,
    write_scan_metadata,
    write_stitched,
)
from xrd_tools.io.read import (
    Integrated1D,
    Integrated2D,
    ProcessedScan,
    Scan,  # deprecated alias for ProcessedScan (S5 rename)
    get_1d,
    get_2d,
    get_frames,
    get_metadata,
    get_raw_frame,
    get_thumbnail,
    open_scan,
    read_scan_data,
    relative_source_path,
    resolve_source_master,
)
from xrd_tools.io.aggregate import (
    Aggregated1D,
    Aggregated2D,
    aggregate_1d,
    aggregate_2d,
)
from xrd_tools.io.frame_view import (
    FrameViewReader,
    iter_frame_records,
    iter_frame_views,
    read_frame_record,
    read_frame_records,
    read_frame_view,
    read_frame_views,
)
from xrd_tools.io.nexus_inspect import (
    NexusAxisSummary,
    NexusDatasetData,
    NexusDatasetPreview,
    NexusFileSummary,
    NexusNodeSummary,
    NexusReducedSummary,
    NexusXDartSummary,
    inspect_nexus,
    preview_nexus_dataset,
    read_nexus_dataset,
)
from xrd_tools.io.chunk_size import adaptive_chunk_size
from xrd_tools.io.image_source import (
    ImageSourceInfo,
    ImageSourceKind,
    RawFrameResult,
    classify_image_source,
    load_image_frame,
    load_processed_raw_or_thumbnail,
)

try:
    from xrd_tools.io.tiled import (
        connect_tiled,
        list_scans,
        read_tiled_run,
    )
except ImportError:
    pass
