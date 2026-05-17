"""
Reciprocal Space Mapping (RSM) utilities for x-ray diffraction data.

Processes SPEC scans and detector images into 3D HKL volumes, with I/O,
gridding, combination, line cuts, slices. Image I/O and SPEC parsing
live in ssrl_xrd_tools.io; fitting in ssrl_xrd_tools.analysis.fitting.

Geometry primitives (``DiffractometerConfig``, ``DetectorHeader``,
``PixelQMap``) are re-exported here for convenience; their canonical
home is :mod:`ssrl_xrd_tools.core.geometry`.
"""

from ssrl_xrd_tools.core.geometry import (
    DetectorHeader,
    DiffractometerConfig,
    PixelQMap,
)
from ssrl_xrd_tools.rsm.volume import (
    RSMVolume,
    extract_2d_slice,
    extract_line_cut,
    mask_data,
    save_vtk,
)
from ssrl_xrd_tools.rsm.gridding import (
    StreamingGridder,
    StreamingScan,
    combine_grids,
    get_common_grid,
    grid_img_data,
    grid_img_data_streaming,
    grid_scans_streaming,
)
from ssrl_xrd_tools.rsm.pipeline import (
    ExperimentConfig,
    ScanInfo,
    SphereInput,
    grid_spheres_streaming,
    load_images,
    process_scan,
    process_scan_data,
    process_scan_from_sphere,
)

# Backward-compatible alias
extract_slice = extract_2d_slice

__all__ = [
    # Geometry primitives (re-exports from core.geometry)
    "DetectorHeader",
    "DiffractometerConfig",
    "PixelQMap",
    # RSM volume + utils
    "RSMVolume",
    "extract_2d_slice",
    "extract_slice",
    "extract_line_cut",
    "mask_data",
    "save_vtk",
    # Gridding
    "StreamingGridder",
    "StreamingScan",
    "combine_grids",
    "get_common_grid",
    "grid_img_data",
    "grid_img_data_streaming",
    "grid_scans_streaming",
    # Pipeline
    "ExperimentConfig",
    "ScanInfo",
    "SphereInput",
    "grid_spheres_streaming",
    "load_images",
    "process_scan",
    "process_scan_data",
    "process_scan_from_sphere",
]
