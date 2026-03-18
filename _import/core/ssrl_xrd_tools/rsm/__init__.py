"""
Reciprocal Space Mapping (RSM) utilities for x-ray diffraction data.

Processes SPEC scans and detector images into 3D HKL volumes, with I/O,
gridding, combination, line cuts, slices. Image I/O and SPEC parsing
live in ssrl_xrd_tools.io; fitting in ssrl_xrd_tools.analysis.fitting.
"""

from ssrl_xrd_tools.rsm.volume import (
    RSMVolume,
    extract_2d_slice,
    extract_line_cut,
    mask_data,
    save_vtk,
)
from ssrl_xrd_tools.rsm.geometry import DiffractometerConfig
from ssrl_xrd_tools.rsm.gridding import (
    combine_grids,
    get_common_grid,
    grid_img_data,
)
from ssrl_xrd_tools.rsm.pipeline import (
    ExperimentConfig,
    ScanInfo,
    load_images,
    process_scan,
    process_scan_data,
)

# Backward-compatible alias
extract_slice = extract_2d_slice
