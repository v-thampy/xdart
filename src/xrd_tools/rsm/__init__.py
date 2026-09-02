"""Reciprocal-space mapping API with lazy implementation imports.

Importing :mod:`xrd_tools.rsm` exposes the stable public surface without
loading array, file-format, xrayutilities, VTK, or GUI dependencies. Each
implementation module is loaded only when its public symbol is first used.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import Any as _Any


_EXPORTS = {
    # Canonical geometry owners.
    "DetectorHeader": ("xrd_tools.core.geometry", "DetectorHeader"),
    "DiffractometerConfig": (
        "xrd_tools.core.geometry",
        "DiffractometerConfig",
    ),
    "PixelQMap": ("xrd_tools.core.geometry", "PixelQMap"),
    # Volume and presentation helpers.
    "RSMVolume": ("xrd_tools.rsm.volume", "RSMVolume"),
    "extract_2d_slice": ("xrd_tools.rsm.volume", "extract_2d_slice"),
    "extract_slice": ("xrd_tools.rsm.volume", "extract_2d_slice"),
    "extract_line_cut": ("xrd_tools.rsm.volume", "extract_line_cut"),
    "mask_data": ("xrd_tools.rsm.volume", "mask_data"),
    "save_vtk": ("xrd_tools.rsm.volume", "save_vtk"),
    # In-memory gridding.
    "StreamingGridder": ("xrd_tools.rsm.gridding", "StreamingGridder"),
    "StreamingScan": ("xrd_tools.rsm.gridding", "StreamingScan"),
    "combine_grids": ("xrd_tools.rsm.gridding", "combine_grids"),
    "get_common_grid": ("xrd_tools.rsm.gridding", "get_common_grid"),
    "grid_img_data": ("xrd_tools.rsm.gridding", "grid_img_data"),
    "grid_img_data_streaming": (
        "xrd_tools.rsm.gridding",
        "grid_img_data_streaming",
    ),
    # Configuration and source-backed pipeline. The historical package name
    # resolves to this pipeline implementation, not the incompatible in-memory
    # helper of the same name in rsm.gridding.
    "ExperimentConfig": ("xrd_tools.core.config", "ExperimentConfig"),
    "ScanInfo": ("xrd_tools.rsm.pipeline", "ScanInfo"),
    "ScanInput": ("xrd_tools.rsm.pipeline", "ScanInput"),
    "grid_scans_streaming": (
        "xrd_tools.rsm.pipeline",
        "grid_scans_streaming",
    ),
    "load_images": ("xrd_tools.rsm.pipeline", "load_images"),
    "process_scan": ("xrd_tools.rsm.pipeline", "process_scan"),
    "process_scan_data": ("xrd_tools.rsm.pipeline", "process_scan_data"),
    "process_scan_from_nexus": (
        "xrd_tools.rsm.pipeline",
        "process_scan_from_nexus",
    ),
}

_SUBMODULES = {
    "volume": "xrd_tools.rsm.volume",
    "gridding": "xrd_tools.rsm.gridding",
    "pipeline": "xrd_tools.rsm.pipeline",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> _Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        try:
            module_name = _SUBMODULES[name]
        except KeyError:
            raise AttributeError(name) from error
        value = _import_module(module_name)
    else:
        value = getattr(_import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS) | set(_SUBMODULES))
