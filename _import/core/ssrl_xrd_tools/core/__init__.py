"""Headless core contracts and containers for ``ssrl_xrd_tools``.

The public ``core`` namespace intentionally exposes one canonical input-side
``FrameGeometry`` from :mod:`ssrl_xrd_tools.core.scan`.  The display/round-trip
geometry type used by :class:`FrameView` remains available as
``ViewFrameGeometry`` so callers never depend on import-order shadowing.
"""

from __future__ import annotations

from ssrl_xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
from ssrl_xrd_tools.core.frame_view import (
    Axis,
    FrameGeometry as ViewFrameGeometry,
    FrameView,
    TwoDKind,
    assert_frameview_equivalent,
    axis_from_unit,
    two_d_kind_from_units,
)
from ssrl_xrd_tools.core.hdf5 import (
    arr_to_h5,
    attributes_to_h5,
    catch_h5py_file,
    check_encoded,
    data_to_h5,
    dataframe_to_h5,
    dict_to_h5,
    encoded_h5,
    h5_to_attributes,
    h5_to_data,
    h5_to_dict,
    h5_to_index,
    index_to_h5,
    none_to_h5,
    scalar_to_h5,
    series_to_h5,
    soft_list_eval,
    str_to_h5,
)
from ssrl_xrd_tools.core.metadata import (
    HeterogeneousMetadata,
    ScanMetadata,
    numeric_metadata,
)
from ssrl_xrd_tools.core.scan import (
    Frame,
    FrameGeometry,
    FrameSource,
    MaskSpec,
    Scan,
    ScanFrame,
    SourceCapabilities,
    SourceKind,
    SourceSpec,
    coerce_source_kind,
)

__all__ = [
    "IntegrationResult1D",
    "IntegrationResult2D",
    "PONI",
    "Axis",
    "FrameGeometry",
    "ViewFrameGeometry",
    "FrameView",
    "TwoDKind",
    "assert_frameview_equivalent",
    "axis_from_unit",
    "two_d_kind_from_units",
    "ScanMetadata",
    "Frame",
    "FrameSource",
    "HeterogeneousMetadata",
    "MaskSpec",
    "Scan",
    "ScanFrame",
    "SourceCapabilities",
    "SourceKind",
    "SourceSpec",
    "coerce_source_kind",
    "numeric_metadata",
    # HDF5 codec
    "arr_to_h5",
    "attributes_to_h5",
    "catch_h5py_file",
    "check_encoded",
    "data_to_h5",
    "dataframe_to_h5",
    "dict_to_h5",
    "encoded_h5",
    "h5_to_attributes",
    "h5_to_data",
    "h5_to_dict",
    "h5_to_index",
    "index_to_h5",
    "none_to_h5",
    "scalar_to_h5",
    "series_to_h5",
    "soft_list_eval",
    "str_to_h5",
]
