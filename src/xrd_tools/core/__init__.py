"""Headless core contracts and containers for ``xrd_tools``.

The public ``core`` namespace intentionally exposes one canonical input-side
``FrameGeometry`` from :mod:`xrd_tools.core.scan`.  The display/round-trip
geometry type used by :class:`FrameView` remains available as
``ViewFrameGeometry`` so callers never depend on import-order shadowing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # static mirror of the lazy __getattr__ exports below
    from xrd_tools.core.hdf5 import (
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

from xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
from xrd_tools.core.filters import compile_filter
from xrd_tools.core.invalid import (
    UINT32_CEILING,
    integer_saturation_ceiling,
    saturation_pixels,
)
from xrd_tools.core.frame_view import (
    DEFAULT_MODE_KEY,
    Axis,
    FrameGeometry as ViewFrameGeometry,
    FrameRecord,
    FrameView,
    TwoDKind,
    assert_framerecord_equivalent,
    assert_frameview_equivalent,
    axis_from_unit,
    two_d_kind_from_units,
)
from xrd_tools.core.metadata import (
    HeterogeneousMetadata,
    ScanMetadata,
    numeric_metadata,
)
from xrd_tools.core.scan import (
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

# The h5py-backed codec helpers re-export lazily (PEP 562): importing
# ``xrd_tools.core`` must stay LIGHT (no h5py/pandas/yaml/fabio) so the
# Qt-free contracts (frame_view, scan, containers, metadata) are loadable
# in pure display-logic / minimal CI environments.  ``from xrd_tools.core
# import dict_to_h5`` etc. keep working via ``__getattr__``.
_HDF5_EXPORTS = frozenset({
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
})


def __getattr__(name: str):
    if name in _HDF5_EXPORTS:
        from xrd_tools.core import hdf5 as _hdf5

        return getattr(_hdf5, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _HDF5_EXPORTS)

__all__ = [
    "IntegrationResult1D",
    "IntegrationResult2D",
    "PONI",
    "Axis",
    "FrameGeometry",
    "ViewFrameGeometry",
    "FrameRecord",
    "FrameView",
    "TwoDKind",
    "DEFAULT_MODE_KEY",
    "assert_framerecord_equivalent",
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
    "compile_filter",
    "numeric_metadata",
    # detector invalid-pixel policy (R3-C)
    "UINT32_CEILING",
    "integer_saturation_ceiling",
    "saturation_pixels",
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
