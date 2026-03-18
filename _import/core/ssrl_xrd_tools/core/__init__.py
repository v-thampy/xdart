# ssrl_xrd_tools/core/__init__.py
"""
Shared primitives: scan metadata and integration/calibration containers.
"""

from __future__ import annotations

from ssrl_xrd_tools.core.containers import (
    IntegrationResult1D,
    IntegrationResult2D,
    PONI,
)
from ssrl_xrd_tools.core.metadata import ScanMetadata
from ssrl_xrd_tools.core.hdf5 import (
    data_to_h5,
    none_to_h5,
    dict_to_h5,
    str_to_h5,
    scalar_to_h5,
    arr_to_h5,
    series_to_h5,
    dataframe_to_h5,
    index_to_h5,
    encoded_h5,
    attributes_to_h5,
    h5_to_data,
    h5_to_dict,
    h5_to_attributes,
    h5_to_index,
    check_encoded,
    soft_list_eval,
    catch_h5py_file,
)

__all__ = [
    "IntegrationResult1D",
    "IntegrationResult2D",
    "PONI",
    "ScanMetadata",
    # HDF5 codec
    "data_to_h5",
    "none_to_h5",
    "dict_to_h5",
    "str_to_h5",
    "scalar_to_h5",
    "arr_to_h5",
    "series_to_h5",
    "dataframe_to_h5",
    "index_to_h5",
    "encoded_h5",
    "attributes_to_h5",
    "h5_to_data",
    "h5_to_dict",
    "h5_to_attributes",
    "h5_to_index",
    "check_encoded",
    "soft_list_eval",
    "catch_h5py_file",
]
