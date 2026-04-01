# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import os

# Other imports
import numpy as np
from pathlib import Path
from collections import OrderedDict

import fabio

# Detector File Sizes
detector_file_sizes = {
    'Rayonix MX225': 18878464,
    'Rayonix SX165': 8392704,
    'Pilatus 100k': 379860,
    'Pilatus 1M': 4092732,
}

# ---------------------------------------------------------------------------
# I/O helpers — re-exported from ssrl_xrd_tools
# ---------------------------------------------------------------------------
from ssrl_xrd_tools.io.export import write_xye  # noqa: E402, F401

# ---------------------------------------------------------------------------
# HDF5 codec — imported from ssrl_xrd_tools.core.hdf5
# ---------------------------------------------------------------------------
from ssrl_xrd_tools.core.hdf5 import (  # noqa: E402
    check_encoded,
    soft_list_eval,
    catch_h5py_file,
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
    h5_to_index,
    h5_to_dict,
    h5_to_attributes,
)


def get_fname_dir():
    """
    Returns directory on local drive to save temporary h5 files in.

    Returns:
        path: {str} Path where h5 file is saved
    """
    home_path = str(Path.home())
    path = os.path.join(home_path, 'xdart_processed_data')
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def split_file_name(fname):
    """Splits filename to get directory, file root and extension.

    Arguments:
        fname {str} -- full image file name with path
    """
    directory = os.path.dirname(fname)
    root, ext = os.path.splitext(os.path.basename(fname))

    if len(ext) > 0:
        if ext[0] == '.':
            ext = ext[1:]

    return directory, root, ext


def get_sname_img_number(fname):
    """Splits filename to get scan name and image number.

    Arguments:
        fname {str} -- full image file name with path
    Returns:
        series_name {str} -- series name (or root if single acquisition)
        img_number {int/None} -- image number (if part of series)
    """
    directory, root, ext = split_file_name(fname)
    try:
        img_number = int(root[root.rindex('_') + 1:])
        root = root[:root.rindex('_')]
    except ValueError:
        img_number = None

    return root, img_number


def get_series_avg(fname, detector, meta_ext):
    """Returns the averaged image and meta data for a series.

    Arguments:
        fname {str} -- full image file name with path
        detector {obj} -- pyFAI detector object
    Returns:
        series_name {str} -- series name (if series exists)
        series_file_names {str array} -- file names in series
        img_data {ndarray} -- averaged 2D intensity over a series
        img_meta {dict} -- averaged meta data over a series
    """
    fpath = Path(fname)
    series_name, img_number = get_sname_img_number(fname)
    if img_number is None:
        return None, None, None, None

    fnames = [str(f) for f in (fpath.parent.glob(f'{series_name}*[0-9][0-9][0-9][0-9]{fpath.suffix}'))]
    data, img_meta = 0, {}
    for ii, fname in enumerate(fnames):
        data += get_img_data(fname, detector, return_float=True)
        if data is None:
            return None, None, None, None

        meta = get_img_meta(fname, meta_ext) if meta_ext else {}
        if ii == 0:
            img_meta = meta
        else:
            for (k, v) in meta.items():
                try:
                    img_meta[k] += meta[k]
                except TypeError:
                    pass

    n = len(fnames)
    data /= n
    for (k, v) in img_meta.items():
        try:
            img_meta[k] /= n
        except TypeError:
            pass

    return series_name, fnames, data, img_meta


def match_img_detector(img_file, poni):
    """Check if the file is created by the detector specified by *poni*.

    Parameters
    ----------
    img_file : str | Path
        Path to the candidate image file.
    poni : ssrl_xrd_tools.core.containers.PONI
        Calibration object whose ``detector`` field is the pyFAI detector name.
    """
    if poni is None:
        return True
    detector_name = poni.detector if poni.detector else ''
    if detector_name not in detector_file_sizes:
        return True
    return os.stat(img_file).st_size == detector_file_sizes[detector_name]


def get_img_meta(img_file, meta_ext, spec_path=None, rv='all'):
    """Get image meta data from pdi/txt files for different beamlines.

    Delegates to ssrl_xrd_tools.io.metadata.read_image_metadata.

    Args:
        img_file (str): Image file for which meta data is required
        meta_ext (str): Extension of Meta file ('txt', 'pdi', or 'SPEC')
        spec_path (str): Unused; kept for backward compatibility
        rv (str, optional): Unused; kept for backward compatibility

    Returns:
        [dict]: Dictionary with all the meta data
    """
    from ssrl_xrd_tools.io.metadata import read_image_metadata
    fmt = 'spec' if (meta_ext or '').upper() == 'SPEC' else (meta_ext or 'txt')
    return read_image_metadata(img_file, meta_format=fmt)


def get_img_data(
        fname, detector, orientation='horizontal',
        flip=False, fliplr=False, transpose=False,
        return_float=False, im=0):
    """Read image file and return numpy array.

    Args:
        fname (str): File Name with path
        detector (detector object): pyFAI detector object
        orientation (str, optional): Orientation of detector.
            Options: 'horizontal', 'vertical'. Defaults to 'horizontal'.
        flip (bool, optional): Flip up-down. Defaults to False.
        fliplr (bool, optional): Flip left-right. Defaults to False.
        transpose (bool, optional): Transpose image. Defaults to False.
        return_float (bool, optional): Convert to float. Defaults to False.
        im (integer, optional): Frame number for multi-image files. Defaults to 0.

    Returns:
        ndarray: Image data read into numpy array
    """
    try:
        with fabio.open(fname) as f:
            if im == 0:
                img_data = f.data
            else:
                img_data = f.get_frame(im).data
    except Exception:
        # Fallback: raw binary for non-standard formats
        try:
            img_data = np.fromfile(fname, dtype='int32').reshape(detector.shape)
        except Exception:
            return None

    try:
        if img_data.shape != detector.shape:
            return None
    except AttributeError:
        return None

    if return_float:
        img_data = np.asarray(img_data, dtype=float)

    if (orientation == 'vertical') or transpose:
        img_data = img_data.T

    if flip:
        img_data = np.flipud(img_data)

    if fliplr:
        img_data = np.fliplr(img_data)

    return img_data


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if self._max > 0:
            if len(self) >= self._max:
                keys = list(self.keys())
                try:
                    k = int(key)
                    diffs = [abs(int(k_)-k) for k_ in keys]
                    out_key = keys[diffs.index(max(diffs))]
                    self.pop(out_key)
                except ValueError:
                    self.popitem(False)

        OrderedDict.__setitem__(self, key, value)
