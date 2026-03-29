# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import time
import os
import subprocess
import sys

# Other imports
import xml.etree.ElementTree

import numpy as np
from pathlib import Path
from collections import OrderedDict

import scipy.ndimage

import scipy.ndimage as ndimage
from scipy.signal import medfilt2d

import pandas as pd
import yaml
import json
import h5py
import fabio

# Detector File Sizes
detector_file_sizes = {
    'Rayonix MX225': 18878464,
    'Rayonix SX165': 8392704,
    'Pilatus 100k': 379860,
    'Pilatus 1M': 4092732,
}


def write_xye(fname, xdata, ydata, variance=None):
    """Saves data to an xye file. Variance is the square root of the
    signal.
    
    args:
        fname: str, path to file
        xdata: angle or q data
        ydata: intensity
    """
    if variance is None:
        _variance = np.sqrt(abs(ydata))
    else:
        _variance = variance
    with open(fname, "w") as file:
        for i in range(0, len(xdata)):
            file.write(
                str(xdata[i]) + "\t" +
                str(ydata[i]) + "\t" +
                str(_variance[i]) + "\n"
            )


def write_csv(fname, xdata, ydata, variance=None):
    """Saves data to a csv file.
    
    args:
        fname: str, path to file
        xdata: angle or q data
        ydata: intensity
    """
    if variance is None:
        _variance = np.sqrt(abs(ydata))
    else:
        _variance = variance
    with open(fname, 'w') as file:
        for i in range(0, len(xdata)):
            file.write(str(xdata[i]) + ', ' +
                       str(ydata[i]) + ', ' +
                       str(_variance[i]) + '\n')


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


def find_between( s, first, last ):
    """find first occurence of substring in string s
     between two substrings (first and last)

    Args:
        s (str): input string
        first (str): first substring
        last (str): second substring

    Returns:
        str: substring between first and last
    """
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""


def find_between_r( s, first, last ):
    """find last occurence of substring in string s
     between two substrings (first and last)

    Args:
        s (str): input string
        first (str): first substring
        last (str): second substring

    Returns:
        str: substring between first and last
    """
    try:
        start = s.rindex(first) + len(first)
        end = s.rindex(last, start)
        return s[start:end]
    except ValueError:
        return ""


# def get_fname_dir(fname):
def get_fname_dir():
    """
    Returns directory on local drive to save temporary h5 files in

    Args:
        fname: {str} Name of scan name used to create subdirectory

    Returns:
        path: {str} Path where h5 file is saved
    """
    home_path = str(Path.home())

    path = os.path.join(home_path, 'xdart_processed_data')
    Path(path).mkdir(parents=True, exist_ok=True)

    return path


def split_file_name(fname):
    """Splits filename to get directory, file root and extension

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
    """Splits filename to get scan name and image number

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
    """ Returns the averaged image and meta data for a series

    Arguments:
        fname {str} -- full image file name with path
        detector {obj} -- pyFAI detector object
    Returns:
        series_name {str} -- series name (if series exists)
        series_file_names {str array} -- file names in series
        img_data {ndarray} -- averaged 2D intensity over a series
        img_mete {dict} -- averaged meta data over a series
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


def get_scan_name(fname):
    """Splits filename to get scan name

    Arguments:
        fname {str} -- full image file name with path
    Returns:
        scan_name {str}
    """
    directory, root, ext = split_file_name(fname)
    try:
        img_number = root[root.rindex('_') + 1:]
    except ValueError:
        img_number = ''

    if img_number:
        try:
            _ = int(img_number)
            return root[:root.rindex('_')]
        except ValueError:
            return root

    return root


def get_img_number(fname):
    """Splits filename to get scan name and image number

    Arguments:
        fname {str} -- full image file name with path
    Returns:
        scan_name {str}
        nImage {int}
    """
    # directory, root, ext = split_file_name(fname)
    root = os.path.splitext(fname)[0]
    try:
        img_number = root[root.rindex('_') + 1:]
        img_number = int(img_number)
    except ValueError:
        img_number = 1

    return img_number


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


def get_motor_val(pdi_file, motor):
    """Return position of a particular motor from PDI file

    Args:
        pdi_file (str): PDI file name with path
        motor (str): Motor name

    Returns:
        float: Motor position
    """
    from ssrl_xrd_tools.io.metadata import read_pdi_metadata
    motors = read_pdi_metadata(pdi_file)
    return motors[motor]


def get_img_data(
        fname, detector, orientation='horizontal',
        flip=False, fliplr=False, transpose=False,
        return_float=False, im=0):
    """Read image file and return numpy array

    Args:
        fname (str): File Name with path
        detector (detector object): pyFAI detector object
        orientation (str, optional): Orientation of detector. Options: 'horizontal', 'vertical'. Defaults to 'horizontal'.
        flip (bool, optional): Flag to flip the image up-down (required by pyFAI at times). Defaults to False.
        fliplr (bool, optional): Flag to flip the image left-right (required by pyFAI at times). Defaults to False.
        transpose (bool, optional): Flag to transpose the image (required by pyFAI at times). Defaults to False.
        return_float (bool, optional): Convert array to float. Defaults to False.
        im (integer, optional): image number if input is h5 file from Eiger. Defaults to 0

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


def get_mask_array(detector, mask_file=None, det_orientation=0):
    """Get mask array from mask file
    """
    mask = detector.calc_mask()
    if mask_file and os.path.exists(mask_file):
        if mask is not None:
            try:
                mask += fabio.open(mask_file).data
            except ValueError:
                print('Mask file not valid for Detector')
                pass
        else:
            mask = fabio.open(mask_file).data

    if mask is None:
        return None

    if mask.shape != detector.shape:
        print('Mask file not valid for Detector')
        return None

    mask = scipy.ndimage.rotate(mask, det_orientation)
    return np.flatnonzero(mask)


def get_norm_fac(normChannel, scan_data, arch_ids=None, return_sum=True):
    """Check to see if normalization channel exists in metadata and return name"""
    normChannel = get_normChannel(normChannel, scan_data_keys=scan_data.columns)
    if arch_ids is None:
        arch_ids = scan_data.index
    norm_fac = scan_data[normChannel][arch_ids] if normChannel else 1
    if return_sum and not isinstance(norm_fac, int):
        norm_fac = norm_fac.mean()

    return norm_fac


def get_normChannel(normChannel, scan_data_keys):
    """Check to see if normalization channel exists in metadata and return name"""
    if normChannel == 'sec':
        normChannel = {'sec', 'seconds', 'Seconds', 'Sec', 'SECONDS', 'SEC'}
    else:
        normChannel = {normChannel, normChannel.lower(), normChannel.upper()}
    normChannel = normChannel.intersection(scan_data_keys)
    return normChannel.pop() if len(normChannel) > 0 else None


def smooth_img(img, kernel_size=3, window_size=3, order=0):
    """Apply a Gaussian filter to smooth image

    Args:
        img (ndarray): 2D numpy array for image
        kernel_size (int, optional): Gaussian filter kernel. Defaults to 3.
        window_size (int, optional): Gaussian filter window (should be odd). Defaults to 3.
        order (int, optional): Order of the filter. Defaults to 0.

    Returns:
        ndarray: Smoothed image
    """
    if (np.mod(kernel_size, 2) == 0) or (np.mod(window_size, 2) == 0):
        print('Smoothing windows should be odd integers')
        return img

    if order >= window_size:
        order = window_size - 1

    if kernel_size > 1:
        img = medfilt2d(img, 3)
    if window_size > 1:
        img = ndimage.gaussian_filter(img, sigma=(window_size, window_size), order=order)

    return img


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
                    # pos = False if (abs(k - keys[0]) > abs(k - keys[-1])) else True
                except ValueError:
                    self.popitem(False)

        OrderedDict.__setitem__(self, key, value)


def launch(program):
    """launch(program)
      Run program as if it had been double-clicked in Finder, Explorer,
      Nautilus, etc. On OS X, the program should be a .app bundle, not a
      UNIX executable. When used with a URL, a non-executable file, etc.,
      the behavior is implementation-defined.

      Returns something false (0 or None) on success; returns something
      True (e.g., an error code from open or xdg-open) or throws on failure.
      However, note that in some cases the command may succeed without
      actually launching the targeted program."""
    if sys.platform == 'darwin':
        ret = subprocess.call(['open', program])
    elif sys.platform.startswith('win'):
        ret = os.startfile(os.path.normpath(program))
    else:
        ret = subprocess.call(['xdg-open', program])
    return ret
