"""
xrd_tools/core/hdf5.py
----------------------------
Type-aware HDF5 serialization codec.

Extracted from xdart/_utils.py so that both xrd_tools and the xdart GUI
can share the same on-disk format.

Public API
----------
write : data_to_h5, dict_to_h5, attributes_to_h5
read  : h5_to_data, h5_to_dict, h5_to_attributes
util  : check_encoded, catch_h5py_file
"""
from __future__ import annotations

import ast
import json
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager

import h5py
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ``ast.literal_eval`` is the safe drop-in for the old ``eval`` on persisted
# HDF5 keys/values.  It parses Python literals (numbers, strings, tuples,
# lists, dicts, bools, None) without executing arbitrary code.  The old
# codec used ``eval`` so external or corrupted files could execute code on
# read — replacing with ``literal_eval`` closes that hole while preserving
# compatibility for every legitimate persisted value we have in the wild
# (tuple keys like ``(0, 1)``, ints, floats, repr(str)).


def _safe_literal(value):
    """Safely decode a persisted literal.

    Accepts a bytes/str/numpy scalar produced by an earlier call to
    ``repr()``/``str()`` and returns the original Python literal. Falls
    back to ``bytes.decode()`` then the raw value on parse failure.
    Never executes code.
    """
    if isinstance(value, bytes):
        try:
            value = value.decode()
        except UnicodeDecodeError:
            return value
    if not isinstance(value, str):
        # Non-string (already-typed) values pass through unchanged.
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return value

__all__ = [
    # write
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
    # read
    "h5_to_data",
    "h5_to_dict",
    "h5_to_attributes",
    "h5_to_index",
    # util
    "check_encoded",
    "soft_list_eval",
    "catch_h5py_file",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def check_encoded(grp, name: str) -> bool:
    """Return True if ``grp.attrs['encoded'] == name``."""
    return grp.attrs.get("encoded", "not_found") == name


def soft_list_eval(data, scope: dict | None = None) -> list:
    """Return a list by safely parsing each element as a Python literal.

    Uses :func:`ast.literal_eval` so corrupted or externally-sourced HDF5
    files cannot execute code at read time.  Falls back to
    ``bytes.decode()`` then the raw value on failure.

    Parameters
    ----------
    data:
        Iterable of items (typically byte-strings from an HDF5 index).
    scope:
        Accepted for backwards compatibility but ignored — the safe parser
        does not take a namespace.
    """
    if scope is not None:
        logger.debug(
            "soft_list_eval received a scope argument — ignored by the safe "
            "literal parser. Remove the argument at the call site."
        )
    return [_safe_literal(x) for x in data]


def _h5_open_error_message(filename, mode, exc) -> str:
    """A clear, actionable message for a failed HDF5 open, distinguishing a
    file-LOCK collision (errno EAGAIN / 'unable to lock') from other OSErrors."""
    text = str(exc or "")
    locked = (
        isinstance(exc, BlockingIOError)
        or getattr(exc, "errno", None) in (11, 35)   # EAGAIN (Linux 11, macOS 35)
        or "unable to lock" in text.lower()
        or "resource temporarily unavailable" in text.lower()
    )
    if locked:
        return (
            f"Could not open '{filename}' (mode '{mode}'): the file is LOCKED. "
            "Another process has it open — most likely a second xdart instance "
            "writing to the same Save Path, or a stale lock left by a crashed run. "
            "Close the other instance (or point it at a different Project Folder / "
            f"Save Path) and try again.  [{type(exc).__name__}: {exc}]"
        )
    return (
        f"Could not open '{filename}' (mode '{mode}') after retries: "
        f"{type(exc).__name__}: {exc}"
    )


@contextmanager
def catch_h5py_file(filename: str, mode: str = "r", tries: int = 100,
                    *args, **kwargs):
    """Context manager that opens an HDF5 file, retrying on ``OSError``.

    Useful for network-mounted filesystems (NFS) common at beamlines where
    transient lock contention raises ``OSError``.

    Parameters
    ----------
    filename:
        Path to the HDF5 file.
    mode:
        Open mode passed to ``h5py.File``.
    tries:
        Maximum number of attempts before re-raising.
    """
    hdf5_file = None
    last_exc = None
    for i in range(tries):
        if i > 0 and i % 10 == 0:
            logger.debug("catch_h5py_file: attempt %d for %s", i, filename)
        try:
            hdf5_file = h5py.File(filename, mode, *args, **kwargs)
            break
        except OSError as exc:
            last_exc = exc
            time.sleep(0.05)
    if hdf5_file is None:
        # Retries exhausted — raise a CLEAR, actionable message instead of the raw
        # BlockingIOError deep in h5py.  The common cause is a file-LOCK collision
        # (a second xdart instance writing the same Save Path), which no amount of
        # retrying can clear.
        raise OSError(_h5_open_error_message(filename, mode, last_exc)) from last_exc
    try:
        yield hdf5_file
    finally:
        hdf5_file.close()


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def none_to_h5(grp: h5py.Group, key: str) -> None:
    """Store ``None`` as an HDF5 empty dataset tagged ``encoded='None'``."""
    if key in grp:
        del grp[key]
    grp.create_dataset(key, data=h5py.Empty("f"))
    grp[key].attrs["encoded"] = "None"


def str_to_h5(data: str, grp: h5py.Group, key: str) -> None:
    """Store a Python ``str`` as an HDF5 variable-length string dataset."""
    if key in grp:
        if check_encoded(grp[key], "str"):
            grp[key][()] = data
            return
        del grp[key]
    grp.create_dataset(key, data=data, dtype=h5py.string_dtype())


def scalar_to_h5(data, grp: h5py.Group, key: str) -> None:
    """Store a scalar (int, float, bool, …) as a 0-d HDF5 dataset."""
    if key in grp:
        if check_encoded(grp[key], "scalar"):
            if grp[key].dtype == np.array(data).dtype:
                grp[key][()] = data
                return
        del grp[key]
    grp.create_dataset(key, data=data)
    grp[key].attrs["encoded"] = "scalar"


def arr_to_h5(data, grp: h5py.Group, key: str, compression) -> None:
    """Store a numpy array (or list) as a resizable HDF5 dataset.

    Arrays named ``map_raw`` or ``bg_raw`` are stored as ``int32``;
    ``i_tthChi``, ``i_qChi``, and ``i_QxyQz`` as ``float32``; everything
    else at its natural dtype.
    """
    if key in ("map_raw", "bg_raw"):
        if np.issubdtype(np.asarray(data).dtype, np.integer):
            arr = np.asarray(data)  # already integer — no NaN possible, skip cast
        else:
            arr = np.nan_to_num(data, nan=0).astype("int32")
    elif key in ("i_tthChi", "i_qChi", "i_QxyQz"):
        arr = np.array(data, dtype="float32")
    else:
        arr = np.array(data)

    if key in grp:
        if check_encoded(grp[key], "arr"):
            if grp[key].dtype == arr.dtype:
                grp[key].resize(arr.shape)
                grp[key][()] = arr[()]
                return
        del grp[key]
    grp.create_dataset(
        key, data=arr,
        maxshape=tuple(None for _ in arr.shape),
    )
    grp[key].attrs["encoded"] = "arr"


def _portable_compression(compression):
    """Normalize a filter to the portable policy: ``lzf`` -> ``gzip`` on EVERY
    platform.  lzf is h5py-only and bus-errors on some ARM64-macOS builds; gzip
    is in every HDF5 build and stock-h5py readable.  Mirrors
    ``xrd_tools.io.nexus._comp_kwargs`` / ``containers._h5_replace`` so these
    pandas-table create_dataset paths never emit lzf either (no core->io import)."""
    return "gzip" if compression == "lzf" else compression


def series_to_h5(data: pd.Series, grp: h5py.Group, key: str,
                 compression) -> None:
    """Store a :class:`pandas.Series` in a labelled HDF5 group."""
    compression = _portable_compression(compression)
    if key in grp:
        if check_encoded(grp[key], "Series"):
            new_grp = grp[key]
            new_grp["data"][()] = np.array(data)
            index_to_h5(data.index, "index", new_grp, compression)
            new_grp.attrs["name"] = data.name
            return
        del grp[key]
    new_grp = grp.create_group(key)
    new_grp.attrs["encoded"] = "Series"
    new_grp.create_dataset(
        "data", data=np.array(data), compression=compression, chunks=True
    )
    index_to_h5(data.index, "index", new_grp, compression)
    new_grp.attrs["name"] = data.name


def dataframe_to_h5(data: pd.DataFrame, grp: h5py.Group, key: str,
                    compression) -> None:
    """Store a :class:`pandas.DataFrame` in a labelled HDF5 group."""
    compression = _portable_compression(compression)
    if key in grp:
        if check_encoded(grp[key], "DataFrame"):
            new_grp = grp[key]
        else:
            del grp[key]
            new_grp = grp.create_group(key)
            new_grp.attrs["encoded"] = "DataFrame"
    else:
        new_grp = grp.create_group(key)
        new_grp.attrs["encoded"] = "DataFrame"
    index_to_h5(data.index, "index", new_grp, compression)
    index_to_h5(data.columns, "columns", new_grp, compression)
    if "data" in new_grp:
        new_grp["data"].resize(np.array(data).shape)
        new_grp["data"][()] = np.array(data)[()]
    else:
        new_grp.create_dataset(
            "data", data=np.array(data), compression=compression,
            chunks=True, maxshape=(None, None),
        )


def index_to_h5(index, key: str, grp: h5py.Group, compression) -> None:
    """Store a :class:`pandas.Index` (or column labels) inside *grp*."""
    compression = _portable_compression(compression)
    if key in grp:
        if grp[key].shape == (0,):
            del grp[key]

    if index.dtype == "object":
        if len(index) > 0:
            strindex = np.array([np.bytes_(str(x)) for x in index])
            if key in grp:
                grp[key].resize(strindex.shape)
                grp[key][()] = strindex[()]
            else:
                grp.create_dataset(
                    key, data=strindex, dtype=h5py.string_dtype(),
                    chunks=True, maxshape=(None,),
                )
        else:
            if key in grp:
                del grp[key]
            grp.create_dataset(key, data=np.array([]))
    else:
        arrindex = np.array(index)
        if key in grp:
            grp[key].resize(arrindex.shape)
            grp[key][()] = arrindex[()]
        else:
            grp.create_dataset(
                key, data=np.array(index), compression=compression,
                chunks=True, maxshape=(None,),
            )


def encoded_h5(data, grp: h5py.Group, key: str, encoder: str) -> None:
    """Store an object of unknown type by encoding it as YAML or JSON bytes."""
    if encoder == "yaml":
        string = np.bytes_(yaml.dump(data))
    elif encoder == "json":
        string = np.bytes_(json.dumps(data))
    else:
        raise ValueError(f"Unknown encoder: {encoder!r}")
    if key in grp:
        if check_encoded(grp[key], encoder):
            grp[key][()] = string
            return
        del grp[key]
    grp.create_dataset(key, data=string, dtype=h5py.string_dtype())
    grp[key].attrs["encoded"] = encoder


def dict_to_h5(data: dict, grp: h5py.Group, key: str, **kwargs) -> None:
    """Recursively store a dictionary in an HDF5 group.

    Each dictionary key becomes a sub-key inside a new group tagged
    ``encoded='dict'``.  Values are dispatched through :func:`data_to_h5`.
    """
    if key in grp:
        if not check_encoded(grp[key], "dict"):
            del grp[key]
            new_grp = grp.create_group(key)
            new_grp.attrs["encoded"] = "dict"
        else:
            new_grp = grp[key]
    else:
        new_grp = grp.create_group(key)
        new_grp.attrs["encoded"] = "dict"

    for jey in data:
        data_to_h5(data[jey], new_grp, str(jey), **kwargs)


def attributes_to_h5(obj, grp: h5py.Group, lst_attr: list | None = None,
                     priv: bool = False, dpriv: bool = False,
                     **kwargs) -> None:
    """Serialize a list of object attributes into an HDF5 group.

    Parameters
    ----------
    obj:
        Object whose attributes are to be saved.
    grp:
        Destination HDF5 group.
    lst_attr:
        Explicit list of attribute names.  If ``None``, all public
        attributes (or all attributes if *priv*/*dpriv* are set) are used.
    priv:
        Include single-underscore attributes when *lst_attr* is ``None``.
    dpriv:
        Include double-underscore attributes when *lst_attr* is ``None``.
    kwargs:
        Passed through to :func:`data_to_h5`.
    """
    if lst_attr is None:
        if dpriv:
            lst_attr = list(obj.__dict__.keys())
        elif priv:
            lst_attr = [x for x in obj.__dict__.keys() if "__" not in x]
        else:
            lst_attr = [x for x in obj.__dict__.keys() if "_" not in x]
    for attr in lst_attr:
        data_to_h5(getattr(obj, attr), grp, attr, **kwargs)


def data_to_h5(data, grp: h5py.Group, key: str, encoder: str = "yaml",
               compression: str = "gzip") -> None:
    """Type-aware dispatcher: save *data* to ``grp[key]``.

    Dispatches to the appropriate typed writer based on
    ``type(data)``.  Falls back to YAML/JSON encoding, then to a raw
    bytes representation.

    Parameters
    ----------
    data:
        Any Python object to serialise.
    grp:
        Destination HDF5 file or group.
    key:
        Dataset / sub-group name within *grp*.
    encoder:
        Fallback encoder for unknown types: ``'yaml'`` (default) or
        ``'json'``.
    compression:
        HDF5 compression filter (default ``'gzip'``; ``'lzf'`` is normalized to
        gzip on every platform — see :func:`_portable_compression`).
    """
    if data is None:
        none_to_h5(grp, key)
    elif type(data) is dict:
        dict_to_h5(data, grp, key, compression=compression)
    elif type(data) is str:
        str_to_h5(data, grp, key)
    elif type(data) is pd.Series:
        series_to_h5(data, grp, key, compression)
    elif type(data) is pd.DataFrame:
        dataframe_to_h5(data, grp, key, compression)
    else:
        try:
            if np.array(data).shape == ():
                scalar_to_h5(data, grp, key)
            else:
                arr_to_h5(data, grp, key, compression)
        except TypeError:
            try:
                encoded_h5(data, grp, key, encoder)
            except Exception as exc:
                logger.debug("encoded_h5 failed for key=%r: %s", key, exc)
                try:
                    if key in grp:
                        if check_encoded(grp[key], "unknown"):
                            grp[key][()] = np.bytes_(data)
                            return
                        del grp[key]
                    grp.create_dataset(
                        key, data=np.bytes_(data),
                        dtype=h5py.string_dtype(),
                    )
                    grp[key].attrs["encoded"] = "unknown"
                except Exception as exc2:
                    logger.warning("Unable to dump key=%r: %s", key, exc2)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def h5_to_index(grp: h5py.Dataset) -> list | np.ndarray:
    """Restore a :class:`pandas.Index` from an HDF5 dataset."""
    if np.issubdtype(grp.dtype, np.number):
        return grp[()]
    return soft_list_eval(grp)


def h5_to_data(grp, encoder: bool = True,
               Loader=yaml.SafeLoader):
    """Read a value from an HDF5 dataset or group, restoring its Python type.

    Parameters
    ----------
    grp:
        HDF5 dataset or group to read.
    encoder:
        If ``True``, inspect the ``encoded`` attribute to select the decoder.
    Loader:
        :mod:`yaml` Loader class used when ``encoded == 'yaml'``. Defaults
        to :class:`yaml.SafeLoader`, which only materialises standard YAML
        types. Callers that need to restore arbitrary Python objects from
        trusted files may pass :class:`yaml.UnsafeLoader` explicitly, but
        this is discouraged — use an explicit serialization format for
        non-standard types instead.
    """
    if encoder and "encoded" in grp.attrs:
        encoded = grp.attrs["encoded"]
        if encoded == "None":
            return None
        if encoded == "dict":
            return h5_to_dict(grp, encoder=encoder, Loader=Loader)
        if encoded == "str":
            try:
                return grp[...].item().decode()
            except AttributeError:
                return grp[()]
        if encoded == "Series":
            return pd.Series(
                data=grp["data"][()],
                index=h5_to_index(grp["index"]),
                name=grp.attrs["name"],
            )
        if encoded == "DataFrame":
            return pd.DataFrame(
                data=grp["data"][()],
                index=h5_to_index(grp["index"]),
                columns=h5_to_index(grp["columns"]),
            )
        if encoded in ("data", "arr", "scalar"):
            return grp[()]
        if encoded == "yaml":
            return yaml.load(grp[...].item(), Loader=Loader)
        if encoded == "json":
            return json.loads(grp[...].item())
        if encoded == "unknown":
            raw = grp[...].item()
            # Safe decode path: try literal_eval, then fall back to the
            # raw decoded string. Never execute code on persisted bytes.
            decoded = _safe_literal(raw)
            if isinstance(decoded, str) or not isinstance(raw, (bytes, str)):
                # literal_eval may legitimately return a str (repr of str)
                # or may have fallen back to the raw decoded form.
                return decoded
            return decoded
    else:
        if isinstance(grp, h5py.Group):
            return h5_to_dict(grp, encoder=encoder, Loader=Loader)
        if grp.shape == ():
            temp = grp[...].item()
            if isinstance(temp, bytes):
                temp = temp.decode()
            return None if temp == "None" else temp
        if grp.shape is None:
            return None
        return grp[()]


def h5_to_dict(grp: h5py.Group, **kwargs) -> dict:
    """Convert an HDF5 group to a Python dictionary.

    Each key in the group is parsed with :func:`ast.literal_eval` to
    restore its original type (e.g. integer keys, tuple keys such as
    ``(0, 1)``).  Values are dispatched through :func:`h5_to_data`. The
    parser is safe against arbitrary code execution.
    """
    data: dict = {}
    for key in grp.keys():
        e_key = _safe_literal(key)
        data[e_key] = h5_to_data(grp[key], **kwargs)
    return data


def h5_to_attributes(obj, grp: h5py.Group,
                     lst_attr: list | None = None, **kwargs) -> None:
    """Set attributes of *obj* from matching keys in an HDF5 group.

    Parameters
    ----------
    obj:
        Target object whose attributes are updated.
    grp:
        Source HDF5 group.
    lst_attr:
        Explicit list of attribute names to restore.  If ``None``, all
        keys in the group that correspond to existing attributes are used.
    kwargs:
        Passed through to :func:`h5_to_data`.
    """
    if lst_attr is None:
        lst_attr = grp.keys()
    for attr in lst_attr:
        if attr in obj.__dict__:
            try:
                setattr(obj, attr, h5_to_data(grp[attr], **kwargs))
            except KeyError:
                pass
