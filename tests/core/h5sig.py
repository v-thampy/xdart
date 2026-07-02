"""Canonical content signature of an HDF5 tree, for write-compat gates.

Walks every group/dataset: names, shapes, dtypes, HDF5 storage layout, value
digests, and attributes -- EXCLUDING volatile provenance values (timestamps,
package versions, hostname, python) whose change is expected run-to-run.
"""
from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np

#: attr/dataset LEAF names whose VALUES are volatile (still recorded as
#: present, just not by value)
VOLATILE_LEAVES = {
    "date", "timestamp", "host", "hostname", "python", "platform",
    # per-run environment noise: the output tmp path, write clock, and the
    # (absolute, env-dependent) project root -- presence still asserted
    "file_name", "file_time", "source_base",
    # environment builds, not record content: the root attrs nexusformat
    # stamps from the running libhdf5/h5py/nexusformat, and the
    # reduction-group program version (importlib metadata of the active
    # install).  Hashing these pinned the TEST MACHINE into the fixture —
    # the gate failed under any other env (e.g. the shipped conda env) and
    # on every h5py upgrade.  Presence is still asserted.
    "HDF5_Version", "h5py_version", "creator_version", "version",
}
VOLATILE_PARENTS = ("versions",)   # /entry/reduction/versions/* values


def _digest(value) -> str:
    arr = np.asarray(value)
    if arr.dtype.kind in ("O", "U", "S"):
        data = np.asarray(arr, dtype="S").tobytes()
    else:
        data = arr.tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_volatile(path: str, leaf: str) -> bool:
    if leaf in VOLATILE_LEAVES:
        return True
    return any(f"/{p}/" in f"/{path}/" for p in VOLATILE_PARENTS)


def h5_content_signature(path) -> dict:
    sig: dict = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            entry: dict = {"kind": "group" if isinstance(obj, h5py.Group) else "dataset"}
            if isinstance(obj, h5py.Dataset):
                entry["shape"] = list(obj.shape)
                entry["dtype"] = str(obj.dtype)
                entry["compression"] = _jsonable(obj.compression)
                entry["compression_opts"] = _jsonable(obj.compression_opts)
                entry["chunks"] = _jsonable(obj.chunks)
                entry["maxshape"] = _jsonable(obj.maxshape)
                leaf = name.rsplit("/", 1)[-1]
                entry["value"] = ("<volatile>" if _is_volatile(name, leaf)
                                  else _digest(obj[()]))
            attrs = {}
            for k in sorted(obj.attrs):
                if _is_volatile(name, k):
                    attrs[k] = "<volatile>"
                else:
                    attrs[k] = _digest(obj.attrs[k])
            entry["attrs"] = attrs
            sig[name] = entry
        f.visititems(visit)
        sig["/"] = {"kind": "group",
                    "attrs": {k: ("<volatile>" if _is_volatile("/", k)
                                  else _digest(f.attrs[k]))
                              for k in sorted(f.attrs)}}
    return sig


def dump(path, out_json):
    with open(out_json, "w") as fh:
        json.dump(h5_content_signature(path), fh, indent=1, sort_keys=True)
