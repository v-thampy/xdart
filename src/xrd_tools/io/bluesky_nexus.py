"""Bluesky / apstools ``NXWriter`` acquisition-file reader helpers.

SSRL bl11-3 now acquires via `Bluesky <https://blueskyproject.io>`_ and writes
``.nxs`` files with apstools' ``NXWriter``.  These files are NeXus but do NOT
follow xdart's own processed-``.nxs`` conventions, so the generic readers in
:mod:`xrd_tools.io.nexus` / :mod:`xrd_tools.io.image` mis-harvest them (80 junk
"motor" columns, NaN wavelength, ``UNKNOWN`` image classification).  This module
concentrates all Bluesky-specific knowledge in one import-light place; the
existing readers dispatch to it at their detection seams.

Ground truth (``test_data/nexus/Pt_10nm_00013.nxs``):

* root attr ``creator == "NXWriter"``; ``entry/instrument/bluesky`` (NXnote)
  present; the entry has NO ``ssrl_schema`` attribute (that marks an xdart file).
* scan motor names are the group children of ``entry/instrument/positioners``
  (authoritative real h5 groups — here ``hy``); the ``!!python/tuple`` YAML in
  ``…/bluesky/metadata/motors`` is only a cross-check (never unsafe-loaded).
* per-frame motor/counter arrays live flat in ``entry/data/<name>``
  (``hy``, ``i0``, ``i1``, ``i2``, ``pd``, ``EPOCH``, …).
* wavelength/energy live in the eiger detector config under
  ``…/bluesky/metadata/configuration/eiger/data/`` (``eiger_cam_wavelength`` Å,
  ``eiger_cam_photon_energy`` eV).
* the detector image stack is EMBEDDED at ``entry/data/eiger_image``; the NXdata
  ``@signal`` points at a *scalar counter* instead, so the detector is marked by
  an ``@signal_type == 'detector'`` attribute (on ``eiger_image`` and the
  ``entry/instrument/detectors/eiger/data`` NXdata).

Import-light: depends only on :mod:`h5py`, :mod:`numpy`, stdlib — so
:mod:`xrd_tools.io.nexus`, :mod:`~xrd_tools.io.image` and
:mod:`~xrd_tools.io.image_source` can all import it without a cycle.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import h5py
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "is_bluesky_nxwriter",
    "resolve_nxentry",
    "bluesky_motor_names",
    "bluesky_angles",
    "bluesky_counters",
    "bluesky_positioner_values",
    "bluesky_baseline_values",
    "bluesky_fixed_motor_values",
    "bluesky_wavelength",
    "bluesky_energy_kev",
    "bluesky_per_frame_table",
    "bluesky_scalar_metadata",
    "find_detector_signal_dataset",
]

# The entry attribute xdart stamps on its own processed files.  A Bluesky file
# never carries it; its presence positively excludes the Bluesky branch.  Kept
# as a literal (not imported from ``io.schema``) to keep this module import-light
# — the key is frozen (``schema.SCHEMA_NAME_ATTR``).
_XDART_SCHEMA_ATTR = "ssrl_schema"

# Default norm-channel counters (ion chambers + photodiode).  ``gate`` and
# ``eiger`` from ``…/metadata/detectors`` are excluded (a timer and the image).
_DEFAULT_BLUESKY_COUNTERS = ("i0", "i1", "i2", "pd")

_EIGER_CONFIG_BASE = "instrument/bluesky/metadata/configuration"


# ---------------------------------------------------------------------------
# small decoders
# ---------------------------------------------------------------------------

def _to_str(v: Any) -> str:
    """Decode an HDF5 attr/scalar (bytes / np.bytes_ / 0-d array) to ``str``."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _to_str(v[()])
        return _to_str(v.ravel()[0]) if v.size else ""
    return str(v) if v is not None else ""


def _nx_class(obj: Any) -> str:
    try:
        return _to_str(obj.attrs.get("NX_class", ""))
    except Exception:
        return ""


def _scalar(ds: h5py.Dataset) -> Any:
    val = ds[()]
    if isinstance(val, np.ndarray):
        return val.ravel()[0] if val.size else None
    return val


# ---------------------------------------------------------------------------
# entry resolution + detection
# ---------------------------------------------------------------------------

def resolve_nxentry(h5: h5py.File | h5py.Group,
                    entry_hint: str = "entry") -> h5py.Group | None:
    """Return the NXentry group, resolved by ``NX_class`` rather than name.

    Prefers ``entry_hint`` when it exists and is an NXentry (or any group);
    otherwise returns the first top-level group whose ``NX_class == 'NXentry'``.
    Returns *None* if none can be found.
    """
    # Already an entry group?
    if isinstance(h5, h5py.Group) and not isinstance(h5, h5py.File):
        if _nx_class(h5) == "NXentry":
            return h5
    root = h5.file if isinstance(h5, h5py.Group) else h5
    hint = root.get(entry_hint)
    if isinstance(hint, h5py.Group) and _nx_class(hint) in ("NXentry", ""):
        # honor the hint when it is an NXentry (or class-less but named 'entry')
        if _nx_class(hint) == "NXentry" or entry_hint == "entry":
            return hint
    for name in root:
        obj = root.get(name)
        if isinstance(obj, h5py.Group) and _nx_class(obj) == "NXentry":
            return obj
    if isinstance(hint, h5py.Group):
        return hint
    return None


def _entry_and_root(h5_or_entry: h5py.File | h5py.Group,
                    ) -> tuple[h5py.Group | None, h5py.File | None]:
    """Normalize the argument into ``(entry_group, root_file)``."""
    try:
        if isinstance(h5_or_entry, h5py.Group) and not isinstance(h5_or_entry, h5py.File):
            root = h5_or_entry.file
            if _nx_class(h5_or_entry) == "NXentry":
                return h5_or_entry, root
            return resolve_nxentry(root), root
        root = h5_or_entry  # h5py.File
        return resolve_nxentry(root), root
    except Exception:
        return None, None


def is_bluesky_nxwriter(h5_or_entry: h5py.File | h5py.Group) -> bool:
    """True if the open file is a Bluesky/apstools ``NXWriter`` acquisition file.

    Accepts either the open :class:`h5py.File` or its NXentry group.  A file
    qualifies when it does NOT carry xdart's ``ssrl_schema`` entry attribute
    (which would mark a processed xdart ``.nxs``) AND shows a positive Bluesky
    signal: the root ``creator == 'NXWriter'`` attribute and/or an
    ``entry/instrument/bluesky`` group.
    """
    entry, root = _entry_and_root(h5_or_entry)
    if root is None:
        return False
    # Positive exclusion: an xdart-processed file is never Bluesky.
    if entry is not None and _XDART_SCHEMA_ATTR in entry.attrs:
        return False
    try:
        creator = _to_str(root.attrs.get("creator", ""))
    except Exception:
        creator = ""
    has_bluesky = entry is not None and "instrument/bluesky" in entry
    return creator == "NXWriter" or bool(has_bluesky)


# ---------------------------------------------------------------------------
# motors / counters / per-frame columns
# ---------------------------------------------------------------------------

def _parse_motors_yaml(entry: h5py.Group) -> list[str]:
    """Best-effort names from ``…/bluesky/metadata/motors`` WITHOUT yaml-loading.

    The value is an apstools ``!!python/tuple`` dump such as
    ``b'!!python/tuple\\n- hy\\n'``.  We only line-scan for ``- <name>`` entries
    (never ``yaml.load`` — that would execute the ``!!python/tuple`` tag).  Used
    only as a cross-check against the authoritative positioner groups.
    """
    ds = entry.get("instrument/bluesky/metadata/motors")
    if not isinstance(ds, h5py.Dataset):
        return []
    try:
        text = _to_str(_scalar(ds))
    except Exception:
        return []
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            name = line[2:].strip().strip("'\"")
            if name:
                names.append(name)
    return names


def bluesky_motor_names(entry: h5py.Group) -> list[str]:
    """Authoritative scan-motor names: the group children of
    ``entry/instrument/positioners`` (real NXpositioner groups).

    Falls back to the (safely) parsed ``…/metadata/motors`` list only if the
    positioners group is missing.  A stray non-group child is never a motor.
    """
    pos = entry.get("instrument/positioners")
    names: list[str] = []
    if isinstance(pos, h5py.Group):
        names = [name for name in pos if isinstance(pos.get(name), h5py.Group)]
    parsed = _parse_motors_yaml(entry)
    if parsed and set(parsed) - set(names):
        logger.debug("Bluesky metadata/motors %s not all present as positioners %s",
                     parsed, names)
    return names or parsed


def _data_group(entry: h5py.Group) -> h5py.Group | None:
    d = entry.get("data")
    return d if isinstance(d, h5py.Group) else None


def _read_1d_numeric(group: h5py.Group, name: str) -> np.ndarray | None:
    """Read ``group/name`` as a 1-D float array, or *None* if unsuitable.

    Skips non-numeric columns by ``dtype.kind`` WITHOUT reading them (Bluesky
    ``entry/data`` carries string/label columns that would raise on cast)."""
    ds = group.get(name)
    if not isinstance(ds, h5py.Dataset):
        return None
    if getattr(ds.dtype, "kind", "O") not in "fiub":
        return None
    try:
        arr = np.asarray(ds, dtype=float)
    except (TypeError, ValueError, OSError):
        return None
    return arr if arr.ndim == 1 else None


def bluesky_angles(entry: h5py.Group,
                   motor_names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Per-frame scan-motor arrays: ``{motor: array}`` from ``entry/data/<motor>``."""
    names = motor_names if motor_names is not None else bluesky_motor_names(entry)
    data = _data_group(entry)
    out: dict[str, np.ndarray] = {}
    if data is None:
        return out
    for name in names:
        arr = _read_1d_numeric(data, name)
        if arr is not None:
            out[name] = arr
    return out


def bluesky_counters(entry: h5py.Group,
                     counter_names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Per-frame counter arrays (ion chambers + photodiode) from ``entry/data``.

    Defaults to :data:`_DEFAULT_BLUESKY_COUNTERS` (``i0/i1/i2/pd``); only the
    ones actually present as 1-D numeric columns are returned.
    """
    names = counter_names if counter_names is not None else list(_DEFAULT_BLUESKY_COUNTERS)
    data = _data_group(entry)
    out: dict[str, np.ndarray] = {}
    if data is None:
        return out
    for name in names:
        arr = _read_1d_numeric(data, name)
        if arr is not None:
            out[name] = arr
    return out


def bluesky_per_frame_table(entry: h5py.Group) -> dict[str, np.ndarray]:
    """The plottable per-frame metadata table: motors + counters + ``EPOCH``.

    This is the ``scan_data``-equivalent surfaced to Plot Metadata and
    :func:`xrd_tools.io.read.get_metadata`.
    """
    table: dict[str, np.ndarray] = {}
    table.update(bluesky_angles(entry))
    table.update(bluesky_counters(entry))
    data = _data_group(entry)
    if data is not None:
        epoch = _read_1d_numeric(data, "EPOCH")
        if epoch is not None:
            table["EPOCH"] = epoch
    return table


# ---------------------------------------------------------------------------
# fixed (non-scanned) motor values — positioners + baseline
# ---------------------------------------------------------------------------
#
# A GI incidence motor is usually held FIXED while something else (or nothing)
# is scanned, so its angle is NOT an ``entry/data/<motor>`` per-frame column.
# apstools records a fixed device's value in ``positioners/<motor>/value`` and/or
# the baseline stream (``…/bluesky/streams/baseline``, read once at scan
# start/end).  These harvesters expose those constants so the chosen incidence
# motor resolves even when it was never scanned (broadcast across all frames).

def _first_numeric(ds: h5py.Dataset) -> float | None:
    """First element of a dataset as a float (a fixed motor's constant value)."""
    try:
        arr = np.asarray(ds, dtype=float).ravel()
    except (TypeError, ValueError, OSError):
        return None
    return float(arr[0]) if arr.size else None


def bluesky_positioner_values(entry: h5py.Group) -> dict[str, float]:
    """Constant value of each positioner from ``positioners/<motor>/value``.

    Returns ``{motor: first_value}`` for every ``NXpositioner`` group child of
    ``entry/instrument/positioners``.  For a scanned motor this is just the
    first frame's value; the fixed-motor use (:func:`bluesky_fixed_motor_values`)
    excludes the scanned columns so only genuinely-fixed motors are broadcast.
    """
    pos = entry.get("instrument/positioners")
    out: dict[str, float] = {}
    if not isinstance(pos, h5py.Group):
        return out
    for name in pos:
        grp = pos.get(name)
        if not isinstance(grp, h5py.Group):
            continue
        ds = grp.get("value")
        if isinstance(ds, h5py.Dataset):
            val = _first_numeric(ds)
            if val is not None:
                out[name] = val
    return out


def _baseline_group(entry: h5py.Group) -> h5py.Group | None:
    base = entry.get("instrument/bluesky/streams/baseline")
    return base if isinstance(base, h5py.Group) else None


def bluesky_baseline_values(entry: h5py.Group) -> dict[str, float]:
    """RAW baseline-stream constants: ``{signal: value_start}`` for every signal
    in ``entry/instrument/bluesky/streams/baseline`` (``value_start`` preferred,
    else ``value_end``/``value``).

    apstools writes each baseline signal as a subgroup with ``value_start`` +
    ``value_end`` scalars (the device read once at scan start/end).  This is the
    FULL set — it includes the ``EpicsMotor`` field spray (``<m>_hlm`` /
    ``<m>_llm`` / ``<m>_dial`` / …); use :func:`bluesky_fixed_motor_values` for
    the motor-filtered incidence values (only authoritative positioner names).
    """
    base = _baseline_group(entry)
    out: dict[str, float] = {}
    if base is None:
        return out
    for name in base:
        obj = base.get(name)
        if isinstance(obj, h5py.Group):
            for field in ("value_start", "value_end", "value"):
                ds = obj.get(field)
                if isinstance(ds, h5py.Dataset):
                    val = _first_numeric(ds)
                    if val is not None:
                        out[name] = val
                        break
        elif isinstance(obj, h5py.Dataset):
            val = _first_numeric(obj)
            if val is not None:
                out[name] = val
    return out


def _baseline_motor_value(entry: h5py.Group, motor: str) -> float | None:
    """``value_start`` of a motor's USER-POSITION baseline signal, or *None*.

    Tries the authoritative motor name then the readback aliases — the motor
    VALUE is the bare ``<m>`` (or ``<m>_user_readback`` / ``<m>_rbv``), NEVER the
    limit/dial/alarm fields.  ``value_start`` (the position at scan start) is the
    acquisition condition; a materially different ``value_end`` is logged at
    debug but ``value_start`` is still returned.
    """
    base = _baseline_group(entry)
    if base is None:
        return None
    for signal in (motor, f"{motor}_user_readback", f"{motor}_rbv",
                   f"{motor}_readback"):
        grp = base.get(signal)
        if isinstance(grp, h5py.Dataset):          # flat, no start/end split
            return _first_numeric(grp)
        if not isinstance(grp, h5py.Group):
            continue
        start = None
        for field in ("value_start", "value_end", "value"):
            ds = grp.get(field)
            if isinstance(ds, h5py.Dataset):
                start = _first_numeric(ds)
                if start is not None:
                    break
        if start is None:
            continue
        end_ds = grp.get("value_end")
        end = _first_numeric(end_ds) if isinstance(end_ds, h5py.Dataset) else None
        if end is not None and abs(end - start) > 1e-9 * max(1.0, abs(start)):
            logger.debug("Bluesky baseline motor %r moved during scan "
                         "(value_start=%g, value_end=%g); using value_start",
                         motor, start, end)
        return start
    return None


def bluesky_fixed_motor_values(entry: h5py.Group,
                               exclude: Iterable[str] = ()) -> dict[str, float]:
    """Constant per-scan values for FIXED positioner motors (not scanned per
    frame).

    For each motor in :func:`bluesky_motor_names` (authoritative
    ``entry/instrument/positioners/*``) NOT in *exclude* (the scanned
    ``entry/data`` columns), resolves its constant angle from — in order — the
    baseline user-position ``value_start`` (:func:`_baseline_motor_value`), then
    ``positioners/<m>/value``.  A FIXED GI incidence motor (e.g. ``halpha`` held
    constant) lands here so its value broadcasts across every frame for incidence
    resolution; the ``EpicsMotor`` limit/dial/alarm field spray is never
    surfaced because only positioner names are considered.
    """
    skip = {str(x) for x in exclude}
    pos_vals = bluesky_positioner_values(entry)
    out: dict[str, float] = {}
    for motor in bluesky_motor_names(entry):
        if motor in skip:
            continue
        val = _baseline_motor_value(entry, motor)
        if val is None:
            val = pos_vals.get(motor)
        if val is not None:
            out[motor] = val
    return out


# ---------------------------------------------------------------------------
# wavelength / energy (eiger detector config)
# ---------------------------------------------------------------------------

def _eiger_config_field(entry: h5py.Group, field: str) -> h5py.Dataset | None:
    """Find ``…/configuration/<det>/data/<field>`` — preferring the ``eiger``
    device, else any configured device that carries the field."""
    cfg = entry.get(_EIGER_CONFIG_BASE)
    if not isinstance(cfg, h5py.Group):
        return None
    order = ["eiger"] + [k for k in cfg if k != "eiger"]
    for dev in order:
        ds = cfg.get(f"{dev}/data/{field}")
        if isinstance(ds, h5py.Dataset):
            return ds
    return None


def bluesky_wavelength(entry: h5py.Group) -> float:
    """Wavelength in Å from ``…/configuration/eiger/data/eiger_cam_wavelength``."""
    ds = _eiger_config_field(entry, "eiger_cam_wavelength")
    if ds is not None:
        try:
            return float(_scalar(ds))
        except Exception:
            logger.warning("Could not read Bluesky eiger_cam_wavelength", exc_info=True)
    return float(np.nan)


def bluesky_energy_kev(entry: h5py.Group) -> float:
    """Beam energy in keV from ``…/eiger_cam_photon_energy`` (stored in eV)."""
    ds = _eiger_config_field(entry, "eiger_cam_photon_energy")
    if ds is not None:
        try:
            ev = float(_scalar(ds))
            if np.isfinite(ev) and ev > 0:
                return ev / 1000.0
        except Exception:
            logger.warning("Could not read Bluesky eiger_cam_photon_energy", exc_info=True)
    return float(np.nan)


# ---------------------------------------------------------------------------
# scalar provenance
# ---------------------------------------------------------------------------

def bluesky_scalar_metadata(entry: h5py.Group) -> dict[str, Any]:
    """Scalar provenance for the metadata table (title / plan / times / count)."""
    out: dict[str, Any] = {}
    for key in ("title", "plan_name", "program_name", "start_time",
                "end_time", "duration", "entry_identifier"):
        ds = entry.get(key)
        if isinstance(ds, h5py.Dataset):
            try:
                val = _scalar(ds)
                out[key] = _to_str(val) if isinstance(val, (bytes, np.bytes_)) else val
            except Exception:
                continue
    npts = entry.get("instrument/bluesky/metadata/num_points")
    if isinstance(npts, h5py.Dataset):
        try:
            out["num_points"] = int(_scalar(npts))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# detector image resolver (shared by image.py + nexus.py)
# ---------------------------------------------------------------------------

def find_detector_signal_dataset(group: h5py.File | h5py.Group,
                                 ) -> h5py.Dataset | None:
    """Return the image dataset marked ``@signal_type == 'detector'``.

    Bluesky points an NXdata ``@signal`` at a hinted *scalar counter*, so the
    detector image is instead flagged by a ``signal_type='detector'`` attribute
    on the pixel dataset(s).  Searches ``group`` recursively and returns the
    largest such dataset with ``ndim >= 2`` (the pixel stack wins over any
    small detector-tagged stat), or *None*.
    """
    best: h5py.Dataset | None = None
    best_size = -1

    def _visit(_name: str, obj: Any) -> None:
        nonlocal best, best_size
        if not isinstance(obj, h5py.Dataset) or obj.ndim < 2:
            return
        if _to_str(obj.attrs.get("signal_type", "")) != "detector":
            return
        size = int(obj.size)
        if size > best_size:
            best = obj  # type: ignore[assignment]
            best_size = size

    try:
        group.visititems(_visit)
    except Exception:
        logger.debug("find_detector_signal_dataset: traversal error", exc_info=True)
    return best
