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
    "is_unfinalized_nxwriter",
    "resolve_nxentry",
    "bluesky_motor_names",
    "bluesky_all_motor_names",
    "bluesky_angles",
    "bluesky_counters",
    "bluesky_positioner_values",
    "bluesky_baseline_values",
    "bluesky_fixed_motor_values",
    "bluesky_eiger_count_time",
    "bluesky_constant_metadata",
    "bluesky_wavelength",
    "bluesky_energy_kev",
    "bluesky_per_frame_table",
    "bluesky_scalar_metadata",
    "find_detector_signal_dataset",
    "validate_average_container_metadata_inputs",
]

# The entry attribute xdart stamps on its own processed files.  A Bluesky file
# never carries it; its presence positively excludes the Bluesky branch.  Kept
# as a literal (not imported from ``io.schema``) to keep this module import-light
# — the key is frozen (``schema.SCHEMA_NAME_ATTR``).
_XDART_SCHEMA_ATTR = "ssrl_schema"

# Default norm-channel counters (ion chambers + photodiode).  ``gate`` and
# ``eiger`` from ``…/metadata/detectors`` are excluded (a timer and the image).
_DEFAULT_BLUESKY_COUNTERS = ("i0", "i1", "i2", "pd")

# Per-frame counting time (gate/timer channel in ``entry/data``).
_BLUESKY_COUNT_TIME_COL = "gate_actual_counting_time"

# An ``EpicsMotor`` baseline signal ``<m>`` sprays companion sub-signals
# (``<m>_user_setpoint``, ``<m>_rbv``, dial/limit fields, …).  The presence of
# ANY of these setpoint/readback companions is what positively marks a bare
# baseline name as a MOTOR — cleanly excluding scaler counters (``i0``), the
# detector (``eiger``), the timer (``gate``) and the field-spray sub-signals
# themselves (``detx_hlm`` has no ``detx_hlm_user_setpoint``).  Both EpicsMotor
# variants at bl11-3 are covered: ``*_user_setpoint``/``*_rbv`` and the
# ``*_readback``/``*_dial_readback`` class.
_MOTOR_COMPANION_SUFFIXES = (
    "_user_setpoint", "_rbv", "_user_readback", "_dial_readback", "_readback",
)

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
                    entry_hint: str = "entry", *,
                    exact_hint: bool = False) -> h5py.Group | None:
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
    if exact_hint:
        link = root.get(entry_hint, getlink=True)
        if not isinstance(link, h5py.HardLink):
            return None
        value = root.get(entry_hint)
        return value if isinstance(value, h5py.Group) else None
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


def is_unfinalized_nxwriter(path) -> bool:
    """True when *path* is a Bluesky/apstools ``NXWriter`` container that is
    still being written — i.e. NOT yet safe to consume-and-retire.

    The NXWriter lifecycle stamps ``entry/end_time`` when the run closes, so a
    Bluesky container WITHOUT it is in-progress: a live directory watch must
    DEFER it (re-check on the next poll) rather than read it, or the reader
    exhausts the partial file and permanently retires it while frames are still
    arriving (v1.1.1 F5).  Also reported unfinalized:

    * an UNREADABLE file (h5py cannot open a half-written HDF5 — the same
      mid-write state, observed earlier), and
    * an NXWriter file whose entry has not been created yet (root
      ``creator='NXWriter'`` is stamped before the tree is populated).

    A non-Bluesky container is NEVER deferred — it has no ``end_time``
    contract, and deferring it would silently exclude plain ``.nxs``/``.h5``
    stacks from directory watches forever.
    """
    try:
        with h5py.File(path, "r") as f:
            if not is_bluesky_nxwriter(f):
                return False
            entry = resolve_nxentry(f)
            return entry is None or "end_time" not in entry
    except Exception:
        logger.debug("is_unfinalized_nxwriter: %s unreadable -> defer",
                     path, exc_info=True)
        return True


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


def bluesky_motor_names(
    entry: h5py.Group, *, skip_advisory_if_authoritative: bool = False,
) -> list[str]:
    """Authoritative scan-motor names: the group children of
    ``entry/instrument/positioners`` (real NXpositioner groups).

    Falls back to the (safely) parsed ``…/metadata/motors`` list only if the
    positioners group is missing.  A stray non-group child is never a motor.
    """
    pos = entry.get("instrument/positioners")
    names: list[str] = []
    if isinstance(pos, h5py.Group):
        names = [name for name in pos if isinstance(pos.get(name), h5py.Group)]
    parsed = ([] if names and skip_advisory_if_authoritative
              else _parse_motors_yaml(entry))
    if parsed and set(parsed) - set(names):
        logger.debug("Bluesky metadata/motors %s not all present as positioners %s",
                     parsed, names)
    return names or parsed


def bluesky_all_motor_names(
    entry: h5py.Group, scanned_motor_names: Iterable[str] | None = None,
) -> list[str]:
    """EVERY motor present in the scan — scanned AND held-fixed.

    :func:`bluesky_motor_names` returns only the SCANNED motors (the
    ``positioners`` groups), which is right for the GI incidence dropdown and
    per-frame angle columns.  But a Bluesky scan also records every FIXED motor's
    position once, in the baseline stream — and those are real motors the user
    wants to see in the metadata table (``detx``, ``dety``, ``hx``/``hy``/``hz``,
    …).  This returns the union: the scanned positioners first (stable order),
    then the baseline motors, identified by their ``EpicsMotor`` setpoint/readback
    companion sub-signals (:data:`_MOTOR_COMPANION_SUFFIXES`) so scaler counters,
    the detector, the timer and the limit/dial field-spray are all excluded.
    """
    names = list(bluesky_motor_names(entry) if scanned_motor_names is None
                 else scanned_motor_names)
    seen = set(names)
    base = _baseline_group(entry)
    if isinstance(base, h5py.Group):
        signals = set(base.keys())
        for name in sorted(signals):
            if name in seen:
                continue
            if any(f"{name}{suffix}" in signals
                   for suffix in _MOTOR_COMPANION_SUFFIXES):
                names.append(name)
                seen.add(name)
    return names


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
    """The plottable per-frame metadata table: motors + counters + counting time
    + ``EPOCH``.

    This is the ``scan_data``-equivalent surfaced to Plot Metadata and
    :func:`xrd_tools.io.read.get_metadata`.  ``gate_actual_counting_time`` (the
    per-frame gate/timer dwell in seconds) is included when present.
    """
    table: dict[str, np.ndarray] = {}
    table.update(bluesky_angles(entry))
    table.update(bluesky_counters(entry))
    data = _data_group(entry)
    if data is not None:
        for extra in (_BLUESKY_COUNT_TIME_COL, "EPOCH"):
            arr = _read_1d_numeric(data, extra)
            if arr is not None:
                table[extra] = arr
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


def _bounded_first_numeric(ds: h5py.Dataset) -> float | None:
    """Read exactly one already-admitted fixed numeric element."""
    try:
        value = ds[()] if ds.shape == () else ds[0]
        return float(value)
    except (TypeError, ValueError, OSError):
        return None


def bluesky_positioner_values(
    entry: h5py.Group, *, bounded: bool = False,
) -> dict[str, float]:
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
            val = (_bounded_first_numeric(ds) if bounded else _first_numeric(ds))
            if val is not None:
                out[name] = val
    return out


def _baseline_group(entry: h5py.Group) -> h5py.Group | None:
    base = entry.get("instrument/bluesky/streams/baseline")
    return base if isinstance(base, h5py.Group) else None


def bluesky_baseline_values(
    entry: h5py.Group, *, bounded: bool = False,
) -> dict[str, float]:
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
                    val = (_bounded_first_numeric(ds) if bounded else _first_numeric(ds))
                    if val is not None:
                        out[name] = val
                        break
        elif isinstance(obj, h5py.Dataset):
            val = (_bounded_first_numeric(obj) if bounded else _first_numeric(obj))
            if val is not None:
                out[name] = val
    return out


def _baseline_motor_value(
    entry: h5py.Group, motor: str, *, bounded: bool = False,
) -> float | None:
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
            return (_bounded_first_numeric(grp) if bounded else _first_numeric(grp))
        if not isinstance(grp, h5py.Group):
            continue
        start = None
        for field in ("value_start", "value_end", "value"):
            ds = grp.get(field)
            if isinstance(ds, h5py.Dataset):
                start = (_bounded_first_numeric(ds) if bounded else _first_numeric(ds))
                if start is not None:
                    break
        if start is None:
            continue
        end_ds = grp.get("value_end")
        end = ((_bounded_first_numeric(end_ds) if bounded else _first_numeric(end_ds))
               if isinstance(end_ds, h5py.Dataset) else None)
        if end is not None and abs(end - start) > 1e-9 * max(1.0, abs(start)):
            logger.debug("Bluesky baseline motor %r moved during scan "
                         "(value_start=%g, value_end=%g); using value_start",
                         motor, start, end)
        return start
    return None


def bluesky_fixed_motor_values(
    entry: h5py.Group, exclude: Iterable[str] = (), *,
    motor_names: Iterable[str] | None = None, bounded: bool = False,
) -> dict[str, float]:
    """Constant per-scan values for FIXED positioner motors (not scanned per
    frame).

    For each motor in :func:`bluesky_all_motor_names` (scanned positioners PLUS
    the baseline motors) NOT in *exclude* (the scanned ``entry/data`` columns),
    resolves its constant value from — in order — the baseline user-position
    ``value_start`` (:func:`_baseline_motor_value`), then ``positioners/<m>/value``.
    A FIXED GI incidence motor (e.g. ``halpha`` held constant) lands here so its
    value broadcasts across every frame for incidence resolution, AND so do the
    other held-fixed motors (``detx``/``dety``/``hx``/``hy``/``hz``/…) for the
    metadata table; the ``EpicsMotor`` limit/dial/alarm field spray is never
    surfaced because only motor names are considered.
    """
    skip = {str(x) for x in exclude}
    pos_vals = bluesky_positioner_values(entry, bounded=bounded)
    out: dict[str, float] = {}
    for motor in (bluesky_all_motor_names(entry) if motor_names is None
                  else motor_names):
        if motor in skip:
            continue
        val = _baseline_motor_value(entry, motor, bounded=bounded)
        if val is None:
            val = pos_vals.get(motor)
        if val is not None:
            out[motor] = val
    return out


def bluesky_eiger_count_time(entry: h5py.Group) -> float:
    """Eiger detector counting time in seconds, or NaN when no eiger is present.

    Read from the detector configuration — ``eiger_cam_acquire_time`` (the
    exposure), falling back to ``eiger_cam_acquire_period``.  Returned separately
    from the gate/timer dwell (:data:`_BLUESKY_COUNT_TIME_COL`) so both the
    detector exposure and the per-frame gate time are available.
    """
    for field in ("eiger_cam_acquire_time", "eiger_cam_acquire_period"):
        ds = _eiger_config_field(entry, field)
        if ds is not None:
            try:
                v = float(_scalar(ds))
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                return v
    return float(np.nan)


def bluesky_constant_metadata(
    entry: h5py.Group, exclude: Iterable[str] = (), *,
    motor_names: Iterable[str] | None = None, bounded: bool = False,
) -> dict[str, float]:
    """Per-scan CONSTANT metadata columns to broadcast across every frame.

    The fixed (non-scanned) motor positions (:func:`bluesky_fixed_motor_values`)
    plus the eiger detector counting time (``eiger_count_time``) when an eiger is
    present.  This is the DISPLAY overlay for the per-frame table — it keeps the
    detector counting time out of the motor-only paths (the GI dropdown / angle
    columns) while still surfacing it in Plot Metadata and the frame-metadata
    table.  Names already in *exclude* (e.g. per-frame ``entry/data`` columns)
    are skipped so a scanned column always wins.
    """
    skip = {str(x) for x in exclude}
    out: dict[str, float] = dict(bluesky_fixed_motor_values(
        entry, exclude=skip, motor_names=motor_names, bounded=bounded,
    ))
    if "eiger_count_time" not in skip:
        ct = bluesky_eiger_count_time(entry)
        if np.isfinite(ct):
            out["eiger_count_time"] = float(ct)
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
# Average-only bounded metadata admission
# ---------------------------------------------------------------------------

_AVERAGE_METADATA_CHUNK_CAP = 1 << 20
_AVERAGE_METADATA_NAME_BYTES = 1 << 20
_AVERAGE_METADATA_NAME_COUNT = {
    "positioners": 256, "baseline": 4096, "configuration": 256, "data": 4096,
}


def _average_hard_path(entry: h5py.Group, path: str) -> Any:
    current: Any = entry
    for component in path.split("/"):
        if not isinstance(current, h5py.Group):
            return None
        link = current.get(component, getlink=True)
        if link is None:
            return None
        if not isinstance(link, h5py.HardLink):
            raise ValueError("AVERAGE_METADATA_INDIRECTION_UNSUPPORTED")
        current = current.get(component)
    return current


def _average_catalog(
    entry: h5py.Group, path: str, role: str, aggregate: list[int],
) -> h5py.Group | None:
    value = _average_hard_path(entry, path)
    if value is None:
        return None
    if not isinstance(value, h5py.Group):
        raise ValueError("AVERAGE_METADATA_INPUT_UNBOUNDED")
    names = tuple(value.keys())
    if len(names) > _AVERAGE_METADATA_NAME_COUNT[role]:
        raise ValueError("AVERAGE_METADATA_CATALOG_TOO_LARGE")
    for name in names:
        encoded = str(name).encode("utf-8")
        if len(encoded) > 256:
            raise ValueError("AVERAGE_METADATA_CATALOG_TOO_LARGE")
        aggregate[0] += len(encoded)
        if aggregate[0] > _AVERAGE_METADATA_NAME_BYTES:
            raise ValueError("AVERAGE_METADATA_CATALOG_TOO_LARGE")
    return value


def _average_fixed_numeric_dataset(
    value: Any, *, scalar_only: bool = False,
) -> h5py.Dataset:
    if not isinstance(value, h5py.Dataset):
        raise ValueError("AVERAGE_METADATA_INPUT_UNBOUNDED")
    dtype = value.dtype
    metadata = getattr(dtype, "metadata", None)
    has_vlen = isinstance(metadata, dict) and metadata.get("vlen") is not None
    itemsize = getattr(dtype, "itemsize", None)
    if (getattr(dtype, "fields", None) is not None
            or getattr(dtype, "subdtype", None) is not None
            or has_vlen
            or type(getattr(dtype, "kind", None)) is not str
            or getattr(dtype, "kind", None) not in {"f", "i", "u", "b"}
            or type(itemsize) is not int or not 1 <= itemsize <= 8):
        raise ValueError("AVERAGE_METADATA_INPUT_UNBOUNDED")
    if bool(value.is_virtual):
        raise ValueError("AVERAGE_METADATA_VIRTUAL_UNSUPPORTED")
    if value.external:
        raise ValueError("AVERAGE_METADATA_INDIRECTION_UNSUPPORTED")
    if value.chunks is not None:
        logical = itemsize
        for extent in value.chunks:
            logical *= int(extent)
        if logical > _AVERAGE_METADATA_CHUNK_CAP:
            raise ValueError("AVERAGE_METADATA_CHUNK_TOO_LARGE")
    shape = tuple(int(item) for item in value.shape)
    if scalar_only:
        valid_shape = shape == () or shape == (1,)
    else:
        valid_shape = len(shape) == 1 and shape[0] >= 1
    if not valid_shape:
        raise ValueError("AVERAGE_METADATA_INPUT_UNBOUNDED")
    return value


def _average_direct_name(value: str) -> str:
    if (type(value) is not str or not value or value in {".", ".."}
            or "/" in value or "\0" in value):
        raise ValueError("AVERAGE_METADATA_MOTOR_NAME_INVALID")
    if len(value.encode("utf-8")) > 256:
        raise ValueError("AVERAGE_METADATA_CATALOG_TOO_LARGE")
    return value


def _average_manifest_motor_names(
    manifest: h5py.Dataset | None,
) -> tuple[str, ...]:
    """Strict direct names from one already-bounded Average manifest."""
    if manifest is None:
        return ()
    try:
        text = _to_str(_scalar(manifest))
    except Exception:
        return ()
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            names.append(
                _average_direct_name(line[2:].strip().strip("'\""))
            )
    return tuple(names)


def _average_default_counter_names(entry: h5py.Group) -> tuple[str, ...]:
    """Select canonical counters case-insensitively, preserving source names."""
    data = _average_hard_path(entry, "data")
    if data is None:
        return ()
    if not isinstance(data, h5py.Group):
        raise ValueError("AVERAGE_METADATA_INPUT_UNBOUNDED")
    observed = tuple(str(name) for name in data.keys())
    selected: list[str] = []
    for canonical in _DEFAULT_BLUESKY_COUNTERS:
        matches = tuple(name for name in observed if name.casefold() == canonical)
        if len(matches) > 1:
            raise ValueError("AVERAGE_METADATA_COUNTER_AMBIGUOUS")
        selected.extend(matches)
    return tuple(selected)


def validate_average_container_metadata_inputs(
    entry: h5py.Group, *, policy: str = "average_bounded_v1",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Admit every metadata route before an Average payload read.

    The returned pair is ``(scanned_motor_names, all_motor_names)``.  It is
    retained by the graph and passed unchanged to the descriptor and streaming
    provider, eliminating per-row YAML/catalog work.
    """
    if policy != "average_bounded_v1" or not isinstance(entry, h5py.Group):
        raise ValueError("unsupported Average metadata policy")
    aggregate = [0]
    positioners = _average_catalog(
        entry, "instrument/positioners", "positioners", aggregate,
    )
    baseline = _average_catalog(
        entry, "instrument/bluesky/streams/baseline", "baseline", aggregate,
    )
    configuration = _average_catalog(
        entry, _EIGER_CONFIG_BASE, "configuration", aggregate,
    )
    data = _average_catalog(entry, "data", "data", aggregate)

    authoritative_scanned: list[str] = []
    if positioners is not None:
        for name in positioners.keys():
            direct_name = _average_direct_name(str(name))
            group = _average_hard_path(
                entry, f"instrument/positioners/{direct_name}",
            )
            if not isinstance(group, h5py.Group):
                continue
            authoritative_scanned.append(direct_name)
            value = _average_hard_path(
                entry, f"instrument/positioners/{direct_name}/value",
            )
            if value is not None:
                _average_fixed_numeric_dataset(value)
    manifest = None
    if not authoritative_scanned:
        observed_manifest = _average_hard_path(
            entry, "instrument/bluesky/metadata/motors",
        )
        if observed_manifest is not None:
            if not isinstance(observed_manifest, h5py.Dataset):
                raise ValueError("AVERAGE_METADATA_INPUT_UNBOUNDED")
            dtype = observed_manifest.dtype
            if bool(observed_manifest.is_virtual):
                raise ValueError("AVERAGE_METADATA_VIRTUAL_UNSUPPORTED")
            if observed_manifest.external:
                raise ValueError("AVERAGE_METADATA_INDIRECTION_UNSUPPORTED")
            if observed_manifest.chunks is not None and (
                    int(getattr(dtype, "itemsize", 0))
                    * int(np.prod(observed_manifest.chunks))) > _AVERAGE_METADATA_CHUNK_CAP:
                raise ValueError("AVERAGE_METADATA_CHUNK_TOO_LARGE")
            if (observed_manifest.shape != ()
                    or getattr(dtype, "kind", None) not in {"S", "U"}
                    or int(getattr(dtype, "itemsize", 0)) > (1 << 16)):
                raise ValueError("AVERAGE_METADATA_INPUT_UNBOUNDED")
            manifest = observed_manifest
    scanned_tuple = (tuple(authoritative_scanned) if authoritative_scanned
                     else _average_manifest_motor_names(manifest))
    if len(scanned_tuple) > 256:
        raise ValueError("AVERAGE_METADATA_CATALOG_TOO_LARGE")

    if baseline is not None:
        for name in baseline.keys():
            direct_name = _average_direct_name(str(name))
            obj = _average_hard_path(
                entry,
                f"instrument/bluesky/streams/baseline/{direct_name}",
            )
            if isinstance(obj, h5py.Dataset):
                _average_fixed_numeric_dataset(obj)
            elif isinstance(obj, h5py.Group):
                for field in ("value_start", "value_end", "value"):
                    value = _average_hard_path(
                        entry,
                        f"instrument/bluesky/streams/baseline/{direct_name}/{field}",
                    )
                    if value is not None:
                        _average_fixed_numeric_dataset(value, scalar_only=True)

    all_names = tuple(
        _average_direct_name(str(name))
        for name in bluesky_all_motor_names(
            entry, scanned_motor_names=scanned_tuple,
        )
    )
    if len(all_names) > 256:
        raise ValueError("AVERAGE_METADATA_CATALOG_TOO_LARGE")

    if data is not None:
        relevant = set(scanned_tuple) | set(_average_default_counter_names(entry)) | {
            _BLUESKY_COUNT_TIME_COL, "EPOCH",
        }
        for name in relevant:
            value = _average_hard_path(entry, f"data/{name}")
            if value is not None:
                _average_fixed_numeric_dataset(value)

    if configuration is not None:
        for device in configuration.keys():
            _average_direct_name(str(device))
            device_data = _average_hard_path(
                entry, f"{_EIGER_CONFIG_BASE}/{device}/data",
            )
            if device_data is None:
                continue
            if not isinstance(device_data, h5py.Group):
                raise ValueError("AVERAGE_METADATA_INPUT_UNBOUNDED")
            for field in (
                "eiger_cam_wavelength", "eiger_cam_photon_energy",
                "eiger_cam_acquire_time", "eiger_cam_acquire_period",
            ):
                value = _average_hard_path(
                    entry, f"{_EIGER_CONFIG_BASE}/{device}/data/{field}",
                )
                if value is not None:
                    _average_fixed_numeric_dataset(value, scalar_only=True)

    for field in ("energy", "wavelength"):
        value = _average_hard_path(
            entry, f"instrument/monochromator/{field}",
        )
        if value is not None:
            _average_fixed_numeric_dataset(value, scalar_only=True)
    return scanned_tuple, tuple(all_names)


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
    from xrd_tools.io.processed_scan_id import (
        ProcessedXdartInputError,
        require_raw_input,
    )

    require_raw_input(group)
    best: h5py.Dataset | None = None
    best_size = -1

    def _visit(_name: str, obj: Any) -> None:
        nonlocal best, best_size
        if not isinstance(obj, h5py.Dataset):
            return
        require_raw_input(obj)
        if obj.ndim < 2:
            return
        if _to_str(obj.attrs.get("signal_type", "")) != "detector":
            return
        size = int(obj.size)
        if size > best_size:
            best = obj  # type: ignore[assignment]
            best_size = size

    try:
        group.visititems(_visit)
    except ProcessedXdartInputError:
        raise
    except Exception:
        logger.debug("find_detector_signal_dataset: traversal error", exc_info=True)
    return best
