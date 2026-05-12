"""NXprocess provenance capture for the v2 NeXus schema.

Populates ``/entry/reduction/`` per the spec in
``xdart/docs/nexus_stitch_refactor_plan.md`` §2.1.  The layout written
here is:

::

    reduction/                                  (NXprocess)
        @NX_class = "NXprocess"
        program   = "xdart"
        version   = <xdart version>
        date      = ISO 8601 timestamp (UTC)
        host      = (str, optional)
        versions/                               (NXcollection)
            xdart          (str)
            ssrl_xrd_tools (str)
            pyFAI          (str)
            h5py           (str)
            numpy          (str)
            pymatgen       (str)
            python         (str)
        config/                                 (NXcollection)
            <name> = JSON-encoded string per supplied config dict
            ...
        inputs/                                 (NXcollection)
            raw_files (N,)  str (h5py vlen str array)
            meta_file       str

Versions are pulled via :mod:`importlib.metadata` — **never hard-coded**.
This is the reason both ``xdart`` and ``ssrl_xrd_tools`` use the
``__version__ = importlib.metadata.version(...)`` pattern in their
package ``__init__``.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import socket
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:  # pragma: no cover
    import h5py

# Canonical list of packages whose versions we record on every write.
# Order is preserved in the written `versions/` group for readability.
CANONICAL_PACKAGES: tuple[str, ...] = (
    "xdart",
    "ssrl_xrd_tools",
    "pyFAI",
    "h5py",
    "numpy",
    "pymatgen",
)


def capture_versions(extra_packages: Iterable[str] = ()) -> dict[str, str]:
    """Return a dict mapping package name → installed version string.

    Adds the Python interpreter version under the ``"python"`` key
    (formatted as ``"3.12.4"`` — major.minor.patch only, no compiler /
    build suffix).  Packages that aren't importable in the current env
    are recorded as ``""`` rather than omitted, so the schema is stable
    across deployments.

    Parameters
    ----------
    extra_packages
        Optional iterable of additional distribution names to record
        beyond :data:`CANONICAL_PACKAGES`.
    """
    versions: dict[str, str] = {}
    seen: set[str] = set()
    for pkg in tuple(CANONICAL_PACKAGES) + tuple(extra_packages):
        if pkg in seen:
            continue
        seen.add(pkg)
        try:
            versions[pkg] = _pkg_version(pkg)
        except PackageNotFoundError:
            versions[pkg] = ""
    versions["python"] = "{0.major}.{0.minor}.{0.micro}".format(sys.version_info)
    return versions


def _iso_now_utc() -> str:
    """ISO 8601 timestamp in UTC, second-precision, with ``Z`` suffix."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _ensure_group(h5f: "h5py.File", path: str, nx_class: str | None = None):
    """Require a group at an absolute path, optionally setting NX_class."""
    grp = h5f.require_group(path)
    if nx_class is not None:
        existing = grp.attrs.get("NX_class", None)
        if isinstance(existing, bytes):
            existing = existing.decode("utf-8")
        if existing != nx_class:
            grp.attrs["NX_class"] = nx_class
    return grp


def _write_scalar(grp, name: str, value: Any) -> None:
    """Store a scalar (str/int/float/bool) at ``grp[name]``, replacing prior."""
    if name in grp:
        del grp[name]
    grp[name] = value


def _write_json(grp, name: str, payload: Any) -> None:
    """Store ``payload`` as a compact JSON string at ``grp[name]``."""
    text = json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str)
    _write_scalar(grp, name, text)


def write_provenance(
    h5f: "h5py.File",
    *,
    entry: str = "entry",
    program: str = "xdart",
    program_version: str | None = None,
    config: Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    host: str | None = None,
    extra: Mapping[str, Any] | None = None,
    extra_packages: Iterable[str] = (),
    date: str | None = None,
) -> None:
    """Write the full ``/{entry}/reduction/`` NXprocess group.

    Parameters
    ----------
    h5f
        Open writable :class:`h5py.File`.
    entry
        Name of the NXentry group (default ``"entry"``).
    program
        Value of ``reduction/program``.  Default ``"xdart"``.
    program_version
        Value of ``reduction/version``.  If ``None``, looked up via
        :func:`importlib.metadata.version` for ``program``.
    config
        Optional mapping ``name -> JSON-serializable value`` written
        under ``reduction/config/<name>`` as compact JSON strings.
        Typical names: ``bai_1d_args``, ``bai_2d_args``, ``mg_1d_args``,
        ``mg_2d_args``, ``gi_config``, ``geometry``.
    inputs
        Optional mapping written under ``reduction/inputs/<name>``.
        Lists of strings (e.g. ``"raw_files"``) are written as
        variable-length string arrays; scalars are written as scalars.
    host
        Hostname recorded in ``reduction/host``.  If ``None``, defaults
        to :func:`socket.gethostname()`.  Pass ``""`` to suppress.
    extra
        Additional flat key-value pairs to write directly into
        ``reduction/`` (e.g. ``cli_args``, ``gui_state``).  Non-string
        values are JSON-encoded.
    extra_packages
        Additional distribution names to record in ``versions/``
        beyond :data:`CANONICAL_PACKAGES`.
    date
        Override the ISO 8601 timestamp written to ``reduction/date``.
        Useful for reproducible testing; default is the current UTC
        time.
    """
    import h5py  # local import — h5py is heavy

    entry_grp = _ensure_group(h5f, entry, "NXentry")
    red = _ensure_group(h5f, f"{entry}/reduction", "NXprocess")

    # program / version / date / host -----------------------------------
    if program_version is None:
        try:
            program_version = _pkg_version(program)
        except PackageNotFoundError:
            program_version = ""
    _write_scalar(red, "program", program)
    _write_scalar(red, "version", program_version)
    _write_scalar(red, "date", date if date is not None else _iso_now_utc())
    if host is None:
        host = socket.gethostname()
    if host:
        _write_scalar(red, "host", host)
    # Optional descriptive entry not in the strict schema but useful:
    _write_scalar(red, "platform", platform.platform())

    # versions/ ---------------------------------------------------------
    versions_grp = _ensure_group(h5f, f"{entry}/reduction/versions", "NXcollection")
    for pkg, ver in capture_versions(extra_packages).items():
        _write_scalar(versions_grp, pkg, ver)

    # config/ -----------------------------------------------------------
    if config:
        cfg_grp = _ensure_group(h5f, f"{entry}/reduction/config", "NXcollection")
        for name, value in config.items():
            # Geometry config has its own subgroup with structured fields,
            # not a single JSON blob — caller can pass it as a sub-mapping.
            if isinstance(value, Mapping) and name == "geometry":
                geom_grp = _ensure_group(
                    h5f, f"{entry}/reduction/config/geometry", "NXcollection"
                )
                for k, v in value.items():
                    if isinstance(v, str):
                        _write_scalar(geom_grp, k, v)
                    else:
                        _write_json(geom_grp, k, v)
            else:
                _write_json(cfg_grp, name, value)

    # inputs/ -----------------------------------------------------------
    if inputs:
        in_grp = _ensure_group(h5f, f"{entry}/reduction/inputs", "NXcollection")
        vlen_str = h5py.string_dtype(encoding="utf-8")
        for name, value in inputs.items():
            if name in in_grp:
                del in_grp[name]
            if isinstance(value, (list, tuple)) and all(
                isinstance(v, str) for v in value
            ):
                in_grp.create_dataset(name, data=list(value), dtype=vlen_str)
            else:
                in_grp[name] = value

    # extra flat scalars -----------------------------------------------
    if extra:
        for k, v in extra.items():
            if isinstance(v, str):
                _write_scalar(red, k, v)
            else:
                _write_json(red, k, v)


def _decode(v: Any) -> Any:
    """h5py-decoded scalar: bytes → str, 0-d arrays → python scalars."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    try:
        # numpy scalar
        return v.item()
    except (AttributeError, ValueError):
        return v


def read_provenance(
    path: str,
    *,
    entry: str = "entry",
) -> dict[str, Any]:
    """Read ``/{entry}/reduction/`` back into a nested dict.

    Returns keys (subset depending on what was written):

    * ``program``, ``version``, ``date``, ``host``, ``platform`` — scalars
    * ``versions`` — dict[str, str] from ``versions/`` subgroup
    * ``config`` — dict where values are parsed JSON (or sub-dicts for
      structured groups like ``geometry``)
    * ``inputs`` — dict where list-valued items decode as
      ``list[str]``

    Anything else under ``reduction/`` lands in the top-level dict.
    Compact JSON strings written via :func:`write_provenance` are
    auto-parsed back to Python objects; raw strings that don't parse are
    returned as-is.
    """
    import h5py

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as h5f:
        if entry not in h5f or "reduction" not in h5f[entry]:
            return out
        red = h5f[f"{entry}/reduction"]

        for key, item in red.items():
            if isinstance(item, h5py.Dataset):
                v = _decode(item[()])
                if isinstance(v, str):
                    # Try JSON-decode; fall back to raw string
                    try:
                        out[key] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        out[key] = v
                else:
                    out[key] = v
            elif isinstance(item, h5py.Group):
                out[key] = _read_group(item)
    return out


def _read_group(grp) -> dict[str, Any]:
    """Recursive group reader used by :func:`read_provenance`."""
    import h5py

    d: dict[str, Any] = {}
    for k, v in grp.items():
        if isinstance(v, h5py.Group):
            d[k] = _read_group(v)
        else:
            raw = _decode(v[()])
            if isinstance(raw, str):
                try:
                    d[k] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d[k] = raw
            elif hasattr(raw, "tolist"):
                # Vlen string array → list[str]
                lst = raw.tolist()
                d[k] = [
                    b.decode("utf-8") if isinstance(b, bytes) else b for b in lst
                ]
            else:
                d[k] = raw
    return d


__all__ = [
    "CANONICAL_PACKAGES",
    "capture_versions",
    "read_provenance",
    "write_provenance",
]
