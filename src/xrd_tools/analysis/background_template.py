"""File-sourced background templates, one owner for the Peak Fitter and Phase Analysis (NEXT-128-PEAK-FIT).

The maintainer, 2026-10-30: "That would have also made sure that the background template is read from a file, I
think. This could be a Nexus file with 1d data or perhaps a xye file. The function itself should be agnostic to where
the BG comes from". So a template is named by one exportable source (``BackgroundTemplateSource``), read by one loader
(``prepare_background_template``), which the GUI's fit job and the exported notebooks both reach through
``run_displayed_peak_fit`` / ``run_displayed_phase_fit``, and fitted as arrays only
(``fitting.models.template_background_model``).

- A source is a frozen value kept as canonical primitive pairs with a version (a recipe setting, a settings section).
  Parsing is strict: an unknown key, kind or version, or a missing field, is refused by name.
- A NeXus source names its entry, frame and 1-D mode and is read through the mode-aware frame reader; a missing entry,
  frame or mode is refused, never replaced by the primary. An .xye source names its x unit; a filename prefix is only a
  hint (``xye_unit_hint``).
- The loader admits one finite paired curve (at least two distinct x, at most 1,000,000 points; an .xye over 64 MiB is
  refused before it is parsed) and returns it sorted, in the source's own unit, with a content identity: SHA-256 of the
  descriptor without the path, the unit, x and y, so moving the data folder keeps it. A recorded identity that no
  longer matches the file refuses the read (BACKGROUND_TEMPLATE_CHANGED). An .xye σ column is not propagated.
- ``on_axis`` places a template on a fit's axis through ``convert_radial_axis`` with no wavelength: Å⁻¹ and nm⁻¹ of one
  axis, and degrees and radians, convert; 2θ and q, or two GI axes, are refused naming both units.
- Reads are cached by source, path and the file's stat facts, so a batch or a re-fit reads its file once.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import KW_ONLY, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Mapping

import numpy as np

from xrd_tools.io.export import read_xye

__all__ = [
    "BackgroundTemplateRefused", "BackgroundTemplateSource", "PreparedBackgroundTemplate", "NEXUS_MODES",
    "XYE_UNITS", "align_template", "is_template_identity", "prepare_background_template", "template_file_problem",
    "xye_unit_hint",
]

SOURCE_VERSION = 1
MAX_POINTS = 1_000_000
MAX_XYE_BYTES = 64 * 1024 * 1024
#: an .xye source's x units, as the template row offers them: (unit, label)
XYE_UNITS = (("q_A^-1", "q (Å⁻¹)"), ("q_nm^-1", "q (nm⁻¹)"), ("2th_deg", "2θ (°)"), ("2th_rad", "2θ (rad)"),
             ("qip_A^-1", "q_ip (Å⁻¹)"), ("qoop_A^-1", "q_oop (Å⁻¹)"), ("exit_angle_deg", "Exit angle (°)"))
#: a NeXus source's 1-D modes: a single-result file's one mode, then the grazing-incidence modes (io/schema.py)
NEXUS_MODES = ("default", "q_total", "q_ip", "q_oop", "exit_angle", "chi_gi")
_UNITS = frozenset(unit for unit, _label in XYE_UNITS)
_KEYS = {"nexus": frozenset({"entry", "frame", "kind", "mode", "path", "version"}),
         "xye": frozenset({"kind", "path", "unit", "version"})}
#: xdart's .xye names are ``<prefix>_<stem>.xye`` (``xye_prefix_for_unit``); the GI prefixes before ``iq_``
_XYE_PREFIXES = (("iqip_", "qip_A^-1"), ("iqoop_", "qoop_A^-1"), ("iexit_", "exit_angle_deg"), ("itth_", "2th_deg"),
                 ("iq_", "q_A^-1"))
_CACHE_ENTRIES, _CACHE_BYTES = 8, 64 * 1024 * 1024
_cache: "OrderedDict[tuple, tuple]" = OrderedDict()
_cache_lock = threading.Lock()


class BackgroundTemplateRefused(ValueError):
    """A template refused by name; the message starts with its code (``BACKGROUND_TEMPLATE_…``)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


def _refuse(code: str, detail: str) -> BackgroundTemplateRefused:
    return BackgroundTemplateRefused(f"BACKGROUND_TEMPLATE_{code}", detail)


def _unit_name(unit: Any) -> str:
    """A unit as conversions name it: the inverse-Å aliases (``A^-1``, ``1/angstrom``…) as ``q_A^-1``."""
    from xrd_tools.analysis.axis_units import canonical_q_unit
    text = str(unit or "").strip()
    return "q_A^-1" if canonical_q_unit(text) == "q_A^-1" else text


def is_template_identity(value: Any) -> bool:
    """Whether ``value`` is a recorded template identity: 64 lowercase hexadecimal digits."""
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True, slots=True)
class BackgroundTemplateSource:
    """One named template: kind "nexus" (a file's entry, frame and 1-D mode; its unit is the file's) or "xye" (a file
    and its explicit x unit). ``path`` is kept as given; a relative one is resolved against a data root when read."""

    kind: str
    path: str
    _: KW_ONLY
    frame: int | None = None
    mode: str | None = None
    entry: str = "entry"
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KEYS:
            raise _refuse("INVALID_SOURCE", f"a template is a NeXus 1-D frame or an .xye file, not {self.kind!r}")
        path = os.fspath(self.path) if isinstance(self.path, (str, os.PathLike)) else None
        if type(path) is not str or not path.strip():
            raise _refuse("INVALID_SOURCE", "a template source names its file")
        object.__setattr__(self, "path", path)
        if self.kind == "nexus":
            if type(self.frame) is not int or self.frame < 0:
                raise _refuse("INVALID_SOURCE", f"a NeXus template names its frame, a whole number, not {self.frame!r}")
            if type(self.mode) is not str or not self.mode:
                raise _refuse("INVALID_SOURCE", "a NeXus template names its 1-D mode (\"default\" for a single result)")
            if type(self.entry) is not str or not self.entry:
                raise _refuse("INVALID_SOURCE", "a NeXus template names its entry")
            if self.unit is not None:
                raise _refuse("INVALID_SOURCE", "a NeXus template's unit is the file's own")
            return
        if self.frame is not None or self.mode is not None or self.entry != "entry":
            raise _refuse("INVALID_SOURCE", "an .xye template names only its file and its x unit")
        if self.unit is None:
            raise _refuse("UNIT_REQUIRED", "an .xye template needs its x unit; choose it, a file name only hints at it")
        unit = _unit_name(self.unit) if type(self.unit) is str else None
        if unit not in _UNITS:
            raise _refuse("INVALID_SOURCE", f"an .xye template's unit is one of {', '.join(sorted(_UNITS))}, "
                                            f"not {self.unit!r}")
        object.__setattr__(self, "unit", unit)

    def to_pairs(self) -> tuple[tuple[str, str | int], ...]:
        """The canonical primitive pairs a recipe setting or a settings section keeps (sorted, with the version)."""
        values: dict[str, Any] = {"kind": self.kind, "path": self.path, "version": SOURCE_VERSION}
        if self.kind == "nexus":
            values.update(entry=self.entry, frame=self.frame, mode=self.mode)
        else:
            values["unit"] = self.unit
        return tuple(sorted(values.items()))

    @classmethod
    def from_pairs(cls, pairs: Any) -> "BackgroundTemplateSource":
        """The source ``to_pairs`` kept, from its pairs, a JSON list of pairs or a JSON object; anything else, an
        unknown key, kind or version, or a missing field is refused by name."""
        if isinstance(pairs, Mapping):
            items = list(pairs.items())
        elif isinstance(pairs, (tuple, list)):
            if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
                raise _refuse("INVALID_SOURCE", "a template source is (key, value) pairs")
            items = [(pair[0], pair[1]) for pair in pairs]
        else:
            raise _refuse("INVALID_SOURCE", "a template source is (key, value) pairs")
        values: dict[str, Any] = {}
        for key, value in items:
            if type(key) is not str or key in values:
                raise _refuse("INVALID_SOURCE", f"a template source names each field once by text, not {key!r}")
            values[key] = value
        kind = values.get("kind")
        if kind not in _KEYS:
            raise _refuse("INVALID_SOURCE", f"a template is a NeXus 1-D frame or an .xye file, not {kind!r}")
        version = values.get("version")
        if type(version) is not int or version != SOURCE_VERSION:
            raise _refuse("INVALID_SOURCE", f"this xdart reads template source version {SOURCE_VERSION}, "
                                            f"not {version!r}")
        expected = _KEYS[kind]
        if kind == "xye" and set(values) == expected - {"unit"}:
            raise _refuse("UNIT_REQUIRED", "an .xye template needs its x unit")
        if set(values) != expected:
            raise _refuse("INVALID_SOURCE", f"a {kind} template source holds {', '.join(sorted(expected))}; "
                                            f"this one holds {', '.join(sorted(values))}")
        if kind == "nexus":
            return cls("nexus", values["path"], frame=values["frame"], mode=values["mode"], entry=values["entry"])
        return cls("xye", values["path"], unit=values["unit"])

    def describe(self) -> str:
        """The source as the template row names it: its file name and what is read from it."""
        name = Path(self.path).name
        if self.kind == "xye":
            return f"{name} ({dict(XYE_UNITS).get(self.unit, self.unit)})"
        entry = "" if self.entry == "entry" else f", {self.entry}"
        return f"{name}, frame {self.frame}, {self.mode}{entry}"


def _readonly(values: Any) -> np.ndarray:
    array = np.array(values, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def align_template(x: Any, y: Any, from_unit: str, to_unit: str) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` placed on a ``to_unit`` axis (x converted, y unchanged), or refused naming both units: Å⁻¹ and nm⁻¹
    of one axis, and degrees and radians, convert; 2θ and q need a wavelength a template does not record, and two
    grazing-incidence axes never convert."""
    source, target = _unit_name(from_unit), _unit_name(to_unit)
    if source == target:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    from xrd_tools.core.containers import convert_radial_axis
    try:
        converted = convert_radial_axis(np.asarray(x, dtype=float), source, target)
    except ValueError:
        converted = None
    if converted is None or not np.all(np.isfinite(converted)):
        raise _refuse("UNIT_UNSUPPORTED", f"a template in {from_unit} cannot be placed on a {to_unit} axis: only "
                                          "Å⁻¹ and nm⁻¹ of one axis, or degrees and radians, convert (2θ and q need a "
                                          "wavelength the template does not record); choose a template in the fit's "
                                          "unit")
    order = np.argsort(converted, kind="stable")
    return _readonly(converted[order]), _readonly(np.asarray(y, dtype=float)[order])


@dataclass(frozen=True, slots=True, eq=False)
class PreparedBackgroundTemplate:
    """A read template: ``x`` (sorted) and ``y`` read-only float64 in ``unit``, its ``source`` and its content
    ``identity``."""

    x: np.ndarray
    y: np.ndarray
    unit: str
    source: BackgroundTemplateSource
    identity: str

    def on_axis(self, unit: str) -> tuple[np.ndarray, np.ndarray]:
        """The template on a fit's ``unit`` axis (``align_template``)."""
        if _unit_name(unit) == _unit_name(self.unit):
            return self.x, self.y
        return align_template(self.x, self.y, self.unit, unit)


def xye_unit_hint(name: Any) -> str | None:
    """The unit an xdart .xye name's prefix records (``iq_``, ``itth_``, ``iqip_``, ``iqoop_``, ``iexit_``), or None:
    only a hint, which the user confirms."""
    base = str(name).replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return next((unit for prefix, unit in _XYE_PREFIXES if base.startswith(prefix)), None)


def _template_path(source: BackgroundTemplateSource, data_root: Any) -> Path:
    path = Path(source.path).expanduser()
    if not path.is_absolute() and data_root is not None:
        path = Path(data_root).expanduser() / path
    return path


def _file_facts(path: Path) -> tuple[int, int, int, int, int]:
    try:
        facts = os.stat(path)
    except FileNotFoundError:
        raise _refuse("UNREADABLE", f"the template file is missing: {path}") from None
    except OSError as error:
        raise _refuse("UNREADABLE", f"the template file cannot be read: {path} ({error.strerror or error})") from None
    if not stat.S_ISREG(facts.st_mode):
        raise _refuse("UNREADABLE", f"the template is not a regular file: {path}")
    return facts.st_dev, facts.st_ino, facts.st_size, facts.st_mtime_ns, facts.st_ctime_ns


def template_file_problem(source: BackgroundTemplateSource, *, data_root: Any = None) -> str:
    """Why ``source``'s file cannot be read now ("" when it is there): a stat, never a read (a restored setting)."""
    try:
        _file_facts(_template_path(source, data_root))
    except BackgroundTemplateRefused as refused:
        return str(refused)
    return ""


def _read_xye(source: BackgroundTemplateSource, path: Path, facts: tuple) -> tuple[str, Any, Any]:
    if facts[2] > MAX_XYE_BYTES:
        raise _refuse("INVALID_CURVE", f"{path.name} is over 64 MiB; a template holds at most 1,000,000 points")
    if facts[2] == 0:
        raise _refuse("INVALID_CURVE", f"{path.name} holds no data")
    try:
        x, y, _sigma = read_xye(path, strict=True)  # the σ column is not propagated (Codex REVIEW-1 R3)
    except Exception as error:  # noqa: BLE001 - a template file is user input
        raise _refuse("UNREADABLE", f"{path.name} is not a two- or three-column .xye table: {error}") from None
    return str(source.unit), x, y


def _read_nexus(source: BackgroundTemplateSource, path: Path, _facts: tuple) -> tuple[str, Any, Any]:
    import h5py

    from xrd_tools.io.frame_view import FrameViewReader
    try:
        with h5py.File(path, "r") as handle:
            present = isinstance(handle.get(source.entry), h5py.Group)
    except Exception as error:  # noqa: BLE001 - a template file is user input
        raise _refuse("UNREADABLE", f"{path.name} is not a readable NeXus file: {error}") from None
    if not present:
        raise _refuse("ENTRY_MISSING", f"{path.name} has no entry {source.entry!r}")
    try:
        with FrameViewReader(path, entry=source.entry, include_thumbnail=False, resolve_source=False) as reader:
            modes = tuple(reader.modes_1d())
            if source.mode not in modes:
                raise _refuse("MODE_MISSING", f"{path.name} has no 1-D mode {source.mode!r}; its modes are "
                                              f"{', '.join(modes) or 'none'}")
            if not reader.has_frame(source.frame):
                raise _refuse("FRAME_MISSING", f"{path.name} has no frame {source.frame}")
            view = reader.read(source.frame, mode_1d=source.mode)
    except BackgroundTemplateRefused:
        raise
    except Exception as error:  # noqa: BLE001 - a template file is user input
        raise _refuse("UNREADABLE", f"{path.name} is not a readable xdart NeXus record: {error}") from None
    if view.axis_1d is None or view.intensity_1d is None:
        raise _refuse("FRAME_MISSING", f"{path.name} frame {source.frame} has no 1-D curve in mode {source.mode!r}")
    return str(view.axis_1d.unit), view.axis_1d.values, view.intensity_1d


def _admit(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """One finite paired curve, sorted: at least two distinct x and at most 1,000,000 points."""
    try:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        raise _refuse("INVALID_CURVE", "a template is numeric") from None
    if x.ndim != 1 or x.shape != y.shape:
        raise _refuse("INVALID_CURVE", "a template is one curve: x and y of equal length")
    if x.size > MAX_POINTS:
        raise _refuse("INVALID_CURVE", f"a template holds at most 1,000,000 points, not {x.size}")
    if x.size < 2:
        raise _refuse("INVALID_CURVE", "a template needs at least two points")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise _refuse("INVALID_CURVE", "a template holds only finite values")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise _refuse("INVALID_CURVE", "a template's x values are distinct")
    return _readonly(x), _readonly(y)


def _identity(source: BackgroundTemplateSource, unit: str, x: np.ndarray, y: np.ndarray) -> str:
    descriptor: dict[str, Any] = {"version": SOURCE_VERSION, "kind": source.kind, "unit": unit}
    if source.kind == "nexus":
        descriptor.update(entry=source.entry, frame=source.frame, mode=source.mode)
    digest = hashlib.sha256()
    for part in (json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                 np.ascontiguousarray(x, dtype="<f8").tobytes(), np.ascontiguousarray(y, dtype="<f8").tobytes()):
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def _cached(key: tuple) -> tuple | None:
    with _cache_lock:
        value = _cache.get(key)
        if value is not None:
            _cache.move_to_end(key)
        return value


def _remember(key: tuple, value: tuple) -> None:
    if value[1].nbytes + value[2].nbytes > _CACHE_BYTES:
        return
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_ENTRIES or sum(v[1].nbytes + v[2].nbytes for v in _cache.values()) > _CACHE_BYTES:
            _cache.popitem(last=False)


def prepare_background_template(source: BackgroundTemplateSource, *, data_root: Any = None,
                                expected_identity: str | None = None) -> PreparedBackgroundTemplate:
    """Read ``source`` (a relative path under ``data_root``) as one admitted curve with its identity; refuse by name
    (``BackgroundTemplateRefused``) a source that cannot be read, a curve that is not admitted, or content that no
    longer has ``expected_identity``. Blocking: callers run it off the GUI thread."""
    if not isinstance(source, BackgroundTemplateSource):
        raise _refuse("INVALID_SOURCE", "a template source is a BackgroundTemplateSource")
    if expected_identity is not None and not is_template_identity(expected_identity):
        raise _refuse("INVALID_SOURCE", "a recorded template identity is 64 lowercase hexadecimal digits")
    path = _template_path(source, data_root)
    before = _file_facts(path)
    key = (source.kind, str(path), source.entry, source.frame, source.mode, source.unit, before)
    value = _cached(key)
    if value is None:
        unit, x, y = (_read_nexus if source.kind == "nexus" else _read_xye)(source, path, before)
        x, y = _admit(x, y)
        value = (unit, x, y, _identity(source, unit, x, y))
        if _file_facts(path) != before:
            raise _refuse("UNREADABLE", f"{path.name} changed while it was read; fit again to read it")
        _remember(key, value)
    unit, x, y, identity = value
    if expected_identity is not None and identity != expected_identity:
        raise _refuse("CHANGED", f"{path.name} no longer holds the content this template recorded; choose the file "
                                 "again to fit its current content")
    return PreparedBackgroundTemplate(x, y, unit, source, identity)
