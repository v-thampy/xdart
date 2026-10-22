"""Stitched results as Int records (maintainer, 2026-09-29).

A stitched pattern is an Int 1D record: I(q) is row 1 of ``integrated_1d`` and each GI cut (q_ip,
q_oop or both) is another mode of that row (the primary mode is then ``q_total``), so Int's reader,
view, units and shared axes apply unchanged. A stitched map is an Int 2D record: the (χ, q) map is row
1 of ``integrated_2d`` beside I(q) in ``integrated_1d``; a GI Stitch 2D's is the 2-D mode ``q_chi`` (the
primary), its q_ip-q_oop map the mode ``qip_qoop`` and its cuts the 1-D modes beside I(q), as Int writes
a GI run's. ``entry/stitch`` holds what an Int record has no place for:

- each pattern's ``coverage`` (pixels per bin) and ``normalization`` (Σ weights, the denominator of I),
  a GI cut's under ``stitch/<cut>/``, the map's under ``stitch/map/`` and the GI map's under
  ``stitch/qip_qoop/``, each map's in its stored (y, x) orientation;
- the contributing frames: their source records under ``stitch/frames/`` (the raw popup and the
  metadata dialog read them) and their manifest, ``stitch/contributing_frames_json``;
- the Stitch output owner's request envelope (``analysis_artifact``).

The contributors live under ``stitch`` so that Int's ``frames/`` holds only the result row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

STITCH_GROUP = "stitch"
FRAMES_JSON = "contributing_frames_json"
#: The record's one row.
LABEL = 1
#: The GI cuts a stitched record may carry beside I(q) (Int's GI 1-D mode keys), in their one order.
GI_MODES = ("q_ip", "q_oop")
_PRIMARY_GI = "q_total"
#: A stitched map's diagnostics, under ``stitch``.
MAP = "map"
#: A GI Stitch 2D's q_ip-q_oop map: its 2-D mode, and its diagnostics under ``stitch``; the (χ, q) map
#: beside it is the primary mode.
GI_MAP = "qip_qoop"
_PRIMARY_GI_MAP = "q_chi"


@dataclass(frozen=True, eq=False)
class StitchedPattern:
    """One stitched 1-D pattern: its axis and unit, intensity (sigma optional), coverage and
    normalization, one value per bin."""

    axis: np.ndarray
    unit: str
    intensity: np.ndarray
    sigma: np.ndarray | None
    coverage: np.ndarray
    normalization: np.ndarray

    @classmethod
    def from_projection(cls, projection) -> "StitchedPattern":
        """The pattern a Stitch 1D result projection (``analysis_artifact``) stores."""
        ((_name, axis),) = projection.axes
        ((_axis, unit),) = projection.axis_units
        return cls(axis, unit, projection.intensity, projection.sigma, projection.coverage,
                   projection.normalization)


@dataclass(frozen=True, eq=False)
class StitchedMap:
    """One stitched 2-D map in the container's (radial, azimuthal) orientation (``IntegrationResult2D``):
    its q and χ axes and units, intensity (sigma optional), coverage and normalization, one value per
    bin."""

    radial: np.ndarray
    azimuthal: np.ndarray
    unit: str
    azimuthal_unit: str
    intensity: np.ndarray
    sigma: np.ndarray | None
    coverage: np.ndarray
    normalization: np.ndarray

    @classmethod
    def from_projection(cls, projection) -> "StitchedMap":
        """The map a Stitch 2D result projection (``analysis_artifact``) stores."""
        (_q, radial), (_chi, azimuthal) = projection.axes
        (_q, unit), (_chi, azimuthal_unit) = projection.axis_units
        return cls(radial, azimuthal, unit, azimuthal_unit, projection.intensity, projection.sigma,
                   projection.coverage, projection.normalization)


@dataclass(frozen=True, eq=False)
class StitchRecord:
    """A stitched record as read back: I(q), the GI cuts as ``(mode, pattern)`` pairs in
    ``GI_MODES`` order (() without GI), the wavelength (Å) or None, the contributing-frames
    manifest (JSON) or None, a Stitch 2D's map or None, and a GI Stitch 2D's q_ip-q_oop map or None.
    ``angular_support`` is I(q)'s saved angular support (origin ``saved``), or the reason it has none: an older
    Stitch result without it is ``legacy_stitch`` (D6), never reconstructed."""

    pattern: StitchedPattern
    gi: tuple[tuple[str, StitchedPattern], ...]
    wavelength_A: float | None
    frames_json: str | None
    map: StitchedMap | None = None
    gi_map: StitchedMap | None = None
    angular_support: Any = None


def _float32(group: h5py.Group, name: str, values) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype="<f4"))


def write_stitch_record(entry: h5py.Group, target: str | Path, pattern: StitchedPattern, *,
                        gi: tuple[tuple[str, StitchedPattern], ...] = (), wavelength_A: float | None = None,
                        frame_records=None, frames_json: str | None = None,
                        source_base: str | Path | None = None,
                        stitched_map: StitchedMap | None = None, gi_map: StitchedMap | None = None,
                        angular_support=None) -> h5py.Group:
    """Write a stitched pattern into ``entry`` of a fresh document from ``open_nexus_writer`` whose
    logical path is ``target``, and return the new ``entry/stitch`` group.

    ``gi`` holds ``(mode, pattern)`` pairs with distinct modes 'q_ip' and 'q_oop', written in
    ``GI_MODES`` order. ``frame_records`` are ``write_contributing_frames`` records and ``frames_json``
    their manifest. ``stitched_map`` is a Stitch 2D's map, written with ``pattern`` as one Int 2D row;
    ``gi_map`` a GI Stitch 2D's q_ip-q_oop map, beside it. ``angular_support`` (an ``AngularSupportProfile``) is
    saved with the primary I(q) row, label 1, outside the Stitch fingerprint (COVERAGE-SUPPORT-20261013).
    """
    from xrd_tools.core.containers import IntegrationResult1D, IntegrationResult2D
    from xrd_tools.core.frame_view import DEFAULT_MODE_KEY
    from xrd_tools.io.nexus_record import write_contributing_frames
    from xrd_tools.io.record_writer import NexusRecordWriter, RecordWrite

    modes = [mode for mode, _value in gi]
    if any(mode not in GI_MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError(f"stitched GI cuts must be distinct modes of {GI_MODES}")
    if gi_map is not None and (stitched_map is None or not gi):
        raise ValueError("a GI map goes beside a GI Stitch 2D's map and its cuts")
    gi = tuple(sorted(gi, key=lambda cut: GI_MODES.index(cut[0])))
    for value in (pattern, *(value for _mode, value in gi),
                  *(value for value in (stitched_map, gi_map) if value is not None)):
        if value.coverage is None or value.normalization is None:
            raise ValueError("a stitched pattern needs its coverage and normalization")
    if STITCH_GROUP in entry:
        raise ValueError("the stitched record's entry already holds a stitch group")
    if wavelength_A is not None:
        from xrd_tools.core.metadata import ScanMetadata
        from xrd_tools.io.nexus import _write_metadata  # the record header open_nexus_writer writes
        from xrd_tools.transforms import wavelength_to_energy

        _write_metadata(entry, ScanMetadata(scan_id=Path(target).stem, energy=float(wavelength_to_energy(wavelength_A)),
                                            wavelength=float(wavelength_A), angles={}, counters={}), {})
    writer = NexusRecordWriter.for_finite_document(target, entry.file, entry=str(entry.name).strip("/"),
                                                   source_base=None)
    primary = _PRIMARY_GI if gi else DEFAULT_MODE_KEY
    # a GI Stitch 2D's (χ, q) map is a named primary beside its q_ip-q_oop map, as Int's GI run's
    primary_2d = _PRIMARY_GI_MAP if gi_map is not None else DEFAULT_MODE_KEY
    writer.begin(primary_mode_1d=primary, primary_mode_2d=primary_2d)

    def result_2d(value):
        return IntegrationResult2D(value.radial, value.azimuthal, value.intensity, value.sigma, value.unit,
                                   value.azimuthal_unit)

    rows = [(primary, pattern), *gi]
    support = angular_support
    if support is not None and support.supported_deg is not None \
            and support.supported_deg.size != np.asarray(pattern.axis).size:
        support = None  # never bind support to bins it does not describe
    for position, (mode, value) in enumerate(rows):
        writer.write_batch([RecordWrite(
            label=LABEL, result_1d=IntegrationResult1D(value.axis, value.intensity, value.sigma, value.unit),
            result_2d=None if stitched_map is None or position else result_2d(stitched_map),
            mode_1d=mode, mode_2d=primary_2d, write_frame_record=position == 0,  # an extra mode joins the row
            angular_support=support if position == 0 else None)])
    if gi_map is not None:
        writer.write_batch([RecordWrite(label=LABEL, result_2d=result_2d(gi_map), mode_2d=GI_MAP,
                                        write_frame_record=False)])
    writer.finish()
    from xrd_tools.io.nexus import _bounded_text_attr  # the strict envelope's fixed-width text

    stitch = entry.create_group(STITCH_GROUP)
    _bounded_text_attr(stitch, "NX_class", "NXcollection")
    _float32(stitch, "coverage", pattern.coverage)
    _float32(stitch, "normalization", pattern.normalization)
    for mode, value in gi:
        cut = stitch.create_group(mode)
        _bounded_text_attr(cut, "NX_class", "NXcollection")
        _float32(cut, "coverage", value.coverage)
        _float32(cut, "normalization", value.normalization)
    for name, value in ((MAP, stitched_map), (GI_MAP, gi_map)):
        if value is not None:  # as the map is stored, (y, x)
            group = stitch.create_group(name)
            _bounded_text_attr(group, "NX_class", "NXcollection")
            _float32(group, "coverage", np.asarray(value.coverage).T)
            _float32(group, "normalization", np.asarray(value.normalization).T)
    if frame_records:
        write_contributing_frames(stitch, frame_records, source_base=source_base, bounded=True)
    if frames_json is not None:
        stitch.create_dataset(FRAMES_JSON, data=np.bytes_(frames_json.encode("utf-8")))
    return stitch


def write_stitch_projection(entry: h5py.Group, target: str | Path, projection, *, frame_records=None,
                            source_base: str | Path | None = None, angular_support=None) -> h5py.Group:
    """Write a Stitch 1D result projection (``analysis_artifact``), the exact bytes its fingerprint
    describes: I(q), its GI cuts, the wavelength and contributors manifest it carries; or a Stitch 2D
    one's map and I(q) (its ``profile``)."""
    two_d = projection.profile is not None
    return write_stitch_record(entry, target,
                               StitchedPattern.from_projection(projection.profile if two_d else projection),
                               gi=tuple((mode, StitchedPattern.from_projection(cut)) for mode, cut in projection.gi),
                               wavelength_A=projection.wavelength_A, frame_records=frame_records,
                               frames_json=projection.frame_manifest_json, source_base=source_base,
                               stitched_map=StitchedMap.from_projection(projection) if two_d else None,
                               gi_map=None if projection.gi_map is None else StitchedMap.from_projection(projection.gi_map),
                               angular_support=angular_support)


def is_stitch_record(entry: h5py.Group) -> bool:
    """Whether ``entry`` is a stitched record's (an Int record with a stitch group)."""
    from xrd_tools.io.processed_scan_id import has_processed_output_markers_entry

    return isinstance(entry.get(STITCH_GROUP), h5py.Group) and has_processed_output_markers_entry(entry)


def _stored(group: h5py.Group, name: str, ndim: int = 1) -> np.ndarray:
    dataset = group.get(name)
    if not isinstance(dataset, h5py.Dataset) or dataset.dtype != np.dtype("<f4") or dataset.ndim != ndim:
        raise ValueError(f"stitched {name} must be a {ndim}-D float32 dataset")
    return np.asarray(dataset[()])


def _row(group: h5py.Group, name: str, ndim: int = 1) -> np.ndarray:
    dataset = group.get(name)
    if (not isinstance(dataset, h5py.Dataset) or dataset.dtype != np.dtype("<f4")
            or dataset.ndim != ndim + 1 or dataset.shape[0] != 1):
        raise ValueError(f"a stitched record's {name} must be one float32 row")
    return np.asarray(dataset[0])


def _unit(dataset) -> str:
    unit = dataset.attrs.get("units")
    unit = unit.decode("utf-8") if isinstance(unit, bytes) else unit
    if not isinstance(unit, str) or not unit:
        raise ValueError("a stitched record's axis has no unit")
    return unit


def _pattern(group: h5py.Group, coverage: np.ndarray, normalization: np.ndarray) -> StitchedPattern:
    """One mode's row of the record (the Int layout: axis_1, frame_index, intensity, sigma)."""
    _one_row(group)
    sigma = _row(group, "sigma") if "sigma" in group else None
    pattern = StitchedPattern(_stored(group, "axis_1"), _unit(group["axis_1"]), _row(group, "intensity"), sigma,
                              coverage, normalization)
    if not (pattern.axis.shape == pattern.intensity.shape == coverage.shape == normalization.shape):
        raise ValueError("a stitched pattern's diagnostics differ in length from it")
    return pattern


def _one_row(group: h5py.Group) -> None:
    index = group.get("frame_index")
    if not isinstance(index, h5py.Dataset) or np.asarray(index[()]).tolist() != [LABEL]:
        raise ValueError("a stitched record holds exactly row 1")


def _map(group: h5py.Group, diagnostics: h5py.Group) -> StitchedMap:
    """The record's map row (Int's 2-D layout: axis_1 q, axis_2 χ, intensity (row, χ, q)), in the
    container's (radial, azimuthal) orientation."""
    _one_row(group)
    for name in ("axis_1", "axis_2"):
        if not isinstance(group.get(name), h5py.Dataset):
            raise ValueError(f"a stitched map has no {name}")
    sigma = _row(group, "sigma", 2).T if "sigma" in group else None
    value = StitchedMap(_stored(group, "axis_1"), _stored(group, "axis_2"), _unit(group["axis_1"]),
                        _unit(group["axis_2"]), _row(group, "intensity", 2).T, sigma,
                        _stored(diagnostics, "coverage", 2).T, _stored(diagnostics, "normalization", 2).T)
    shape = (value.radial.size, value.azimuthal.size)
    if not (value.intensity.shape == value.coverage.shape == value.normalization.shape == shape):
        raise ValueError("a stitched map's diagnostics differ in shape from it")
    return value


def _read(handle: h5py.File, entry: str, container: Path) -> StitchRecord:
    from xrd_tools.io.nexus import _read_energy, _read_wavelength
    from xrd_tools.io.processed_scan_id import require_current_processed_groups

    groups = require_current_processed_groups(handle, entry, container=container)
    root = groups.entry
    stitch = root.get(STITCH_GROUP)
    if not isinstance(stitch, h5py.Group):
        raise ValueError("the record has no stitch group")
    if groups.integrated_1d is None or (groups.integrated_2d is not None) is not (MAP in stitch):
        raise ValueError("a stitched record holds I(q), and a map with its diagnostics or none")
    maps = {mode for mode, _group in groups.mode_groups_2d}
    if groups.integrated_2d is not None and maps not in ({groups.primary_mode_2d}, {_PRIMARY_GI_MAP, GI_MAP}):
        raise ValueError("a stitched record holds one map, or a GI Stitch 2D's (χ, q) and q_ip-q_oop maps")
    if (GI_MAP in maps) is not (GI_MAP in stitch) or (GI_MAP in maps and groups.primary_mode_2d != _PRIMARY_GI_MAP):
        raise ValueError("a GI Stitch 2D's q_ip-q_oop map goes beside its (χ, q) map, with its diagnostics")
    extras = {mode for mode, _group in groups.mode_groups_1d if mode != groups.primary_mode_1d}
    if extras and (groups.primary_mode_1d != _PRIMARY_GI or not extras <= set(GI_MODES)):
        raise ValueError(f"a stitched record carries only the GI cuts {GI_MODES} beside q_total")
    pattern = _pattern(groups.integrated_1d, _stored(stitch, "coverage"), _stored(stitch, "normalization"))
    gi = []
    for mode in (mode for mode in GI_MODES if mode in extras):  # one order, not the file's
        cut = stitch.get(mode)
        if not isinstance(cut, h5py.Group):
            raise ValueError(f"the stitched GI cut {mode} has no coverage or normalization")
        value = _pattern(groups.mode_group("1d", mode), _stored(cut, "coverage"), _stored(cut, "normalization"))
        if value.sigma is not None:  # no fingerprint covers one (Codex review F6)
            raise ValueError("a stitched GI cut carries no sigma")
        gi.append((mode, value))
    text = stitch.get(FRAMES_JSON)
    if text is not None and not isinstance(text, h5py.Dataset):
        raise ValueError("the stitched frames manifest must be a dataset")
    frames_json = None if text is None else bytes(text[()]).decode("utf-8")
    wavelength = float(_read_wavelength(root, _read_energy(root))) if "instrument" in root else None
    stitched = {}
    for name, mode in ((MAP, groups.primary_mode_2d), (GI_MAP, GI_MAP)):
        diagnostics = stitch.get(name)
        if diagnostics is not None and not isinstance(diagnostics, h5py.Group):
            raise ValueError("a stitched map's coverage and normalization must be a group")
        stitched[name] = (None if diagnostics is None
                          else _map(groups.integrated_2d if name == MAP else groups.mode_group("2d", mode), diagnostics))
    if stitched[GI_MAP] is not None and not gi:
        raise ValueError("a GI map goes beside a GI Stitch 2D's cuts")
    return StitchRecord(pattern, tuple(gi), wavelength, frames_json, stitched[MAP], stitched[GI_MAP],
                        _saved_support(groups.integrated_1d, pattern.intensity))


def _saved_support(group: h5py.Group, intensity) -> Any:
    """I(q)'s saved angular support (label 1), or its reason: absent is an older Stitch result (D6)."""
    from xrd_tools.core.angular_support import AngularSupportProfile, SupportOrigin, SupportReason
    from xrd_tools.io.angular_support import AngularSupportFormatError, SupportTableReader
    try:
        profile = SupportTableReader(group, frame_count=1).profile(LABEL, intensity)
    except AngularSupportFormatError as error:
        return AngularSupportProfile.unavailable(SupportReason.MALFORMED_SAVED_SUPPORT, origin=SupportOrigin.SAVED,
                                                 detail=str(error))
    if profile.reason is SupportReason.NO_SAVED_SUPPORT:
        return AngularSupportProfile.unavailable(SupportReason.LEGACY_STITCH, origin=SupportOrigin.SAVED)
    return profile


def read_stitch_record(source: str | Path | h5py.File, *, entry: str = "entry") -> StitchRecord:
    """Read a stitched record from its path or an open file (the Stitch output owner reads through
    the one handle it admitted). Int's own reader reads the same patterns (``read_frame_view``)."""
    if isinstance(source, h5py.File):
        return _read(source, entry, Path(source.filename))
    with h5py.File(Path(source), "r") as handle:
        return _read(handle, entry, Path(source))


__all__ = ["FRAMES_JSON", "GI_MAP", "GI_MODES", "LABEL", "MAP", "STITCH_GROUP", "StitchRecord", "StitchedMap",
           "StitchedPattern", "is_stitch_record", "read_stitch_record", "write_stitch_projection",
           "write_stitch_record"]
