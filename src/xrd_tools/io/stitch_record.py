"""Stitched results as Int records (maintainer, 2026-09-29).

A stitched pattern is an Int 1D record: I(q) is row 1 of ``integrated_1d`` and a GI cut is that row's
second mode (the primary mode is then ``q_total``), so Int's reader, view, units and shared axes apply
unchanged. ``entry/stitch`` holds what an Int record has no place for:

- each pattern's ``coverage`` (pixels per bin) and ``normalization`` (Σ weights, the denominator of I),
  the GI cut's under ``stitch/<cut>/``;
- the contributing frames: their source records under ``stitch/frames/`` (the raw popup and the
  metadata dialog read them) and their manifest, ``stitch/frames_json``;
- the Stitch output owner's request envelope (``analysis_artifact``).

The contributors live under ``stitch`` so that Int's ``frames/`` holds only the result row.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

STITCH_GROUP = "stitch"
FRAMES_JSON = "frames_json"
#: The record's one row.
LABEL = 1
#: The GI cuts a stitched record may carry beside I(q) (Int's GI 1-D mode keys).
GI_MODES = ("q_ip", "q_oop")
_PRIMARY_GI = "q_total"


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


@dataclass(frozen=True, eq=False)
class StitchRecord:
    """A stitched record as read back: I(q), the GI cut as ``(mode, pattern)`` or None, the
    wavelength (Å) or None, and the contributing-frames manifest (JSON) or None."""

    pattern: StitchedPattern
    gi: tuple[str, StitchedPattern] | None
    wavelength_A: float | None
    frames_json: str | None


def _float32(group: h5py.Group, name: str, values) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype="<f4"))


def write_stitch_record(entry: h5py.Group, target: str | Path, pattern: StitchedPattern, *,
                        gi: tuple[str, StitchedPattern] | None = None, wavelength_A: float | None = None,
                        frame_records=None, frames_json: str | None = None,
                        source_base: str | Path | None = None) -> h5py.Group:
    """Write a stitched pattern into ``entry`` of a fresh document from ``open_nexus_writer`` whose
    logical path is ``target``, and return the new ``entry/stitch`` group.

    ``gi`` is ``(mode, pattern)`` with mode 'q_ip' or 'q_oop'. ``frame_records`` are
    ``write_contributing_frames`` records and ``frames_json`` their manifest.
    """
    from xrd_tools.core.containers import IntegrationResult1D
    from xrd_tools.core.frame_view import DEFAULT_MODE_KEY
    from xrd_tools.io.nexus_record import write_contributing_frames
    from xrd_tools.io.record_writer import NexusRecordWriter, RecordWrite

    if gi is not None and gi[0] not in GI_MODES:
        raise ValueError(f"a stitched GI cut must be one of {GI_MODES}")
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
    primary = _PRIMARY_GI if gi is not None else DEFAULT_MODE_KEY
    writer.begin(primary_mode_1d=primary)
    rows = [(primary, pattern)] + ([gi] if gi is not None else [])
    for position, (mode, value) in enumerate(rows):
        writer.write_batch([RecordWrite(
            label=LABEL, result_1d=IntegrationResult1D(value.axis, value.intensity, value.sigma, value.unit),
            mode_1d=mode, write_frame_record=position == 0)])  # an extra mode joins the row's record
    writer.finish()
    stitch = entry.create_group(STITCH_GROUP)
    stitch.attrs["NX_class"] = "NXcollection"
    _float32(stitch, "coverage", pattern.coverage)
    _float32(stitch, "normalization", pattern.normalization)
    if gi is not None:
        cut = stitch.create_group(gi[0])
        cut.attrs["NX_class"] = "NXcollection"
        _float32(cut, "coverage", gi[1].coverage)
        _float32(cut, "normalization", gi[1].normalization)
    if frame_records:
        write_contributing_frames(stitch, frame_records, source_base=source_base, bounded=True)
    if frames_json is not None:
        stitch.create_dataset(FRAMES_JSON, data=np.bytes_(frames_json.encode("utf-8")))
    return stitch


def is_stitch_record(entry: h5py.Group) -> bool:
    """Whether ``entry`` is a stitched record's (an Int record with a stitch group)."""
    from xrd_tools.io.processed_scan_id import has_processed_output_markers_entry

    return isinstance(entry.get(STITCH_GROUP), h5py.Group) and has_processed_output_markers_entry(entry)


def _stored(group: h5py.Group, name: str) -> np.ndarray:
    dataset = group.get(name)
    if not isinstance(dataset, h5py.Dataset) or dataset.dtype != np.dtype("<f4") or dataset.ndim != 1:
        raise ValueError(f"stitched {name} must be a 1-D float32 dataset")
    return np.asarray(dataset[()])


def read_stitch_record(path: str | Path, *, entry: str = "entry") -> StitchRecord:
    """Read a stitched record through Int's reader (the patterns) and its stitch group."""
    from xrd_tools.io.frame_view import read_frame_view
    from xrd_tools.io.nexus import _read_energy, _read_wavelength
    from xrd_tools.io.processed_scan_id import require_current_processed_groups

    with h5py.File(Path(path), "r") as handle:
        groups = require_current_processed_groups(handle, entry, container=Path(path))
        if groups.integrated_2d is not None:
            raise ValueError("a stitched 1-D record holds no 2-D result")
        root = groups.entry
        stitch = root.get(STITCH_GROUP)
        if not isinstance(stitch, h5py.Group):
            raise ValueError("the record has no stitch group")
        modes = tuple(mode for mode, _group in groups.mode_groups_1d)
        extras = tuple(mode for mode in modes if mode != groups.primary_mode_1d)
        if extras and (groups.primary_mode_1d != _PRIMARY_GI or len(extras) != 1 or extras[0] not in GI_MODES):
            raise ValueError("a stitched record carries at most one GI cut beside q_total")
        diagnostics = {None: (_stored(stitch, "coverage"), _stored(stitch, "normalization"))}
        for mode in extras:
            cut = stitch.get(mode)
            if not isinstance(cut, h5py.Group):
                raise ValueError(f"the stitched GI cut {mode} has no coverage or normalization")
            diagnostics[mode] = (_stored(cut, "coverage"), _stored(cut, "normalization"))
        text = stitch.get(FRAMES_JSON)
        if text is not None and not isinstance(text, h5py.Dataset):
            raise ValueError("the stitched frames manifest must be a dataset")
        frames_json = None if text is None else bytes(text[()]).decode("utf-8")
        wavelength = None
        if "instrument" in root:
            wavelength = float(_read_wavelength(root, _read_energy(root)))
    patterns = {}
    for mode in (None, *extras):
        view = read_frame_view(path, LABEL, entry=entry, include_thumbnail=False, mode_1d=mode)
        coverage, normalization = diagnostics[mode]
        sigma = None if view.sigma_1d is None else np.asarray(view.sigma_1d, dtype="<f4")
        patterns[mode] = StitchedPattern(np.asarray(view.axis_1d.values, dtype="<f4"), view.axis_1d.unit,
                                         np.asarray(view.intensity_1d, dtype="<f4"), sigma, coverage, normalization)
    for mode, value in patterns.items():
        if not (value.coverage.shape == value.normalization.shape == value.axis.shape == value.intensity.shape):
            raise ValueError(f"stitched {mode or 'I(q)'} diagnostics differ in length from its pattern")
    return StitchRecord(patterns[None], None if not extras else (extras[0], patterns[extras[0]]),
                        wavelength, frames_json)


__all__ = ["FRAMES_JSON", "GI_MODES", "LABEL", "STITCH_GROUP", "StitchRecord", "StitchedPattern",
           "is_stitch_record", "read_stitch_record", "write_stitch_record"]
