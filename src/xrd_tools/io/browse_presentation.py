"""Bounded, presentation-only reads for processed Browse artifacts.

This deliberately is not a provenance reader.  Browse needs four detached
configuration facts plus the persisted wavelength; walking the complete
``reduction`` tree (and then the complete metadata table a second time) makes
terminal handoff latency depend on unrelated artifact history.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from xrd_tools.core.energy import WavelengthUnit, canonical_wavelength_m
from xrd_tools.io.processed_scan_id import require_current_processed
from xrd_tools.transforms import energy_to_wavelength


_MAX_PRESENTATION_SCALAR_BYTES = 64 * 1024
#: The writer's own per-entry ceiling for ``run_configuration`` and ``energy_selection`` (append.py), also applied
#: to ``append_lineage`` here: a longer lineage refuses an energy scan's rings, never the Browse load.
_MAX_RING_RECORD_UTF8_BYTES = 262_144
_MAX_GEOMETRY_FIELDS = 32
_PRESENTATION_KEYS = ("poni_file", "gi", "mask_file")


def _direct_dataset(group: h5py.Group, name: str) -> h5py.Dataset | None:
    """Return one local hard-linked dataset without following foreign links."""

    try:
        link = group.get(name, getlink=True)
        node = group.get(name) if type(link) is h5py.HardLink else None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if not isinstance(node, h5py.Dataset):
        return None
    try:
        external = tuple(node.external or ())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    return None if node.is_virtual or external else node


def _direct_group(group: h5py.Group, name: str) -> h5py.Group | None:
    try:
        link = group.get(name, getlink=True)
        node = group.get(name) if type(link) is h5py.HardLink else None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return node if isinstance(node, h5py.Group) else None


def _optional_group(
    group: h5py.Group, name: str, *, role: str,
) -> h5py.Group | None:
    try:
        link = group.get(name, getlink=True)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} link is unreadable") from error
    if link is None:
        return None
    if type(link) is not h5py.HardLink:
        raise ValueError(f"{role} is not a local hard-linked group")
    value = _direct_group(group, name)
    if value is None:
        raise ValueError(f"{role} is not a local hard-linked group")
    return value


def _optional_dataset(
    group: h5py.Group, name: str, *, role: str,
) -> h5py.Dataset | None:
    try:
        link = group.get(name, getlink=True)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} link is unreadable") from error
    if link is None:
        return None
    if type(link) is not h5py.HardLink:
        raise ValueError(f"{role} is not a bounded local hard-linked dataset")
    value = _direct_dataset(group, name)
    if value is None:
        raise ValueError(f"{role} is not a bounded local hard-linked dataset")
    return value


def _bounded_utf8_scalar(
    dataset: h5py.Dataset, *, role: str, max_bytes: int = _MAX_PRESENTATION_SCALAR_BYTES,
) -> str:
    """Read a UTF-8 scalar through a fixed-size HDF5 conversion buffer."""

    info = h5py.check_string_dtype(dataset.dtype)
    try:
        external = tuple(dataset.external or ())
    except (OSError, RuntimeError, TypeError, ValueError):
        external = ("unreadable",)
    if (
        dataset.shape != ()
        or dataset.is_virtual
        or external
        or info is None
        or info.encoding != "utf-8"
        or (info.length is not None and info.length > max_bytes)
    ):
        raise ValueError(f"{role} is not a bounded local UTF-8 scalar")
    destination = np.empty((), dtype=f"S{max_bytes + 1}")
    try:
        dataset.read_direct(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} could not be read as bounded UTF-8") from error
    raw = bytes(destination[()])
    if len(raw) > max_bytes:
        raise ValueError(f"{role} exceeds the UTF-8 byte ceiling")
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} is not UTF-8") from error


def _json_value(
    dataset: h5py.Dataset, *, role: str, max_bytes: int = _MAX_PRESENTATION_SCALAR_BYTES,
) -> Any:
    text = _bounded_utf8_scalar(dataset, role=role, max_bytes=max_bytes)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # This is the historical read_provenance behaviour for a plain string.
        return text


def _positive_scalar(dataset: h5py.Dataset | None, *, role: str, scalar_only: bool = False) -> float | None:
    if scalar_only and dataset is not None and dataset.shape != ():
        raise ValueError(f"{role} is not a scalar")
    if dataset is None or dataset.dtype.kind not in "iuf" or dataset.size < 1:
        return None
    if dataset.ndim == 0:
        selection = ()
    elif dataset.ndim == 1:
        selection = 0
    else:
        return None
    try:
        value = float(dataset[selection])
    except (IndexError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def read_browse_presentation(
    path: str | Path, *, entry: str = "entry",
) -> tuple[dict[str, Any], Any | None]:
    """Return only Browse's detached presentation facts and mask identity.

    The artifact is opened once, exact config nodes are addressed directly,
    JSON scalars are bounded to 64 KiB, and geometry is capped at 32 scalar
    fields.  No frame inventory, axis, metadata column, input lineage, or
    unrelated reduction node is traversed.
    """

    presentation: dict[str, Any] = {}
    mask: Any | None = None
    with h5py.File(Path(path), "r") as handle:
        require_current_processed(handle, entry)
        entry_group = _optional_group(handle, entry, role=f"/{entry}")
        if entry_group is None:
            return presentation, mask
        reduction = _optional_group(
            entry_group, "reduction", role=f"/{entry}/reduction",
        )
        config = (
            None
            if reduction is None
            else _optional_group(
                reduction, "config", role=f"/{entry}/reduction/config",
            )
        )
        if config is not None:
            for key in _PRESENTATION_KEYS:
                dataset = _optional_dataset(
                    config, key, role=f"Browse config {key}",
                )
                if dataset is None:
                    continue
                value = _json_value(dataset, role=f"Browse config {key}")
                if key == "mask_file":
                    mask = value
                else:
                    presentation[key] = value
            geometry = _optional_group(
                config, "geometry", role="Browse geometry",
            )
            if geometry is not None:
                if len(geometry) > _MAX_GEOMETRY_FIELDS:
                    raise ValueError("Browse geometry field inventory exceeds limit")
                values: dict[str, Any] = {}
                for name in geometry:
                    dataset = _optional_dataset(
                        geometry, str(name), role=f"Browse geometry {name}",
                    )
                    if dataset is None:
                        raise ValueError("Browse geometry contains a non-scalar field")
                    values[str(name)] = _json_value(
                        dataset, role=f"Browse geometry {name}",
                    )
                if values:
                    presentation["geometry"] = values

        wavelength_m = _monochromator_wavelength_m(entry_group, entry)
        if wavelength_m is not None:
            presentation["wavelength_m"] = wavelength_m
    return presentation, mask


def _monochromator_wavelength_m(entry_group: h5py.Group, entry: str, *, scalar_only: bool = False) -> float | None:
    """The output's recorded wavelength (m): the monochromator's wavelength, else its energy."""

    instrument = _optional_group(
        entry_group, "instrument", role=f"/{entry}/instrument",
    )
    monochromator = (
        None
        if instrument is None
        else _optional_group(
            instrument,
            "monochromator",
            role=f"/{entry}/instrument/monochromator",
        )
    )
    if monochromator is None:
        return None
    wavelength_a = _positive_scalar(
        _optional_dataset(
            monochromator,
            "wavelength",
            role="Browse wavelength",
        ),
        role="Browse wavelength",
        scalar_only=scalar_only,
    )
    if wavelength_a is None:
        energy_kev = _positive_scalar(
            _optional_dataset(
                monochromator, "energy", role="Browse energy",
            ),
            role="Browse energy",
            scalar_only=scalar_only,
        )
        if energy_kev is not None:
            wavelength_a = float(energy_to_wavelength(energy_kev))
    return canonical_wavelength_m(wavelength_a, WavelengthUnit.ANGSTROM)


@dataclass(frozen=True, slots=True)
class SavedRingRecord:
    """The detached record nodes a processed output's predicted rings are rebuilt from.

    ``kind`` is ``"run"``, ``"average"`` or ``"reintegrate"``. ``epochs`` holds the Append lineage's labels per
    epoch, read only for an energy scan (per-frame energies). ``unreadable`` names nodes that exist but could not
    be read within their bounds; an absent node is ``None`` instead.
    """

    kind: str
    run_configuration: Mapping[str, Any] | None
    energy_selection: Mapping[str, Any] | None
    monochromator_wavelength_m: float | None
    epochs: tuple[tuple[int, ...], ...] | None
    unreadable: tuple[str, ...]


def read_ring_record(path: str | Path, *, entry: str = "entry") -> SavedRingRecord:
    """Read only the declared local scalar nodes predicted rings need (BROWSE-REFERENCE-RINGS-20261013).

    ``reduction/config`` ``run_configuration`` and ``energy_selection`` (each at most 262,144 UTF-8 bytes, the
    writer's ceiling), ``append_lineage`` under the same bound and only for an energy scan, and the monochromator
    wavelength. Links, virtual, external or non-scalar nodes and over-ceiling values are unreadable, not errors;
    nothing detector-sized and no handle survives the call.
    """

    from xrd_tools.io.append import _lineage_labels
    from xrd_tools.io.frame_view import _is_current_average_frame_one

    unreadable: list[str] = []
    with h5py.File(Path(path), "r") as handle:
        require_current_processed(handle, entry)
        entry_group = _optional_group(handle, entry, role=f"/{entry}")
        if entry_group is None:
            raise ValueError(f"processed output has no {entry!r} entry")
        reduction = _optional_group(entry_group, "reduction", role=f"/{entry}/reduction")
        config = (None if reduction is None
                  else _optional_group(reduction, "config", role=f"/{entry}/reduction/config"))

        def node(name: str) -> Any:
            try:
                dataset = _optional_dataset(config, name, role=f"Ring config {name}")
                if dataset is None:
                    return None
                value = _json_value(
                    dataset, role=f"Ring config {name}", max_bytes=_MAX_RING_RECORD_UTF8_BYTES)
            except ValueError:
                value = ""
            if type(value) is not dict:
                unreadable.append(name)
                return None
            return value

        names = () if config is None else tuple(str(name) for name in config)
        kind = ("average" if _is_current_average_frame_one(entry_group)
                else "reintegrate" if any(name.startswith("dimension_replacement_") for name in names)
                else "run")
        run_configuration = None if config is None else node("run_configuration")
        energy_selection = None if config is None else node("energy_selection")
        epochs = None
        if energy_selection is not None and "frame_energies_eV" in energy_selection:
            lineage = node("append_lineage")
            if lineage is not None:
                try:
                    _lineage_labels(lineage)  # the Append owner's own validation
                    epochs = tuple(tuple(epoch["labels"]) for epoch in lineage["epochs"])
                except (KeyError, TypeError, ValueError):
                    unreadable.append("append_lineage")
        try:
            wavelength_m = _monochromator_wavelength_m(entry_group, entry, scalar_only=True)
        except ValueError:
            wavelength_m = None
            unreadable.append("monochromator")
    return SavedRingRecord(kind, run_configuration, energy_selection, wavelength_m, epochs, tuple(unreadable))


#: The longest saved 1-D axis an older output's support reconstruction reads (COVERAGE-SUPPORT-20261013).
_MAX_SUPPORT_AXIS_POINTS = 1 << 20


@dataclass(frozen=True, slots=True, eq=False)
class SavedSupportFacts:
    """The detached nodes an older Int output's angular support is rebuilt from, beyond its ring record (D5).

    ``mask_signed``: the record states whether a user mask applied (its accepted ``mask_sha256``); ``mask_sha256``
    None means none did, and a digest's pixels are ``mask_indices`` (flat, C order) when the output embedded them.
    ``axis_values`` is the primary 1-D profile's saved float32 axis, read-only.
    """

    mask_signed: bool
    mask_sha256: str | None
    mask_indices: np.ndarray | None
    axis_values: np.ndarray | None


def read_saved_support_facts(path: str | Path, *, entry: str = "entry", run_configuration: Any,
                             pixels: int) -> SavedSupportFacts:
    """Read the user-mask facts and the saved 1-D axis an older output's support needs, once and bounded.

    A separate optional read (never part of :func:`read_ring_record`, which stays free of detector arrays): the
    embedded mask's flat indices (``instrument/detector/mask``, at most ``pixels``, the detector's size) and the
    primary ``integrated_1d/axis_1``. Links, virtual or oversized nodes are unreadable: None, never a guess.
    """

    signature = run_configuration.get("scientific_signature") if isinstance(run_configuration, Mapping) else None
    assets = signature.get("accepted_scientific_assets") if isinstance(signature, Mapping) else None
    signed = isinstance(assets, Mapping) and "mask_sha256" in assets
    digest = assets.get("mask_sha256") if signed else None
    if digest is not None and (type(digest) is not str or len(digest) != 64):
        signed, digest = False, None
    indices = axis = None
    with h5py.File(Path(path), "r") as handle:
        require_current_processed(handle, entry)
        entry_group = _optional_group(handle, entry, role=f"/{entry}")
        if entry_group is None:
            raise ValueError(f"processed output has no {entry!r} entry")

        def read(group_path: str, name: str, limit: int, kinds: str):
            try:
                group = entry_group
                for part in group_path.split("/"):
                    group = None if group is None else _optional_group(group, part, role=f"/{entry}/{group_path}")
                node = None if group is None else _optional_dataset(group, name, role=f"/{entry}/{group_path}/{name}")
            except ValueError:
                return None
            if (node is None or node.ndim != 1 or node.dtype.kind not in kinds or not 0 < node.shape[0] <= limit
                    or node.is_virtual or node.external):
                return None
            value = np.array(node[()], copy=True)
            value.setflags(write=False)
            return value

        if digest is not None:
            raw = read("instrument/detector", "mask", int(pixels), "iu")
            indices = None if raw is None else raw.astype(np.int64)
        axis = read("integrated_1d", "axis_1", _MAX_SUPPORT_AXIS_POINTS, "f")
        if axis is not None:
            axis = np.ascontiguousarray(axis, dtype=np.float32)
            axis.setflags(write=False)
    return SavedSupportFacts(signed, digest, indices, axis)


__all__ = ["SavedRingRecord", "SavedSupportFacts", "read_browse_presentation", "read_ring_record",
           "read_saved_support_facts"]
