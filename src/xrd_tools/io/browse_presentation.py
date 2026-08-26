"""Bounded, presentation-only reads for processed Browse artifacts.

This deliberately is not a provenance reader.  Browse needs four detached
configuration facts plus the persisted wavelength; walking the complete
``reduction`` tree (and then the complete metadata table a second time) makes
terminal handoff latency depend on unrelated artifact history.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xrd_tools.core.energy import WavelengthUnit, canonical_wavelength_m
from xrd_tools.transforms import energy_to_wavelength


_MAX_PRESENTATION_SCALAR_BYTES = 64 * 1024
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


def _bounded_utf8_scalar(dataset: h5py.Dataset, *, role: str) -> str:
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
        or (info.length is not None and info.length > _MAX_PRESENTATION_SCALAR_BYTES)
    ):
        raise ValueError(f"{role} is not a bounded local UTF-8 scalar")
    destination = np.empty(
        (), dtype=f"S{_MAX_PRESENTATION_SCALAR_BYTES + 1}",
    )
    try:
        dataset.read_direct(destination)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"{role} could not be read as bounded UTF-8") from error
    raw = bytes(destination[()])
    if len(raw) > _MAX_PRESENTATION_SCALAR_BYTES:
        raise ValueError(f"{role} exceeds the UTF-8 byte ceiling")
    try:
        return raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} is not UTF-8") from error


def _json_value(dataset: h5py.Dataset, *, role: str) -> Any:
    text = _bounded_utf8_scalar(dataset, role=role)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # This is the historical read_provenance behaviour for a plain string.
        return text


def _positive_scalar(dataset: h5py.Dataset | None, *, role: str) -> float | None:
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
        if monochromator is not None:
            wavelength_a = _positive_scalar(
                _optional_dataset(
                    monochromator,
                    "wavelength",
                    role="Browse wavelength",
                ),
                role="Browse wavelength",
            )
            if wavelength_a is None:
                energy_kev = _positive_scalar(
                    _optional_dataset(
                        monochromator, "energy", role="Browse energy",
                    ),
                    role="Browse energy",
                )
                if energy_kev is not None:
                    wavelength_a = float(energy_to_wavelength(energy_kev))
            wavelength_m = canonical_wavelength_m(
                wavelength_a, WavelengthUnit.ANGSTROM,
            )
            if wavelength_m is not None:
                presentation["wavelength_m"] = wavelength_m
    return presentation, mask


__all__ = ["read_browse_presentation"]
