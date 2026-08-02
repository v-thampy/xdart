"""Qt-free image-sidecar motor discovery for the Scattering Workspace."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from xrd_tools.core.metadata import numeric_metadata
from xrd_tools.io.metadata import (
    ImageMetadataRead,
    read_image_metadata,
    read_image_metadata_observed,
)


MotorCatalog = tuple[str, ...] | None


def _finite_motor_mapping(raw: dict[str, object]) -> dict[str, float]:
    filtered = {
        key: value
        for key, value in raw.items()
        if not isinstance(value, (bool, np.bool_))
    }
    numeric = numeric_metadata(filtered)
    return {
        name: value
        for name, value in numeric.items()
        if (
            name
            and "roi" not in name.casefold()
            and "pd" not in name.casefold()
        )
    }


def image_motor_metadata(
    path: Path | str,
    metadata_format: str | None,
) -> dict[str, float]:
    """Return one ordered finite numeric motor mapping from one metadata read."""

    if metadata_format is None:
        return {}
    return _finite_motor_mapping(
        read_image_metadata(Path(path), metadata_format)
    )


def read_image_motor_metadata(
    path: Path | str,
    metadata_format: str | None,
    *,
    meta_dir: Path | str | None = None,
) -> ImageMetadataRead:
    """Return finite motors plus the exact accepted metadata input."""

    if metadata_format is None:
        return ImageMetadataRead({}, None)
    observed = read_image_metadata_observed(
        Path(path),
        metadata_format,
        meta_dir=meta_dir,
    )
    return ImageMetadataRead(
        _finite_motor_mapping(dict(observed.values)),
        observed.source_path,
    )


def image_metadata_motor_names(
    path: Path | str,
    metadata_format: str | None,
) -> tuple[str, ...]:
    """Return the ordered finite numeric sidecar keys usable as motors.

    Python ``None`` is the typed metadata-off value at this GUI boundary.  It
    must not reach ``read_image_metadata``, whose headless API deliberately
    interprets ``None`` as automatic discovery.
    """

    return tuple(image_motor_metadata(path, metadata_format))


def ordered_motor_intersection(
    catalogs: Iterable[MotorCatalog],
) -> MotorCatalog:
    """Intersect known catalogs in first-catalog order.

    ``None`` means at least one catalog is unknown.  ``()`` means every
    catalog was inspected and there is no common motor.
    """

    values = tuple(catalogs)
    if not values or any(value is None for value in values):
        return None
    first = values[0] or ()
    common = set(first)
    for catalog in values[1:]:
        common.intersection_update(catalog or ())
    return tuple(
        name
        for name in first
        if name and name != "Manual" and name in common
    )


__all__ = [
    "MotorCatalog",
    "image_motor_metadata",
    "image_metadata_motor_names",
    "read_image_motor_metadata",
    "ordered_motor_intersection",
]
