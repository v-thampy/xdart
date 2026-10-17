"""Qt-free image-sidecar motor discovery for the Scattering Workspace."""

from __future__ import annotations

from pathlib import Path

from xrd_tools.core.metadata import (
    finite_motor_metadata as _finite_motor_mapping,
    ordered_motor_intersection,
)
from xrd_tools.io.metadata import (
    ImageMetadataRead,
    read_image_metadata,
    read_image_metadata_observed,
)


MotorCatalog = tuple[str, ...] | None


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


__all__ = [
    "MotorCatalog",
    "image_motor_metadata",
    "image_metadata_motor_names",
    "read_image_motor_metadata",
    "ordered_motor_intersection",
]
