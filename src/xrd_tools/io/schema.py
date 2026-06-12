# -*- coding: utf-8 -*-
"""Schema-as-code for the processed-scan NeXus record (v2).

The single declarative description of the on-disk layout that
``xrd_tools.io.nexus`` writes and the readers consume.  Layout facts —
the schema stamp, which datasets are row-aligned (one leading per-frame
dimension), the axis dataset names, the capability attributes — live HERE
so writers, validators, readers, and row surgery share one source of
truth instead of each re-hard-coding strings.

This module describes the format; it never changes it.  Everything below
is **persisted** in existing user files — treat every string as frozen.
Schema evolution = bump :data:`PROCESSED_SCHEMA_VERSION` and extend the
structures additively; never rename an attribute key or dataset name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "SCHEMA_NAME_ATTR",
    "SCHEMA_VERSION_ATTR",
    "DTYPE_ATTR",
    "MONOTONIC_ATTR",
    "SOURCE_BASE_ATTR",
    "THUMBNAIL_LUT_ATTRS",
    "PROCESSED_SCHEMA_NAME",
    "ACCEPTED_SCHEMA_NAMES",
    "PROCESSED_SCHEMA_VERSION",
    "INTEGRATED_ROW_ALIGNED",
    "GroupSchema",
    "ProcessedScanSchema",
    "SCHEMA",
]


# ── persisted attribute KEYS (entry/group/dataset attrs) ─────────────────────
# The "ssrl_" prefix is the historical (pre-monorepo) name and is part of the
# on-disk format — keys stay even though the package is now xrd_tools.

#: entry attr naming the schema this file follows.
SCHEMA_NAME_ATTR = "ssrl_schema"
#: entry attr carrying the integer schema version (readers' C1 check).
SCHEMA_VERSION_ATTR = "ssrl_schema_version"
#: scan_data column attr recording the logical dtype ("string"/"float32").
DTYPE_ATTR = "ssrl_dtype"

# ── capability attributes (optional; readers feature-detect, never require) ──

#: group attr: frame_index is strictly increasing → readers may binary-search
#: / fast-append instead of scanning all labels.
MONOTONIC_ATTR = "_frame_index_strictly_increasing"
#: entry attr: POSIX project root that relative ``source/path`` pointers
#: resolve against (the N1 portability contract).
SOURCE_BASE_ATTR = "source_base"
#: thumbnail dataset attrs storing the quantization LUT for inversion.
THUMBNAIL_LUT_ATTRS = ("vmin", "vmax", "dtype")

# ── schema identity ──────────────────────────────────────────────────────────

#: stamped on every newly written file.
PROCESSED_SCHEMA_NAME = "xrd_tools.processed_scan"
#: names a reader should treat as this schema — files written before the
#: monorepo rename (xdart ≤0.40 / ssrl_xrd_tools ≤0.41) carry the old name.
ACCEPTED_SCHEMA_NAMES = (
    PROCESSED_SCHEMA_NAME,
    "ssrl_xrd_tools.processed_scan",
)
#: current schema version; readers warn (never refuse) on newer files.
PROCESSED_SCHEMA_VERSION = 2

# ── row-aligned datasets ─────────────────────────────────────────────────────

#: datasets inside ``integrated_1d``/``integrated_2d`` whose LEADING dimension
#: is the per-frame row — exactly these are sliced/rebuilt by row surgery
#: (``drop_integrated_rows``) and grown by the appenders.  Axis datasets
#: (``q``/``chi``) are shared across rows and are NOT in this set.
INTEGRATED_ROW_ALIGNED = frozenset({"frame_index", "intensity", "sigma"})


@dataclass(frozen=True)
class GroupSchema:
    """Declarative description of one entry-level group."""

    name: str
    #: shared (non-row) axis DATASET NAMES, (radial, azimuthal) order.
    #: NOTE: not the intensity storage order — integrated_2d intensity rows
    #: are stored (chi, q) = (azimuthal, radial); see the 2D-orientation
    #: convention in CLAUDE.md before consuming axes positionally.
    axes: tuple[str, ...] = ()
    #: datasets with a per-frame leading dimension.
    row_aligned: frozenset = frozenset()


@dataclass(frozen=True)
class ProcessedScanSchema:
    """The whole v2 processed-scan record, as data."""

    name: str = PROCESSED_SCHEMA_NAME
    accepted_names: tuple[str, ...] = ACCEPTED_SCHEMA_NAMES
    version: int = PROCESSED_SCHEMA_VERSION
    name_attr: str = SCHEMA_NAME_ATTR
    version_attr: str = SCHEMA_VERSION_ATTR
    groups: Mapping[str, GroupSchema] = field(
        default_factory=lambda: MappingProxyType({
            "integrated_1d": GroupSchema(
                "integrated_1d", axes=("q",),
                row_aligned=INTEGRATED_ROW_ALIGNED,
            ),
            "integrated_2d": GroupSchema(
                "integrated_2d", axes=("q", "chi"),
                row_aligned=INTEGRATED_ROW_ALIGNED,
            ),
            "per_frame_geometry": GroupSchema(
                "per_frame_geometry",
                row_aligned=frozenset({
                    "frame_index", "rot1", "rot2", "rot3", "incident_angle",
                }),
            ),
        })
    )


#: the singleton consumers import.
SCHEMA = ProcessedScanSchema()
