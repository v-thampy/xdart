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

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from xrd_tools.core.frame_view import DEFAULT_MODE_KEY  # "default"; the top-level slot

logger = logging.getLogger(__name__)

__all__ = [
    "CAPABILITIES",
    "CapabilityAttr",
    "DatasetSpec",
    "detect_capabilities",
    "is_known_schema_name",
    "SCHEMA_NAME_ATTR",
    "SCHEMA_VERSION_ATTR",
    "DTYPE_ATTR",
    "MONOTONIC_ATTR",
    "PRIMARY_MODE_ATTR",
    "MULTI_RESULT_MODES_ATTR",
    "GI_MODE_KEYS_1D",
    "GI_MODE_KEYS_2D",
    "MODE_SUBGROUP_NAMES",
    "mode_subgroup_name",
    "subgroup_mode_key",
    "resolve_mode_path",
    "DEFAULT_MODE_KEY",
    "SOURCE_BASE_ATTR",
    "THUMBNAIL_LUT_ATTRS",
    "PROCESSED_SCHEMA_NAME",
    "ACCEPTED_SCHEMA_NAMES",
    "PROCESSED_SCHEMA_VERSION",
    "INTEGRATED_ROW_ALIGNED",
    "GroupSchema",
    "ProcessedScanSchema",
    "SCHEMA",
    "REINTEGRATE_SHADOW_SUFFIX",
    "REINTEGRATE_SHADOW_COMPLETE_ATTR",
    "resolve_integrated_group",
]


# ── streaming-reintegrate shadow recovery ────────────────────────────────────
#: suffix of the shadow group a streaming reintegrate stages rows into before
#: the atomic swap (writer side: xdart.modules.ewald.nexus_writer).  Shared here
#: so the headless readers can recover an orphan left by a crash mid-swap.
REINTEGRATE_SHADOW_SUFFIX = "__reint"
#: attr the writer stamps on a shadow ONLY once its coverage is validated, right
#: before the swap deletes the canonical group.  An orphan shadow is adopted as
#: the authoritative result ONLY when it carries this marker -- a shadow left by
#: a crash MID-WRITE (still streaming, never validated; e.g. a 2D reintegrate on
#: a 1D-only scan whose canonical 2D never existed) is partial and must NOT be
#: presented as complete.
REINTEGRATE_SHADOW_COMPLETE_ATTR = "reintegrate_shadow_complete"


def resolve_integrated_group(entry_grp, group_name: str):
    """Resolve a canonical integrated group, recovering from a crash mid-swap.

    Returns ``(group, adopted)``: the canonical ``group_name`` group when
    present; else, when only its ``<group_name>__reint`` orphan shadow survives
    AND that shadow was marked COMPLETE by the writer (a crash precisely between
    the swap's delete-canonical and move-shadow), the shadow is adopted
    READ-ONLY (the file is not mutated) so consumers "open sanely" on the
    complete result.  An UNMARKED orphan (a reintegrate crashed mid-write -- its
    rows are partial) is ignored, never presented as the result; else
    ``(None, False)``.  A writer pass repairs/clears the orphan via
    ``cleanup_reintegrate_shadow_groups``.
    """
    g = entry_grp.get(group_name)
    if g is not None:
        return g, False
    shadow = entry_grp.get(f"{group_name}{REINTEGRATE_SHADOW_SUFFIX}")
    if shadow is None:
        return None, False
    if shadow.attrs.get(REINTEGRATE_SHADOW_COMPLETE_ATTR):
        logger.warning(
            "Adopting completed orphan reintegration shadow %s%s as %s "
            "(read-only; the file crashed mid-swap -- run a writer pass to "
            "repair).", group_name, REINTEGRATE_SHADOW_SUFFIX, group_name,
        )
        return shadow, True
    logger.warning(
        "Ignoring incomplete orphan reintegration shadow %s%s: a reintegrate "
        "crashed mid-write, so its rows are partial and are NOT presented as "
        "%s (which is absent on disk).",
        group_name, REINTEGRATE_SHADOW_SUFFIX, group_name,
    )
    return None, False


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
#: integrated_1d/2d scan-level attr: the mode_key whose result occupies the
#: TOP-LEVEL slot (per-scan, self-describing reader pointer).  Absent ⇒ a
#: standard/old single-result file whose top-level slot is DEFAULT_MODE_KEY.
PRIMARY_MODE_ATTR = "primary_mode"
#: integrated_1d/2d scan-level attr: ORDERED list[str] of EVERY persisted
#: mode_key for that dimension (primary FIRST).  Presence is the capability
#: marker for a mode-aware file; a single named GI mode lists just ``[primary]``.
MULTI_RESULT_MODES_ATTR = "multi_result_modes"
#: entry attr: POSIX project root that relative ``source/path`` pointers
#: resolve against (the N1 portability contract).
SOURCE_BASE_ATTR = "source_base"
#: thumbnail dataset attrs storing the quantization LUT for inversion
#: (consumed by nexus_record.write_thumbnail and read._dequantize_thumbnail).
THUMBNAIL_LUT_ATTRS = ("vmin", "vmax", "dtype")


@dataclass(frozen=True)
class CapabilityAttr:
    """One optional v2 feature, feature-detected by PRESENCE (ADR-0002).

    The integer schema version never moves for additive features; a
    reader uses a capability iff its marker is present AND the registry
    knows it.  ``marker`` is the on-disk name; ``kind`` says what to look
    for at ``location`` (relative to the entry group)."""

    marker: str
    location: str                 # "" = the entry group itself
    kind: str                     # "attr" | "group" | "dataset"
    meaning: str
    introduced: int = 2


#: the optional features of the v2 record (additive-only; never remove).
CAPABILITIES: "Mapping[str, CapabilityAttr]" = None  # set below GroupSchema

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

# ── multi-result GI mode keys (the per-mode nested-subgroup layout) ──────────
#: Canonical on-disk GI mode_keys == ``GI1DMode.value`` / ``GI2DMode.value``
#: (reduction/core.py:193-204) == the FrameEvent.mode_key vocabulary.  Hardcoded
#: HERE (not imported) so ``io`` never imports ``reduction`` (which imports io →
#: cycle) and on-disk names are never derived from GUI labels — the
#: ``frame.gi_1d`` / ``gi_2d`` dict keys (``qtotal``/``polar``/``gi2d``…) are
#: GUI/legacy spellings and MUST NOT reach disk.  Frozen forever.
GI_MODE_KEYS_1D = frozenset({"q_total", "q_ip", "q_oop", "exit_angle", "chi_gi"})
GI_MODE_KEYS_2D = frozenset({"qip_qoop", "q_chi", "exit_angles"})

#: mode_key → on-disk NXdata subgroup name.  Identity for GI keys (the enum
#: values are valid HDF5 names) but declared explicitly so the on-disk name is
#: canonical-by-declaration, decoupled from any future enum/label respelling.
#: DEFAULT_MODE_KEY is intentionally absent: the primary/default slot is the
#: top-level group, never a subgroup.
MODE_SUBGROUP_NAMES: "Mapping[str, str]" = MappingProxyType(
    {k: k for k in (GI_MODE_KEYS_1D | GI_MODE_KEYS_2D)}
)
_SUBGROUP_TO_MODE: "Mapping[str, str]" = MappingProxyType(
    {v: k for k, v in MODE_SUBGROUP_NAMES.items()}
)


def mode_subgroup_name(mode_key: str) -> str:
    """On-disk subgroup name for a NON-primary GI ``mode_key``.

    Fail-loud: ``DEFAULT_MODE_KEY`` has no subgroup (it is the top-level slot)
    and an unknown key raises (callers must use a registered GI mode_key)."""
    if mode_key == DEFAULT_MODE_KEY:
        raise ValueError(
            "DEFAULT_MODE_KEY has no subgroup (it lives at the top-level group)"
        )
    try:
        return MODE_SUBGROUP_NAMES[mode_key]
    except KeyError:
        raise ValueError(f"unknown GI mode_key: {mode_key!r}") from None


def subgroup_mode_key(subgroup_name: str) -> "str | None":
    """Inverse: on-disk child-group name → mode_key, or ``None`` if it is not a
    registered GI subgroup (so an unknown on-disk child never becomes a phantom
    mode)."""
    return _SUBGROUP_TO_MODE.get(subgroup_name)


def resolve_mode_path(group_name: str, mode_key: str, primary_mode: str) -> str:
    """Reader rule: ``mode == primary ? top-level : <group>/<subgroup>``."""
    if mode_key == primary_mode or mode_key == DEFAULT_MODE_KEY:
        return group_name
    return f"{group_name}/{mode_subgroup_name(mode_key)}"


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset of the v2 record, as data (Phase 2a).

    Everything here is PERSISTED layout fact: ``name`` and ``dtype`` are
    frozen on disk; ``role``/``row_aligned``/``required`` drive the
    writer, validators, readers, and fixture factory.  ``chunk_style``
    names the writer's chunking strategy (shapes are runtime values):
    ``"rows"`` = (min(N,32), n_q) 1D row blocks, ``"frame"`` =
    (1, n_chi, n_q) one frame per chunk, ``"labels"`` = (64,) label
    blocks, ``None`` = h5py default (contiguous).
    """

    name: str
    dtype: str                       # "float32" | "int64"
    role: str                        # "signal" | "axis" | "row_label" | "error"
    row_aligned: bool
    required: bool = True
    compressed: bool = False         # honors the writer's compression= arg
    chunk_style: str | None = None
    #: where the units attr value comes from at write time
    #: ("radial_unit" | "azimuthal_unit" | a literal like "rad"/"deg").
    units_from: str | None = None


def _integrated_datasets(axes: tuple[str, ...]) -> "Mapping[str, DatasetSpec]":
    """The shared integrated_1d/2d dataset family (2D adds the chi axis)."""
    two_d = len(axes) == 2
    specs = {
        "intensity": DatasetSpec(
            "intensity", "float32", "signal", row_aligned=True,
            compressed=True, chunk_style="frame" if two_d else "rows",
        ),
        "frame_index": DatasetSpec(
            "frame_index", "int64", "row_label", row_aligned=True,
            chunk_style="labels",
        ),
        axes[0]: DatasetSpec(
            axes[0], "float32", "axis", row_aligned=False,
            units_from="radial_unit",
        ),
        "sigma": DatasetSpec(
            "sigma", "float32", "error", row_aligned=True, required=False,
            compressed=True, chunk_style="frame" if two_d else "rows",
        ),
    }
    if two_d:
        specs[axes[1]] = DatasetSpec(
            axes[1], "float32", "axis", row_aligned=False,
            units_from="azimuthal_unit",
        )
    return MappingProxyType(specs)


_GEOMETRY_DATASETS: "Mapping[str, DatasetSpec]" = MappingProxyType({
    "frame_index": DatasetSpec("frame_index", "int64", "row_label",
                               row_aligned=True, chunk_style="labels"),
    # the derived angle series ARE compressed (unlike the label column)
    "rot1": DatasetSpec("rot1", "float32", "signal", row_aligned=True,
                        required=False, compressed=True,
                        chunk_style="labels", units_from="rad"),
    "rot2": DatasetSpec("rot2", "float32", "signal", row_aligned=True,
                        required=False, compressed=True,
                        chunk_style="labels", units_from="rad"),
    "rot3": DatasetSpec("rot3", "float32", "signal", row_aligned=True,
                        required=False, compressed=True,
                        chunk_style="labels", units_from="rad"),
    "incident_angle": DatasetSpec("incident_angle", "float32", "signal",
                                  row_aligned=True, required=False,
                                  compressed=True, chunk_style="labels",
                                  units_from="deg"),
})


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
    #: full per-dataset declarations (2a); row_aligned above stays as the
    #: legacy fast set — test_schema_as_code pins their consistency.
    datasets: Mapping[str, DatasetSpec] = field(
        default_factory=lambda: MappingProxyType({})
    )
    #: static NX attrs stamped at group creation (runtime-valued attrs —
    #: two_d_kind, the monotonic flag — are capability attrs, not here).
    nx_attrs: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )


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
                datasets=_integrated_datasets(("q",)),
                nx_attrs=MappingProxyType({
                    "NX_class": "NXdata",
                    "signal": "intensity",
                    "axes": ("frame_index", "q"),
                }),
            ),
            "integrated_2d": GroupSchema(
                "integrated_2d", axes=("q", "chi"),
                row_aligned=INTEGRATED_ROW_ALIGNED,
                datasets=_integrated_datasets(("q", "chi")),
                nx_attrs=MappingProxyType({
                    "NX_class": "NXdata",
                    "signal": "intensity",
                    "axes": ("frame_index", "chi", "q"),
                }),
            ),
            "per_frame_geometry": GroupSchema(
                "per_frame_geometry",
                row_aligned=frozenset({
                    "frame_index", "rot1", "rot2", "rot3", "incident_angle",
                }),
                datasets=_GEOMETRY_DATASETS,
                nx_attrs=MappingProxyType({"NX_class": "NXcollection"}),
            ),
            # The canonical Diffractometer as a single JSON blob (config_json):
            # the declarative instrument (both adapter views), the fitted
            # DetectorCalibration (PONI + Detector_config + image mount), the
            # preset tag + motor map.  Scan-level (no per-frame rows), so it
            # carries no schema datasets — the blob is written by hand as a
            # vlen-UTF8 string (it is not a numeric row-aligned stack).
            "diffractometer": GroupSchema(
                "diffractometer",
                nx_attrs=MappingProxyType({"NX_class": "NXcollection"}),
            ),
        })
    )

    # -- lookups -----------------------------------------------------------
    def get_dataset(self, group: str, name: str) -> DatasetSpec | None:
        g = self.groups.get(group)
        return g.datasets.get(name) if g is not None else None

    def is_row_aligned(self, group: str, name: str) -> bool:
        ds = self.get_dataset(group, name)
        return bool(ds is not None and ds.row_aligned)


CAPABILITIES = MappingProxyType({
    "source_base": CapabilityAttr(
        SOURCE_BASE_ATTR, "", "attr",
        "N1 portability: relative source/path pointers resolve against "
        "this POSIX project root"),
    "frames_record": CapabilityAttr(
        "frames", "", "group",
        "per-frame record groups (thumbnails, source refs, timestamps)"),
    "per_frame_geometry": CapabilityAttr(
        "per_frame_geometry", "", "group",
        "derived diffractometer rotations + incident angle per frame"),
    "sigma_1d": CapabilityAttr(
        "sigma", "integrated_1d", "dataset", "1D error estimates"),
    "sigma_2d": CapabilityAttr(
        "sigma", "integrated_2d", "dataset", "2D error estimates"),
    "two_d_kind": CapabilityAttr(
        "two_d_kind", "integrated_2d", "attr",
        "explicit GI axis identity (else inferred from units)"),
    "axis_kind_1d": CapabilityAttr(
        "axis_kind", "integrated_1d", "attr",
        "explicit 1D axis identity -- 'azimuthal' for I-vs-chi "
        "(chi_deg/chigi_deg), else 'radial' (inferred from units)"),
    "multi_result_1d": CapabilityAttr(
        MULTI_RESULT_MODES_ATTR, "integrated_1d", "attr",
        "per-GI-mode results: the primary at integrated_1d, others under "
        "integrated_1d/<mode>/ nested NXdata subgroups"),
    "multi_result_2d": CapabilityAttr(
        MULTI_RESULT_MODES_ATTR, "integrated_2d", "attr",
        "per-GI-mode results: the primary at integrated_2d, others under "
        "integrated_2d/<mode>/ nested NXdata subgroups"),
    "diffractometer": CapabilityAttr(
        "diffractometer", "", "group",
        "canonical Diffractometer geometry blob (config_json: both adapter "
        "views + fitted DetectorCalibration + preset + motor map) for offline "
        "stitch/RSM"),
    "ub_matrix": CapabilityAttr(
        "ub_matrix", "sample", "dataset",
        "UB sample orientation matrix (the dataset already round-trips; this "
        "only advertises its presence)"),
})


def detect_capabilities(entry_grp) -> set[str]:
    """Feature-detect the optional v2 capabilities present in an open
    entry group (h5py).  Unknown on-disk extras are ignored; unknown
    registry entries simply absent — additive evolution by construction.
    """
    found: set[str] = set()
    for name, cap in CAPABILITIES.items():
        node = entry_grp
        if cap.location:
            if cap.location not in entry_grp:
                continue
            node = entry_grp[cap.location]
        if cap.kind == "attr":
            if cap.marker in node.attrs:
                found.add(name)
        elif cap.marker in node:
            found.add(name)
    return found


def is_known_schema_name(value) -> bool:
    """True if an ``ssrl_schema`` value names this schema (current or any
    accepted historical spelling)."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value) in ACCEPTED_SCHEMA_NAMES


#: the singleton consumers import.
SCHEMA = ProcessedScanSchema()
