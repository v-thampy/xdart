# -*- coding: utf-8 -*-
"""Schema-as-code for the processed-scan NeXus record (v3).

The single declarative description of the on-disk layout that
``xrd_tools.io.nexus`` writes and the readers consume.  Layout facts —
the schema stamp, which datasets are row-aligned (one leading per-frame
dimension), the axis dataset names, the capability attributes — live HERE
so writers, validators, readers, and row surgery share one source of
truth instead of each re-hard-coding strings.

Version 3 intentionally replaces integrated q/chi dataset names with neutral
axis_1/axis_2 names. Scientific units, mode identities and array orientation
are unchanged. Other persisted keys remain frozen; format changes require
an explicit schema-version boundary, not silent in-place file migration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import h5py
import numpy as np

from xrd_tools.core.frame_view import DEFAULT_MODE_KEY, axis_from_unit

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
    "canonical_gi_mode_key",
    "local_hard_group",
    "local_hard_group_path",
    "local_hard_dataset",
    "read_current_mode_layout",
    "mode_subgroup_name",
    "subgroup_mode_key",
    "resolve_mode_path",
    "DEFAULT_MODE_KEY",
    "SOURCE_BASE_ATTR",
    "THUMBNAIL_LUT_ATTRS",
    "PROCESSED_SCHEMA_NAME",
    "ACCEPTED_SCHEMA_NAMES",
    "PROCESSED_SCHEMA_VERSION",
    "axis_display_metadata",
    "integrated_axis_names",
    "INTEGRATED_ROW_ALIGNED",
    "GroupSchema",
    "ProcessedScanSchema",
    "SCHEMA",
    "REINTEGRATE_SHADOW_SUFFIX",
    "REINTEGRATE_SHADOW_COMPLETE_ATTR",
    "is_complete_reintegration_shadow",
    "resolve_integrated_group",
]


# ── streaming-reintegrate shadow recovery ────────────────────────────────────
#: suffix of the shadow group a streaming reintegrate stages rows into before
#: the atomic swap performed by the canonical NeXus writer.  Shared here
#: so the headless readers can recover an orphan left by a crash mid-swap.
REINTEGRATE_SHADOW_SUFFIX = "__reint"
#: attr the writer stamps on a shadow ONLY once its coverage is validated, right
#: before the swap deletes the canonical group.  An orphan shadow is adopted as
#: the authoritative result ONLY when it carries this marker -- a shadow left by
#: a crash MID-WRITE (still streaming, never validated; e.g. a 2D reintegrate on
#: a 1D-only scan whose canonical 2D never existed) is partial and must NOT be
#: presented as complete.
REINTEGRATE_SHADOW_COMPLETE_ATTR = "reintegrate_shadow_complete"


def is_complete_reintegration_shadow(group) -> bool:
    """Accept only the scalar boolean ``True`` written by the current writer."""
    try:
        attr = group.attrs.get_id(REINTEGRATE_SHADOW_COMPLETE_ATTR)
        if attr.shape != () or attr.dtype != np.dtype(np.bool_):
            return False
        value = group.attrs[REINTEGRATE_SHADOW_COMPLETE_ATTR]
        return isinstance(value, (bool, np.bool_)) and bool(value)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _local_group(parent, name: str):
    """Return one local hard-linked group without following indirection."""
    try:
        if type(parent.get(name, getlink=True)) is not h5py.HardLink:
            return None
        value = parent.get(name)
        return value if isinstance(value, h5py.Group) else None
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None


def local_hard_group(parent, name: str, *, role: str | None = None):
    """Return one optional local hard-linked group, refusing indirection.

    ``None`` means the component is absent.  A present soft/external link, or
    a hard-linked non-group, is malformed current generated output and raises.
    This helper is deliberately for processed-result ownership only; raw
    detector links (including Eiger ``ExternalLink`` image segments) are not
    traversed through it.
    """
    shown = role or f"{getattr(parent, 'name', '<group>')}/{name}"
    if type(name) is not str or not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"{shown} must be one exact group component")
    try:
        link = parent.get(name, getlink=True)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"cannot inspect generated result group {shown}") from error
    if link is None:
        return None
    if type(link) is not h5py.HardLink:
        raise ValueError(f"generated result group {shown} is not local hard storage")
    try:
        value = parent.get(name)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"cannot open generated result group {shown}") from error
    if not isinstance(value, h5py.Group):
        raise ValueError(f"generated result node {shown} is not a group")
    return value


def local_hard_group_path(parent, path: str, *, role: str | None = None):
    """Resolve an optional generated-result path one local-hard component at a time."""
    if type(path) is not str:
        raise ValueError("generated result path must be exact text")
    parts = tuple(path.split("/"))
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise ValueError(f"invalid generated result path {path!r}")
    current = parent
    for index, part in enumerate(parts):
        current = local_hard_group(
            current,
            part,
            role=(role if index == len(parts) - 1 else "/".join(parts[: index + 1])),
        )
        if current is None:
            return None
    return current


def local_hard_dataset(parent, name: str, *, role: str | None = None):
    """Return one optional local hard-linked generated-result dataset."""
    shown = role or f"{getattr(parent, 'name', '<group>')}/{name}"
    if type(name) is not str or not name or "/" in name or name in {".", ".."}:
        raise ValueError(f"{shown} must be one exact dataset component")
    try:
        link = parent.get(name, getlink=True)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"cannot inspect generated result dataset {shown}") from error
    if link is None:
        return None
    if type(link) is not h5py.HardLink:
        raise ValueError(f"generated result dataset {shown} is not local hard storage")
    value = parent.get(name)
    if not isinstance(value, h5py.Dataset) or value.is_virtual or value.external:
        raise ValueError(f"generated result node {shown} is not a local dataset")
    return value


def resolve_integrated_group(entry_grp, group_name: str):
    """Resolve a canonical integrated group, recovering from a crash mid-swap.

    Returns ``(group, adopted)``: the canonical ``group_name`` group when
    present; else, when only its ``<group_name>__reint`` orphan shadow survives
    AND that shadow was marked COMPLETE by the writer (a crash precisely between
    the swap's delete-canonical and move-shadow), the shadow is adopted
    READ-ONLY (the file is not mutated) so consumers "open sanely" on the
    complete result.  An UNMARKED orphan (a reintegrate crashed mid-write -- its
    rows are partial) is ignored, never presented as the result; else
    ``(None, False)``.  Ordinary append writers refuse this recovery-only state;
    promotion requires an explicit recovery operation.
    """
    try:
        canonical_link = entry_grp.get(group_name, getlink=True)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"cannot inspect generated result group {group_name}"
        ) from error
    if canonical_link is not None:
        # A shadow is recovery storage only when the canonical namespace slot
        # is genuinely absent.  Never let an indirect or wrong-kind canonical
        # node disappear behind an otherwise valid completed shadow.
        return local_hard_group(
            entry_grp,
            group_name,
            role=group_name,
        ), False
    shadow = _local_group(
        entry_grp,
        f"{group_name}{REINTEGRATE_SHADOW_SUFFIX}",
    )
    if shadow is None:
        return None, False
    if is_complete_reintegration_shadow(shadow):
        logger.warning(
            "Adopting completed orphan reintegration shadow %s%s as %s "
            "(read-only; the file crashed mid-swap and requires explicit "
            "recovery).", group_name, REINTEGRATE_SHADOW_SUFFIX, group_name,
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

#: Entry attribute holding the ROOT FAMILY this artifact belongs to
#: (ADR-0010, carried as 491b9413).  Stable public slots are named
#: `<family><slot>.nexus`, and a later operation must CONSUME this value rather
#: than derive one from the file's stem -- deriving from `sample_int2d.nexus`
#: yields `sample_int2d_average.nexus`, the chained name the ADR forbids, and
#: stripping the slot to recover the root is an explicit stop condition.
ARTIFACT_FAMILY_ATTR = "artifact_family_v1"
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
#: Exact schema identities accepted by the forward-only reader.
ACCEPTED_SCHEMA_NAMES = (PROCESSED_SCHEMA_NAME,)
#: New writes use v3; qualified v2 records remain readable, never appendable.
PROCESSED_SCHEMA_VERSION = 3


def integrated_axis_names(version: int) -> tuple[str, str]:
    """Physical integrated coordinates for the two supported record layouts."""
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, (int, np.integer)):
        raise ValueError("unsupported processed schema version")
    if version == 2:
        return "q", "chi"
    if version == PROCESSED_SCHEMA_VERSION:
        return "axis_1", "axis_2"
    raise ValueError("unsupported processed schema version")


def axis_display_metadata(unit: str) -> dict[str, str]:
    """Scientific labels for neutral axes, computed only at dataset creation.

    Keep the existing machine-readable unit token intact. NeXpy and silx use
    long_name verbatim, so the human label includes the physical unit too.
    """
    physical_unit = unit.rsplit("_", 1)[-1]
    display_unit = {
        "A^-1": "Å⁻¹", "nm^-1": "nm⁻¹", "deg": "°", "rad": "rad",
        "degrees": "°", "1/angstrom": "Å⁻¹", "angstrom^-1": "Å⁻¹",
    }.get(physical_unit, physical_unit)
    return {
        "units": unit,
        "long_name": f"{axis_from_unit(unit).label} ({display_unit})",
    }

# ── row-aligned datasets ─────────────────────────────────────────────────────

#: datasets inside ``integrated_1d``/``integrated_2d`` whose LEADING dimension
#: is the per-frame row — exactly these are sliced/rebuilt by row surgery
#: (``drop_integrated_rows``) and grown by the appenders.  Axis datasets
#: (``axis_1``/``axis_2``) are shared across rows and are NOT in this set.
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


def canonical_gi_mode_key(
    value,
    dimension: str,
    *,
    allow_default: bool = True,
) -> str:
    """Return one exact canonical current mode key.

    Enum-like producer values are accepted only through their exact ``.value``
    string.  Arbitrary values are never stringified, and historical GUI keys
    such as ``qtotal`` are intentionally not translated.
    """
    if dimension not in {"1d", "2d"}:
        raise ValueError(f"unknown result dimension {dimension!r}")
    raw = value if type(value) is str else getattr(value, "value", value)
    if type(raw) is not str:
        raise ValueError(f"{dimension} mode key requires exact text")
    allowed = GI_MODE_KEYS_1D if dimension == "1d" else GI_MODE_KEYS_2D
    if raw == DEFAULT_MODE_KEY:
        if allow_default:
            return raw
        raise ValueError(f"named {dimension} result mode cannot be default")
    if raw not in allowed:
        raise ValueError(f"unknown canonical {dimension} mode key {raw!r}")
    return raw


def _exact_text_attr(group: h5py.Group, name: str) -> str:
    try:
        attr = group.attrs.get_id(name)
        if attr.shape != ():
            raise ValueError
        value = group.attrs[name]
        if isinstance(value, (bytes, np.bytes_)):
            value = bytes(value).decode("utf-8", errors="strict")
        if type(value) is not str or not value or len(value.encode("utf-8")) > 256:
            raise ValueError
        return value
    except (KeyError, OSError, RuntimeError, TypeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{group.name}@{name} requires one bounded text scalar") from error


def _exact_text_vector_attr(group: h5py.Group, name: str) -> tuple[str, ...]:
    try:
        attr = group.attrs.get_id(name)
        value = group.attrs[name]
        if attr.shape != np.asarray(value).shape or len(attr.shape) != 1:
            raise ValueError
        raw = np.asarray(value).tolist()
        if not raw or len(raw) > len(GI_MODE_KEYS_1D | GI_MODE_KEYS_2D):
            raise ValueError
        decoded: list[str] = []
        for item in raw:
            if isinstance(item, (bytes, np.bytes_)):
                item = bytes(item).decode("utf-8", errors="strict")
            if type(item) is not str or not item or len(item.encode("utf-8")) > 256:
                raise ValueError
            decoded.append(item)
        return tuple(decoded)
    except (KeyError, OSError, RuntimeError, TypeError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"{group.name}@{name} requires a bounded 1-D text vector") from error


def read_current_mode_layout(
    group: h5py.Group,
    dimension: str,
) -> tuple[str, tuple[str, ...], tuple[tuple[str, h5py.Group], ...]]:
    """Authenticate the complete current mode inventory for one result stack.

    Returns ``(primary, ordered_modes, ordered_mode_groups)``.  An attr-free
    stack is the standard ``default`` layout.  Named GI output must carry both
    attrs, list the primary first, and own every declared child as a local hard
    group; undeclared/unknown group siblings and indirect links are refused.
    """
    if not isinstance(group, h5py.Group):
        raise ValueError("current result layout requires an HDF5 group")
    allowed = GI_MODE_KEYS_1D if dimension == "1d" else GI_MODE_KEYS_2D
    has_primary = PRIMARY_MODE_ATTR in group.attrs
    has_modes = MULTI_RESULT_MODES_ATTR in group.attrs
    if has_primary != has_modes:
        raise ValueError(f"{group.name} has an incomplete current mode inventory")
    if has_primary:
        primary = canonical_gi_mode_key(
            _exact_text_attr(group, PRIMARY_MODE_ATTR),
            dimension,
            allow_default=False,
        )
        modes = _exact_text_vector_attr(group, MULTI_RESULT_MODES_ATTR)
        if (
            modes[0] != primary
            or len(modes) != len(set(modes))
            or any(mode not in allowed for mode in modes)
        ):
            raise ValueError(f"{group.name} has a malformed current mode inventory")
    else:
        primary = DEFAULT_MODE_KEY
        modes = (DEFAULT_MODE_KEY,)

    expected_children = {
        mode_subgroup_name(mode): mode
        for mode in modes
        if mode != primary
    }
    pairs: list[tuple[str, h5py.Group]] = [(primary, group)]
    for name in group:
        try:
            link = group.get(name, getlink=True)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise ValueError(f"cannot inspect generated result node {group.name}/{name}") from error
        if type(link) is not h5py.HardLink:
            raise ValueError(
                f"generated result node {group.name}/{name} is not local hard storage"
            )
        node = group.get(name)
        if isinstance(node, h5py.Group) and name not in expected_children:
            raise ValueError(f"unrequested generated result sibling {group.name}/{name}")
    for child_name, mode in expected_children.items():
        child = local_hard_group(
            group,
            child_name,
            role=f"{group.name}/{child_name}",
        )
        if child is None:
            raise ValueError(f"missing declared result group {group.name}/{child_name}")
        if any(child.id == owned.id for _key, owned in pairs):
            raise ValueError(
                f"declared result group {group.name}/{child_name} aliases another mode"
            )
        pairs.append((mode, child))
    return primary, modes, tuple(pairs)


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
    """The shared integrated_1d/2d dataset family (2D adds axis_2)."""
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


def _stitched_datasets(axes: tuple[str, ...]) -> "Mapping[str, DatasetSpec]":
    """The stitched_1d/2d dataset family — a single MERGED pattern (no per-frame
    leading dim, so nothing is row-aligned).  The ``provenance_json`` blob (the
    StitchPlan + CorrectionStack) is written by hand as a vlen-UTF8 string, like
    the diffractometer ``config_json`` — not declared here (additive extra)."""
    two_d = len(axes) == 2
    specs = {
        "intensity": DatasetSpec(
            "intensity", "float32", "signal", row_aligned=False, compressed=True),
        axes[0]: DatasetSpec(
            axes[0], "float32", "axis", row_aligned=False, units_from="radial_unit"),
    }
    if two_d:
        specs[axes[1]] = DatasetSpec(
            axes[1], "float32", "axis", row_aligned=False,
            units_from="azimuthal_unit")
    specs["sigma"] = DatasetSpec(
        "sigma", "float32", "error", row_aligned=False, required=False,
        compressed=True)
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
    #: are stored (axis_2, axis_1) = (azimuthal, radial); see the 2D-orientation
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
    """The current processed-scan record, as data."""

    name: str = PROCESSED_SCHEMA_NAME
    accepted_names: tuple[str, ...] = ACCEPTED_SCHEMA_NAMES
    version: int = PROCESSED_SCHEMA_VERSION
    name_attr: str = SCHEMA_NAME_ATTR
    version_attr: str = SCHEMA_VERSION_ATTR
    groups: Mapping[str, GroupSchema] = field(
        default_factory=lambda: MappingProxyType({
            "integrated_1d": GroupSchema(
                "integrated_1d", axes=("axis_1",),
                row_aligned=INTEGRATED_ROW_ALIGNED,
                datasets=_integrated_datasets(("axis_1",)),
                nx_attrs=MappingProxyType({
                    "NX_class": "NXdata",
                    "signal": "intensity",
                    "axes": ("frame_index", "axis_1"),
                }),
            ),
            "integrated_2d": GroupSchema(
                "integrated_2d", axes=("axis_1", "axis_2"),
                row_aligned=INTEGRATED_ROW_ALIGNED,
                datasets=_integrated_datasets(("axis_1", "axis_2")),
                nx_attrs=MappingProxyType({
                    "NX_class": "NXdata",
                    "signal": "intensity",
                    "axes": ("frame_index", "axis_2", "axis_1"),
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
            # Scan-level MERGED stitch outputs (one pattern, not a per-frame
            # stack): the offline stitch/RSM result + its provenance_json blob
            # (the StitchPlan + applied CorrectionStack). Capability-gated;
            # written by write_stitched, read by read_stitched.
            "stitched_1d": GroupSchema(
                "stitched_1d", axes=("q",),
                datasets=_stitched_datasets(("q",)),
                nx_attrs=MappingProxyType({
                    "NX_class": "NXdata",
                    "signal": "intensity",
                    "axes": ("q",),
                }),
            ),
            "stitched_2d": GroupSchema(
                "stitched_2d", axes=("q", "chi"),
                datasets=_stitched_datasets(("q", "chi")),
                nx_attrs=MappingProxyType({
                    "NX_class": "NXdata",
                    "signal": "intensity",
                    "axes": ("q", "chi"),
                }),
            ),
            # The gridded RSM volume (one 3D H-K-L grid; not a per-frame stack):
            # h/k/l axes + the 3D intensity + a provenance_json blob. Scan-level,
            # capability-gated; written by write_rsm, read by read_rsm.
            "rsm": GroupSchema(
                "rsm", axes=("h", "k", "l"),
                datasets=MappingProxyType({
                    "intensity": DatasetSpec(
                        "intensity", "float32", "signal", row_aligned=False,
                        compressed=True),
                    "h": DatasetSpec("h", "float32", "axis", row_aligned=False),
                    "k": DatasetSpec("k", "float32", "axis", row_aligned=False),
                    "l": DatasetSpec("l", "float32", "axis", row_aligned=False),
                }),
                nx_attrs=MappingProxyType({
                    "NX_class": "NXdata",
                    "signal": "intensity",
                    "axes": ("h", "k", "l"),
                }),
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
    "average_finite_counts": CapabilityAttr(
        "finite_counts", "frames/frame_0001", "dataset",
        "Average Scan per-pixel finite-contributor count map"),
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
    "stitched_1d": CapabilityAttr(
        "stitched_1d", "", "group",
        "merged 1D stitch pattern (offline stitch/RSM) + provenance_json "
        "(StitchPlan + applied CorrectionStack)"),
    "stitched_2d": CapabilityAttr(
        "stitched_2d", "", "group",
        "merged 2D (q,χ) stitch pattern + provenance_json (StitchPlan + "
        "applied CorrectionStack)"),
    "rsm": CapabilityAttr(
        "rsm", "", "group",
        "gridded reciprocal-space-map volume (h/k/l NXdata) + provenance_json "
        "(RSMPlan + applied CorrectionStack)"),
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

# Stitch uses a separate physical layout while retaining logical q/chi APIs.
# The preceding layouts stay available for exact old artifact/embedded reads.
STITCH_NEUTRAL_AXIS_NAMES = MappingProxyType({"q": "axis_1", "chi": "axis_2"})
STITCH_NEUTRAL_GROUPS = MappingProxyType({
    name: GroupSchema(
        name, axes=axes, datasets=_stitched_datasets(axes),
        nx_attrs=MappingProxyType({
            "NX_class": "NXdata", "signal": "intensity", "axes": axes,
        }),
    )
    for name, axes in (
        ("stitched_1d", ("axis_1",)),
        ("stitched_2d", ("axis_1", "axis_2")),
    )
})
