# -*- coding: utf-8 -*-
"""Shared, Qt-free control readiness and render-state contracts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import logging
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from xrd_tools.session.run_configuration import FrozenRunConfiguration

logger = logging.getLogger(__name__)


class Tool(str, Enum):
    INT_1D = "int_1d"
    INT_2D = "int_2d"
    IMAGE_VIEWER = "image_viewer"
    XYE_VIEWER = "xye_viewer"
    NEXUS_VIEWER = "nexus_viewer"
    STITCH = "stitch"
    RSM = "rsm"


class MeasMode(str, Enum):
    STANDARD = "standard"
    GI = "gi"


_UNSET = object()


_PROCESSING_COMPARED_FIELDS: tuple[tuple[str, str], ...] = (
    ("mode", "mode"),
    ("axis_1d", "1D axis"),
    ("axis_2d", "2D axis"),
    ("unit_1d", "1D unit"),
    ("unit_2d", "2D unit"),
    ("npt_1d", "1D points"),
    ("npt_oop_1d", "1D oop points"),
    ("npt_rad_2d", "2D radial points"),
    ("npt_azim_2d", "2D azimuth points"),
    ("radial_range_1d", "1D radial range"),
    ("azimuth_range_1d", "1D azimuth range"),
    ("radial_range_2d", "2D radial range"),
    ("azimuth_range_2d", "2D azimuth range"),
    # S-3: value-affecting, GRID-PRESERVING params.  These change the written
    # numbers while leaving the axis/npt/range grid identical, so they pass both
    # the modal and the axis backstop -> mixed provenance under a /entry/reduction
    # that claims the first run's config.  Compared BACKWARD-TOLERANTLY: a field
    # absent from a pre-upgrade stored config is _UNSET and skipped (no
    # false-positive modal on every existing processed artifact).
    ("chi_offset_1d", "1D chi offset"),
    ("chi_offset_2d", "2D chi offset"),
    ("monitor_1d", "1D monitor"),
    ("monitor_2d", "2D monitor"),
    ("polarization_1d", "1D polarization"),
    ("polarization_2d", "2D polarization"),
    ("error_model_1d", "1D error model"),
    ("error_model_2d", "2D error model"),
    ("gi_incidence", "GI incidence angle"),
    ("background_policy", "Background"),
)


@dataclass(frozen=True, slots=True)
class ProcessingConfigSignature:
    """Data-affecting integration config used by Append run gates."""

    mode: MeasMode
    axis_1d: str
    axis_2d: str
    unit_1d: str
    unit_2d: str
    npt_1d: int | None
    npt_oop_1d: int | None
    npt_rad_2d: int | None
    npt_azim_2d: int | None
    radial_range_1d: object
    azimuth_range_1d: object
    radial_range_2d: object
    azimuth_range_2d: object
    # S-3 value-affecting/grid-preserving.  Default _UNSET == "not present in this
    # config"; the comparison skips any field that is _UNSET on either side, so a
    # pre-upgrade stored config (missing these keys) never triggers a modal.
    chi_offset_1d: object = _UNSET
    chi_offset_2d: object = _UNSET
    monitor_1d: object = _UNSET
    monitor_2d: object = _UNSET
    polarization_1d: object = _UNSET
    polarization_2d: object = _UNSET
    error_model_1d: object = _UNSET
    error_model_2d: object = _UNSET
    gi_incidence: object = _UNSET
    background_policy: tuple[tuple[str, object], ...] | None = None

    @property
    def display_mode(self) -> str:
        return "Grazing" if self.mode == MeasMode.GI else "Standard"

    def compared_items(self) -> tuple[tuple[str, object], ...]:
        return tuple(
            (field_name, getattr(self, attr))
            for attr, field_name in _PROCESSING_COMPARED_FIELDS
        )


@dataclass(frozen=True, slots=True)
class AppendConfigCheck:
    ok: bool
    reason: str = ""
    compared_fields: tuple[str, ...] = ()
    mismatched_fields: tuple[str, ...] = ()
    processed_label: str = ""
    current_label: str = ""


class AppendConfigMismatchError(RuntimeError):
    """An Append target's stored config differs from the current run config.

    Raised by run paths (e.g. the xdart image wrangler's ``initialize_scan``)
    when :func:`append_config_mismatch_check` fails after the Run-click modal
    can no longer intervene — a mid-run settings change, or a later
    auto-discovered scan of a directory run.  Subclasses ``RuntimeError`` so
    pre-existing broad handlers keep working, while run loops can catch THIS
    mismatch specifically and stop cleanly instead of crashing the worker
    thread.  Carries the :class:`AppendConfigCheck` (``reason``,
    ``mismatched_fields``, ``processed_label``/``current_label``) so the GUI
    can name what changed without re-deriving the comparison.  The append
    target itself is untouched — the guard preserved it.
    """

    def __init__(self, message: str, check: AppendConfigCheck):
        super().__init__(message)
        self.check = check


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}

def _background_policy(value: object):
    if value is None: return None
    from xrd_tools.reduction.background import FrameBackgroundPlan
    plan = value if type(value) is FrameBackgroundPlan else FrameBackgroundPlan.from_mapping(dict(value))
    return None if plan.mode == "None" else tuple(sorted(plan.to_mapping().items()))


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("", "none", "null")
    return False


def _first_value(mapping: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in mapping and not _is_empty_value(mapping[key]):
            return mapping[key]
    return default


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int | None:
    if _is_empty_value(value):
        return default
    return _int_or_none(value)


def _bool_or_none(value: Any) -> bool | None:
    if _is_empty_value(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("1", "true", "t", "yes", "y", "on", "gi", "grazing"):
        return True
    if text in ("0", "false", "f", "no", "n", "off", "standard"):
        return False
    return None


def _text_or_default(value: Any, default: str) -> str:
    if _is_empty_value(value):
        return default
    return str(value).strip()


def _range_or_none(value: Any) -> object:
    if _is_empty_value(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return text
    try:
        if len(value) != 2:  # type: ignore[arg-type]
            return repr(value)
        lo, hi = value  # type: ignore[misc]
    except TypeError:
        return repr(value)
    if lo is None or hi is None:
        return None
    try:
        return (round(float(lo), 12), round(float(hi), 12))
    except (TypeError, ValueError):
        return (str(lo), str(hi))


def _num_or_unset(mapping: Mapping[str, Any], keys: Sequence[str]) -> object:
    """S-3: a numeric config value, or ``_UNSET`` when NONE of the keys is
    present (so a pre-upgrade config that never stored the field is skipped)."""
    for key in keys:
        if key in mapping:
            value = mapping[key]
            if value is None or value == "":
                return None
            try:
                return round(float(value), 12)
            except (TypeError, ValueError):
                return str(value)
    return _UNSET


def _str_or_unset(mapping: Mapping[str, Any], keys: Sequence[str]) -> object:
    """S-3: a string-valued config field, or ``_UNSET`` when absent."""
    for key in keys:
        if key in mapping:
            value = mapping[key]
            return None if value is None else str(value).strip().lower()
    return _UNSET


def _standard_axis_from_unit(unit: object, *, dim: str) -> str:
    text = str(unit or "").lower()
    if "chi" in text and dim == "1d":
        return "chi"
    if "2th" in text or "2θ" in text:
        return "2theta" if dim == "1d" else "2theta-chi"
    return "q" if dim == "1d" else "q-chi"


def _config_indicates_gi(
    bai_1d_args: Mapping[str, Any],
    bai_2d_args: Mapping[str, Any],
    gi_config: Mapping[str, Any],
) -> bool:
    return (
        bool(gi_config)
        or "gi_mode_1d" in bai_1d_args
        or "gi_mode_2d" in bai_2d_args
        or "npt_oop" in bai_1d_args
        or "npt_oop" in bai_2d_args
    )

def processing_config_from_args(
    bai_1d_args: Mapping[str, Any] | None,
    bai_2d_args: Mapping[str, Any] | None,
    *,
    gi_enabled: bool | None = None,
    gi_config: Mapping[str, Any] | None = None,
    background_policy: object = None,
) -> ProcessingConfigSignature:
    """Return the data-shape/axis signature for an integration setup."""

    a1 = _mapping(bai_1d_args)
    a2 = _mapping(bai_2d_args)
    gic = _mapping(gi_config)
    is_gi = (
        _config_indicates_gi(a1, a2, gic)
        if gi_enabled is None
        else bool(gi_enabled)
    )
    mode = MeasMode.GI if is_gi else MeasMode.STANDARD
    unit_1d = _text_or_default(a1.get("unit"), "q_A^-1")
    unit_2d = _text_or_default(a2.get("unit"), "q_A^-1")
    if is_gi:
        axis_1d = _text_or_default(
            a1.get("gi_mode_1d")
            or gic.get("gi_mode_1d")
            or None,
            "q_total",
        )
        axis_2d = _text_or_default(
            a2.get("gi_mode_2d")
            or gic.get("gi_mode_2d")
            or None,
            "qip_qoop",
        )
    else:
        axis_1d = _standard_axis_from_unit(unit_1d, dim="1d")
        axis_2d = _standard_axis_from_unit(unit_2d, dim="2d")
    return ProcessingConfigSignature(
        mode=mode,
        axis_1d=axis_1d,
        axis_2d=axis_2d,
        unit_1d=unit_1d,
        unit_2d=unit_2d,
        npt_1d=_int_or_none(
            _first_value(a1, ("npt", "numpoints", "npt_rad"), 3000)
        ),
        npt_oop_1d=_int_or_none(a1.get("npt_oop")),
        npt_rad_2d=_int_or_none(_first_value(a2, ("npt_rad", "npt"), 500)),
        npt_azim_2d=_int_or_default(a2.get("npt_azim"), 500),
        radial_range_1d=_range_or_none(a1.get("radial_range")),
        azimuth_range_1d=_range_or_none(a1.get("azimuth_range")),
        radial_range_2d=_range_or_none(a2.get("radial_range")),
        azimuth_range_2d=_range_or_none(a2.get("azimuth_range")),
        # chi_offset is INERT for GI (S-4 zeroes the GI azimuth_offset; GI chi
        # goes to FiberIntegrator's own convention), so comparing it on a GI scan
        # trips a FALSE Append modal for a change that does not alter written GI
        # data.  Leave it unknown for GI; standard mode still compares it (it does
        # move the written chi axis).  [review follow-up]
        chi_offset_1d=(_UNSET if is_gi else _num_or_unset(a1, ("chi_offset",))),
        chi_offset_2d=(_UNSET if is_gi
                       else _num_or_unset(a2, ("azimuth_offset", "chi_offset"))),
        monitor_1d=_str_or_unset(a1, ("monitor",)),
        monitor_2d=_str_or_unset(a2, ("monitor",)),
        polarization_1d=_num_or_unset(a1, ("polarization_factor", "polarization")),
        polarization_2d=_num_or_unset(a2, ("polarization_factor", "polarization")),
        error_model_1d=_str_or_unset(a1, ("error_model",)),
        error_model_2d=_str_or_unset(a2, ("error_model",)),
        gi_incidence=_num_or_unset(
            gic, ("th_val", "incidence", "incident_angle", "incidence_angle")),
        background_policy=_background_policy(background_policy),
    )


def processing_config_from_mapping(
    value: Mapping[str, Any] | ProcessingConfigSignature | None,
) -> ProcessingConfigSignature | None:
    if value is None:
        return None
    if isinstance(value, ProcessingConfigSignature):
        return value
    config = _mapping(value)
    if "config" in config and "bai_1d_args" not in config:
        config = _mapping(config.get("config"))
    gi_marker = config.get("gi", _UNSET)
    gi_enabled = None if gi_marker is _UNSET else _bool_or_none(gi_marker)
    return processing_config_from_args(
        _mapping(config.get("bai_1d_args")),
        _mapping(config.get("bai_2d_args")),
        gi_enabled=gi_enabled,
        gi_config=_mapping(config.get("gi_config")),
        background_policy=config.get("background"),
    )


def processing_config_from_scan(
    scan: Any,
    *,
    prefer_stored: bool = False,
) -> ProcessingConfigSignature | None:
    """Build an Append/readiness signature from a scan-like object."""

    if scan is None:
        return None
    if prefer_stored:
        for attr in ("reduction_config", "_display_reduction_config"):
            stored = getattr(scan, attr, None)
            if isinstance(stored, Mapping) and stored:
                return processing_config_from_mapping(stored)
        return None
    return processing_config_from_args(
        _mapping(getattr(scan, "bai_1d_args", {}) or {}),
        _mapping(getattr(scan, "bai_2d_args", {}) or {}),
        gi_enabled=bool(getattr(scan, "gi", False)),
        gi_config=_mapping(getattr(scan, "gi_config", {}) or {}),
        background_policy=getattr(getattr(scan, "run_configuration", None), "background", getattr(scan, "background", None)),
    )


def _format_processing_config_value(attr: str, value: object) -> str:
    if value is _UNSET:
        return "not recorded"
    if value is None:
        return "Auto" if "range" in attr else "not set"
    if attr == "mode":
        try:
            return "Grazing" if MeasMode(value) == MeasMode.GI else "Standard"
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, tuple) and len(value) == 2:
        def _part(item: object) -> str:
            return f"{item:g}" if isinstance(item, (int, float)) else str(item)
        return f"{_part(value[0])} to {_part(value[1])}"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def append_config_difference_lines(
    processed_config: Mapping[str, Any] | ProcessingConfigSignature | None,
    current_config: Mapping[str, Any] | ProcessingConfigSignature | None,
    mismatched_fields: Sequence[str] = (),
) -> tuple[str, ...]:
    """Describe each Append mismatch with its existing and current values."""
    processed = processing_config_from_mapping(processed_config)
    current = processing_config_from_mapping(current_config)
    if processed is None or current is None:
        return ()
    requested = set(mismatched_fields)
    lines = []
    for attr, label in _PROCESSING_COMPARED_FIELDS:
        if requested and label not in requested:
            continue
        old = getattr(processed, attr)
        new = getattr(current, attr)
        if old is _UNSET or new is _UNSET or old == new:
            continue
        lines.append(
            f"{label}: existing {_format_processing_config_value(attr, old)}; "
            f"current {_format_processing_config_value(attr, new)}"
        )
    return tuple(lines)


def append_config_mismatch_check(
    write_mode: object,
    processed_config: Mapping[str, Any] | ProcessingConfigSignature | None,
    current_config: Mapping[str, Any] | ProcessingConfigSignature | None,
) -> AppendConfigCheck:
    """Return the Append config comparison result when configs differ."""

    compared_fields = tuple(label for _attr, label in _PROCESSING_COMPARED_FIELDS)
    if str(write_mode or "").strip().lower() != "append":
        return AppendConfigCheck(ok=True, compared_fields=compared_fields)
    processed = processing_config_from_mapping(processed_config)
    current = processing_config_from_mapping(current_config)
    if processed is None or current is None:
        return AppendConfigCheck(ok=True, compared_fields=compared_fields)

    # S-3 backward-tolerant: skip any field that is _UNSET on EITHER side (a
    # pre-upgrade stored config that never recorded it, or a run that doesn't set
    # it) -- comparing it would raise a false modal on every existing processed artifact.
    mismatches = tuple(
        label for attr, label in _PROCESSING_COMPARED_FIELDS
        if getattr(processed, attr) is not _UNSET
        and getattr(current, attr) is not _UNSET
        and getattr(processed, attr) != getattr(current, attr)
    )
    if not mismatches:
        return AppendConfigCheck(
            ok=True,
            compared_fields=compared_fields,
            processed_label=processed.display_mode,
            current_label=current.display_mode,
        )

    logger.debug(
        "append config mismatch: processed=%s current=%s checked=%s "
        "differences=%s",
        processed.compared_items(),
        current.compared_items(),
        compared_fields,
        mismatches,
    )
    # Name WHAT changed: display_mode only carries the GI/Standard axis, so a
    # same-mode mismatch (e.g. a mid-run Int 1D -> Int 2D settings change) used
    # to read "processed: Standard · current: Standard" — true but useless.
    reason = (
        f"processed: {processed.display_mode} · current: {current.display_mode} "
        f"(differs: {', '.join(mismatches)}) "
        "— switch write mode to Replace, or revert settings"
    )
    return AppendConfigCheck(
        ok=False,
        reason=reason,
        compared_fields=compared_fields,
        mismatched_fields=mismatches,
        processed_label=processed.display_mode,
        current_label=current.display_mode,
    )


class ProcessingPage(str, Enum):
    INT_1D = "int_1d"
    INT_2D = "int_2d"
    VIEWER = "viewer"


class SectionId(str, Enum):
    PROJECT = "project"
    SOURCE = "source"
    EXPERIMENT = "experiment"
    PROCESSING = "processing"


class ControlAction(str, Enum):
    """Intent emitted by the Controls projection.

    The Qt widget renders these as buttons, while the Scattering Workspace owns
    command routing. Keeping the action list pure prevents the renderer from
    acquiring execution or scientific-state ownership.
    """

    CALIBRATE = "calibrate"
    MAKE_MASK = "make_mask"
    REINTEGRATE_1D = "reintegrate_1d"
    REINTEGRATE_2D = "reintegrate_2d"
    ADVANCED_PROCESSING = "advanced_processing"
    #: Another page's own Processing actions on the same row (Stitch: check the
    #: inputs and list the outputs; retry a finalization that did not complete).
    PREVIEW = "preview"
    RETRY = "retry"
    PLOT_METADATA = "plot_metadata"
    #: Presentation only: shared help buttons route directly to their context
    #: owner, never through ControlsProjection or a scientific command.
    HELP = "help"


@dataclass(frozen=True, slots=True)
class SourceCaps:
    has_frames: bool = False
    has_raw: bool = False
    raw_reachable: bool = False
    has_metadata: bool = False
    has_motors: bool = False
    has_energy: bool = False
    has_geometry: bool = False
    has_psi_metadata: bool = False
    #: O-1b R4A-6.  A valid configured source whose discovery is DEFERRED to the
    #: run (a lazy container directory: nobody has walked or opened anything
    #: yet).  It is the typed distinction between "proved to have nothing" and
    #: "not looked yet", and it exists so the capability fields above can stay
    #: strictly evidence-based while Run eligibility keys on configured-intent
    #: validity.  A caller must never read this as evidence OF frames -- it is
    #: the explicit statement that there is no evidence either way.  Defaults
    #: false, so every existing producer keeps its exact meaning.
    discovery_deferred: bool = False


@dataclass(frozen=True, slots=True)
class ResultCaps:
    has_1d: bool = False
    has_2d: bool = False
    has_raw: bool = False
    raw_reachable: bool = False
    has_scan_metadata: bool = False
    has_rsm: bool = False
    has_phase_result: bool = False
    has_psi_metadata: bool = False
    # None means "unknown / let the dialog show its own friendly dependency
    # hint".  A concrete frozenset means the launcher can gate on it.
    available_optional_deps: frozenset[str] | None = None

    def optional_dep_known(self) -> bool:
        return self.available_optional_deps is not None

    def has_optional_dep(self, name: str) -> bool:
        deps = self.available_optional_deps
        return deps is None or name in deps or "all" in deps


@dataclass(frozen=True, slots=True)
class ControlActionSpec:
    action: ControlAction
    label: str
    section: SectionId
    enabled: bool = True
    reason: str = ""
    production_ready: bool = True


class ControlFieldKind(str, Enum):
    LINE = "line"
    BOOL = "bool"
    COMBO = "combo"
    #: A whole number typed into a line: the text is kept as typed and refused, never reverted, when it is
    #: no whole number within the field's bounds (:func:`integer_field_refusal`).
    INT = "int"


def integer_field_refusal(text, *, minimum: int | None = None, maximum: int | None = None,
                          allow_blank: bool = False) -> str:
    """Why ``text`` is not a whole number within the bounds, in a few words ("" when it is one).

    The one rule an integer field's row shows and a page's readiness names. It never replaces the
    typed value (Codex's RSM stage-1 review F1: a refused value must not leave an older one in force).
    ``allow_blank``: an empty field is valid (the field's automatic value, as a blank Last frame).
    """
    value = "" if text is None else str(text).strip()
    if not value:
        return "" if allow_blank else "a whole number is needed"
    try:
        number = int(value)
    except ValueError:
        return f"{value!r} is not a whole number"
    if minimum is not None and number < minimum:
        return f"must be at least {minimum}"
    if maximum is not None and number > maximum:
        return f"must be at most {maximum}"
    return ""


@dataclass(frozen=True, slots=True)
class ControlFormField:
    """One immutable Controls field projected for the shared Qt renderer.

    ``path`` is a presentation-neutral field identifier.  A page owns the
    value and edit semantics; this object carries only the rendered snapshot.
    """

    section: SectionId
    label: str
    path: tuple[str, ...]
    value: object = ""
    kind: ControlFieldKind = ControlFieldKind.LINE
    choices: tuple[str, ...] = ()
    browse: bool = False
    enabled: bool = True
    reason: str = ""
    parameter_group: str = ""
    #: The grey hint an empty range bound shows (None leaves the renderer's own), e.g. the
    #: automatic threshold ceiling, which is never the field's value (decision 39).
    placeholder: str | None = None
    #: INT fields only: the bounds, and whether a blank (automatic) value is valid
    #: (:func:`integer_field_refusal`).
    minimum: int | None = None
    maximum: int | None = None
    allow_blank: bool = False


@dataclass(frozen=True, slots=True)
class ControlFormEdit:
    """One value-change intent emitted by the shared Controls renderer."""

    path: tuple[str, ...]
    value: object


@dataclass(frozen=True, slots=True)
class ControlsProjection:
    """One complete, immutable render projection for the Controls panel."""

    processing_page: ProcessingPage
    fields: tuple[ControlFormField, ...]
    section_actions: Mapping[SectionId, tuple[ControlActionSpec, ...]]
    detector_summary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(
            self,
            "section_actions",
            MappingProxyType({
                section: tuple(actions)
                for section, actions in self.section_actions.items()
            }),
        )

    def fields_for(self, section: SectionId) -> tuple[ControlFormField, ...]:
        return tuple(field for field in self.fields if field.section is section)

    def value_for(self, path: tuple[str, ...], default: object = "") -> object:
        path = tuple(path)
        for field in self.fields:
            if field.path == path:
                return field.value
        return default

    def actions_for(self, section: SectionId) -> tuple[ControlActionSpec, ...]:
        return tuple(self.section_actions.get(section, ()))


def tool_from_mode_text(mode_text: str | None) -> Tool:
    """Map the GUI mode-combo text to the typed Controls tool."""
    text = str(mode_text or "").strip().lower()
    if text in {"2d viewer", "image viewer"}:
        return Tool.IMAGE_VIEWER
    if text in {"1d viewer", "xye viewer"}:
        return Tool.XYE_VIEWER
    if "image viewer" in text:
        return Tool.IMAGE_VIEWER
    if "xye viewer" in text:
        return Tool.XYE_VIEWER
    if "nexus viewer" in text or "nexus" in text and "viewer" in text:
        return Tool.NEXUS_VIEWER
    if "stitch" in text:
        return Tool.STITCH
    if "rsm" in text:
        return Tool.RSM
    if "2d" in text and "1d" not in text:
        return Tool.INT_2D
    return Tool.INT_1D


_NATIVE_GI_ONLY_ARGS: frozenset[str] = frozenset(
    {
        "incident_angle",
        "incidence_motor",
        "tilt_angle",
        "sample_orientation",
        "method",
        "mode_1d",
        "mode_2d",
        "npt_oop",
        "gi_mode_1d",
        "gi_mode_2d",
        "npt_ip",
        "x_range",
        "y_range",
    }
)


def _native_pop_first(
    args: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        if key in args:
            return args.pop(key)
    return default


def _native_npt_2d(args_2d: dict[str, Any]) -> tuple[int, int]:
    npt = args_2d.pop("npt", None)
    if isinstance(npt, (tuple, list)) and len(npt) == 2:
        return int(npt[0]), int(npt[1])
    npt_rad = args_2d.pop("npt_rad", None)
    npt_azim = args_2d.pop("npt_azim", None)
    if npt_rad is None:
        npt_rad = npt if npt is not None else 1000
    if npt_azim is None:
        npt_azim = 360
    return int(npt_rad), int(npt_azim)


def _native_strip_nonstandard_args(args: dict[str, Any]) -> None:
    for key in _NATIVE_GI_ONLY_ARGS:
        args.pop(key, None)


def _native_gi_1d_unit_default(unit: Any, mode: str, *, is_gi: bool) -> str:
    if not is_gi:
        return str(unit or "q_A^-1")
    if mode == "q_ip":
        return "qip_A^-1"
    if mode == "q_oop":
        return "qoop_A^-1"
    return str(unit or "q_A^-1")


def _native_gi_2d_unit_default(unit: Any, mode: str, *, is_gi: bool) -> str:
    text = str(unit or "").strip()
    if not is_gi:
        return text or "q_A^-1"
    if mode == "qip_qoop":
        return text if text.startswith("qip_") else "qip_A^-1"
    return text or "q_A^-1"


#: 2-D integration argument naming the GI 2-D modes computed ALONGSIDE the
#: selected one (the Processing "Q-χ + Qip-Qoop" choice is ``qip_qoop`` plus
#: ``["q_chi"]`` here).  A selection of existing modes, never a mode key.
GI_COMPANION_MODES_2D_ARG = "gi_companion_modes_2d"


def gi_companion_modes_2d(
    bai_2d_args: Mapping[str, Any] | None, *, primary: str | None = None,
) -> tuple[str, ...]:
    """The declared companion GI 2-D modes of one 2-D argument mapping.

    A companion equal to *primary* is not a companion: the selected mode is
    calculated once, whatever a stale or hand-written selection says.
    """
    declared = (bai_2d_args or {}).get(GI_COMPANION_MODES_2D_ARG) or ()
    if isinstance(declared, (str, bytes)) or not all(
        type(mode) is str and mode for mode in declared
    ):
        raise ValueError(f"{GI_COMPANION_MODES_2D_ARG} must be a list of mode names")
    return tuple(dict.fromkeys(mode for mode in declared if mode != primary))


def build_native_int_reduction_plan_from_args(
    bai_1d_args: Mapping[str, Any] | None,
    bai_2d_args: Mapping[str, Any] | None,
    *,
    declare_companion_modes_2d: bool = False,
    gi_enabled: bool = False,
    gi_incident_angle: Any = None,
    incidence_motor: Any = None,
    tilt_angle: Any = 0.0,
    sample_orientation: Any = 4,
    gi_exit_angle_convention: str = "xdart_reflection_qoop_v1",
    integrate_1d: bool = True,
    integrate_2d: bool = True,
    threshold_min: Any = None,
    threshold_max: Any = None,
    mask_saturation: bool = False,
    saturation: int | None = None,
    detector_mask: Any = None,
    detector_shape: tuple[int, int] | None = None,
):
    """Build the native Controls integration plan.

    This consumes the already-synced Controls argument dictionaries
    directly and remains the pure controls-layer production plan builder.
    """

    from xrd_tools.reduction import (  # lazy: preserve readiness import purity
        GIMode,
        Integration1DPlan,
        Integration2DPlan,
        ReductionPlan,
    )
    from xrd_tools.reduction.masks import _mask_for_plan

    args_1d = dict(bai_1d_args or {})
    args_2d = dict(bai_2d_args or {})

    unit_1d = _native_pop_first(args_1d, ("unit",), "q_A^-1")
    unit_2d = _native_pop_first(args_2d, ("unit",), "q_A^-1")
    method_1d = _native_pop_first(args_1d, ("method",), "csr")
    method_2d = _native_pop_first(args_2d, ("method",), "csr")

    npt_1d = int(_native_pop_first(args_1d, ("npt", "numpoints", "npt_rad"), 1000))
    npt_rad_1d = int(_native_pop_first(args_1d, ("chi_npt_rad",), 1000))
    npt_rad_2d, npt_azim_2d = _native_npt_2d(args_2d)

    radial_range_1d = _native_pop_first(args_1d, ("radial_range",), None)
    azimuth_range_1d = _native_pop_first(args_1d, ("azimuth_range",), None)
    radial_range_2d = _native_pop_first(args_2d, ("radial_range",), None)
    azimuth_range_2d = _native_pop_first(args_2d, ("azimuth_range",), None)
    azimuth_offset_2d = float(
        _native_pop_first(args_2d, ("azimuth_offset", "chi_offset"), 0.0) or 0.0
    )
    chi_offset_1d = float(_native_pop_first(args_1d, ("chi_offset",), 0.0) or 0.0)
    # S-4: carry chi_offset as the 1D plan's azimuth_offset and re-add it to the
    # OUTPUT chi axis (in reduction), mirroring the 2D EXACTLY -- instead of
    # shifting the INPUT range and leaving the written 1D chi axis in the raw
    # pyFAI frame 90deg out of frame with the 2D cake chi.  GI keeps offset 0
    # (its chi handling is separate; the 2D also zeroes azimuth_offset for GI).
    azimuth_offset_1d = chi_offset_1d if not gi_enabled else 0.0

    monitor_1d = _native_pop_first(args_1d, ("monitor",), None)
    monitor_2d = _native_pop_first(args_2d, ("monitor",), None)
    error_1d = _native_pop_first(args_1d, ("error_model",), None)
    error_2d = _native_pop_first(args_2d, ("error_model",), None)
    pol_1d = _native_pop_first(args_1d, ("polarization_factor",), None)
    pol_2d = _native_pop_first(args_2d, ("polarization_factor",), None)
    _native_pop_first(args_1d, ("normalization_factor",), None)
    _native_pop_first(args_2d, ("normalization_factor",), None)

    gi_mode_1d = str(_native_pop_first(args_1d, ("gi_mode_1d",), "q_total"))
    gi_mode_2d = str(_native_pop_first(args_2d, ("gi_mode_2d",), "qip_qoop"))
    # Always consumed here, so it can never reach pyFAI as an unknown keyword.
    # Only an ordinary Run declares it; Average and Reintegration stay one-mode.
    companions_2d = gi_companion_modes_2d(args_2d, primary=gi_mode_2d)
    args_2d.pop(GI_COMPANION_MODES_2D_ARG, None)
    npt_oop = _native_pop_first(args_1d, ("npt_oop",), None)
    if npt_oop is None:
        npt_oop = _native_pop_first(args_2d, ("npt_oop",), None)
    gi_method = _native_pop_first(args_1d, ("gi_method_1d",), None)
    if gi_method is None:
        gi_method = _native_pop_first(
            args_2d,
            ("gi_method_2d",),
            "cython",
        )
    gi_method = str(gi_method)

    if not gi_enabled:
        _native_strip_nonstandard_args(args_1d)
        _native_strip_nonstandard_args(args_2d)

    gi = None
    if gi_enabled:
        incident = gi_incident_angle
        motor = None if incidence_motor is None else str(incidence_motor)
        if incident is None and motor is not None:
            try:
                incident = float(motor)
                motor = None
            except (TypeError, ValueError):
                pass
        if incident is None and motor is None:
            raise ValueError(
                "GI reduction requires an incident angle or incidence motor"
            )
        gi = GIMode(
            incident_angle=incident,
            incidence_motor=motor,
            tilt_angle=float(tilt_angle or 0.0),
            sample_orientation=int(sample_orientation or 4),
            method=gi_method,
            mode_1d=gi_mode_1d,
            mode_2d=gi_mode_2d,
            npt_oop=None if npt_oop is None else int(npt_oop),
            gi_exit_angle_convention=gi_exit_angle_convention,
        )

    integration_1d = None
    if integrate_1d:
        integration_1d = Integration1DPlan(
            npt=npt_1d,
            npt_rad=npt_rad_1d,
            unit=_native_gi_1d_unit_default(
                unit_1d, gi_mode_1d, is_gi=gi_enabled
            ),
            method=str(method_1d),
            radial_range=radial_range_1d,
            azimuth_range=azimuth_range_1d,
            monitor_key=monitor_1d,
            error_model=error_1d,
            polarization_factor=pol_1d,
            azimuth_offset=azimuth_offset_1d,
            extra=args_1d,
        )

    integration_2d = None
    if integrate_2d:
        integration_2d = Integration2DPlan(
            npt_rad=npt_rad_2d,
            npt_azim=npt_azim_2d,
            unit=_native_gi_2d_unit_default(
                unit_2d, gi_mode_2d, is_gi=gi_enabled
            ),
            method=str(method_2d),
            radial_range=radial_range_2d,
            azimuth_range=azimuth_range_2d,
            azimuth_offset=azimuth_offset_2d,
            monitor_key=monitor_2d,
            error_model=error_2d,
            polarization_factor=pol_2d,
            extra=args_2d,
        )

    # Native Run produces one selected GI mode per enabled dimension, plus any
    # declared companion 2-D modes. Declare those sets so resource accounting
    # funds exactly them and does not reserve every schema mode.
    modes = {}
    if gi is not None:
        if integration_1d is not None:
            modes["enabled_modes_1d"] = (gi.mode_1d.value,)
        if integration_2d is not None:
            modes["enabled_modes_2d"] = (gi.mode_2d.value,) + (
                companions_2d if declare_companion_modes_2d else ()
            )
    return ReductionPlan(
        integration_1d=integration_1d,
        integration_2d=integration_2d,
        gi=gi,
        mask=_mask_for_plan(detector_mask, detector_shape),
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        mask_saturation=bool(mask_saturation),
        saturation=saturation,
        extra=modes,
    )


def build_native_int_reduction_plan_from_scan(
    scan: Any,
    *,
    integrate_1d: bool = True,
    integrate_2d: bool | None = None,
    threshold_min: Any = None,
    threshold_max: Any = None,
    mask_saturation: bool = False,
):
    """Build the native Controls integration plan from a live-scan-like object.

    This is the Qt-free run-path form of
    :func:`build_native_int_reduction_plan_from_args`: it reads only typed scan
    attributes and ``bai_*_args`` dictionaries, not Qt widgets.
    """

    if integrate_2d is None:
        integrate_2d = not bool(getattr(scan, "skip_2d", False))

    gi_config = dict(getattr(scan, "gi_config", {}) or {})
    if gi_config and "gi_exit_angle_convention" not in gi_config:
        gi_convention = "legacy_xdart_2025_reflection"
    else:
        gi_convention = gi_config.get(
            "gi_exit_angle_convention", "xdart_reflection_qoop_v1"
        )

    def _gi_value(name: str, default: Any) -> Any:
        value = getattr(scan, name, None)
        if value is None:
            value = gi_config.get(name, None)
        return default if value is None else value

    detector_shape = getattr(scan, "detector_shape", None)
    if detector_shape is not None:
        try:
            detector_shape = (int(detector_shape[0]), int(detector_shape[1]))
        except (TypeError, ValueError, IndexError):
            detector_shape = None
    if detector_shape is None:
        try:
            first_idx = scan.frames.index[0]
            first_img = getattr(scan.frames[int(first_idx)], "map_raw", None)
            detector_shape = getattr(first_img, "shape", None)
        except Exception:
            detector_shape = None

    return build_native_int_reduction_plan_from_args(
        dict(getattr(scan, "bai_1d_args", {}) or {}),
        dict(getattr(scan, "bai_2d_args", {}) or {}),
        gi_enabled=bool(getattr(scan, "gi", False)),
        gi_incident_angle=getattr(scan, "_cached_fiber_integrator_angle", None),
        incidence_motor=getattr(scan, "incidence_motor", None),
        tilt_angle=_gi_value("tilt_angle", 0.0),
        sample_orientation=_gi_value("sample_orientation", 4),
        gi_exit_angle_convention=gi_convention,
        integrate_1d=integrate_1d,
        integrate_2d=bool(integrate_2d),
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        mask_saturation=mask_saturation,
        detector_mask=getattr(scan, "global_mask", None),
        detector_shape=detector_shape,
    )


def execution_plan_shape(
    bai_1d_args: Mapping[str, Any],
    bai_2d_args: Mapping[str, Any],
    gi: Any,
    processing_mode: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The plan arguments a Run's controls imply: its argument copies, GI and dimensions.

    One rule for the Run's frozen configuration and for a saved output's ``run_configuration``
    (BROWSE-REFERENCE-RINGS-20261013), whose rings must rebuild the plan facts the Run published.
    """
    one, two = dict(bai_1d_args), dict(bai_2d_args)
    if gi.enabled:
        one["gi_mode_1d"], two["gi_mode_2d"] = gi.mode_1d, gi.mode_2d
    manual = gi.enabled and gi.effective_motor == "Manual"
    return one, two, {
        "gi_enabled": gi.enabled,
        "gi_incident_angle": gi.th_val if manual else None,
        "incidence_motor": gi.effective_motor if gi.enabled and not manual else None,
        "tilt_angle": gi.tilt_angle, "sample_orientation": gi.sample_orientation,
        "integrate_1d": True,
        "integrate_2d": not (
            "Viewer" not in processing_mode
            and "1D" in processing_mode
            and "2D" not in processing_mode
        ),
    }


def execution_plan_values(
    configuration: FrozenRunConfiguration,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    threshold = configuration.threshold
    one, two, shape = execution_plan_shape(
        configuration.bai_1d_args, configuration.bai_2d_args,
        configuration.gi, configuration.processing_mode,
    )
    return one, two, {
        **shape,
        "threshold_min": (
            threshold.threshold_min if threshold.apply_threshold else None
        ),
        "threshold_max": (
            threshold.threshold_max if threshold.apply_threshold else None
        ),
        "mask_saturation": threshold.mask_saturation,
        # the source's resolved ceiling (decision 39); absent: the data type's rule
        "saturation": threshold.saturation,
    }


class EnergyScanRefused(ValueError):
    """An Int energy scan that cannot be reduced as recorded.

    Raised when the scan is bound, before any frame is reduced or anything is written; its
    message says what to change, and the page shows it in a notification.
    """


def effective_ring_geometry(calibration, *, integrator, plan=None):
    """One immutable, transient snapshot of geometry already bound for Run.

    Construct once, before publication. JSON owns the nested detector config;
    no integrator, array, source or RSM-only image transform is retained.
    """
    import json
    detector = integrator.detector
    return MappingProxyType({
        **{name: float(getattr(integrator, name)) for name in
           ('dist', 'poni1', 'poni2', 'rot1', 'rot2', 'rot3')},
        'wavelength': integrator.wavelength,
        'detector': detector.name,
        'detector_config_json': json.dumps(dict(calibration.detector_config), sort_keys=True),
        'parallax': calibration.parallax,
        'is_flat': bool(detector.IS_FLAT),
        'uniform_pixel': bool(detector.uniform_pixel),
        'has_spline': detector.spline is not None,
        'domain': 'Grazing' if plan is not None and plan.gi is not None else 'Standard',
        'chiDiscAtPi': bool(integrator.chiDiscAtPi),
        'azimuth_offset_1d': float(getattr(getattr(plan, 'integration_1d', None), 'azimuth_offset', 0.)),
        'azimuth_offset_2d': float(getattr(getattr(plan, 'integration_2d', None), 'azimuth_offset', 0.)),
    })


def frame_ring_wavelengths(selection: Mapping[str, Any] | None) -> dict[int, float]:
    """Each listed frame's effective wavelength (m) from an energy selection's per-frame energies.

    The one per-label rule the Run's ring publication and a saved output's rings share: a label in
    ``frame_labels`` takes its own ``frame_energies_eV`` entry; any other keeps the snapshot's wavelength.
    """
    from xrd_tools.core.energy import energy_eV_to_wavelength_m
    selection = selection or {}
    return dict(zip(selection.get('frame_labels', ()),
                    map(energy_eV_to_wavelength_m, selection.get('frame_energies_eV', ()))))


@dataclass(frozen=True, slots=True)
class RingRefusal:
    """A saved output whose frames get no predicted rings, with the reason for each of them.

    ``kind`` is the record's: an Average result or a Reintegrate version is not an Int Run's record, so a
    display keeps its own generic reason for it rather than this one.
    """

    reason: str
    kind: str = 'run'

    @property
    def shape(self) -> None:
        return None

    def frame_snapshot(self, label: int) -> None:
        return None

    def frame_reason(self, label: int) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class RingBasis:
    """Detached per-frame ring snapshots rebuilt from one saved Int output (BROWSE-REFERENCE-RINGS-20261013).

    ``base`` is the output's 17-field snapshot; an eligible label's own wavelength replaces its ``wavelength``,
    as the Run published it. ``shape`` is the saved detector's native shape. Labels outside the output, and an
    energy scan's earlier Append epochs (D5), refuse with their own reasons. No handle, integrator or array is held.
    """

    base: Mapping[str, Any]
    wavelengths: Mapping[int, float]
    labels: frozenset[int]
    eligible: frozenset[int]
    shape: tuple[int, int]
    #: The record's own accepted calibration and plan, as the basis resolved them (COVERAGE-SUPPORT-20261013's saved
    #: support basis extends this one rather than reading the record again); None only for a hand-built basis.
    calibration: Any = None
    plan: Any = None

    @property
    def kind(self) -> str:
        return 'run'

    def frame_snapshot(self, label: int) -> Mapping[str, Any] | None:
        if self.frame_reason(label):
            return None
        wavelength = self.wavelengths.get(label, self.base['wavelength'])
        return (self.base if wavelength == self.base['wavelength']
                else MappingProxyType({**self.base, 'wavelength': wavelength}))

    def frame_reason(self, label: int) -> str:
        from xrd_tools import geometry_figures as gf
        if label not in self.labels:
            return gf.SAVED_LABEL_ABSENT_REASON
        return '' if label in self.eligible else gf.SAVED_HISTORY_REASON


def _valid_energy_selection(selection: Any) -> bool:
    """A saved energy selection the Run's rule can trust: a finite positive or absent representative wavelength,
    and equal-length lists of unique nonnegative integer labels and finite positive energies, or neither list."""
    def positive(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    if not isinstance(selection, Mapping) or 'wavelength_m' not in selection:
        return False
    wavelength = selection.get('wavelength_m')
    if wavelength is not None and not positive(wavelength):
        return False
    labels, energies = selection.get('frame_labels'), selection.get('frame_energies_eV')
    if labels is None and energies is None:
        return True
    return (type(labels) is list and type(energies) is list and len(labels) == len(energies)
            and all(type(label) is int and label >= 0 for label in labels) and len(set(labels)) == len(labels)
            and all(positive(energy) for energy in energies))


def ring_basis_from_saved(run_configuration, energy_selection, *, labels, monochromator_wavelength_m=None,
                          epochs=None, kind='run', unreadable=()):
    """One saved Int output's ring basis, or a typed refusal; malformed content refuses rather than raises.

    Geometry comes only from the output's own record: its accepted PONI values and detector configuration, its
    saved plan arguments through the Run's own plan route (:func:`execution_plan_shape`), and its effective
    energy — the energy selection, or for an older output without one its recorded monochromator wavelength (D2).
    When the selection lists per-frame energies, only the last Append epoch's labels are eligible (D5).
    ``unreadable`` names the record nodes that exist but could not be read.
    """
    import json
    from dataclasses import replace
    from xrd_tools import geometry_figures as gf
    from xrd_tools.integrate.calibration import (
        detector_calibration_from_projection, detector_calibration_to_integrator)
    from xrd_tools.session.run_configuration import FrozenGIConfiguration
    if kind == 'average':
        return RingRefusal(gf.SAVED_AVERAGE_REASON, kind)
    if kind == 'reintegrate':
        return RingRefusal(gf.SAVED_REINTEGRATE_REASON, kind)
    if 'run_configuration' in unreadable or (run_configuration is not None
                                             and not isinstance(run_configuration, Mapping)):
        return RingRefusal(gf.SAVED_RECORD_MALFORMED_REASON)
    signature = (run_configuration or {}).get('scientific_signature')
    assets = signature.get('accepted_scientific_assets') if isinstance(signature, Mapping) else None
    if not isinstance(assets, Mapping) or not isinstance(assets.get('poni_values'), Mapping):
        return RingRefusal(gf.SAVED_NO_CALIBRATION_REASON)
    if assets.get('poni_detector_config_json') is None:
        return RingRefusal(gf.SAVED_NO_DETECTOR_CONFIG_REASON)
    if 'energy_selection' in unreadable or (energy_selection is not None
                                            and not _valid_energy_selection(energy_selection)):
        return RingRefusal(gf.SAVED_ENERGY_MALFORMED_REASON)
    if energy_selection is None and 'monochromator' in unreadable:
        return RingRefusal(gf.SAVED_ENERGY_MALFORMED_REASON)
    if energy_selection is not None and energy_selection['wavelength_m'] is None:
        return RingRefusal(gf.SAVED_NO_ENERGY_REASON)
    if energy_selection is None and monochromator_wavelength_m is None:
        return RingRefusal(gf.SAVED_NO_ENERGY_REASON)
    per_frame = energy_selection is not None and 'frame_energies_eV' in energy_selection
    if per_frame and (epochs is None or 'append_lineage' in unreadable):
        return RingRefusal(gf.SAVED_HISTORY_UNREADABLE_REASON)
    wavelength = monochromator_wavelength_m if energy_selection is None else energy_selection.get('wavelength_m')
    try:
        calibration = detector_calibration_from_projection(
            dict(assets['poni_values']), detector_config=json.loads(assets['poni_detector_config_json']))
        if wavelength is not None:
            calibration = replace(calibration, poni=replace(calibration.poni, wavelength=wavelength))
        integrator = detector_calibration_to_integrator(calibration)
        one, two, values = execution_plan_shape(
            run_configuration['bai_1d_args'], run_configuration['bai_2d_args'],
            FrozenGIConfiguration.from_dict(run_configuration['gi']), run_configuration['processing_mode'])
        plan = build_native_int_reduction_plan_from_args(one, two, **values)
        base = effective_ring_geometry(calibration, integrator=integrator, plan=plan)
        shape = tuple(int(size) for size in integrator.detector.shape)
    except (KeyError, TypeError, ValueError, AttributeError):
        return RingRefusal(gf.SAVED_RECORD_MALFORMED_REASON)
    reason = gf.detector_ring_unavailable(base, shape)
    if reason:
        return RingRefusal(reason)
    held = frozenset(labels)
    return RingBasis(base, MappingProxyType(frame_ring_wavelengths(energy_selection)), held,
                     held & frozenset(epochs[-1] if epochs else ()) if per_frame else held, shape, calibration, plan)


def ring_basis_for_record(path, *, entry='entry', labels):
    """Read one processed output's ring facts (bounded, local, once) and return its basis or refusal."""
    from xrd_tools.io.browse_presentation import read_ring_record
    record = read_ring_record(path, entry=entry)
    return ring_basis_from_saved(
        record.run_configuration, record.energy_selection, labels=labels,
        monochromator_wavelength_m=record.monochromator_wavelength_m, epochs=record.epochs, kind=record.kind,
        unreadable=record.unreadable)


def bind_native_int_source_energy(scan, calibration, *, metadata_wavelength_A=None,
                                  metadata_energy_eV=()):
    """Bind already-read source metadata to a new Int scan, without loading pixels.

    Explicitly called by new Run/replay only; persisted and Reintegrate scans retain
    their existing interpretation. The returned calibration is an effective copy.

    Each frame's own recorded energy decides (eV or keV by its size; an impossible one
    counts as none). Only when no frame records one does the source's energy or wavelength
    apply. A varying series is an energy scan, which Int reduces frame by frame at each
    frame's own energy (``frame_labels`` and ``frame_energies_eV``); the sensor parallax
    correction, computed at one energy, cannot follow it. Such a scan, and one whose per-frame
    energies would not fit the run's saved settings, raise :class:`EnergyScanRefused`.
    """
    from dataclasses import replace
    from xrd_tools.core.energy import resolve_acquisition_energy, metadata_energy_values, WavelengthUnit
    from xrd_tools.integrate.calibration import detector_calibration_to_integrator
    labels, recorded = [], []
    for frame in scan.frames:
        values = metadata_energy_values(frame.metadata, energy_is_eV=frame.metadata_energy_is_eV)
        if len(values) > 1:
            resolve_acquisition_energy(None, values)  # a row's energy and wavelengths must agree
        labels.append(int(frame.index))
        recorded.append(values[0] if values else None)
    per_frame = any(value is not None for value in recorded)
    if per_frame:
        energies = recorded
    else:
        energies = list(metadata_energy_eV)
        if metadata_wavelength_A is not None:
            energies.extend(metadata_energy_values({}, wavelength=metadata_wavelength_A,
                                                  wavelength_unit=WavelengthUnit.ANGSTROM))
    selection = resolve_acquisition_energy(calibration.poni.wavelength, energies, allow_energy_scan=per_frame)
    if 'frame_energies_eV' in selection:
        if calibration.parallax is not None:
            raise EnergyScanRefused(
                'an energy scan cannot use the sensor parallax correction, which is computed at one energy. To '
                'reduce each frame at its own energy, turn parallax off: in Advanced, "Detector sensor / parallax '
                '(PONI v3)", check "Override selected PONI sensor settings" and clear "Enable parallax"')
        selection = {**selection, 'frame_labels': labels,
                     'frame_energies_eV': [float(value) for value in selection['frame_energies_eV']]}
        from xrd_tools.core.provenance import config_entry_overflow
        overflow = config_entry_overflow('energy_selection', selection)  # as the run saves it at its end
        if overflow is not None:
            size, ceiling = overflow
            raise EnergyScanRefused(
                f"an energy scan of {len(labels):,} frames cannot be saved: Int saves each frame's energy with the "
                f"run's settings, which hold at most {ceiling:,} bytes, and these energies need {size:,}. Reduce at "
                f"most about {len(labels) * ceiling // size:,} frames per run")
    if selection['disagreement']:
        logger.warning('X-ray energy mismatch: metadata=%.1f eV, PONI=%.1f eV; using metadata',
                       selection['source_energy_eV'], selection['calibration_energy_eV'])
    wavelength = selection['wavelength_m']
    if wavelength is not None:
        calibration = replace(calibration, poni=replace(calibration.poni, wavelength=wavelength))
    scan.poni = calibration.poni
    scan.integrator = detector_calibration_to_integrator(calibration)
    scan.wavelength = None if wavelength is None else wavelength * 1e10
    scan.energy = None if selection['energy_eV'] is None else selection['energy_eV'] / 1000
    scan.extra = {**scan.extra, 'energy_selection': selection}
    return calibration


def native_int_reduction_plan(
    configuration: FrozenRunConfiguration, *, companion_modes_2d: bool = False,
    energy_selection: Mapping[str, Any] | None = None,
):
    """Translate one frozen Controls configuration into its mask-free plan.

    *companion_modes_2d* is set only by an ordinary Run: it alone integrates the
    GI companion map the "Q-χ + Qip-Qoop" choice selects.  Average and the XYE
    naming lookup stay single-mode.

    Scientific assets are admitted independently of control intent. The
    ordinary run path attaches its already-authenticated detector mask after
    this pure translation; Average keeps the mask in its calibration state so
    it can apply the accepted static mask while accumulating detector pixels.
    """

    from xrd_tools.session.readiness import (
        build_native_int_reduction_plan_from_args,
    )

    one, two, values = execution_plan_values(configuration)
    plan = build_native_int_reduction_plan_from_args(
        one, two, declare_companion_modes_2d=companion_modes_2d, **values,
    )
    if energy_selection is not None:
        from dataclasses import replace
        plan = replace(plan, extra={**plan.extra, 'energy_selection': dict(energy_selection)})
    return plan
