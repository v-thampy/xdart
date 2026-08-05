"""Pure validation and detached-candidate edits for Controls intent."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from xrd_tools.core.scan import SourceSpec
from xrd_tools.session.intent_store import (
    IntentCommitAccepted,
    RunIntentSnapshot,
    RunIntentStore,
)
from xrd_tools.session.readiness import INTEGRATION_CONTROL_SPECS
from xrd_tools.session.run_configuration import RunIntent
from xrd_tools.sources.selection import DirectorySourceSpec

from .contracts import SourceSelection
from .detector_projection import poni_saturation_ceiling
from .controls_inventory import (
    GI_1D_AXES,
    GI_2D_AXES,
    GI_ENABLED,
    GI_MOTOR,
    GI_ORIENTATION,
    GI_THETA,
    GI_TILT,
    INT_1D_AXIS,
    INT_1D_POINTS,
    INT_1D_RADIAL_AUTO,
    INT_1D_AZIM_AUTO,
    INT_2D_AXIS,
    INT_2D_RADIAL_POINTS,
    INT_2D_AZIM_POINTS,
    INT_2D_RADIAL_AUTO,
    INT_2D_AZIM_AUTO,
    INT_PATHS,
    MASK_FILE,
    MASK_SATURATION,
    OUTPUT_MODE,
    PONI_FILE,
    PROJECT_ROOT,
    SAVE_PATH,
    SOURCE_DIRECTORY,
    SOURCE_EDIT_PATHS,
    SOURCE_FILTER,
    SOURCE_META,
    SOURCE_FORMAT_SUFFIXES,
    SOURCE_RECURSIVE,
    SOURCE_SUFFIX,
    STANDARD_1D_AXES,
    STANDARD_2D_AXES,
    THRESHOLD_ENABLED,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    integration_values,
    points_2d,
)
from .output_values import APPEND_UNAVAILABLE


@dataclass(frozen=True, slots=True)
class EditRefusal:
    reason: str


@dataclass(frozen=True, slots=True)
class EditNoChange:
    pass


EditResult = RunIntent | EditRefusal | EditNoChange


ADVANCED_METHODS_1D = (
    "numpy",
    "cython",
    "BBox",
    "splitpixel",
    "lut",
    "csr",
    "nosplit_csr",
    "full_csr",
    "lut_ocl",
    "csr_ocl",
)
ADVANCED_METHODS_2D = (
    "numpy",
    "cython",
    "BBox",
    "splitpixel",
    "lut",
    "csr",
    "lut_ocl",
    "csr_ocl",
)
GI_HISTOGRAM_METHODS = ("cython", "python")
DEFAULT_GI_HISTOGRAM_METHOD = "cython"
DEFAULT_POLARIZATION_FACTOR = 0.99
_ADVANCED_LEAVES = (
    "correctSolidAngle",
    "apply_polarization",
    "polarization_factor",
    "method",
    "dummy",
    "delta_dummy",
    "chi_offset",
    "safe",
)
ADVANCED_FIELD_PATHS = tuple(
    spec.path
    for spec in INTEGRATION_CONTROL_SPECS
    if len(spec.path) == 2 and spec.path[1] in _ADVANCED_LEAVES
)
if set(ADVANCED_FIELD_PATHS) != {
    (root, leaf)
    for root in ("Int1D", "Int2D")
    for leaf in _ADVANCED_LEAVES
}:
    raise RuntimeError(
        "Advanced integration fields disagree with the shared schema."
    )


@dataclass(frozen=True, slots=True)
class AdvancedDimensionValues:
    correct_solid_angle: object
    apply_polarization: object
    polarization_factor: object
    method: object
    dummy: object
    delta_dummy: object
    chi_offset: object
    safe: object


@dataclass(frozen=True, slots=True)
class AdvancedSettingsValues:
    one_d: AdvancedDimensionValues
    two_d: AdvancedDimensionValues
    gi_enabled: object = False
    gi_method: object = DEFAULT_GI_HISTOGRAM_METHOD


def advanced_settings_values(
    snapshot: RunIntentSnapshot,
) -> AdvancedSettingsValues:
    """Project the value-only Advanced editor from one revisioned snapshot."""

    if not isinstance(snapshot, RunIntentSnapshot):
        raise TypeError("snapshot must be a RunIntentSnapshot")
    intent = snapshot.thaw()
    return AdvancedSettingsValues(
        one_d=_advanced_dimension_values(
            intent.bai_1d_args,
            root="Int1D",
        ),
        two_d=_advanced_dimension_values(
            intent.bai_2d_args,
            root="Int2D",
        ),
        gi_enabled=intent.gi.enabled,
        gi_method=_advanced_gi_method(intent),
    )


def reduce_advanced_settings(
    snapshot: RunIntentSnapshot,
    values: AdvancedSettingsValues,
) -> EditResult:
    """Validate and apply one complete Advanced-dialog submission."""

    if not isinstance(snapshot, RunIntentSnapshot):
        return EditRefusal("Advanced settings lost their revision.")
    if type(values) is not AdvancedSettingsValues:
        return EditRefusal("Advanced settings are invalid.")
    current = advanced_settings_values(snapshot)
    one_d = _validate_advanced_dimension(
        values.one_d,
        current.one_d,
        root="1D",
        methods=ADVANCED_METHODS_1D,
    )
    if isinstance(one_d, EditRefusal):
        return one_d
    two_d = _validate_advanced_dimension(
        values.two_d,
        current.two_d,
        root="2D",
        methods=ADVANCED_METHODS_2D,
    )
    if isinstance(two_d, EditRefusal):
        return two_d
    if type(values.gi_enabled) is not bool:
        return EditRefusal("GI Advanced mode requires true or false.")
    if values.gi_enabled is not current.gi_enabled:
        return EditRefusal("GI mode changed outside the Advanced editor.")
    if type(values.gi_method) is not str:
        return EditRefusal("GI histogram backend is invalid.")
    if current.gi_enabled:
        if (
            values.gi_method not in GI_HISTOGRAM_METHODS
            and values.gi_method != current.gi_method
        ):
            return EditRefusal("GI histogram backend is unsupported.")
    elif values.gi_method != current.gi_method:
        return EditRefusal(
            "GI histogram backend is available only in Grazing mode."
        )
    desired = AdvancedSettingsValues(
        one_d=one_d,
        two_d=two_d,
        gi_enabled=values.gi_enabled,
        gi_method=values.gi_method,
    )
    if desired == current:
        return EditNoChange()
    candidate = snapshot.thaw()
    _install_advanced_dimension(
        candidate.bai_1d_args,
        current.one_d,
        desired.one_d,
        root="Int1D",
    )
    _install_advanced_dimension(
        candidate.bai_2d_args,
        current.two_d,
        desired.two_d,
        root="Int2D",
    )
    if (
        current.gi_enabled
        and desired.gi_method != current.gi_method
    ):
        candidate.bai_1d_args["gi_method_1d"] = desired.gi_method
        candidate.bai_2d_args["gi_method_2d"] = desired.gi_method
    return candidate


def _advanced_gi_method(intent: RunIntent) -> object:
    """Return the effective Fiber backend without rewriting legacy intent."""

    method = intent.bai_1d_args.get("gi_method_1d")
    if method is None:
        method = intent.bai_2d_args.get(
            "gi_method_2d",
            DEFAULT_GI_HISTOGRAM_METHOD,
        )
    # pyFAI's historical ``"no"`` shorthand resolves to its Python
    # histogram implementation.  Present the semantic name to the operator,
    # while leaving the stored shorthand untouched unless they choose a
    # different backend.
    return "python" if method == "no" else method


def _advanced_dimension_values(
    args: dict[object, object],
    *,
    root: str,
) -> AdvancedDimensionValues:
    polarization = args.get("polarization_factor")
    offset = (
        args.get("azimuth_offset", args.get("chi_offset", 0.0))
        if root == "Int2D"
        else args.get("chi_offset", 0.0)
    )
    return AdvancedDimensionValues(
        bool(args.get("correctSolidAngle", True)),
        polarization is not None,
        (
            DEFAULT_POLARIZATION_FACTOR
            if polarization is None
            else polarization
        ),
        args.get("method", "csr"),
        args.get("dummy"),
        args.get("delta_dummy"),
        offset,
        bool(args.get("safe", True)),
    )


def _validate_advanced_dimension(
    values: object,
    current: AdvancedDimensionValues,
    *,
    root: str,
    methods: tuple[str, ...],
) -> AdvancedDimensionValues | EditRefusal:
    if type(values) is not AdvancedDimensionValues:
        return EditRefusal(f"{root} advanced settings are invalid.")
    for label, value in (
        ("solid-angle correction", values.correct_solid_angle),
        ("polarization correction", values.apply_polarization),
        ("safe integration", values.safe),
    ):
        if type(value) is not bool:
            return EditRefusal(f"{root} {label} requires true or false.")
    method = values.method
    if (
        type(method) is not str
        or (method not in methods and method != current.method)
    ):
        return EditRefusal(f"{root} integration method is unsupported.")
    factor = _finite_float(values.polarization_factor)
    if isinstance(factor, EditRefusal):
        return EditRefusal(f"{root} polarization factor must be finite.")
    if not -1.0 <= factor <= 1.0:
        return EditRefusal(
            f"{root} polarization factor must be from -1 through 1."
        )
    dummy = _optional_finite_float(values.dummy)
    if isinstance(dummy, EditRefusal):
        return EditRefusal(f"{root} dummy value must be finite or blank.")
    delta_dummy = _optional_finite_float(values.delta_dummy)
    if isinstance(delta_dummy, EditRefusal):
        return EditRefusal(
            f"{root} dummy tolerance must be finite or blank."
        )
    chi_offset = _finite_float(values.chi_offset)
    if isinstance(chi_offset, EditRefusal):
        return EditRefusal(f"{root} chi offset must be finite.")
    return AdvancedDimensionValues(
        values.correct_solid_angle,
        values.apply_polarization,
        (
            factor
            if values.apply_polarization
            else DEFAULT_POLARIZATION_FACTOR
        ),
        method,
        dummy,
        delta_dummy,
        chi_offset,
        values.safe,
    )


def _optional_finite_float(value: object) -> float | None | EditRefusal:
    if value is None or (type(value) is str and not value.strip()):
        return None
    return _finite_float(value)


def _install_advanced_dimension(
    args: dict[object, object],
    current: AdvancedDimensionValues,
    desired: AdvancedDimensionValues,
    *,
    root: str,
) -> None:
    if desired.correct_solid_angle != current.correct_solid_angle:
        args["correctSolidAngle"] = desired.correct_solid_angle
    if (
        desired.apply_polarization != current.apply_polarization
        or (
            desired.apply_polarization
            and desired.polarization_factor != current.polarization_factor
        )
    ):
        args["polarization_factor"] = (
            desired.polarization_factor
            if desired.apply_polarization
            else None
        )
    if desired.method != current.method:
        args["method"] = desired.method
    for key, value, old in (
        ("dummy", desired.dummy, current.dummy),
        ("delta_dummy", desired.delta_dummy, current.delta_dummy),
    ):
        if value == old:
            continue
        if value is None:
            args.pop(key, None)
        else:
            args[key] = value
    if desired.chi_offset != current.chi_offset:
        if root == "Int2D":
            args.pop("azimuth_offset", None)
        args["chi_offset"] = desired.chi_offset
    if desired.safe != current.safe:
        args["safe"] = desired.safe


def reduce_control_edit(
    snapshot: RunIntentSnapshot,
    path: tuple[str, ...],
    value: object,
) -> EditResult:
    """Validate one presentation edit before changing an independent candidate."""

    if type(path) is not tuple or not all(type(part) is str for part in path):
        return EditRefusal("Unknown control.")
    if path in INT_PATHS:
        return _reduce_integration_edit(snapshot, path, value)
    if path in SOURCE_EDIT_PATHS:
        return _reduce_source_edit(snapshot, path, value)
    intent = snapshot.thaw()
    parsed = _control_value(intent, path, value)
    if isinstance(parsed, EditRefusal):
        return parsed
    current = _current_value(intent, path)
    # Design checkpoint 2026-08-04: a touch of any threshold control must NOT
    # absorb as EditNoChange while the intent's threshold identity differs
    # from what the panel displays (degenerate booleans OR unmaterialized
    # displayed defaults) — the same-value edit falls through so the shared
    # canonicalizer below makes identity match display.  The probe runs on a
    # detached thawed copy.
    noncanonical_threshold_touch = (
        path in {
            MASK_SATURATION, THRESHOLD_ENABLED, THRESHOLD_MIN, THRESHOLD_MAX,
        }
        and canonicalize_threshold_intent(snapshot.thaw())
    )
    if parsed == current and not noncanonical_threshold_touch:
        return EditNoChange()
    if path in {THRESHOLD_MIN, THRESHOLD_MAX}:
        low = parsed if path == THRESHOLD_MIN else intent.threshold.threshold_min
        high = parsed if path == THRESHOLD_MAX else intent.threshold.threshold_max
        if low is not None and high is not None and low > high:
            return EditRefusal("Threshold minimum cannot exceed maximum.")
    candidate = snapshot.thaw()
    _install_value(candidate, path, parsed)
    if path == GI_ENABLED:
        _normalize_gi_units(
            candidate,
            was_enabled=intent.gi.enabled,
        )
    elif path in {THRESHOLD_MIN, THRESHOLD_MAX}:
        # LV-UI-11: setting a manual bound IS choosing manual thresholding —
        # the sentinel Auto masking and the manual band are exclusive.  A
        # CLEARED bound (parsed None) keeps the mode and re-materializes to
        # the displayed default through the canonicalizer below.
        if parsed is not None:
            candidate.threshold.apply_threshold = True
            candidate.threshold.mask_saturation = False
        canonicalize_threshold_intent(candidate)
    elif path == MASK_SATURATION:
        # LV-UI-11: the Threshold row's Auto toggle.  ON = mask saturated
        # pixels, no manual band; OFF = the manual [min, max] band applies
        # at exactly the displayed defaults.
        candidate.threshold.apply_threshold = not parsed
        canonicalize_threshold_intent(candidate)
    elif path == THRESHOLD_ENABLED:
        # Kept for the legacy static_scan binding and programmatic edits; the
        # same exclusivity holds in both directions.
        candidate.threshold.mask_saturation = not parsed
        canonicalize_threshold_intent(candidate)
    return candidate


def canonicalize_threshold_intent(intent: RunIntent) -> bool:
    """THE shared vNext threshold canonicalizer (frozen design checkpoint,
    2026-08-04): one function makes an intent's threshold identity describe
    exactly what the panel displays.  Used by BOTH the edit reducer and the
    start capture; returns True when the intent was changed.

    - Exclusive booleans: a degenerate pair adopts the displayed Auto fact —
      ``apply_threshold = not mask_saturation``.
    - Manual mode MATERIALIZES the displayed defaults into the identity:
      missing minimum -> 0.0; missing maximum -> the detector family's
      display default when known, otherwise None (the box renders blank and
      execution stays open-ended above — display and identity agree either
      way).  The ceiling is a display default, not an acquisition fact;
      reduction-time masking keys off the acquired frame's own dtype.
    - Runs even when the booleans are already exclusive: the defaulted-bound
      gap (Codex DESIGN_STOP) was a boolean-canonical pair whose cleared
      bounds stored None while the projection displayed substituted
      defaults, so the run executed a band the panel never showed.
    - No min<=max guard here: materialization mirrors the display verbatim,
      and a nonsensical band refuses LOUDLY at freeze
      (``FrozenThresholdPolicy`` validation) instead of being silently
      un-materialized.
    """
    threshold = intent.threshold
    changed = False
    if bool(threshold.apply_threshold) == bool(threshold.mask_saturation):
        threshold.apply_threshold = not bool(threshold.mask_saturation)
        changed = True
    if threshold.apply_threshold:
        if threshold.threshold_min is None:
            threshold.threshold_min = 0.0
            changed = True
        if threshold.threshold_max is None:
            ceiling = poni_saturation_ceiling(intent.poni_file)
            if ceiling is not None:
                threshold.threshold_max = ceiling
                changed = True
    return changed


def commit_canonical_threshold(
    store: RunIntentStore, snapshot: RunIntentSnapshot
) -> RunIntentSnapshot | None:
    """Run the shared canonicalizer THROUGH the revisioned store and return
    the canonical snapshot (``snapshot`` itself when already canonical).

    Hosted HERE rather than in the start pipeline because candidate
    mutation belongs to the editing module: callers hand in a snapshot and
    receive a snapshot back, so no raw thawed ``RunIntent`` ever crosses a
    kernel-module boundary (the semantic architecture guard permits a raw
    intent outside the reducer only as the direct candidate of a store
    ``commit``).  The bounded retry re-runs only on a genuine concurrent-
    revision race; exhaustion returns ``None`` and the caller must refuse
    with a typed outcome — a raw non-canonical capture is never produced.
    """
    for _ in range(3):
        intent = snapshot.thaw()
        if not canonicalize_threshold_intent(intent):
            return snapshot
        result = store.commit(intent, expected_revision=snapshot.revision)
        if isinstance(result, IntentCommitAccepted):
            return result.snapshot
        snapshot = result.snapshot
    return None


def _normalize_gi_units(intent: RunIntent, *, was_enabled: bool) -> None:
    current_1d = str(intent.bai_1d_args.get("unit") or "q_A^-1")
    current_2d = str(intent.bai_2d_args.get("unit") or "q_A^-1")
    prior_1d = _axis_semantic(was_enabled, intent.gi.mode_1d, current_1d)
    prior_2d = _axis_semantic(was_enabled, intent.gi.mode_2d, current_2d)
    if intent.gi.enabled:
        unit_1d = (
            "qip_A^-1"
            if intent.gi.mode_1d == "q_ip"
            else "qoop_A^-1"
            if intent.gi.mode_1d == "q_oop"
            else "q_A^-1"
        )
        unit_2d = (
            "qip_A^-1"
            if intent.gi.mode_2d == "qip_qoop"
            else "q_A^-1"
        )
    else:
        unit_1d = (
            current_1d
            if current_1d in STANDARD_1D_AXES.values()
            else "q_A^-1"
        )
        unit_2d = (
            current_2d
            if current_2d in STANDARD_2D_AXES.values()
            else "q_A^-1"
        )
    next_1d = _axis_semantic(intent.gi.enabled, intent.gi.mode_1d, unit_1d)
    next_2d = _axis_semantic(intent.gi.enabled, intent.gi.mode_2d, unit_2d)
    _normalize_unit(
        intent.bai_1d_args,
        unit_1d,
        clear_range=prior_1d != next_1d,
    )
    _normalize_unit(
        intent.bai_2d_args,
        unit_2d,
        clear_range=prior_2d != next_2d,
    )


def _axis_semantic(enabled: bool, mode: str, unit: str) -> str:
    if not enabled:
        return unit
    return {
        "q_total": "q_A^-1",
        "q_chi": "q_A^-1",
        "q_ip": "qip_A^-1",
        "qip_qoop": "qip_A^-1",
        "q_oop": "qoop_A^-1",
    }.get(mode, f"gi:{mode}")


def _normalize_unit(
    values: dict[object, object], unit: str, *, clear_range: bool
) -> None:
    values["unit"] = unit
    if clear_range:
        values.pop("radial_range", None)


def _reduce_integration_edit(
    snapshot: RunIntentSnapshot,
    path: tuple[str, ...],
    value: object,
) -> EditResult:
    intent = snapshot.thaw()
    current_values = integration_values(intent)
    current = current_values[path]
    candidate = snapshot.thaw()

    if path in {
        INT_1D_POINTS,
        INT_2D_RADIAL_POINTS,
        INT_2D_AZIM_POINTS,
    }:
        parsed = _integer(value)
        if isinstance(parsed, EditRefusal) or parsed <= 0:
            return EditRefusal("Integration points must be a positive integer.")
        # A loaded GI intent may carry an intentional asymmetric ``npt_oop``
        # for the future Advanced editor.  Re-entering the visible main value
        # is still an edit when that hidden dimension differs: the compact
        # one-Pts surface owns the symmetric setting.
        gi_1d_grid_already_symmetric = (
            path != INT_1D_POINTS
            or not intent.gi.enabled
            or intent.bai_1d_args.get("npt_oop", parsed) == parsed
        )
        if parsed == current and gi_1d_grid_already_symmetric:
            return EditNoChange()
        if path == INT_1D_POINTS:
            _install_alias(
                candidate.bai_1d_args,
                ("npt", "numpoints", "npt_rad"),
                "npt",
                parsed,
            )
            # The compact GI surface deliberately exposes one point count, as
            # production does.  A main-form edit therefore re-synchronizes the
            # two FiberIntegrator dimensions.  An asymmetric ``npt_oop`` loaded
            # from an existing intent remains untouched until the user edits
            # this field, preserving the value for a future fully-owned
            # Advanced override instead of exposing a half-wired second box.
            if intent.gi.enabled:
                candidate.bai_1d_args["npt_oop"] = parsed
        else:
            radial, azimuthal = points_2d(candidate.bai_2d_args)
            if path == INT_2D_RADIAL_POINTS:
                radial = parsed
            else:
                azimuthal = parsed
            candidate.bai_2d_args.pop("npt", None)
            candidate.bai_2d_args["npt_rad"] = radial
            candidate.bai_2d_args["npt_azim"] = azimuthal
        return candidate

    if path in {INT_1D_AXIS, INT_2D_AXIS}:
        if type(value) is not str:
            return EditRefusal("Choose a supported integration axis.")
        axes = (
            GI_1D_AXES
            if intent.gi.enabled and path == INT_1D_AXIS
            else GI_2D_AXES
            if intent.gi.enabled
            else STANDARD_1D_AXES
            if path == INT_1D_AXIS
            else STANDARD_2D_AXES
        )
        selected = axes.get(value)
        if selected is None:
            return EditRefusal("Choose a supported integration axis.")
        if value == current:
            return EditNoChange()
        if intent.gi.enabled:
            if path == INT_1D_AXIS:
                candidate.gi.mode_1d = selected
                candidate.bai_1d_args["unit"] = (
                    "qip_A^-1"
                    if selected == "q_ip"
                    else "qoop_A^-1"
                    if selected == "q_oop"
                    else "q_A^-1"
                )
                candidate.bai_1d_args.pop("radial_range", None)
            else:
                candidate.gi.mode_2d = selected
                candidate.bai_2d_args["unit"] = (
                    "qip_A^-1"
                    if selected == "qip_qoop"
                    else "q_A^-1"
                )
                candidate.bai_2d_args.pop("radial_range", None)
        elif path == INT_1D_AXIS:
            candidate.bai_1d_args["unit"] = selected
            candidate.bai_1d_args.pop("radial_range", None)
        else:
            candidate.bai_2d_args["unit"] = selected
            candidate.bai_2d_args.pop("radial_range", None)
        return candidate

    if path in {
        INT_1D_RADIAL_AUTO,
        INT_1D_AZIM_AUTO,
        INT_2D_RADIAL_AUTO,
        INT_2D_AZIM_AUTO,
    }:
        if type(value) is not bool:
            return EditRefusal("An Auto control requires true or false.")
        if value is current:
            return EditNoChange()
        args, key = _range_owner(candidate, path)
        if value:
            args.pop(key, None)
        else:
            root, name = path
            dimension = name.removesuffix("_auto")
            args[key] = (
                float(current_values[(root, f"{dimension}_low")]),
                float(current_values[(root, f"{dimension}_high")]),
            )
        return candidate

    parsed = _finite_float(value)
    if isinstance(parsed, EditRefusal):
        return parsed
    if parsed == current:
        return EditNoChange()
    root, leaf = path
    dimension, edge = leaf.rsplit("_", 1)
    other_edge = "high" if edge == "low" else "low"
    other = float(current_values[(root, f"{dimension}_{other_edge}")])
    low, high = (parsed, other) if edge == "low" else (other, parsed)
    if low > high:
        return EditRefusal("Integration range minimum cannot exceed maximum.")
    args, key = _range_owner(candidate, path)
    args[key] = (low, high)
    return candidate


def _range_owner(
    intent: RunIntent,
    path: tuple[str, ...],
) -> tuple[dict[object, object], str]:
    root, leaf = path
    args = intent.bai_1d_args if root == "Int1D" else intent.bai_2d_args
    key = "radial_range" if leaf.startswith("radial_") else "azimuth_range"
    return args, key


def _install_alias(
    values: dict[object, object],
    aliases: tuple[str, ...],
    default: str,
    value: object,
) -> None:
    owner = next((alias for alias in aliases if alias in values), default)
    for alias in aliases:
        if alias != owner:
            values.pop(alias, None)
    values[owner] = value


def _reduce_source_edit(
    snapshot: RunIntentSnapshot,
    path: tuple[str, ...],
    value: object,
) -> EditResult:
    candidate = snapshot.thaw()
    source = candidate.source_spec
    if path == SOURCE_META:
        metadata_format = _metadata_format_value(value)
        if isinstance(metadata_format, EditRefusal):
            return metadata_format
        if type(source) is DirectorySourceSpec:
            if metadata_format == source.metadata_format:
                return EditNoChange()
            candidate.source_spec = DirectorySourceSpec(
                root=source.root,
                recursive=source.recursive,
                suffixes=source.suffixes,
                name_filter=source.name_filter,
                generation=source.generation + 1,
                metadata_format=metadata_format,
            )
            return candidate
        if type(source) is SourceSpec:
            options = dict(source.options)
            current = options.get("metadata_format", "auto")
            if "metadata_format" in options and metadata_format == current:
                return EditNoChange()
            options["metadata_format"] = metadata_format
            candidate.source_spec = SourceSpec(
                uri=source.uri,
                kind=source.kind,
                metadata_uri=source.metadata_uri,
                entry=source.entry,
                options=options,
            )
            return candidate
        return EditRefusal("Choose an image source before editing Meta Type.")
    if type(source) is not DirectorySourceSpec:
        return EditRefusal(
            "Choose an Image Directory source before editing directory options."
        )
    root = source.root
    recursive = source.recursive
    suffixes = source.suffixes
    name_filter = source.name_filter
    if path == SOURCE_DIRECTORY:
        if type(value) is not str or not value.strip():
            return EditRefusal("A source directory path is required.")
        root = Path(value).expanduser()
    elif path == SOURCE_RECURSIVE:
        if type(value) is not bool:
            return EditRefusal("Subdirs requires true or false.")
        recursive = value
    elif path == SOURCE_SUFFIX:
        if source.suffixes not in SOURCE_FORMAT_SUFFIXES.values():
            return EditRefusal(
                "Use Choose source to replace this exact complete source."
            )
        if type(value) is not str or value not in SOURCE_FORMAT_SUFFIXES:
            return EditRefusal("Choose a file type.")
        suffixes = SOURCE_FORMAT_SUFFIXES[value]
    else:
        if type(value) is not str:
            return EditRefusal("A source filter must be text.")
        name_filter = value.strip() or None
    desired = (root, recursive, suffixes, name_filter)
    current = (
        source.root,
        source.recursive,
        source.suffixes,
        source.name_filter,
    )
    if desired == current:
        return EditNoChange()
    candidate.source_spec = DirectorySourceSpec(
        root=root,
        recursive=recursive,
        suffixes=suffixes,
        name_filter=name_filter,
        generation=source.generation + 1,
        metadata_format=source.metadata_format,
    )
    return candidate


def _metadata_format_value(value: object) -> str | None | EditRefusal:
    if type(value) is not str:
        return EditRefusal("Choose a metadata type.")
    normalized = value.strip().lower()
    if normalized == "none":
        return None
    if normalized in {"auto", "txt", "pdi", "metadata", "spec"}:
        return normalized
    return EditRefusal("Choose a metadata type.")


def reduce_source_selection(
    snapshot: RunIntentSnapshot,
    source: SourceSelection,
) -> EditResult:
    """Replace the complete selected source in one candidate value."""

    if type(source) not in {SourceSpec, DirectorySourceSpec}:
        return EditRefusal("Choose a complete supported source.")
    candidate = snapshot.thaw()
    if candidate.source_spec == source:
        return EditNoChange()
    candidate.source_spec = source
    return candidate


def _control_value(
    intent: RunIntent,
    path: tuple[str, ...],
    value: object,
) -> object | EditRefusal:
    if path in {PROJECT_ROOT, SAVE_PATH, PONI_FILE, MASK_FILE}:
        return value if type(value) is str else EditRefusal("A path must be text.")
    if path == OUTPUT_MODE:
        return (
            value
            if value == "Overwrite"
            else EditRefusal(APPEND_UNAVAILABLE)
        )
    if path in {GI_ENABLED, THRESHOLD_ENABLED, MASK_SATURATION}:
        return (
            value
            if type(value) is bool
            else EditRefusal("A boolean control requires true or false.")
        )
    if path == GI_MOTOR:
        if type(value) is not str or not value:
            return EditRefusal("Unknown incidence-motor selection.")
        return value
    if path in {GI_THETA, GI_TILT}:
        return _finite_float(value)
    if path == GI_ORIENTATION:
        parsed = _integer(value)
        if isinstance(parsed, EditRefusal) or not 1 <= parsed <= 8:
            return EditRefusal(
                "Sample orientation must be an integer from 1 through 8."
            )
        return parsed
    if path in {THRESHOLD_MIN, THRESHOLD_MAX}:
        if type(value) is str and not value.strip():
            return None
        return _finite_float(value)
    return EditRefusal("Unknown control.")


def _finite_float(value: object) -> float | EditRefusal:
    if type(value) is bool:
        return EditRefusal("A finite numeric value is required.")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return EditRefusal("A finite numeric value is required.")
    return (
        parsed
        if math.isfinite(parsed)
        else EditRefusal("A finite numeric value is required.")
    )


def _integer(value: object) -> int | EditRefusal:
    if type(value) is int:
        return value
    if type(value) is str and value.strip().lstrip("+-").isdigit():
        return int(value)
    return EditRefusal("An integer value is required.")


def _current_value(intent: RunIntent, path: tuple[str, ...]) -> object:
    values = {
        PROJECT_ROOT: intent.project_root,
        SAVE_PATH: intent.save_path,
        OUTPUT_MODE: intent.output_mode,
        PONI_FILE: intent.poni_file,
        MASK_FILE: intent.mask_file,
        GI_ENABLED: intent.gi.enabled,
        GI_MOTOR: intent.gi.incidence_motor,
        GI_THETA: intent.gi.th_val,
        GI_ORIENTATION: intent.gi.sample_orientation,
        GI_TILT: intent.gi.tilt_angle,
        THRESHOLD_ENABLED: intent.threshold.apply_threshold,
        THRESHOLD_MIN: intent.threshold.threshold_min,
        THRESHOLD_MAX: intent.threshold.threshold_max,
        MASK_SATURATION: intent.threshold.mask_saturation,
    }
    return values[path]


def _install_value(
    intent: RunIntent,
    path: tuple[str, ...],
    value: object,
) -> None:
    if path == PROJECT_ROOT:
        intent.project_root = value  # type: ignore[assignment]
    elif path == SAVE_PATH:
        intent.save_path = value  # type: ignore[assignment]
    elif path == OUTPUT_MODE:
        intent.output_mode = value  # type: ignore[assignment]
    elif path == PONI_FILE:
        intent.poni_file = value  # type: ignore[assignment]
    elif path == MASK_FILE:
        intent.mask_file = value  # type: ignore[assignment]
    elif path == GI_ENABLED:
        intent.gi.enabled = value  # type: ignore[assignment]
    elif path == GI_MOTOR:
        intent.gi.incidence_motor = value  # type: ignore[assignment]
    elif path == GI_THETA:
        intent.gi.th_val = value  # type: ignore[assignment]
    elif path == GI_ORIENTATION:
        intent.gi.sample_orientation = value  # type: ignore[assignment]
    elif path == GI_TILT:
        intent.gi.tilt_angle = value  # type: ignore[assignment]
    elif path == THRESHOLD_ENABLED:
        intent.threshold.apply_threshold = value  # type: ignore[assignment]
    elif path == THRESHOLD_MIN:
        intent.threshold.threshold_min = value  # type: ignore[assignment]
    elif path == THRESHOLD_MAX:
        intent.threshold.threshold_max = value  # type: ignore[assignment]
    else:
        intent.threshold.mask_saturation = value  # type: ignore[assignment]
