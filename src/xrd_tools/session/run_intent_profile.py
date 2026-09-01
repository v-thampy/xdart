"""Versioned, Qt-free JSON profiles for editable :class:`RunIntent` values.

Profiles describe the next-run intent only.  Run generations, frozen
configuration fingerprints, and runtime-resolved values deliberately never
cross this boundary.  Loading returns a fully decoded and validated candidate
whose generation is zero; committing that candidate remains the intent store's
responsibility.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import SourceSpec
from xrd_tools.reduction.background import FrameBackgroundPlan
from xrd_tools.reduction.provenance_config import jsonable_run_value
from xrd_tools.session.run_configuration import (
    GIIntent,
    PoniV3OverrideIntent,
    RunIntent,
    ThresholdIntent,
)
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
)


PROFILE_SCHEMA = "xdart.run-intent-profile"
PROFILE_VERSION = 1

_ENVELOPE_KEYS = frozenset({"schema", "version", "intent"})
_INTENT_KEYS = frozenset({
    "source_spec",
    "processing_mode",
    "output_mode",
    "live_mode",
    "batch_mode",
    "max_cores",
    "bai_1d_args",
    "bai_2d_args",
    "gi",
    "threshold",
    "poni_file",
    "poni_values",
    "mask_file",
    "background",
    "project_root",
    "save_path",
    "run_options",
})
_GI_KEYS = frozenset({
    "enabled",
    "incidence_motor",
    "th_val",
    "sample_orientation",
    "tilt_angle",
    "mode_1d",
    "mode_2d",
})
_THRESHOLD_KEYS = frozenset({
    "apply_threshold",
    "threshold_min",
    "threshold_max",
    "mask_saturation",
})
_DIRECTORY_SOURCE_KEYS = frozenset({
    "family",
    "root",
    "recursive",
    "suffixes",
    "name_filter",
    "metadata_format",
})
_SOURCE_KEYS = frozenset({
    "family",
    "uri",
    "uri_is_path",
    "kind",
    "metadata_uri",
    "metadata_uri_is_path",
    "entry",
    "options",
})
_VALUE_TAG = "$xdart_profile_type"


class RunIntentProfileError(ValueError):
    """A profile is malformed, unsupported, or not a valid next-run intent."""


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RunIntentProfileError(f"{path} must be a JSON object")
    return value


def _keyset(value: Mapping[str, Any], expected: frozenset[str], *, path: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise RunIntentProfileError(
            f"{path} has an invalid keyset (missing={missing}, extra={extra})"
        )


def _text(value: object, *, path: str, empty: bool = True) -> str:
    if type(value) is not str or (not empty and not value.strip()):
        qualifier = "nonempty " if not empty else ""
        raise RunIntentProfileError(f"{path} must be {qualifier}text")
    return value


def _optional_text(value: object, *, path: str) -> str | None:
    if value is None:
        return None
    return _text(value, path=path)


def _boolean(value: object, *, path: str) -> bool:
    if type(value) is not bool:
        raise RunIntentProfileError(f"{path} must be a JSON boolean")
    return value


def _integer(value: object, *, path: str) -> int:
    if type(value) is not int:
        raise RunIntentProfileError(f"{path} must be an integer")
    return value


def _number(value: object, *, path: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise RunIntentProfileError(f"{path} must be a finite number")
    return float(value)


def _optional_number(value: object, *, path: str) -> float | None:
    if value is None:
        return None
    return _number(value, path=path)


def _json_mapping(value: object, *, path: str) -> dict[str, Any]:
    raw = _mapping(value, path=path)
    try:
        projected = jsonable_run_value(raw, path=path)
    except (TypeError, ValueError) as error:
        raise RunIntentProfileError(str(error)) from error
    if type(projected) is not dict:  # pragma: no cover - guarded above
        raise RunIntentProfileError(f"{path} must be a JSON object")
    return projected


def _encode_profile_value(value: object, *, path: str) -> Any:
    """Encode JSON-native values while preserving tuple and Path semantics."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RunIntentProfileError(f"{path} must not contain a non-finite number")
        return value
    if isinstance(value, Path):
        return {_VALUE_TAG: "path", "value": str(value)}
    if isinstance(value, tuple):
        return {
            _VALUE_TAG: "tuple",
            "items": [
                _encode_profile_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, list):
        return [
            _encode_profile_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        for key in value:
            if type(key) is not str:
                raise RunIntentProfileError(
                    f"{path} must contain only text object keys"
                )
        if _VALUE_TAG in value:
            return {
                _VALUE_TAG: "mapping",
                "items": [
                    [key, _encode_profile_value(item, path=f"{path}.{key}")]
                    for key, item in value.items()
                ],
            }
        return {
            key: _encode_profile_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    raise RunIntentProfileError(
        f"{path} contains unsupported {type(value).__module__}."
        f"{type(value).__qualname__}"
    )


def _decode_profile_value(value: object, *, path: str) -> Any:
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):  # pragma: no cover - parse hook rejects it
            raise RunIntentProfileError(f"{path} must not contain a non-finite number")
        return value
    if type(value) is list:
        return [
            _decode_profile_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is not dict:
        raise RunIntentProfileError(f"{path} contains a non-JSON value")
    if _VALUE_TAG not in value:
        return {
            key: _decode_profile_value(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    tag = value[_VALUE_TAG]
    if tag == "path" and set(value) == {_VALUE_TAG, "value"}:
        return Path(_text(value["value"], path=f"{path}.value"))
    if tag == "tuple" and set(value) == {_VALUE_TAG, "items"}:
        items = value["items"]
        if type(items) is not list:
            raise RunIntentProfileError(f"{path}.items must be an array")
        return tuple(
            _decode_profile_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(items)
        )
    if tag == "mapping" and set(value) == {_VALUE_TAG, "items"}:
        items = value["items"]
        if type(items) is not list:
            raise RunIntentProfileError(f"{path}.items must be an array")
        result: dict[str, Any] = {}
        for index, pair in enumerate(items):
            if (
                type(pair) is not list
                or len(pair) != 2
                or type(pair[0]) is not str
                or pair[0] in result
            ):
                raise RunIntentProfileError(
                    f"{path}.items[{index}] is not a unique text-key pair"
                )
            result[pair[0]] = _decode_profile_value(
                pair[1], path=f"{path}.{pair[0]}"
            )
        return result
    raise RunIntentProfileError(f"{path} has an invalid profile value tag")


def _profile_mapping(value: object, *, path: str) -> dict[str, Any]:
    decoded = _decode_profile_value(value, path=path)
    if type(decoded) is not dict:
        raise RunIntentProfileError(f"{path} must be a JSON object")
    return decoded


def _encode_source(
    source: SourceSpec | DirectorySourceSpec | None,
) -> dict[str, Any] | None:
    if source is None:
        return None
    if type(source) is DirectorySourceSpec:
        return {
            "family": "directory",
            "root": str(source.root),
            "recursive": bool(source.recursive),
            "suffixes": list(source.suffixes),
            "name_filter": source.name_filter,
            "metadata_format": source.metadata_format,
        }
    if type(source) is SourceSpec:
        metadata_uri = source.metadata_uri
        return {
            "family": "source",
            "uri": str(source.uri),
            "uri_is_path": isinstance(source.uri, Path),
            "kind": str(getattr(source.kind, "value", source.kind)),
            "metadata_uri": (
                None if metadata_uri is None else str(metadata_uri)
            ),
            "metadata_uri_is_path": isinstance(metadata_uri, Path),
            "entry": source.entry,
            "options": _encode_profile_value(
                dict(source.options), path="intent.source_spec.options"
            ),
        }
    raise TypeError("source_spec must be SourceSpec, DirectorySourceSpec, or None")


def _decode_source(value: object) -> SourceSpec | DirectorySourceSpec | None:
    if value is None:
        return None
    source = _mapping(value, path="intent.source_spec")
    family = _text(source.get("family"), path="intent.source_spec.family")
    if family == "directory":
        _keyset(source, _DIRECTORY_SOURCE_KEYS, path="intent.source_spec")
        suffixes = source["suffixes"]
        if type(suffixes) is not list or any(
            type(item) is not str or not item for item in suffixes
        ):
            raise RunIntentProfileError(
                "intent.source_spec.suffixes must be an array of nonempty text"
            )
        result = DirectorySourceSpec(
            Path(_text(source["root"], path="intent.source_spec.root", empty=False)),
            recursive=_boolean(
                source["recursive"], path="intent.source_spec.recursive"
            ),
            suffixes=tuple(suffixes),
            name_filter=_optional_text(
                source["name_filter"], path="intent.source_spec.name_filter"
            ),
            # A directory observer's revision is runtime state, not profile state.
            generation=0,
            metadata_format=_optional_text(
                source["metadata_format"],
                path="intent.source_spec.metadata_format",
            ),
        )
        if list(result.suffixes) != suffixes:
            raise RunIntentProfileError(
                "intent.source_spec.suffixes are not canonical"
            )
        return result
    if family != "source":
        raise RunIntentProfileError(
            "intent.source_spec.family must be 'source' or 'directory'"
        )
    _keyset(source, _SOURCE_KEYS, path="intent.source_spec")
    uri_text = _text(source["uri"], path="intent.source_spec.uri", empty=False)
    uri_is_path = _boolean(
        source["uri_is_path"], path="intent.source_spec.uri_is_path"
    )
    metadata_text = _optional_text(
        source["metadata_uri"], path="intent.source_spec.metadata_uri"
    )
    metadata_is_path = _boolean(
        source["metadata_uri_is_path"],
        path="intent.source_spec.metadata_uri_is_path",
    )
    if metadata_text is None and metadata_is_path:
        raise RunIntentProfileError(
            "intent.source_spec.metadata_uri_is_path requires metadata_uri"
        )
    options = _profile_mapping(
        source["options"], path="intent.source_spec.options"
    )
    try:
        return SourceSpec(
            Path(uri_text) if uri_is_path else uri_text,
            _text(source["kind"], path="intent.source_spec.kind", empty=False),
            metadata_uri=(
                None
                if metadata_text is None
                else Path(metadata_text) if metadata_is_path else metadata_text
            ),
            entry=_optional_text(
                source["entry"], path="intent.source_spec.entry"
            ),
            options=options,
        )
    except (TypeError, ValueError) as error:
        raise RunIntentProfileError(
            f"intent.source_spec is invalid: {error}"
        ) from error


def _decode_gi(value: object) -> GIIntent:
    data = _mapping(value, path="intent.gi")
    _keyset(data, _GI_KEYS, path="intent.gi")
    result = GIIntent(
        enabled=_boolean(data["enabled"], path="intent.gi.enabled"),
        incidence_motor=_text(
            data["incidence_motor"],
            path="intent.gi.incidence_motor",
            empty=False,
        ),
        th_val=_number(data["th_val"], path="intent.gi.th_val"),
        sample_orientation=_integer(
            data["sample_orientation"],
            path="intent.gi.sample_orientation",
        ),
        tilt_angle=_number(data["tilt_angle"], path="intent.gi.tilt_angle"),
        mode_1d=_text(data["mode_1d"], path="intent.gi.mode_1d", empty=False),
        mode_2d=_text(data["mode_2d"], path="intent.gi.mode_2d", empty=False),
    )
    try:
        result.freeze(choices=None)
    except (TypeError, ValueError) as error:
        raise RunIntentProfileError(f"intent.gi is invalid: {error}") from error
    return result


def _decode_threshold(value: object) -> ThresholdIntent:
    data = _mapping(value, path="intent.threshold")
    _keyset(data, _THRESHOLD_KEYS, path="intent.threshold")
    result = ThresholdIntent(
        apply_threshold=_boolean(
            data["apply_threshold"], path="intent.threshold.apply_threshold"
        ),
        threshold_min=_optional_number(
            data["threshold_min"], path="intent.threshold.threshold_min"
        ),
        threshold_max=_optional_number(
            data["threshold_max"], path="intent.threshold.threshold_max"
        ),
        mask_saturation=_boolean(
            data["mask_saturation"], path="intent.threshold.mask_saturation"
        ),
    )
    try:
        result.freeze()
    except (TypeError, ValueError) as error:
        raise RunIntentProfileError(
            f"intent.threshold is invalid: {error}"
        ) from error
    return result


def _validate_candidate(candidate: RunIntent) -> RunIntent:
    """Validate through the run boundary without advancing the candidate."""
    if type(candidate.processing_mode) is not str or not candidate.processing_mode.strip():
        raise RunIntentProfileError("intent.processing_mode must be nonempty text")
    try:
        candidate.clone_candidate().freeze()
    except (TypeError, ValueError, OverflowError) as error:
        raise RunIntentProfileError(f"profile run intent is invalid: {error}") from error
    if candidate.generation != 0:
        raise RunIntentProfileError("loaded profile candidate generation must be zero")
    return candidate


def _decode_versioned(document: dict[str, Any]) -> RunIntent:
    _keyset(document, _ENVELOPE_KEYS, path="profile")
    if document["schema"] != PROFILE_SCHEMA:
        raise RunIntentProfileError("profile schema is unsupported")
    if type(document["version"]) is not int or document["version"] != PROFILE_VERSION:
        raise RunIntentProfileError("profile version is unsupported")
    data = _mapping(document["intent"], path="intent")
    intent_keys = set(data)
    if (
        intent_keys != set(_INTENT_KEYS)
        and intent_keys != set(_INTENT_KEYS) | {"poni_v3_override"}
    ):
        missing = sorted(_INTENT_KEYS - intent_keys)
        extra = sorted(intent_keys - _INTENT_KEYS - {"poni_v3_override"})
        raise RunIntentProfileError(
            "intent has an invalid keyset "
            f"(missing={missing}, extra={extra})"
        )
    output_mode = _text(
        data["output_mode"], path="intent.output_mode", empty=False
    )
    if output_mode not in {"Append", "Overwrite"}:
        raise RunIntentProfileError(
            "intent.output_mode must be 'Append' or 'Overwrite'"
        )
    max_cores = _integer(data["max_cores"], path="intent.max_cores")
    if max_cores < 1:
        raise RunIntentProfileError("intent.max_cores must be at least 1")
    live_mode = _boolean(data["live_mode"], path="intent.live_mode")
    batch_mode = _boolean(data["batch_mode"], path="intent.batch_mode")
    if live_mode and batch_mode:
        raise RunIntentProfileError(
            "intent.live_mode and intent.batch_mode cannot both be enabled"
        )
    poni_values = data["poni_values"]
    if poni_values is not None:
        poni_values = _profile_mapping(
            poni_values, path="intent.poni_values"
        )
    override = None
    if "poni_v3_override" in data:
        try:
            override = PoniV3OverrideIntent.from_mapping(
                _mapping(
                    data["poni_v3_override"],
                    path="intent.poni_v3_override",
                )
            )
        except (TypeError, ValueError) as error:
            raise RunIntentProfileError(
                f"intent.poni_v3_override is invalid: {error}"
            ) from error
    try:
        background = FrameBackgroundPlan.from_mapping(
            _mapping(data["background"], path="intent.background")
        )
    except (TypeError, ValueError) as error:
        raise RunIntentProfileError(
            f"intent.background is invalid: {error}"
        ) from error
    candidate = RunIntent(
        source_spec=_decode_source(data["source_spec"]),
        processing_mode=_text(
            data["processing_mode"],
            path="intent.processing_mode",
            empty=False,
        ),
        output_mode=output_mode,
        live_mode=live_mode,
        batch_mode=batch_mode,
        max_cores=max_cores,
        bai_1d_args=_profile_mapping(
            data["bai_1d_args"], path="intent.bai_1d_args"
        ),
        bai_2d_args=_profile_mapping(
            data["bai_2d_args"], path="intent.bai_2d_args"
        ),
        gi=_decode_gi(data["gi"]),
        threshold=_decode_threshold(data["threshold"]),
        poni_file=_text(data["poni_file"], path="intent.poni_file"),
        poni_values=poni_values,
        mask_file=_text(data["mask_file"], path="intent.mask_file"),
        poni_v3_override=override,
        background=background,
        project_root=_text(data["project_root"], path="intent.project_root"),
        save_path=_text(data["save_path"], path="intent.save_path"),
        run_options=_profile_mapping(
            data["run_options"], path="intent.run_options"
        ),
        generation=0,
    )
    return _validate_candidate(candidate)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RunIntentProfileError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise RunIntentProfileError(f"non-finite JSON constant {value!r} is unsupported")


def dump_run_intent_profile(intent: RunIntent) -> str:
    """Return a deterministic version-1 JSON profile for *intent*."""

    if type(intent) is not RunIntent:
        raise TypeError("intent must be a RunIntent")
    try:
        frozen = intent.clone_candidate().freeze()
        if type(frozen.processing_mode) is not str or not frozen.processing_mode.strip():
            raise RunIntentProfileError("intent.processing_mode must be nonempty text")
        document = {
            "schema": PROFILE_SCHEMA,
            "version": PROFILE_VERSION,
            "intent": {
                "source_spec": _encode_source(frozen.thaw_source_spec()),
                "processing_mode": frozen.processing_mode,
                "output_mode": frozen.output_mode,
                "live_mode": bool(frozen.live_mode),
                "batch_mode": bool(frozen.batch_mode),
                "max_cores": int(frozen.max_cores),
                "bai_1d_args": _encode_profile_value(
                    frozen.bai_1d_args, path="intent.bai_1d_args"
                ),
                "bai_2d_args": _encode_profile_value(
                    frozen.bai_2d_args, path="intent.bai_2d_args"
                ),
                "gi": {
                    "enabled": bool(frozen.gi.enabled),
                    "incidence_motor": frozen.gi.incidence_motor,
                    "th_val": float(frozen.gi.th_val),
                    "sample_orientation": int(frozen.gi.sample_orientation),
                    "tilt_angle": float(frozen.gi.tilt_angle),
                    "mode_1d": frozen.gi.mode_1d,
                    "mode_2d": frozen.gi.mode_2d,
                },
                "threshold": frozen.threshold.as_dict(),
                "poni_file": frozen.poni_file,
                "poni_values": _encode_profile_value(
                    frozen.poni_values, path="intent.poni_values"
                ),
                "mask_file": frozen.mask_file,
                "background": frozen.background.to_mapping(),
                "project_root": frozen.project_root,
                "save_path": frozen.save_path,
                "run_options": _encode_profile_value(
                    frozen.run_options, path="intent.run_options"
                ),
            },
        }
        if frozen.poni_v3_override is not None:
            document["intent"]["poni_v3_override"] = (
                frozen.poni_v3_override.as_dict()
            )
        return json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
    except RunIntentProfileError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise RunIntentProfileError(f"cannot encode run-intent profile: {error}") from error


def load_run_intent_profile(text: str) -> RunIntent:
    """Decode and fully validate one versioned vNext profile document."""

    if type(text) is not str:
        raise TypeError("profile text must be str")
    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
        document = _mapping(document, path="profile")
        return _decode_versioned(document)
    except RunIntentProfileError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError, OverflowError) as error:
        raise RunIntentProfileError(f"cannot decode run-intent profile: {error}") from error


__all__ = [
    "PROFILE_SCHEMA",
    "PROFILE_VERSION",
    "RunIntentProfileError",
    "dump_run_intent_profile",
    "load_run_intent_profile",
]
