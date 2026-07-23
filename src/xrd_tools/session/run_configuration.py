"""Immutable run-boundary configuration values.

The GUI may edit a :class:`RunIntent` over time, but a processing run must not
consult mutable widgets, legacy ParameterTrees, or a display ``LiveScan`` after
the operator presses Run.  ``RunIntent.freeze()`` therefore creates one deeply
immutable :class:`FrozenRunConfiguration` that can be shared by the wrangler,
worker, reduction-plan builder, and writer provenance.

This module is deliberately Qt-free and import-light.  It stores arbitrary
integration argument mappings as a tagged immutable value tree, then returns
fresh mutable copies through explicit accessors.  A content fingerprint excludes
the run generation: two successive runs with identical settings have the same
fingerprint while their ``generation`` values remain distinct.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import SourceSpec
from xrd_tools.sources.selection import DirectorySourceSpec


_SCHEMA_VERSION = 1
_FrozenValue = tuple[Any, ...]


def _float_token(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return value.hex()


def _float_from_token(value: str) -> float:
    if value == "nan":
        return float("nan")
    if value == "+inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    return float.fromhex(value)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    )


def _freeze_value(value: Any) -> _FrozenValue:
    """Return a tagged, recursively immutable representation of *value*.

    Unsupported objects fail closed instead of falling back to ``repr``; an
    address-bearing repr would make fingerprints process-dependent.
    """

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        return ("float", _float_token(value))
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", base64.b64encode(value).decode("ascii"))
    if isinstance(value, Path):
        return ("path", str(value))
    if isinstance(value, Enum):
        return (
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            _freeze_value(value.value),
        )
    if isinstance(value, Mapping):
        items = [
            (_freeze_value(key), _freeze_value(item))
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: _canonical_json(pair[0]))
        return ("mapping", tuple(items))
    if isinstance(value, list):
        return ("list", tuple(_freeze_value(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze_value(item) for item in value))
    if isinstance(value, frozenset):
        items = [_freeze_value(item) for item in value]
        items.sort(key=_canonical_json)
        return ("frozenset", tuple(items))
    if isinstance(value, set):
        items = [_freeze_value(item) for item in value]
        items.sort(key=_canonical_json)
        return ("set", tuple(items))

    # NumPy scalars/arrays and similar value containers are accepted without
    # importing their libraries into this lightweight module.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _freeze_value(converted)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            converted = tolist()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _freeze_value(converted)

    raise TypeError(
        "run configuration values must be value-like; "
        f"unsupported {type(value).__module__}.{type(value).__qualname__}"
    )


def _thaw_value(value: _FrozenValue) -> Any:
    tag = value[0]
    if tag == "none":
        return None
    if tag == "bool":
        return bool(value[1])
    if tag == "int":
        return int(value[1])
    if tag == "float":
        return _float_from_token(value[1])
    if tag == "str":
        return str(value[1])
    if tag == "bytes":
        return base64.b64decode(value[1].encode("ascii"))
    if tag == "path":
        return Path(value[1])
    if tag == "enum":
        # Reconstructing arbitrary Enum classes would require dynamic imports.
        # Run consumers need the serialized value, not the original class.
        return _thaw_value(value[3])
    if tag == "mapping":
        return {
            _thaw_value(key): _thaw_value(item)
            for key, item in value[1]
        }
    if tag == "list":
        return [_thaw_value(item) for item in value[1]]
    if tag == "tuple":
        return tuple(_thaw_value(item) for item in value[1])
    if tag == "frozenset":
        return frozenset(_thaw_value(item) for item in value[1])
    if tag == "set":
        return {_thaw_value(item) for item in value[1]}
    raise ValueError(f"unknown frozen run-configuration value tag {tag!r}")


def _mapping_value(value: Mapping[Any, Any] | None) -> _FrozenValue:
    return _freeze_value(dict(value or {}))


def _thaw_mapping(value: _FrozenValue) -> dict[Any, Any]:
    thawed = _thaw_value(value)
    if not isinstance(thawed, dict):
        raise TypeError("frozen run-configuration value is not a mapping")
    return thawed


@dataclass(slots=True)
class GIIntent:
    """Mutable Controls-owned grazing-incidence intent."""

    enabled: bool = False
    incidence_motor: str = "Manual"
    th_val: float = 0.1
    sample_orientation: int = 4
    tilt_angle: float = 0.0
    mode_1d: str = "q_total"
    mode_2d: str = "qip_qoop"

    def freeze(self) -> "FrozenGIConfiguration":
        return FrozenGIConfiguration(
            enabled=bool(self.enabled),
            incidence_motor=str(self.incidence_motor or "Manual"),
            th_val=float(self.th_val),
            sample_orientation=int(self.sample_orientation),
            tilt_angle=float(self.tilt_angle),
            mode_1d=str(self.mode_1d or "q_total"),
            mode_2d=str(self.mode_2d or "qip_qoop"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "GIIntent":
        data = dict(value or {})
        return cls(
            enabled=bool(data.get("enabled", data.get("gi", False))),
            incidence_motor=str(
                data.get("incidence_motor", data.get("th_motor", "Manual"))
                or "Manual"
            ),
            th_val=float(data.get("th_val", 0.1) or 0.0),
            sample_orientation=int(data.get("sample_orientation", 4) or 4),
            tilt_angle=float(data.get("tilt_angle", 0.0) or 0.0),
            mode_1d=str(
                data.get("mode_1d", data.get("gi_mode_1d", "q_total"))
                or "q_total"
            ),
            mode_2d=str(
                data.get("mode_2d", data.get("gi_mode_2d", "qip_qoop"))
                or "qip_qoop"
            ),
        )


@dataclass(frozen=True, slots=True)
class FrozenGIConfiguration:
    """Deeply immutable grazing-incidence configuration for one run."""

    enabled: bool = False
    incidence_motor: str = "Manual"
    th_val: float = 0.1
    sample_orientation: int = 4
    tilt_angle: float = 0.0
    mode_1d: str = "q_total"
    mode_2d: str = "qip_qoop"

    def __post_init__(self) -> None:
        if int(self.sample_orientation) not in range(1, 9):
            raise ValueError("sample_orientation must be between 1 and 8")
        if not math.isfinite(float(self.th_val)):
            raise ValueError("th_val must be finite")
        if not math.isfinite(float(self.tilt_angle)):
            raise ValueError("tilt_angle must be finite")

    @property
    def scan_incidence_motor(self) -> str:
        if self.incidence_motor == "Manual":
            return str(float(self.th_val))
        return str(self.incidence_motor)

    def scan_config(self) -> dict[str, Any]:
        """Return a fresh ``LiveScan.gi_config`` mapping."""

        if not self.enabled:
            return {}
        return {
            "gi_mode_1d": self.mode_1d,
            "gi_mode_2d": self.mode_2d,
            "incidence_motor": self.incidence_motor,
            "th_val": float(self.th_val),
            "sample_orientation": int(self.sample_orientation),
            "tilt_angle": float(self.tilt_angle),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "incidence_motor": self.incidence_motor,
            "th_val": float(self.th_val),
            "sample_orientation": int(self.sample_orientation),
            "tilt_angle": float(self.tilt_angle),
            "mode_1d": self.mode_1d,
            "mode_2d": self.mode_2d,
        }


@dataclass(slots=True)
class ThresholdIntent:
    """Mutable Controls-owned pixel-rejection intent."""

    apply_threshold: bool = False
    threshold_min: float | None = None
    threshold_max: float | None = None
    mask_saturation: bool = True

    def freeze(self) -> "FrozenThresholdPolicy":
        return FrozenThresholdPolicy(
            apply_threshold=bool(self.apply_threshold),
            threshold_min=(
                None if self.threshold_min is None else float(self.threshold_min)
            ),
            threshold_max=(
                None if self.threshold_max is None else float(self.threshold_max)
            ),
            mask_saturation=bool(self.mask_saturation),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "ThresholdIntent":
        data = dict(value or {})
        return cls(
            apply_threshold=bool(data.get("apply_threshold", False)),
            threshold_min=data.get("threshold_min"),
            threshold_max=data.get("threshold_max"),
            mask_saturation=bool(data.get("mask_saturation", True)),
        )


@dataclass(frozen=True, slots=True)
class FrozenThresholdPolicy:
    """Deeply immutable threshold/saturation policy for one run."""

    apply_threshold: bool = False
    threshold_min: float | None = None
    threshold_max: float | None = None
    mask_saturation: bool = True

    def __post_init__(self) -> None:
        for name in ("threshold_min", "threshold_max"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
        if (
            self.threshold_min is not None
            and self.threshold_max is not None
            and float(self.threshold_min) > float(self.threshold_max)
        ):
            raise ValueError("threshold_min cannot exceed threshold_max")

    def as_dict(self) -> dict[str, Any]:
        return {
            "apply_threshold": bool(self.apply_threshold),
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
            "mask_saturation": bool(self.mask_saturation),
        }


@dataclass(frozen=True, slots=True)
class FrozenSourceSpec:
    """Deeply immutable value projection of a supported source selection."""

    family: str
    uri: str
    source_kind: str = ""
    metadata_uri: str | None = None
    entry: str | None = None
    options: _FrozenValue = ("mapping", ())
    recursive: bool = False
    suffixes: tuple[str, ...] = ()
    name_filter: str | None = None
    generation: int = 0
    uri_was_path: bool = True
    metadata_uri_was_path: bool = True

    @classmethod
    def from_source(
        cls,
        value: SourceSpec | DirectorySourceSpec,
    ) -> "FrozenSourceSpec":
        if isinstance(value, DirectorySourceSpec):
            return cls(
                family="directory",
                uri=str(value.root),
                recursive=bool(value.recursive),
                suffixes=tuple(str(item) for item in value.suffixes),
                name_filter=value.name_filter,
                generation=int(value.generation),
            )
        if isinstance(value, SourceSpec):
            metadata_uri = value.metadata_uri
            kind = getattr(value.kind, "value", value.kind)
            return cls(
                family="source",
                uri=str(value.uri),
                source_kind=str(kind),
                metadata_uri=(
                    None if metadata_uri is None else str(metadata_uri)
                ),
                entry=value.entry,
                options=_mapping_value(value.options),
                uri_was_path=isinstance(value.uri, Path),
                metadata_uri_was_path=isinstance(metadata_uri, Path),
            )
        raise TypeError(
            "source_spec must be SourceSpec, DirectorySourceSpec, or None"
        )

    def thaw(self) -> SourceSpec | DirectorySourceSpec:
        """Return a fresh typed source selection."""

        if self.family == "directory":
            return DirectorySourceSpec(
                root=Path(self.uri),
                recursive=bool(self.recursive),
                suffixes=tuple(self.suffixes),
                name_filter=self.name_filter,
                generation=int(self.generation),
            )
        if self.family == "source":
            uri: str | Path = Path(self.uri) if self.uri_was_path else self.uri
            metadata_uri: str | Path | None = self.metadata_uri
            if metadata_uri is not None and self.metadata_uri_was_path:
                metadata_uri = Path(metadata_uri)
            return SourceSpec(
                uri=uri,
                kind=self.source_kind,
                metadata_uri=metadata_uri,
                entry=self.entry,
                options=_thaw_mapping(self.options),
            )
        raise ValueError(f"unknown frozen source family {self.family!r}")

    def as_dict(self) -> dict[str, Any]:
        if self.family == "directory":
            return {
                "family": "directory",
                "root": self.uri,
                "recursive": bool(self.recursive),
                "suffixes": list(self.suffixes),
                "name_filter": self.name_filter,
                "generation": int(self.generation),
            }
        return {
            "family": "source",
            "uri": self.uri,
            "kind": self.source_kind,
            "metadata_uri": self.metadata_uri,
            "entry": self.entry,
            "options": _thaw_mapping(self.options),
        }

    def _fingerprint_value(self) -> tuple[Any, ...]:
        return (
            self.family,
            self.uri,
            self.source_kind,
            self.metadata_uri,
            self.entry,
            self.options,
            bool(self.recursive),
            tuple(self.suffixes),
            self.name_filter,
            int(self.generation),
            bool(self.uri_was_path),
            bool(self.metadata_uri_was_path),
        )


def _normalize_output_mode(value: str) -> str:
    text = str(value or "Append").strip().lower()
    if text == "append":
        return "Append"
    if text in {"overwrite", "replace"}:
        return "Overwrite"
    raise ValueError("output_mode must be Append, Overwrite, or Replace")


@dataclass(frozen=True, slots=True)
class FrozenRunConfiguration:
    """One generation-stamped, deeply immutable processing-run configuration."""

    generation: int
    source: FrozenSourceSpec | None
    processing_mode: str
    output_mode: str
    live_mode: bool
    batch_mode: bool
    max_cores: int
    gi: FrozenGIConfiguration
    threshold: FrozenThresholdPolicy
    poni_file: str = ""
    mask_file: str = ""
    project_root: str = ""
    save_path: str = ""
    _bai_1d_args: _FrozenValue = field(
        default=("mapping", ()),
        repr=False,
    )
    _bai_2d_args: _FrozenValue = field(
        default=("mapping", ()),
        repr=False,
    )
    _poni_values: _FrozenValue = field(
        default=("none",),
        repr=False,
    )
    _run_options: _FrozenValue = field(
        default=("mapping", ()),
        repr=False,
    )
    fingerprint: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        generation = int(self.generation)
        if generation < 1:
            raise ValueError("generation must be at least 1")
        max_cores = int(self.max_cores)
        if max_cores < 1:
            raise ValueError("max_cores must be at least 1")
        if bool(self.live_mode) and bool(self.batch_mode):
            raise ValueError("live_mode and batch_mode cannot both be enabled")
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "max_cores", max_cores)
        object.__setattr__(
            self,
            "output_mode",
            _normalize_output_mode(self.output_mode),
        )
        payload = self._content_fingerprint_value()
        digest = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "fingerprint", digest)

    def _content_fingerprint_value(self) -> tuple[Any, ...]:
        """Canonical content identity, deliberately excluding generation."""

        return (
            "xdart-frozen-run-configuration",
            _SCHEMA_VERSION,
            (
                None
                if self.source is None
                else self.source._fingerprint_value()
            ),
            str(self.processing_mode),
            _normalize_output_mode(self.output_mode),
            bool(self.live_mode),
            bool(self.batch_mode),
            int(self.max_cores),
            (
                bool(self.gi.enabled),
                str(self.gi.incidence_motor),
                _float_token(self.gi.th_val),
                int(self.gi.sample_orientation),
                _float_token(self.gi.tilt_angle),
                str(self.gi.mode_1d),
                str(self.gi.mode_2d),
            ),
            (
                bool(self.threshold.apply_threshold),
                (
                    None
                    if self.threshold.threshold_min is None
                    else _float_token(self.threshold.threshold_min)
                ),
                (
                    None
                    if self.threshold.threshold_max is None
                    else _float_token(self.threshold.threshold_max)
                ),
                bool(self.threshold.mask_saturation),
            ),
            str(self.poni_file),
            str(self.mask_file),
            str(self.project_root),
            str(self.save_path),
            self._bai_1d_args,
            self._bai_2d_args,
            self._poni_values,
            self._run_options,
        )

    @property
    def identity(self) -> tuple[int, str]:
        return int(self.generation), self.fingerprint

    @property
    def skip_2d(self) -> bool:
        text = str(self.processing_mode or "")
        return "Viewer" not in text and "1D" in text and "2D" not in text

    @property
    def bai_1d_args(self) -> dict[Any, Any]:
        """Return a fresh mutable copy for one consumer."""

        return _thaw_mapping(self._bai_1d_args)

    @property
    def bai_2d_args(self) -> dict[Any, Any]:
        """Return a fresh mutable copy for one consumer."""

        return _thaw_mapping(self._bai_2d_args)

    @property
    def poni_values(self) -> dict[Any, Any] | None:
        values = _thaw_value(self._poni_values)
        if values is None:
            return None
        if not isinstance(values, dict):
            raise TypeError("frozen PONI values are not a mapping")
        return values

    @property
    def run_options(self) -> dict[Any, Any]:
        return _thaw_mapping(self._run_options)

    def thaw_source_spec(self) -> SourceSpec | DirectorySourceSpec | None:
        return None if self.source is None else self.source.thaw()

    def scan_args(self) -> dict[str, dict[Any, Any]]:
        """Return fresh ``LiveScan`` integration keyword arguments."""

        return {
            "bai_1d_args": self.bai_1d_args,
            "bai_2d_args": self.bai_2d_args,
        }

    def scan_kwargs(self) -> dict[str, Any]:
        """Return fresh run-owned ``LiveScan`` configuration values."""

        values: dict[str, Any] = self.scan_args()
        values.update({
            "gi": bool(self.gi.enabled),
            "incidence_motor": self.gi.scan_incidence_motor,
            "skip_2d": bool(self.skip_2d),
            "apply_threshold": bool(self.threshold.apply_threshold),
            "threshold_min": self.threshold.threshold_min,
            "threshold_max": self.threshold.threshold_max,
            "mask_sentinel": bool(self.threshold.mask_saturation),
        })
        return values

    def processing_mapping(self) -> dict[str, Any]:
        """Return the canonical mapping consumed by Append/readiness helpers."""

        return {
            "bai_1d_args": self.bai_1d_args,
            "bai_2d_args": self.bai_2d_args,
            "gi": bool(self.gi.enabled),
            "gi_config": self.gi.scan_config(),
        }

    def threshold_mapping(self) -> dict[str, Any]:
        return self.threshold.as_dict()

    def as_provenance(self) -> dict[str, Any]:
        """Return a fresh JSON-friendly writer-provenance mapping."""

        return {
            "schema_version": _SCHEMA_VERSION,
            "generation": int(self.generation),
            "fingerprint": self.fingerprint,
            "source": None if self.source is None else self.source.as_dict(),
            "processing_mode": self.processing_mode,
            "output_mode": self.output_mode,
            "live_mode": bool(self.live_mode),
            "batch_mode": bool(self.batch_mode),
            "max_cores": int(self.max_cores),
            "gi": self.gi.as_dict(),
            "threshold": self.threshold.as_dict(),
            "poni_file": self.poni_file,
            "poni_values": self.poni_values,
            "mask_file": self.mask_file,
            "project_root": self.project_root,
            "save_path": self.save_path,
            "bai_1d_args": self.bai_1d_args,
            "bai_2d_args": self.bai_2d_args,
            "run_options": self.run_options,
        }


@dataclass(slots=True)
class RunIntent:
    """Mutable Controls-owned intent that freezes into one run generation."""

    source_spec: SourceSpec | DirectorySourceSpec | None = None
    processing_mode: str = "Int 2D"
    output_mode: str = "Append"
    live_mode: bool = False
    batch_mode: bool = False
    max_cores: int = 1
    bai_1d_args: dict[Any, Any] = field(default_factory=dict)
    bai_2d_args: dict[Any, Any] = field(default_factory=dict)
    gi: GIIntent = field(default_factory=GIIntent)
    threshold: ThresholdIntent = field(default_factory=ThresholdIntent)
    poni_file: str = ""
    poni_values: Mapping[Any, Any] | None = None
    mask_file: str = ""
    project_root: str = ""
    save_path: str = ""
    run_options: dict[Any, Any] = field(default_factory=dict)
    generation: int = 0

    def __post_init__(self) -> None:
        self.bai_1d_args = dict(self.bai_1d_args or {})
        self.bai_2d_args = dict(self.bai_2d_args or {})
        self.run_options = dict(self.run_options or {})
        if isinstance(self.gi, Mapping):
            self.gi = GIIntent.from_mapping(self.gi)
        if not isinstance(self.gi, GIIntent):
            raise TypeError("gi must be GIIntent or a mapping")
        if isinstance(self.threshold, Mapping):
            self.threshold = ThresholdIntent.from_mapping(self.threshold)
        if not isinstance(self.threshold, ThresholdIntent):
            raise TypeError("threshold must be ThresholdIntent or a mapping")
        if self.source_spec is not None and not isinstance(
            self.source_spec,
            (SourceSpec, DirectorySourceSpec),
        ):
            raise TypeError(
                "source_spec must be SourceSpec, DirectorySourceSpec, or None"
            )
        self.generation = int(self.generation)
        if self.generation < 0:
            raise ValueError("generation cannot be negative")

    def freeze(
        self,
        *,
        generation: int | None = None,
    ) -> FrozenRunConfiguration:
        """Freeze the current values and advance the monotonic run generation."""

        if generation is None:
            next_generation = int(self.generation) + 1
        else:
            next_generation = int(generation)
            if next_generation <= int(self.generation):
                raise ValueError(
                    "explicit generation must be greater than the current generation"
                )
        source = (
            None
            if self.source_spec is None
            else FrozenSourceSpec.from_source(self.source_spec)
        )
        frozen = FrozenRunConfiguration(
            generation=next_generation,
            source=source,
            processing_mode=str(self.processing_mode),
            output_mode=_normalize_output_mode(self.output_mode),
            live_mode=bool(self.live_mode),
            batch_mode=bool(self.batch_mode),
            max_cores=int(self.max_cores),
            gi=self.gi.freeze(),
            threshold=self.threshold.freeze(),
            poni_file=str(self.poni_file or ""),
            mask_file=str(self.mask_file or ""),
            project_root=str(self.project_root or ""),
            save_path=str(self.save_path or ""),
            _bai_1d_args=_mapping_value(self.bai_1d_args),
            _bai_2d_args=_mapping_value(self.bai_2d_args),
            _poni_values=_freeze_value(
                None if self.poni_values is None else dict(self.poni_values)
            ),
            _run_options=_mapping_value(self.run_options),
        )
        # Advance only after the complete snapshot has validated.  A rejected
        # Run click must not leave a hole in the Controls-owned generation
        # sequence or make a retry look like a different accepted run.
        self.generation = next_generation
        return frozen

    @classmethod
    def from_frozen(cls, value: FrozenRunConfiguration) -> "RunIntent":
        """Return a mutable intent initialized from a frozen configuration."""

        if not isinstance(value, FrozenRunConfiguration):
            raise TypeError("value must be FrozenRunConfiguration")
        gi = value.gi.as_dict()
        threshold = value.threshold.as_dict()
        return cls(
            source_spec=value.thaw_source_spec(),
            processing_mode=value.processing_mode,
            output_mode=value.output_mode,
            live_mode=value.live_mode,
            batch_mode=value.batch_mode,
            max_cores=value.max_cores,
            bai_1d_args=value.bai_1d_args,
            bai_2d_args=value.bai_2d_args,
            gi=GIIntent.from_mapping(gi),
            threshold=ThresholdIntent.from_mapping(threshold),
            poni_file=value.poni_file,
            poni_values=value.poni_values,
            mask_file=value.mask_file,
            project_root=value.project_root,
            save_path=value.save_path,
            run_options=value.run_options,
            generation=value.generation,
        )


__all__ = [
    "FrozenGIConfiguration",
    "FrozenRunConfiguration",
    "FrozenSourceSpec",
    "FrozenThresholdPolicy",
    "GIIntent",
    "RunIntent",
    "ThresholdIntent",
]
