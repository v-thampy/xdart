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
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from xrd_tools.core.scan import SourceSpec
from xrd_tools.session.gi_motor import pick_default_gi_motor
from xrd_tools.sources.selection import (
    DirectorySourceSpec,
    normalize_image_source_metadata,
)


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


def resolve_gi_motor(
    raw: str,
    choices: "list[str] | tuple[str, ...] | None",
) -> str:
    """Resolve the effective GI incidence motor ONCE, from a choices list.

    ``raw`` is the operator's stored selection; ``choices`` is the source's real
    motor list (may include ``'Manual'``).  Rule (R4B-8, one policy):

    * a deliberate ``'Manual'`` stays ``'Manual'``;
    * a valid explicit selection (``raw`` is a real motor of the source) wins;
    * otherwise the shared default policy (:func:`pick_default_gi_motor`) picks
      over the source's real motors — never injecting a motor absent from the
      source;
    * when no choices list is supplied (``choices is None``) the raw selection
      is honored as-is — the caller could not offer a source motor list to
      verify against, so degrading an explicit motor to Manual would silently
      diverge from what the operator sees.  ``()`` means genuinely
      empty-and-known (the source was probed and has no real motors); a caller
      that has NOT probed must pass ``None``, not ``()``.
    """
    raw = str(raw or "Manual")
    if choices is None:
        return raw
    real = [str(c) for c in choices if str(c) and str(c) != "Manual"]
    if raw == "Manual":
        return "Manual"
    if raw in real:
        return raw
    return pick_default_gi_motor(real)


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

    def freeze(
        self,
        *,
        choices: "list[str] | tuple[str, ...] | None" = None,
    ) -> "FrozenGIConfiguration":
        raw = str(self.incidence_motor or "Manual")
        return FrozenGIConfiguration(
            enabled=bool(self.enabled),
            incidence_motor=raw,
            resolved_motor=resolve_gi_motor(raw, choices),
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
    """Deeply immutable grazing-incidence configuration for one run.

    ``incidence_motor`` is the operator's RAW selection; ``resolved_motor`` is
    the effective incidence axis chosen once by the shared policy from the
    source's motor list (R4B-8).  Every RUN consumer (plan builder, worker,
    writer provenance) uses the resolved value; the raw selection is retained
    for provenance/audit.
    """

    enabled: bool = False
    incidence_motor: str = "Manual"
    resolved_motor: str = ""
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
        # Direct construction (tests, restore) may omit resolution; fall back to
        # the raw selection so the resolved value is always populated.
        if not str(self.resolved_motor or ""):
            object.__setattr__(
                self, "resolved_motor", str(self.incidence_motor or "Manual"))

    @property
    def effective_motor(self) -> str:
        """The resolved incidence-motor axis every RUN consumer must use."""
        return str(self.resolved_motor or self.incidence_motor or "Manual")

    @property
    def scan_incidence_motor(self) -> str:
        motor = self.effective_motor
        if motor == "Manual":
            return str(float(self.th_val))
        return str(motor)

    def scan_config(self) -> dict[str, Any]:
        """Return a fresh ``LiveScan.gi_config`` mapping."""

        if not self.enabled:
            return {}
        return {
            "gi_mode_1d": self.mode_1d,
            "gi_mode_2d": self.mode_2d,
            "incidence_motor": self.effective_motor,
            "th_val": float(self.th_val),
            "sample_orientation": int(self.sample_orientation),
            "tilt_angle": float(self.tilt_angle),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "incidence_motor": self.incidence_motor,
            "resolved_motor": self.effective_motor,
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


def _detached_value(value: Any) -> Any:
    """Recursively detach one mutable intent value from its producer.

    O-1a-W1R-D1 (review §41.3.A).  A copy boundary that shares a nested container
    is not a copy: the candidate could mutate the canonical intent through it.
    ``Mapping`` covers both ``dict`` and the ``MappingProxyType`` that
    ``SourceSpec`` wraps its options in.
    """
    if isinstance(value, Mapping):
        # O-1a-W1R-D2 (review §43.3): KEYS ride the same boundary as values.
        return {
            _detached_value(key): _detached_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_detached_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detached_value(item) for item in value)
    # O-1a-W1R-D2 (review §42.2 item 2): preserve set VERSUS frozenset -- the
    # first version collapsed both to a mutable ``set`` -- and detach members.
    if isinstance(value, frozenset):
        return frozenset(_detached_value(item) for item in value)
    if isinstance(value, set):
        return {_detached_value(item) for item in value}
    return _detached_container(value)


def _detached_container(value: Any) -> Any:
    """Detach one duck-typed value container (NumPy arrays/scalars and kin).

    O-1a-W1R-D2 (review §42.2 items 1 and 3, §43.3).  :func:`_freeze_value`
    accepts ANY object that converts itself through ``item()``/``tolist()``, so
    those -- not just the ``dtype``-bearing subset -- are exactly what this
    boundary owes a recursive copy.

    An arbitrary ``copy()`` or constructor is not evidence: either can alias
    the original or change the value.  Generic containers reduce to their
    recursively detached built-in value.  Only NumPy's narrow dtype/shape value
    protocol uses ``deepcopy`` to preserve its public type, and every retained
    result must freeze to the exact pre-copy value.
    """
    converters = [name for name in ("item", "tolist")
                  if callable(getattr(value, name, None))]
    if not converters:
        return value

    frozen_before = _freeze_value(value)
    for name in converters:
        try:
            converted = getattr(value, name)()
        except (TypeError, ValueError):
            continue
        if converted is value:
            continue
        detached = _detached_value(converted)
        if _freeze_value(detached) != frozen_before:
            continue

        value_type = type(value)
        if (value_type.__module__.split(".", 1)[0] == "numpy"
                and hasattr(value, "dtype") and hasattr(value, "shape")):
            try:
                copied = copy.deepcopy(value)
            except Exception:
                copied = value
            if (copied is not value
                    and _freeze_value(copied) == frozen_before):
                return copied

        return detached
    raise TypeError(
        "run configuration values must be detachable; "
        f"{type(value).__module__}.{type(value).__qualname__} cannot provide "
        "an equivalent detached value")


def _detached_source_spec(value):
    """A detached copy of one typed source selection (review §41.3.A)."""
    if value is None:
        return None
    from xrd_tools.sources import DirectorySourceSpec

    if isinstance(value, DirectorySourceSpec):
        # Frozen dataclass of immutable scalars and a tuple; rebuild anyway so a
        # candidate never shares identity with the canonical intent's value.
        return DirectorySourceSpec(
            root=value.root,
            recursive=bool(value.recursive),
            suffixes=tuple(value.suffixes),
            name_filter=value.name_filter,
            generation=int(value.generation),
            metadata_format=value.metadata_format,
        )
    if isinstance(value, SourceSpec):
        return SourceSpec(
            uri=value.uri,
            kind=value.kind,
            metadata_uri=value.metadata_uri,
            entry=value.entry,
            options=_detached_value(dict(value.options or {})),
        )
    return value


def _non_directory_suffixes(value: SourceSpec) -> tuple[str, ...]:
    """The format token of a non-directory source, from its canonical data.

    O-1a-W1R-D1 (review §40.1 P1-B).  ``image_series_spec`` freezes the
    containing directory as the ``uri`` and records the selected member in
    ``options``, so the member is the authority for a series; a single image or a
    container master names its own file.  Returns ``()`` only when neither
    carries a suffix, which is not a supported production shape.
    """
    selected = ""
    options = getattr(value, "options", None) or {}
    try:
        selected = str(options.get("selected_file") or "")
    except AttributeError:
        selected = ""
    token = Path(selected).suffix if selected else Path(str(value.uri)).suffix
    token = str(token).strip().lower()
    return (token,) if token else ()


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
    metadata_format: str | None = "auto"
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
                metadata_format=value.metadata_format,
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
                # O-1a-W1R-D1 (review §40.1 P1-B, §40.3 D1 item 3): a
                # non-directory source must be TOTAL.  A numbered series freezes
                # the CONTAINING DIRECTORY as its uri, so a consumer deriving the
                # format from the uri got nothing and fell back to a mutable
                # mirror.  The format comes from canonical frozen data -- the
                # selected member when the source names one, else the uri itself
                # -- and the directory-only questions get their neutral answers
                # rather than "not applicable".
                suffixes=_non_directory_suffixes(value),
                recursive=False,
                name_filter="",
                uri_was_path=isinstance(value.uri, Path),
                metadata_uri_was_path=isinstance(metadata_uri, Path),
            )
        raise TypeError(
            "source_spec must be SourceSpec, DirectorySourceSpec, or None"
        )

    @property
    def _uri_names_a_directory(self) -> bool:
        # A numbered series freezes its CONTAINING DIRECTORY as the uri, so both
        # source-value answers below turn on this one fact (review §43.1).
        return self.family == "directory" or self.source_kind == "tiff_series"

    @property
    def format_tokens(self) -> tuple[str, ...]:
        """Every frozen format alternative, as a bare token ('_master.h5' -> 'h5').

        All alternatives are preserved: an Eiger master froze as
        ``("_master.hdf5", "_master.h5")`` reads as ``("hdf5", "h5")``, because a
        consumer matching one spelling must still recognize the other.
        """
        candidates = self.suffixes or (
            () if self._uri_names_a_directory else (Path(self.uri).suffix,))
        tokens = (str(item).strip().lower().rsplit(".", 1)[-1]
                  for item in candidates)
        return tuple(token for token in tokens if token)

    @property
    def filesystem_root(self) -> str:
        """The directory this source reads from; fail closed when it has none."""
        if self._uri_names_a_directory:
            return str(self.uri)
        if self.source_kind in ("image_file", "nexus_stack", "eiger_master",
                                "processed_nexus", "spec"):
            return str(Path(self.uri).parent)
        raise ValueError(
            f"frozen source kind {self.source_kind!r} has no filesystem root")

    def thaw(self) -> SourceSpec | DirectorySourceSpec:
        """Return a fresh typed source selection."""

        if self.family == "directory":
            return DirectorySourceSpec(
                root=Path(self.uri),
                recursive=bool(self.recursive),
                suffixes=tuple(self.suffixes),
                name_filter=self.name_filter,
                generation=int(self.generation),
                metadata_format=self.metadata_format,
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
                "metadata_format": self.metadata_format,
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
        value = (
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
        )
        if self.family == "directory":
            value += (self.metadata_format,)
        return value + (
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
                str(self.gi.effective_motor),
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
        """Return a fresh, detached, JSON-NATIVE writer-provenance mapping.

        O-1a-W1R (review §39.2 W1R-P1-7): the mapping is normalized through the
        ONE bounded recursive normalizer (:func:`jsonable_run_value`) so
        ``json.loads(json.dumps(mapping)) == mapping`` holds for every supported
        frozen value, and an unsupported/live value REFUSES here -- before the
        writer opens any output file -- instead of being stringified into the
        persisted record.
        """

        return jsonable_run_value(self._provenance_values(), path="provenance")

    def _provenance_values(self) -> dict[str, Any]:
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

    def native_int_snapshot(self) -> dict[str, Any]:
        """Return a reduction snapshot dictionary derived from this frozen configuration."""
        gi_enabled = bool(self.gi.enabled)
        incidence = self.gi.scan_incidence_motor if gi_enabled else "Manual"
        return {
            "bai_1d_args": copy.deepcopy(dict(self.bai_1d_args or {})),
            "bai_2d_args": copy.deepcopy(dict(self.bai_2d_args or {})),
            "gi": gi_enabled,
            "gi_config": self.gi.scan_config() if gi_enabled else {},
            "incidence_motor": incidence,
            "th_mtr": incidence,
            "sample_orientation": int(self.gi.sample_orientation),
            "tilt_angle": float(self.gi.tilt_angle),
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
        if type(self.source_spec) is SourceSpec:
            self.source_spec = normalize_image_source_metadata(self.source_spec)
        self.generation = int(self.generation)
        if self.generation < 0:
            raise ValueError("generation cannot be negative")

    def freeze(
        self,
        *,
        generation: int | None = None,
        gi_motor_choices: "list[str] | tuple[str, ...] | None" = None,
    ) -> FrozenRunConfiguration:
        """Freeze the current values and advance the monotonic run generation.

        ``gi_motor_choices`` is the source's real motor list, supplied by the
        caller (the GUI) so the effective GI motor is resolved ONCE here (R4B-8)
        rather than independently by each run consumer.
        """

        if generation is None:
            next_generation = int(self.generation) + 1
        else:
            next_generation = int(generation)
            if next_generation <= int(self.generation):
                raise ValueError(
                    "explicit generation must be greater than the current generation"
                )
        source_spec = self.source_spec
        if type(source_spec) is SourceSpec:
            source_spec = normalize_image_source_metadata(source_spec)
        source = (
            None
            if source_spec is None
            else FrozenSourceSpec.from_source(source_spec)
        )
        frozen = FrozenRunConfiguration(
            generation=next_generation,
            source=source,
            processing_mode=str(self.processing_mode),
            output_mode=_normalize_output_mode(self.output_mode),
            live_mode=bool(self.live_mode),
            batch_mode=bool(self.batch_mode),
            max_cores=int(self.max_cores),
            gi=self.gi.freeze(choices=gi_motor_choices),
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

    def clone_candidate(self) -> "RunIntent":
        """Return a DETACHED candidate copy, safe to populate, freeze and discard.

        O-1a-W1R-D1 (review §41.3.A and §41.3.B).  Two defects share one root:
        ``copy.deepcopy`` cannot copy this intent once a real source is on it,
        because ``SourceSpec.__post_init__`` stores ``options`` in a
        ``MappingProxyType``; and freezing the canonical intent to build a run
        candidate advances its generation even when admission then refuses.

        One explicit value-copy boundary answers both.  Every nested mutable value
        is detached recursively -- a shared nested container is not a copy -- and
        the clone starts from this intent's generation without advancing it, so
        ``freeze()`` on the clone leaves the canonical sequence untouched until an
        accepted run commits it.
        """
        return RunIntent(
            source_spec=_detached_source_spec(self.source_spec),
            processing_mode=str(self.processing_mode),
            output_mode=str(self.output_mode),
            live_mode=bool(self.live_mode),
            batch_mode=bool(self.batch_mode),
            max_cores=int(self.max_cores),
            bai_1d_args=_detached_value(dict(self.bai_1d_args or {})),
            bai_2d_args=_detached_value(dict(self.bai_2d_args or {})),
            gi=GIIntent(
                enabled=_detached_value(self.gi.enabled),
                incidence_motor=_detached_value(self.gi.incidence_motor),
                th_val=_detached_value(self.gi.th_val),
                sample_orientation=_detached_value(self.gi.sample_orientation),
                tilt_angle=_detached_value(self.gi.tilt_angle),
                mode_1d=_detached_value(self.gi.mode_1d),
                mode_2d=_detached_value(self.gi.mode_2d),
            ),
            threshold=ThresholdIntent(
                apply_threshold=_detached_value(self.threshold.apply_threshold),
                threshold_min=_detached_value(self.threshold.threshold_min),
                threshold_max=_detached_value(self.threshold.threshold_max),
                mask_saturation=_detached_value(self.threshold.mask_saturation),
            ),
            poni_file=self.poni_file,
            poni_values=(
                None if self.poni_values is None
                else _detached_value(dict(self.poni_values))
            ),
            mask_file=self.mask_file,
            project_root=self.project_root,
            save_path=self.save_path,
            run_options=_detached_value(dict(self.run_options or {})),
            generation=int(self.generation),
        )

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


class RunConfigurationRefused(RuntimeError):
    """One typed refusal for a run configuration that may not be executed.

    O-1a-W1: after admission the frozen configuration is the SOLE run-configuration
    authority, so a consumer that does not hold the accepted object must refuse
    rather than fall back to display state.  Exactly three reasons exist:

    ``absent``   nothing was published for this run;
    ``foreign``  the carrier is not THE object this run admitted -- either not a
                 :class:`FrozenRunConfiguration` at all, or a genuine but
                 different one (same-generation, future-generation, or
                 equal-valued reconstruction);
    ``stale``    the carrier was frozen for an earlier accepted click.

    O-1a-W1R (review §39.2 W1R-P1-1): ``foreign`` deliberately covers a GENUINE
    frozen object that is not the admitted one.  Generation is monotonic
    admission metadata; it is never execution authorization, so two distinct
    genuine objects at the same generation are not interchangeable.
    """

    __slots__ = ("reason", "stage", "detail", "generation", "floor")

    def __init__(
        self,
        reason: str,
        *,
        stage: str,
        detail: str = "",
        generation: int | None = None,
        floor: int | None = None,
    ) -> None:
        self.reason = str(reason)
        self.stage = str(stage)
        self.detail = str(detail)
        self.generation = generation
        self.floor = floor
        message = f"run configuration refused ({self.reason}) at {self.stage}"
        if self.detail:
            message = f"{message}: {self.detail}"
        super().__init__(message)

    def as_event_fields(self) -> dict[str, Any]:
        """Structured fields for one refusal event (no formatting policy here)."""

        return {
            "reason": self.reason,
            "stage": self.stage,
            "detail": self.detail,
            "generation": self.generation,
            "floor": self.floor,
        }


def _require_frozen_carrier(
    value: Any,
    *,
    stage: str,
    floor: int,
) -> FrozenRunConfiguration:
    """The absence/type half both gates share."""

    if value is None:
        raise RunConfigurationRefused(
            "absent",
            stage=stage,
            detail="no frozen run configuration was published for this run",
            floor=int(floor),
        )
    if not isinstance(value, FrozenRunConfiguration):
        raise RunConfigurationRefused(
            "foreign",
            stage=stage,
            detail=f"carrier is {type(value).__name__}, not FrozenRunConfiguration",
            floor=int(floor),
        )
    return value


def require_run_configuration(
    value: Any,
    *,
    stage: str,
    floor: int = 0,
    expected: FrozenRunConfiguration | None = None,
) -> FrozenRunConfiguration:
    """CONSUMPTION gate: return the EXACT admitted object, or refuse.

    O-1a-W1R (review §39.2 W1R-P1-1, §39.5 Phase 1 item 3).  Admission and
    consumption have different rules, and this is the consumption half:

    * ``expected`` is the object the owner bound at admission.  The carrier must
      be that object by ``is``.  A genuine but different ``FrozenRunConfiguration``
      -- same generation, a future generation, or an equal-valued reconstruction
      with an identical fingerprint -- is ``foreign``, because equal values are
      not run identity;
    * ``expected=None`` means this owner has no admitted binding.  A carrier
      without a binding is refused: a bare generation ``floor`` is admission
      evidence, NOT execution authorization, so it can never license a
      substitution (this is exactly what the parent got wrong).

    ``floor`` is retained for the refusal event fields and for the ``stale``
    diagnosis; it no longer decides anything on its own.
    """

    frozen = _require_frozen_carrier(value, stage=stage, floor=floor)
    if expected is None:
        raise RunConfigurationRefused(
            "foreign",
            stage=stage,
            detail=(
                "no admitted run configuration is bound at this stage; a "
                "generation floor is not execution authorization"
            ),
            generation=int(frozen.generation),
            floor=int(floor),
        )
    if frozen is expected:
        return frozen
    if int(frozen.generation) < int(expected.generation):
        raise RunConfigurationRefused(
            "stale",
            stage=stage,
            detail=(
                f"generation {int(frozen.generation)} was superseded by "
                f"generation {int(expected.generation)}"
            ),
            generation=int(frozen.generation),
            floor=int(expected.generation),
        )
    same_values = frozen.fingerprint == expected.fingerprint
    raise RunConfigurationRefused(
        "foreign",
        stage=stage,
        detail=(
            "carrier is a different FrozenRunConfiguration than the one this "
            f"run admitted (carrier generation {int(frozen.generation)}, "
            f"admitted generation {int(expected.generation)}, "
            + ("identical" if same_values else "different")
            + " fingerprint)"
        ),
        generation=int(frozen.generation),
        floor=int(expected.generation),
    )


def admit_run_configuration(
    value: Any,
    *,
    stage: str,
    floor: int = 0,
    bound: FrozenRunConfiguration | None = None,
) -> FrozenRunConfiguration:
    """FIRST-ADMISSION gate: accept ONE newly frozen object, exactly once.

    O-1a-W1R (review §39.5 Phase 1 items 1 and 3).  The rules here are
    deliberately NOT the consumption rules:

    * ``bound`` is the object this owner is currently holding.  While an object
      is bound at the SAME generation, the only acceptable admission is that
      exact object again (an idempotent re-publication).  A different genuine
      object at the bound generation is a REBIND and is refused ``foreign``;
    * a strictly newer generation is a genuinely new accepted click and binds;
    * an older generation is ``stale``.

    Generation therefore stays monotonic first-admission metadata, which is all
    §39.2 W1R-P1-1 leaves it authorized to be.
    """

    frozen = _require_frozen_carrier(value, stage=stage, floor=floor)
    generation = int(frozen.generation)
    reference = int(
        bound.generation if bound is not None else floor)
    if generation < reference:
        raise RunConfigurationRefused(
            "stale",
            stage=stage,
            detail=(
                f"generation {generation} was superseded by generation "
                f"{reference}"
            ),
            generation=generation,
            floor=reference,
        )
    if generation == reference and reference > 0:
        if bound is None or frozen is not bound:
            raise RunConfigurationRefused(
                "foreign",
                stage=stage,
                detail=(
                    "a different FrozenRunConfiguration was offered at the "
                    f"already-admitted generation {reference}; admission binds "
                    "one object once"
                ),
                generation=generation,
                floor=reference,
            )
    return frozen


# --------------------------------------------------------------------------- #
# Detached JSON-native provenance (review §39.2 W1R-P1-7).
# --------------------------------------------------------------------------- #

def jsonable_run_value(value: Any, *, path: str = "provenance") -> Any:
    """Return a detached, deterministic, JSON-NATIVE copy of *value*.

    O-1a-W1R (review §39.2 W1R-P1-7, §39.5 Phase 3 item 4).  ``_freeze_value``
    accepts several value types that survive :meth:`FrozenRunConfiguration.thaw`
    but are NOT JSON — most importantly the pyFAI method TUPLE — so
    ``json.loads(json.dumps(provenance))`` differed from the mapping the writer
    was handed.  This is the ONE normalizer for that conversion; it follows the
    bounded rules the vNext provenance path already used
    (``reduction.provenance_config._jsonable_range`` / ``_enum_value``), which
    now delegates here so the tree keeps a single owner.

    Unsupported or live values RAISE, before any output file is created: a
    lossy ``default=str`` stringify at write time is what W-1 forbids.
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"{path}: non-finite float {number!r} has no JSON form")
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return jsonable_run_value(value.value, path=f"{path}.value")
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{path}: JSON object keys must be str, got "
                    f"{type(key).__module__}.{type(key).__qualname__}"
                )
            out[key] = jsonable_run_value(item, path=f"{path}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [
            jsonable_run_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path}: run provenance values must be JSON-native; unsupported "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


__all__ = [
    "FrozenGIConfiguration",
    "FrozenRunConfiguration",
    "FrozenSourceSpec",
    "FrozenThresholdPolicy",
    "GIIntent",
    "RunConfigurationRefused",
    "RunIntent",
    "ThresholdIntent",
    "admit_run_configuration",
    "jsonable_run_value",
    "require_run_configuration",
    "resolve_gi_motor",
]
