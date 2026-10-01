"""Immutable, Qt-free operator values for the standalone Stitch tool.

The GUI edits :class:`StitchToolForm`.  Preflight turns that form into one
exact :class:`~xrd_tools.analysis.stitch_operation.StitchOperationRequest` and
a bounded summary of the raw members it owns.  Run must consume the request in
that preflight unchanged; rebuilding it at dispatch would reopen a source and
silently change the scientific input.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math
import os
from pathlib import Path
import re

import numpy as np

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleArtifactRefused,
    ModuleKind,
    ModuleOutputRequest,
    ModuleSourceReceipt,
    ModuleSourceGroupReceipt,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    MetadataTablePlan,
    analysis_canonical_fingerprint,
    run_metadata_table,
)
from xrd_tools.analysis.stitch_operation import (
    SAMPLE_POSITIONS,
    STITCH_ERROR_MODELS,
    StitchGeometryInput,
    StitchGeometryKind,
    StitchOperationPlan,
    StitchOperationRefused,
    StitchOperationRequest,
    XuStitchOperationPlan,
    capture_stitch_geometry,
    prepare_stitch_operation,
)
from xrd_tools.integrate.frame_screen import EXCLUDE_CHOICES
from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput,
    XuStitchCalibrationRefused,
    capture_xu_stitch_calibration,
)
from xrd_tools.core.geometry.xu_runtime import (
    XuRuntimeUnsupported,
    xu_runtime_session,
)
from xrd_tools.integrate.xu_stitch import (
    XuStitchScienceRefused,
    resolve_xu_stitch_effective_geometry,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.integrate.stitch_gi import StitchGI
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCAN = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_PREFLIGHT_FACTORY = object()
_MAX_SELECTED_FRAMES = 4096


class StitchToolPreflightRefused(ValueError):
    """A form could not be admitted as one exact Stitch preflight."""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        diagnostics: tuple[str, ...] = (),
    ) -> None:
        if type(code) is not str or not code:
            raise TypeError("preflight refusal code must be a nonempty string")
        if type(diagnostics) is not tuple or any(
            type(item) is not str for item in diagnostics
        ):
            raise TypeError("preflight diagnostics must be an exact string tuple")
        self.code = code
        self.diagnostics = diagnostics
        super().__init__(code if message is None else f"{code}: {message}")


def _absolute_path(value: object, name: str) -> str:
    try:
        spelling = os.fspath(value)
    except TypeError as error:
        raise TypeError(f"{name} must be path-like") from error
    if type(spelling) is not str:
        raise TypeError(f"{name} must resolve to a string path")
    if not spelling or not spelling.strip() or "\0" in spelling:
        raise ValueError(f"{name} is required")
    path = os.path.abspath(os.path.expanduser(spelling))
    if not path or "\0" in path:
        raise ValueError(f"{name} is invalid")
    return os.path.normcase(path)


def _pairs(value: object, name: str) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    parsed: list[tuple[str, str]] = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or any(
                type(part) is not str or not part or part.strip() != part
                for part in item
            )
        ):
            raise TypeError(f"{name} entries must be exact nonempty string pairs")
        parsed.append(item)
    result = tuple(parsed)
    if (
        not result
        or result != tuple(sorted(result))
        or len({left for left, _right in result}) != len(result)
        or len({right for _left, right in result}) != len(result)
    ):
        raise ValueError(f"{name} must be nonempty, sorted, and one-to-one")
    return result


def _references(value: object) -> tuple[tuple[str, float], ...]:
    if type(value) is not tuple:
        raise TypeError("PONI references must be an exact tuple")
    parsed: list[tuple[str, float]] = []
    for item in value:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or not item[0]
            or item[0].strip() != item[0]
            or type(item[1]) not in {int, float}
            or not math.isfinite(float(item[1]))
        ):
            raise TypeError("PONI references must be finite name/value pairs")
        parsed.append((item[0], float(item[1])))
    result = tuple(parsed)
    if result != tuple(sorted(result)) or len({name for name, _ in result}) != len(
        result
    ):
        raise ValueError("PONI references must be sorted and unique")
    return result


def _ordered_range(value: object, name: str) -> tuple[float, float]:
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(item) not in {int, float} for item in value)
        or any(not math.isfinite(float(item)) for item in value)
        or float(value[0]) >= float(value[1])
    ):
        raise ValueError(f"{name} must be one finite increasing pair")
    return float(value[0]), float(value[1])


@dataclass(frozen=True, slots=True)
class StitchFrameSelector:
    """Inclusive numeric frame-label range, never an ordinal table slice."""

    start_label: int = 0
    stop_label: int | None = None
    step: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.start_label) is not int
            or self.start_label < 0
            or (
                self.stop_label is not None
                and (
                    type(self.stop_label) is not int
                    or self.stop_label < self.start_label
                )
            )
            or type(self.step) is not int
            or self.step < 1
        ):
            raise ValueError("Stitch frame-label selector is invalid")

    def select(self, labels: tuple[int, ...]) -> tuple[int, ...]:
        if type(labels) is not tuple or not labels or any(
            type(label) is not int or label < 0 for label in labels
        ):
            raise StitchToolPreflightRefused(
                "SOURCE_LABELS_INVALID", "source frame labels are invalid"
            )
        stop = max(labels) if self.stop_label is None else self.stop_label
        count = (stop - self.start_label) // self.step + 1
        if count > _MAX_SELECTED_FRAMES:
            raise StitchToolPreflightRefused(
                "FRAME_SELECTION_LIMIT_EXCEEDED",
                "Stitch selection exceeds 4096 frames",
            )
        expected = tuple(range(self.start_label, stop + 1, self.step))
        expected_set = set(expected)
        selected = tuple(label for label in labels if label in expected_set)
        if not expected or selected != expected:
            raise StitchToolPreflightRefused(
                "FRAME_SELECTION_OUTSIDE_SOURCE",
                "requested frame labels are not one exact ordered source subset",
            )
        return selected

    @property
    def fingerprint_value(self) -> tuple[object, ...]:
        return self.start_label, self.stop_label, self.step


@dataclass(frozen=True, slots=True)
class StitchToolForm:
    """All operator-editable Stitch inputs, normalized into one frozen value."""

    project_root: str | Path
    spec_path: str | Path
    scan: str
    image_dir: str | Path
    image_stem: str
    frame_selector: StitchFrameSelector
    detector_shape: tuple[int, int]
    raw_dtype: str
    geometry_path: str | Path
    geometry_kind: StitchGeometryKind
    source_motors: tuple[tuple[str, str], ...]
    q_range: tuple[float, float] | None
    npt_1d: int
    output_path: str | Path
    raw_header_skip: int = 0
    threshold: float | None = None
    expected_geometry_sha256: str | None = None
    poni_references: tuple[tuple[str, float], ...] = ()
    image_rotation: int = 0
    monitor_selector: MetadataColumnSelector | None = None
    use_detector_mask: bool = True
    overwrite: AnalysisArtifactOverwrite = AnalysisArtifactOverwrite.REPLACE
    max_frame_bytes: int = 256 * 1024 * 1024
    mode: str = "1d"
    backend: str = "multigeometry"
    source_selection: StitchSourceSelection | None = None
    unit: str = "q_A^-1"
    conditioning: StitchConditioning | None = None
    mask_path: str | None = None
    output_family: str | None = None
    output_families: tuple[str, ...] = ()
    #: Native options: dropped-frame check, sample position and the pyFAI
    #: corrections. The XU backend takes the frame check only.
    frame_screen: str = "empty"
    sample_position: str = "calibrated"
    correct_solid_angle: bool = True
    polarization_factor: float | None = 0.99
    error_model: str | None = None
    dark_path: str | None = None
    flat_path: str | None = None
    #: Flat pixels below this fraction of the flat's median are masked as dead.
    flat_min_fraction: float = 0.1
    #: Grazing incidence: GI cuts written beside I(q), and a Stitch 2D's q_ip-q_oop map (native, in Q).
    gi: StitchGI | None = None
    #: Stitch 2D (``mode`` '2d', native pyFAI MultiGeometry): the map's radial and azimuthal points
    #: and its χ range (°), as Int 2D's; I(q) beside it takes ``npt_1d`` and the radial range.
    npt_rad_2d: int = 1000
    npt_azim_2d: int = 360
    #: The map's χ range; None (Auto): the frames' own χ extent, as Int 2D's (the data, not −180 to 180).
    chi_range: tuple[float, float] | None = None
    #: The map's radial axis, as Int 2D's own: Q or 2θ (Q with GI), over its range (None, Auto: the
    #: frames' extent); ``unit`` and ``q_range`` are I(q)'s beside it.
    map_unit: str = "q_A^-1"
    map_radial_range: tuple[float, float] | None = None
    #: The frames are also mirrored left-right after ``image_rotation``: a mirrored detector camera's
    #: transform into Stitch's frame (``camera.stitch_orientation``).
    image_flip: bool = False
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if (self.output_family is not None and type(self.output_family) is not str
            or type(self.output_families) is not tuple
            or any(type(value) is not str for value in self.output_families)
            or self.output_family and self.output_families):
            raise ValueError("Specify one result family or ordered output families")
        project = _absolute_path(self.project_root, "Project root")
        spec = _absolute_path(self.spec_path, "SPEC path")
        image_dir = _absolute_path(self.image_dir, "image directory")
        geometry = _absolute_path(self.geometry_path, "geometry path")
        output = _absolute_path(self.output_path, "output path")
        if self.backend not in {"multigeometry", "xu_hist"}:
            raise ValueError("Stitch backend must be multigeometry or xu_hist")
        native = self.source_selection is not None
        if native and type(self.source_selection) is not StitchSourceSelection:
            raise TypeError("native source selection must be exact StitchSourceSelection")
        if self.unit not in {"q_A^-1", "2th_deg"}:
            raise ValueError("Choose Q or 2theta for Stitch binning")
        if self.conditioning is not None and type(self.conditioning) is not StitchConditioning:
            raise TypeError("conditioning must be exact StitchConditioning")
        if not native and (self.unit != "q_A^-1" or self.conditioning is not None or self.mask_path is not None):
            raise ValueError("native processing requires a native source selection")
        if self.mask_path is not None:
            object.__setattr__(self, "mask_path", _absolute_path(self.mask_path, "mask path"))
        self._native_options(native)
        if self.gi is not None and (type(self.gi) is not StitchGI or not native or self.unit != "q_A^-1"):
            raise ValueError("a GI cut needs a native Stitch in Q")
        if not native and Path(spec).suffix:
            raise ValueError("Stitch SPEC path must be extensionless")
        if Path(output).suffix.casefold() != ".nexus":
            raise ValueError("Stitch output path must end in .nexus")
        if not native and (type(self.scan) is not str or _SCAN.fullmatch(self.scan) is None):
            raise ValueError("SPEC scan must be an exact N or N.M selector")
        if (
            type(self.image_stem) is not str
            or (not native and not self.image_stem)
            or self.image_stem.strip() != self.image_stem
            or len(self.image_stem.encode("utf-8")) > 4096
            or any(mark in self.image_stem for mark in ("\0", "/", "\\"))
        ):
            raise ValueError("image stem must be one bounded filename substring")
        if type(self.frame_selector) is not StitchFrameSelector:
            raise TypeError("frame selector must be exact StitchFrameSelector")
        if (
            type(self.detector_shape) is not tuple
            or len(self.detector_shape) != 2
            or any(type(item) is not int or item < 1 for item in self.detector_shape)
        ):
            raise ValueError("detector shape must contain two positive exact integers")
        if type(self.raw_dtype) is not str or not self.raw_dtype:
            raise TypeError("raw dtype must be a nonempty string")
        try:
            dtype = np.dtype(self.raw_dtype)
        except TypeError as error:
            raise ValueError("raw dtype is not a NumPy dtype") from error
        if dtype.fields is not None or dtype.subdtype is not None or dtype.kind not in "biuf":
            raise ValueError("raw dtype must be one scalar real numeric dtype")
        if (
            type(self.raw_header_skip) is not int
            or not 0 <= self.raw_header_skip <= (1 << 30)
        ):
            raise ValueError("raw header skip must be an integer from 0 through 1 GiB")
        threshold = self.threshold
        if threshold is not None and (
            type(threshold) not in {int, float}
            or not math.isfinite(float(threshold))
        ):
            raise ValueError("detector threshold must be finite or absent")
        if type(self.geometry_kind) is not StitchGeometryKind:
            raise TypeError("geometry kind must be exact StitchGeometryKind")
        suffix = Path(geometry).suffix.casefold()
        expected_suffix = (
            ".json"
            if self.geometry_kind is StitchGeometryKind.PYFAI_GONIOMETER_JSON
            else ".poni"
        )
        if suffix != expected_suffix:
            raise ValueError(f"geometry path must end in {expected_suffix}")
        if self.expected_geometry_sha256 is not None and (
            type(self.expected_geometry_sha256) is not str
            or _SHA256.fullmatch(self.expected_geometry_sha256) is None
        ):
            raise ValueError("expected geometry SHA-256 must be lowercase hex")
        motors = _pairs(self.source_motors, "source motor mapping")
        references = _references(self.poni_references)
        if self.geometry_kind is StitchGeometryKind.PYFAI_GONIOMETER_JSON:
            if references:
                raise ValueError("fitted JSON geometry cannot carry PONI references")
        elif (  # a single detector arm maps its one moving circle, del or nu
            {name for name, _source in motors} not in ({"del", "nu"}, {"del"}, {"nu"})
            or {name for name, _value in references} != {name for name, _source in motors}
        ):
            raise ValueError("PONI geometry requires del and/or nu mappings and references")
        if type(self.image_rotation) is not int or self.image_rotation not in {
            0,
            90,
            180,
            270,
        }:
            raise ValueError("image rotation must be 0, 90, 180, or 270")
        if type(self.image_flip) is not bool:
            raise ValueError("image flip must be an exact boolean")
        q_range = None if native and self.q_range is None else _ordered_range(self.q_range, "radial range")
        if type(self.npt_1d) is not int or not 1 <= self.npt_1d <= 1_000_000:
            raise ValueError("1-D bin count must be a bounded positive integer")
        if self.monitor_selector is not None and type(
            self.monitor_selector
        ) is not MetadataColumnSelector:
            raise TypeError("monitor selector must be exact MetadataColumnSelector")
        if type(self.use_detector_mask) is not bool:
            raise TypeError("detector mask toggle must be an exact bool")
        if type(self.overwrite) is not AnalysisArtifactOverwrite:
            raise TypeError("overwrite policy must be exact AnalysisArtifactOverwrite")
        if (
            type(self.max_frame_bytes) is not int
            or not 1 <= self.max_frame_bytes <= 4 * 1024 * 1024 * 1024
        ):
            raise ValueError("max frame bytes must be a bounded positive integer")
        if math.prod(self.detector_shape) * 8 > self.max_frame_bytes:
            raise ValueError("float64 detector frame exceeds max frame bytes")
        if self.mode not in {"1d", "2d"}:
            raise ValueError("Stitch mode must be 1d or 2d")
        if self.mode == "2d":
            if not native:
                raise ValueError("Stitch 2D writes a native record: choose its sources on the Stitch page")
            for name in ("npt_rad_2d", "npt_azim_2d"):
                value = getattr(self, name)
                if type(value) is not int or not 1 <= value <= 1_000_000:
                    raise ValueError("2-D point counts must be bounded positive integers")
            if self.npt_rad_2d * self.npt_azim_2d > 64_000_000:
                raise ValueError("a 2-D map may hold at most 64 million points")
            if self.chi_range is not None:
                object.__setattr__(self, "chi_range", _ordered_range(self.chi_range, "χ range"))
            if self.map_unit not in {"q_A^-1", "2th_deg"}:
                raise ValueError("Choose Q-χ or 2θ-χ for the map")
            if self.gi is not None and self.map_unit != "q_A^-1":
                raise ValueError("a grazing-incidence map is binned in Q")
            if self.map_radial_range is not None:
                object.__setattr__(self, "map_radial_range", _ordered_range(self.map_radial_range, "map range"))
        if not native and self.backend == "xu_hist" and (
            self.geometry_kind is not StitchGeometryKind.PYFAI_GONIOMETER_JSON
            or self.expected_geometry_sha256 is not None
            or motors != (("del", "del"), ("nu", "nu"))
            or references
            or self.image_rotation != 0
            or self.image_flip
            or self.detector_shape != (195, 1475)
            or dtype.str != "<i4"
            or self.raw_header_skip != 0
            or threshold != 800000.0
            or self.use_detector_mask is not True
        ):
            raise ValueError(
                "xu_hist requires the locked SURFACE detector, decoder, mapping, "
                "orientation, threshold, and detector-mask contract"
            )
        object.__setattr__(self, "project_root", project)
        object.__setattr__(self, "spec_path", spec)
        object.__setattr__(self, "image_dir", image_dir)
        object.__setattr__(self, "geometry_path", geometry)
        object.__setattr__(self, "output_path", output)
        object.__setattr__(self, "detector_shape", tuple(self.detector_shape))
        object.__setattr__(self, "raw_dtype", dtype.str)
        object.__setattr__(
            self,
            "threshold",
            None if threshold is None else float(threshold),
        )
        object.__setattr__(self, "source_motors", motors)
        object.__setattr__(self, "poni_references", references)
        object.__setattr__(self, "q_range", q_range)
        monitor = self.monitor_selector
        canonical = (
            "stitch-tool-form-v1",
            project,
            spec,
            self.scan,
            image_dir,
            self.image_stem,
            self.frame_selector.fingerprint_value,
            self.detector_shape,
            dtype.str,
            self.raw_header_skip,
            None if threshold is None else float(threshold),
            geometry,
            self.geometry_kind,
            self.expected_geometry_sha256,
            motors,
            references,
            self.image_rotation,
            q_range,
            self.npt_1d,
            None if monitor is None else (monitor.name, monitor.occurrence),
            self.use_detector_mask,
            output,
            self.overwrite,
            self.max_frame_bytes,
            self.mode,
        )
        domain = "stitch-tool-form-v1"
        if self.backend == "xu_hist" and not native:
            domain = "stitch-tool-form-v2-xu-hist"
            canonical = (
                domain,
                self.backend,
                project,
                spec,
                self.scan,
                image_dir,
                self.image_stem,
                self.frame_selector.fingerprint_value,
                self.detector_shape,
                dtype.str,
                self.raw_header_skip,
                float(threshold),
                geometry,
                motors,
                q_range,
                self.npt_1d,
                None if monitor is None else (monitor.name, monitor.occurrence),
                output,
                self.overwrite,
                self.max_frame_bytes,
                self.mode,
            )
        if native:
            domain = "stitch-tool-native-form-v1"
            canonical = (domain, canonical, self.backend,
                         self.source_selection.fingerprint_value, self.unit,
                         None if self.conditioning is None else
                         (self.conditioning.mode, self.conditioning.threshold_min, self.conditioning.threshold_max),
                         self.mask_path, self.output_family, self.output_families,
                         ("native-options-v1", self.frame_screen, self.sample_position,
                          self.correct_solid_angle, self.polarization_factor, self.error_model,
                          self.dark_path, self.flat_path)
                         # a cutoff away from 10 % changes the form; the default keeps it
                         + (() if self.flat_min_fraction == 0.1
                            else (("flat-min-fraction", self.flat_min_fraction),))
                         + (() if self.gi is None else (self.gi.identity(),))
                         # a 1-D form keeps the identity it had
                         + (() if self.mode == "1d" else
                            (("stitch-2d-v2", self.npt_rad_2d, self.npt_azim_2d, self.chi_range,
                              self.map_unit, self.map_radial_range),)))
        if self.image_flip:  # a mirrored camera's; an unmirrored form keeps the identity it had
            canonical = (canonical, ("image-flip", True))
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(domain, canonical),
        )

    def _native_options(self, native: bool) -> None:
        if self.frame_screen not in EXCLUDE_CHOICES:
            raise ValueError(f"frame check must be one of {EXCLUDE_CHOICES}")
        if self.sample_position not in SAMPLE_POSITIONS:
            raise ValueError(f"sample position must be one of {SAMPLE_POSITIONS}")
        if type(self.correct_solid_angle) is not bool:
            raise TypeError("solid angle correction must be an exact bool")
        factor = self.polarization_factor
        if factor is not None:
            if (type(factor) not in {int, float} or not math.isfinite(float(factor))
                    or not -1.0 <= float(factor) <= 1.0):
                raise ValueError("polarization factor must be from -1 to 1, or absent")
            object.__setattr__(self, "polarization_factor", float(factor))
        if self.error_model not in STITCH_ERROR_MODELS:
            raise ValueError(f"error model must be one of {STITCH_ERROR_MODELS}")
        for name in ("dark_path", "flat_path"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _absolute_path(getattr(self, name), name.replace("_", " ")))
        cutoff = self.flat_min_fraction
        if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
                or not math.isfinite(cutoff) or not 0.0 <= cutoff < 1.0):
            raise ValueError("flat cutoff must be from 0 % to below 100 % of the median")
        object.__setattr__(self, "flat_min_fraction", float(cutoff))
        corrected = (self.polarization_factor != 0.99 or self.dark_path is not None
                     or self.flat_path is not None or self.flat_min_fraction != 0.1)
        pyfai_only = (self.sample_position != "calibrated" or not self.correct_solid_angle
                      or self.error_model is not None)
        if not native and (pyfai_only or corrected or self.frame_screen != "empty"):
            raise ValueError("native processing requires a native source selection")
        if self.backend == "xu_hist" and pyfai_only:
            raise ValueError("the XU backend always applies solid angle and has no uncertainty "
                             "or sample position; those need pyFAI MultiGeometry")


@dataclass(frozen=True, slots=True)
class StitchPreflightMember:
    """One selected frame and its exact Project-relative raw locator."""

    label: int
    relative_path: str
    source_frame_index: int
    values: tuple[tuple[str, int, float], ...]

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if (
            type(self.label) is not int
            or type(self.relative_path) is not str
            or not self.relative_path
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or type(self.source_frame_index) is not int
            or self.source_frame_index < 0
            or type(self.values) is not tuple
        ):
            raise TypeError("Stitch preflight member is invalid")
        for value in self.values:
            if (
                type(value) is not tuple
                or len(value) != 3
                or type(value[0]) is not str
                or not value[0]
                or type(value[1]) is not int
                or value[1] < 0
                or type(value[2]) is not float
                or not math.isfinite(value[2])
            ):
                raise TypeError("Stitch preflight member value is invalid")


@dataclass(frozen=True, slots=True)
class StitchPreflightSummary:
    """Bounded operator-facing projection of one exact prepared request."""

    form_fingerprint: str
    request_fingerprint: str
    source_fingerprint: str
    module_source_fingerprint: str
    table_fingerprint: str
    manifest_fingerprint: str
    geometry_fingerprint: str
    geometry_sha256: str
    project_root: str
    source_relative_path: str
    source_scan: str
    image_directory_relative_path: str
    image_stem: str
    geometry_relative_path: str
    geometry_kind: StitchGeometryKind
    source_motors: tuple[tuple[str, str], ...]
    poni_references: tuple[tuple[str, float], ...]
    image_rotation: int
    output_relative_path: str
    selected_labels: tuple[int, ...]
    members: tuple[StitchPreflightMember, ...]
    dependency_files: tuple[str, ...]
    required_selectors: tuple[MetadataColumnSelector, ...]
    detector_shape: tuple[int, int]
    raw_dtype: str
    raw_header_skip: int
    threshold: float | None
    q_range: tuple[float, float]
    npt_1d: int
    monitor_selector: MetadataColumnSelector | None
    use_detector_mask: bool
    overwrite: AnalysisArtifactOverwrite
    max_frame_bytes: int
    holds: tuple[str, ...]
    backend: str = "multigeometry"
    asset_semantic_fingerprint: str | None = None
    effective_geometry_fingerprint: str | None = None
    image_flip: bool = False
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        hashes = (
            self.form_fingerprint,
            self.request_fingerprint,
            self.source_fingerprint,
            self.module_source_fingerprint,
            self.table_fingerprint,
            self.manifest_fingerprint,
            self.geometry_fingerprint,
            self.geometry_sha256,
        )
        if (
            _claim is not _PREFLIGHT_FACTORY
            or any(type(item) is not str or _SHA256.fullmatch(item) is None for item in hashes)
            or type(self.source_scan) is not str
            or not self.source_scan
            or type(self.project_root) is not str
            or not Path(self.project_root).is_absolute()
            or type(self.image_stem) is not str
            or not self.image_stem
            or type(self.geometry_kind) is not StitchGeometryKind
            or type(self.source_motors) is not tuple
            or not self.source_motors
            or type(self.poni_references) is not tuple
            or type(self.image_rotation) is not int
            or self.image_rotation not in {0, 90, 180, 270}
            or type(self.image_flip) is not bool
            or type(self.selected_labels) is not tuple
            or not self.selected_labels
            or type(self.members) is not tuple
            or len(self.members) != len(self.selected_labels)
            or any(type(item) is not StitchPreflightMember for item in self.members)
            or tuple(item.label for item in self.members) != self.selected_labels
            or type(self.dependency_files) is not tuple
            or not self.dependency_files
            or len(self.dependency_files) > 8192
            or any(type(item) is not str or not item for item in self.dependency_files)
            or type(self.required_selectors) is not tuple
            or any(
                type(item) is not MetadataColumnSelector
                for item in self.required_selectors
            )
            or type(self.detector_shape) is not tuple
            or len(self.detector_shape) != 2
            or any(type(item) is not int or item < 1 for item in self.detector_shape)
            or type(self.raw_dtype) is not str
            or not self.raw_dtype
            or type(self.raw_header_skip) is not int
            or self.raw_header_skip < 0
            or (
                self.threshold is not None
                and (
                    type(self.threshold) is not float
                    or not math.isfinite(self.threshold)
                )
            )
            or (
                self.monitor_selector is not None
                and type(self.monitor_selector) is not MetadataColumnSelector
            )
            or type(self.use_detector_mask) is not bool
            or type(self.overwrite) is not AnalysisArtifactOverwrite
            or type(self.holds) is not tuple
            or any(type(item) is not str or not item for item in self.holds)
            or self.backend not in {"multigeometry", "xu_hist"}
            or (
                self.backend == "multigeometry"
                and (
                    self.asset_semantic_fingerprint is not None
                    or self.effective_geometry_fingerprint is not None
                )
            )
            or (
                self.backend == "xu_hist"
                and (
                    type(self.asset_semantic_fingerprint) is not str
                    or _SHA256.fullmatch(self.asset_semantic_fingerprint) is None
                    or type(self.effective_geometry_fingerprint) is not str
                    or _SHA256.fullmatch(self.effective_geometry_fingerprint) is None
                )
            )
        ):
            raise TypeError("Stitch preflight summary is invalid")
        canonical = (
            "stitch-preflight-summary-v1",
            *hashes,
            self.project_root,
            self.source_relative_path,
            self.source_scan,
            self.image_directory_relative_path,
            self.image_stem,
            self.geometry_relative_path,
            self.geometry_kind,
            self.source_motors,
            self.poni_references,
            self.image_rotation,
            self.output_relative_path,
            self.selected_labels,
            tuple(
                (
                    item.label,
                    item.relative_path,
                    item.source_frame_index,
                    item.values,
                )
                for item in self.members
            ),
            self.dependency_files,
            tuple(
                (selector.name, selector.occurrence)
                for selector in self.required_selectors
            ),
            self.detector_shape,
            self.raw_dtype,
            self.raw_header_skip,
            self.threshold,
            self.q_range,
            self.npt_1d,
            (
                None
                if self.monitor_selector is None
                else (
                    self.monitor_selector.name,
                    self.monitor_selector.occurrence,
                )
            ),
            self.use_detector_mask,
            self.overwrite,
            self.max_frame_bytes,
            self.holds,
        )
        domain = "stitch-preflight-summary-v1"
        if self.backend == "xu_hist":
            domain = "stitch-preflight-summary-v2-xu-hist"
            canonical = (
                domain,
                canonical,
                self.backend,
                self.asset_semantic_fingerprint,
                self.effective_geometry_fingerprint,
            )
        if self.image_flip:  # a mirrored camera's; an unmirrored summary keeps the identity it had
            canonical = (canonical, ("image-flip", True))
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(domain, canonical),
        )


@dataclass(eq=False, frozen=True, slots=True)
class StitchToolPreflight:
    """The exact request and summary admitted for a specific frozen form."""

    form: StitchToolForm
    request: StitchOperationRequest | None
    summary: StitchPreflightSummary | None
    _claim: InitVar[object] = None
    requests: tuple[StitchOperationRequest, ...] = ()
    skipped: tuple[tuple[str, str, str], ...] = ()
    # Detached display facts only; the request/provenance remain Run authority.
    preview_members: tuple[tuple[dict, ...], ...] = ()
    warnings: tuple[str, ...] = ()
    #: The outputs (request indices) already stitched with these settings, which Append skips.
    unchanged: tuple[int, ...] = ()

    def __post_init__(self, _claim: object) -> None:
        if type(self.form) is StitchToolForm and self.form.source_selection is not None:
            if (_claim is not _PREFLIGHT_FACTORY or type(self.requests) is not tuple
                or any(type(item) is not StitchOperationRequest for item in self.requests)
                or self.request is not (self.requests[0] if self.requests else None)
                or self.summary is not None or type(self.skipped) is not tuple
                or type(self.preview_members) is not tuple
                or type(self.warnings) is not tuple
                or any(type(warning) is not str or not warning for warning in self.warnings)
                or len(self.preview_members) != len(self.requests)
                or any(type(group) is not tuple or any(type(member) is not dict for member in group)
                       for group in self.preview_members)
                or any(type(item) is not tuple or len(item) != 3 or any(type(v) is not str for v in item) for item in self.skipped)
                or type(self.unchanged) is not tuple
                or any(type(index) is not int for index in self.unchanged)
                or list(self.unchanged) != sorted(set(self.unchanged))
                or any(not 0 <= index < len(self.requests) for index in self.unchanged)):
                raise TypeError("Native Stitch preflight is invalid")
            targets = tuple(item.module.output.target for item in self.requests)
            if len(set(targets)) != len(targets):
                raise ValueError("Stitch outputs must have distinct stable families")
            return
        if (
            _claim is not _PREFLIGHT_FACTORY
            or self.unchanged != ()
            or type(self.form) is not StitchToolForm
            or type(self.request) is not StitchOperationRequest
            or type(self.summary) is not StitchPreflightSummary
            or self.summary.form_fingerprint != self.form.fingerprint
            or self.summary.request_fingerprint != self.request.module.fingerprint
            or self.summary.manifest_fingerprint != self.request.manifest.fingerprint
            or self.summary.geometry_fingerprint
            != (
                self.request.plan.geometry.fingerprint
                if type(self.request.plan) is StitchOperationPlan
                else self.request.plan.calibration.fingerprint
            )
            or self.summary.backend != self.form.backend
            or (
                self.form.backend == "multigeometry"
                and type(self.request.plan) is not StitchOperationPlan
            )
            or (
                self.form.backend == "xu_hist"
                and (
                    type(self.request.plan) is not XuStitchOperationPlan
                    or self.summary.asset_semantic_fingerprint
                    != self.request.plan.calibration.semantic_fingerprint
                    or self.summary.effective_geometry_fingerprint
                    != self.request.plan.effective_geometry.fingerprint
                )
            )
            or self.summary.selected_labels
            != self.request.module.source.selected_labels
        ):
            raise TypeError("Stitch tool preflight is invalid")
        object.__setattr__(self, "requests", (self.request,))

    def is_current(self, form: StitchToolForm | None) -> bool:
        return type(form) is StitchToolForm and form.fingerprint == self.form.fingerprint


def _inside_project(path: str | Path, root: Path, name: str) -> str:
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise StitchToolPreflightRefused(
            f"{name.upper()}_OUTSIDE_PROJECT",
            f"{name.replace('_', ' ')} must resolve inside the selected Project",
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise StitchToolPreflightRefused(
            f"{name.upper()}_INVALID", f"{name.replace('_', ' ')} is invalid"
        )
    return relative.as_posix()


def prepare_stitch_tool(form: StitchToolForm) -> StitchToolPreflight:
    """Perform blocking, read-only preflight for one exact operator form."""

    if type(form) is not StitchToolForm:
        raise TypeError("Stitch preflight requires exact StitchToolForm")
    try:
        project = Path(form.project_root).resolve(strict=True)
    except OSError as error:
        raise StitchToolPreflightRefused(
            "PROJECT_UNAVAILABLE", "selected Project root is unavailable"
        ) from error
    if not project.is_dir():
        raise StitchToolPreflightRefused(
            "PROJECT_UNAVAILABLE", "selected Project root is not a directory"
        )
    if form.source_selection is not None:
        from .stitch_native_values import prepare_native_stitch_tool
        return prepare_native_stitch_tool(form, project)
    output_relative = _inside_project(form.output_path, project, "output")
    if not Path(form.output_path).parent.is_dir():
        raise StitchToolPreflightRefused(
            "OUTPUT_PARENT_UNAVAILABLE",
            "Stitch output parent directory must already exist",
        )
    image_directory_relative = _inside_project(
        form.image_dir, project, "image_directory"
    )
    read_options: dict[str, object] = {
        "detector_shape": form.detector_shape,
        "raw_dtype": form.raw_dtype,
        "raw_header_skip": form.raw_header_skip,
        "threshold": form.threshold,
        # Source reads stay unrotated.  Stitch owns and records orientation.
        "rotation": 0,
    }
    source_spec = SourceSpec(
        form.spec_path,
        SourceKind.SPEC,
        options={
            "scan": form.scan,
            "image_dir": form.image_dir,
            "image_stem": form.image_stem,
            "read_image_kwargs": read_options,
        },
    )
    table = run_metadata_table(MetadataTablePlan(source_spec))
    if table.disposition is not AnalysisDisposition.COMPLETED:
        raise StitchToolPreflightRefused(
            table.code or "SOURCE_PREFLIGHT_REFUSED",
            "SPEC source could not be admitted for Stitch",
            diagnostics=table.diagnostics,
        )
    selected_labels = form.frame_selector.select(table.labels)
    requested_selectors = tuple(
        MetadataColumnSelector(source_name)
        for _geometry_name, source_name in form.source_motors
    ) + (
        () if form.monitor_selector is None else (form.monitor_selector,)
    )
    if form.backend == "xu_hist":
        names = tuple(column.name for column in table.columns)
        energy_count = names.count("energy")
        if energy_count > 1:
            raise StitchToolPreflightRefused(
                "AMBIGUOUS_METADATA_SELECTOR",
                "the optional XU source energy column must be unique",
            )
        if energy_count == 1:
            requested_selectors += (MetadataColumnSelector("energy"),)
    selectors = tuple(
        sorted(
            {
                (selector.name, selector.occurrence): selector
                for selector in requested_selectors
            }.values(),
            key=lambda selector: (selector.name, selector.occurrence),
        )
    )
    try:
        source = ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.STITCH,
            selected_labels=selected_labels,
            resolved_selectors=selectors,
        )
        if form.backend == "multigeometry":
            geometry = capture_stitch_geometry(
                StitchGeometryInput(
                    form.geometry_path,
                    form.geometry_kind,
                    expected_sha256=form.expected_geometry_sha256,
                    source_motors=form.source_motors,
                    reference_motor_positions=form.poni_references,
                    image_rotation=form.image_rotation,
                    image_flip=form.image_flip,
                )
            )
            plan = StitchOperationPlan(
                geometry,
                mode="1d",
                npt_1d=form.npt_1d,
                radial_range=form.q_range,
                monitor_selector=form.monitor_selector,
                use_detector_mask=form.use_detector_mask,
                max_frame_bytes=form.max_frame_bytes,
            )
        else:
            from xrd_tools.integrate.stitch_calibration import xu_calibration_path
            calibration = capture_xu_stitch_calibration(  # a pyFAI file's partner, saved beside it
                XuStitchCalibrationInput(xu_calibration_path(form.geometry_path, project)),
                project_root=project,
            )
            runtime_owner = xu_runtime_session()
            with runtime_owner as session:
                effective = resolve_xu_stitch_effective_geometry(
                    calibration,
                    session,
                ).projection
            if runtime_owner.execution_record is None:
                raise StitchToolPreflightRefused(
                    "XU_RUNTIME_CONFIG_RESTORE_FAILED",
                    "XU runtime restoration was not attested",
                )
            plan = XuStitchOperationPlan(
                calibration,
                effective,
                float(form.q_range[0]),
                float(form.q_range[1]),
                npt_1d=form.npt_1d,
                monitor_selector=form.monitor_selector,
                max_frame_bytes=form.max_frame_bytes,
            )
        output = ModuleOutputRequest(
            form.output_path,
            AnalysisArtifactKind.STITCH_1D,
            form.overwrite,
        )
        request = prepare_stitch_operation(
            source,
            output,
            plan,
            project_root=project,
        )
    except StitchToolPreflightRefused:
        raise
    except (
        XuRuntimeUnsupported,
        XuStitchCalibrationRefused,
        XuStitchScienceRefused,
        StitchOperationRefused,
        ModuleArtifactRefused,
    ) as error:
        raise StitchToolPreflightRefused(error.code) from error
    manifest = request.manifest
    members = tuple(
        StitchPreflightMember(
            contribution.label,
            manifest.files[contribution.file_ordinal].relative_path,
            contribution.source_frame_index,
            contribution.values,
        )
        for contribution in manifest.contributions
    )
    if type(request.plan) is StitchOperationPlan:
        geometry_relative = _inside_project(
            request.plan.geometry.resolved_path, project, "geometry"
        )
        geometry_fingerprint = request.plan.geometry.fingerprint
        geometry_sha256 = request.plan.geometry.sha256
        asset_semantic_fingerprint = None
        effective_geometry_fingerprint = None
    else:
        geometry_relative = request.plan.calibration.resolved_relative_path
        geometry_fingerprint = request.plan.calibration.fingerprint
        geometry_sha256 = request.plan.calibration.raw_sha256
        asset_semantic_fingerprint = request.plan.calibration.semantic_fingerprint
        effective_geometry_fingerprint = request.plan.effective_geometry.fingerprint
    summary = StitchPreflightSummary(
        form_fingerprint=form.fingerprint,
        request_fingerprint=request.module.fingerprint,
        source_fingerprint=request.module.source.analysis.source_fingerprint,
        module_source_fingerprint=request.module.source.fingerprint,
        table_fingerprint=request.module.source.table_fingerprint,
        manifest_fingerprint=manifest.fingerprint,
        geometry_fingerprint=geometry_fingerprint,
        geometry_sha256=geometry_sha256,
        project_root=str(project),
        source_relative_path=manifest.source_relative_path,
        source_scan=manifest.source_scan or "",
        image_directory_relative_path=image_directory_relative,
        image_stem=form.image_stem,
        geometry_relative_path=geometry_relative,
        geometry_kind=form.geometry_kind,
        source_motors=form.source_motors,
        poni_references=form.poni_references,
        image_rotation=form.image_rotation,
        image_flip=form.image_flip,
        output_relative_path=output_relative,
        selected_labels=request.module.source.selected_labels,
        members=members,
        dependency_files=tuple(item.relative_path for item in manifest.files),
        required_selectors=selectors,
        detector_shape=form.detector_shape,
        raw_dtype=form.raw_dtype,
        raw_header_skip=form.raw_header_skip,
        threshold=form.threshold,
        q_range=form.q_range,
        npt_1d=form.npt_1d,
        monitor_selector=form.monitor_selector,
        use_detector_mask=form.use_detector_mask,
        overwrite=form.overwrite,
        max_frame_bytes=form.max_frame_bytes,
        holds=(
            ("2d-orientation-parity", "gi-corrections")
            if form.backend == "multigeometry"
            else (
                "xu-2d-and-chi",
                "sensor-parallax-xu-geometry",
                "xu-sigma-and-variance",
                "mixed-energy-grouped-xu-stitch",
                "unvalidated-platforms-and-live-gates",
            )
        ),
        backend=form.backend,
        asset_semantic_fingerprint=asset_semantic_fingerprint,
        effective_geometry_fingerprint=effective_geometry_fingerprint,
        _claim=_PREFLIGHT_FACTORY,
    )
    return StitchToolPreflight(form, request, summary, _PREFLIGHT_FACTORY)


__all__ = [
    "StitchFrameSelector",
    "StitchPreflightMember",
    "StitchPreflightSummary",
    "StitchToolForm",
    "StitchToolPreflight",
    "StitchToolPreflightRefused",
    "prepare_stitch_tool",
]
