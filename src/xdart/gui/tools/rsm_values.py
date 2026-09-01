"""Immutable, Qt-free operator values for the standalone RSM tool.

The editable form carries one already-typed :class:`RSMOperationPlan`.
Preflight binds it to an exact SPEC scan and returns the request that Run must
consume unchanged; the GUI never reconstructs scientific authority at dispatch.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import threading

import numpy as np

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector,
    ModuleKind,
    ModuleOutputRequest,
    ModuleSourceReceipt,
)
from xrd_tools.analysis.rsm_operation import (
    RSMDetectorGeometry,
    RSMImageConditioning,
    RSMNormalizationMode,
    RSMNormalizationPolicy,
    RSMOperationPlan,
    RSMOperationRefused,
    RSMOperationRequest,
    prepare_rsm_operation,
    required_rsm_selectors,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    MetadataTablePlan,
    analysis_canonical_fingerprint,
    run_metadata_table,
)
from xrd_tools.core.geometry import DetectorHeader, ImageOrientation
from xrd_tools.core.scan import SourceKind, SourceSpec
from xrd_tools.io.analysis_artifact import (
    AnalysisArtifactKind,
    AnalysisArtifactOverwrite,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCAN = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_PREFLIGHT_FACTORY = object()
_MAX_SELECTED_FRAMES = 4096
_PSIC_ROLES = ("mu", "eta", "chi", "phi", "nu", "del")


class RSMToolPreflightRefused(ValueError):
    """One RSM form could not be admitted as an exact runnable request."""

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
    if type(spelling) is not str or not spelling.strip() or "\0" in spelling:
        raise ValueError(f"{name} is required")
    return os.path.normcase(os.path.abspath(os.path.expanduser(spelling)))


def _inside_project(path: str | Path, root: Path, name: str) -> str:
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise RSMToolPreflightRefused(
            f"{name.upper()}_OUTSIDE_PROJECT",
            f"{name.replace('_', ' ')} must resolve inside the selected Project",
        ) from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RSMToolPreflightRefused(
            f"{name.upper()}_INVALID",
            f"{name.replace('_', ' ')} is invalid",
        )
    return relative.as_posix()


def _relative_path(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be one safe relative path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class RSMFrameSelector:
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
            raise ValueError("RSM frame-label selector is invalid")

    def select(self, labels: tuple[int, ...]) -> tuple[int, ...]:
        if type(labels) is not tuple or not labels or any(
            type(label) is not int or label < 0 for label in labels
        ):
            raise RSMToolPreflightRefused(
                "SOURCE_LABELS_INVALID", "source frame labels are invalid"
            )
        stop = max(labels) if self.stop_label is None else self.stop_label
        count = (stop - self.start_label) // self.step + 1
        if count > _MAX_SELECTED_FRAMES:
            raise RSMToolPreflightRefused(
                "FRAME_SELECTION_LIMIT_EXCEEDED",
                "RSM selection exceeds 4096 frames",
            )
        expected = tuple(range(self.start_label, stop + 1, self.step))
        expected_set = set(expected)
        selected = tuple(label for label in labels if label in expected_set)
        if not expected or selected != expected:
            raise RSMToolPreflightRefused(
                "FRAME_SELECTION_OUTSIDE_SOURCE",
                "requested frame labels are not one exact ordered source subset",
            )
        return selected

    @property
    def fingerprint_value(self) -> tuple[object, ...]:
        return self.start_label, self.stop_label, self.step


@dataclass(frozen=True, slots=True)
class RSMToolForm:
    """All operator-editable source/output intent plus one typed science plan."""

    project_root: str | Path
    spec_path: str | Path
    scan: str
    image_dir: str | Path
    image_stem: str
    frame_selector: RSMFrameSelector
    detector_shape: tuple[int, int]
    raw_dtype: str
    plan: RSMOperationPlan
    output_path: str | Path
    raw_header_skip: int = 0
    overwrite: AnalysisArtifactOverwrite = AnalysisArtifactOverwrite.CREATE_NEW
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        project = _absolute_path(self.project_root, "Project root")
        spec = _absolute_path(self.spec_path, "SPEC path")
        image_dir = _absolute_path(self.image_dir, "image directory")
        output = _absolute_path(self.output_path, "output path")
        if Path(spec).suffix:
            raise ValueError("RSM SPEC path must be extensionless")
        if Path(output).suffix.casefold() != ".nexus":
            raise ValueError("RSM output path must end in .nexus")
        if type(self.scan) is not str or _SCAN.fullmatch(self.scan) is None:
            raise ValueError("SPEC scan must be an exact N or N.M selector")
        if (
            type(self.image_stem) is not str
            or not self.image_stem
            or self.image_stem.strip() != self.image_stem
            or len(self.image_stem.encode("utf-8")) > 4096
            or any(mark in self.image_stem for mark in ("\0", "/", "\\"))
        ):
            raise ValueError("image stem must be one bounded filename substring")
        if type(self.frame_selector) is not RSMFrameSelector:
            raise TypeError("frame selector must be exact RSMFrameSelector")
        if (
            type(self.detector_shape) is not tuple
            or len(self.detector_shape) != 2
            or any(type(item) is not int or item < 2 for item in self.detector_shape)
        ):
            raise ValueError("detector shape must contain two exact integers >= 2")
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
        if type(self.plan) is not RSMOperationPlan:
            raise TypeError("RSM form plan must be exact RSMOperationPlan")
        plan_shape = (
            self.plan.geometry.header.Nch1,
            self.plan.geometry.header.Nch2,
        )
        if self.detector_shape != plan_shape:
            raise ValueError("raw detector shape must match the RSM detector header")
        if type(self.overwrite) is not AnalysisArtifactOverwrite:
            raise TypeError("overwrite policy must be exact AnalysisArtifactOverwrite")
        object.__setattr__(self, "project_root", project)
        object.__setattr__(self, "spec_path", spec)
        object.__setattr__(self, "image_dir", image_dir)
        object.__setattr__(self, "output_path", output)
        object.__setattr__(self, "detector_shape", tuple(self.detector_shape))
        object.__setattr__(self, "raw_dtype", dtype.str)
        canonical = (
            "rsm-tool-form-v1",
            project,
            spec,
            self.scan,
            image_dir,
            self.image_stem,
            self.frame_selector.fingerprint_value,
            self.detector_shape,
            dtype.str,
            self.raw_header_skip,
            self.plan.fingerprint,
            output,
            self.overwrite,
        )
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint("rsm-tool-form-v1", canonical),
        )


class RSMScanPreset(str, Enum):
    STO_ALIGN_SCAN43 = "sto-align-scan43"


@dataclass(frozen=True, slots=True)
class RSMToolPresetValues:
    """Filesystem-free defaults for one documented operator/science preset."""

    preset: RSMScanPreset
    spec_relative_path: str
    scan: str
    image_directory_relative_path: str
    image_stem: str
    frame_selector: RSMFrameSelector
    detector_shape: tuple[int, int]
    raw_dtype: str
    raw_header_skip: int
    header: DetectorHeader
    roi: tuple[int, int, int, int]
    motor_selectors: tuple[tuple[str, MetadataColumnSelector], ...]
    conditioning: RSMImageConditioning
    normalization: RSMNormalizationPolicy
    quick_bins: tuple[int, int, int]
    full_bins: tuple[int, int, int]
    chunk_size: int
    max_frame_bytes: int
    max_chunk_bytes: int

    def __post_init__(self) -> None:
        if type(self.preset) is not RSMScanPreset:
            raise TypeError("RSM preset kind must be exact")
        _relative_path(self.spec_relative_path, "preset SPEC path")
        _relative_path(
            self.image_directory_relative_path,
            "preset image directory",
        )

    def plan(self, bins: tuple[int, int, int] | None = None) -> RSMOperationPlan:
        selected_bins = self.quick_bins if bins is None else bins
        if selected_bins not in {self.quick_bins, self.full_bins}:
            raise ValueError("preset grid must be the quick or full exact choice")
        return RSMOperationPlan(
            RSMDetectorGeometry(
                self.header,
                self.motor_selectors,
                ImageOrientation(),
                self.roi,
            ),
            self.conditioning,
            self.normalization,
            bins=selected_bins,
            chunk_size=self.chunk_size,
            max_frame_bytes=self.max_frame_bytes,
            max_chunk_bytes=self.max_chunk_bytes,
        )

    def form(
        self,
        project_root: str | Path,
        output_path: str | Path,
        *,
        bins: tuple[int, int, int] | None = None,
        overwrite: AnalysisArtifactOverwrite = AnalysisArtifactOverwrite.CREATE_NEW,
    ) -> RSMToolForm:
        root = Path(project_root)
        return RSMToolForm(
            root,
            root / self.spec_relative_path,
            self.scan,
            root / self.image_directory_relative_path,
            self.image_stem,
            self.frame_selector,
            self.detector_shape,
            self.raw_dtype,
            self.plan(bins),
            output_path,
            self.raw_header_skip,
            overwrite,
        )


def rsm_tool_preset(
    preset: RSMScanPreset = RSMScanPreset.STO_ALIGN_SCAN43,
) -> RSMToolPresetValues:
    """Return immutable scan-43 defaults without probing the filesystem."""

    if preset is not RSMScanPreset.STO_ALIGN_SCAN43:
        raise ValueError("unknown RSM preset")
    return RSMToolPresetValues(
        preset,
        "STO_align",
        "43.1",
        "images",
        "b_thampy_STO_align_scan43_",
        RSMFrameSelector(0, 60, 1),
        (195, 487),
        "int32",
        0,
        DetectorHeader(97.0, 243.0, 0.172, 0.172, 1014.7173, 195, 487),
        (0, -1, 0, -1),
        tuple(
            (role, MetadataColumnSelector(role, 0)) for role in _PSIC_ROLES
        ),
        RSMImageConditioning(1.0e-6, 2.0e10, 100.0),
        RSMNormalizationPolicy(
            RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE,
            MetadataColumnSelector("foil status", 0),
            MetadataColumnSelector("Seconds", 0),
            (1.06, 3.04, 4.65, 9.5),
        ),
        (40, 40, 40),
        (200, 200, 200),
        8,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    )


@dataclass(frozen=True, slots=True)
class RSMPreflightMember:
    label: int
    relative_path: str
    source_frame_index: int
    values: tuple[tuple[str, int, float], ...]
    normalization_divisor: float

    def __post_init__(self) -> None:
        path = Path(self.relative_path)
        if (
            type(self.label) is not int
            or self.label < 0
            or type(self.relative_path) is not str
            or not self.relative_path
            or len(self.relative_path.encode("utf-8")) > 4096
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or type(self.source_frame_index) is not int
            or self.source_frame_index < 0
            or type(self.values) is not tuple
            or not self.values
            or any(
                type(item) is not tuple
                or len(item) != 3
                or type(item[0]) is not str
                or not item[0]
                or item[0].strip() != item[0]
                or type(item[1]) is not int
                or item[1] < 0
                or type(item[2]) is not float
                or not math.isfinite(item[2])
                for item in self.values
            )
            or tuple(self.values)
            != tuple(sorted(self.values, key=lambda item: (item[0], item[1])))
            or len({(item[0], item[1]) for item in self.values})
            != len(self.values)
            or type(self.normalization_divisor) is not float
            or not math.isfinite(self.normalization_divisor)
            or self.normalization_divisor <= 0
        ):
            raise TypeError("RSM preflight member is invalid")


@dataclass(frozen=True, slots=True)
class RSMToolPreflightSummary:
    form_fingerprint: str
    request_fingerprint: str
    source_fingerprint: str
    module_source_fingerprint: str
    table_fingerprint: str
    preflight_fingerprint: str
    plan_fingerprint: str
    project_root: str
    source_relative_path: str
    source_scan: str
    image_directory_relative_path: str
    image_stem: str
    output_relative_path: str
    selected_labels: tuple[int, ...]
    members: tuple[RSMPreflightMember, ...]
    dependency_files: tuple[str, ...]
    required_selectors: tuple[MetadataColumnSelector, ...]
    detector_shape: tuple[int, int]
    cropped_shape: tuple[int, int]
    raw_dtype: str
    raw_header_skip: int
    energy_eV: float
    ub: tuple[tuple[float, float, float], ...]
    q_bounds: tuple[tuple[float, float], ...]
    bins: tuple[int, int, int]
    chunk_size: int
    max_frame_bytes: int
    max_chunk_bytes: int
    conditioning: tuple[float, float | None, float | None]
    normalization_mode: RSMNormalizationMode
    absorption_lengths: tuple[float, float, float, float]
    overwrite: AnalysisArtifactOverwrite
    holds: tuple[str, ...]
    fingerprint: str = field(init=False)
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        hashes = (
            self.form_fingerprint,
            self.request_fingerprint,
            self.source_fingerprint,
            self.module_source_fingerprint,
            self.table_fingerprint,
            self.preflight_fingerprint,
            self.plan_fingerprint,
        )
        try:
            source_relative = _relative_path(
                self.source_relative_path, "RSM summary source path"
            )
            image_relative = _relative_path(
                self.image_directory_relative_path,
                "RSM summary image directory",
            )
            output_relative = _relative_path(
                self.output_relative_path, "RSM summary output path"
            )
            dependencies = tuple(
                _relative_path(item, "RSM summary dependency")
                for item in self.dependency_files
            )
            dtype = np.dtype(self.raw_dtype)
        except (TypeError, ValueError) as error:
            raise TypeError("RSM preflight summary is invalid") from error
        shapes = (self.detector_shape, self.cropped_shape)
        if (
            _claim is not _PREFLIGHT_FACTORY
            or any(type(item) is not str or _SHA256.fullmatch(item) is None for item in hashes)
            or type(self.project_root) is not str
            or not Path(self.project_root).is_absolute()
            or source_relative != self.source_relative_path
            or type(self.source_scan) is not str
            or _SCAN.fullmatch(self.source_scan) is None
            or image_relative != self.image_directory_relative_path
            or type(self.image_stem) is not str
            or not self.image_stem
            or self.image_stem.strip() != self.image_stem
            or len(self.image_stem.encode("utf-8")) > 4096
            or any(mark in self.image_stem for mark in ("\0", "/", "\\"))
            or output_relative != self.output_relative_path
            or type(self.selected_labels) is not tuple
            or not self.selected_labels
            or any(type(item) is not int or item < 0 for item in self.selected_labels)
            or len(set(self.selected_labels)) != len(self.selected_labels)
            or type(self.members) is not tuple
            or any(type(item) is not RSMPreflightMember for item in self.members)
            or tuple(item.label for item in self.members) != self.selected_labels
            or type(self.dependency_files) is not tuple
            or not self.dependency_files
            or dependencies != self.dependency_files
            or len(set(self.dependency_files)) != len(self.dependency_files)
            or type(self.required_selectors) is not tuple
            or not self.required_selectors
            or any(type(item) is not MetadataColumnSelector for item in self.required_selectors)
            or len(
                {(item.name, item.occurrence) for item in self.required_selectors}
            )
            != len(self.required_selectors)
            or any(
                type(shape) is not tuple
                or len(shape) != 2
                or any(type(value) is not int or value < 2 for value in shape)
                for shape in shapes
            )
            or any(
                cropped > detector
                for cropped, detector in zip(
                    self.cropped_shape, self.detector_shape
                )
            )
            or type(self.raw_dtype) is not str
            or dtype.str != self.raw_dtype
            or dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.kind not in "biuf"
            or type(self.raw_header_skip) is not int
            or not 0 <= self.raw_header_skip <= (1 << 30)
            or type(self.energy_eV) is not float
            or not math.isfinite(self.energy_eV)
            or self.energy_eV <= 0
            or type(self.ub) is not tuple
            or len(self.ub) != 3
            or any(type(row) is not tuple or len(row) != 3 for row in self.ub)
            or any(
                type(value) is not float or not math.isfinite(value)
                for row in self.ub
                for value in row
            )
            or type(self.q_bounds) is not tuple
            or len(self.q_bounds) != 3
            or any(
                type(item) is not tuple
                or len(item) != 2
                or any(type(value) is not float for value in item)
                or not all(math.isfinite(value) for value in item)
                or item[1] <= item[0]
                for item in self.q_bounds
            )
            or type(self.bins) is not tuple
            or len(self.bins) != 3
            or any(type(item) is not int or item < 2 for item in self.bins)
            or math.prod(self.bins) > 8_000_000
            or type(self.chunk_size) is not int
            or self.chunk_size < 1
            or type(self.max_frame_bytes) is not int
            or self.max_frame_bytes < 1
            or type(self.max_chunk_bytes) is not int
            or self.max_chunk_bytes < 1
            or type(self.conditioning) is not tuple
            or len(self.conditioning) != 3
            or type(self.conditioning[0]) is not float
            or not math.isfinite(self.conditioning[0])
            or any(
                item is not None
                and (type(item) is not float or not math.isfinite(item))
                for item in self.conditioning[1:]
            )
            or type(self.normalization_mode) is not RSMNormalizationMode
            or type(self.absorption_lengths) is not tuple
            or len(self.absorption_lengths) != 4
            or any(
                type(item) is not float or not math.isfinite(item) or item < 0
                for item in self.absorption_lengths
            )
            or (
                self.normalization_mode is RSMNormalizationMode.IDENTITY
                and self.absorption_lengths != (0.0, 0.0, 0.0, 0.0)
            )
            or (
                self.normalization_mode
                is RSMNormalizationMode.FOIL_TRANSMISSION_EXPOSURE
                and not any(item > 0 for item in self.absorption_lengths)
            )
            or type(self.overwrite) is not AnalysisArtifactOverwrite
            or type(self.holds) is not tuple
            or any(type(item) is not str or not item for item in self.holds)
            or len(set(self.holds)) != len(self.holds)
        ):
            raise TypeError("RSM preflight summary is invalid")
        canonical = (
            "rsm-tool-preflight-summary-v1",
            hashes,
            self.project_root,
            self.source_relative_path,
            self.source_scan,
            self.image_directory_relative_path,
            self.image_stem,
            self.output_relative_path,
            self.selected_labels,
            tuple(
                (
                    item.label,
                    item.relative_path,
                    item.source_frame_index,
                    item.values,
                    item.normalization_divisor,
                )
                for item in self.members
            ),
            self.dependency_files,
            tuple((item.name, item.occurrence) for item in self.required_selectors),
            self.detector_shape,
            self.cropped_shape,
            self.raw_dtype,
            self.raw_header_skip,
            self.energy_eV,
            self.ub,
            self.q_bounds,
            self.bins,
            self.chunk_size,
            self.max_frame_bytes,
            self.max_chunk_bytes,
            self.conditioning,
            self.normalization_mode,
            self.absorption_lengths,
            self.overwrite,
            self.holds,
        )
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                "rsm-tool-preflight-summary-v1", canonical
            ),
        )


@dataclass(eq=False, frozen=True, slots=True)
class RSMToolPreflight:
    form: RSMToolForm
    request: RSMOperationRequest
    summary: RSMToolPreflightSummary
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _PREFLIGHT_FACTORY
            or type(self.form) is not RSMToolForm
            or type(self.request) is not RSMOperationRequest
            or type(self.summary) is not RSMToolPreflightSummary
        ):
            raise TypeError("RSM tool preflight is invalid")
        receipt = self.request.preflight
        source = self.request.module.source
        plan = self.request.plan
        try:
            options = json.loads(receipt.source_options_json)
            read_options = options["read_image_kwargs"]
            project = Path(self.form.project_root).resolve(strict=True)
            form_spec = Path(self.form.spec_path).resolve(strict=True)
            form_image = Path(self.form.image_dir).resolve(strict=True)
            form_output = Path(self.form.output_path).resolve(strict=False)
            output_relative = Path(
                self.request.module.output.target
            ).relative_to(Path(receipt.project_root)).as_posix()
            expected_spec = Path(receipt.project_root) / receipt.source_relative_path
            expected_image = Path(receipt.project_root) / options["image_dir"]
            selected_from_form = self.form.frame_selector.select(
                source.analysis.labels
            )
            holds_value = self.request.provenance["holds"]
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise TypeError("RSM tool preflight is invalid") from error
        expected_members = tuple(
            RSMPreflightMember(
                contribution.label,
                receipt.files[contribution.file_ordinal].relative_path,
                contribution.source_frame_index,
                contribution.values,
                contribution.normalization_divisor,
            )
            for contribution in receipt.contributions
        )
        expected_holds = (
            tuple(holds_value)
            if type(holds_value) is list
            and all(type(item) is str for item in holds_value)
            else ()
        )
        expected_conditioning = (
            plan.conditioning.additive_offset,
            plan.conditioning.high_threshold,
            plan.conditioning.static_hot_threshold,
        )
        if (
            type(read_options) is not dict
            or project != Path(receipt.project_root)
            or form_spec != expected_spec
            or self.form.scan != receipt.source_scan
            or form_image != expected_image
            or self.form.image_stem != options.get("image_stem")
            or selected_from_form != source.selected_labels
            or self.form.detector_shape != receipt.detector_shape
            or read_options.get("detector_shape") != list(self.form.detector_shape)
            or read_options.get("raw_dtype") != self.form.raw_dtype
            or read_options.get("raw_header_skip") != self.form.raw_header_skip
            or read_options.get("threshold") is not None
            or read_options.get("rotation") != 0
            or self.form.plan.fingerprint != plan.fingerprint
            or form_output != Path(self.request.module.output.target)
            or self.form.overwrite != self.request.module.output.overwrite
            or self.summary.form_fingerprint != self.form.fingerprint
            or self.summary.request_fingerprint
            != self.request.module.fingerprint
            or self.summary.source_fingerprint
            != source.analysis.source_fingerprint
            or self.summary.module_source_fingerprint != source.fingerprint
            or self.summary.table_fingerprint != source.table_fingerprint
            or self.summary.preflight_fingerprint != receipt.fingerprint
            or self.summary.plan_fingerprint != plan.fingerprint
            or self.summary.project_root != receipt.project_root
            or self.summary.source_relative_path != receipt.source_relative_path
            or self.summary.source_scan != receipt.source_scan
            or self.summary.image_directory_relative_path
            != options.get("image_dir")
            or self.summary.image_stem != options.get("image_stem")
            or self.summary.output_relative_path != output_relative
            or self.summary.selected_labels != source.selected_labels
            or self.summary.members != expected_members
            or self.summary.dependency_files
            != tuple(item.relative_path for item in receipt.files)
            or self.summary.required_selectors != required_rsm_selectors(plan)
            or self.summary.detector_shape != receipt.detector_shape
            or self.summary.cropped_shape != receipt.cropped_shape
            or self.summary.raw_dtype != self.form.raw_dtype
            or self.summary.raw_header_skip != self.form.raw_header_skip
            or self.summary.energy_eV != receipt.energy_eV
            or self.summary.ub != receipt.ub
            or self.summary.q_bounds != receipt.q_bounds
            or self.summary.bins != plan.bins
            or self.summary.chunk_size != plan.chunk_size
            or self.summary.max_frame_bytes != plan.max_frame_bytes
            or self.summary.max_chunk_bytes != plan.max_chunk_bytes
            or self.summary.conditioning != expected_conditioning
            or self.summary.normalization_mode != plan.normalization.mode
            or self.summary.absorption_lengths
            != plan.normalization.absorption_lengths
            or self.summary.overwrite != self.request.module.output.overwrite
            or self.summary.holds != expected_holds
        ):
            raise TypeError("RSM tool preflight facts disagree with its request")

    def is_current(self, form: RSMToolForm | None) -> bool:
        return type(form) is RSMToolForm and form.fingerprint == self.form.fingerprint


def prepare_rsm_tool(
    form: RSMToolForm,
    *,
    cancel_token: threading.Event | None = None,
) -> RSMToolPreflight:
    """Perform blocking, image-free preflight for one exact operator form."""

    if type(form) is not RSMToolForm:
        raise TypeError("RSM preflight requires exact RSMToolForm")
    if cancel_token is not None and type(cancel_token) is not threading.Event:
        raise TypeError("RSM preflight cancellation must be threading.Event")
    try:
        project = Path(form.project_root).resolve(strict=True)
    except OSError as error:
        raise RSMToolPreflightRefused(
            "PROJECT_UNAVAILABLE", "selected Project root is unavailable"
        ) from error
    if not project.is_dir():
        raise RSMToolPreflightRefused(
            "PROJECT_UNAVAILABLE", "selected Project root is not a directory"
        )
    spec_relative = _inside_project(form.spec_path, project, "spec")
    output_relative = _inside_project(form.output_path, project, "output")
    if Path(form.output_path).is_symlink():
        raise RSMToolPreflightRefused(
            "OUTPUT_SYMLINK_UNSUPPORTED",
            "RSM output target must not be a symbolic link",
        )
    if not (project / output_relative).parent.is_dir():
        raise RSMToolPreflightRefused(
            "OUTPUT_PARENT_UNAVAILABLE",
            "RSM output parent directory must already exist",
        )
    image_relative = _inside_project(form.image_dir, project, "image_directory")
    spec_path = project / spec_relative
    image_path = project / image_relative
    output_path = project / output_relative
    selectors = required_rsm_selectors(form.plan)
    source_spec = SourceSpec(
        spec_path,
        SourceKind.SPEC,
        options={
            "scan": form.scan,
            "image_dir": image_path,
            "image_stem": form.image_stem,
            "read_image_kwargs": {
                "detector_shape": form.detector_shape,
                "raw_dtype": form.raw_dtype,
                "raw_header_skip": form.raw_header_skip,
                "threshold": None,
                "rotation": 0,
            },
            "metadata_column_projection": tuple(
                (selector.name, selector.occurrence) for selector in selectors
            ),
        },
    )
    table = run_metadata_table(
        MetadataTablePlan(source_spec),
        cancel_token=cancel_token,
    )
    if table.disposition is not AnalysisDisposition.COMPLETED:
        raise RSMToolPreflightRefused(
            table.code or "SOURCE_PREFLIGHT_REFUSED",
            "SPEC source could not be admitted for RSM",
            diagnostics=table.diagnostics,
        )
    selected = form.frame_selector.select(table.labels)
    try:
        source = ModuleSourceReceipt.from_metadata_table(
            table,
            kind=ModuleKind.RSM,
            selected_labels=selected,
            resolved_selectors=selectors,
        )
        output = ModuleOutputRequest(
            output_path,
            AnalysisArtifactKind.RSM,
            form.overwrite,
        )
        request = prepare_rsm_operation(
            source,
            output,
            form.plan,
            project_root=project,
            cancel_token=cancel_token,
        )
    except RSMOperationRefused as error:
        raise RSMToolPreflightRefused(error.code, str(error)) from error
    except ValueError as error:
        if str(error) != "metadata table is no longer an exact source fact":
            raise
        raise RSMToolPreflightRefused(
            "SOURCE_REVISION_CHANGED",
            "SPEC source changed while binding the RSM Preview",
            diagnostics=(str(error),),
        ) from error
    receipt = request.preflight
    if (
        receipt.source_relative_path != spec_relative
        or request.module.output.target != str(output_path)
    ):
        raise RSMToolPreflightRefused(
            "PROJECT_BINDING_MISMATCH",
            "RSM source or output no longer matches its Project binding",
        )
    members = tuple(
        RSMPreflightMember(
            contribution.label,
            receipt.files[contribution.file_ordinal].relative_path,
            contribution.source_frame_index,
            contribution.values,
            contribution.normalization_divisor,
        )
        for contribution in receipt.contributions
    )
    provenance = request.provenance
    holds_value = provenance.get("holds", ())
    holds = tuple(holds_value) if type(holds_value) is list else ()
    conditioning = request.plan.conditioning
    summary = RSMToolPreflightSummary(
        form.fingerprint,
        request.module.fingerprint,
        request.module.source.analysis.source_fingerprint,
        request.module.source.fingerprint,
        request.module.source.table_fingerprint,
        receipt.fingerprint,
        request.plan.fingerprint,
        str(project),
        receipt.source_relative_path,
        receipt.source_scan,
        image_relative,
        form.image_stem,
        output_relative,
        request.module.source.selected_labels,
        members,
        tuple(item.relative_path for item in receipt.files),
        selectors,
        receipt.detector_shape,
        receipt.cropped_shape,
        form.raw_dtype,
        form.raw_header_skip,
        receipt.energy_eV,
        receipt.ub,
        receipt.q_bounds,
        request.plan.bins,
        request.plan.chunk_size,
        request.plan.max_frame_bytes,
        request.plan.max_chunk_bytes,
        (
            conditioning.additive_offset,
            conditioning.high_threshold,
            conditioning.static_hot_threshold,
        ),
        request.plan.normalization.mode,
        request.plan.normalization.absorption_lengths,
        form.overwrite,
        holds,
        _claim=_PREFLIGHT_FACTORY,
    )
    return RSMToolPreflight(form, request, summary, _PREFLIGHT_FACTORY)


__all__ = [
    "RSMFrameSelector",
    "RSMPreflightMember",
    "RSMScanPreset",
    "RSMToolForm",
    "RSMToolPreflight",
    "RSMToolPreflightRefused",
    "RSMToolPreflightSummary",
    "RSMToolPresetValues",
    "prepare_rsm_tool",
    "rsm_tool_preset",
]
