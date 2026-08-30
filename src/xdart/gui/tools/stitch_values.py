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
    ModuleKind,
    ModuleOutputRequest,
    ModuleSourceReceipt,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition,
    MetadataTablePlan,
    analysis_canonical_fingerprint,
    run_metadata_table,
)
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryInput,
    StitchGeometryKind,
    StitchOperationPlan,
    StitchOperationRequest,
    capture_stitch_geometry,
    prepare_stitch_operation,
)
from xrd_tools.core.scan import SourceKind, SourceSpec
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
    q_range: tuple[float, float]
    npt_1d: int
    output_path: str | Path
    raw_header_skip: int = 0
    threshold: float | None = None
    expected_geometry_sha256: str | None = None
    poni_references: tuple[tuple[str, float], ...] = ()
    image_rotation: int = 0
    monitor_selector: MetadataColumnSelector | None = None
    use_detector_mask: bool = True
    overwrite: AnalysisArtifactOverwrite = AnalysisArtifactOverwrite.CREATE_NEW
    max_frame_bytes: int = 256 * 1024 * 1024
    mode: str = "1d"
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        project = _absolute_path(self.project_root, "Project root")
        spec = _absolute_path(self.spec_path, "SPEC path")
        image_dir = _absolute_path(self.image_dir, "image directory")
        geometry = _absolute_path(self.geometry_path, "geometry path")
        output = _absolute_path(self.output_path, "output path")
        if Path(spec).suffix:
            raise ValueError("Stitch SPEC path must be extensionless")
        if Path(output).suffix.casefold() != ".nexus":
            raise ValueError("Stitch output path must end in .nexus")
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
        elif (
            {name for name, _source in motors} != {"del", "nu"}
            or {name for name, _value in references} != {"del", "nu"}
        ):
            raise ValueError("PONI geometry requires del/nu mappings and references")
        if type(self.image_rotation) is not int or self.image_rotation not in {
            0,
            90,
            180,
            270,
        }:
            raise ValueError("image rotation must be 0, 90, 180, or 270")
        q_range = _ordered_range(self.q_range, "q range")
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
        if self.mode != "1d":
            raise ValueError("2-D Stitch remains held pending orientation parity")
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
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint("stitch-tool-form-v1", canonical),
        )


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
        object.__setattr__(
            self,
            "fingerprint",
            analysis_canonical_fingerprint(
                "stitch-preflight-summary-v1", canonical
            ),
        )


@dataclass(eq=False, frozen=True, slots=True)
class StitchToolPreflight:
    """The exact request and summary admitted for a specific frozen form."""

    form: StitchToolForm
    request: StitchOperationRequest
    summary: StitchPreflightSummary
    _claim: InitVar[object] = None

    def __post_init__(self, _claim: object) -> None:
        if (
            _claim is not _PREFLIGHT_FACTORY
            or type(self.form) is not StitchToolForm
            or type(self.request) is not StitchOperationRequest
            or type(self.summary) is not StitchPreflightSummary
            or self.summary.form_fingerprint != self.form.fingerprint
            or self.summary.request_fingerprint != self.request.module.fingerprint
            or self.summary.manifest_fingerprint != self.request.manifest.fingerprint
            or self.summary.geometry_fingerprint
            != self.request.plan.geometry.fingerprint
            or self.summary.selected_labels
            != self.request.module.source.selected_labels
        ):
            raise TypeError("Stitch tool preflight is invalid")

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
        geometry = capture_stitch_geometry(
            StitchGeometryInput(
                form.geometry_path,
                form.geometry_kind,
                expected_sha256=form.expected_geometry_sha256,
                source_motors=form.source_motors,
                reference_motor_positions=form.poni_references,
                image_rotation=form.image_rotation,
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
    geometry_relative = _inside_project(
        request.plan.geometry.resolved_path, project, "geometry"
    )
    summary = StitchPreflightSummary(
        form_fingerprint=form.fingerprint,
        request_fingerprint=request.module.fingerprint,
        source_fingerprint=request.module.source.analysis.source_fingerprint,
        module_source_fingerprint=request.module.source.fingerprint,
        table_fingerprint=request.module.source.table_fingerprint,
        manifest_fingerprint=manifest.fingerprint,
        geometry_fingerprint=request.plan.geometry.fingerprint,
        geometry_sha256=request.plan.geometry.sha256,
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
        holds=("2d-orientation-parity", "gi-and-custom-corrections"),
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
