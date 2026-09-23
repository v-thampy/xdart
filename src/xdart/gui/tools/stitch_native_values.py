"""Native Stitch selections projected onto existing finite operation owners."""
from pathlib import Path
import re

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector, ModuleKind, ModuleOutputRequest, ModuleSourceReceipt,
    ModuleSourceGroupReceipt,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition, MetadataTablePlan, analysis_canonical_fingerprint,
    run_metadata_table,
)
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryInput, StitchOperationPlan, XuStitchOperationPlan,
    capture_stitch_geometry, prepare_stitch_operation,
)
from xrd_tools.analysis.stitch_sources import resolve_stitch_sources
from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput, capture_xu_stitch_calibration,
)
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
from xrd_tools.io.output_path import resolve_finite_output_target


def _family(group, selected, project, prefix=None):
    """Persist explicit identity, independent of processing/settings revisions."""
    identity = tuple((str(Path(spec.uri).resolve().relative_to(project)), spec.entry,
                      spec.options.get("scan"), labels)
                     for spec, labels in zip(group, selected))
    readable = prefix or "-".join(
        Path(spec.uri).stem + ("-" + str(spec.options["scan"]).removesuffix(".1")
                               if "scan" in spec.options else "")
        for spec in group)
    readable = re.sub(r"[^A-Za-z0-9_-]+", "-", readable).strip("-_")[:66] or "stitch"
    digest = analysis_canonical_fingerprint("stitch-family-v1", identity)[:12]
    return f"{readable}-{digest}"


def prepare_native_stitch_tool(form, project):
    from .stitch_values import (
        StitchToolPreflight, StitchToolPreflightRefused, _PREFLIGHT_FACTORY,
        _inside_project,
    )
    output_dir = Path(form.output_path).parent
    _inside_project(form.output_path, project, "output")
    if not output_dir.is_dir():
        raise StitchToolPreflightRefused("OUTPUT_PARENT_UNAVAILABLE", "Choose an existing save directory")
    for path in form.source_selection.paths:
        # The Project root itself is a valid directory input.
        if Path(path).resolve() != project:
            _inside_project(path, project, "source")
    read_options = dict(detector_shape=form.detector_shape, raw_dtype=form.raw_dtype,
                        raw_header_skip=form.raw_header_skip, rotation=0)
    resolution = resolve_stitch_sources(form.source_selection, read_options)
    if not resolution.groups:
        return StitchToolPreflight(form, None, None, _PREFLIGHT_FACTORY,
                                   requests=(), skipped=resolution.skipped)
    common = dict(npt_1d=form.npt_1d, unit=form.unit, radial_range=form.q_range,
                  monitor_selector=form.monitor_selector, max_frame_bytes=form.max_frame_bytes,
                  native_records=True, conditioning=form.conditioning, mask_path=form.mask_path)
    if form.backend == "multigeometry":
        geometry = capture_stitch_geometry(StitchGeometryInput(
            form.geometry_path, form.geometry_kind,
            expected_sha256=form.expected_geometry_sha256, source_motors=form.source_motors,
            reference_motor_positions=form.poni_references, image_rotation=form.image_rotation,
        ))
        plan = StitchOperationPlan(geometry, use_detector_mask=form.use_detector_mask, **common)
        motors = tuple(source for _name, source in form.source_motors)
        energy_name = "energy"
    else:
        calibration = capture_xu_stitch_calibration(
            XuStitchCalibrationInput(form.geometry_path), project_root=project)
        runtime_owner = xu_runtime_session()
        with runtime_owner as session:
            effective = resolve_xu_stitch_effective_geometry(calibration, session).projection
        if runtime_owner.execution_record is None:
            raise StitchToolPreflightRefused("XU_RUNTIME_CONFIG_RESTORE_FAILED")
        plan = XuStitchOperationPlan(calibration, effective, **common)
        acquisition = calibration.projection.value["acquisition"]
        motors = tuple(acquisition["source_motors"].values())
        energy_name = acquisition["source_energy_selector"]
    requests = []
    # Metadata is loaded once per exact source. The receipt is reused without
    # changing membership when a scan appears in more than one output group.
    cached = {}
    for group in resolution.groups:
        receipts, labels = [], []
        for spec in group:
            cache_key = (str(spec.uri), spec.entry, str(spec.options))
            if cache_key not in cached:
                table = run_metadata_table(MetadataTablePlan(spec))
                if table.disposition is not AnalysisDisposition.COMPLETED:
                    raise StitchToolPreflightRefused(table.code or "SOURCE_PREFLIGHT_REFUSED",
                        "Source metadata could not be admitted", diagnostics=table.diagnostics)
                selected = form.frame_selector.select(table.labels)
                selectors = tuple(MetadataColumnSelector(str(name)) for name in motors)
                selectors += () if form.monitor_selector is None else (form.monitor_selector,)
                names = tuple(column.name for column in table.columns)
                if names.count(energy_name) > 1:
                    raise StitchToolPreflightRefused("AMBIGUOUS_METADATA_SELECTOR", "Energy must be unique")
                if energy_name in names:
                    selectors += (MetadataColumnSelector(energy_name),)
                selectors = tuple(sorted({(s.name, s.occurrence): s for s in selectors}.values(),
                                         key=lambda s: (s.name, s.occurrence)))
                cached[cache_key] = ModuleSourceReceipt.from_metadata_table(table,
                    kind=ModuleKind.STITCH, selected_labels=selected, resolved_selectors=selectors)
            receipt = cached[cache_key]
            receipts.append(receipt)
            labels.append(receipt.selected_labels)
        source = receipts[0] if len(receipts) == 1 else ModuleSourceGroupReceipt.from_members(
            tuple(receipts), kind=ModuleKind.STITCH)
        family = _family(group, labels, project, form.output_family)
        target = resolve_finite_output_target(output_dir, family, operation_token="stitch-1d")
        output = ModuleOutputRequest(target, AnalysisArtifactKind.STITCH_1D,
                                     form.overwrite, artifact_family=family)
        requests.append(prepare_stitch_operation(source, output, plan, project_root=project))
    return StitchToolPreflight(form, requests[0], None, _PREFLIGHT_FACTORY,
                               requests=tuple(requests), skipped=resolution.skipped)
