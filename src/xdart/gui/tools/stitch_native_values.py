"""Native Stitch selections projected onto existing finite operation owners."""
import json
from pathlib import Path

from xrd_tools.analysis.module_transaction import (
    MetadataColumnSelector, ModuleKind, ModuleOutputRequest, ModuleSourceReceipt,
    ModuleSourceGroupReceipt,
)
from xrd_tools.analysis.scan_operations import (
    AnalysisDisposition, MetadataTablePlan,
    run_metadata_table,
)
from xrd_tools.analysis.stitch_operation import (
    StitchGeometryInput, StitchGeometryKind, StitchOperationPlan, XuStitchOperationPlan,
    capture_stitch_geometry, prepare_stitch_operation,
)
from xrd_tools.analysis.stitch_sources import resolve_stitch_sources
from xrd_tools.analysis.xu_stitch_calibration import (
    XuStitchCalibrationInput, capture_xu_stitch_calibration,
)
from xrd_tools.core.geometry.xu_runtime import xu_runtime_session
from xrd_tools.integrate.xu_stitch import resolve_xu_stitch_effective_geometry
from xrd_tools.io.analysis_artifact import AnalysisArtifactKind
from xrd_tools.io.output_path import is_artifact_family, resolve_finite_output_target


def _combined(group):
    # Multiple entries of one NeXus file remain one source/file output.
    return len({(str(Path(spec.uri).resolve()), spec.options.get("scan"))
                for spec in group}) > 1


def _families(form, resolution, project):
    """Readable slots are independent of frame subsets and processing settings."""
    from .stitch_values import StitchToolPreflightRefused
    groups = resolution.groups

    def refused(code, message):
        if resolution.skipped:
            skipped = "; ".join(f"{Path(path).relative_to(project)} · scan {scan}: {reason}"
                                for path, scan, reason in resolution.skipped)
            message += f". Skipped scans: {skipped}"
        return StitchToolPreflightRefused(code, message)

    explicit = list(form.output_families or ("",) * len(groups))
    if len(explicit) != len(groups):
        raise refused("OUTPUT_FAMILY_COUNT_MISMATCH",
            "Enter one result name per output, separated by ; (blank names use single-source defaults)")
    if form.output_family:
        combined = [i for i, group in enumerate(groups) if _combined(group)]
        if len(groups) == 1:
            explicit[0] = form.output_family
        elif len(combined) == 1:
            explicit[combined[0]] = form.output_family
        else:
            raise refused("OUTPUT_FAMILY_COUNT_MISMATCH",
                "For multiple outputs, enter result names in output order separated by ;")
    families, seen = [], set()
    for index, (group, name) in enumerate(zip(groups, explicit), 1):
        if not name:
            if _combined(group):
                raise refused("OUTPUT_FAMILY_REQUIRED",
                    f"Output {index} combines sources; enter an explicit result name")
            spec = group[0]
            relative = Path(spec.uri).resolve().relative_to(project)
            # Keep source suffixes meaningful to identity: SPEC may have a suffix.
            stem = relative if spec.options.get("scan") is not None else relative.with_suffix("")
            name = "-".join(stem.parts)
            if "scan" in spec.options:
                name += "-" + str(spec.options["scan"]).removesuffix(".1")
        if not is_artifact_family(name):
            raise refused("OUTPUT_FAMILY_INVALID",
                f"Output {index} name {name!r} is invalid; choose a readable name of at most 80 characters")
        if name.casefold() in seen:
            raise refused("OUTPUT_FAMILY_COLLISION",
                f"Multiple outputs use {name!r}; choose distinct result names")
        seen.add(name.casefold())
        families.append(name)
    return tuple(families)


def _check_output_directory(target, project):
    """Qualify existing ancestry without creating directories during Preview."""
    from .stitch_values import StitchToolPreflightRefused, _inside_project
    _inside_project(target, project, "output")
    ancestor = Path(target).parent
    while not ancestor.exists() and not ancestor.is_symlink():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise StitchToolPreflightRefused("OUTPUT_PARENT_UNAVAILABLE",
            f"Output ancestor is not a directory: {ancestor}")


def create_native_output_directory(request):
    """Run-only setup; the existing publisher still owns target admission."""
    target = Path(request.module.output.target)
    project = Path(request.manifest.project_root)
    _check_output_directory(target, project)
    target.parent.mkdir(parents=True, exist_ok=True)
    _check_output_directory(target, project)


def _affine_geometry_warning(request):
    """Display-only caution from captured geometry and selected motor facts."""
    if type(request.plan) is not StitchOperationPlan:
        return None
    geometry = request.plan.geometry
    if geometry.request.kind is not StitchGeometryKind.PYFAI_GONIOMETER_JSON:
        return None
    asset = json.loads(geometry.content)
    constants = asset["trans_function"].get("constants", {})
    marker = constants.get("xdart_stacked_arm_v1") if type(constants) is dict else None
    if type(marker) in (int, float) and marker == 1:
        return None
    bindings = dict(geometry.request.source_motors)
    nu = bindings.get("nu_angle", bindings.get("nu"))
    delta = bindings.get("del_angle", bindings.get("del"))
    if not nu or not delta:
        return None
    from xrd_tools.core.geometry import Diffractometer
    try:
        instrument = Diffractometer.from_pyfai_goniometer(asset, source_motors=bindings)
    except (KeyError, ValueError, TypeError, NotImplementedError, SyntaxError):
        # Preview's warning does not change existing model admission semantics.
        return None
    if (getattr(instrument, "stacked_arm", None) is not None
        or not instrument.rot1.is_active or instrument.rot1.source_motor != nu
        or not instrument.rot2.is_active or instrument.rot2.source_motor != delta
        or instrument.rot3.is_active):
        return None
    for contribution in request.manifest.contributions:
        values = {(name, occurrence): value for name, occurrence, value in contribution.values}
        if values[(nu, 0)] != 0 and values[(delta, 0)] != 0:
            return (
                f"{Path(request.module.output.target).name}: this calibration uses independent Euler "
                "mappings, and selected frames have both del and nu nonzero. If del is mounted on nu, "
                "these mappings do not describe that rotation order; create a stacked calibration "
                "before using these results."
            )
    return None


def prepare_native_stitch_tool(form, project):
    from .stitch_values import (
        StitchToolPreflight, StitchToolPreflightRefused, _PREFLIGHT_FACTORY,
        _inside_project,
    )
    output_dir = Path(form.output_path).parent
    _check_output_directory(form.output_path, project)
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
    families = _families(form, resolution, project)
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
    requests, preview_members, warnings = [], [], []
    # Metadata is loaded once per exact source. The receipt is reused without
    # changing membership when a scan appears in more than one output group.
    cached = {}
    for group, family in zip(resolution.groups, families):
        receipts = []
        for spec in group:
            cache_key = (str(spec.uri), spec.entry, str(spec.options))
            if cache_key not in cached:
                table = run_metadata_table(MetadataTablePlan(spec))
                if table.disposition is not AnalysisDisposition.COMPLETED:
                    raise StitchToolPreflightRefused(table.code or "SOURCE_PREFLIGHT_REFUSED",
                        f"Source metadata could not be admitted: {spec.uri} (entry {spec.entry or '-'}, scan {spec.options.get('scan', '-')})",
                        diagnostics=table.diagnostics)
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
                missing = tuple(s.name for s in selectors if names.count(s.name) <= s.occurrence)
                if missing:
                    raise StitchToolPreflightRefused("MISSING_REQUIRED_METADATA",
                        f"{spec.uri} (entry {spec.entry or '-'}, scan {spec.options.get('scan', '-')}): "
                        f"missing required column(s): {', '.join(missing)}")
                cached[cache_key] = ModuleSourceReceipt.from_metadata_table(table,
                    kind=ModuleKind.STITCH, selected_labels=selected, resolved_selectors=selectors)
            receipt = cached[cache_key]
            receipts.append(receipt)
        target = resolve_finite_output_target(output_dir, family, operation_token="stitch-1d")
        output = ModuleOutputRequest(target, AnalysisArtifactKind.STITCH_1D,
                                     form.overwrite, artifact_family=family)
        try:
            source = receipts[0] if len(receipts) == 1 else ModuleSourceGroupReceipt.from_members(
                tuple(receipts), kind=ModuleKind.STITCH)
            request = prepare_stitch_operation(source, output, plan, project_root=project)
            from xrd_tools.analysis.stitch_native import native_frame_members
            members = tuple(native_frame_members(request))
        except (ValueError, TypeError) as error:
            count = sum(len(receipt.selected_labels) for receipt in receipts)
            code = getattr(error, "code", "STITCH_PREFLIGHT_REFUSED")
            reason = str(error)
            if code == "MANIFEST_BYTE_LIMIT_EXCEEDED":
                reason = "source manifest exceeds 512 KiB; reduce scans or frames in this output"
            elif "analysis provenance exceeds the bounded input size" in reason:
                reason = "provenance exceeds 1 MiB; reduce scans or frames in this output"
            elif len(receipts) > 256:
                reason = "a grouped result supports at most 256 source entries"
            raise StitchToolPreflightRefused(code,
                f"Output {len(requests) + 1} ({family}), {len(receipts)} sources, {count} frames: {reason}") from error
        requests.append(request)
        preview_members.append(members)
        warning = _affine_geometry_warning(request)
        if warning is not None:
            warnings.append(warning)
    return StitchToolPreflight(form, requests[0], None, _PREFLIGHT_FACTORY,
                               requests=tuple(requests), skipped=resolution.skipped,
                               preview_members=tuple(preview_members), warnings=tuple(warnings))
