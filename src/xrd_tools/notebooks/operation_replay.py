"""Portable operator intent replayed by the existing Stitch and RSM transaction owners.

Capture is filesystem-free. Only explicit Run creates a fresh output namespace beneath DATA_ROOT;
external input paths stay explicit, and no replay defaults to replacing an earlier result.
"""
from dataclasses import fields, replace
from pathlib import Path
import math
import threading
from types import MappingProxyType

from xrd_tools.analysis.module_transaction import MetadataColumnSelector, ModuleDisposition
from xrd_tools.analysis.stitch_values import StitchToolForm, StitchFrameSelector, prepare_stitch_tool
from xrd_tools.analysis.stitch_native_values import create_native_output_directory
from xrd_tools.analysis.rsm_values import (
    RSMToolFormV2, RSMScanMemberForm, RSMNexusMemberForm, RSMFrameSelector, prepare_rsm_tool_v2,
)
from xrd_tools.analysis.stitch_operation import StitchGeometryKind, StitchFoilNormalization, run_stitch_operation
from xrd_tools.analysis.stitch_sources import StitchSourceSelection
from xrd_tools.analysis.rsm_operation import (
    RSMImageConditioning, RSMNormalizationMode, RSMNormalizationPolicy, RSMFrameCorrections,
    RSMUBOverride, run_rsm_operation_v2,
)
from xrd_tools.analysis.rsm_geometry_asset import (
    RSMGeometryAssetInput, RSMCalibrationGeometryInput, rsm_geometry_asset_input, rsm_calibration_geometry_input,
)
from xrd_tools.core.scan import SourceSpec, SourceKind
from xrd_tools.integrate.stitch_conditioning import StitchConditioning
from xrd_tools.integrate.stitch_gi import StitchGI
from xrd_tools.io.analysis_artifact import AnalysisArtifactOverwrite
from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame


_SCHEMA = "operation-replay-v1"
_TYPES = {cls.__name__: cls for cls in (
    StitchToolForm, StitchFrameSelector, StitchSourceSelection, StitchConditioning, StitchGI,
    StitchFoilNormalization, MetadataColumnSelector, RSMToolFormV2, RSMScanMemberForm,
    RSMNexusMemberForm, RSMFrameSelector, RSMImageConditioning, RSMNormalizationPolicy,
    RSMFrameCorrections, RSMUBOverride, RSMGeometryAssetInput, RSMCalibrationGeometryInput, SourceSpec,
)}
_ENUMS = {cls.__name__: cls for cls in (
    StitchGeometryKind, AnalysisArtifactOverwrite, SourceKind, RSMNormalizationMode, RSMCoordinateFrame,
)}
_PATHS = {
    StitchToolForm: {"project_root", "spec_path", "image_dir", "geometry_path", "output_path",
                     "mask_path", "dark_path", "flat_path"},
    StitchSourceSelection: {"paths", "image_dir"},
    RSMToolFormV2: {"project_root", "output_path"},
    RSMScanMemberForm: {"spec_path", "image_dir"},
    RSMNexusMemberForm: {"nexus_path"},
    RSMFrameCorrections: {"mask_path", "dark_path", "flat_path"},
    RSMUBOverride: {"spec_path"},
    SourceSpec: {"uri", "metadata_uri"},
}


def _path(value, project, *, output=False):
    if value is None or value == "":
        return value
    if type(value) is tuple:
        return {"$tuple": [_path(item, project, output=output) for item in value]}
    path = Path(value).expanduser()
    if not path.is_absolute():
        relative = path
    else:
        try:
            relative = path.relative_to(project)
        except ValueError:
            if output:
                raise ValueError("WORKFLOW_OUTPUT_OUTSIDE_PROJECT: the captured output must be in Project")
            return {"$path": str(path), "root": "external"}
    if ".." in relative.parts:
        raise ValueError("WORKFLOW_PATH_INVALID: Project paths cannot escape DATA_ROOT")
    return {"$path": relative.as_posix(), "root": "output" if output else "data"}


def _encode(value, project):
    kind = type(value)
    if value is None or kind in {str, bool, int}:
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("WORKFLOW_VALUE_INVALID: numbers must be finite")
        return value
    if kind in _ENUMS.values():
        return {"$enum": kind.__name__, "value": value.value}
    if kind in _TYPES.values():
        settings = {}
        for field in fields(value):
            if not field.init:
                continue  # fingerprints are recomputed by the exact form constructors
            item = getattr(value, field.name)
            if kind is StitchToolForm and field.name == "allow_extrapolation" and not item:
                continue  # Preserve default replay payloads; old notebooks restore the strict default.
            settings[field.name] = (_path(item, project, output=field.name == "output_path")
                                    if field.name in _PATHS.get(kind, ()) else _encode(item, project))
        return {"$type": kind.__name__, "fields": settings}
    if kind is tuple:
        return {"$tuple": [_encode(item, project) for item in value]}
    if kind in {dict, MappingProxyType} and all(type(key) is str for key in value):
        return {"$mapping": {key: _encode(item, project) for key, item in value.items()}}
    raise TypeError(f"WORKFLOW_TYPE_UNSUPPORTED: {kind.__name__} is not a concrete operator value")


def _decode(value, data_root, output_root):
    kind = type(value)
    if value is None or kind in {str, bool, int}:
        return value
    if kind is float and math.isfinite(value):
        return value
    if kind is not dict:
        raise ValueError("WORKFLOW_VALUE_INVALID: expected a primitive operator value")
    if set(value) == {"$path", "root"}:
        path, root = value["$path"], value["root"]
        if type(path) is not str or not path:
            raise ValueError("WORKFLOW_PATH_INVALID")
        relative = Path(path)
        if root == "external" and relative.is_absolute():
            return str(relative)
        if root not in {"data", "output"} or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("WORKFLOW_PATH_INVALID: relative paths must stay in their explicit root")
        return str((data_root if root == "data" else output_root) / relative)
    if set(value) == {"$tuple"} and type(value["$tuple"]) is list:
        return tuple(_decode(item, data_root, output_root) for item in value["$tuple"])
    if set(value) == {"$mapping"} and type(value["$mapping"]) is dict:
        if any(type(key) is not str for key in value["$mapping"]):
            raise ValueError("WORKFLOW_VALUE_INVALID")
        return {key: _decode(item, data_root, output_root) for key, item in value["$mapping"].items()}
    if set(value) == {"$enum", "value"} and value["$enum"] in _ENUMS:
        return _ENUMS[value["$enum"]](value["value"])
    if set(value) == {"$type", "fields"} and value["$type"] in _TYPES and type(value["fields"]) is dict:
        cls = _TYPES[value["$type"]]
        allowed = {field.name for field in fields(cls) if field.init}
        if cls is StitchToolForm and "allow_extrapolation" not in value["fields"]:
            allowed.remove("allow_extrapolation")
        if set(value["fields"]) != allowed:
            raise ValueError("WORKFLOW_FIELDS_INVALID: operator fields must be exact")
        settings = {key: _decode(item, data_root, output_root) for key, item in value["fields"].items()}
        if cls is RSMGeometryAssetInput:
            return rsm_geometry_asset_input(**settings)
        if cls is RSMCalibrationGeometryInput:
            return rsm_calibration_geometry_input(**settings)
        return cls(**settings)
    raise ValueError("WORKFLOW_TYPE_UNSUPPORTED: unknown operator value tag")


def capture_stitch_form(form):
    """Capture current Stitch settings without preparation, source reads or output creation."""
    if type(form) is not StitchToolForm:
        raise TypeError("WORKFLOW_FORM_REQUIRED: exact StitchToolForm required")
    return {"schema": _SCHEMA, "forms": [_encode(form, Path(form.project_root))]}


def capture_rsm_forms(forms):
    """Capture the ordered RSM v2 queue and every member's concrete source selectors."""
    if (type(forms) is not tuple or not forms or len(forms) > 256
            or any(type(form) is not RSMToolFormV2 for form in forms)
            or len({form.project_root for form in forms}) != 1):
        raise TypeError("WORKFLOW_FORM_REQUIRED: one bounded RSM v2 queue in one Project required")
    return {"schema": _SCHEMA, "forms": [_encode(form, Path(form.project_root)) for form in forms]}


def _restore_forms(mode, configuration, *, data_root, output_root):
    if (mode not in {"stitch", "rsm"} or type(configuration) is not dict
            or set(configuration) != {"schema", "forms"} or configuration["schema"] != _SCHEMA
            or type(configuration["forms"]) is not list or not 1 <= len(configuration["forms"]) <= 256):
        raise ValueError("WORKFLOW_CONFIGURATION_INVALID: expected bounded concrete operator forms")
    forms = tuple(_decode(item, Path(data_root).expanduser().absolute(), Path(output_root).expanduser().absolute())
                  for item in configuration["forms"])
    expected = StitchToolForm if mode == "stitch" else RSMToolFormV2
    if any(type(form) is not expected for form in forms) or mode == "stitch" and len(forms) != 1:
        raise ValueError("WORKFLOW_FORM_REQUIRED: recipe mode must match its exact forms")
    return forms


def run_operation_replay(mode, configuration, *, data_root, output_root, on_progress=None, should_cancel=None):
    """Prepare and run the saved operator queue, serially, in a fresh descendant output namespace.

    Existing owners keep all scientific, publication and cleanup authority. Cancellation is sampled before
    preparation, between jobs and at each owner's progress callback; the owner receives its exact Event.
    """
    data_root = Path(data_root).expanduser().resolve(strict=True)
    output_root = Path(output_root).expanduser().absolute()
    if not data_root.is_dir() or output_root == data_root or not output_root.resolve().is_relative_to(data_root):
        raise ValueError("WORKFLOW_OUTPUT_OUTSIDE_PROJECT: OUTPUT_ROOT must be a fresh descendant of DATA_ROOT")
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("WORKFLOW_OUTPUT_EXISTS: choose a fresh OUTPUT_ROOT; replay never replaces outputs")
    forms = _restore_forms(mode, configuration, data_root=data_root, output_root=output_root)
    if any(Path(form.project_root).resolve() != data_root
           or not Path(form.output_path).resolve().is_relative_to(output_root.resolve()) for form in forms):
        raise ValueError("WORKFLOW_OUTPUT_OUTSIDE_NAMESPACE: every output must stay in the fresh OUTPUT_ROOT")
    cancel = threading.Event()

    def cancelled():
        if should_cancel is not None and should_cancel():
            cancel.set()
        return cancel.is_set()

    def progress(value):
        cancelled()
        if on_progress is not None:
            on_progress(value)
        cancelled()

    if cancelled():
        return ()
    # Reserve only the operator's explicit new namespace. No calibration, input, or original output is copied.
    output_root.mkdir(parents=True, exist_ok=False)
    results = []
    for form in forms:
        if cancelled():
            break
        form = replace(form, overwrite=AnalysisArtifactOverwrite.CREATE_NEW)
        # A legacy Stitch preflight needs its parent already present; native owners otherwise create it on Run.
        if mode == "stitch" and form.source_selection is None:
            Path(form.output_path).parent.mkdir(parents=True, exist_ok=True)
        prepared = (prepare_stitch_tool(form) if mode == "stitch"
                    else prepare_rsm_tool_v2(form, cancel_token=cancel))
        requests = prepared.requests if mode == "stitch" else (prepared.request,)
        for request in requests:
            if cancelled():
                break
            if mode == "stitch" and form.source_selection is not None:
                create_native_output_directory(request)  # the same Run-only setup as StitchToolOwner
            result = (run_stitch_operation if mode == "stitch" else run_rsm_operation_v2)(
                request, cancel_token=cancel, progress_callback=progress)
            results.append(result)
            if result.terminal.disposition is not ModuleDisposition.COMMITTED:
                return tuple(results)  # preserve refusal/cancellation/publication truth, without another job
    return tuple(results)
