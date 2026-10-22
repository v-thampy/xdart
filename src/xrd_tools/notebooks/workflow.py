"""Literal page-Run recipes over the existing finite reduction/operation owners."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path


def _plain(value):
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in {list, tuple}:
        return [_plain(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _plain(item) for key, item in value.items()}
    raise TypeError("workflow configuration contains only finite JSON primitives")


@dataclass(frozen=True, slots=True)
class WorkflowRecipe:
    mode: str
    configuration: dict

    def __post_init__(self):
        if self.mode not in {"int", "stitch", "rsm"}:
            raise ValueError("workflow mode is int, stitch or rsm")
        if type(self.configuration) is not dict:
            raise TypeError("workflow configuration is a primitive JSON object")
        object.__setattr__(self, "configuration", _plain(self.configuration))


def workflow_to_json(recipe: WorkflowRecipe) -> str:
    return json.dumps({"schema": "xdart.workflow-recipe-v1", "mode": recipe.mode,
                       "configuration": _plain(recipe.configuration)}, ensure_ascii=False, allow_nan=False, indent=2)


def workflow_from_json(text: str) -> WorkflowRecipe:
    document = json.loads(text)
    if document.pop("schema") != "xdart.workflow-recipe-v1":
        raise ValueError("unsupported workflow recipe")
    return WorkflowRecipe(**document)


def run_workflow(recipe: WorkflowRecipe, *, data_root, output_root, on_progress=None, should_cancel=None):
    """Explicit processing call; export and notebook opening never call it automatically.

    Outputs use a fresh descendant of the source Project. Existing targets are refused before the scientific
    owner is entered. That owner retains source admission, finite/live policy, cancellation and publication.
    """
    recipe = WorkflowRecipe(recipe.mode, recipe.configuration)
    data_root, output_root = Path(data_root).expanduser().resolve(), Path(output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_EXISTS: choose a fresh output folder: {output_root}")
    if data_root not in output_root.parents:
        raise ValueError("OUTPUT_ROOT_INVALID: outputs must be a fresh descendant of DATA_ROOT")
    if recipe.mode == "int":
        from .int_replay import run_int_replay
        return run_int_replay(recipe.configuration, data_root=data_root, output_root=output_root,
                              on_progress=on_progress, should_cancel=should_cancel)
    from .operation_replay import run_operation_replay
    return run_operation_replay(recipe.mode, recipe.configuration, data_root=data_root, output_root=output_root,
                                on_progress=on_progress, should_cancel=should_cancel)


def workflow_result_paths(outputs):
    """Project published paths without discarding the operation result/terminal values returned to the caller."""
    paths = []
    for output in outputs:
        if isinstance(output, (str, Path)):
            paths.append(Path(output))
        else:
            from xrd_tools.analysis.module_transaction import ModuleDisposition
            if output.terminal.disposition is ModuleDisposition.COMMITTED:
                paths.append(Path(output.request.module.output.target))
    return tuple(paths)


@dataclass(frozen=True, slots=True)
class WorkflowResultReference:
    path: Path
    mode: str
    frame: int | None = None

    @property
    def label(self):
        return str(self.path) + (f" :: frame {self.frame}" if self.frame is not None else "")


def workflow_result_references(outputs, mode):
    references = []
    for path in workflow_result_paths(outputs):
        if mode == "int" and path.suffix.lower() != ".xye":
            from xrd_tools.io import open_scan
            references.extend(WorkflowResultReference(path, mode, label) for label in open_scan(path).frame_indices)
        else:
            references.append(WorkflowResultReference(path, mode))
    return tuple(references)


def read_workflow_result(reference: WorkflowResultReference):
    """One selected public processed result, with its recorded native coordinates; never detector input."""
    from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
    key = FitInputKey(reference.label, str(reference.path), "workflow-result", 0)
    if reference.mode == "int":
        if reference.path.suffix.lower() == ".xye":
            from xrd_tools.io import read_xye
            x, y, sigma = read_xye(reference.path)
            return FitInputSnapshot(key, (FitAxis("x", "", x),), y, sigma=sigma)
        from xrd_tools.io import open_scan
        pattern = open_scan(reference.path).get_1d(reference.frame)
        return FitInputSnapshot(key, (FitAxis("radial", pattern.q_unit or "", pattern.q),),
                                pattern.intensity, sigma=pattern.sigma)
    from xrd_tools.io.analysis_artifact import read_analysis_artifact
    from xrd_tools.analysis.fit_sources import stitch_fit_snapshot, rsm_region_snapshot
    payload = read_analysis_artifact(reference.path)
    if reference.mode == "stitch":
        unit = payload.inspection.axis_units[0][1]
        return stitch_fit_snapshot(payload, key, coordinate="2theta" if unit.startswith("2th") else "Q")
    held, coordinates = payload.axes[-1]
    bounds = tuple((name, (float(axis.min()), float(axis.max()))) for name, axis in payload.axes[:-1])
    return rsm_region_snapshot(payload, key, bounds=bounds, plane=held, index=len(coordinates)//2)
