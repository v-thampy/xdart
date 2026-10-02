"""Small immutable recipes that call the existing public analyzers; no independent fitting policy."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .inputs import SnapshotReference, SourceReference, load_snapshot


def _plain(value):
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in {list, tuple}:
        return tuple(_plain(item) for item in value)
    raise TypeError("notebook settings contain only finite primitive values and sequences")


@dataclass(frozen=True, slots=True)
class FitNotebookRecipe:
    tool: str
    input: SnapshotReference | SourceReference
    settings: tuple[tuple[str, object], ...] = ()
    batch: tuple[SnapshotReference | SourceReference, ...] = ()
    title: str = "Exported current fit request"
    # Native->Cartesian transform for HKL region fits, explicit primitive rows; no inferred lattice metric.
    reciprocal_transform: tuple[tuple[float, ...], ...] | None = None
    orthonormal: bool = False

    def __post_init__(self):
        if self.tool not in {"peak", "phase", "region2d", "region3d"}:
            raise ValueError("notebook tool is peak, phase, region2d or region3d")
        if type(self.input) not in {SnapshotReference, SourceReference} or type(self.title) is not str:
            raise TypeError("notebook input is a snapshot/source reference and title is text")
        pairs = self.settings.items() if type(self.settings) is dict else self.settings
        settings = tuple((name, _plain(value)) for name, value in pairs)
        if any(type(name) is not str for name, _ in settings) or len(dict(settings)) != len(settings):
            raise ValueError("each notebook setting has one text name")
        object.__setattr__(self, "settings", settings)
        batch = tuple(self.batch)
        if any(type(item) not in {SnapshotReference, SourceReference} for item in batch):
            raise TypeError("a notebook batch keeps cheap input references, not hydrated arrays")
        object.__setattr__(self, "batch", batch)
        object.__setattr__(self, "reciprocal_transform", _plain(self.reciprocal_transform))


def recipe_to_json(recipe: FitNotebookRecipe) -> str:
    def reference(item):
        from dataclasses import asdict
        return {"kind": "source" if type(item) is SourceReference else "snapshot", **asdict(item)}
    return json.dumps({"schema": "fit-notebook-recipe-v1", "tool": recipe.tool, "input": reference(recipe.input),
                       "settings": dict(recipe.settings), "batch": [reference(item) for item in recipe.batch],
                       "title": recipe.title, "reciprocal_transform": recipe.reciprocal_transform,
                       "orthonormal": recipe.orthonormal}, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def recipe_from_json(text: str) -> FitNotebookRecipe:
    value = json.loads(text)
    if value.pop("schema") != "fit-notebook-recipe-v1":
        raise ValueError("unsupported fit notebook recipe")
    def reference(item):
        kind = item.pop("kind", "snapshot")
        if kind == "source":
            item["selection"] = tuple((name, _plain(setting)) for name, setting in item["selection"])
            item["file_facts"] = None if item["file_facts"] is None else tuple(item["file_facts"])
            return SourceReference(**item)
        if kind != "snapshot":
            raise ValueError("unsupported notebook input reference")
        return SnapshotReference(**item)
    value["input"] = reference(value["input"])
    value["batch"] = tuple(reference(item) for item in value["batch"])
    return FitNotebookRecipe(**value)


def recipe_analyzer(recipe: FitNotebookRecipe, *, data_root: str | Path):
    """Public analyzer with literal admitted settings; external CIF paths have an explicit root."""
    from xrd_tools.analysis.runner import DisplayedPeakFitAnalyzer, DisplayedPhaseFitAnalyzer
    settings = dict(recipe.settings)
    if recipe.tool == "peak":
        return DisplayedPeakFitAnalyzer(settings)
    if recipe.tool == "phase":
        settings.pop("fit_bounds", None)  # a native selection mask, applied before central phase conversion
        paths = settings.get("cif_paths", ())
        settings["cif_paths"] = tuple(Path(path) if Path(path).is_absolute() else Path(data_root) / path for path in paths)
        return DisplayedPhaseFitAnalyzer(settings)
    from xrd_tools.analysis.fitting.region import PeakSeed
    from xrd_tools.analysis.fitting.region_analyzers import Region2DFitAnalyzer, Region3DFitAnalyzer
    # Seed tuples: stable ID, centre, optional FWHMs, optional height, all native coordinates.
    seeds = tuple(PeakSeed(*seed) for seed in settings.pop("seeds", ()))
    settings.pop("region_bounds", None)
    analyzer = Region2DFitAnalyzer if recipe.tool == "region2d" else Region3DFitAnalyzer
    return analyzer(seeds=seeds, **settings)


def analyze_recipe_snapshot(recipe, snapshot, *, data_root):
    """Adapt native selection coordinates through the shared source owner, then call the public analyzer."""
    from dataclasses import replace
    from xrd_tools.analysis.runner import Overlay
    import numpy as np
    native = snapshot
    if recipe.tool == "phase":
        from xrd_tools.analysis.fit_sources import phase_fit_snapshot
        bounds = dict(recipe.settings).get("fit_bounds")
        if bounds is not None:
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError("FIT_BOUNDS_INVALID: native low must be less than high")
            axis = snapshot.axes[0].values
            valid = np.ones(axis.shape, dtype=bool) if snapshot.mask is None else snapshot.mask
            snapshot = replace(snapshot, mask=valid & (axis >= bounds[0]) & (axis <= bounds[1]))
        snapshot = phase_fit_snapshot(snapshot)
    analyzer = recipe_analyzer(recipe, data_root=data_root)
    outcome = analyzer.analyze(_region_input(recipe, snapshot) if recipe.tool.startswith("region") else snapshot)
    if recipe.tool == "phase" and native.axes[0].unit in {"2th_deg", "2th_rad"} and outcome.overlay is not None:
        from xrd_tools.core.containers import convert_radial_axis
        overlay = outcome.overlay
        convert = lambda values: convert_radial_axis(np.asarray(values), "q_A^-1", native.axes[0].unit,
                                                     native.wavelength_angstrom)
        outcome = replace(outcome, overlay=Overlay(convert(overlay.x), overlay.traces,
                                                   list(convert(overlay.markers))))
    return outcome


def _region_input(recipe, snapshot):
    from xrd_tools.analysis.fitting.region import RegionAxis
    from xrd_tools.analysis.fitting.region_analyzers import REGION_2D_KEY, REGION_3D_KEY, Region2DInput, Region3DInput
    from xrd_tools.analysis.runner import AnalysisInput
    import numpy as np
    from dataclasses import replace
    from xrd_tools.analysis.fit_snapshot import FitAxis
    bounds = dict(recipe.settings).get("region_bounds")
    if bounds is not None:
        bounds = dict(bounds)
        selectors, selected_axes = [], []
        for axis in snapshot.axes:
            lo, hi = bounds[axis.name]
            if not lo < hi:
                raise ValueError("FIT_REGION_EMPTY: native axis low must be less than high")
            indices = np.flatnonzero((axis.values >= lo) & (axis.values <= hi))
            if indices.size < 2:
                raise ValueError(f"FIT_REGION_EMPTY: {axis.name} needs at least two native bins")
            selectors.append(slice(int(indices[0]), int(indices[-1]) + 1))
            selected_axes.append(FitAxis(axis.name, axis.unit, axis.values[selectors[-1]]))
        selectors = tuple(selectors)
        snapshot = replace(snapshot, axes=tuple(selected_axes), values=snapshot.values[selectors],
                           mask=None if snapshot.mask is None else snapshot.mask[selectors],
                           sigma=None if getattr(snapshot, "sigma", None) is None else snapshot.sigma[selectors])
    axes = tuple(RegionAxis(axis.name, axis.values, axis.unit) for axis in snapshot.axes)
    mask = None if snapshot.mask is None else ~snapshot.mask  # snapshot True includes; region True excludes
    sigma = getattr(snapshot, "sigma", None)
    if recipe.tool == "region2d":
        if len(axes) != 2:
            raise ValueError("region2d recipe requires a selected two-dimensional slice")
        key, region = REGION_2D_KEY, Region2DInput(snapshot.values, *axes, mask, sigma)
    else:
        if len(axes) != 3:
            raise ValueError("region3d recipe requires a selected fitting box")
        transform = None if recipe.reciprocal_transform is None else np.asarray(recipe.reciprocal_transform)
        key, region = REGION_3D_KEY, Region3DInput(snapshot.values, axes, mask, sigma, transform, recipe.orthonormal)
    return AnalysisInput(snapshot.key.result_identity, np.empty(0), np.empty(0), metadata={key: region})


def run_recipe(recipe: FitNotebookRecipe, *, bundle_root: str | Path, data_root: str | Path | None = None,
               reference: SnapshotReference | SourceReference | None = None, loader=load_snapshot):
    """Execute one current request normally, including headless/script runs with no Fit click."""
    kwargs = {"data_root": data_root} if loader is load_snapshot else {}
    snapshot = loader(reference or recipe.input, bundle_root=bundle_root, **kwargs)
    return analyze_recipe_snapshot(recipe, snapshot, data_root=bundle_root if data_root is None else data_root)


def run_recipe_batch(recipe: FitNotebookRecipe, references=None, *, bundle_root: str | Path,
                     data_root: str | Path | None = None, loader=load_snapshot, on_progress=None,
                     should_cancel=None, on_frame=None):
    """Lazy hydrate-fit-release batch using the existing compact runner; no arrays retained in descriptors."""
    from xrd_tools.analysis.runner import run_batch
    class RecipeAnalyzer:
        unit = "frame"
        kind = recipe.tool
        def analyze(self, snapshot):
            return analyze_recipe_snapshot(recipe, snapshot, data_root=bundle_root if data_root is None else data_root)
    analyzer = RecipeAnalyzer()
    def descriptor(reference):
        def load():
            kwargs = {"data_root": data_root} if loader is load_snapshot else {}
            snapshot = loader(reference, bundle_root=bundle_root, **kwargs)
            return snapshot
        load.label = reference.label
        return load
    descriptors = tuple(descriptor(reference) for reference in (recipe.batch if references is None else references))
    return run_batch(analyzer, descriptors, compact=True, on_progress=on_progress,
                     should_cancel=should_cancel, on_frame=on_frame)
