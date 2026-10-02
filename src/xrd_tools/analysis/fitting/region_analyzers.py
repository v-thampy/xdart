"""Region-fit analyzers for the analysis runner (headless, Qt-free). FIT-TOOLS-2026-10 H4/H5.

They implement :class:`xrd_tools.analysis.runner.Analyzer` (``kind``, ``unit = "frame"``, ``analyze``) for the 2-D
and 3-D region fits of :mod:`xrd_tools.analysis.fitting.region`. A runner input is 1-D by design, so a region
travels on ``AnalysisInput.metadata`` under :data:`REGION_2D_KEY` / :data:`REGION_3D_KEY` as a detached
:class:`Region2DInput` / :class:`Region3DInput` (the hosts build it from their fit-input snapshot). The outcome's
``params`` are :meth:`RegionFitOutcome.flat_params` (keys carry the stable seed IDs, so a batch's columns stay
aligned); ``overlay`` is None (a region outcome carries its own axes, units, shape order and validity mask); the
:class:`RegionFitOutcome` itself is ``result.payload``. A refusal or failure is an inert ``ok=False`` outcome.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from xrd_tools.analysis.fitting.region import (
    PeakSeed, RegionAxis, RegionFitCancelled, RegionFitRefused, fit_2d_region, fit_3d_region,
)
from xrd_tools.analysis.plans import AnalysisResult
from xrd_tools.analysis.runner import AnalysisInput, AnalysisOutcome

__all__ = [
    "REGION_2D_KEY", "REGION_3D_KEY", "Region2DFitAnalyzer", "Region2DInput", "Region3DFitAnalyzer",
    "Region3DInput",
]

REGION_2D_KEY = "fit_region_2d"
REGION_3D_KEY = "fit_region_3d"


@dataclass(frozen=True, slots=True)
class Region2DInput:
    """A detached 2-D region in ``(x, y)`` data order; ``mask`` True excludes; ``sigma`` admitted or None."""

    data: np.ndarray
    x: RegionAxis
    y: RegionAxis
    mask: np.ndarray | None = None
    sigma: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Region3DInput:
    """A detached 3-D region (axis order) with its optional reciprocal transform (native -> Cartesian Q)."""

    data: np.ndarray
    axes: tuple[RegionAxis, RegionAxis, RegionAxis]
    mask: np.ndarray | None = None
    sigma: np.ndarray | None = None
    reciprocal_transform: np.ndarray | None = None
    orthonormal: bool = False


def _run(label: str, region, expected: type, key: str, fit) -> AnalysisOutcome:
    if type(region) is not expected:
        return AnalysisOutcome(label, ok=False, message=f"no {expected.__name__} on metadata[{key!r}]")
    try:
        outcome = fit(region)
    except (RegionFitRefused, RegionFitCancelled, ValueError) as error:
        return AnalysisOutcome(label, ok=False, message=str(error))
    return AnalysisOutcome(
        label=label, ok=outcome.diagnostics.success, params=outcome.flat_params(), overlay=None,
        result=AnalysisResult(kind=key, payload=outcome), message=outcome.diagnostics.message,
    )


@dataclass
class Region2DFitAnalyzer:
    """Frame-unit 2-D region fit: rotated peaks (declared seeds) plus a plane."""

    seeds: Sequence[PeakSeed] = field(default_factory=tuple)
    profile: str = "gaussian"
    background: str = "linear"
    use_uncertainties: bool = False
    kind: str = "peak_fit_2d"
    unit: str = "frame"

    def analyze(self, inp: AnalysisInput) -> AnalysisOutcome:
        region = (inp.metadata or {}).get(REGION_2D_KEY)

        def fit(item: Region2DInput):
            if self.use_uncertainties and item.sigma is None:
                raise RegionFitRefused("WEIGHTS_UNUSABLE", "the input carries no admitted uncertainties")
            return fit_2d_region(item.data, item.x, item.y, seeds=tuple(self.seeds), profile=self.profile,
                                 background=self.background, mask=item.mask,
                                 sigma=item.sigma if self.use_uncertainties else None)

        return _run(inp.label, region, Region2DInput, REGION_2D_KEY, fit)

    def analyze_scan(self, inputs: Sequence[AnalysisInput]) -> AnalysisOutcome:
        raise NotImplementedError("Region2DFitAnalyzer is frame-unit; the runner calls analyze() per input.")


@dataclass
class Region3DFitAnalyzer:
    """Frame-unit 3-D region fit: one to three seeded Gaussians plus an affine background."""

    seeds: Sequence[PeakSeed] = field(default_factory=tuple)
    background: str = "linear"
    use_uncertainties: bool = False
    kind: str = "peak_fit_3d"
    unit: str = "frame"

    def analyze(self, inp: AnalysisInput) -> AnalysisOutcome:
        region = (inp.metadata or {}).get(REGION_3D_KEY)

        def fit(item: Region3DInput):
            if self.use_uncertainties and item.sigma is None:
                raise RegionFitRefused("WEIGHTS_UNUSABLE", "the input carries no admitted uncertainties")
            return fit_3d_region(item.data, item.axes, seeds=tuple(self.seeds), background=self.background,
                                 mask=item.mask, sigma=item.sigma if self.use_uncertainties else None,
                                 reciprocal_transform=item.reciprocal_transform, orthonormal=item.orthonormal)

        return _run(inp.label, region, Region3DInput, REGION_3D_KEY, fit)

    def analyze_scan(self, inputs: Sequence[AnalysisInput]) -> AnalysisOutcome:
        raise NotImplementedError("Region3DFitAnalyzer is frame-unit; the runner calls analyze() per input.")
