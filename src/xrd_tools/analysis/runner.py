"""Analysis-agnostic runner protocol (headless, Qt-free).

The contract xdart's live + batch runners drive so they can apply ANY analysis
(peak fitting today; phase fitting / sin2psi / texture later) WITHOUT knowing
which one.  An :class:`Analyzer` wraps an analysis Plan
(:mod:`xrd_tools.analysis.plans`) and projects its :class:`AnalysisResult` into
a uniform :class:`AnalysisOutcome`:

* an :class:`Overlay` — ``fit`` / ``background`` / ``residual`` traces (+ markers
  such as peak centers) to layer onto the pattern plot for live display, and
* a flat ``params`` dict (``center_0``, ``fwhm_0``, …) — the per-frame parameter
  series the batch runner accumulates and plots vs frame index.

``unit`` declares granularity: ``"frame"`` (one pattern -> one outcome, e.g.
peak/phase fitting) or ``"scan"`` (a SET of patterns -> one outcome, e.g.
sin2psi/texture across tilts/orientations).  The runner branches ONCE on
``unit`` and is otherwise analysis-agnostic — that single branch is what keeps
the live/batch machinery reusable across every analysis.

Importing this module is cheap and GUI-free; ``lmfit`` (the ``[fitting]`` extra)
is only pulled when an analyzer actually runs a fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from xrd_tools.analysis.plans import AnalysisResult, PeakFitPlan, run_peak_fit


@dataclass
class AnalysisInput:
    """One unit of work for an analyzer: a 1-D pattern (+ optional context)."""

    label: str
    x: np.ndarray
    y: np.ndarray
    sigma: "np.ndarray | None" = None
    x_unit: str = ""
    #: optional richer context (e.g. an IntegrationResult1D) for analyzers that
    #: need more than the bare (x, y) arrays.
    result_1d: Any = None
    metadata: "dict[str, Any]" = field(default_factory=dict)


@dataclass
class Overlay:
    """Layered display traces an analyzer produces over the pattern plot."""

    x: np.ndarray
    #: keyed by trace kind: ``"fit"`` / ``"background"`` / ``"residual"`` / …
    traces: "dict[str, np.ndarray]" = field(default_factory=dict)
    #: x-positions of point markers (e.g. fitted peak centers).
    markers: "list[float]" = field(default_factory=list)


@dataclass
class AnalysisOutcome:
    """Uniform analyzer result: display overlay + flat params + raw result."""

    label: str
    ok: bool
    #: flat scalar params for the per-frame series (``center_0``, ``fwhm_0``, …).
    params: "dict[str, float]" = field(default_factory=dict)
    overlay: "Overlay | None" = None
    #: the raw :class:`AnalysisResult` for callers that need full detail.
    result: "AnalysisResult | None" = None
    message: str = ""


@runtime_checkable
class Analyzer(Protocol):
    """The analysis-agnostic contract the live/batch runners drive."""

    kind: str
    unit: str  # 'frame' | 'scan'

    def analyze(self, inp: AnalysisInput) -> AnalysisOutcome:
        """Analyze ONE input (frame-unit analyzers)."""
        ...

    def analyze_scan(self, inputs: Sequence[AnalysisInput]) -> AnalysisOutcome:
        """Analyze a SET of inputs as one unit (scan-unit analyzers)."""
        ...


# --------------------------------------------------------------------------
# Peak fitting (frame-unit) — the first concrete analyzer.
# --------------------------------------------------------------------------


@dataclass
class PeakFitAnalyzer:
    """Frame-unit analyzer wrapping :class:`PeakFitPlan` + :func:`run_peak_fit`.

    Projects the resulting ``PeakFitResult1D`` payload to an :class:`Overlay`
    (fit / background / residual traces + peak-center markers) and flat
    ``params`` (``center_i`` / ``center_err_i`` / ``fwhm_i`` / ``amplitude_i``).
    PhaseFit / Sin2Psi analyzers follow the same wrap-a-Plan-and-project shape.
    """

    plan: PeakFitPlan = field(default_factory=PeakFitPlan)
    kind: str = "peak_fit"
    unit: str = "frame"

    def analyze(self, inp: AnalysisInput) -> AnalysisOutcome:
        x = np.asarray(inp.x, dtype=float)
        y = np.asarray(inp.y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        try:
            result = run_peak_fit(self.plan, x, y)
        except Exception as exc:  # backend/fit failure -> inert outcome
            return AnalysisOutcome(inp.label, ok=False, message=str(exc))
        payload = result.payload
        return AnalysisOutcome(
            label=inp.label,
            ok=bool(getattr(payload, "success", True)),
            params=_peak_params(payload),
            overlay=_peak_overlay(x, y, payload),
            result=result,
        )

    def analyze_scan(self, inputs: Sequence[AnalysisInput]) -> AnalysisOutcome:
        raise NotImplementedError(
            "PeakFitAnalyzer is frame-unit; the runner calls analyze() per frame."
        )


def _peak_params(payload: Any) -> "dict[str, float]":
    out: "dict[str, float]" = {}
    centers = list(payload.peak_centers or [])
    sigmas = list(payload.peak_sigmas or [])
    amps = list(payload.peak_amplitudes or [])
    cerrs = list(getattr(payload, "peak_centers_err", []) or [])
    params = getattr(payload, "params", None)
    for i, c in enumerate(centers):
        out[f"center_{i}"] = float(c)
        if i < len(cerrs) and cerrs[i] is not None:
            out[f"center_err_{i}"] = float(cerrs[i])
        fwhm = None
        if params is not None:
            p = params.get(f"p{i}_fwhm")
            if p is not None:
                fwhm = float(p.value)
        if fwhm is None and i < len(sigmas):
            fwhm = 2.3548 * float(sigmas[i])  # Gaussian-equivalent fallback
        if fwhm is not None:
            out[f"fwhm_{i}"] = fwhm
        if i < len(amps):
            out[f"amplitude_{i}"] = float(amps[i])
    return out


def _peak_overlay(x: np.ndarray, y: np.ndarray, payload: Any) -> Overlay:
    best = np.asarray(payload.best_fit, dtype=float)
    traces: "dict[str, np.ndarray]" = {"fit": best, "residual": y - best}
    try:
        comps = payload.fit_result.eval_components(x=x)
        bg = sum(v for k, v in comps.items() if str(k).startswith("bg"))
        if np.ndim(bg) == 1:
            traces["background"] = np.asarray(bg, dtype=float)
    except Exception:
        pass
    return Overlay(
        x=np.asarray(x, dtype=float),
        traces=traces,
        markers=[float(c) for c in (payload.peak_centers or [])],
    )
