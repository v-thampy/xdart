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
# Batch driver (analysis-agnostic) — the headless core the GUI batch worker
# wraps in a thread, and the same call notebooks/pipelines use directly.
# --------------------------------------------------------------------------


def run_batch(analyzer, inputs, on_progress=None, should_cancel=None):
    """Apply ``analyzer`` across a sequence of :class:`AnalysisInput`.

    Branches ONCE on ``analyzer.unit`` (the single analysis-specific decision):

    * ``"frame"`` — one outcome per input (e.g. peak/phase fitting), the
      per-frame series a vs-index plot is built from.
    * ``"scan"`` — the whole set is one unit (e.g. sin2psi/texture across
      tilts); returns a single-element list.

    Never raises for an analysis failure — a bad input yields an inert
    ``ok=False`` outcome so the batch always completes.  ``on_progress(done,
    total)`` is called after each input (and once at the end for scan-unit).
    ``should_cancel()`` is polled before each frame (frame-unit only); if it
    returns true the loop stops and the outcomes gathered so far are returned."""
    inputs = list(inputs)
    total = len(inputs)
    if getattr(analyzer, "unit", "frame") == "scan":
        try:
            out = analyzer.analyze_scan(inputs)
        except Exception as exc:  # noqa: BLE001 — inert on failure
            out = AnalysisOutcome(label="scan", ok=False, message=str(exc))
        if on_progress is not None:
            on_progress(total, total)
        return [out]
    outcomes = []
    for i, inp in enumerate(inputs):
        if should_cancel is not None and should_cancel():
            break
        try:
            out = analyzer.analyze(inp)
        except Exception as exc:  # noqa: BLE001 — one bad frame can't abort the batch
            out = AnalysisOutcome(label=inp.label, ok=False, message=str(exc))
        outcomes.append(out)
        if on_progress is not None:
            on_progress(i + 1, total)
    return outcomes


def batch_params_table(outcomes):
    """Flatten a list of :class:`AnalysisOutcome` to ``(labels, columns)``.

    ``labels`` is one label per outcome (the frame index for frame-unit work);
    ``columns`` is an order-preserving ``{param_name: [value per outcome]}`` —
    the vs-index series for plotting or CSV export.  A param absent from a given
    outcome (e.g. a frame where that peak wasn't fit) is ``nan``, so every column
    is the same length and the series stay aligned with ``labels``."""
    labels = [o.label for o in outcomes]
    keys: "list[str]" = []
    seen: "set[str]" = set()
    for o in outcomes:
        for k in o.params:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    columns = {k: [float(o.params.get(k, np.nan)) for o in outcomes] for k in keys}
    return labels, columns


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
