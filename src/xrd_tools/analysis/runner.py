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

from xrd_tools.analysis.plans import (
    AnalysisResult,
    PeakFitPlan,
    PhaseFitPlan,
    Sin2PsiPlan,
    run_peak_fit,
    run_phase_fit,
    run_sin2psi,
)
from xrd_tools.core.containers import IntegrationResult2D


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


def run_batch(analyzer, inputs, on_progress=None, should_cancel=None, on_frame=None):
    """Apply ``analyzer`` across a sequence of :class:`AnalysisInput`.

    Branches ONCE on ``analyzer.unit`` (the single analysis-specific decision):

    * ``"frame"`` — one outcome per input (e.g. peak/phase fitting), the
      per-frame series a vs-index plot is built from.
    * ``"scan"`` — the whole set is one unit (e.g. sin2psi/texture across
      tilts); returns a single-element list.

    Never raises for an analysis failure — a bad input yields an inert
    ``ok=False`` outcome so the batch always completes.  ``on_progress(done,
    total)`` is called after each input (and once at the end for scan-unit).
    ``on_frame(outcome)`` streams each outcome as it is computed (so a live
    vs-frame plot can grow during the run).  ``should_cancel()`` is polled
    before each frame (frame-unit only); if it returns true the loop stops and
    the outcomes gathered so far are returned."""
    inputs = list(inputs)
    total = len(inputs)
    if getattr(analyzer, "unit", "frame") == "scan":
        try:
            out = analyzer.analyze_scan(inputs)
        except Exception as exc:  # noqa: BLE001 — inert on failure
            out = AnalysisOutcome(label="scan", ok=False, message=str(exc))
        if on_frame is not None:
            on_frame(out)
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
        if on_frame is not None:
            on_frame(out)
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


# --------------------------------------------------------------------------
# sin2psi strain (scan-unit) — the first scan-unit analyzer.
# --------------------------------------------------------------------------


@dataclass
class Sin2PsiAnalyzer:
    """Scan-unit analyzer wrapping :class:`Sin2PsiPlan` + :func:`run_sin2psi`.

    sin²ψ strain analysis aggregates χ-sectors WITHIN one GI polar map (it fits
    the same Bragg peak in each tilt sector, then regresses d vs sin²ψ), so it is
    ``unit="scan"`` — one *set* of inputs collapses to one outcome.  The
    ``(q, χ)`` polar map is an :class:`IntegrationResult2D`; carry it on an
    :class:`AnalysisInput` via ``result_1d`` (or ``metadata['result_2d']``).
    ``analyze_scan`` reads the first input that carries one, so it works whether
    the runner hands it the whole batch or a single carrier input.

    Projects :class:`Sin2PsiResult` to flat strain params (``d0`` / ``slope`` /
    ``r_squared`` / optional ``stress``) + a sin²ψ-vs-d regression overlay
    (``data`` points + the fitted ``fit`` line).  Follows the same
    wrap-a-Plan-and-project shape as :class:`PeakFitAnalyzer`.
    """

    plan: Sin2PsiPlan
    kind: str = "sin2psi"
    unit: str = "scan"

    def analyze(self, inp: AnalysisInput) -> AnalysisOutcome:
        raise NotImplementedError(
            "Sin2PsiAnalyzer is scan-unit; the runner calls analyze_scan()."
        )

    def analyze_scan(self, inputs: Sequence[AnalysisInput]) -> AnalysisOutcome:
        inputs = list(inputs)
        label = inputs[0].label if inputs else "sin2psi"
        result2d = _extract_result2d(inputs)
        if result2d is None:
            return AnalysisOutcome(
                label, ok=False,
                message="sin2psi needs an IntegrationResult2D (q, χ) polar map on "
                        "an input's result_1d (or metadata['result_2d']).",
            )
        try:
            result = run_sin2psi(self.plan, result2d)
        except Exception as exc:  # backend/fit/regression failure -> inert
            return AnalysisOutcome(label, ok=False, message=str(exc))
        payload = result.payload
        n = np.asarray(getattr(payload, "d_values", []), dtype=float).size
        ok = (n >= 2 and np.isfinite(getattr(payload, "d0", np.nan))
              and np.isfinite(getattr(payload, "slope", np.nan)))
        return AnalysisOutcome(
            label=label,
            ok=bool(ok),
            params=_sin2psi_params(payload),
            overlay=_sin2psi_overlay(payload),
            result=result,
        )


def _extract_result2d(inputs: "Sequence[AnalysisInput]"):
    """The 2-D polar map from the first input that carries one, else None."""
    for inp in inputs:
        if isinstance(inp.result_1d, IntegrationResult2D):
            return inp.result_1d
        cand = (inp.metadata or {}).get("result_2d")
        if isinstance(cand, IntegrationResult2D):
            return cand
    return None


def _sin2psi_params(payload: Any) -> "dict[str, float]":
    out: "dict[str, float]" = {
        "d0": float(payload.d0),
        "d0_err": float(payload.d0_err),
        "slope": float(payload.slope),
        "slope_err": float(payload.slope_err),
        "r_squared": float(payload.r_squared),
        "n_sectors": float(len(payload.peak_fits or [])),
    }
    psi = np.asarray(payload.psi_deg, dtype=float)
    if psi.size:
        out["psi_min"] = float(np.min(psi))
        out["psi_max"] = float(np.max(psi))
    if getattr(payload, "stress", None) is not None:
        out["stress"] = float(payload.stress)
    if getattr(payload, "stress_err", None) is not None:
        out["stress_err"] = float(payload.stress_err)
    return out


def _sin2psi_overlay(payload: Any) -> Overlay:
    x = np.asarray(payload.sin2psi, dtype=float)
    d = np.asarray(payload.d_values, dtype=float)
    fit = np.asarray(payload.d0 + payload.slope * x, dtype=float)
    return Overlay(x=x, traces={"data": d, "fit": fit}, markers=[])


# --------------------------------------------------------------------------
# Phase fitting (frame-unit) — structure-informed multi-phase fit.
# --------------------------------------------------------------------------


@dataclass
class PhaseFitAnalyzer:
    """Frame-unit analyzer wrapping :class:`PhaseFitPlan` + :func:`run_phase_fit`.

    Unlike :class:`PeakFitAnalyzer`, phase fitting needs the crystallographic
    ``phases`` (``PhaseModel`` objects from CIFs) — the Plan's ``FitConfig``
    carries only phase NAMES — so they are injected at construction.  Projects
    the per-frame ``MultiPhaseResult`` to flat params (``frac_<phase>``,
    ``<phase>_a`` / ``_b`` / ``_c``, ``q_shift``, ``redchi``) — the vs-frame
    series a batch run plots — plus a fit / residual overlay
    (``fitter.eval_model``).
    """

    plan: PhaseFitPlan
    phases: "list[Any]" = field(default_factory=list)
    kind: str = "phase_fit"
    unit: str = "frame"

    def analyze(self, inp: AnalysisInput) -> AnalysisOutcome:
        if not self.phases:
            return AnalysisOutcome(
                inp.label, ok=False,
                message="No phases — add one or more CIF files.")
        x = np.asarray(inp.x, dtype=float)
        y = np.asarray(inp.y, dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        try:
            result = run_phase_fit(self._effective_plan(), [(x, y)],
                                   self.phases, labels=[inp.label])
        except Exception as exc:  # backend/fit failure -> inert outcome
            return AnalysisOutcome(inp.label, ok=False, message=str(exc))
        store = result.payload
        if len(store) == 0:
            return AnalysisOutcome(inp.label, ok=False,
                                   message="phase fit produced no result")
        entry = store[0]
        return AnalysisOutcome(
            label=inp.label,
            ok=bool(entry.get("success", False)),
            params=_phase_params(entry),
            overlay=_phase_overlay(x, y, entry),
            result=result,
        )

    def analyze_scan(self, inputs: Sequence[AnalysisInput]) -> AnalysisOutcome:
        raise NotImplementedError(
            "PhaseFitAnalyzer is frame-unit; the runner calls analyze() per frame."
        )

    def _effective_plan(self) -> PhaseFitPlan:
        """A plan whose FitConfig selects every injected phase by name (an empty
        phase_names selects NONE in fit_sequence — so fill it from self.phases)."""
        config = self.plan.config
        if getattr(config, "phase_names", None):
            return self.plan
        from xrd_tools.analysis.fitting import FitConfig
        names = [getattr(p, "name", f"phase_{i}")
                 for i, p in enumerate(self.phases)]
        filled = FitConfig(init_kw=dict(config.init_kw),
                           fit_kw=dict(config.fit_kw),
                           phase_names=names,
                           min_intensity=config.min_intensity,
                           name=config.name)
        return PhaseFitPlan(config=filled, sequential=self.plan.sequential)


def _phase_params(entry: "dict[str, Any]") -> "dict[str, float]":
    out: "dict[str, float]" = {}
    for name, frac in (entry.get("phase_fractions") or {}).items():
        out[f"frac_{name}"] = float(frac)
    for name, lat in (entry.get("lattice_params") or {}).items():
        for k, v in (lat or {}).items():
            out[f"{name}_{k}"] = float(v)
    out["q_shift"] = float(entry.get("q_shift", 0.0) or 0.0)
    redchi = entry.get("redchi")
    if redchi is not None and np.isfinite(redchi):
        out["redchi"] = float(redchi)
    return out


def _phase_overlay(x: np.ndarray, y: np.ndarray, entry: "dict[str, Any]") -> Overlay:
    x = np.asarray(x, dtype=float)
    result = entry.get("result")
    traces: "dict[str, np.ndarray]" = {}
    if result is not None:
        try:
            model = np.asarray(result.fitter.eval_model(result.params), dtype=float)
            if model.shape == x.shape:
                traces["fit"] = model
                traces["residual"] = np.asarray(y, dtype=float) - model
        except Exception:
            pass
    return Overlay(x=x, traces=traces, markers=[])
