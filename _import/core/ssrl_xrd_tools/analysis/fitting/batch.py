"""
Batch phase-fitting utilities: configuration, result storage, and sequential fitting.

The three main pieces:

* :class:`FitConfig` — serialisable snapshot of all PhaseFitter / fit() kwargs.
* :class:`FitResultStore` — accumulates :class:`MultiPhaseResult` objects from
  a batch run and can export to a pandas DataFrame.
* :func:`fit_sequence` — fits a list of patterns with optional sequential
  seeding (result *N* → starting guess for *N+1*).
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["FitConfig", "FitResultStore", "fit_sequence", "fit_nexus"]


# ---------------------------------------------------------------------------
# FitConfig
# ---------------------------------------------------------------------------

@dataclass
class FitConfig:
    """Serialisable snapshot of PhaseFitter init + fit keyword arguments.

    Two groups of settings mirror the two dicts returned by
    ``PhaseFitViewer._build_fitter_kwargs()``:

    * **init_kw** — passed to the ``PhaseFitter(...)`` constructor
      (prefit background, fit background, amorphous peak settings).
    * **fit_kw** — passed to ``PhaseFitter.fit(...)``
      (profile, Caglioti, texture, lattice bounds, width bounds, …).

    Additionally:

    * **phase_names** — which phases to include (by name).
    * **min_intensity** — minimum template intensity for peak filtering.
    * **name** — optional human-readable label for this config.

    The whole thing round-trips through JSON for easy save / load.
    """

    # PhaseFitter.__init__ kwargs
    init_kw: dict[str, Any] = field(default_factory=dict)

    # PhaseFitter.fit() kwargs
    fit_kw: dict[str, Any] = field(default_factory=dict)

    # Which phases to select (by name)
    phase_names: list[str] = field(default_factory=list)

    # Minimum template-intensity threshold for add_phase
    min_intensity: float = 5.0

    # Human-readable label
    name: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict (JSON-safe after tuple→list coercion)."""
        d = asdict(self)
        # Tuples become lists in JSON — normalise on save so round-trip
        # comparisons work.
        return json.loads(json.dumps(d))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FitConfig":
        """Reconstruct from a dict (e.g. loaded from JSON)."""
        d = dict(d)  # shallow copy
        # Re-tuple march_axis if present
        fit_kw = d.get("fit_kw", {})
        if "march_axis" in fit_kw and isinstance(fit_kw["march_axis"], list):
            fit_kw["march_axis"] = tuple(fit_kw["march_axis"])
        if "q_range" in fit_kw and isinstance(fit_kw["q_range"], list):
            fit_kw["q_range"] = tuple(fit_kw["q_range"])
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("FitConfig saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FitConfig":
        """Load config from a JSON file."""
        d = json.loads(Path(path).read_text())
        return cls.from_dict(d)

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        return (
            f"FitConfig{label}("
            f"phases={self.phase_names}, "
            f"profile={self.fit_kw.get('phase_profile', '?')}, "
            f"texture={self.fit_kw.get('texture', '?')}, "
            f"prefit={self.init_kw.get('prefit_background', 'none')})"
        )


# ---------------------------------------------------------------------------
# FitResultStore
# ---------------------------------------------------------------------------

class FitResultStore:
    """Accumulate :class:`MultiPhaseResult` objects from a batch run.

    Each entry stores the result, the pattern index/label, elapsed time,
    and optionally a snapshot of the :class:`~lmfit.Parameters` (for
    sequential seeding).

    The store can be exported to a pandas ``DataFrame`` for plotting
    phase fractions vs. sequence index or composition.
    """

    def __init__(self):
        self._entries: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._entries[idx]

    def __iter__(self):
        return iter(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def append(
        self,
        result: Any,  # MultiPhaseResult
        *,
        index: int = 0,
        label: str = "",
        elapsed: float = 0.0,
        params_snapshot: Any | None = None,
    ) -> None:
        """Record a single fit result.

        Parameters
        ----------
        result : MultiPhaseResult
            The fit result object.
        index : int
            Sequence index (e.g. pattern number).
        label : str
            Optional label (e.g. composition tag, scan ID).
        elapsed : float
            Wall-clock time for this fit in seconds.
        params_snapshot : lmfit.Parameters or None
            Deep copy of the fitted parameters, useful for seeding the
            next pattern in sequential mode.  If *None* the store takes
            a copy from ``result.params``.
        """
        if params_snapshot is None:
            params_snapshot = deepcopy(result.params)
        self._entries.append({
            "index": index,
            "label": label,
            "result": result,
            "elapsed": elapsed,
            "params": params_snapshot,
            "success": result.success,
            "redchi": result.redchi,
            "phase_fractions": result.phase_fractions(),
        })

    @property
    def results(self) -> list[Any]:
        """List of MultiPhaseResult objects."""
        return [e["result"] for e in self._entries]

    def to_dataframe(self):
        """Export to a pandas DataFrame.

        Columns: index, label, success, redchi, elapsed, plus one column
        per phase fraction (e.g. ``frac_Ortho``, ``frac_Mono``).

        Lattice parameters are included as ``{phase}_{param}`` columns
        (e.g. ``Ortho_a``, ``Mono_beta``).
        """
        import pandas as pd

        rows = []
        for e in self._entries:
            row = {
                "index": e["index"],
                "label": e["label"],
                "success": e["success"],
                "redchi": e["redchi"],
                "elapsed": e["elapsed"],
            }
            for phase_name, frac in e["phase_fractions"].items():
                row[f"frac_{phase_name}"] = frac
            # Lattice params per phase
            result = e["result"]
            for i, ph in enumerate(result.fitter.phases):
                for k, v in result.lattice_params(i).items():
                    row[f"{ph.name}_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """One-line-per-pattern summary."""
        lines = []
        for e in self._entries:
            tag = e["label"] or f"#{e['index']}"
            ok = "OK" if e["success"] else "STOP"
            fracs = "  ".join(
                f"{k}={v:.3f}" for k, v in e["phase_fractions"].items()
            )
            lines.append(
                f"[{tag}] {ok}  redχ²={e['redchi']:.3g}  "
                f"t={e['elapsed']:.1f}s  {fracs}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# fit_sequence — batch + optional sequential seeding
# ---------------------------------------------------------------------------

def fit_sequence(
    patterns: Sequence[tuple[np.ndarray, np.ndarray]
                        | tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    phases: list[Any],
    config: FitConfig,
    *,
    sequential: bool = False,
    labels: Sequence[str] | None = None,
    progress_callback=None,
    fit_background_template: np.ndarray
                             | tuple[np.ndarray, np.ndarray]
                             | None = None,
) -> FitResultStore:
    """Fit a sequence of patterns and return a :class:`FitResultStore`.

    Parameters
    ----------
    patterns : list of (q, y) or (q, y, sigma)
        Each element is one diffraction pattern.
    phases : list of PhaseModel
        The full set of phases.  ``config.phase_names`` selects which
        ones to include in the fit.
    config : FitConfig
        Shared init / fit kwargs (see :class:`FitConfig`).
    sequential : bool
        If *True*, the fitted ``Parameters`` from pattern *N* are used
        as the starting guess for pattern *N+1*.  This is useful for
        continuous composition series where neighbouring patterns are
        similar.
    labels : list of str or None
        Optional per-pattern labels (e.g. composition tags).
    progress_callback : callable or None
        ``progress_callback(i, n, result)`` called after each fit.
    fit_background_template : ndarray or (x_ref, y_ref), optional
        Reference spectrum for a scaled-template background (e.g. a
        substrate measurement).  If supplied, it's injected into every
        fitter's ``init_kw`` and ``fit_background`` defaults to
        ``"template"``.  Kept out of :class:`FitConfig` because ndarrays
        don't round-trip through JSON.

    Returns
    -------
    FitResultStore
    """
    import time
    from ssrl_xrd_tools.analysis.fitting.phase_fitting import PhaseFitter

    store = FitResultStore()
    n = len(patterns)
    if labels is None:
        labels = [str(i) for i in range(n)]

    # Select phases by name.  Zero phases are allowed if the config
    # includes an amorphous or in-fit background component (or a
    # template is supplied at call time).
    selected_phases = [
        p for p in phases if getattr(p, "name", None) in config.phase_names
    ]
    has_bg = bool(
        config.init_kw.get("fit_background")
        or config.init_kw.get("amorphous_peak")
        or fit_background_template is not None
    )
    if not selected_phases and not has_bg:
        raise ValueError(
            f"No phases matched config.phase_names={config.phase_names} "
            f"and no amorphous/fit-background/template component is set. "
            f"Available phases: {[getattr(p, 'name', '?') for p in phases]}"
        )

    prev_params = None

    for i, pat in enumerate(patterns):
        q, y = pat[0], pat[1]
        sigma = pat[2] if len(pat) > 2 else None

        # Build fitter
        init_kw = dict(config.init_kw)
        if fit_background_template is not None:
            init_kw.setdefault("fit_background", "template")
            init_kw["fit_background_template"] = fit_background_template
        if sigma is not None:
            fitter = PhaseFitter(q, y, sigma=sigma, **init_kw)
        else:
            fitter = PhaseFitter(q, y, **init_kw)

        for ph in selected_phases:
            fitter.add_phase(ph, min_intensity=config.min_intensity)

        # Build fit kwargs — optionally seed from previous result
        fit_kw = dict(config.fit_kw)
        if sequential and prev_params is not None:
            fit_kw["params"] = prev_params

        t0 = time.perf_counter()
        try:
            result = fitter.fit(**fit_kw)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            logger.warning("Pattern %s (%s) failed: %s", i, labels[i], exc)
            elapsed = time.perf_counter() - t0
            # Create a minimal failure entry — skip this pattern
            if progress_callback is not None:
                progress_callback(i, n, None)
            continue

        store.append(
            result,
            index=i,
            label=labels[i],
            elapsed=elapsed,
        )

        if sequential:
            prev_params = deepcopy(result.params)

        if progress_callback is not None:
            progress_callback(i, n, result)

    return store


# ---------------------------------------------------------------------------
# NeXus convenience entry point  (xdart 0.37+ schema)
# ---------------------------------------------------------------------------

def fit_nexus(
    path: str | Path,
    phases: list[Any],
    config: FitConfig,
    *,
    entry: str = "entry",
    sequential: bool = False,
    progress_callback=None,
    fit_background_template: np.ndarray
                             | tuple[np.ndarray, np.ndarray]
                             | None = None,
    label_motor: str | None = None,
) -> "FitResultStore":
    """Fit every 1-D pattern in an xdart NeXus file (v1 or v2 schema).

    Reads via :func:`ssrl_xrd_tools.io.nexus.read_sphere`, which
    auto-detects schema version — works on both pre-0.37 and 0.37+
    files.  Replaces per-frame ``h5py.File`` round-trips with a single
    load.
    """
    from ssrl_xrd_tools.io.nexus import read_sphere

    ds = read_sphere(path, entry=entry, groups=("1d",))

    if "intensity_1d" not in ds.data_vars:
        raise ValueError(
            f"{path}:{entry} has no /integrated_1d data — nothing to fit"
        )

    q = np.asarray(ds["q"].values, dtype=float)
    intensity = np.asarray(ds["intensity_1d"].values, dtype=float)
    sigma_arr = (
        np.asarray(ds["sigma_1d"].values, dtype=float)
        if "sigma_1d" in ds.data_vars
        else None
    )

    n = intensity.shape[0]
    patterns: list[tuple] = []
    for i in range(n):
        if sigma_arr is not None:
            patterns.append((q, intensity[i], sigma_arr[i]))
        else:
            patterns.append((q, intensity[i]))

    if label_motor is not None and label_motor in ds.data_vars:
        labels: list[str] | None = [str(v) for v in ds[label_motor].values]
    elif "frame" in ds.coords:
        labels = [str(v) for v in ds["frame"].values]
    else:
        labels = None

    return fit_sequence(
        patterns,
        phases,
        config,
        sequential=sequential,
        labels=labels,
        progress_callback=progress_callback,
        fit_background_template=fit_background_template,
    )
