"""
Synthetic 1D XRD pattern generator for ML training and fit validation.

The forward model is deliberately kept consistent with
:mod:`ssrl_xrd_tools.analysis.fitting.phase_fitting`: it uses the same
reciprocal metric tensor, the same March-Dollase texture correction, and
the same area-normalised pseudo-Voigt convention.  Anything trained on
these synthetic patterns will therefore live in the same parameter space
the real fitter optimises.

Three main entry points
-----------------------

* :func:`simulate_pattern` — single forward model call: given phases,
  fractions, widths, texture, substrate template, and noise settings,
  return an I(q) array.

* :func:`sample_prior` — draw one random parameter set (fractions,
  lattice jitter, widths, textures, substrate amplitude, q-shift, noise
  level) from physically motivated priors.

* :func:`generate_dataset` — combine the two into an (N_samples, N_q)
  training matrix X with a matched target matrix y of phase fractions
  (plus optional auxiliary targets).

Width models
------------

Two options, selected with ``width_model``:

* ``"scherrer"`` — Williamson–Hall style in q-space::

      FWHM²(q) = (2π·K / D)² + (2·ε·q)²

  Two parameters per phase:
      ``D``  — apparent crystallite size in Å
      ``eps`` — microstrain (fractional, dimensionless)

* ``"caglioti"`` — classical instrument-resolution form in q-space::

      σ²(q) = U·q² + V·q + W

  Three parameters per phase: ``U``, ``V``, ``W``.

Both produce per-peak pseudo-Voigt widths compatible with the fitter.

Substrate template
------------------

If ``substrate_template=(q_ref, I_ref)`` is supplied, it is interpolated
onto ``q_grid`` and added as ``amp · I_template(q)``.  Use this to bake
the actual fused-silica spectrum (or any other reference baseline) into
the training set so the ML model learns to factor it out rather than
hallucinate peaks from residual humps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ssrl_xrd_tools.analysis.fitting.phase_fitting import (
    _GAUSS_FWHM_FACTOR,
    _march_dollase,
    _metric_tensor,
    _q_from_hkl,
    _vector_pseudo_voigt,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SimulatorPriors",
    "simulate_pattern",
    "sample_prior",
    "generate_dataset",
]


# ---------------------------------------------------------------------------
# Width-model helpers
# ---------------------------------------------------------------------------

# Scherrer shape factor K (spherical crystallites → 0.94).
_SCHERRER_K = 0.94


def _fwhm_to_sigma(fwhm: np.ndarray | float) -> np.ndarray | float:
    """Convert FWHM → pseudo-Voigt sigma used by the fitter.

    The fitter's pseudo-Voigt (see :func:`_vector_pseudo_voigt`) treats
    ``sigma`` as the Lorentzian HWHM.  FWHM = 2·σ for that component,
    and the Gaussian side uses σ_g = σ / √(2 ln 2) so *its* FWHM also
    equals 2·σ.  One number, one consistent interpretation.
    """
    return 0.5 * np.asarray(fwhm, dtype=float)


def _scherrer_sigma(q: np.ndarray, D_angstrom: float, eps: float) -> np.ndarray:
    """Williamson–Hall FWHM → σ, for a single phase.

    ``D_angstrom`` is the apparent crystallite size in Å; ``eps`` is the
    microstrain (fractional, e.g. ``0.005`` for 0.5 %).
    """
    size_term = (2.0 * np.pi * _SCHERRER_K / max(float(D_angstrom), 1e-6)) ** 2
    strain_term = (2.0 * float(eps) * np.asarray(q, dtype=float)) ** 2
    fwhm = np.sqrt(size_term + strain_term)
    return _fwhm_to_sigma(fwhm)


def _caglioti_sigma(q: np.ndarray, U: float, V: float, W: float) -> np.ndarray:
    """σ(q) from the Caglioti-in-q parameterisation used by the fitter."""
    q = np.asarray(q, dtype=float)
    sigma2 = float(U) * q * q + float(V) * np.abs(q) + float(W)
    return np.sqrt(np.clip(sigma2, 1e-12, None))


# ---------------------------------------------------------------------------
# Phase utilities
# ---------------------------------------------------------------------------

def _phase_geometry(phase: Any) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Return (hkl, template_amp, lattice_params) for a PhaseModel.

    ``lattice_params`` is ``(a, b, c, alpha, beta, gamma)`` in pymatgen's
    convention.  ``template_amp`` is the peak-list intensity, normalised
    to unit maximum (same convention as the fitter).
    """
    if not getattr(phase, "peaks", None):
        raise ValueError(
            f"PhaseModel {getattr(phase, 'name', '?')!r} has no peaks "
            "populated.  Call .calculate_peaks() first."
        )

    hkls: list[tuple[int, int, int]] = []
    intensities: list[float] = []
    for pk in phase.peaks:
        hkl = pk.hkl
        if len(hkl) == 4:  # Miller–Bravais → Miller
            hkl = (hkl[0], hkl[1], hkl[3])
        hkls.append(tuple(int(v) for v in hkl))
        intensities.append(float(pk.intensity))
    hkl = np.asarray(hkls, dtype=float)
    intens = np.asarray(intensities, dtype=float)
    if intens.max() > 0:
        intens = intens / intens.max()

    lat = phase.structure.lattice
    lattice_params = (lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma)
    return hkl, intens, lattice_params


def _apply_lattice_jitter(
    phase: Any,
    lattice_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    angle_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, ...]:
    """Return jittered (a, b, c, α, β, γ) without mutating the phase."""
    _, _, (a, b, c, alpha, beta, gamma) = _phase_geometry(phase)
    sa, sb, sc = lattice_scale
    oa, ob, og = angle_offsets
    return (a * sa, b * sb, c * sc, alpha + oa, beta + ob, gamma + og)


# ---------------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------------

def simulate_pattern(
    q_grid: np.ndarray,
    phases: Sequence[Any],
    fractions: Sequence[float],
    *,
    # --- width model ---
    width_model: str = "scherrer",
    widths: Sequence[dict[str, float]] | None = None,
    # --- profile shape ---
    profile: str = "pseudovoigt",
    eta: float = 0.5,
    # --- lattice perturbations ---
    lattice_scales: Sequence[tuple[float, float, float]] | None = None,
    angle_offsets: Sequence[tuple[float, float, float]] | None = None,
    # --- texture ---
    march_r: Sequence[float] | None = None,
    march_axes: Sequence[tuple[int, int, int]] | None = None,
    # --- global shifts & scaling ---
    q_shift: float = 0.0,
    total_counts: float = 1e4,
    # --- substrate template ---
    substrate_template: tuple[np.ndarray, np.ndarray] | None = None,
    substrate_amplitude: float = 0.0,
    # --- noise ---
    noise: str | None = "poisson",
    background_floor: float = 0.0,
    rng: np.random.Generator | None = None,
    # --- diagnostic return ---
    return_components: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    """Generate one synthetic XRD pattern I(q).

    Parameters
    ----------
    q_grid : (M,) ndarray
        Output q-axis (Å⁻¹), strictly increasing.
    phases : list of PhaseModel
        Phase library.  Each phase must have ``.peaks`` populated.
    fractions : list of float
        One weight per phase.  Normalised internally so they sum to 1 —
        the resulting fractions are the ground-truth ML targets.
    width_model : {"scherrer", "caglioti"}
        Per-peak width parameterisation; see module docstring.
    widths : list of dict, optional
        One dict per phase:

        * ``"scherrer"`` → ``{"D": float (Å), "eps": float}``
        * ``"caglioti"`` → ``{"U": float, "V": float, "W": float}``

        If omitted, sensible defaults are used (D = 80 Å, ε = 0.003;
        U = V = 0, W = 0.001 Å⁻²).
    profile : str
        Peak profile name.  Currently only ``"pseudovoigt"`` is wired
        through; other choices fall back to pseudo-Voigt with a warning.
    eta : float
        Pseudo-Voigt mixing fraction (0 = pure Gaussian, 1 = pure
        Lorentzian).
    lattice_scales : list of (sa, sb, sc), optional
        Multiplicative jitter on a, b, c per phase.  ``(1, 1, 1)``
        leaves the lattice at its PhaseModel value.
    angle_offsets : list of (Δα, Δβ, Δγ), optional
        Additive angle jitter in degrees.
    march_r : list of float, optional
        March-Dollase parameter per phase.  ``1.0`` → random powder.
    march_axes : list of (h, k, l), optional
        Texture axes per phase.  Defaults to ``(0, 0, 1)``.
    q_shift : float
        Global q-axis shift applied to all peaks (Å⁻¹).
    total_counts : float
        Target maximum-of-pattern count level *before* noise.  The
        pattern is rescaled so its peak is at ``total_counts``.
    substrate_template : (q_ref, I_ref) tuple, optional
        Interpolated onto ``q_grid`` and added with amplitude
        ``substrate_amplitude * total_counts``.
    substrate_amplitude : float
        Amplitude of the substrate template, relative to
        ``total_counts``.  Typical range 0.3–1.5 for a substrate-
        dominated film.
    noise : {"poisson", None}
        Noise model.  ``"poisson"`` replaces the pattern with a Poisson
        draw; ``None`` returns the noise-free intensity.
    background_floor : float
        Constant added to the pattern before noise (in counts).  Keeps
        the Poisson variance strictly positive in dark regions.
    rng : numpy.random.Generator, optional
        RNG for reproducibility.  Defaults to ``np.random.default_rng()``.
    return_components : bool
        If *True*, return a dict with per-phase contributions, the
        substrate term, the clean pattern, and the noisy pattern.  If
        *False* (default), return the noisy ``I(q)`` array only.

    Returns
    -------
    ndarray or dict
        See ``return_components``.
    """
    if rng is None:
        rng = np.random.default_rng()

    q = np.asarray(q_grid, dtype=float)
    n_phases = len(phases)
    fractions = np.asarray(fractions, dtype=float)
    if fractions.shape != (n_phases,):
        raise ValueError(
            f"fractions must have length {n_phases}, got {fractions.shape}"
        )
    frac_sum = fractions.sum()
    if frac_sum <= 0:
        raise ValueError("fractions must sum to a positive value.")
    fractions = fractions / frac_sum

    widths = list(widths) if widths is not None else [None] * n_phases
    lattice_scales = (list(lattice_scales) if lattice_scales is not None
                      else [(1.0, 1.0, 1.0)] * n_phases)
    angle_offsets = (list(angle_offsets) if angle_offsets is not None
                     else [(0.0, 0.0, 0.0)] * n_phases)
    march_r = list(march_r) if march_r is not None else [1.0] * n_phases
    march_axes = (list(march_axes) if march_axes is not None
                  else [(0, 0, 1)] * n_phases)

    if profile.lower() not in ("pseudovoigt", "pvoigt", "pseudo"):
        logger.warning(
            "simulate_pattern: profile=%r not yet wired, using pseudovoigt.",
            profile,
        )

    # Evaluate in the q-shifted frame so peak centres match the fitter.
    x = q - float(q_shift)

    phase_components: list[np.ndarray] = []
    for i, phase in enumerate(phases):
        if fractions[i] <= 0:
            phase_components.append(np.zeros_like(q))
            continue

        hkl, template_amp, _ = _phase_geometry(phase)
        a, b, c, alpha, beta, gamma = _apply_lattice_jitter(
            phase,
            lattice_scale=lattice_scales[i],
            angle_offsets=angle_offsets[i],
        )
        G = _metric_tensor(a, b, c, alpha, beta, gamma)
        centers = _q_from_hkl(hkl, G)

        # Width per peak
        w = widths[i]
        if width_model == "scherrer":
            w = w or {"D": 80.0, "eps": 3e-3}
            sigmas = _scherrer_sigma(centers, w.get("D", 80.0), w.get("eps", 3e-3))
        elif width_model == "caglioti":
            w = w or {"U": 0.0, "V": 0.0, "W": 1e-3}
            sigmas = _caglioti_sigma(centers, w.get("U", 0.0),
                                     w.get("V", 0.0), w.get("W", 1e-3))
        else:
            raise ValueError(
                f"Unknown width_model={width_model!r}. Use 'scherrer' or 'caglioti'."
            )

        # March-Dollase texture correction
        r = float(march_r[i])
        if abs(r - 1.0) > 1e-9:
            axis = np.asarray(march_axes[i], dtype=float)
            md = _march_dollase(hkl, G, axis, r)
        else:
            md = np.ones(hkl.shape[0])

        # Restrict to peaks inside the grid (plus a small margin) for speed.
        q_lo, q_hi = q.min() - 0.05, q.max() + 0.05
        keep = (centers >= q_lo) & (centers <= q_hi)
        if not keep.any():
            phase_components.append(np.zeros_like(q))
            continue

        amps = template_amp[keep] * md[keep]
        # Unit-area per phase before weighting by fraction.
        amps = amps / max(amps.sum(), 1e-30)
        y_phase = _vector_pseudo_voigt(
            x, centers[keep], amps, sigmas[keep], float(eta),
        )
        phase_components.append(fractions[i] * y_phase)

    phase_total = np.sum(phase_components, axis=0) if phase_components else np.zeros_like(q)

    # Substrate template — interpolate + scale.
    substrate = np.zeros_like(q)
    if substrate_template is not None and substrate_amplitude > 0:
        q_ref, I_ref = substrate_template
        q_ref = np.asarray(q_ref, dtype=float)
        I_ref = np.asarray(I_ref, dtype=float)
        order = np.argsort(q_ref)
        substrate_unit = np.interp(q, q_ref[order], I_ref[order])
        # Normalise the template so substrate_amplitude is relative to
        # total_counts rather than absolute template counts.
        peak = substrate_unit.max()
        if peak > 0:
            substrate_unit = substrate_unit / peak
        substrate = substrate_amplitude * substrate_unit

    # Rescale phase_total so its peak equals 1 *before* mixing with the
    # substrate, then multiply everything by total_counts.  This gives a
    # pattern whose brightest single point is ~total_counts (substrate
    # may push it higher).
    peak_phase = phase_total.max()
    if peak_phase > 0:
        phase_total = phase_total / peak_phase
    clean = total_counts * (phase_total + substrate) + background_floor

    if noise == "poisson":
        noisy = rng.poisson(np.clip(clean, 0.0, None)).astype(float)
    elif noise in (None, "none", "off"):
        noisy = clean
    else:
        raise ValueError(f"Unknown noise={noise!r}. Use 'poisson' or None.")

    if return_components:
        return {
            "q": q,
            "I_clean": clean,
            "I_noisy": noisy,
            "phase_components": {
                getattr(phases[i], "name", f"phase_{i}"):
                    total_counts * phase_components[i] / max(peak_phase, 1e-30)
                for i in range(n_phases)
            },
            "substrate": total_counts * substrate,
            "fractions": fractions,
        }
    return noisy


# ---------------------------------------------------------------------------
# Priors + dataset generator
# ---------------------------------------------------------------------------

@dataclass
class SimulatorPriors:
    """Prior ranges for :func:`sample_prior` / :func:`generate_dataset`.

    All ranges are inclusive ``(low, high)`` uniform-draw tuples unless
    noted otherwise.
    """

    # Phase fractions come from a Dirichlet.  Default: uniform on the
    # simplex (all α = 1).  Set to smaller α to sparsify (more patterns
    # dominated by one phase).
    dirichlet_alpha: float | Sequence[float] = 1.0

    # Scherrer width priors
    D_range: tuple[float, float] = (40.0, 200.0)        # Å
    eps_range: tuple[float, float] = (1e-3, 8e-3)       # fractional

    # Caglioti width priors (used only if width_model='caglioti')
    U_range: tuple[float, float] = (0.0, 5e-4)
    V_range: tuple[float, float] = (-5e-4, 5e-4)
    W_range: tuple[float, float] = (5e-5, 5e-3)

    # Lattice jitter: per-phase multiplicative scale on (a, b, c)
    lattice_jitter_pct: float = 0.02

    # Texture: March-Dollase r per phase.  1.0 → no correction.
    march_r_range: tuple[float, float] = (0.6, 1.4)
    march_axis_choices: Sequence[tuple[int, int, int]] = field(
        default_factory=lambda: [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
    )

    # Global alignment
    q_shift_range: tuple[float, float] = (-0.02, 0.02)  # Å⁻¹

    # Substrate amplitude relative to total_counts
    substrate_amp_range: tuple[float, float] = (0.2, 1.2)
    # Probability of including the substrate at all (set <1 to train the
    # model on both bare and substrate-heavy regimes).
    substrate_prob: float = 1.0

    # Counts level (peak counts before noise)
    counts_range: tuple[float, float] = (3e3, 3e4)

    # Pseudo-Voigt mixing
    eta_range: tuple[float, float] = (0.2, 0.8)

    # Background floor (flat counts added before Poisson noise)
    bg_floor_range: tuple[float, float] = (0.0, 50.0)


def _loguniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_prior(
    phases: Sequence[Any],
    *,
    priors: SimulatorPriors | None = None,
    width_model: str = "scherrer",
    substrate_available: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Draw one random parameter set from ``priors``.

    The returned dict is ready to splat into :func:`simulate_pattern`
    (minus the required positional args and the substrate template
    itself).
    """
    if rng is None:
        rng = np.random.default_rng()
    if priors is None:
        priors = SimulatorPriors()

    n = len(phases)

    # Dirichlet fractions
    alpha = priors.dirichlet_alpha
    if np.isscalar(alpha):
        alpha = np.full(n, float(alpha))
    else:
        alpha = np.asarray(alpha, dtype=float)
        if alpha.size != n:
            raise ValueError(
                f"dirichlet_alpha length {alpha.size} != n_phases {n}"
            )
    fractions = rng.dirichlet(alpha)

    # Widths
    if width_model == "scherrer":
        widths = [
            {"D": _loguniform(rng, *priors.D_range),
             "eps": _loguniform(rng, *priors.eps_range)}
            for _ in range(n)
        ]
    elif width_model == "caglioti":
        widths = [
            {"U": rng.uniform(*priors.U_range),
             "V": rng.uniform(*priors.V_range),
             "W": _loguniform(rng, *priors.W_range)}
            for _ in range(n)
        ]
    else:
        raise ValueError(f"Unknown width_model={width_model!r}")

    # Lattice jitter (symmetric around 1)
    pct = priors.lattice_jitter_pct
    lattice_scales = [
        tuple(1.0 + rng.uniform(-pct, pct, size=3))
        for _ in range(n)
    ]
    angle_offsets = [(0.0, 0.0, 0.0)] * n  # keep angles fixed by default

    # Texture
    march_r = [rng.uniform(*priors.march_r_range) for _ in range(n)]
    axis_choices = list(priors.march_axis_choices)
    march_axes = [axis_choices[rng.integers(len(axis_choices))] for _ in range(n)]

    # Global shifts and scaling
    q_shift = float(rng.uniform(*priors.q_shift_range))
    total_counts = _loguniform(rng, *priors.counts_range)
    eta = float(rng.uniform(*priors.eta_range))
    bg_floor = float(rng.uniform(*priors.bg_floor_range))

    # Substrate
    if substrate_available and rng.random() < priors.substrate_prob:
        substrate_amplitude = float(rng.uniform(*priors.substrate_amp_range))
    else:
        substrate_amplitude = 0.0

    return dict(
        fractions=fractions,
        width_model=width_model,
        widths=widths,
        lattice_scales=lattice_scales,
        angle_offsets=angle_offsets,
        march_r=march_r,
        march_axes=march_axes,
        q_shift=q_shift,
        total_counts=total_counts,
        eta=eta,
        substrate_amplitude=substrate_amplitude,
        background_floor=bg_floor,
    )


def generate_dataset(
    q_grid: np.ndarray,
    phases: Sequence[Any],
    n_samples: int,
    *,
    priors: SimulatorPriors | None = None,
    width_model: str = "scherrer",
    substrate_template: tuple[np.ndarray, np.ndarray] | None = None,
    noise: str | None = "poisson",
    profile: str = "pseudovoigt",
    include_aux_targets: bool = False,
    seed: int | None = None,
    progress_callback=None,
) -> dict[str, np.ndarray]:
    """Generate a synthetic training set.

    Returns a dict with at minimum:

    * ``X`` — shape ``(N, len(q_grid))``, float32 intensities.
    * ``y_fractions`` — shape ``(N, n_phases)``, float32 phase fractions
      summing to 1 along axis 1.
    * ``q`` — shape ``(len(q_grid),)``, the shared q-axis.
    * ``phase_names`` — list of str, column order for ``y_fractions``.

    When ``include_aux_targets=True``, adds:

    * ``y_D`` — shape ``(N, n_phases)``, apparent crystallite size per
      phase in Å (only for ``width_model='scherrer'``).
    * ``y_eps`` — shape ``(N, n_phases)``, microstrain per phase.
    * ``y_march_r`` — shape ``(N, n_phases)``.
    * ``y_q_shift`` — shape ``(N,)``.
    * ``y_substrate_amp`` — shape ``(N,)``.

    Parameters
    ----------
    q_grid : (M,) ndarray
        Shared q-axis.
    phases : list of PhaseModel
    n_samples : int
        Number of patterns to generate.
    priors : SimulatorPriors, optional
        Prior ranges.  Defaults to :class:`SimulatorPriors()`.
    width_model : {"scherrer", "caglioti"}
    substrate_template : (q_ref, I_ref) tuple, optional
        Reference substrate spectrum.  If ``None``, substrate amplitude
        is forced to 0 regardless of ``priors.substrate_prob``.
    noise : {"poisson", None}
    profile : str
    include_aux_targets : bool
        See above.
    seed : int, optional
        Seed for the RNG.
    progress_callback : callable or None
        ``progress_callback(i, n)`` after each sample.
    """
    rng = np.random.default_rng(seed)
    if priors is None:
        priors = SimulatorPriors()

    q = np.asarray(q_grid, dtype=float)
    M = q.size
    n_phases = len(phases)

    X = np.empty((n_samples, M), dtype=np.float32)
    y_fractions = np.empty((n_samples, n_phases), dtype=np.float32)

    aux: dict[str, np.ndarray] = {}
    if include_aux_targets:
        if width_model == "scherrer":
            aux["y_D"] = np.empty((n_samples, n_phases), dtype=np.float32)
            aux["y_eps"] = np.empty((n_samples, n_phases), dtype=np.float32)
        aux["y_march_r"] = np.empty((n_samples, n_phases), dtype=np.float32)
        aux["y_q_shift"] = np.empty(n_samples, dtype=np.float32)
        aux["y_substrate_amp"] = np.empty(n_samples, dtype=np.float32)
        aux["y_total_counts"] = np.empty(n_samples, dtype=np.float32)

    for i in range(n_samples):
        params = sample_prior(
            phases,
            priors=priors,
            width_model=width_model,
            substrate_available=(substrate_template is not None),
            rng=rng,
        )
        pattern = simulate_pattern(
            q,
            phases,
            params["fractions"],
            width_model=width_model,
            widths=params["widths"],
            profile=profile,
            eta=params["eta"],
            lattice_scales=params["lattice_scales"],
            angle_offsets=params["angle_offsets"],
            march_r=params["march_r"],
            march_axes=params["march_axes"],
            q_shift=params["q_shift"],
            total_counts=params["total_counts"],
            substrate_template=substrate_template,
            substrate_amplitude=params["substrate_amplitude"],
            background_floor=params["background_floor"],
            noise=noise,
            rng=rng,
        )
        X[i] = pattern.astype(np.float32)
        y_fractions[i] = params["fractions"].astype(np.float32)

        if include_aux_targets:
            if width_model == "scherrer":
                aux["y_D"][i] = [w["D"] for w in params["widths"]]
                aux["y_eps"][i] = [w["eps"] for w in params["widths"]]
            aux["y_march_r"][i] = params["march_r"]
            aux["y_q_shift"][i] = params["q_shift"]
            aux["y_substrate_amp"][i] = params["substrate_amplitude"]
            aux["y_total_counts"][i] = params["total_counts"]

        if progress_callback is not None:
            progress_callback(i + 1, n_samples)

    phase_names = [getattr(p, "name", f"phase_{i}") for i, p in enumerate(phases)]
    out = {
        "X": X,
        "y_fractions": y_fractions,
        "q": q.astype(np.float32),
        "phase_names": phase_names,
    }
    out.update(aux)
    return out
