"""
Multi-phase 1D XRD pattern fitting with crystal-structure-informed models.

Uses pymatgen for initial peak positions/intensities from CIF files, then
performs fast analytical lattice refinement during fitting (no repeated
pymatgen calls).  Background is handled via SNIP.  Peak profiles are
pseudo-Voigt with optional Caglioti (U, V, W) Q-dependent widths.

Example
-------
>>> from ssrl_xrd_tools.analysis.phase import PhaseModel
>>> from ssrl_xrd_tools.analysis.fitting.phase_fitting import PhaseFitter
>>>
>>> au = PhaseModel.from_cif("Au.cif")
>>> cu = PhaseModel.from_cif("Cu.cif")
>>>
>>> fitter = PhaseFitter(q, intensity)
>>> fitter.add_phase(au)
>>> fitter.add_phase(cu)
>>> result = fitter.fit()
>>> fitter.report(result)
>>> fitter.plot(result)
"""
from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from lmfit import Parameters, minimize

from ssrl_xrd_tools.analysis.fitting.background import snip_1d

logger = logging.getLogger(__name__)

__all__ = ["PhaseFitter", "MultiPhaseResult"]


# ---------------------------------------------------------------------------
# Analytical d-spacing helpers (avoid pymatgen in the inner loop)
# ---------------------------------------------------------------------------

def _metric_tensor(a: float, b: float, c: float,
                   alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Return the reciprocal metric tensor G* for a general triclinic lattice.

    Angles are in **degrees**.  The tensor is 3×3 and symmetric; G*_ij lets
    you compute  1/d²(hkl) = h_i G*_ij h_j  for Miller indices (h, k, l).
    """
    ar, br, gr = np.radians(alpha), np.radians(beta), np.radians(gamma)
    ca, cb, cg = np.cos(ar), np.cos(br), np.cos(gr)
    sa, sb, sg = np.sin(ar), np.sin(br), np.sin(gr)

    vol_sq_term = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg
    if vol_sq_term <= 0:
        raise ValueError(
            f"Invalid lattice angles (alpha={alpha}, beta={beta}, gamma={gamma}): "
            f"volume determinant is non-positive ({vol_sq_term:.6g})"
        )
    vol = a * b * c * np.sqrt(vol_sq_term)
    if vol < 1e-12:
        raise ValueError(
            f"Degenerate lattice: volume {vol:.6g} is too small "
            f"(a={a}, b={b}, c={c})"
        )

    # Reciprocal lattice parameters
    a_star = b * c * sa / vol
    b_star = a * c * sb / vol
    c_star = a * b * sg / vol

    cos_alpha_star = (cb * cg - ca) / (sb * sg)
    cos_beta_star  = (ca * cg - cb) / (sa * sg)
    cos_gamma_star = (ca * cb - cg) / (sa * sb)

    G = np.array([
        [a_star**2,
         a_star * b_star * cos_gamma_star,
         a_star * c_star * cos_beta_star],
        [a_star * b_star * cos_gamma_star,
         b_star**2,
         b_star * c_star * cos_alpha_star],
        [a_star * c_star * cos_beta_star,
         b_star * c_star * cos_alpha_star,
         c_star**2],
    ])
    return G


def _q_from_hkl(hkl: np.ndarray, G_star: np.ndarray) -> np.ndarray:
    """Compute q = 2π/d for an array of (N, 3) Miller indices.

    Parameters
    ----------
    hkl : (N, 3) int array
    G_star : (3, 3) reciprocal metric tensor

    Returns
    -------
    q : (N,) float array  —  values in Å⁻¹
    """
    # 1/d² = h^T G* h
    inv_d2 = np.einsum("ij,jk,ik->i", hkl.astype(float), G_star, hkl.astype(float))
    inv_d2 = np.clip(inv_d2, 1e-30, None)
    return 2.0 * np.pi * np.sqrt(inv_d2)


# ---------------------------------------------------------------------------
# Peak profile helpers
# ---------------------------------------------------------------------------

def _pseudo_voigt(x: np.ndarray, center: float, amplitude: float, #TODO - this should probably use the lmfit Pseudovoigt
                  sigma: float, fraction: float) -> np.ndarray:
    """Evaluate a pseudo-Voigt profile (area-normalised Gaussian + Lorentzian).

    This is the same convention as lmfit's PseudoVoigtModel:
        pV = (1 - η) G + η L
    where G and L have the *same* FWHM = 2σ.
    """
    # Gaussian component  (FWHM = 2σ  →  σ_gauss = σ / sqrt(2 ln2))
    sig_g = sigma / np.sqrt(2.0 * np.log(2.0)) if sigma > 0 else 1e-30
    gauss = (amplitude / (sig_g * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - center) / sig_g) ** 2
    )
    # Lorentzian component  (HWHM = σ)
    lorentz = (amplitude / np.pi) * (sigma / ((x - center) ** 2 + sigma**2))

    return (1.0 - fraction) * gauss + fraction * lorentz


def _caglioti_sigma(q: float | np.ndarray, U: float, V: float, W: float) -> float | np.ndarray:
    """Caglioti FWHM in Q-space.

    The classical Caglioti formula is defined in 2θ-space:
        FWHM²(2θ) = U tan²θ + V tanθ + W

    Here we use an analogous polynomial in Q:
        σ²(Q) = U·Q² + V·Q + W

    which gives a smooth, Q-dependent peak width without requiring an
    explicit 2θ↔Q conversion inside the fitting loop.  U, V, W have
    dimensions of Å⁻² , Å⁻¹ , and dimensionless, respectively.
    """
    sigma2 = U * q**2 + V * np.abs(q) + W
    return np.sqrt(np.clip(sigma2, 1e-10, None))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class MultiPhaseResult:
    """Thin wrapper around lmfit MinimizerResult with convenience accessors."""

    def __init__(self, lmfit_result, fitter: "PhaseFitter"):
        self.lmfit_result = lmfit_result
        self.params = lmfit_result.params
        self.fitter = fitter

    # ---- quick accessors --------------------------------------------------

    @property
    def q_shift(self) -> float:
        return self.params["q_shift"].value

    def phase_scale(self, idx: int) -> float:
        return self.params[f"p{idx}_scale"].value

    def phase_fractions(self) -> dict[str, float]:
        """Return normalised phase fractions (scale_i / Σ scales)."""
        scales = {
            ph.name: self.params[f"p{i}_scale"].value
            for i, ph in enumerate(self.fitter.phases)
        }
        total = sum(scales.values()) or 1.0
        return {k: v / total for k, v in scales.items()}

    def lattice_params(self, idx: int) -> dict[str, float]:
        """Return refined lattice parameters for phase *idx*."""
        pre = f"p{idx}_"
        return {
            k.replace(pre, ""): self.params[k].value
            for k in ("a", "b", "c", "alpha", "beta", "gamma")
            if f"{pre}{k}" in self.params
        }

    def width_params(self, idx: int) -> dict[str, float]:
        pre = f"p{idx}_"
        out = {}
        for k in ("U", "V", "W", "fraction"):
            key = f"{pre}{k}"
            if key in self.params:
                out[k] = self.params[key].value
        return out

    @property
    def redchi(self) -> float:
        return self.lmfit_result.redchi

    @property
    def success(self) -> bool:
        return self.lmfit_result.success

    def summary(self) -> str:
        lines = [
            f"Fit success: {self.success}",
            f"Reduced χ²: {self.redchi:.6g}",
            f"Q-shift: {self.q_shift:.6f} Å⁻¹",
            "",
        ]
        fracs = self.phase_fractions()
        for i, ph in enumerate(self.fitter.phases):
            lines.append(f"--- {ph.name} ---")
            lines.append(f"  scale      = {self.phase_scale(i):.4g}")
            lines.append(f"  fraction   = {fracs[ph.name]:.4f}")
            lp = self.lattice_params(i)
            if lp:
                lines.append(f"  a={lp.get('a',0):.5f}  b={lp.get('b',0):.5f}  "
                             f"c={lp.get('c',0):.5f}")
            wp = self.width_params(i)
            if wp:
                lines.append(f"  U={wp.get('U',0):.4g}  V={wp.get('V',0):.4g}  "
                             f"W={wp.get('W',0):.4g}  η={wp.get('fraction',0.5):.3f}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main fitter
# ---------------------------------------------------------------------------

class PhaseFitter: #TODO thes SNIP background should be optional
    """
    Multi-phase 1D XRD pattern fitter.

    Builds a composite model of N crystallographic phases, each contributing
    pseudo-Voigt peaks at positions determined analytically from (hkl, lattice)
    with template intensities from pymatgen.  A SNIP background is added, and
    a global Q-shift corrects for calibration offset.

    Parameters
    ----------
    x : array-like
        Q values (Å⁻¹).
    y : array-like
        Measured intensity.
    sigma : array-like or None
        Per-point uncertainties (used as weights = 1/sigma if provided).
    snip_width : int or None
        SNIP clipping width.  *None* → auto (5 % of data length).
    """

    def __init__(
        self,
        x: np.ndarray | Any,
        y: np.ndarray | Any,
        sigma: np.ndarray | None = None,
        snip_width: int | None = None,
    ):
        # Accept IntegrationResult1D transparently
        from ssrl_xrd_tools.core.containers import IntegrationResult1D
        if isinstance(x, IntegrationResult1D):
            res = x
            x, y = res.radial, res.intensity
            if sigma is None:
                sigma = res.sigma

        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float) if sigma is not None else None

        self.phases: list[Any] = []          # PhaseModel instances
        self._hkl_arrays: list[np.ndarray] = []    # (N_peaks, 3) per phase
        self._template_amps: list[np.ndarray] = []  # relative intensities
        self._init_lattice: list[dict] = []          # initial lattice params

        # Background
        self._snip_width = snip_width or max(int(len(self.x) * 0.05), 3)
        self.background = snip_1d(self.y, snip_width=self._snip_width)

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def add_phase(
        self,
        phase: Any,
        *,
        q_range: tuple[float, float] | None = None,
        min_intensity: float = 0.5,
    ) -> None:
        """Register a :class:`PhaseModel` for the fit.

        Parameters
        ----------
        phase : PhaseModel
            Must already have ``peaks`` populated (via ``calculate_peaks``).
        q_range : (q_min, q_max) or None
            Restrict to peaks within this Q range.  *None* → use the data
            range with a 10 % margin.
        min_intensity : float
            Drop peaks whose template intensity is below this fraction of
            the strongest peak in the phase (0–100 scale from pymatgen).
        """
        if not phase.peaks:
            raise ValueError(
                f"Phase '{phase.name}' has no peaks. "
                "Call phase.calculate_peaks() first."
            )

        if q_range is None:
            margin = 0.1 * (self.x.max() - self.x.min())
            q_range = (self.x.min() - margin, self.x.max() + margin)

        hkls, amps = [], []
        for pk in phase.peaks:
            if pk.q < q_range[0] or pk.q > q_range[1]:
                continue
            if pk.intensity < min_intensity:
                continue
            hkls.append(pk.hkl)
            amps.append(pk.intensity)

        if not hkls:
            logger.warning("Phase '%s': no peaks in the fitting range.", phase.name)

        hkl_arr = np.array(hkls, dtype=float) if hkls else np.empty((0, 3))
        amp_arr = np.array(amps, dtype=float) if amps else np.empty(0)

        # Normalise template intensities to max = 1
        if amp_arr.size and amp_arr.max() > 0:
            amp_arr = amp_arr / amp_arr.max()

        self.phases.append(phase)
        self._hkl_arrays.append(hkl_arr)
        self._template_amps.append(amp_arr)

        # Store initial lattice for bounds
        if phase.structure:
            lat = phase.structure.lattice
            self._init_lattice.append(dict(
                a=lat.a, b=lat.b, c=lat.c,
                alpha=lat.alpha, beta=lat.beta, gamma=lat.gamma,
            ))
        else:
            self._init_lattice.append({})

    # ------------------------------------------------------------------
    # Parameter construction
    # ------------------------------------------------------------------

    def build_parameters(
        self,
        q_shift_bound: float = 0.05,
        lattice_pct: float = 0.05,
        caglioti: bool = True,
    ) -> Parameters:
        """Build the lmfit :class:`Parameters` object.

        Parameters
        ----------
        q_shift_bound : float
            Max |Q-shift| allowed (Å⁻¹).
        lattice_pct : float
            Fractional tolerance on lattice constants (e.g. 0.05 → ±5 %).
        caglioti : bool
            If *True*, use U/V/W Caglioti broadening per phase; otherwise
            a single constant σ per phase.
        """
        params = Parameters()

        # Global calibration offset
        params.add("q_shift", value=0.0, min=-q_shift_bound, max=q_shift_bound)

        for i, phase in enumerate(self.phases):
            pre = f"p{i}_"
            lat0 = self._init_lattice[i]

            # Scale factor (overall intensity multiplier)
            params.add(f"{pre}scale", value=1.0, min=0.0)

            # Peak width parameters
            if caglioti:
                params.add(f"{pre}U", value=1e-4, min=0.0)
                params.add(f"{pre}V", value=0.0)
                params.add(f"{pre}W", value=4e-4, min=1e-6)
            else:
                params.add(f"{pre}sigma", value=0.02, min=1e-4, max=2.0)

            # Pseudo-Voigt mixing fraction
            params.add(f"{pre}fraction", value=0.5, min=0.0, max=1.0)

            # Lattice parameters (only if structure available)
            if lat0:
                for key in ("a", "b", "c"):
                    v0 = lat0[key]
                    params.add(
                        f"{pre}{key}", value=v0,
                        min=v0 * (1 - lattice_pct),
                        max=v0 * (1 + lattice_pct),
                    )

                # Symmetry constraints: tie equal axes
                if np.isclose(lat0["a"], lat0["b"], rtol=1e-4):
                    params[f"{pre}b"].expr = f"{pre}a"
                if np.isclose(lat0["a"], lat0["c"], rtol=1e-4):
                    params[f"{pre}c"].expr = f"{pre}a"
                elif np.isclose(lat0["b"], lat0["c"], rtol=1e-4):
                    params[f"{pre}c"].expr = f"{pre}b"

                # Angles fixed by default (no shear refinement)
                for key in ("alpha", "beta", "gamma"):
                    params.add(f"{pre}{key}", value=lat0[key], vary=False)

        return params

    # ------------------------------------------------------------------
    # Model evaluation
    # ------------------------------------------------------------------

    def _eval_phase(
        self,
        idx: int,
        x: np.ndarray,
        params: Parameters,
    ) -> np.ndarray:
        """Evaluate the contribution of phase *idx* at positions *x*."""
        pre = f"p{idx}_"
        scale = params[f"{pre}scale"].value
        fraction = params[f"{pre}fraction"].value

        hkl = self._hkl_arrays[idx]
        template_amp = self._template_amps[idx]

        if hkl.size == 0:
            return np.zeros_like(x)

        # Current lattice → reciprocal metric tensor → Q positions
        lat0 = self._init_lattice[idx]
        if lat0:
            a = params[f"{pre}a"].value
            b = params[f"{pre}b"].value
            c = params[f"{pre}c"].value
            alpha = params[f"{pre}alpha"].value
            beta = params[f"{pre}beta"].value
            gamma = params[f"{pre}gamma"].value
            G_star = _metric_tensor(a, b, c, alpha, beta, gamma)
            q_positions = _q_from_hkl(hkl, G_star)
        else:
            # Fall back to stored peak positions
            q_positions = np.array([pk.q for pk in self.phases[idx].peaks])

        # Width parameters
        has_caglioti = f"{pre}U" in params
        if has_caglioti:
            U = params[f"{pre}U"].value
            V = params[f"{pre}V"].value
            W = params[f"{pre}W"].value

        y_phase = np.zeros_like(x)
        for j, (q_c, amp_t) in enumerate(zip(q_positions, template_amp)):
            if has_caglioti:
                sig = _caglioti_sigma(q_c, U, V, W)
            else:
                sig = params[f"{pre}sigma"].value
            y_phase += _pseudo_voigt(x, q_c, amp_t * scale, float(sig), fraction)

        return y_phase

    def eval_model(self, params: Parameters) -> np.ndarray:
        """Evaluate the full model (all phases + background) at self.x."""
        q_shift = params["q_shift"].value
        x_shifted = self.x - q_shift

        y_model = self.background.copy()
        for i in range(len(self.phases)):
            y_model += self._eval_phase(i, x_shifted, params)
        return y_model

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _residual(self, params: Parameters) -> np.ndarray:
        y_calc = self.eval_model(params)
        resid = y_calc - self.y
        if self.sigma is not None:
            w = np.where(self.sigma > 0, 1.0 / self.sigma, 1.0)
            resid = resid * w
        return resid

    # ------------------------------------------------------------------
    # Fit driver
    # ------------------------------------------------------------------

    def fit(
        self,
        params: Parameters | None = None,
        method: str = "leastsq",
        caglioti: bool = True,
        q_shift_bound: float = 0.05,
        lattice_pct: float = 0.05,
        **minimize_kw: Any,
    ) -> MultiPhaseResult:
        """Run the multi-phase fit.

        Parameters
        ----------
        params : Parameters or None
            Pre-built parameters; if *None*, ``build_parameters`` is called.
        method : str
            lmfit minimisation method (default: Levenberg–Marquardt).
        caglioti : bool
            Use Caglioti U/V/W broadening (passed to ``build_parameters``).
        q_shift_bound, lattice_pct
            Forwarded to ``build_parameters`` when *params* is None.
        **minimize_kw
            Extra keyword arguments for :func:`lmfit.minimize`.

        Returns
        -------
        MultiPhaseResult
        """
        if not self.phases:
            raise ValueError("No phases added. Call add_phase() first.")

        if params is None:
            params = self.build_parameters(
                q_shift_bound=q_shift_bound,
                lattice_pct=lattice_pct,
                caglioti=caglioti,
            )

        result = minimize(self._residual, params, method=method, **minimize_kw)
        return MultiPhaseResult(result, self)

    # ------------------------------------------------------------------
    # Reporting / plotting
    # ------------------------------------------------------------------

    def report(self, result: MultiPhaseResult) -> str:
        """Print a human-readable fit summary."""
        txt = result.summary()
        print(txt)
        return txt

    def plot(
        self,
        result: MultiPhaseResult,
        *,
        show_components: bool = True,
        show_residual: bool = True,
        show_background: bool = True,
        ax: Any = None,
    ) -> Any:
        """Plot measured data, total fit, per-phase components, and residual.

        Returns the matplotlib Axes object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        params = result.params
        q_shift = params["q_shift"].value
        x_shifted = self.x - q_shift

        # Measured
        ax.plot(self.x, self.y, "k.", markersize=2, alpha=0.6, label="Data")

        # Total fit
        y_total = self.eval_model(params)
        ax.plot(self.x, y_total, "r-", linewidth=1.5, label="Total fit")

        # Background
        if show_background:
            ax.plot(self.x, self.background, "--", color="gray",
                    linewidth=1, label="SNIP background")

        # Per-phase components
        if show_components:
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.phases), 1)))
            for i, phase in enumerate(self.phases):
                y_ph = self._eval_phase(i, x_shifted, params)
                ax.plot(self.x, y_ph + self.background, "-",
                        color=colors[i], linewidth=1,
                        alpha=0.7, label=phase.name)

        # Residual
        if show_residual:
            residual = self.y - y_total
            offset = -0.1 * self.y.max()
            ax.plot(self.x, residual + offset, "g-", linewidth=0.8,
                    alpha=0.6, label="Residual (shifted)")
            ax.axhline(offset, color="g", linewidth=0.3, alpha=0.4)

        ax.set_xlabel(r"Q ($\AA^{-1}$)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title("Multi-phase XRD fit")
        ax.legend(fontsize=8)
        return ax
