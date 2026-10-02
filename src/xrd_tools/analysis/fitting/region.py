"""Joint peak fitting of a 2-D or 3-D region (headless, Qt-free, lmfit-free).

FIT-TOOLS-2026-10 H4 (the 2-D extension) and H5 (the 3-D fitter), one numerical owner for both:

    I(r) = b0 + b . (r - r_ref) + sum_j A_j g(r2_j),    r2_j = (r - c_j)^T Sigma_j^-1 (r - c_j)

over the region's valid samples, fitted jointly (never as separate 1-D fits). ``A_j`` is the peak HEIGHT, ``c_j``
the centre and ``Sigma_j = L_j L_j^T`` a full symmetric positive-definite shape matrix through a Cholesky factor
with positive diagonal bounds (no unconstrained covariance entries, no Euler angles). Profiles ``g``:

* ``gaussian``: ``exp(-r2/2)``; ``Sigma`` is the covariance; FWHM = 2 sqrt(2 ln 2) sqrt(lambda) (2-D and 3-D);
* ``lorentzian2``: ``(1 + r2)^-2``; FWHM = 2 sqrt(sqrt(2) - 1) sqrt(lambda) (2-D);
* ``pseudovoigt``: common-HWHM mixture ``(1-eta) exp(-ln2 r2) + eta (1 + (sqrt2-1) r2)^-2``; FWHM = 2 sqrt(lambda)
  for every ``eta`` (2-D), the same convention as the corrected ``PseudoVoigt2DModel``.

The background is ``none``, ``constant`` or ``linear`` (a plane in 2-D, affine in 3-D) about the region's centre.

Conventions. Data are in AXIS order: ``data[i, j(, k)]`` sits at ``(axes[0][i], axes[1][j](, axes[2][k]))``; a
displayed 2-D image is the transpose (rows = the second axis). Coordinates are the native bin centres, uniform or
not, ascending or not; internally each axis is mapped onto [-1, 1] over the region for conditioning and every
reported value is mapped back. Only finite, unmasked samples are fitted (``mask`` True = excluded); an uncovered
bin is never a measured zero. Fitting is unweighted by default; ``sigma`` (admitted uncertainties) selects a
weighted fit over the joint finite-positive-sigma mask. Uncertainties come from ``s^2 (J^T J)^-1`` (scaled by the
reduced chi-square, lmfit's default) and the delta method; they are unavailable (None), never zero, when a
parameter sits on a bound or the normal matrix is singular.

Bounds: centres inside the region, positive heights, each Cholesky diagonal (the conditional width of its axis)
at least ``WIDTH_FLOOR_BINS`` of that axis's median bin spacing. Caps, checked before any mesh, box copy or
Jacobian exists: ``MAX_POINTS`` region points, ``MAX_COMPONENTS`` components, ``MAX_PARAMETERS`` free parameters.
A fit needs at least twice as many usable samples as free parameters. A ``cancel_token`` is checked at every
model and Jacobian evaluation, so an active solve stops (``RegionFitCancelled``) rather than draining.
Components are the declared seeds (stable IDs, kept in declared order); without seeds one component starts at
the data maximum. Automatic multi-peak discovery is deliberately absent.

Reported meaning. 2-D: principal FWHMs and the major axis's orientation in the PLOTTED coordinate plane (native
units; not a sample tilt), unidentifiable for a circular peak. 3-D: native centre, covariance entries, marginal
FWHMs ``2 sqrt(2 ln 2) sqrt(Sigma_ii)`` and thin central-slice FWHMs, height and the continuous model integral
``A (2 pi)^(3/2) sqrt(det Sigma)`` (coordinate-volume units, not counts); physical principal FWHMs and directions
only through a reciprocal transform ``J`` (native -> Cartesian Q, ``Sigma_Q = J Sigma J^T``, the RSM UB
authority's 2 pi convention), unavailable without one; near-degenerate directions are marked unidentifiable.
Strain, tilt and relaxation are a separate reference-dependent calculation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import threading

import numpy as np

__all__ = [
    "Estimate", "MAX_COMPONENTS", "MAX_PARAMETERS", "MAX_POINTS", "PeakSeed", "PhysicalComponent", "RegionAxis",
    "RegionBackground", "RegionComponent", "RegionDiagnostics", "RegionFitCancelled", "RegionFitOutcome",
    "RegionFitRefused", "RegionSlice", "fit_2d_region", "fit_3d_region", "fit_volume_region",
]

LN2 = math.log(2.0)
_KAPPA = math.sqrt(2.0) - 1.0
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * LN2)

MAX_POINTS = 100_000
MAX_COMPONENTS = 3
MAX_PARAMETERS = 34
#: Lower bound on each Cholesky diagonal, in units of that axis's median bin spacing.
WIDTH_FLOOR_BINS = 0.25
#: Upper bound on Cholesky entries, in normalized units (the region spans [-1, 1]).
_SHAPE_BOUND = 4.0
#: Principal directions whose eigenvalues differ by less than this fraction are unidentifiable.
IDENTIFIABLE_RELATIVE_GAP = 1e-2
#: Jacobians up to this many elements (8 MiB) use the exact trust-region solver, larger ones LSMR.
_EXACT_SOLVER_ELEMENTS = 1 << 20

_PROFILES = {
    "gaussian": "gaussian", "gauss": "gaussian",
    "lorentzian2": "lorentzian2", "lor2": "lorentzian2", "lorentzian_squared": "lorentzian2",
    "pseudovoigt": "pseudovoigt", "pvoigt": "pseudovoigt",
}
#: profile(r_half^2) = 1/2 on the half-height contour.
_HALF_HEIGHT_RADIUS = {"gaussian": math.sqrt(2.0 * LN2), "lorentzian2": math.sqrt(_KAPPA), "pseudovoigt": 1.0}
_BACKGROUNDS = {"none": "none", "constant": "constant", "linear": "linear", "plane": "linear", "affine": "linear"}


class RegionFitRefused(ValueError):
    """A named refusal (``code``): REGION_TOO_LARGE, TOO_MANY_COMPONENTS, TOO_MANY_PARAMETERS, REGION_TOO_THIN,
    REGION_EMPTY, INSUFFICIENT_SAMPLES, WEIGHTS_UNUSABLE or SEED_OUTSIDE_REGION."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class RegionFitCancelled(RuntimeError):
    """The caller's cancel token was set; the solve stopped at its next model evaluation."""


@dataclass(frozen=True, slots=True)
class RegionAxis:
    """One named native axis of bin centres (in data order)."""

    name: str
    values: np.ndarray
    unit: str | None = None

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        if type(self.name) is not str or not self.name:
            raise ValueError("a region axis needs a name")
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"axis {self.name!r} needs at least two finite bin centres")
        if not values.max() > values.min():
            raise ValueError(f"axis {self.name!r} has no extent")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PeakSeed:
    """A declared component: stable ``id``, starting centre (native, axis order), optional FWHMs and height."""

    id: str
    center: tuple[float, ...]
    fwhm: tuple[float, ...] | None = None
    height: float | None = None


@dataclass(frozen=True, slots=True)
class Estimate:
    """A fitted or derived value; ``stderr`` None means unavailable (never zero for unknown)."""

    value: float
    stderr: float | None = None


@dataclass(frozen=True, slots=True)
class PhysicalComponent:
    """A 3-D component in Cartesian Q (Angstrom^-1) through the supplied reciprocal transform."""

    center_q: tuple[Estimate, ...]
    covariance_q: np.ndarray
    principal_fwhm: tuple[Estimate, ...]  # descending
    principal_directions: tuple[np.ndarray | None, ...]  # unit vectors in Q; None when unidentifiable
    identifiable: tuple[bool, ...]
    integral: Estimate  # intensity x Angstrom^-3


@dataclass(frozen=True, slots=True)
class RegionComponent:
    id: str
    profile: str
    height: Estimate
    center: tuple[Estimate, ...]
    covariance: np.ndarray  # native Sigma (shape matrix), axis order
    covariance_stderr: np.ndarray | None
    central_fwhm: tuple[Estimate, ...]  # along each axis through the centre
    marginal_fwhm: tuple[Estimate, ...] | None  # Gaussian only
    integral: Estimate  # intensity x native coordinate volume
    principal_fwhm: tuple[Estimate, ...] | None = None  # 2-D, plotted plane, descending
    rotation_deg: Estimate | None = None  # 2-D major axis from +x toward +y, (-90, 90]
    rotation_identifiable: bool | None = None
    fraction: Estimate | None = None  # pseudo-Voigt
    physical: PhysicalComponent | None = None


@dataclass(frozen=True, slots=True)
class RegionBackground:
    kind: str
    reference: tuple[float, ...]  # r_ref: the region centre (native)
    offset: Estimate  # b0 at r_ref
    slopes: tuple[Estimate, ...]  # per native unit; empty unless linear


@dataclass(frozen=True, slots=True)
class RegionDiagnostics:
    success: bool
    message: str
    nfev: int
    n_used: int
    n_parameters: int
    dof: int
    weighted: bool
    ssr: float  # sum of squared (unweighted) residuals
    rms: float
    normalized_rms: float  # rms over the fitted samples' range
    chisqr: float
    redchi: float
    uncertainty: str  # "available" or why not


@dataclass(frozen=True, slots=True)
class RegionSlice:
    """Data, model and signed linear residual on one plane, with its named axes (data order)."""

    axes: tuple[RegionAxis, RegionAxis]
    fixed: tuple[str, int, float] | None
    data: np.ndarray
    model: np.ndarray
    residual: np.ndarray
    valid: np.ndarray

    def displayed(self) -> RegionSlice:
        """The same plane as an image: rows are the second axis, columns the first (the transpose)."""
        return RegionSlice((self.axes[1], self.axes[0]), self.fixed, self.data.T, self.model.T,
                           self.residual.T, self.valid.T)


class _Layout:
    def __init__(self, ndim: int, n_components: int, profile: str, background: str) -> None:
        self.ndim = ndim
        self.profile = profile
        self.background = background
        self.n_background = {"none": 0, "constant": 1, "linear": 1 + ndim}[background]
        self.tril = tuple((a, b) for a in range(ndim) for b in range(a + 1))
        self.per_component = 1 + ndim + len(self.tril) + (profile == "pseudovoigt")
        self.n_components = n_components
        self.size = self.n_background + n_components * self.per_component

    def offset(self, j: int) -> int:
        return self.n_background + j * self.per_component

    def unpack(self, p: np.ndarray, j: int):
        o = self.offset(j)
        n = self.ndim
        lower = np.zeros((n, n))
        for k, (a, b) in enumerate(self.tril):
            lower[a, b] = p[o + 1 + n + k]
        eta = p[o + 1 + n + len(self.tril)] if self.profile == "pseudovoigt" else None
        return p[o], p[o + 1:o + 1 + n], lower, eta


def _profile(kind: str, r2: np.ndarray, eta):
    """``g(r2)``, ``dg/dr2`` and (pseudo-Voigt) ``dg/deta``."""
    if kind == "gaussian":
        g = np.exp(-0.5 * r2)
        return g, -0.5 * g, None
    if kind == "lorentzian2":
        q = 1.0 / (1.0 + r2)
        return q * q, -2.0 * q ** 3, None
    e = np.exp(-LN2 * r2)
    q = 1.0 / (1.0 + _KAPPA * r2)
    lor = q * q
    return (1.0 - eta) * e + eta * lor, -(1.0 - eta) * LN2 * e - 2.0 * eta * _KAPPA * q ** 3, lor - e


def _profile_integral(kind: str, ndim: int, eta) -> float:
    """``integral of g(|z|^2) d^n z`` (so a component integrates to ``A sqrt(det Sigma)`` times this)."""
    if kind == "gaussian":
        return (2.0 * math.pi) ** (ndim / 2.0)
    if kind == "lorentzian2":
        return math.pi
    return (1.0 - eta) * math.pi / LN2 + eta * math.pi / _KAPPA


def _component_terms(layout: _Layout, p: np.ndarray, j: int, t: np.ndarray):
    amplitude, centre, lower, eta = layout.unpack(p, j)
    inverse = np.linalg.inv(lower)
    d = t - centre
    z = d @ inverse.T  # z = L^-1 d
    r2 = np.einsum("ij,ij->i", z, z)
    return amplitude, eta, inverse, z, r2


def _model(layout: _Layout, p: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = t.shape[0]
    out = np.full(n, p[0] if layout.n_background else 0.0)
    if layout.background == "linear":
        out += t @ p[1:1 + layout.ndim]
    for j in range(layout.n_components):
        amplitude, eta, _inverse, _z, r2 = _component_terms(layout, p, j, t)
        out += amplitude * _profile(layout.profile, r2, eta)[0]
    return out


def _jacobian(layout: _Layout, p: np.ndarray, t: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """The model's Jacobian; every column is written, so ``out`` may be a reused buffer."""
    n, ndim = t.shape
    jac = np.empty((n, layout.size)) if out is None else out
    if layout.n_background:
        jac[:, 0] = 1.0
    if layout.background == "linear":
        jac[:, 1:1 + ndim] = t
    for j in range(layout.n_components):
        o = layout.offset(j)
        amplitude, eta, inverse, z, r2 = _component_terms(layout, p, j, t)
        g, dg, deta = _profile(layout.profile, r2, eta)
        u = z @ inverse  # u = Sigma^-1 d
        scale = -2.0 * amplitude * dg  # d model / d r2 = A dg; d r2 / d c = -2 u; d r2 / d L_ab = -2 u_a z_b
        jac[:, o] = g
        jac[:, o + 1:o + 1 + ndim] = scale[:, None] * u
        for k, (a, b) in enumerate(layout.tril):
            jac[:, o + 1 + ndim + k] = scale * u[:, a] * z[:, b]
        if deta is not None:
            jac[:, o + 1 + ndim + len(layout.tril)] = amplitude * deta
    return jac


def _products_only(matrix: np.ndarray):
    """A LinearOperator doing only products with ``matrix`` (real): no adjoint copy is ever made."""
    from scipy.sparse.linalg import LinearOperator

    class _Products(LinearOperator):
        def __init__(self, a):
            super().__init__(a.dtype, a.shape)
            self.a = a

        def _matvec(self, x):
            return self.a @ x

        def _matmat(self, x):
            return self.a @ x

        def _rmatvec(self, x):
            return self.a.T @ x

        def _rmatmat(self, x):
            return self.a.T @ x

    return _Products(matrix)


def _sym_entries(ndim: int):
    return tuple((a, b) for a in range(ndim) for b in range(a, ndim))


class _Report:
    """Every reported quantity as a function of the normalized parameters (for values and the delta method)."""

    def __init__(self, layout: _Layout, mid: np.ndarray, half: np.ndarray, transform: np.ndarray | None):
        self.layout, self.mid, self.half, self.transform = layout, mid, half, transform
        n = layout.ndim
        names: list[tuple] = []
        if layout.n_background:
            names.append(("bg", "offset"))
        if layout.background == "linear":
            names += [("bg", "slope", i) for i in range(n)]
        for j in range(layout.n_components):
            names += [(j, "height")] + [(j, "center", i) for i in range(n)]
            names += [(j, "cov", a, b) for a, b in _sym_entries(n)]
            names += [(j, "integral")] + [(j, "central", i) for i in range(n)]
            if layout.profile == "gaussian":
                names += [(j, "marginal", i) for i in range(n)]
            if n == 2:
                names += [(j, "eig", 0), (j, "eig", 1), (j, "angle")]
            if layout.profile == "pseudovoigt":
                names.append((j, "fraction"))
            if transform is not None:
                names += [(j, "qcenter", i) for i in range(3)] + [(j, "qcov", a, b) for a, b in _sym_entries(3)]
                names += [(j, "qeig", i) for i in range(3)] + [(j, "qintegral")]
        self.names = tuple(names)
        self.index = {name: k for k, name in enumerate(self.names)}
        self.angles = np.array([name[-1] == "angle" for name in self.names])

    def native(self, p: np.ndarray, j: int):
        amplitude, centre, lower, eta = self.layout.unpack(p, j)
        sigma = (lower @ lower.T) * np.outer(self.half, self.half)
        return amplitude, self.mid + self.half * centre, sigma, eta

    def values(self, p: np.ndarray) -> np.ndarray:
        layout, n = self.layout, self.layout.ndim
        out: list[float] = []
        if layout.n_background:
            out.append(p[0])
        if layout.background == "linear":
            out += list(p[1:1 + n] / self.half)
        r_half = _HALF_HEIGHT_RADIUS[layout.profile]
        for j in range(layout.n_components):
            amplitude, centre, sigma, eta = self.native(p, j)
            precision = np.linalg.inv(sigma)
            integral = amplitude * math.sqrt(max(np.linalg.det(sigma), 0.0)) * _profile_integral(
                layout.profile, n, eta)
            out += [amplitude, *centre, *(sigma[a, b] for a, b in _sym_entries(n)), integral]
            out += list(2.0 * r_half / np.sqrt(np.diag(precision)))
            if layout.profile == "gaussian":
                out += list(FWHM_PER_SIGMA * np.sqrt(np.diag(sigma)))
            if n == 2:
                eigenvalues, vectors = np.linalg.eigh(sigma)
                major = vectors[:, 1]
                out += [eigenvalues[1], eigenvalues[0], _wrap_angle(math.degrees(math.atan2(major[1], major[0])))]
            if layout.profile == "pseudovoigt":
                out.append(eta)
            if self.transform is not None:
                sigma_q = self.transform @ sigma @ self.transform.T
                out += list(self.transform @ centre) + [sigma_q[a, b] for a, b in _sym_entries(3)]
                out += list(np.linalg.eigvalsh(sigma_q)[::-1])
                out.append(integral * abs(np.linalg.det(self.transform)))
        return np.asarray(out, dtype=np.float64)

    def covariance(self, p: np.ndarray, parameter_covariance: np.ndarray) -> np.ndarray:
        """Delta-method covariance of :meth:`values` (central differences; angle differences wrapped)."""
        gradient = np.empty((len(self.names), p.size))
        for k in range(p.size):
            step = 1e-6 * max(abs(p[k]), 1e-3)
            plus, minus = p.copy(), p.copy()
            plus[k] += step
            minus[k] -= step
            difference = self.values(plus) - self.values(minus)
            difference[self.angles] = [_wrap_angle(item) for item in difference[self.angles]]
            gradient[:, k] = difference / (2.0 * step)
        return gradient @ parameter_covariance @ gradient.T


def _wrap_angle(degrees: float) -> float:
    """Map an axis orientation onto (-90, 90]."""
    wrapped = (degrees + 90.0) % 180.0 - 90.0
    return 90.0 if wrapped == -90.0 else wrapped


def _separated(values: np.ndarray, i: int, j: int, covariance: np.ndarray | None, index: tuple[int, int]) -> bool:
    gap = abs(float(values[i]) - float(values[j]))
    if gap < IDENTIFIABLE_RELATIVE_GAP * max(abs(float(values[i])), abs(float(values[j]))):
        return False
    if covariance is None:
        return True
    a, b = index
    variance = covariance[a, a] + covariance[b, b] - 2.0 * covariance[a, b]
    return bool(gap >= 2.0 * math.sqrt(max(float(variance), 0.0)))


def _normalized_seed(seed: PeakSeed, axes: Sequence[RegionAxis], mid, half) -> np.ndarray:
    if type(seed) is not PeakSeed or type(seed.id) is not str or not seed.id:
        raise ValueError("seeds must be PeakSeed objects with a non-empty id")
    centre = np.asarray(seed.center, dtype=np.float64)
    if centre.shape != (len(axes),) or not np.all(np.isfinite(centre)):
        raise ValueError(f"seed {seed.id!r} needs one finite centre value per axis")
    if seed.fwhm is not None:
        fwhm = np.asarray(seed.fwhm, dtype=np.float64)
        if fwhm.shape != (len(axes),) or not np.all(np.isfinite(fwhm)) or np.any(fwhm <= 0):
            raise ValueError(f"seed {seed.id!r} needs one finite positive FWHM per axis")
    if seed.height is not None and not (np.isfinite(seed.height) and seed.height > 0):
        raise ValueError(f"seed {seed.id!r} needs a finite positive height")
    for value, axis in zip(centre, axes):
        if not axis.values.min() <= value <= axis.values.max():
            raise RegionFitRefused("SEED_OUTSIDE_REGION",
                                   f"seed {seed.id!r} lies outside the region on axis {axis.name!r}")
    return (centre - mid) / half


def _half_width(line: np.ndarray, values: np.ndarray, start: int, level: float) -> float | None:
    """Distance from ``start`` to where ``line`` falls to ``level`` (linear interpolation), averaged over sides."""
    if not np.isfinite(line[start]) or line[start] <= level:
        return None
    sides = []
    for step in (-1, 1):
        q = start
        while 0 <= q + step < line.size and np.isfinite(line[q + step]) and line[q + step] > level:
            q += step
        nxt = q + step
        distance = abs(values[q] - values[start])
        if 0 <= nxt < line.size and np.isfinite(line[nxt]):
            distance += (line[q] - level) / (line[q] - line[nxt]) * abs(values[nxt] - values[q])
            sides.append(distance)
        elif distance > 0:
            sides.append(distance)
    return float(np.mean(sides)) if sides else None


def _fit_region(
    data,
    axes: Sequence[RegionAxis],
    *,
    seeds: Sequence[PeakSeed],
    profile: str,
    background: str,
    mask,
    sigma,
    transform: np.ndarray | None,
    physical_reason: str | None,
    cancel_token: threading.Event | None,
) -> RegionFitOutcome:
    from scipy.optimize import least_squares

    profile = _PROFILES.get(str(profile).lower().replace("-", "").replace(" ", ""))
    if profile is None:
        raise ValueError(f"profile must be one of {sorted(set(_PROFILES.values()))}")
    background = _BACKGROUNDS.get(str(background).lower())
    if background is None:
        raise ValueError(f"background must be one of {sorted(_BACKGROUNDS)}")
    ndim = len(axes)
    shape = tuple(axis.values.size for axis in axes)
    seeds = tuple(seeds or ())
    # caps first: from shapes and counts only, before any mesh, copy or Jacobian
    points = math.prod(shape)
    if points > MAX_POINTS:
        raise RegionFitRefused("REGION_TOO_LARGE", f"the region holds {points:,} points; at most "
                               f"{MAX_POINTS:,} are fitted: narrow the box")
    n_components = max(1, len(seeds))
    if n_components > MAX_COMPONENTS:
        raise RegionFitRefused("TOO_MANY_COMPONENTS", f"{n_components} components; at most {MAX_COMPONENTS}")
    layout = _Layout(ndim, n_components, profile, background)
    if layout.size > MAX_PARAMETERS:
        raise RegionFitRefused("TOO_MANY_PARAMETERS", f"{layout.size} free parameters; at most {MAX_PARAMETERS}")
    if len({seed.id for seed in seeds}) != len(seeds):
        raise ValueError("seed ids must be unique")

    data = np.asarray(data)
    if data.shape != shape:
        raise ValueError(f"data shape {data.shape} must be the axes' lengths {shape} (axis order)")
    valid = np.isfinite(data)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"mask shape {mask.shape} must be {shape}")
        valid &= ~mask
    lows = np.array([axis.values.min() for axis in axes])
    highs = np.array([axis.values.max() for axis in axes])
    mid, half = (highs + lows) / 2.0, (highs - lows) / 2.0
    normalized_seeds = [_normalized_seed(seed, axes, mid, half) for seed in seeds]
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise RegionFitRefused("REGION_EMPTY", "the region holds no finite, unmasked sample")
    required = 2 * layout.size
    if n_valid < required:
        raise RegionFitRefused("INSUFFICIENT_SAMPLES", f"{n_valid} usable samples; at least {required} are "
                               f"needed for {layout.size} parameters")
    weights = None
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        if sigma.shape != shape:
            raise ValueError(f"sigma shape {sigma.shape} must be {shape}")
        valid &= np.isfinite(sigma) & (sigma > 0)
        n_valid = int(valid.sum())
        if n_valid < required:
            raise RegionFitRefused("WEIGHTS_UNUSABLE", f"{n_valid} samples carry a finite positive sigma; at "
                                   f"least {required} are needed")
        weights = 1.0 / sigma[valid]

    normalized_axes = [(axis.values - m) / h for axis, m, h in zip(axes, mid, half)]
    index = np.nonzero(valid)
    t = np.stack([values[i] for values, i in zip(normalized_axes, index)], axis=1)
    y = np.asarray(data[valid], dtype=np.float64)
    masked = np.where(valid, data, np.nan)

    # starting values and bounds
    lower_bound = np.full(layout.size, -np.inf)
    upper_bound = np.full(layout.size, np.inf)
    x0 = np.zeros(layout.size)
    base = float(np.percentile(y, 5)) if layout.n_background else 0.0
    if layout.n_background:
        x0[0] = base
    floors = np.array([WIDTH_FLOOR_BINS * np.median(np.abs(np.diff(axis.values))) for axis in axes]) / half
    r_half = _HALF_HEIGHT_RADIUS[profile]
    if not seeds:
        peak = np.unravel_index(int(np.nanargmax(masked)), shape)
        seeds = (PeakSeed("p0", tuple(axis.values[i] for axis, i in zip(axes, peak))),)
        normalized_seeds = [_normalized_seed(seeds[0], axes, mid, half)]
    for j, (seed, centre) in enumerate(zip(seeds, normalized_seeds)):
        o = layout.offset(j)
        nearest = tuple(int(np.argmin(np.abs(axis.values - value))) for axis, value in zip(axes, seed.center))
        local = masked[nearest]
        height = seed.height if seed.height is not None else (
            local - base if np.isfinite(local) and local > base else float(y.max()) - base)
        height = max(float(height), 1e-12 * max(float(np.abs(y).max()), 1.0))
        widths = np.empty(ndim)
        for i, axis in enumerate(axes):
            if seed.fwhm is not None:
                widths[i] = float(seed.fwhm[i]) / (2.0 * r_half) / half[i]
                continue
            line = masked[nearest[:i] + (slice(None),) + nearest[i + 1:]]
            hw = _half_width(line, axis.values, nearest[i], base + height / 2.0)
            spacing = float(np.median(np.abs(np.diff(axis.values))))
            widths[i] = (hw if hw else 2.0 * spacing) / r_half / half[i]
        x0[o] = height
        lower_bound[o] = 0.0
        x0[o + 1:o + 1 + ndim] = centre
        lower_bound[o + 1:o + 1 + ndim], upper_bound[o + 1:o + 1 + ndim] = -1.0, 1.0
        for k, (a, b) in enumerate(layout.tril):
            position = o + 1 + ndim + k
            if a == b:
                x0[position] = widths[a]
                lower_bound[position], upper_bound[position] = floors[a], _SHAPE_BOUND
            else:
                lower_bound[position], upper_bound[position] = -_SHAPE_BOUND, _SHAPE_BOUND
        if profile == "pseudovoigt":
            position = o + 1 + ndim + len(layout.tril)
            x0[position], lower_bound[position], upper_bound[position] = 0.5, 0.0, 1.0
    finite_bounds = np.isfinite(lower_bound) & np.isfinite(upper_bound)
    margin = np.where(finite_bounds, 1e-9 * (upper_bound - lower_bound), 0.0)
    x0 = np.where(finite_bounds, np.clip(x0, lower_bound + margin, upper_bound - margin), x0)
    x0 = np.where(np.isfinite(lower_bound) & ~finite_bounds, np.maximum(x0, lower_bound), x0)

    def residuals(p):
        if cancel_token is not None and cancel_token.is_set():
            raise RegionFitCancelled("region fit cancelled")
        out = _model(layout, p, t) - y
        return out if weights is None else out * weights

    def dense_jacobian(p):
        if cancel_token is not None and cancel_token.is_set():
            raise RegionFitCancelled("region fit cancelled")
        jac = _jacobian(layout, p, t)
        return jac if weights is None else jac * weights[:, None]

    final_jacobian = dense_jacobian
    if t.shape[0] * layout.size <= _EXACT_SOLVER_ELEMENTS:
        options = dict(jac=dense_jacobian, x_scale="jac", tr_solver="exact")
    else:
        # A large Jacobian (the 128 MiB workspace target, measured): SciPy's least_squares copies every array
        # Jacobian it hands the solver, keeps its first one for the whole solve, and the exact solver adds
        # scaled, augmented and SVD copies; aslinearoperator caches a conjugate (copied) adjoint. Instead one
        # buffer is refilled in place behind a products-only operator for LSMR: the solver never uses a previous
        # Jacobian once it asks for the next. The start point's column norms stand in for x_scale="jac".
        buffer = np.empty((t.shape[0], layout.size))

        def fill(p):
            if cancel_token is not None and cancel_token.is_set():
                raise RegionFitCancelled("region fit cancelled")
            _jacobian(layout, p, t, out=buffer)
            if weights is not None:
                np.multiply(buffer, weights[:, None], out=buffer)
            return buffer

        norms = np.sqrt(np.einsum("ij,ij->j", fill(x0), buffer))
        norms[norms == 0] = 1.0
        operator = _products_only(buffer)
        options = dict(jac=lambda p: (fill(p), operator)[1], x_scale=1.0 / norms, tr_solver="lsmr")
        final_jacobian = fill
    solution = least_squares(residuals, x0, bounds=(lower_bound, upper_bound), method="trf",
                             ftol=1e-10, xtol=1e-10, gtol=1e-10, **options)
    p = solution.x
    success = bool(solution.status > 0)
    message, nfev, at_bound = str(solution.message), int(solution.nfev), bool(np.any(solution.active_mask != 0))
    chisqr = float(2.0 * solution.cost)
    del solution, options  # release the solver's last Jacobian before the covariance's own
    dof = n_valid - layout.size
    redchi = chisqr / dof
    raw = _model(layout, p, t) - y
    ssr = float(raw @ raw)
    rms = math.sqrt(ssr / n_valid)
    spread = float(y.max() - y.min())
    del raw, masked
    jac = final_jacobian(p) if success and not at_bound else None
    del t, y

    report = _Report(layout, mid, half, transform)
    reported = report.values(p)
    reported_covariance = None
    if not success:
        uncertainty = "the solve did not converge"
    elif at_bound:
        uncertainty = "a parameter is at its bound"
    else:
        normal = jac.T @ jac
        scale = np.sqrt(np.diag(normal))
        eigenvalues = np.zeros(1)
        if np.all(scale > 0):  # judge singularity scale-free, on the column-normalized normal matrix
            eigenvalues, vectors = np.linalg.eigh(normal / np.outer(scale, scale))
        if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
            uncertainty = "the normal matrix is singular (an unidentifiable parameter)"
        else:
            inverse = (vectors / eigenvalues) @ vectors.T / np.outer(scale, scale)
            reported_covariance = report.covariance(p, redchi * inverse)
            uncertainty = "available"
    del jac

    def estimate(name) -> Estimate:
        k = report.index[name]
        stderr = None
        if reported_covariance is not None:  # G C G^T is positive semi-definite up to rounding
            stderr = float(math.sqrt(max(float(reported_covariance[k, k]), 0.0)))
        return Estimate(float(reported[k]), stderr)

    components = []
    for j, seed in enumerate(seeds):
        sigma_native = np.empty((ndim, ndim))
        sigma_err = None if reported_covariance is None else np.empty((ndim, ndim))
        for a, b in _sym_entries(ndim):
            entry = estimate((j, "cov", a, b))
            sigma_native[a, b] = sigma_native[b, a] = entry.value
            if sigma_err is not None:
                sigma_err[a, b] = sigma_err[b, a] = entry.stderr
        sigma_native.flags.writeable = False
        if sigma_err is not None:
            sigma_err.flags.writeable = False
        principal = rotation = identifiable = physical = None
        if ndim == 2:
            eigen = np.array([reported[report.index[(j, "eig", i)]] for i in range(2)])
            identifiable = _separated(eigen, 0, 1, reported_covariance,
                                      (report.index[(j, "eig", 0)], report.index[(j, "eig", 1)]))
            principal = tuple(Estimate(2.0 * r_half * math.sqrt(max(value, 0.0)), None if item.stderr is None
                                       else r_half * item.stderr / math.sqrt(value))
                              for value, item in ((eigen[i], estimate((j, "eig", i))) for i in range(2)))
            rotation = estimate((j, "angle")) if identifiable else None
        if transform is not None:
            physical = _physical(report, reported, reported_covariance, estimate, j)
        components.append(RegionComponent(
            id=seed.id, profile=profile, height=estimate((j, "height")),
            center=tuple(estimate((j, "center", i)) for i in range(ndim)),
            covariance=sigma_native, covariance_stderr=sigma_err,
            central_fwhm=tuple(estimate((j, "central", i)) for i in range(ndim)),
            marginal_fwhm=(tuple(estimate((j, "marginal", i)) for i in range(ndim))
                           if profile == "gaussian" else None),
            integral=estimate((j, "integral")), principal_fwhm=principal, rotation_deg=rotation,
            rotation_identifiable=identifiable,
            fraction=estimate((j, "fraction")) if profile == "pseudovoigt" else None, physical=physical,
        ))
    background_outcome = RegionBackground(
        background, tuple(float(value) for value in mid),
        estimate(("bg", "offset")) if layout.n_background else Estimate(0.0, None),
        tuple(estimate(("bg", "slope", i)) for i in range(ndim)) if background == "linear" else (),
    )
    diagnostics = RegionDiagnostics(
        success=success, message=message, nfev=nfev, n_used=n_valid,
        n_parameters=layout.size, dof=dof, weighted=weights is not None, ssr=ssr, rms=rms,
        normalized_rms=rms / spread if spread > 0 else math.nan, chisqr=chisqr, redchi=redchi,
        uncertainty=uncertainty,
    )
    valid.flags.writeable = False
    parameters = p.copy()
    parameters.flags.writeable = False
    return RegionFitOutcome(tuple(axes), profile, tuple(components), background_outcome, diagnostics, valid,
                            physical_reason if transform is None else None, layout, parameters, mid, half)


def _physical(report: _Report, reported, reported_covariance, estimate, j: int) -> PhysicalComponent:
    sigma_q = np.empty((3, 3))
    for a, b in _sym_entries(3):
        sigma_q[a, b] = sigma_q[b, a] = reported[report.index[(j, "qcov", a, b)]]
    eigenvalues, vectors = np.linalg.eigh(sigma_q)
    eigenvalues, vectors = eigenvalues[::-1], vectors[:, ::-1]
    names = [report.index[(j, "qeig", i)] for i in range(3)]
    flags = tuple(all(_separated(eigenvalues, i, k, reported_covariance, (names[i], names[k]))
                      for k in range(3) if k != i) for i in range(3))
    directions = []
    for i in range(3):
        if not flags[i]:
            directions.append(None)
            continue
        direction = vectors[:, i] * (1.0 if vectors[np.argmax(np.abs(vectors[:, i])), i] >= 0 else -1.0)
        direction.flags.writeable = False
        directions.append(direction)
    widths = []
    for i in range(3):
        item = estimate((j, "qeig", i))
        value = max(eigenvalues[i], 0.0)
        widths.append(Estimate(FWHM_PER_SIGMA * math.sqrt(value), None if item.stderr is None or value == 0
                               else FWHM_PER_SIGMA * item.stderr / (2.0 * math.sqrt(value))))
    sigma_q.flags.writeable = False
    return PhysicalComponent(tuple(estimate((j, "qcenter", i)) for i in range(3)), sigma_q, tuple(widths),
                             tuple(directions), flags, estimate((j, "qintegral")))


@dataclass(frozen=True, slots=True)
class RegionFitOutcome:
    """One detached region fit: named axes in data order, the samples used, components, background, diagnostics.

    It keeps compact parameters, the boolean validity mask and the axes (never a data or model cube); models and
    comparison planes are evaluated on demand.
    """

    axes: tuple[RegionAxis, ...]
    profile: str
    components: tuple[RegionComponent, ...]
    background: RegionBackground
    diagnostics: RegionDiagnostics
    valid: np.ndarray  # True = sample fitted (finite, unmasked, usable weight), data order
    physical_reason: str | None  # why physical (Q) values are unavailable, else None
    _layout: _Layout
    _parameters: np.ndarray
    _mid: np.ndarray
    _half: np.ndarray

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(axis.values.size for axis in self.axes)

    @property
    def data_order(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)

    def evaluate(self, *coordinates) -> np.ndarray:
        """The fitted model at native coordinates (one broadcastable array per axis, axis order)."""
        if len(coordinates) != self.ndim:
            raise ValueError(f"evaluate needs {self.ndim} coordinate arrays")
        arrays = np.broadcast_arrays(*(np.asarray(c, dtype=np.float64) for c in coordinates))
        t = np.stack([(a.ravel() - m) / h for a, m, h in zip(arrays, self._mid, self._half)], axis=1)
        return _model(self._layout, self._parameters, t).reshape(arrays[0].shape)

    def model(self) -> np.ndarray:
        """The fitted model on the region's full grid (data order), invalid samples included."""
        return self.evaluate(*np.meshgrid(*(axis.values for axis in self.axes), indexing="ij"))

    def _plane(self, data: np.ndarray, axis: str | None, index: int | None) -> RegionSlice:
        data = np.asarray(data)
        if data.shape != self.shape:
            raise ValueError(f"comparison data must have the region's shape {self.shape}")
        if axis is None:
            plane_axes, fixed, values, valid = self.axes, None, data, self.valid
            coordinates = np.meshgrid(*(a.values for a in self.axes), indexing="ij")
        else:
            k = self.data_order.index(axis)
            if not 0 <= index < self.shape[k]:
                raise ValueError(f"index {index} is outside axis {axis!r}")
            plane_axes = tuple(a for i, a in enumerate(self.axes) if i != k)
            fixed = (axis, int(index), float(self.axes[k].values[index]))
            values = np.take(data, index, axis=k)
            valid = np.take(self.valid, index, axis=k)
            grids = iter(np.meshgrid(*(a.values for a in plane_axes), indexing="ij"))
            coordinates = [np.full(values.shape, fixed[2]) if i == k else next(grids) for i in range(self.ndim)]
        model = self.evaluate(*coordinates)
        residual = np.where(valid, values - model, np.nan)
        return RegionSlice(tuple(plane_axes), fixed, values, model, residual, valid)

    def comparison(self, data: np.ndarray) -> RegionSlice:
        """A 2-D fit's data, model and signed residual over the whole plane (data order)."""
        if self.ndim != 2:
            raise ValueError("comparison() is for 2-D fits; use comparison_slice() for a volume")
        return self._plane(data, None, None)

    def comparison_slice(self, data: np.ndarray, axis: str, index: int) -> RegionSlice:
        """One plane of a 3-D fit at ``index`` of ``axis``: data, model, residual on the same plane and axes."""
        if self.ndim != 3:
            raise ValueError("comparison_slice() is for 3-D fits")
        return self._plane(data, axis, int(index))

    def _flat_items(self):
        """``(key, Estimate, unit)`` for every reported value: ``<id>.<quantity>[_<axis>]``, ``bg.*`` and the
        diagnostics. Units: "I" is the data's intensity unit, native axis units as the axes carry them ("" when
        unknown or dimensionless), Q values in Angstrom^-1."""
        names = self.data_order
        units = {axis.name: axis.unit or "" for axis in self.axes}
        plane = units[names[0]] if len(set(units.values())) == 1 else ""
        volume = "·".join(units[name] for name in names if units[name])
        for c in self.components:
            yield f"{c.id}.height", c.height, "I"
            for name, item in zip(names, c.center):
                yield f"{c.id}.center_{name}", item, units[name]
            for a, b in _sym_entries(self.ndim):
                stderr = None if c.covariance_stderr is None else float(c.covariance_stderr[a, b])
                yield (f"{c.id}.cov_{names[a]}_{names[b]}", Estimate(float(c.covariance[a, b]), stderr),
                       "·".join(item for item in (units[names[a]], units[names[b]]) if item))
            for name, item in zip(names, c.central_fwhm):
                yield f"{c.id}.slice_fwhm_{name}", item, units[name]
            for name, item in zip(names, c.marginal_fwhm or ()):
                yield f"{c.id}.fwhm_{name}", item, units[name]
            yield f"{c.id}.integral", c.integral, "I·" + volume if volume else "I"
            if c.principal_fwhm is not None:
                yield f"{c.id}.fwhm_major", c.principal_fwhm[0], plane
                yield f"{c.id}.fwhm_minor", c.principal_fwhm[1], plane
            yield f"{c.id}.angle_deg", c.rotation_deg, "deg"
            yield f"{c.id}.fraction", c.fraction, ""
            if c.physical is not None:
                for name, item in zip("xyz", c.physical.center_q):
                    yield f"{c.id}.q_center_{name}", item, "A^-1"
                for i, item in enumerate(c.physical.principal_fwhm, start=1):
                    yield f"{c.id}.q_fwhm_{i}", item, "A^-1"
                yield f"{c.id}.q_integral", c.physical.integral, "I·A^-3"
        if self.background.kind != "none":
            yield "bg.offset", self.background.offset, "I"
        for name, item in zip(names, self.background.slopes):
            yield f"bg.slope_{name}", item, f"I/{units[name]}" if units[name] else "I"
        d = self.diagnostics
        for key, value, unit in (("rms", d.rms, "I"), ("normalized_rms", d.normalized_rms, ""),
                                 ("redchi", d.redchi, ""), ("n_used", float(d.n_used), ""),
                                 ("success", float(d.success), "")):
            yield key, Estimate(value), unit

    def flat_params(self) -> dict[str, float]:
        """Flat named values for tables, batches and CSV; ``<key>_err`` is absent when unavailable."""
        out: dict[str, float] = {}
        for key, item, _unit in self._flat_items():
            if item is not None:
                out[key] = item.value
                if item.stderr is not None:
                    out[key + "_err"] = item.stderr
        return out

    def flat_units(self) -> dict[str, str]:
        """The unit of each :meth:`flat_params` key (its ``_err`` shares it)."""
        out: dict[str, str] = {}
        for key, item, unit in self._flat_items():
            if item is not None:
                out[key] = unit
                if item.stderr is not None:
                    out[key + "_err"] = unit
        return out


def _as_axis(value, default_name: str) -> RegionAxis:
    return value if type(value) is RegionAxis else RegionAxis(default_name, value)


def fit_2d_region(
    data,
    x,
    y,
    *,
    seeds: Sequence[PeakSeed] = (),
    profile: str = "gaussian",
    background: str = "linear",
    mask=None,
    sigma=None,
    cancel_token: threading.Event | None = None,
) -> RegionFitOutcome:
    """Fit one or more rotated 2-D peaks plus a plane to ``data`` in ``(x, y)`` order (``data[i, j]`` at
    ``(x[i], y[j])``). ``x``/``y`` are :class:`RegionAxis` or arrays of bin centres; profiles ``gaussian``,
    ``lorentzian2`` or ``pseudovoigt``; ``mask`` True excludes a sample; ``sigma`` selects a weighted fit."""
    return _fit_region(data, (_as_axis(x, "x"), _as_axis(y, "y")), seeds=seeds, profile=profile,
                       background=background, mask=mask, sigma=sigma, transform=None, physical_reason=None,
                       cancel_token=cancel_token)


def _reciprocal_transform(reciprocal_transform, orthonormal: bool):
    from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame, rsm_coordinate_matrix

    if orthonormal:  # Cartesian Q is its own metric; only an identity may be given
        return rsm_coordinate_matrix(RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU, reciprocal_transform), None
    if reciprocal_transform is None:
        return None, "RECIPROCAL_TRANSFORM_REQUIRED"
    return rsm_coordinate_matrix(RSMCoordinateFrame.HKL, reciprocal_transform), None


def fit_3d_region(
    data,
    axes: Sequence[RegionAxis],
    *,
    seeds: Sequence[PeakSeed] = (),
    background: str = "linear",
    mask=None,
    sigma=None,
    reciprocal_transform=None,
    orthonormal: bool = False,
    cancel_token: threading.Event | None = None,
) -> RegionFitOutcome:
    """Fit one to three seeded Gaussians plus an affine background jointly over a 3-D region (axis order).

    ``reciprocal_transform`` is the 3x3 matrix taking native coordinates to Cartesian Q in Angstrom^-1 (for an
    H/K/L volume, its persisted UB with 2 pi included, admitted by ``rsm_coordinate_matrix``); ``orthonormal``
    declares native axes that already are Cartesian Q. Without either, physical principal widths and directions
    are unavailable (``physical_reason``).
    """
    axes = tuple(axes)
    if len(axes) != 3 or any(type(axis) is not RegionAxis for axis in axes):
        raise ValueError("a 3-D region needs three RegionAxis objects")
    transform, reason = _reciprocal_transform(reciprocal_transform, orthonormal)
    return _fit_region(data, axes, seeds=seeds, profile="gaussian", background=background, mask=mask,
                       sigma=sigma, transform=transform, physical_reason=reason, cancel_token=cancel_token)


def fit_volume_region(
    volume,
    box: Mapping[str, tuple[float, float]] | None = None,
    *,
    seeds: Sequence[PeakSeed] = (),
    background: str = "linear",
    mask=None,
    sigma=None,
    reciprocal_transform=None,
    cancel_token: threading.Event | None = None,
) -> RegionFitOutcome:
    """Fit the box of an :class:`~xrd_tools.rsm.volume.RSMVolume` (inclusive ranges by axis name; omitted axes
    keep their full range). The cap is checked from the axis ranges before the box is read; contiguous boxes are
    views of the volume. A Cartesian-Q volume is its own metric; an H/K/L volume needs ``reciprocal_transform``
    (its persisted coordinate matrix) for physical values. ``mask`` (True = excluded) and ``sigma`` are
    full-volume arrays; empty (NaN) voxels are always excluded."""
    from xrd_tools.rsm.coordinate_frame import RSMCoordinateFrame
    from xrd_tools.rsm.volume import RSMVolume, _box_view, _mask_selector

    if type(volume) is not RSMVolume:
        raise TypeError("fit_volume_region needs an RSMVolume")
    frame = volume.coordinate_frame
    names = frame.axis_names
    box = dict(box or {})
    unknown = set(box) - set(names)
    if unknown:
        raise ValueError(f"box names unknown axes {sorted(unknown)!r}; the axes are {names!r}")
    selectors, counts = [], []
    for name, values in zip(names, volume.axis_values):
        lower, upper = box.get(name, (-np.inf, np.inf))
        chosen = (values >= lower) & (values <= upper)
        selectors.append(_mask_selector(chosen))
        counts.append(int(chosen.sum()))
    if math.prod(counts) > MAX_POINTS:
        raise RegionFitRefused("REGION_TOO_LARGE", f"the box holds {math.prod(counts):,} voxels; at most "
                               f"{MAX_POINTS:,} are fitted: narrow the box")
    thin = [name for name, count in zip(names, counts) if count < 2]
    if thin:
        raise RegionFitRefused("REGION_TOO_THIN", f"the box spans fewer than two bins on {thin!r}")
    axes = tuple(RegionAxis(name, values[np.arange(values.size)[selector]], unit)
                 for name, values, selector, unit in zip(names, volume.axis_values, selectors, frame.axis_units))
    boxed = []
    for name, array in (("mask", mask), ("sigma", sigma)):
        if array is not None:
            array = np.asarray(array, dtype=bool if name == "mask" else np.float64)
            if array.shape != volume.shape:
                raise ValueError(f"{name} shape {array.shape} must be the volume's {volume.shape}")
            array = _box_view(array, selectors)
        boxed.append(array)
    data = _box_view(volume.intensity, selectors)
    orthonormal = frame is RSMCoordinateFrame.Q_SAMPLE_CARTESIAN_XU
    transform, reason = _reciprocal_transform(reciprocal_transform, orthonormal)
    return _fit_region(data, axes, seeds=seeds, profile="gaussian", background=background, mask=boxed[0],
                       sigma=boxed[1], transform=transform, physical_reason=reason, cancel_token=cancel_token)
