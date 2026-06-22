# Advanced fitting options — catalog for a future GUI panel (Jun 2026)

Reference for exposing `PhaseFitter` / `fit_peaks` tuning knobs from the xdart
Peak/Phase Fitting GUI later. Every parameter below is verified against the
**current** signatures (`PhaseFitter.__init__`, `PhaseFitter.fit()`,
`PhaseFitter.add_phase()`, `fit_peaks()`) as of Jun 2026.

Tags: **[PEAK+PHASE]** = shared with structure-agnostic peak fitting
(`fit_peaks` / `PeakFitPlan` / `PeakFitAnalyzer`); **[PHASE-ONLY]** = needs a
registered crystallographic phase (`PhaseFitter`).

Panel scope: the **Peak-Fit** advanced panel exposes Groups 1–4 only
([PEAK+PHASE]); the **Phase-Fit** advanced panel adds Groups 5–7 ([PHASE-ONLY]).

> Naming traps (the old `hzo_fitting_example.py` tripped every one of these):
> the profile kwarg value is `pseudovoigt` (one word, **not** `pseudo_voigt`);
> the lattice tolerance is `lattice_pct` (**not** `lattice_float_pct`); the SNIP
> width flows through `prefit_background_kwargs={'snip_width': N}` (there is **no**
> writable `bg_snip_width` attribute and no public `_calculate_background()`).

---

## Group 1 — Background  (set in `PhaseFitter.__init__`)

| kwarg | tag | type / default | what it does |
|---|---|---|---|
| `prefit_background` | [PEAK+PHASE] | str \| ndarray \| None, `'none'` | baseline subtracted BEFORE the fit (no free params): `'none'` \| `'snip'` \| `'chebyshev'` \| a user array |
| `prefit_background_kwargs` | [PEAK+PHASE] | dict \| None, `None` | tuning for the prefit routine, e.g. `{'snip_width': 50}`. (`snip_width=` is a deprecated flat alias — route the GUI through this dict.) |
| `fit_background` | [PEAK+PHASE] | str \| None, `None` | in-fit background refined WITH free params: `'polynomial{N}'` \| `'chebyshev{N}'` \| `'spline{N}'` \| `'template'` \| combo e.g. `'template+chebyshev2'` (adds N+1 free coeffs) |
| `fit_background_template` | [PEAK+PHASE] | ndarray \| (x_ref, y_ref) \| None | substrate-only reference scaled by one amplitude; pair with `fit_background='template'` |
| `amorphous_peak` | [PEAK+PHASE] | str \| None, `None` | optional broad component profile (`'gaussian'`/`'lorentzian'`/`'pseudovoigt'`/`'voigt'`/…) |
| `amorphous_init` | [PEAK+PHASE] | dict \| None, `None` | initial params for it, e.g. `{'center': 1.5, 'sigma': 0.3, 'amplitude': 100}` |

## Group 2 — Peak profile  (`PhaseFitter.fit(phase_profile=…)`)

- `phase_profile` **[PEAK+PHASE]** `str`, default `'pseudovoigt'` — applied to ALL
  peaks: `gaussian` \| `lorentzian` \| `pseudovoigt` \| `voigt` \|
  `lorentzian_squared` \| `pearson7` \| `pearson4` \| `splitlorentzian` \|
  `moffat` \| `studentst` \| `skewedgaussian` (aliases `pvoigt`/`gauss`/`lor`).
- Profile shape params (auto-populate inputs from the chosen profile's spec/bounds):
  `fraction` (def 0.5 [0,1], pseudovoigt G/L mix); `voigt_gamma` (def 0.02, voigt
  L width); `expo` (def 1.5 [0.55,50], pearson7/4 tail); `skew` (def 0.0 [-10,10],
  pearson4); `skew_gamma` (def 0.0, skewedgaussian); `sigma_r` (def 0.02,
  splitlorentzian right HWHM); `moffat_beta` (def 1.0 [1.01,50], moffat).

## Group 3 — Width model  (`PhaseFitter.fit(width_model=…)`)

3-way selector that swaps the inputs (legacy `caglioti: bool` still accepted:
True→caglioti, False→fixed; prefer `width_model`):

- **fixed** — per-phase `sigma` (guessed), bounded by `width_min`/`width_max`.
- **caglioti** — per-phase `U,V,W` with σ²(Q)=U·Q² + V·|Q| + W; `U` def 1e-6
  [0,1e-2], `V` def 0 [-1e-2,1e-2], `W` guessed.
- **scherrer** (Williamson–Hall) — per-phase `D` (crystallite size Å, def
  π·K/σ_guess, [π·K/width_max, 1e5]) and `eps` (microstrain, def 1e-3, [0,0.1]).
- `width_max` / `width_min` **[PEAK+PHASE]** — upper/lower σ bounds (caglioti maps
  W_max = width_max²; None → data_range/4 and 1e-5 respectively).

## Group 4 — Solver  (`PhaseFitter.fit(…)`)

| kwarg | type / default | what it does |
|---|---|---|
| `method` | str, `'leastsq'` | lmfit minimizer: `leastsq` (L-M) \| `least_squares` \| `nelder` \| `emcee` \| … |
| `nan_policy` | str, `'omit'` | `'omit'` (drop NaN/dummy bins) \| `'raise'` |
| `q_range` | (qmin,qmax) \| None | restrict the minimization domain (data outside is evaluated but not fitted) |
| `q_shift_bound` | float, `0.05` | max \|global q-shift\| (Å⁻¹), shared by all phases, bounds [−b,+b] |
| `**fit_kwargs` | dict | passthrough to lmfit `Model.fit()`: `max_nfev` (~5000), `scale_covar`, … — expose as a small JSON/dict editor |

## Group 5 — Lattice & constraints  **[PHASE-ONLY]**  (`PhaseFitter.fit(…)`)

- `lattice_pct` float, def 0.05 — fractional tolerance band on a,b,c: a ∈ a₀(1±pct).
  (α/β/γ are never fit.)
- `lock_lattice_order` bool, def True — preserve a ≥ b ≥ c within a phase via
  non-negative gap params.
- `lock_cross_phase` bool, def False — extend same-axis ordering across phases
  (advanced; only meaningful with `lock_lattice_order=True` — warn otherwise).

## Group 6 — Texture / preferred orientation  **[PHASE-ONLY]**  (`PhaseFitter.fit(…)`)

- `texture` str, def `'none'` — `'none'` \| `'march_dollase'` \| `'free'` (swap the
  input area on selection).
- `march_axis` (h,k,l), def (0,0,1) — texture axis; only with `march_dollase`.
- `march_r` per-phase float, def 1.0 [0.1,5.0] — March–Dollase (<1 plate-like,
  >1 needle-like); requires a phase with a lattice metric (error if missing).
- `pk_scale_range` (min,max), def (0.0,10.0) — bounds for the per-peak multipliers
  when `texture='free'` (adds N_peaks free params — flexible but risky).

## Group 7 — Phase selection / filtering  **[PHASE-ONLY]**  (`PhaseFitter.add_phase(…)`)

- `q_range` (per `add_phase`) (qmin,qmax) \| None — restrict that phase's peaks to
  a q window (None → auto ~10% margin).
- `min_intensity` (per `add_phase`) float, def 0.5 — drop template peaks below this
  fraction of phase max intensity (reduces free params).
- `phase_names` (in `FitConfig`) list[str] — which phases participate.

---

## Serialization & UX notes

- **Canonical form:** use `FitConfig` (an `init_kw` dict for Group 1 + amorphous,
  a `fit_kw` dict for Groups 2–7) as the JSON-serializable record; store
  `march_axis` / `q_range` as lists, rehydrate to tuples via `FitConfig.from_dict`.
- **Profile shape discovery:** when the user picks `phase_profile`, populate the
  shape-param fields from that profile's spec/bounds; show only the relevant ones.
- **Width selector:** a 3-tab control (Fixed / Caglioti / Scherrer) that swaps the
  inputs and stores the choice as the `width_model` string.
- **Texture selector:** none / march_dollase / free swaps the input area
  (`march_axis`+`march_r`, or `pk_scale_range`, or nothing).
- **q-range masking:** shade the masked regions on the pattern plot when a fit
  `q_range` is set.
- **Validation before fit:** ≥1 phase (or `fit_background`/amorphous) registered;
  `march_dollase` requires a lattice metric; warn that `voigt`/`skewedgaussian`
  converge slower; `lock_cross_phase` without `lock_lattice_order` is meaningless.

## Headless gaps this catalog surfaced (follow-ups, not blockers)

1. **No scan-unit concrete `Analyzer`.** The runner protocol has `analyze_scan`,
   but the only concrete analyzer (`PeakFitAnalyzer`) is `unit='frame'`. A
   `PhaseFitAnalyzer` (wrapping `PhaseFitPlan`/`run_phase_fit`, frame-unit) and a
   `Sin2PsiAnalyzer` (wrapping `Sin2PsiPlan`/`run_sin2psi`, scan-unit) would let
   xdart's live/batch runners drive phase-fit + sin²ψ through the same contract.
   The underlying Plans/`run_*` are already headless and callable.
2. **No standalone background helper.** The prefit baselines (`snip`/`chebyshev`)
   are reachable only via a `PhaseFitter` instance — there is no top-level
   `estimate_background(x, y, kind, **kw) -> ndarray` for a pipeline that just
   wants a baseline.
3. **No headless analysis example / CI guard until now.** Added
   `examples/headless_analysis.py` (the runner parallel to `headless_sanity.py`);
   a notebook import-cleanliness CI check (no `xdart`/`pyqtgraph` in `sys.modules`)
   is still worth adding.
