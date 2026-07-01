"""Multi-phase structure-informed fitting of HZO thin films (headless).

Demonstrates the current ``PhaseFitter`` API on a Hf0.5Zr0.5O2 (HZO) thin film
with several crystallographic phases plus crystalline substrate/electrode
layers.  Thin films show large inhomogeneous broadening, so a pseudo-Voigt
profile with generous width bounds is used.

This is a *template*: it needs your own integrated pattern (``q``, ``intensity``)
and a CIF per phase.  Every API call is current; swap in real data + CIF paths
to run it.

Headless: imports only ``xrd_tools`` + numpy/matplotlib — no Qt / xdart.  The
same ``PhaseFitter`` drives xdart's Phase-Fit GUI; for fitting a whole sequence
of patterns see ``FitConfig`` + ``fit_sequence`` (notebook 04), and for one call
from arrays see ``xrd_tools.analysis.plans.make_phase_fitter``.
"""

import matplotlib.pyplot as plt
import numpy as np

from xrd_tools.analysis.fitting import PhaseFitter   # package re-export
from xrd_tools.analysis.phase import PhaseModel


def main():
    # -----------------------------------------------------------------
    # 1. Load your integrated 1-D pattern (q in Å⁻¹, intensity).
    # -----------------------------------------------------------------
    # [Placeholder] Replace with your real data (e.g. xrd_tools.io.get_1d on a
    # processed .nxs).  The dummy noise below just lets the script run end-to-end.
    q_array = np.linspace(1.5, 4.5, 1000)
    intensity_array = np.random.normal(10, 1, size=1000)

    # -----------------------------------------------------------------
    # 2. Define the structural phases (one CIF each).
    # -----------------------------------------------------------------
    # HZO commonly shows monoclinic (P2_1/c), tetragonal (P4_2/nmc), and the
    # ferroelectric orthorhombic (Pca2_1) phases, over a crystalline electrode.
    phase_m = PhaseModel.from_cif("cifs/HZO_monoclinic.cif", name="m-HZO")
    phase_t = PhaseModel.from_cif("cifs/HZO_tetragonal.cif", name="t-HZO")
    phase_o = PhaseModel.from_cif("cifs/HZO_orthorhombic.cif", name="o-HZO")
    phase_tin = PhaseModel.from_cif("cifs/TiN_cubic.cif", name="TiN")
    phase_pt = PhaseModel.from_cif("cifs/Pt_cubic.cif", name="Pt")

    # -----------------------------------------------------------------
    # 3. Set up the fitter.
    # -----------------------------------------------------------------
    # The broad amorphous/substrate hump is removed by a SNIP baseline BEFORE the
    # fit (no free parameters).  A wide SNIP window encapsulates the hump without
    # clipping the broad HZO peaks.  Baseline options flow through
    # prefit_background / prefit_background_kwargs (NOT a writable attribute).
    fitter = PhaseFitter(
        q_array, intensity_array,
        prefit_background="snip",
        prefit_background_kwargs={"snip_width": int(len(q_array) * 0.1)},
    )
    for phase in (phase_m, phase_t, phase_o, phase_tin, phase_pt):
        fitter.add_phase(phase)

    # -----------------------------------------------------------------
    # 4. Fit.  Profile + constraints are passed to fit() — it builds the lmfit
    #    model + parameters internally (no manual build_parameters needed).
    # -----------------------------------------------------------------
    print("Beginning least-squares minimization…")
    result = fitter.fit(
        phase_profile="pseudovoigt",   # one word; handles thin-film tails
        q_shift_bound=0.05,            # global sample-misalignment shift, ±0.05 Å⁻¹
        lattice_pct=0.05,             # allow a/b/c to float ±5%
        width_max=0.5,                # thin films broaden a lot — raise the σ cap
        max_nfev=2000,                # passthrough to lmfit (complex multi-phase)
    )

    # -----------------------------------------------------------------
    # 5. Report (use the result accessors, not raw param-name guessing).
    # -----------------------------------------------------------------
    print(result.summary())
    print(f"Global q-shift: {result.q_shift:.4f} Å⁻¹")
    print("\n[Phase fractions  (scale_i / Σ scale)]")
    for name, frac in result.phase_fractions().items():
        print(f" - {name}: {frac:.4f}")

    # Plot data vs. the full model (phases + background).
    model = fitter.eval_model(result.params)
    plt.figure(figsize=(9, 4))
    plt.plot(q_array, intensity_array, lw=0.8, color="0.6", label="data")
    plt.plot(q_array, model, lw=1.4, color="C3", label="fit")
    plt.xlabel("q (Å⁻¹)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
