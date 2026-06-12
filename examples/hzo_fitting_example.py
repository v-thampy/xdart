# examples/fit_hzo_thin_films.py

import numpy as np
import matplotlib.pyplot as plt

# Import the new Phase Analysis tools
from xrd_tools.analysis.phase import PhaseModel
from xrd_tools.analysis.fitting.phase_fitting import PhaseFitter

def main():
    # -----------------------------------------------------------------
    # 1. Load Dummy/Experimental Data
    # -----------------------------------------------------------------
    # Suppose you have an integrated 1D profile `q_array` and `intensity_array`
    # from your `.nxs` file or the `test_pyfai_fiber.ipynb` notebook.
    
    # [Placeholder] Replace this with your actual data loading:
    q_array = np.linspace(1.5, 4.5, 1000)
    intensity_array = np.random.normal(10, 1, size=1000) # Dummy background noise
    
    # -----------------------------------------------------------------
    # 2. Define the Structural Phases (CIF Files)
    # -----------------------------------------------------------------
    # You will need CIF files downloaded from Materials Project or ICSD.
    # For Hf0.5Zr0.5O2 (HZO), the space groups usually are:
    # - Monoclinic (m-phase, P2_1/c)
    # - Tetragonal (t-phase, P4_2/nmc) 
    # - Orthorhombic (o-phase, Pca2_1 or Pmn2_1 - the ferroelectric one!)
    
    # Load primary HZO phases
    phase_m = PhaseModel.from_cif("cifs/HZO_monoclinic.cif", name="m-HZO")
    phase_t = PhaseModel.from_cif("cifs/HZO_tetragonal.cif", name="t-HZO")
    phase_o = PhaseModel.from_cif("cifs/HZO_orthorhombic.cif", name="o-HZO")
    
    # Load substrate/electrode phases
    phase_tin = PhaseModel.from_cif("cifs/TiN_cubic.cif", name="TiN")
    phase_pt = PhaseModel.from_cif("cifs/Pt_cubic.cif", name="Pt")
    
    # -----------------------------------------------------------------
    # 3. Setup the Phase Fitter
    # -----------------------------------------------------------------
    # For very broad peaks (thin films), a pseudo-voigt or voigt shape
    # handles the localized broadening effects better than pure Gaussians.
    fitter = PhaseFitter(q_array, intensity_array, peak_shape="pseudo_voigt")
    
    # Add phases into the fitter
    for phase in [phase_m, phase_t, phase_o, phase_tin, phase_pt]:
        fitter.add_phase(phase)
        
    # The large amorphous SiO2 background will be naturally targeted 
    # by the SNIP background extractor inside the fitter, but let's 
    # increase the SNIP width slightly to ensure it encapsulates the wide hump 
    # without cutting off the broad HZO peaks.
    fitter.bg_snip_width = int(len(q_array) * 0.1) # 10% of window width
    fitter._calculate_background() # Re-evaluate background
    
    # -----------------------------------------------------------------
    # 4. Customize Fitting Parameters for Thin Films
    # -----------------------------------------------------------------
    # Build standard constraints
    params = fitter.build_parameters(q_shift_bound=0.05, lattice_float_pct=0.05)
    
    # Thin films cause HUGE inhomogeneous broadening. 
    # Let's adjust the bounds and initial guesses for 'sigma' significantly.
    for p_idx in range(len(fitter.phases)):
        prefix = f"phase{p_idx}_"
        # Increase expected broadening defaults
        params[f"{prefix}sigma"].set(value=0.08, min=0.01, max=0.5)
        # Give substrate phases (TiN, Pt) sharper bounds if they are thick/crystalline:
        if fitter.phases[p_idx].name in ["TiN", "Pt"]:
            params[f"{prefix}sigma"].set(value=0.02, min=0.005, max=0.1)
    
    # -----------------------------------------------------------------
    # 5. Execute Fit & Analyze
    # -----------------------------------------------------------------
    print("Beginning Least-Squares Minimization...")
    result = fitter.fit(max_nfev=2000) # Allow more iterations for complex multi-phase
    
    print(result.fit_report())
    
    # Extract calculated fraction metrics
    # Note: Full Rietveld weight fraction is proportional to (Scale * cell_mass / cell_volume).
    # This scale factor gives you raw intensity approximations for phase abundance.
    print("\n[Extracted Phase Scalings]")
    for p_idx, phase in enumerate(fitter.phases):
        scale_val = result.params[f"phase{p_idx}_scale"].value
        print(f" - {phase.name}: {scale_val:.4f}")
        
    print(f"\nGlobal sample misalignment shift (q-shift): {result.params['q_shift'].value:.4f} A^-1")
    
    # Plotting
    fitter.plot_fit(result, show=True)

if __name__ == "__main__":
    main()
