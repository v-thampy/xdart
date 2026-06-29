"""Validate the GI corrections against REAL grazing-incidence data.

Data: the del-only LaB6 stitch (fixed grazing αi≈0.2°, 16 keV, Pilatus300kw) from
``example_notebooks/Stitching/data_del_only`` — the same scans the team's
``Multi120_GI_Corrections_Explorer.ipynb`` (xu q-vector ground truth + the
reference correction ``compute()``) and the xu-vs-pyFAI comparison notebooks use.

Run (env with pyFAI + xrayutilities, e.g. xrd_edit/xrd_test):
    python scripts/gi_real_data_validation.py
Skips cleanly if the (external, private) data is not present.

What it checks, layer by layer:
  A. data load + xu q ground truth  → raw |q| reproduces the known LaB6 peaks.
  B. GICorrectionStack vs the notebook reference.  We merge I = Σraw/Σnorm (the
     boost in the denominator); the notebook merges I = Σ(raw·w)/Σw (1/boost in w),
     so our ``gi_normalization`` must equal the notebook's combined ``1/w``
     factor-for-factor (footprint / Fresnel / absorption).
  C. refract_q vs the notebook q-shift, by exit-angle regime.
  D. THE GATED CONVENTION: does our pyFAI FiberIntegrator exit-angle (αf) map match
     the xu-derived αf (per sample_orientation)?  This is the piece that feeds the
     (validated) corrections their αf, so it must be right before GI stitch is
     unblocked.

Result (Jun 2026): A/B/C PASS — the GI correction physics matches the team
reference exactly.  D FAILS — pyFAI's fiber exit-angle axis (~6.5° span) does not
match the physical vertical (the 1475 detector long axis, ~34.5° span) for any
sample_orientation 1-4 at tilt=0, so the GI stitch guard correctly stays until the
detector-orientation / fiber-vertical-axis convention is resolved.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

STITCH = Path.home() / "repos" / "example_notebooks" / "Stitching"
BASE = STITCH / "data_del_only"
SPEC = BASE / "calibration" / "LaB6_16keV"


def _skip(msg: str) -> None:
    print(f"SKIP: {msg}")
    sys.exit(0)


def main() -> int:
    try:
        import pyFAI  # noqa: F401
        import xrayutilities as xu
        from pyFAI.integrator.fiber import FiberIntegrator
        import pyFAI.units as U
    except Exception as exc:  # noqa: BLE001
        _skip(f"pyFAI/xrayutilities unavailable: {exc!r}")

    if not SPEC.exists():
        _skip(f"GI reference data not found at {BASE} (external/private dataset)")

    from xrd_tools.io.spec import get_angles
    from xrd_tools.io.image import read_image
    from xrd_tools.corrections.grazing import GICorrectionStack

    RAWS = sorted(glob.glob(str(BASE / "images" / "b_thampy_LaB6_16keV_scan7_*.raw")))
    SHAPE, PIX = (195, 1475), 172e-6
    deld = np.array(get_angles(str(SPEC), "7.1", ["del"])[0])
    EN_eV = 15999.985
    WL_A = 12398.4198 / EN_eV
    K = 2 * np.pi / WL_A
    AI_DEG, MAT = 0.20, "Si"
    ai = np.radians(AI_DEG)
    LAB6 = np.array([1.512, 2.138, 2.618, 3.023, 3.380, 3.702, 4.276, 4.541, 5.013])
    PF = dict(dist=0.39353, poni1=0.03312, poni2=0.04979, rot1=0.00273,
              rot2_offset=0.00777, rot2_scale=0.01741)
    detector = pyFAI.detector_factory("Pilatus300kw")

    qc = xu.QConversion(['x+', 'z-', 'y+', 'z-'], ['x+', 'z-'], [0, 1, 0])
    HX = xu.HXRD([0, 1, 0], [0, 0, 1], geometry='real', en=EN_eV, qconv=qc)
    HX.Ang2Q.init_area('x-', 'z+', cch1=193.24, cch2=282.04, Nch1=195, Nch2=1475,
                       pwidth1=PIX * 1e3, pwidth2=PIX * 1e3, distance=392.77)

    def relmax(a, b, sel=None):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if sel is not None:
            a, b = a[sel], b[sel]
        fin = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-9)
        return float(np.max(np.abs(a[fin] - b[fin]) / np.abs(b[fin]))) if fin.any() else float("nan")

    results = {}

    # ---- A. data load + LaB6 peaks --------------------------------------
    Q0, QZ, IRAW = [], [], []
    for k in range(0, len(RAWS), 3):
        d = float(deld[k])
        qx, qy, qz = HX.Ang2Q.area(0., 0., 0., 0., 0., d + 0.353)
        w = np.nan_to_num(read_image(RAWS[k], detector_shape=SHAPE, raw_dtype="int32", threshold=8e5))
        m = w > 0
        Q0.append(np.sqrt(qx**2 + qy**2 + qz**2)[m]); QZ.append(qz[m]); IRAW.append(w[m])
    Q0 = np.concatenate(Q0).astype(float); QZ = np.concatenate(QZ).astype(float)
    IRAW = np.concatenate(IRAW).astype(float)
    qb = np.linspace(1.0, 5.0, 421); qcen = 0.5 * (qb[:-1] + qb[1:])
    num, _ = np.histogram(Q0, bins=qb, weights=IRAW); den, _ = np.histogram(Q0, bins=qb)
    Iq = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    from scipy.signal import find_peaks
    pk, _ = find_peaks(np.nan_to_num(Iq), height=np.nanpercentile(Iq, 90), distance=5)
    matched = sum(any(abs(qcen[p] - r) < 0.02 for p in pk) for r in LAB6)
    results["A_lab6_peaks"] = (matched >= 6, f"{matched}/{len(LAB6)} LaB6 peaks matched in raw |q|")

    # ---- B. corrections vs the notebook reference -----------------------
    af = np.arcsin(np.clip(QZ / K - np.sin(ai), -1, 1))
    afc = np.clip(af, np.radians(0.01), None)
    m = getattr(xu.materials, MAT)
    delta, beta = float(m.delta(EN_eV)), float(m.ibeta(EN_eV))
    ac = np.sqrt(2 * delta)

    def nbT2(a_rad):
        a = np.asarray(a_rad, complex)
        return np.abs(2 * a / (a + np.sqrt(a**2 - 2 * delta - 2j * beta))) ** 2

    Ti = float(nbT2(np.array([ai]))[0])
    nb_w = np.float32(np.sin(ai)) * (1.0 / np.sin(ai) + 1.0 / np.sin(afc)) * (1.0 / (Ti * nbT2(afc)))
    our_norm = GICorrectionStack(material=MAT, energy_eV=EN_eV).gi_normalization(
        incident_angle_deg=AI_DEG, alpha_f_rad=afc)   # footprint+fresnel+absorption (no refraction term in norm)
    # our default stack has refraction=True but that's a q-shift, not in gi_normalization
    rm_corr = relmax(our_norm, 1.0 / nb_w)
    results["B_corrections"] = (rm_corr < 1e-3, f"gi_normalization vs notebook 1/w  relmax={rm_corr:.2e}")

    # ---- C. refraction q-shift, above the critical angle ----------------
    ai_in = np.sqrt(max(ai**2 - ac**2, 0.0)); af_in = np.sqrt(np.clip(af**2 - ac**2, 0, None))
    q_nb = np.sqrt(np.clip(Q0**2 - QZ**2 + (K * (np.sin(af_in) + np.sin(ai_in)))**2, 0, None))
    gi_rf = GICorrectionStack(material=MAT, energy_eV=EN_eV, footprint=False,
                              fresnel=False, absorption=False, refraction=True)
    q_our = gi_rf.refract_q(incident_angle_deg=AI_DEG, alpha_f_rad=af, q_total=Q0, q_z=QZ)
    rm_refr = relmax(q_our, q_nb, sel=af > ac)
    results["C_refraction"] = (rm_refr < 1e-5,
                               f"refract_q vs notebook (af>αc)  relmax={rm_refr:.2e} "
                               f"[below-horizon af≤0 differ by contract: reflection-geometry vs |af|]")

    # ---- D. THE GATED CONVENTION: pyFAI fiber αf vs xu αf ----------------
    k = len(RAWS) // 2; d = float(deld[k])
    qx, qy, qz = HX.Ang2Q.area(0., 0., 0., 0., 0., d + 0.353)
    af_xu = np.arcsin(np.clip(qz / K - np.sin(ai), -1, 1))
    fi = FiberIntegrator(dist=PF["dist"], poni1=PF["poni1"], poni2=PF["poni2"],
                         rot1=PF["rot1"], rot2=PF["rot2_scale"] * d + PF["rot2_offset"],
                         rot3=0.0, detector=detector, wavelength=WL_A * 1e-10)
    q_pf = np.asarray(fi.qArray(shape=SHAPE), dtype=float) / 10.0
    qcal = relmax(q_pf, np.sqrt(qx**2 + qy**2 + qz**2))
    above = af_xu > np.radians(0.5)
    best = np.inf
    for so in (1, 2, 3, 4):
        fi.reset_integrator(incident_angle=ai, tilt_angle=0.0, sample_orientation=so)
        af_u = U.get_unit_fiber("exit_angle_vert_rad", incident_angle=ai,
                                tilt_angle=0.0, sample_orientation=so)
        af_pf = np.asarray(fi.array_from_unit(SHAPE, "center", af_u), dtype=float)
        best = min(best, float(np.degrees(np.nanmax(np.abs(af_pf[above] - af_xu[above])))))
    phys_span = float(np.degrees(np.ptp(af_xu)))
    results["D_fiber_convention"] = (
        best < 1.0,
        f"|q| pf-vs-xu relmax={qcal:.2e} (calib OK); best αf mismatch over "
        f"sample_orientation 1-4 = {best:.1f}° vs physical αf span {phys_span:.1f}° "
        f"=> pyFAI fiber 'vertical' axis is wrong for this geometry (gate stands)")

    print("\n=== GI real-data validation ===")
    ok = True
    for key, (passed, msg) in results.items():
        ok = ok and passed
        print(f"[{'PASS' if passed else 'GAP '}] {key}: {msg}")
    print(f"\nGI correction physics: {'VALIDATED' if all(results[k][0] for k in ('A_lab6_peaks','B_corrections','C_refraction')) else 'REVIEW'}")
    print(f"GI fiber αf convention: {'VALIDATED' if results['D_fiber_convention'][0] else 'GATED (unresolved)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
