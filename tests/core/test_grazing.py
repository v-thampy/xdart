"""Grazing-incidence corrections (P2b) — committed, notebook-free gates.

Material Si @ 10 keV (αc = 0.17913°, β = 7.59e-8, δ = 4.887e-6) — values verified
against xrayutilities 1.7.12.  The gates are the design's (§4 step 2): footprint
∝ 1/sin αi; refraction shifts qz the right way + vanishes far above αc; Fresnel
peaks at αc (Yoneda).
"""
from __future__ import annotations

import numpy as np
import pytest


def _stack(**kw):
    from xrd_tools.corrections.grazing import GICorrectionStack
    return GICorrectionStack(material="Si", energy_eV=10000.0, **kw)


class TestOpticalConstants:
    def test_si_reference_values(self):
        pytest.importorskip("xrayutilities")
        oc = _stack().optical_constants()
        assert np.degrees(oc["ac_rad"]) == pytest.approx(0.17913, abs=1e-4)
        assert oc["delta"] == pytest.approx(4.8873e-6, rel=1e-3)
        assert oc["beta"] == pytest.approx(7.5908e-8, rel=1e-3)
        # αc = sqrt(2δ) (small-angle)
        assert oc["ac_rad"] == pytest.approx(np.sqrt(2 * oc["delta"]), rel=1e-3)

    def test_unknown_material_raises(self):
        from xrd_tools.corrections.grazing import GICorrectionStack
        pytest.importorskip("xrayutilities")
        with pytest.raises(ValueError, match="not a predefined"):
            GICorrectionStack(material="Unobtanium", energy_eV=10000.0).optical_constants()

    def test_missing_material_or_energy_raises(self):
        from xrd_tools.corrections.grazing import GICorrectionStack
        with pytest.raises(ValueError, match="energy_eV is required"):
            GICorrectionStack(material="Si").optical_constants()
        pytest.importorskip("xrayutilities")
        with pytest.raises(ValueError, match="material is required"):
            GICorrectionStack(energy_eV=10000.0).optical_constants()


class TestPrimitiveGates:
    def test_footprint_proportional_to_inv_sin(self):
        from xrd_tools.corrections.grazing import footprint_weight
        ai = np.radians(np.array([0.25, 0.5, 1.0]))
        c = footprint_weight(ai)
        # C·sin αi is constant (== 1)
        np.testing.assert_allclose(c * np.sin(ai), 1.0)
        assert footprint_weight(np.radians(0.5)) / footprint_weight(np.radians(1.0)) \
            == pytest.approx(2.0, abs=1e-3)

    def test_fresnel_peaks_at_critical_angle(self):
        pytest.importorskip("xrayutilities")
        from xrd_tools.corrections.grazing import fresnel_transmission_sq
        oc = _stack().optical_constants()
        a = np.radians(np.linspace(0.02, 0.6, 4000))
        v = fresnel_transmission_sq(a, oc["ac_rad"], oc["beta"])
        # the Yoneda peak sits at αc (within one grid step)
        assert np.degrees(a[np.argmax(v)]) == pytest.approx(
            np.degrees(oc["ac_rad"]), abs=5e-3)
        assert v.max() > 1.0  # genuine enhancement

    def test_refraction_shift_positive_and_vanishes_above_ac(self):
        pytest.importorskip("xrayutilities")
        from xrd_tools.corrections.grazing import refracted_angle
        oc = _stack().optical_constants()
        shifts = []
        for d in (0.25, 0.5, 1.0, 3.0):
            a = np.radians(d)
            shifts.append(np.degrees(a - refracted_angle(a, oc["ac_rad"], oc["beta"])))
        assert all(s > 0 for s in shifts)                  # internal angle < external
        assert all(shifts[i] > shifts[i + 1] for i in range(3))  # monotone-decreasing
        assert shifts[-1] < 0.01                           # ~vanishes at 3° (≫ αc)


class TestGICorrectionStack:
    def test_refract_q_maps_qz_down_keeps_in_plane(self):
        pytest.importorskip("xrayutilities")
        gi = _stack()
        q0 = np.array([2.0, 1.5]); qz = np.array([0.5, 0.3])
        q_ip = np.sqrt(q0 ** 2 - qz ** 2)
        q_new = gi.refract_q(incident_angle_deg=0.3, alpha_f_rad=np.radians([0.4, 0.4]),
                             q_total=q0, q_z=qz)
        assert np.all(q_new < q0)                          # |q| shifted down
        # in-plane component preserved (qz changed, q_ip not)
        qz_new = np.sqrt(np.clip(q_new ** 2 - q_ip ** 2, 0, None))
        assert np.all(qz_new < qz)

    def test_gi_normalization_off_is_identity(self):
        pytest.importorskip("xrayutilities")
        gi = _stack(footprint=False, fresnel=False, absorption=False)
        af = np.radians(np.array([0.1, 0.5, 1.0]))
        np.testing.assert_array_equal(
            gi.gi_normalization(incident_angle_deg=0.3, alpha_f_rad=af),
            np.ones(3))

    def test_gi_normalization_on_is_finite_positive(self):
        pytest.importorskip("xrayutilities")
        gi = _stack()
        af = np.radians(np.array([0.1, 0.5, 1.0]))
        n = gi.gi_normalization(incident_angle_deg=0.3, alpha_f_rad=af)
        assert np.all(np.isfinite(n)) and np.all(n > 0)

    def test_footprint_only_is_constant_sin_ai(self):
        pytest.importorskip("xrayutilities")
        gi = _stack(fresnel=False, absorption=False)
        af = np.radians(np.array([0.1, 0.5, 1.0]))
        n = gi.gi_normalization(incident_angle_deg=0.3, alpha_f_rad=af)
        np.testing.assert_allclose(n, np.sin(np.radians(0.3)))  # per-frame scalar

    def test_fresnel_only_enhances_near_ac(self):
        pytest.importorskip("xrayutilities")
        gi = _stack(footprint=False, absorption=False)
        oc = gi.optical_constants()
        # at αf == αc the |T(αf)|² factor is the Yoneda max
        n_at_ac = gi.gi_normalization(incident_angle_deg=0.3,
                                      alpha_f_rad=np.array([oc["ac_rad"]]))
        n_far = gi.gi_normalization(incident_angle_deg=0.3,
                                    alpha_f_rad=np.radians([1.0]))
        assert n_at_ac[0] > n_far[0]   # norm larger at αc (measured enhanced → divided out)

    def test_provenance_roundtrip(self):
        from xrd_tools.corrections.grazing import GICorrectionStack
        gi = GICorrectionStack(material="SiO2", energy_eV=12000.0,
                               density_kg_m3=2200.0, film_thickness_A=500.0,
                               fresnel=False)
        assert GICorrectionStack.from_dict(gi.to_dict()) == gi
