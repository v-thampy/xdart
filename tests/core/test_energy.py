"""The single canonical energy↔wavelength conversion + consistency guard."""
from __future__ import annotations

import logging

import pytest

from xrd_tools.core.energy import (
    DEFAULT_WAVELENGTH_SENTINEL_M,
    WavelengthUnit,
    canonical_wavelength_m,
    check_energy_consistency,
    energy_eV_to_wavelength_m,
    is_default_wavelength_sentinel_m,
    normalize_wavelength_m,
    wavelength_angstrom_to_m,
    wavelength_m_to_angstrom,
    wavelength_m_to_energy_eV,
)


def test_roundtrip_energy_wavelength():
    for ev in (8000.0, 10000.0, 17479.0):
        lam = energy_eV_to_wavelength_m(ev)
        assert wavelength_m_to_energy_eV(lam) == pytest.approx(ev, rel=1e-12)


def test_known_value():
    # 1 Å ≈ 12398.42 eV (CODATA hc); 10 keV ≈ 1.2398 Å
    assert wavelength_m_to_energy_eV(1.0e-10) == pytest.approx(12398.42, abs=0.1)
    assert energy_eV_to_wavelength_m(10000.0) == pytest.approx(1.23984e-10, rel=1e-4)


def test_consistency_warns_on_mismatch(caplog):
    with caplog.at_level(logging.WARNING, logger="xrd_tools.core.energy"):
        check_energy_consistency(10000.0, 10500.0, what_a="a", what_b="b")
    assert any("energy mismatch" in r.message for r in caplog.records)


def test_consistency_quiet_when_agree(caplog):
    with caplog.at_level(logging.WARNING, logger="xrd_tools.core.energy"):
        check_energy_consistency(10000.0, 10001.0, what_a="a", what_b="b")  # <0.1%
        check_energy_consistency(None, 10000.0, what_a="a", what_b="b")     # None → no-op
    assert not caplog.records


# ── X1 Slice 3a0: the headless wavelength vocabulary (R3-P1) ─────────────────

def test_wavelength_unit_exact_members():
    assert WavelengthUnit.METRE.value == "m"
    assert WavelengthUnit.ANGSTROM.value == "angstrom"
    assert set(WavelengthUnit) == {WavelengthUnit.METRE, WavelengthUnit.ANGSTROM}


def test_canonical_wavelength_m_declared_units_only():
    # explicit metre declaration: value passes through
    assert canonical_wavelength_m(1.54e-10, WavelengthUnit.METRE) \
        == pytest.approx(1.54e-10)
    # explicit angstrom declaration: canonicalized to metres
    assert canonical_wavelength_m(1.54, WavelengthUnit.ANGSTROM) \
        == pytest.approx(1.54e-10)
    # NO unit declared → no evidence, never magnitude inference
    assert canonical_wavelength_m(1.54, None) is None
    assert canonical_wavelength_m(1.54e-10, None) is None
    # non-numeric / non-positive / non-finite → None
    assert canonical_wavelength_m("nan?", WavelengthUnit.METRE) is None
    assert canonical_wavelength_m(-1.0, WavelengthUnit.ANGSTROM) is None
    assert canonical_wavelength_m(0.0, WavelengthUnit.METRE) is None
    assert canonical_wavelength_m(float("nan"), WavelengthUnit.ANGSTROM) is None
    assert canonical_wavelength_m(None, WavelengthUnit.METRE) is None


def test_canonical_wavelength_m_explicit_one_angstrom_is_valid():
    """Sentinel rejection is provenance-sensitive: an explicitly DECLARED
    1.0 Å (or 1e-10 m) source is valid physical evidence even though it equals
    the historical constructor placeholder."""
    assert canonical_wavelength_m(1.0, WavelengthUnit.ANGSTROM) \
        == pytest.approx(1.0e-10)
    assert canonical_wavelength_m(1.0e-10, WavelengthUnit.METRE) \
        == pytest.approx(1.0e-10)


def test_migrated_sentinel_helpers_keep_legacy_behavior():
    """The canonical helpers keep their exact legacy-placeholder semantics:
    the untrusted metre placeholder is rejected by default and admitted only
    for authoritative sources."""
    assert DEFAULT_WAVELENGTH_SENTINEL_M == 1.0e-10
    assert is_default_wavelength_sentinel_m(1.0e-10) is True
    assert is_default_wavelength_sentinel_m(1.5e-10) is False
    assert is_default_wavelength_sentinel_m("bogus") is False

    assert normalize_wavelength_m(1.0e-10) is None            # sentinel rejected
    assert normalize_wavelength_m(1.0e-10, allow_default_sentinel=True) \
        == pytest.approx(1.0e-10)
    assert normalize_wavelength_m(1.5e-10) == pytest.approx(1.5e-10)
    assert normalize_wavelength_m(-1.0) is None
    assert normalize_wavelength_m("x") is None

    assert wavelength_m_to_angstrom(1.5e-10) == pytest.approx(1.5)
    assert wavelength_m_to_angstrom(1.0e-10) is None          # sentinel rejected
    assert wavelength_m_to_angstrom(1.0e-10, allow_default_sentinel=True) \
        == pytest.approx(1.0)
    assert wavelength_angstrom_to_m(1.54) == pytest.approx(1.54e-10)
    assert wavelength_angstrom_to_m(-2.0) is None
    assert wavelength_angstrom_to_m("x") is None
