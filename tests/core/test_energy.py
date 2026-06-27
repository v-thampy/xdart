"""The single canonical energy↔wavelength conversion + consistency guard."""
from __future__ import annotations

import logging

import pytest

from xrd_tools.core.energy import (
    check_energy_consistency,
    energy_eV_to_wavelength_m,
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
