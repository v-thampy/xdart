"""The source energy fact is optional: a source without it reads as raw, as before B1's correction."""
from types import SimpleNamespace

import pytest

from xrd_tools.sources.base import source_metadata_energy_eV


def test_a_source_without_the_energy_fact_reads_its_raw_motor_energy_by_size():
    assert source_metadata_energy_eV(SimpleNamespace()) == ()
    assert source_metadata_energy_eV(SimpleNamespace(motors={'energy': [12.4]})) == pytest.approx((12400.,))
    assert source_metadata_energy_eV(SimpleNamespace(motors={'energy': [12400.]})) == pytest.approx((12400.,))


def test_a_source_with_the_energy_fact_keeps_its_canonical_ev():
    source = SimpleNamespace(metadata_energy_eV=(100.,), motors={'energy': [0.1]})
    assert source_metadata_energy_eV(source) == pytest.approx((100.,))
