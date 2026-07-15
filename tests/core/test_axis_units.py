"""Unit contracts shared by q-based analysis workflows."""

from __future__ import annotations

import pytest

from xrd_tools.analysis import canonical_q_unit, require_inverse_angstrom


@pytest.mark.parametrize(
    ("stored", "canonical"),
    [
        ("q_A^-1", "q_A^-1"),
        ("angstrom^-1", "q_A^-1"),
        ("1/angstrom", "q_A^-1"),
        ("qtot_A^-1", "q_A^-1"),
        ("q_total_A-1", "q_A^-1"),
        ("2th_deg", "2th_deg"),
        ("q_nm^-1", "q_nm^-1"),
        (None, ""),
    ],
)
def test_canonical_q_unit_preserves_or_normalizes_provenance(stored, canonical):
    assert canonical_q_unit(stored) == canonical


@pytest.mark.parametrize("stored", (None, "2th_deg", "q_nm^-1", "qip_A^-1"))
def test_require_inverse_angstrom_rejects_unproven_or_incompatible_q(stored):
    with pytest.raises(ValueError, match="inverse-angstrom q unit"):
        require_inverse_angstrom(stored, operation="test")
