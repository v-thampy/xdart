"""Frozen P3b KP1–KP4: real phase listing, library and reference contracts.

KP1 and KP2 start on existing public behavior and fail on bb0a2be9.
KP3/KP4 introduce APIs and earn no behavioral parent-polarity credit.
The crystal-condition oracles below are independent of the implementation.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pytest

from xrd_tools.analysis.phase import HAS_PYMATGEN, PhaseModel

pytestmark = pytest.mark.skipif(not HAS_PYMATGEN, reason="requires pymatgen")

_CIF = """data_primitive
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
loop_
_symmetry_equiv_pos_as_xyz
'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Si1 Si 0.123 0.234 0.345 1
"""


def _has_d(phase, d):
    return any(np.isclose(peak.d_spacing, d, rtol=1e-7, atol=0) for peak in phase.peaks)


def _q_values(phase):
    return np.asarray(sorted(peak.q for peak in phase.peaks))


def _phase_api():
    from xrd_tools.analysis.phase_references import PhaseLibrary, build_phase_references
    return PhaseLibrary, build_phase_references


def test_lattice_entry_lists_every_space_group_allowed_reflection():
    # Pca21's general reflection conditions allow 110 and 111. A dummy at
    # the origin is a special-position motif and introduces extra absences.
    ortho = PhaseModel.from_lattice(
        "Ortho", a=5.06, b=5.23, c=5.07, space_group="Pca2_1",
    )
    d110 = 1 / np.sqrt(1 / 5.06**2 + 1 / 5.23**2)
    d111 = 1 / np.sqrt(1 / 5.06**2 + 1 / 5.23**2 + 1 / 5.07**2)
    assert _has_d(ortho, d110) and _has_d(ortho, d111), (
        "manual Pca21 lattice must include space-group-allowed 110 and 111"
    )

    # Centering is an independent condition, with deliberately distinct
    # reciprocal lengths avoiding accidental d degeneracy in the checks.
    fcc = PhaseModel.from_lattice("fcc", a=4.0, space_group="Fm-3m")
    bcc = PhaseModel.from_lattice("bcc", a=4.0, space_group="Im-3m")
    for h in range(5):
        for k in range(h + 1):
            for l in range(k + 1):
                if h == k == l == 0:
                    continue
                d = 4 / np.sqrt(h * h + k * k + l * l)
                if d < 1.5406 / 2:
                    continue
                if h % 2 == k % 2 == l % 2:
                    assert _has_d(fcc, d)
                if (h + k + l) % 2 == 0:
                    assert _has_d(bcc, d)
    assert not _has_d(fcc, 4.0)   # 100 forbidden by F centering.
    assert not _has_d(fcc, 4 / np.sqrt(2))  # 110 forbidden.
    assert not _has_d(bcc, 4.0)   # 100 forbidden by I centering.
    assert not _has_d(bcc, 4 / np.sqrt(3))  # 111 forbidden.

    # A 6_3 screw makes 000l require even l; choose a small a so no other
    # reflection reaches the tested low-q axial positions.
    hexagonal = PhaseModel.from_lattice(
        "hex", a=1.0, c=8.0, gamma=120.0, space_group="P6_3/mmc",
    )
    for l in (2, 4, 6):
        assert _has_d(hexagonal, 8 / l)
    for l in (1, 3, 5):
        assert not _has_d(hexagonal, 8 / l)

    # pyFAI's shipped LaB6 cell is Pm-3m, a=4.156826 Å (SRM 660c).
    from pyFAI.calibrant import get_calibrant
    lab6 = PhaseModel.from_lattice("LaB6", a=4.156826, space_group="Pm-3m")
    for d in get_calibrant("LaB6").dSpacing:
        if d > 1.5406 / 2:
            assert _has_d(lab6, d)


def test_reflection_listing_is_bounded_by_the_reachable_range(tmp_path, monkeypatch):
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    from xrd_tools.analysis.display_fit_operations import (
        DisplayedPhaseFitPlan, DisplayedTraceInput, DisplayedTraceReceipt,
        run_displayed_phase_fit,
    )

    cif = tmp_path / "primitive.cif"
    cif.write_text(_CIF, encoding="utf-8")
    q = np.linspace(1.0, 6.0, 251)
    phase = PhaseModel.from_cif(cif)
    phase.calculate_peaks(wavelength=0.25)
    y = np.ones_like(q)
    for peak in phase.peaks:
        if q[0] <= peak.q <= q[-1]:
            y += peak.intensity * np.exp(-((q - peak.q) / 0.025)**2)
    trace = DisplayedTraceInput(
        axis=q, intensity=y, label="bounded", axis_unit="1/angstrom",
        title="bounded listing", epoch=1,
        receipt=DisplayedTraceReceipt(
            "displayed-trace-v1", 1, "run", "1", tmp_path / "output.nxs", "bounded", 0,
        ),
    )
    plan = DisplayedPhaseFitPlan(
        trace=trace, cif_paths=(cif,), expected_hashes=(None,),
        phase_names=("primitive",), wavelength_angstrom=0.25,
        phase_profile="gaussian", max_nfev=1,
    )
    real_pattern = XRDCalculator.get_pattern
    calls = []

    def observed(self, structure, *args, **kwargs):
        calls.append(tuple(kwargs.get("two_theta_range", (0, 90))))
        return real_pattern(self, structure, *args, **kwargs)

    monkeypatch.setattr(XRDCalculator, "get_pattern", observed)
    bounded_fit = run_displayed_phase_fit(plan)
    expected = np.degrees(2 * np.arcsin(q[[0, -1]] * 0.25 / (4 * np.pi)))
    assert calls and all(
        np.allclose(bounds, expected, rtol=1e-12, atol=1e-12) for bounds in calls
    ), f"fit must request the reachable 2theta range {expected}; got {calls}"
    assert bounded_fit.code == "OK", bounded_fit.diagnostics

    full = PhaseModel.from_cif(cif)
    full.calculate_peaks(wavelength=0.729)
    bounded = PhaseModel("bounded", structure=None)
    bounded.structure = full.structure
    bounded.calculate_peaks(wavelength=0.729, two_theta_range=(5.0, 40.0))
    q0, q1 = 4 * np.pi * np.sin(np.radians([2.5, 20])) / 0.729
    np.testing.assert_allclose(
        _q_values(bounded),
        sorted(peak.q for peak in full.peaks if q0 <= peak.q <= q1),
        rtol=1e-12, atol=1e-12,
    )

    # Compare the fitted result with the same real backend listing the full
    # range. Both paths subsequently admit the identical q-window peaks.
    def force_full(self, structure, *args, **kwargs):
        kwargs["two_theta_range"] = (0, 180)
        return real_pattern(self, structure, *args, **kwargs)

    monkeypatch.setattr(XRDCalculator, "get_pattern", force_full)
    full_fit = run_displayed_phase_fit(plan)
    assert full_fit.code == bounded_fit.code
    np.testing.assert_allclose(full_fit.fit, bounded_fit.fit, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(full_fit.parameter_values, bounded_fit.parameter_values, rtol=1e-12, atol=1e-12)

    PhaseLibrary, build = _phase_api()
    library = PhaseLibrary(tmp_path / "library")
    entry = library.add_cif(cif, name="primitive")
    calls.clear()
    monkeypatch.setattr(XRDCalculator, "get_pattern", observed)
    reference = build(entry, wavelength_angstrom=0.25, two_theta_range=(0.0, 30.0))
    assert not reference.reason
    assert calls and all(bounds == (0.0, 30.0) for bounds in calls)
    assert all(4 * np.pi * np.sin(np.radians(15)) / 0.25 >= p.q for p in reference.phases[0].reflections)


def test_phase_library_round_trips_cif_and_lattice_entries(tmp_path, monkeypatch):
    PhaseLibrary, build = _phase_api()  # New API: no parent-polarity credit.
    root = tmp_path / "isolated-library"
    monkeypatch.setenv("XDART_PHASE_LIBRARY", str(root))
    library = PhaseLibrary()
    cif = tmp_path / "primitive.cif"
    raw = _CIF.encode()
    cif.write_bytes(raw)
    replacements = []
    real_replace = os.replace

    def observed_replace(source, target):
        source, target = Path(source), Path(target)
        assert source.exists() and source != target
        assert source.parent == target.parent
        replacements.append((source, target))
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", observed_replace)
    entry = library.add_cif(cif, name="primitive")
    lattice = library.add_lattice("orthorhombic", a=5.06, b=5.23, c=5.07, space_group="Pca2_1")
    assert entry.id == hashlib.sha256(raw).hexdigest()
    assert entry.source_kind == "cif" and lattice.source_kind == "lattice"
    assert replacements and all(target.is_relative_to(root) for _, target in replacements)
    reopened = PhaseLibrary()
    assert {e.id for e in reopened.entries()} == {entry.id, lattice.id}
    assert reopened.resolve(entry.id).entry.cif_bytes == raw
    assert reopened.resolve(lattice.id).entry.lattice == lattice.lattice
    missing = reopened.resolve("absent")
    assert missing.entry is None and missing.reason
    other = PhaseLibrary(tmp_path / "other-library")
    assert other.entries() == ()
    malformed = tmp_path / "malformed.cif"
    malformed.write_text("this is not a CIF", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)cif|pars"):
        library.add_cif(malformed)
    oversized = tmp_path / "large.cif"
    oversized.write_bytes(raw + b"#" * (4 * 1024**2 + 1))
    with pytest.raises(ValueError, match="(?i)byte|limit|MiB|large"):
        library.add_cif(oversized)

    # Each source meets its cap; five selected 4 MiB CIFs exceed the common
    # 16 MiB capture ceiling before a large reflection listing is attempted.
    large_entries = []
    for i in range(5):
        data = raw + f"\n# {i} ".encode()
        data += b"x" * (4 * 1024**2 - len(data))
        path = tmp_path / f"boundary-{i}.cif"
        path.write_bytes(data)
        large_entries.append(library.add_cif(path))
    assert not build(*large_entries[:4], wavelength_angstrom=1.54, q_range=(1, 3)).reason
    overflow = build(*large_entries, wavelength_angstrom=1.54, q_range=(1, 3))
    assert not overflow.phases and "byte" in overflow.reason.lower()


def test_reference_reflections_apply_the_phase_fitters_caps(tmp_path):
    PhaseLibrary, build = _phase_api()  # New API: no parent-polarity credit.
    library = PhaseLibrary(tmp_path / "library")
    cif = tmp_path / "primitive.cif"
    cif.write_text(_CIF, encoding="utf-8")
    entry = library.add_cif(cif, name="primitive")
    all_peaks = build(entry, wavelength_angstrom=0.729, q_range=(1, 6), min_intensity=0)
    default = build(entry, wavelength_angstrom=0.729, q_range=(1, 6))
    assert not all_peaks.reason and not default.reason
    expected = tuple(p for p in all_peaks.phases[0].reflections if p.intensity >= 5)
    assert default.phases[0].reflections == expected
    for floor in (0, 5, 100):
        result = build(entry, wavelength_angstrom=0.729, q_range=(1, 6), min_intensity=floor)
        assert result.phases[0].reflections == tuple(p for p in all_peaks.phases[0].reflections if p.intensity >= floor)
    exact_floor = all_peaks.phases[0].reflections[0].intensity
    at_boundary = build(entry, wavelength_angstrom=0.729, q_range=(1, 6), min_intensity=exact_floor)
    assert all_peaks.phases[0].reflections[0] in at_boundary.phases[0].reflections
    for floor in (-1, 101, math.nan):
        refused = build(entry, wavelength_angstrom=0.729, q_range=(1, 6), min_intensity=floor)
        assert not refused.phases and refused.reason

    lattice = library.add_lattice("lattice", a=4.0, space_group="Pm-3m")
    low = build(lattice, wavelength_angstrom=0.729, q_range=(1, 6), min_intensity=0)
    high = build(lattice, wavelength_angstrom=0.729, q_range=(1, 6), min_intensity=100)
    assert low.phases == high.phases
    assert all(p.intensity is None for p in low.phases[0].reflections)
    other_energy = build(lattice, wavelength_angstrom=1.0, q_range=(1, 6))
    assert other_energy.phases[0].reflections == low.phases[0].reflections
    angles = build(lattice, wavelength_angstrom=1.0, two_theta_range=(10, 30))
    assert all(10 <= np.degrees(2 * np.arcsin(p.q / (4 * np.pi))) <= 30 for p in angles.phases[0].reflections)
    phases = tuple(library.add_lattice(f"phase-{i}", a=4 + i / 100, space_group="Pm-3m") for i in range(9))
    assert not build(*phases[:8], wavelength_angstrom=0.729, q_range=(1, 3)).reason
    phase_cap = build(*phases, wavelength_angstrom=0.729, q_range=(1, 3))
    assert not phase_cap.phases and "phase" in phase_cap.reason.lower()

    dense = library.add_lattice("too-many", a=40.1, b=41.2, c=42.3, space_group=1)
    per_phase = build(dense, wavelength_angstrom=0.25, q_range=(1, 6))
    assert not per_phase.phases and "reflection" in per_phase.reason.lower()
    medium = tuple(library.add_lattice(
        f"medium-{i}", a=14.1, b=15.2, c=16.3, space_group=1,
    ) for i in range(3))
    single = build(medium[0], wavelength_angstrom=0.25, q_range=(1, 6))
    assert not single.reason
    assert 4096 / 3 < len(single.phases[0].reflections) <= 2048
    total = build(*medium, wavelength_angstrom=0.25, q_range=(1, 6))
    assert not total.phases and "reflection" in total.reason.lower()
