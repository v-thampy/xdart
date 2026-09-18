# -*- coding: utf-8 -*-
"""GI-COMPANION-20260918 — the display-only q–χ view derived from a q_ip–q_oop cake.

The helper re-bins a saved Cartesian map; it is an approximation and is never
asserted equal to pyFAI's direct q–χ integration.  What IS asserted: the
coordinate convention and array orientation (against a real written direct map),
explicit treatment of invalid bins, no coverage outside the source map, and
bounded, grid-keyed mapping caches.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.core.test_gi_companion_modes import _plan, _run
from xrd_tools.io import read_frame_record
from xrd_tools.session import gi_derived_view
from xrd_tools.session.gi_derived_view import derive_q_chi, derived_q_chi_cache_bytes

QIP = np.linspace(-1.5, 1.5, 121)
QOOP = np.linspace(0.01, 1.8, 91)


def _ring(radius: float = 1.0) -> np.ndarray:
    qq = np.hypot(QIP[None, :], QOOP[:, None])
    return 5.0 + 100.0 * np.exp(-0.5 * ((qq - radius) / 0.04) ** 2)


def _blob(qip: float, qoop: float) -> np.ndarray:
    return 1.0 + 100.0 * np.exp(
        -0.5 * (((QIP[None, :] - qip) / 0.05) ** 2 + ((QOOP[:, None] - qoop) / 0.05) ** 2)
    )


def test_a_ring_lands_at_its_radius_for_every_chi():
    derived = derive_q_chi(_ring(1.0), QIP, QOOP)
    assert derived.intensity.shape == (derived.chi.size, derived.q.size)
    peaks = derived.q[np.nanargmax(derived.intensity, axis=1)]
    covered = np.isfinite(derived.intensity).sum(axis=1) > derived.q.size // 2
    assert covered.sum() > derived.chi.size // 2
    np.testing.assert_allclose(peaks[covered], 1.0, atol=2 * np.diff(derived.q)[0])


@pytest.mark.parametrize(("qip", "qoop"), ((0.8, 0.6), (-0.8, 0.6), (0.0, 1.2)))
def test_chi_is_measured_from_the_surface_normal_towards_qip(qip, qoop):
    """pyFAI's fiber convention: chi = atan2(q_ip, q_oop), so +q_ip is +chi."""
    derived = derive_q_chi(_blob(qip, qoop), QIP, QOOP)
    row, column = np.unravel_index(np.nanargmax(derived.intensity), derived.intensity.shape)
    assert derived.q[column] == pytest.approx(np.hypot(qip, qoop), abs=0.05)
    assert derived.chi[row] == pytest.approx(np.degrees(np.arctan2(qip, qoop)), abs=4.0)


def test_orientation_and_convention_match_a_written_direct_map(tmp_path, capsys):
    """Derived from the SAVED Cartesian view, compared with the SAVED direct view.

    Both come from one real run (real pyFAI, real writer, real reader), so a
    transposed or mirrored helper cannot pass.  The deviations are reported,
    not asserted: this is double binning, not the direct integration.
    """
    target, _sink = _run(tmp_path, "both", _plan("qip_qoop", "q_chi"))
    record = read_frame_record(target, 1)
    cart, direct = record.view_2d("qip_qoop"), record.view_2d("q_chi")

    derived = derive_q_chi(
        cart.intensity_2d, cart.axis_2d_x.values, cart.axis_2d_y.values,
        q=direct.axis_2d_x.values, chi=direct.axis_2d_y.values,
    )
    assert derived.intensity.shape == direct.intensity_2d.shape
    both = np.isfinite(derived.intensity) & np.isfinite(direct.intensity_2d)
    assert both.sum() > 0.5 * np.isfinite(direct.intensity_2d).sum()
    ours, theirs = derived.intensity[both], direct.intensity_2d[both]
    correlation = float(np.corrcoef(ours, theirs)[0, 1])
    mirrored = float(np.corrcoef(
        derived.intensity[::-1][both[::-1] & both], direct.intensity_2d[both[::-1] & both],
    )[0, 1])
    deviation = np.abs(ours - theirs) / np.maximum(np.abs(theirs), np.percentile(np.abs(theirs), 10))
    with capsys.disabled():
        print(
            f"\n[derived q-chi vs direct, 30x24 bins, Pilatus 100k] correlation {correlation:.4f} "
            f"(chi-mirrored {mirrored:.4f}); median deviation {100 * np.median(deviation):.2f}%, "
            f"p95 {100 * np.percentile(deviation, 95):.2f}%; direct-only bins "
            f"{int((np.isfinite(direct.intensity_2d) & ~np.isfinite(derived.intensity)).sum())}, "
            f"derived-only bins {int((~np.isfinite(direct.intensity_2d) & np.isfinite(derived.intensity)).sum())}"
        )
    assert correlation > 0.9 and correlation > mirrored


def test_invalid_source_bins_are_excluded_not_averaged_in():
    cake = np.full((QOOP.size, QIP.size), 10.0)
    cake[:, : QIP.size // 2] = np.nan                 # everything at q_ip < 0 unmeasured
    derived = derive_q_chi(cake, QIP, QOOP)
    finite = np.isfinite(derived.intensity)
    assert finite.any()
    np.testing.assert_allclose(derived.intensity[finite], 10.0)
    # chi < 0 is the unmeasured half: nothing is invented there.  (Very close to
    # the origin the measured q_ip = 0 column itself reaches slightly negative
    # chi, so the claim is made away from it.)
    assert not finite[np.ix_(derived.chi < -5.0, derived.q > 0.3)].any()
    assert finite[np.ix_(derived.chi > 5.0, derived.q > 0.3)].any()

    flagged = np.full(cake.shape, 10.0)
    flagged[:, : QIP.size // 2] = -1.0                # a caller-known dummy value
    masked = derive_q_chi(flagged, QIP, QOOP, valid=flagged != -1.0)
    np.testing.assert_array_equal(np.isfinite(masked.intensity), finite)


def test_a_cropped_source_yields_a_cropped_view():
    crop = QIP > 0.5
    derived = derive_q_chi(_ring()[:, crop], QIP[crop], QOOP)
    assert derived.q[0] >= 0.5 - 0.05
    assert derived.chi[0] > 0.0                       # only the +q_ip side exists
    assert derived.chi[-1] < 90.0
    full = derive_q_chi(_ring(), QIP, QOOP)
    assert full.chi[0] < -80.0 and full.chi[-1] > 80.0


def test_a_half_covered_target_bin_is_dropped():
    cake = np.full((QOOP.size, QIP.size), 3.0)
    lenient = derive_q_chi(cake, QIP, QOOP, min_valid_fraction=1e-9)
    strict = derive_q_chi(cake, QIP, QOOP)
    assert np.isfinite(strict.intensity).sum() <= np.isfinite(lenient.intensity).sum()
    np.testing.assert_allclose(strict.intensity[np.isfinite(strict.intensity)], 3.0)


def test_the_mapping_cache_is_keyed_by_both_grids_and_bounded(monkeypatch):
    gi_derived_view._cache.clear()
    builds = []
    real = gi_derived_view._build
    monkeypatch.setattr(
        gi_derived_view, "_build",
        lambda *axes: builds.append(1) or real(*axes),
    )
    derive_q_chi(_ring(), QIP, QOOP)
    derive_q_chi(_ring(1.2), QIP, QOOP)                       # same grids: reused
    assert len(builds) == 1
    derive_q_chi(_ring()[:, 1:], QIP[1:], QOOP)               # source grid changed
    derive_q_chi(_ring(), QIP, QOOP, q=np.linspace(0.1, 2.0, 40))   # target changed
    assert len(builds) == 3
    assert len(gi_derived_view._cache) <= gi_derived_view._CACHE_ENTRIES
    assert 0 < derived_q_chi_cache_bytes() <= gi_derived_view._CACHE_BYTES
    derive_q_chi(_ring(), QIP, QOOP)                          # evicted: rebuilt
    assert len(builds) == 4


@pytest.mark.parametrize("bad", ("shape", "axis", "fraction"))
def test_malformed_input_is_refused(bad):
    cake = _ring()
    with pytest.raises(ValueError):
        if bad == "shape":
            derive_q_chi(cake.T, QIP, QOOP)
        elif bad == "axis":
            derive_q_chi(cake, QIP[::-1], QOOP)
        else:
            derive_q_chi(cake, QIP, QOOP, min_valid_fraction=0.0)
