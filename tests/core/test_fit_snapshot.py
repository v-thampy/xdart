"""The fit-input snapshot (``xrd_tools.analysis.fit_snapshot``; FIT-TOOLS-2026-10, Codex's design review F4, oracle
row 6's staleness part): a fit host captures its input detached, with a key of facts it already holds, and adopts a
result only while that key is still its current one."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from xrd_tools.analysis.display_fit_operations import DisplayedPeakFitPlan, run_displayed_peak_fit
from xrd_tools.analysis.fit_snapshot import FitAxis, FitInputKey, FitInputSnapshot
from xrd_tools.analysis.scan_operations import AnalysisDisposition


def _key(**changes):
    key = FitInputKey("stitch:scan_12", "/data/scan_12.nexus", "admission-fp-1", 3,
                      selection=(("panel", "1D"), ("cut", "q")), background="none",
                      settings=(("model", "pseudovoigt"), ("fit_bounds", (1.0, 4.0))))
    return replace(key, **changes)


def _pattern():
    q = np.linspace(1.0, 4.0, 301)
    return q, 20.0 + 500.0 * np.exp(-0.5 * ((q - 2.5) / 0.03) ** 2)


def test_a_fit_is_stale_once_its_result_is_replaced_reloaded_recut_or_refitted_otherwise_current():
    q, y = _pattern()
    snapshot = FitInputSnapshot(_key(), (FitAxis("q", "A^-1", q),), y, wavelength_angstrom=1.0)
    assert snapshot.is_current(_key())  # the same facts, captured again
    for change in ({"fingerprint": "admission-fp-2"},  # replaced at the same path: admitted again
                   {"generation": 4},  # reloaded, or the window reopened
                   {"selection": (("panel", "1D"), ("cut", "chi"))},  # another cut
                   {"background": "presentation"},  # Set BG now applies
                   {"settings": (("model", "gaussian"), ("fit_bounds", (1.0, 4.0)))}):
        assert not snapshot.is_current(_key(**change)), change


def test_a_snapshot_owns_read_only_copies_of_its_values_mask_and_coordinates():
    q, y = _pattern()
    mask = np.isfinite(y)
    mask[:10] = False
    snapshot = FitInputSnapshot(_key(), (FitAxis("q", "A^-1", q),), y, mask, 1.0)
    y[:] = 0.0  # the host's buffers change after the capture
    q[:] = 0.0
    mask[:] = True
    assert snapshot.values.max() > 500.0 and snapshot.axes[0].values[0] == 1.0 and not snapshot.mask[:10].any()
    for array in (snapshot.values, snapshot.mask, snapshot.axes[0].values):
        assert not array.flags.writeable
    fitted = snapshot.fitted_values()
    assert np.isnan(fitted[:10]).all() and np.isfinite(fitted[10:]).all() and not fitted.flags.writeable
    image = FitInputSnapshot(_key(), (FitAxis("K", "r.l.u.", np.arange(3.0)), FitAxis("H", "r.l.u.", np.arange(4.0))),
                             np.zeros((3, 4)))
    assert image.values.shape == (3, 4) and image.fitted_values() is image.values


@pytest.mark.parametrize("bad", (
    lambda q, y: FitInputSnapshot(_key(), (FitAxis("q", "", q),), y[:-1]),
    lambda q, y: FitInputSnapshot(_key(), (FitAxis("q", "", q),), y, np.ones(y.shape, dtype=int)),
    lambda q, y: FitInputSnapshot(_key(), (FitAxis("q", "", q),), y, wavelength_angstrom=0.0),
    lambda q, y: _key(background="log"),
    lambda q, y: _key(settings=(("fraction", float("nan")),)),
    lambda q, y: _key(selection=(("cut", "q"), ("cut", "chi"))),
    lambda q, y: _key(generation="3"),
))
def test_an_ill_formed_input_is_refused(bad):
    q, y = _pattern()
    with pytest.raises((TypeError, ValueError)):
        bad(q, y)


def test_a_one_dimensional_snapshot_is_fitted_by_the_displayed_fit_policy_with_its_key_as_receipt():
    q, y = _pattern()
    snapshot = FitInputSnapshot(_key(), (FitAxis("q", "A^-1", q),), y, wavelength_angstrom=1.0)
    trace = snapshot.trace_input()
    receipt = trace.receipt
    assert (receipt.run_generation, receipt.run_fingerprint, receipt.source_scan, str(receipt.artifact)) == (
        3, "admission-fp-1", "stitch:scan_12", "/data/scan_12.nexus")
    assert receipt.frame_label == "panel='1D';cut='q'" and (trace.label, trace.axis_unit) == ("q", "A^-1")
    result = run_displayed_peak_fit(DisplayedPeakFitPlan(trace))
    assert result.disposition is AnalysisDisposition.COMPLETED and result.trace_fingerprint == trace.trace_fingerprint
    assert result.marker_positions == pytest.approx([2.5], abs=0.01)
    reloaded = FitInputSnapshot(_key(generation=4), (FitAxis("q", "A^-1", q),), y).trace_input()
    assert reloaded.trace_fingerprint != trace.trace_fingerprint  # another input: the result's identity differs
    with pytest.raises(ValueError, match="one-dimensional"):
        FitInputSnapshot(_key(), (FitAxis("K", "", np.arange(2.0)), FitAxis("H", "", np.arange(2.0))),
                         np.zeros((2, 2))).trace_input()
